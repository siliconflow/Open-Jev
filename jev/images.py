"""Opt-in image channel for Open-Jev serving (JEV_IMAGES=1). Channel open, untrained readout.

The Open-Jev checkpoints are trained text-only: their LoRA adapter and decision head
have never seen an image, and the released model.json pins a Qwen3.5 multimodal base
whose vision tower (model.visual.*) is dropped by DecisionModel's load path
(AutoModelForImageTextToText -> full.model.language_model; `full` is deleted right
after init). This module re-attaches the untrained tower from the SAME pinned
snapshot, so the image channel opens without touching the text path. An open channel
is not a capability claim: report TVD/separation, never accuracy, for image requests.

Gating: nothing runs unless jev.server imports attach() under JEV_IMAGES=1. With the
gate unset the module is never imported by the serving path; compile_request strips
state.images for every consumer that does not claim them, so prompts, candidates and
the published text numbers are byte-identical.

Pipeline (one request): decode refs -> official preprocess -> one tower pass ->
splice the merged rows into the candidate prompts' token stream after the state
text -> score with the same head. The prompts keep their exact text; images are
inserted as vision_start + image_pad x N + vision_end token segments inside the
shared Context block, which the chat template otherwise renders verbatim.
"""
import base64
import io
import os
import re
import urllib.request

IMAGE_MIMES = ("image/jpeg", "image/png", "image/webp")
IMAGE_MAX_BYTES = 5 * 1024 * 1024    # decoded, per image
IMAGE_FETCH_LIMIT = 8 * 1024 * 1024  # https fetch cap
MAX_IMAGES = 4

_DATA_URL = re.compile(r"^data:([\w./+-]+);base64,(.*)$", re.S)

# separator between the Context block and the Question block inside a candidate
# prompt (jev.api.candidate_prompts builds "Context:<newline>{state}<blank line>
# Question: ..."; the two newlines before "Question:" mark the Context end)
QUESTION_SEP = "\n\nQuestion:"


def decode_image(ref):
    """data URL | https URL -> PIL.Image (in-memory). Raises ValueError on any refusal."""
    from PIL import Image

    m = _DATA_URL.match(ref.strip())
    if m:
        mime, b64 = m.group(1), m.group(2)
        if mime not in IMAGE_MIMES:
            raise ValueError(f"unsupported image mime {mime} (jpeg/png/webp)")
        raw = base64.b64decode(b64, validate=True)
        if len(raw) > IMAGE_MAX_BYTES:
            raise ValueError("image exceeds 5 MiB decoded")
        im = Image.open(io.BytesIO(raw))
        im.load()
        return im
    if ref.startswith("https://"):
        req = urllib.request.Request(ref, headers={"user-agent": "openjev-images/1"})
        with urllib.request.urlopen(req, timeout=10) as r:  # type: ignore[attr-defined]
            raw = r.read(IMAGE_FETCH_LIMIT + 1)
        if len(raw) > IMAGE_FETCH_LIMIT:
            raise ValueError("image fetch exceeded 8 MiB cap")
        if len(raw) > IMAGE_MAX_BYTES:
            raise ValueError("image exceeds 5 MiB decoded")
        im = Image.open(io.BytesIO(raw))
        im.load()
        return im
    raise ValueError("images must be data URLs or https URLs")


def extract_images(state):
    """Pull the optional images list off a dict state BEFORE compile_request's deep
    copy; returns (text_state, images_or_None). A non-list `images` value is NOT an
    image ref and stays in the state as ordinary data (compiled records then render
    it, matching upstream's json-dump behavior)."""
    if isinstance(state, dict) and isinstance(state.get("images"), list):
        images, rest = state["images"], {k: v for k, v in state.items() if k != "images"}
        if not images:
            return state, None
        if len(images) > MAX_IMAGES:
            raise ValueError(f"at most {MAX_IMAGES} images per request")
        for ref in images:
            if not isinstance(ref, str):
                raise ValueError("state.images entries must be strings (data/https URLs)")
        return rest, images
    return state, None


# ---------------------------------------------------------------------------
# attach: build the tower once at server startup (JEV_IMAGES=1)
# ---------------------------------------------------------------------------

def attach(model):
    """Build the tower for a loaded DecisionModel from its own pinned base snapshot.

    model.model_id / model.revision are the pins DecisionModel.load honours, so the
    tower weights come from the exact snapshot the language model was loaded from
    (already on disk: DecisionModel loaded it). Returns a VisionHook, or None when
    the pinned base has no vision tower - the channel then stays closed and image
    requests get a 422.
    """
    from transformers import AutoConfig

    snapshot = _resolve_snapshot(model.model_id, model.revision)
    cfg = AutoConfig.from_pretrained(snapshot)
    vision_cfg = getattr(cfg, "vision_config", None)
    if vision_cfg is None:
        print(f"[images] {model.model_id}: no vision tower in this base; channel stays closed",
              flush=True)
        return None
    tower = _load_tower(vision_cfg, snapshot)
    ref = next(model.backbone.parameters())
    tower.to(ref.device).to(ref.dtype)
    tower.eval()
    for p in tower.parameters():
        p.requires_grad_(False)
    try:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(snapshot)   # image-only: skips the
        # Qwen3VL video sub-processor (extra deps, ffmpeg for videos) - this route takes images
    except Exception as e:  # processor stack unavailable: refuse rather than hand-roll patches
        raise RuntimeError(f"image channel needs AutoImageProcessor for {snapshot}: {e!r}")
    hook = VisionHook(model, tower, processor,
                      cfg.image_token_id, cfg.vision_start_token_id, cfg.vision_end_token_id)
    print(f"[images] tower attached ({type(tower).__name__}), "
          f"{sum(p.numel() for p in tower.parameters()) / 1e6:.0f}M params, untrained - "
          f"channel open, untrained readout", flush=True)
    return hook


def _resolve_snapshot(model_id, revision):
    """Local snapshot dir for the pinned base (hf hub cache layout, or the id itself
    when it already is a local path)."""
    import glob

    if os.path.isdir(model_id):
        return model_id
    root = os.path.expanduser("~/.cache/huggingface/hub")
    name = f"models--{model_id.replace('/', '--')}"
    pats = [f"{root}/{name}/snapshots/*"]
    if revision:
        pats.insert(0, f"{root}/{name}/snapshots/{revision}")
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    # not cached as a plain dir (e.g. a local .venv-side test base never hit this);
    # fall back to transformers' own resolver, which downloads or raises
    from transformers.utils.hub import cached_file

    return os.path.dirname(cached_file(model_id, "config.json", revision=revision))


def _load_tower(vision_cfg, snapshot):
    """Instantiate the standalone Qwen3.5 tower and load model.visual.* from the snapshot."""
    import glob as _glob

    from safetensors.torch import load_file as _load_file
    from transformers.models.qwen3_5 import modeling_qwen3_5

    tower = modeling_qwen3_5.Qwen3_5VisionModel._from_config(vision_cfg)
    state = {}
    for path in sorted(_glob.glob(os.path.join(_glob.escape(snapshot), "*.safetensors"))):
        for k, v in _load_file(path).items():
            m = re.match(r"^model\.visual\.([0-9A-Za-z_.]+)$", k)
            if m:
                state[m.group(1)] = v
    if not state:
        raise RuntimeError(f"no model.visual.* tensors under {snapshot}")
    missing, unexpected = tower.load_state_dict(state, strict=False)
    missing = [k for k in missing if "inv_freq" not in k]   # lazily-built non-persistent buffer
    if missing:
        raise RuntimeError(f"vision tower incomplete: missing {missing[:5]} ({len(missing)})")
    return tower


# ---------------------------------------------------------------------------
# the hook: official preprocess, one tower run, splice into candidate prompts
# ---------------------------------------------------------------------------

class VisionHook:
    """All state needed to answer image-bearing requests. No per-request state is kept."""

    def __init__(self, model, tower, processor, image_token, vision_start, vision_end):
        self.model, self.tower, self.processor = model, tower, processor
        self.image_token = int(image_token)
        self.vision_start, self.vision_end = int(vision_start), int(vision_end)

    def embed(self, images):
        """PIL images -> (merged rows [N, out_hidden] on the tower's device/dtype,
        rows-per-image list). The official processor supplies pixel_values and
        image_grid_thw; N comes from grid_thw // merge**2 and is asserted against the
        tower's own output - never trusted alone."""
        import torch

        with torch.no_grad():
            kept = images[:MAX_IMAGES]
            out = self.processor(images=kept, return_tensors="pt")
            ref = next(self.tower.parameters())
            pixel_values = out["pixel_values"].to(ref.device, ref.dtype)
            grid_thw = out["image_grid_thw"].to(ref.device)   # the tower derives position/interp
            # indices from it - they must share the tower's device (MPS embedding fails otherwise)
            m = getattr(self.tower.config, "spatial_merge_size", 2) or 2
            n_per = [t * h * w // (m * m) for t, h, w in grid_thw.tolist()]
            # transformers names the first arg `hidden_states` (Qwen2-VL tradition) but it takes
            # pixel_values - patch_embed() is applied inside the tower. Positional, to stay
            # robust against the name.
            res = self.tower(pixel_values, grid_thw=grid_thw, return_dict=True)
            rows = res.pooler_output
            if rows.shape[0] != sum(n_per):
                raise RuntimeError(f"tower produced {rows.shape[0]} rows, "
                                   f"grid_thw says {sum(n_per)} - refusing to guess")
            text_hidden = getattr(self.model.backbone.config, "hidden_size", None)
            if text_hidden is not None and rows.shape[1] != text_hidden:
                raise RuntimeError(f"vision out_hidden {rows.shape[1]} != text hidden "
                                   f"{text_hidden}: this base cannot be spliced without a "
                                   f"projection layer (not implemented)")
            return rows, n_per

    def segment_ids(self, n_per_image):
        """Token ids inserted into every candidate prompt at the end of the Context
        block: per image, vision_start + image_pad x N + vision_end."""
        seg = []
        for n in n_per_image:
            seg += [self.vision_start] + [self.image_token] * n + [self.vision_end]
        return seg

    def score(self, records, images):
        """DecisionModel.forward for a state whose images have been decoded: prompts
        keep their text, the shared Context block gains the image segment, scoring
        uses the same head. Returns (logits per record, encoded token count) in the
        exact shapes TorchScorer.score produces, so Predictor.predict is reused.

        Splice point: every candidate prompt of one request shares the prefix
        "Context:" + newline + {state} + blank line + "Question:" (jev.api.
        candidate_prompts). We render the prompts exactly as DecisionModel.forward
        does, locate that shared prefix inside the first rendered prompt as a token
        subsequence (an assert, not a heuristic), and insert the image segment at
        its end on every row. Question and proposal tokens stay identical to the
        text path; every candidate sees the images at the same position.
        """
        import torch

        from .api import candidate_prompts

        with torch.no_grad():
            rows, n_per = self.embed(images)
            seg = self.segment_ids(n_per)
            tok = self.model.tokenizer

            # prompts identical to the text path (the same rendering forward() uses)
            prompts, counts = [], []
            for record in records:
                entries = candidate_prompts(record)
                counts.append(len(entries))
                for entry in entries:
                    prompts.append(tok.apply_chat_template(
                        [{"role": "user", "content": entry}], tokenize=False,
                        add_generation_prompt=True, enable_thinking=False,
                    ))
            if QUESTION_SEP not in prompts[0]:
                raise RuntimeError("candidate prompt lacks a Context block - template changed?")
            text_prefix = prompts[0].split(QUESTION_SEP)[0] + QUESTION_SEP

            encoded = tok(prompts, padding=True, truncation=False, return_tensors="pt")
            lengths = encoded["attention_mask"].sum(-1)
            max_len = int(lengths.max().item())
            if max_len + len(seg) > self.model.max_length:
                raise ValueError(f"Input length {max_len + len(seg)} with images exceeds "
                                 f"max_length={self.model.max_length}; no silent truncation")

            p0 = encoded["input_ids"][0][: int(lengths[0])]       # first prompt, unpadded
            prefix_ids = tok(text_prefix, add_special_tokens=False)["input_ids"]
            at = _subsequence_index(p0.tolist(), prefix_ids) + len(prefix_ids)

            B, L = int(encoded["input_ids"].shape[0]), int(encoded["input_ids"].shape[1])
            ids, att = encoded["input_ids"], encoded["attention_mask"]
            seg_t = torch.tensor(seg, dtype=ids.dtype)
            # insert the segment at `at` on every row; rows are right-padded, so the
            # tail after `at` shifts uniformly and attention stays per-row correct
            new_ids = torch.full((B, L + len(seg)), tok.pad_token_id, dtype=ids.dtype)
            new_att = torch.zeros((B, L + len(seg)), dtype=att.dtype)
            for i in range(B):
                n = int(lengths[i])
                new_ids[i, :at] = ids[i, :at]
                new_ids[i, at:at + len(seg)] = seg_t
                new_ids[i, at + len(seg):at + len(seg) + n - at] = ids[i, at:n]
                new_att[i, : n + len(seg)] = 1
            lengths = lengths + len(seg)

            dev = next(self.model.backbone.parameters()).device
            ids_dev = new_ids.to(dev)
            emb = self.model.backbone.get_input_embeddings()(ids_dev)
            n_placeholders = sum(n_per)   # per row; vision_start/end are NOT placeholders
            img = ids_dev == self.image_token
            if int(img.sum()) != B * n_placeholders:
                raise RuntimeError("image placeholder count mismatch - refusing to splice")
            emb[img] = rows.to(emb.dtype).repeat(B, 1)   # row-major: each row's placeholders, in order
            outputs = self.model.backbone(
                inputs_embeds=emb, attention_mask=new_att.to(dev), use_cache=False,
                return_dict=True)
            self.model.last_input_tokens = int(lengths.sum().item())
            # under sharding the last hidden state comes out on the LAST device, not
            # the input device (mirrors DecisionModel.forward).
            hdev = outputs.last_hidden_state.device
            idx = torch.arange(B, device=hdev)
            hidden = outputs.last_hidden_state[idx, lengths.to(hdev) - 1]
            scores = self.model.head(
                hidden.float().to(self.model.head.weight.device)).squeeze(-1)
            logits, offset = [], 0
            for record, count in zip(records, counts):
                values = scores[offset:offset + count]
                if record["kind"] == "noul":
                    values = torch.stack([torch.zeros_like(values[0]), values[0]])
                logits.append(values)
                offset += count
            return logits, self.model.last_input_tokens


def _subsequence_index(haystack, needle):
    """First index of `needle` inside `haystack`; raises when absent (no -1 sentinel)."""
    if not needle:
        raise ValueError("empty prefix")
    first, n = needle[0], len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i] == first and haystack[i:i + n] == needle:
            return i
    raise RuntimeError("state prefix not found in rendered prompt - template changed?")


class ImageScorer:
    """Image-channel scorer (deploy pack, JEV_IMAGES gate): wraps a TorchScorer and
    its VisionHook. Records WITHOUT the carried `images` key take the wrapped text
    scorer verbatim - byte-identical to a gate-off server. Records with images run
    through the hook (official preprocess, one tower pass, splice, same head); the
    per-record image decode errors surface as ValueError -> 422.

    Batching note: the hook scores a request's candidates in one padded batch, so
    Predictor.predict's candidate_batches chunking is bypassed for image records -
    a request either fits in one pass or exceeds max_length anyway (the hook
    refuses to truncate silently). The noul logits duplication (the torch.stack
    [zero, score] shape DecisionModel.forward produces) is replicated by the hook,
    so format_response compatibility is exact.
    """

    def __init__(self, wrapped, hook):
        self.wrapped, self.hook = wrapped, hook

    def score(self, records):
        if not any("images" in r for r in records):
            return self.wrapped.score(records)      # text path verbatim
        if not all("images" in r for r in records):
            raise ValueError("mixed image and text records in one request")
        images = [decode_image(ref) for ref in records[0]["images"]]
        rows, tokens = self.hook.score(records, images)
        return [row.float().cpu().tolist() for row in rows], tokens
