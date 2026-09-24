"""Bounded candidate batching shared by HTTP and Python inference clients."""
import math
import time
import hashlib
import os
import subprocess
from pathlib import Path
import json

from .api import compile_request, format_response
from .metrics import softmax


def candidate_batches(records, limit):
    """Yield (record index, partial record) groups with at most limit sequences.

    A large Choice is split before scoring and normalized only after its logits
    are reassembled. Normalizing each chunk would change the decision.
    """
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("candidate batch size must be positive")
    batch, size = [], 0
    for index, record in enumerate(records):
        pieces = [record] if record["kind"] == "noul" else [
            {**record, "options": record["options"][start:start + limit]}
            for start in range(0, len(record["options"]), limit)
        ]
        for piece in pieces:
            count = 1 if piece["kind"] == "noul" else len(piece["options"])
            if batch and size + count > limit:
                yield batch
                batch, size = [], 0
            batch.append((index, piece))
            size += count
    if batch:
        yield batch


class Predictor:
    """scorer.score(records) returns raw logits and actual encoded token count."""

    def __init__(self, scorer, *, model_name, temperature=1.0, batch_size=32,
                 max_questions=4096, max_candidates=65536, method="checkpoint", provenance=None,
                 prefix_cache=False):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if min(batch_size, max_questions, max_candidates) < 1:
            raise ValueError("inference limits must be positive")
        self.scorer, self.model_name, self.method = scorer, model_name, method
        self.temperature, self.batch_size = temperature, batch_size
        self.max_questions, self.max_candidates = max_questions, max_candidates
        self.provenance = dict(provenance or {})
        self.prefix_cache = prefix_cache

    def predict(self, request):
        if not isinstance(request, dict) or not {"state", "questions"} <= request.keys():
            raise ValueError("request requires state and questions")
        # jev-latest is an explicit wire-compatibility alias, not a TypeSafe model.
        if request.get("model") not in (None, "open-jev", "jev-latest", self.model_name):
            raise ValueError("requested model is not loaded; see /v1/models")
        if not isinstance(request["questions"], dict) or len(request["questions"]) > self.max_questions:
            raise ValueError("questions must be an object within the server question limit")
        records = compile_request(request["state"], request["questions"])
        count = sum(1 if r["kind"] == "noul" else len(r["options"]) for r in records)
        if count > self.max_candidates:
            raise ValueError("request exceeds the candidate limit")
        start, input_tokens = time.perf_counter(), 0
        cache_stats = {"enabled": False, "mode": "independent_candidates"}
        if self.prefix_cache:
            score_request = getattr(self.scorer, "score_request", None)
            if not callable(score_request):
                raise ValueError("Selected backend does not support request-local prefix caching")
            logits, cache_stats = score_request(records, batch_size=self.batch_size)
            if len(logits) != len(records):
                raise RuntimeError("backend returned the wrong number of rows")
            for record, row in zip(records, logits):
                expected = 2 if record["kind"] == "noul" else len(record["options"])
                if len(row) != expected or any(not math.isfinite(v) for v in row):
                    raise RuntimeError("backend returned invalid logits")
            input_tokens = cache_stats["logical_input_tokens"]
        else:
            logits = [[] for _ in records]
            for batch in candidate_batches(records, self.batch_size):
                rows, token_count = self.scorer.score([piece for _, piece in batch])
                if len(rows) != len(batch):
                    raise RuntimeError("backend returned the wrong number of rows")
                input_tokens += token_count
                for (index, piece), row in zip(batch, rows):
                    expected = 2 if piece["kind"] == "noul" else len(piece["options"])
                    if len(row) != expected or any(not math.isfinite(v) for v in row):
                        raise RuntimeError("backend returned invalid logits")
                    logits[index].extend(row)
        result = format_response(records, [softmax(row, self.temperature) for row in logits])
        result.update(model=self.model_name, usage={"input_tokens": input_tokens, "output_tokens": 0},
                      metadata={"method": self.method, "temperature": self.temperature,
                                "candidate_sequences": count,
                                "inference_seconds": time.perf_counter() - start,
                                "prefix_cache": cache_stats, **self.provenance})
        return result


class TorchScorer:
    def __init__(self, model):
        self.model = model

    def score(self, records):
        import torch
        with torch.inference_mode():
            rows = self.model(records)
        return [row.float().cpu().tolist() for row in rows], self.model.last_input_tokens

    def score_request(self, records, *, batch_size):
        rows, stats = self.model.score_cached(records, batch_size=batch_size)
        return [row.float().cpu().tolist() for row in rows], stats


class ImageScorer:
    """Image-channel scorer (deploy pack, JEV_IMAGES gate): wraps a TorchScorer and
    its VisionHook. Records WITHOUT the carried `images` key take the wrapped text
    scorer verbatim - byte-identical to a gate-off server. Records with images run
    through the hook (official preprocess, one tower pass, splice, same head); the
    per-record image decode errors surface as ValueError -> 422.

    Batching note: the hook scores a request's candidates in one padded batch, so
    Predictor.predict's candidate_batches chunking is bypassed for image records -
    a request either fits in one pass or exceeds max_candidates anyway. The noul
    duplication logic (forward's torch.stack for kind == "noul") is replicated by
    the hook, so TorchScorer.format_response compatibility is exact.
    """

    def __init__(self, wrapped, hook):
        self.wrapped, self.hook = wrapped, hook

    def score(self, records):
        if not any("images" in r for r in records):
            return self.wrapped.score(records)
        if not all("images" in r for r in records):
            raise ValueError("mixed image and text records in one request")
        from .images import decode_image
        images = [decode_image(ref) for ref in records[0]["images"]]
        rows, tokens = self.hook.score([r for r in records], images)
        return [row.float().cpu().tolist() for row in rows], tokens


def load_predictor(*, checkpoint=None, model_id=None, revision=None, device="cuda:0",
                   max_length=None, batch_size=32, prefix_cache=False):
    from .model import DecisionModel
    if bool(checkpoint) == bool(model_id):
        raise ValueError("choose exactly one checkpoint or base model")
    if checkpoint:
        path = Path(checkpoint)
        model = DecisionModel.load(path, device=device)
        temperature = json.loads((path / "temperature.json").read_text())["temperature"]
        method = "lora_decision_head"
        digest = hashlib.sha256()
        for file in sorted(path.rglob("*")):
            if file.is_file():
                digest.update(str(file.relative_to(path)).encode() + b"\0")
                with file.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
        provenance = {"checkpoint_sha256": digest.hexdigest()}
    else:
        if not revision:
            raise ValueError("base model requires a pinned revision")
        model = DecisionModel(model_id, revision, device=device, lora_rank=0,
                              max_length=max_length or 4096).eval()
        temperature, method = 1.0, "pretrained_yes_minus_no_no_training"
        provenance = {}
    if max_length is not None:
        if max_length < 1:
            raise ValueError("max length must be positive")
        model.max_length = max_length
    provenance.update(base_revision=model.revision, max_length=model.max_length)
    try:
        provenance["code_commit"] = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        provenance["code_commit"] = None
    # Image channel (deploy pack): JEV_IMAGES=1 attaches the base's untrained vision
    # tower and wraps the scorer. Off (default) nothing is imported and the predictor
    # is byte-identical to upstream. On a base without a vision tower the attach call
    # prints and returns None - the scorer stays unwrapped and image requests get a
    # 422 from Predictor (compile_request carried `images`, but no scorer claims it
    # - refused explicitly below, never silently dropped).
    scorer = TorchScorer(model)
    if os.environ.get("JEV_IMAGES") == "1":
        from .images import ImageScorer, attach

        hook = attach(model)
        if hook is not None:
            scorer = ImageScorer(scorer, hook)
            print("[images] scorer wrapped (JEV_IMAGES=1)", flush=True)
    return Predictor(scorer, model_name=model.model_id, temperature=temperature,
                     batch_size=batch_size, method=method, provenance=provenance, prefix_cache=prefix_cache)
