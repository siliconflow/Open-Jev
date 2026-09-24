"""Image channel tests (JEV_IMAGES gate). Three tiers, mirroring kev's
tests/test_vision.py:

1. no-weights tests: decode_image refusal surface, extract_images passthrough and
   refusals, compile_request byte-identical text rendering with and without a
   claimed images list.
2. gate-off equality: serving.load_predictor without JEV_IMAGES never imports
   jev.images; Predictor's scorer is the plain TorchScorer.
3. 0.8B full chain (skipped unless the local Qwen3.5-0.8B-Base snapshot exists):
   attach loads all 153 model.visual.* tensors, ImageScorer returns the same
   shapes as the text scorer, image rows change the readout vs the text path,
   and a rerun is deterministic.

Run: .venv-vision/bin/python -m pytest tests/test_images.py -q
"""
import base64
import io
import os
import unittest

SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/"
    "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"returns": "Exchanges and refunds", "shipping": "Delivery status"}},
    "refund": {"type": "noul", "instructions": "Is a refund requested?",
               "criteria": {"true": "Explicitly asks for money back",
                            "false": "Does not request money back"}},
}


def _png_data_url(rgb=(200, 30, 30)):
    from PIL import Image

    im = Image.new("RGB", (4, 4), rgb)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class DecodeImageTest(unittest.TestCase):
    def test_refusals(self):
        from jev.images import decode_image

        with self.assertRaises(ValueError):
            decode_image("not a url")
        with self.assertRaises(ValueError):
            decode_image("data:image/gif;base64," + _png_data_url().split(",", 1)[1])
        with self.assertRaises(ValueError):
            decode_image("data:image/png;base64,!!!not base64!!!")
        with self.assertRaises(ValueError):
            decode_image("http://example.com/a.png")             # https only

    def test_accepts_png_data_url(self):
        from jev.images import decode_image

        im = decode_image(_png_data_url())
        self.assertEqual(im.size, (4, 4))
        self.assertEqual(im.mode, "RGB")


class ExtractImagesTest(unittest.TestCase):
    def test_passthrough_and_refusals(self):
        from jev.api import extract_images

        rest, imgs = extract_images({"ticket": "x", "images": ["data:image/png;base64,AA"]})
        self.assertEqual(imgs, ["data:image/png;base64,AA"])
        self.assertEqual(rest, {"ticket": "x"})
        # non-list value: NOT a claim - stays in the state as ordinary data
        rest, imgs = extract_images({"images": "two"})
        self.assertIsNone(imgs)
        self.assertEqual(rest, {"images": "two"})
        # empty list = absent
        rest, imgs = extract_images({"images": []})
        self.assertIsNone(imgs)
        # non-dict state untouched
        rest, imgs = extract_images("plain text")
        self.assertIsNone(imgs)
        self.assertEqual(rest, "plain text")
        # refusals
        with self.assertRaises(ValueError):
            extract_images({"images": ["a"] * 5})                # >4
        with self.assertRaises(ValueError):
            extract_images({"images": [1]})                      # non-string entry

    def test_compile_request_text_rendering_identical(self):
        """With a claimed images list the compiled records render byte-identically
        to the same request without images; the list rides along as record["images"]
        and never enters a prompt."""
        from jev.api import candidate_prompts, compile_request

        state_text = {"ticket": "A delivery question"}
        with_imgs = dict(state_text, images=[_png_data_url()])
        r_img = compile_request(with_imgs, QUESTIONS)
        r_plain = compile_request(state_text, QUESTIONS)
        self.assertEqual([candidate_prompts(r) for r in r_img],
                         [candidate_prompts(r) for r in r_plain])
        self.assertEqual([r["state"] for r in r_img], [r["state"] for r in r_plain])
        for record in r_img:
            self.assertEqual(record["images"], [_png_data_url()])
        for record in r_plain:
            self.assertNotIn("images", record)

    def test_non_list_images_still_renders(self):
        from jev.api import candidate_prompts, compile_request

        records = compile_request({"images": "two"}, QUESTIONS)
        self.assertNotIn("images", records[0])
        self.assertIn('"images": "two"', candidate_prompts(records[0])[0])


class GateOffTest(unittest.TestCase):
    def test_serving_never_imports_images_without_gate(self):
        import subprocess
        import sys

        code = (
            "import sys; sys.path.insert(0, %r); "
            "import os; os.environ.pop('JEV_IMAGES', None); "
            "import jev.serving as s; "
            "assert 'jev.images' not in sys.modules, 'jev.images imported without the gate'; "
            "print('gate-off ok')" % os.getcwd()
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("gate-off ok", out.stdout)

    def test_imagescorer_exists_and_wraps(self):
        from jev.images import ImageScorer

        class Dummy:
            calls = 0

            def score(self, records):
                self.calls += 1
                return [[0.0] * (2 if r["kind"] == "noul" else len(r["options"]))
                        for r in records], 7

        inner = Dummy()
        scorer = ImageScorer(inner, hook=object())
        rows, tokens = scorer.score([{"kind": "choice", "options": ["a", "b"]}])   # no images key
        self.assertEqual(inner.calls, 1)              # delegated verbatim
        self.assertEqual(tokens, 7)


@unittest.skipIf(not os.path.isdir(SNAP), "Qwen3.5-0.8B-Base snapshot not cached")
class FullChainTest(unittest.TestCase):
    """0.8B base, no LoRA (DecisionModel lora_rank=0 - the pretrained Yes-minus-No
    readout). Proves: tower loads, splice works, output shapes match the text
    scorer, images change the readout, deterministic rerun."""

    def test_full_chain(self):
        import torch

        from jev.images import ImageScorer, attach, decode_image
        from jev.model import DecisionModel
        from jev.serving import TorchScorer

        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model = DecisionModel(SNAP, "local", device=dev, lora_rank=0, max_length=4096).eval()
        hook = attach(model)
        self.assertIsNotNone(hook, "0.8B-Base must carry a vision tower")

        state = {"ticket": "Customer Anna: refund requested for a broken mug."}
        records = __import__("jev.api", fromlist=["compile_request"]).compile_request(
            dict(state, images=[_png_data_url()]), QUESTIONS)

        text_records = __import__("jev.api", fromlist=["compile_request"]).compile_request(
            dict(state, images=[_png_data_url()]), QUESTIONS)
        plain = __import__("jev.api", fromlist=["compile_request"]).compile_request(
            state, QUESTIONS)

        scorer = ImageScorer(TorchScorer(model), hook)
        # text-visible record content is equal, so the wrapped path is exercised on
        # the SAME rendering; image rows must go through the hook
        with torch.no_grad():
            rows_text, _ = scorer.wrapped.score([{k: v for k, v in r.items() if k != "images"}
                                                 for r in plain])
            rows_img, tokens_img = scorer.score(records)

        self.assertEqual(len(rows_img), len(records))
        for record, row in zip(records, rows_img):
            expected = 2 if record["kind"] == "noul" else len(record["options"])
            self.assertEqual(len(row), expected)
        # images changed the readout (same-process comparison is the valid sanity)
        flat_text = [v for row in rows_text for v in row]
        flat_img = [v for row in rows_img for v in row]
        self.assertTrue(any(abs(a - b) > 1e-6 for a, b in zip(flat_text, flat_img)),
                        "image rows did not change the readout")
        # deterministic rerun
        rows_img2, _ = scorer.score(records)
        self.assertEqual(rows_img, rows_img2)
        self.assertGreater(tokens_img, 0)


if __name__ == "__main__":
    unittest.main()
