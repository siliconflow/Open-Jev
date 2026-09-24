# Open-Jev serving image for SF GPU Functions (4090 g1/g2, open-jev-{2b,9b} routes).
#
# Mirrors the kev serving image pattern (see kev/Dockerfile in the
# os_jev_exp workspace): Open-Jev is a pure PyTorch/peft stack (jev.server is
# a stdlib ThreadingHTTPServer; scorer = torch + transformers + peft). No
# vLLM in the serving path, so this uses a slim python base with user-space
# CUDA (the pip torch wheel bundles its CUDA runtime; the host driver comes
# from the SF GPU Function runtime).
#
# Dependency pins follow the upstream lockfile exactly (requirements.txt at
# the pinned release): torch>=2.8, transformers==5.10.2, peft==0.19.1,
# accelerate==1.13.0, datasets==5.0.0, safetensors. The cu124 index tops out
# at torch 2.6.0 for cp312 — which is BELOW the torch>=2.8 pin — so we use
# the cu126 index here (has 2.8.x for cp312; 4090 sm_89 is covered by every
# CUDA 12.x build). This diverges from kev only because upstream Open-Jev
# pins torch>=2.8; kev's own range allowed 2.6.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1) Deps layer (cached; changes rarely). torch first from the cu126 index,
# then the rest from PyPI against the upstream pins. datasets is only needed
# for training-side imports; jev.server never imports it, but installing it
# keeps `pip install -e .` extras parity — cheap, avoids surprises.
# pillow + torchvision serve the opt-in image channel (JEV_IMAGES=1): the
# transformers qwen2_vl image processor imports torchvision unconditionally.
# torchvision comes from the same cu126 index so its bundled-CUDA torch dep
# resolves against the wheel above instead of pulling a second torch.
COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
      "torch>=2.8,<2.10" \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
      -r requirements.txt

# Optional runtime extras (small): phone-numbers control task. The doom extra
# (vizdoom) is a game-rendering dependency and is NOT shipped.
RUN pip install --no-cache-dir "phonenumbers==9.0.14"

# 2) Source layer (installed separately so source-only commits rebuild fast).
# The examples/ tree (~384K) is served by jev.server as its task-lab UI, so
# it must be present next to the package.
COPY pyproject.toml README.md LICENSE ./
COPY jev ./jev
COPY examples ./examples
RUN pip install --no-cache-dir --no-deps .

# Non-root runtime, same convention as the agent/kev/nimble images.
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /workspace/ckpt \
    && chown -R appuser:appuser /workspace
USER appuser

# Weights come from the SF platform's injected HF proxy at runtime. The hub's
# Xet middleware would bypass the proxy (same failure mode as kev/jev-proxy);
# disable Xet, lengthen the socket timeouts for the slow proxy channel.
ENV HF_HUB_DISABLE_XET=1 \
    HF_HUB_ETAG_TIMEOUT=30 \
    HF_HUB_DOWNLOAD_TIMEOUT=600 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
# The SF cloud-function yaml overrides `command` to add the checkpoint
# bootstrap (hf download) in front of the serve step; see
# sf-jev-deploy/deploy/openjev/openjev-2b-4090.yaml in the os_jev_exp workspace.
CMD ["python", "-m", "jev.server", "--checkpoint", "/workspace/ckpt/package/checkpoint", \
     "--host", "0.0.0.0", "--port", "8000", "--device", "cuda:0", \
     "--max-length", "4096", "--batch-size", "1"]
