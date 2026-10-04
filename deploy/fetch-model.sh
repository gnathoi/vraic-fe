#!/usr/bin/env bash
# One-off: download the pinned model weights (~31 GB) into the jfe-models volume (inside a container).
# Uses podman, or docker if podman is absent; override with ENGINE=docker.
set -euo pipefail
ENGINE=${ENGINE:-$(command -v podman || command -v docker)}
REPO=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
REV=5a5a776300a41aaa681dd7ff0106608ef2bc90db
"$ENGINE" volume inspect jfe-models >/dev/null 2>&1 || "$ENGINE" volume create jfe-models >/dev/null
"$ENGINE" run --rm -v jfe-models:/models --entrypoint hf docker.io/vllm/vllm-openai:v0.30.0 \
  download "$REPO" --revision "$REV" --local-dir /models/qwen3-30b-a3b-instruct-2507-fp8
