<p align="center"><img src="docs/brand/vraic-to-the-future.png" width="160" alt="Vraic Futures Engine logo"></p>

# Vraic Futures Engine (vraic-fe)

A probabilistic cohort-component population scenario engine for Jersey, calibrated to Statistics Jersey (10,000-draw
Monte Carlo ensembles on a GPU). A local LLM (vLLM, Qwen3-30B-A3B-Instruct-2507-FP8) turns plain-English questions into
validated scenario changes; the engine, not the LLM, computes every number.

## Requirements

- NVIDIA GPU (tested on 48 GB; the LLM takes 80% of it) with the NVIDIA Container Toolkit and CDI (`nvidia.com/gpu=0`).
- About 31 GB of disk for the model weights, plus the container images.

## Run

```bash
cp .env.example .env        # set the access code, admin code and database password
deploy/fetch-model.sh       # one-off: model weights into the jfe-models volume
```

Then either Compose:

```bash
docker compose up -d        # or: podman compose up -d
```

or Podman quadlets (systemd user services):

```bash
deploy/deploy.sh            # build the image, install deploy/quadlets, start
```

Open http://localhost:8090 and enter the access code. The build downloads and checksums the open data; the LLM takes a
few minutes to load on first start.

## Docs

- [docs/CAPABILITIES.md](docs/CAPABILITIES.md)
- [docs/DATASETS.md](docs/DATASETS.md)

Code: Apache-2.0. Data © Statistics Jersey and Government of Jersey, Open Government Licence – Jersey v1.0.

To share it privately, `tailscale serve` in front of port 8090 works well.
