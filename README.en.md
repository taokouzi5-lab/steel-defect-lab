# DefectLens

**Steel surface defect classification with Grad-CAM explanations and per-inference carbon accounting.**

[![CI](https://github.com/taokouzi5-lab/steel-defect-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/taokouzi5-lab/steel-defect-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-INT8-005CED.svg)](https://onnxruntime.ai/)

## Live Demo

<!-- DEMO_LINK: replace after ModelScope deployment -->

The hosted demo is being deployed to ModelScope 创空间. This section is the
placeholder for the demo URL and will be replaced after deployment. Until
then, run the stack locally (see [Quick Start](#quick-start)).

Repository: [github.com/taokouzi5-lab/steel-defect-lab](https://github.com/taokouzi5-lab/steel-defect-lab)

## Features

- **Training comparison (ResNet-18 vs ResNet-50).** Six NEU-DET defect
  classes, a fixed 80/10/10 split (1,440/180/180, seed 42), AMP, warmup +
  cosine schedule, and early stopping on validation macro-F1. Training
  carbon is measured with the MACE-F `CarbonTracker` (CodeCarbon 3.2.7).
- **Accuracy-gated INT8 quantization.** ONNX opset 17 export with a
  numerical parity check, then static INT8 quantization. A ≤1.5-point
  top-1 gate automatically falls back to dynamic quantization; if both
  variants miss the gate, the FP32 artifact is kept and the failure is
  logged. Results are written back to `artifacts/metrics.json`.
- **Hand-written Grad-CAM.** The Selvaraju et al. algorithm on `layer4`,
  with paired forward/backward hooks removed in a `finally` block. The
  explanation targets the class predicted by the served ONNX model, so a
  heatmap never explains a different class than the API returned.
- **Carbon cost on every prediction.** Per-inference gCO2eq from measured
  MACE-F inference energy, China grid intensity, factory-floor PUE and the
  Fischer-2025 calibration factor. Every response carries the estimate,
  its method label, and a phone-charging equivalent.
- **Security-oriented API.** Magic-byte upload validation (JPEG/PNG),
  5 MB cap, 4096×4096 pixel budget, EXIF correction, memory-only
  processing (no uploads written to disk at the application layer),
  20 requests/min per client, and envelope errors with request IDs.
- **One container, one origin.** The production image builds the React SPA
  and serves it from FastAPI alongside `/api/v1`: non-root, CPU-only
  torch, single worker by default.

## Results

Training was measured on the development machine (RTX 3070 Ti Laptop,
i7-12700H); CPU latency was measured on the same machine. Sources:
`artifacts/metrics.json` and `artifacts/checkpoints/*_run.json`.

| Model | Params | Test acc | Macro F1 | Epochs | Training CO2eq (measured / calibrated) | INT8 top-1 drop | ONNX size (FP32 → INT8) | CPU p50 (FP32 → INT8) |
|---|---|---:|---:|---:|---|---:|---|---|
| ResNet-18 | 11.18 M | 99.44% | 99.44% | 8 | 0.31 g / 0.43 g | 0.00 pt | 42.64 → 10.77 MB | 9.32 → 10.14 ms |
| ResNet-50 | 23.52 M | 98.89% | 98.89% | 6 | 0.45 g / 0.63 g | 1.11 pt | 89.64 → 22.91 MB | 21.90 → 20.35 ms |

**Negative results are reported as measured.** Static INT8 shrinks each
model about 4× and keeps accuracy within the gate, but it does **not**
deliver a latency speedup on this CPU: ResNet-18 got slower (9.3 → 10.1 ms
p50) and ResNet-50 improved only marginally (21.9 → 20.3 ms, within
run-to-run noise). The gate is about accuracy and artifact size, not speed.

Default inference carbon: **7.4×10⁻⁵ gCO2eq per request** (China grid
582 g/kWh, PUE 2.0, T3 calibration 1.39). ResNet-18 uses the T3
representative fallback, so the number is model-independent by
construction; see [docs/carbon-method.md](docs/carbon-method.md).

## Quick Start

### Local, single process

Requirements: Python 3.10+, Node 20+, pip.

```bash
pip install -e "backend/.[dev]"

cd frontend
npm ci
npm run build
cd ..

uvicorn app.main:create_app --factory --app-dir backend --host 0.0.0.0 --port 3000
```

Open <http://localhost:3000>. The API serves the built SPA and `/api/v1`
from the same origin.

Model assets (`artifacts/onnx`, `artifacts/checkpoints`) are gitignored.
Without them the service still boots: `/api/v1/health` returns 200,
`/api/v1/models` lists the four metrics-known models as `available: false`,
and `POST /api/v1/predict` answers `503 model_not_ready` in the standard
envelope.

### Docker

```bash
docker build -t defectlens .
docker run --rm -p 7860:7860 defectlens
```

Open <http://localhost:7860>. Build from a checkout that contains
`artifacts/onnx` and `artifacts/checkpoints` to serve predictions; a bare
checkout builds the same image in degraded mode (asserted by CI).

### Retrain and re-export (optional)

```bash
pip install -e "training/.[dev]"
python training/scripts/download_weights.py --model resnet18
python training/scripts/train.py --model resnet18
python training/scripts/evaluate.py --model resnet18
python training/scripts/export_onnx.py --model resnet18
python training/scripts/quantize.py --model resnet18
```

This path expects the NEU-DET images under `training/data` (not committed).

## Architecture

```
Browser → upload validation → ONNX INT8 → optional Grad-CAM → carbon estimate → JSON envelope
```

The full data flow, single-container topology, directory map and design
decisions are in **[docs/architecture.md](docs/architecture.md)**.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Plain infrastructure probe |
| GET | `/api/v1/models` | Accuracy, latency and carbon metadata |
| GET | `/api/v1/samples` | Whitelisted demo samples |
| GET | `/api/v1/meta` | Carbon method disclosure |
| POST | `/api/v1/predict` | Multipart image classification (optional Grad-CAM) |
| POST | `/api/v1/predict/sample/{id}` | Upload-free demo sample classification |

## Testing and CI

- Backend: 202 pytest tests (`pytest backend/tests`, coverage gate 70%).
- Training: 193 pytest tests (coverage gate 60%, CPU-only in CI).
- Frontend: 80 Vitest tests; 10 Playwright E2E tests against a built SPA.
- CI (`.github/workflows/ci.yml`): three jobs — `test` (training: ruff,
  bandit, pytest), `backend` (ruff, bandit, pytest), and `docker-build`
  (builds the production Dockerfile from a bare checkout and asserts the
  boot + degradation contract on port 7860).

## Data

The dataset is **NEU-DET** (NEU surface defect database): six hot-rolled
steel strip defect classes — crazing, inclusion, patches, pitted_surface,
rolled-in_scale, scratches — with 300 grayscale images per class (1,800
images in total).

> K. Song and Y. Yan, "A noise robust method based on completed local
> binary patterns for hot-rolled steel strip surface defects," *Applied
> Surface Science*, vol. 285, pp. 858–864, 2013. Dataset: Northeastern
> Univ., 2013.

## Carbon methodology

The estimation formula, every constant with its provenance, the three
explicit boundaries, and the v1.1 container-measurement roadmap are in
**[docs/carbon-method.md](docs/carbon-method.md)**. The same disclosure is
served at runtime by `GET /api/v1/meta`, so the documentation cannot drift
from the implementation.

## Related

- **[mace-f / MACE-F](https://github.com/taokouzi5-lab/mace-f)** — the
  manufacturing AI carbon measurement framework whose method-D energy
  measurements, grid intensities, PUE profiles and calibration factors
  this project uses.

## License

MIT License. Copyright (c) 2026 Chunqi Sun. See the
[MIT license text](https://opensource.org/licenses/MIT).
