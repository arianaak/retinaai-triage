# RetinaAI Triage

**Open, uncertainty-aware diabetic retinopathy screening support for low-cost imaging.**
Grades fundus images (0–4), checks image quality, estimates its own uncertainty,
shows an attention heatmap and assigns a triage category.

> ⚠️ Research and educational prototype. **Not a medical device**, not validated for
> clinical use, not for diagnosis or autonomous decisions.

## Who does what

| Role | Start here |
|---|---|
| **ML** (model) | `notebooks/train_kaggle.ipynb` on a Kaggle GPU → `src/retinaai/` |
| **Backend** | `api/app.py` — runs in **mock mode** today, real model later via one env var |
| **Frontend** | `docs/API_CONTRACT.md` — the exact JSON you will receive |

## Repository layout

```
src/retinaai/
  preprocess.py   loading, fundus crop, Ben Graham enhancement (shared train/inference)
  augment.py      training augmentations + simulated smartphone / low-quality capture
  quality.py      heuristic image-quality check (good / usable / reject + reasons)
  data.py         generic CSV dataset (APTOS now; DDR / IDRiD / Messidor later)
  model.py        timm backbone + multi-task heads, MC-dropout
  train.py        training loop (AMP, class weighting, resume, staged fine-tuning)
  metrics.py      QWK, referable sensitivity / specificity / AUC
  explain.py      Grad-CAM attention maps
  triage.py       transparent triage rules
  result.py       builds the API JSON (single source of truth)
  predict.py      Predictor: image -> full result
  mock.py         MockPredictor: same JSON, no model needed
api/app.py        FastAPI reference server
scripts/quality_report.py   calibrate quality thresholds on real data
notebooks/train_kaggle.ipynb
tests/
```

## Quickstart

### ML — train on Kaggle
1. Push this repo to GitHub.
2. New Kaggle notebook → import `notebooks/train_kaggle.ipynb` → GPU + Internet on →
   add input **APTOS 2019 Blindness Detection**.
3. Set `REPO` in the first cell and run top to bottom.

Locally / Colab: `pip install -e ".[train]"` then
```bash
python -m retinaai.train --data_dir path/to/aptos --cache_dir cache --max_samples 600 --epochs 2
```
All options: `python -m retinaai.train --help`.

### Backend — run the API
```bash
pip install -e ".[api]"
RETINAAI_MOCK=1 uvicorn api.app:app --reload --port 8000          # no model needed
# later, with weights:
pip install -e ".[train,api]"
RETINAAI_CHECKPOINT=weights/model_inference.pt uvicorn api.app:app --port 8000
```
Interactive docs: http://localhost:8000/docs

Or call the model from your own service:
```python
from retinaai.predict import Predictor
result = Predictor("weights/model_inference.pt").predict(image_bytes)
```

### Tests
```bash
pip install -e ".[dev]" && pytest -q      # model tests auto-skip without torch
```

## Sharing model weights
Never commit `.pt` files (they're in `.gitignore`). After training, download
`model_inference.pt` from the Kaggle Output tab and attach it to a **GitHub Release**
(e.g. `v0.1-baseline`). Teammates download it into `weights/`.

## Roadmap
| Stage | Goal | Data |
|---|---|---|
| 0 | Pipeline smoke test | APTOS subset |
| 1 | DR grading baseline + heatmap + triage | APTOS 2019 |
| 2 | Smartphone robustness (degradation fine-tuning), uncertainty evaluation | APTOS + simulated; mBRSET for real smartphone validation |
| 3 | Learned quality head replacing heuristics | EyeQ (Good / Usable / Reject) |
| 4 | Multi-disease findings (glaucoma suspect, AMD, …) as extra heads | RFMiD / ODIR |
| 5 | External validation | Messidor-2 / IDRiD (DME labels) |

## Known limitations
- APTOS labels are noisy and from one population/camera setup → expect domain shift.
- 0–4 grade does not capture diabetic macular edema separately.
- Grad-CAM shows where the model looked, not why.
- Quality thresholds are heuristics until calibrated (`scripts/quality_report.py`).
- Smartphone robustness is tested on *simulated* degradation until validated on real images.

## Data
Datasets are **not** redistributed here. Each has its own license (mostly
research / non-commercial); obtain them from their official sources.
