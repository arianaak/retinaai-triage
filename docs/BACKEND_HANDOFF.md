# RetinaAI model → RetinaAI Nexus backend: handoff

Model release: **v0.1-baseline** · file `model_inference.pt`
All line numbers refer to the `retinaai-triage` repo. Research/educational prototype only.

> **Short version:** use `export/inference.py` as your `backend/app/ml/inference.py`.
> It is self-contained (numpy, opencv, torch, timm), CPU-only and thread-safe, and its
> preprocessing has been checked **bit-for-bit identical** to the training pipeline.

---

## A. Model & weights

**A1. Architecture.** EfficientNet-B0 (timm) + dropout 0.3 + one linear head with 5 outputs.
Defined in `src/retinaai/model.py:20-29`, with backbone and dropout values from `src/retinaai/config.py:26,28`.
Internally: `timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)` + `nn.Linear(1280, 5)`.
Equivalent plain timm model, which `inference.py` uses:
```python
net = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
```
The weights need a key rename to load into it (see A2). Dropout only matters in training, so the two models are identical in `eval()` mode.

**A2. Saving format.** This is a **checkpoint dict**, not a full pickled model (`src/retinaai/train.py:122-129`, file written at `:158`):
```
{"format_version": 1, "model": <state_dict>, "config": {...training config...},
 "heads": {"dr": 5}, "epoch": int, "metrics": {...validation metrics...}}
```
- There's no optimizer in this file (it's only in `last.pt`).
- There's **no `module.` prefix**, because DataParallel wasn't used.
- The state_dict keys are `backbone.*` and `heads.dr.weight` / `heads.dr.bias`.

Loading on CPU, as done in `export/inference.py` → `load_model`:
```python
ckpt = torch.load(path, map_location="cpu", weights_only=True)
state = {k.removeprefix("backbone."): v for k, v in ckpt["model"].items() if k.startswith("backbone.")}
state["classifier.weight"] = ckpt["model"]["heads.dr.weight"]
state["classifier.bias"]   = ckpt["model"]["heads.dr.bias"]
net = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
net.load_state_dict(state, strict=True); net.eval()
```

**A3. File, size and versions.** `model_inference.pt`, about **16 MB** (~4.0M float32 parameters). Confirm the exact size with `ls -lh`.
Library versions were **NOT RECORDED** in the checkpoint. The model was trained on Kaggle's default GPU image. Get the versions with the Kaggle cell in §K1.

## B. Preprocessing (validation/test path)

**B4. Steps, in order.** The model expects RGB throughout.
1. **Crop black border**: grayscale (RGB→GRAY), keep rows/columns with any pixel > 10, then crop to that bounding box (`src/retinaai/preprocess.py:37-45`).
2. **Pad to square** with black, centred (`preprocess.py:48-54`).
3. **Resize to 512×512**, `INTER_AREA` if downscaling, otherwise `INTER_LINEAR` (`preprocess.py:57-61`). This was the training cache step (`src/retinaai/data.py:43`, saved as lossless PNG).
4. **Repeat steps 1–3 at 384×384** (`data.py:75`). So the model was trained on original → 512 → 384, two resizes. `inference.py` reproduces this from `config.cache_size` and `config.img_size`.
5. **No center crop.**
6. **Ben Graham enhancement**: `blur = GaussianBlur(img, sigma = 384/50 = 7.68)`, then `cv2.addWeighted(img, 4, blur, -4, 128)` (uint8, saturating), then everything outside a circle of radius 0.48·384 is set to 128 (`preprocess.py:70-77`). No CLAHE.
7. **Scale to 0–1** (`/255`) and apply **ImageNet normalisation**: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) (`preprocess.py:14-15, 88-92`).
8. HWC → CHW, then add a batch dimension. The input tensor is (1, 3, 384, 384) float32.

**B5. Image reading.** Training read images with **OpenCV** and converted BGR→RGB straight away (`preprocess.py:31,34`). The model saw **RGB** only, so your `img_rgb` is exactly the right input.

**B6. Code.** See `export/inference.py`: `_crop_to_fundus`, `_pad_to_square`, `_to_square_fundus`, `_ben_graham`, `_normalize_chw` and `RetinaDRModel.square/tensor`. These are copied from `src/retinaai/preprocess.py`.
They were checked against the real training path (including the cache PNG write/read) on 4 image sizes: **max abs difference = 0.0**.

## C. Output

**C7. Output type.** **5-class softmax classifier**. Trained with `CrossEntropyLoss` (`train.py:98`), using label smoothing 0.05 and sqrt-inverse-frequency class weights (`config.py:38-39`, `data.py:86-90`).
It is not a regression or ordinal model, so there are no thresholds. `grade = argmax(probabilities)`.

**C8. Calibration.** **NOT DONE YET**: no temperature scaling is applied. Two side effects to know about for your confidence < 0.6 rule:
- Label smoothing keeps the top probability below ~0.95, so the model rarely gives very confident outputs.
- Class weights shift some probability toward the rare grades (1, 3, 4).

Expect a noticeable share of images to be flagged "needs clinician review". Temperature scaling can be fitted from the saved validation probabilities (`val_probs.npy`) if needed.

## D. Grad-CAM

**D9. Target layer.** The output of `net.forward_features(x)`, i.e. after `net.conv_head` → `net.bn2` (BN + SiLU), shape (1, 1280, 12, 12).
For pytorch-grad-cam this would be `target_layers=[net.bn2]`, or `[model.backbone.bn2]` in the repo's own model.
`inference.py` does **not** use pytorch-grad-cam. That library registers hooks on the shared model, which isn't thread-safe. Instead it computes the gradient with respect to the feature map using `torch.autograd.grad`, which keeps no shared state.

**D10. Tested?** Yes, on Kaggle through `src/retinaai/explain.py:18-34` (`Predictor.predict` → heatmap was produced).
The default target is the **expected grade** Σk·p_k, which highlights regions pushing toward more severe disease. `gradcam(..., target="predicted")` instead uses the predicted class score.
The heatmap is blended over the cropped, squared fundus at ≤512 px, the same crop the model saw.

## E. Data & results

**E11. Dataset and split.** **APTOS 2019** `train.csv` only (3,662 images). Split is **stratified 80/20** with seed 42 (`data.py:35-37`, `config.py:20,43`): 2,929 train / 733 validation.
There is **no separate test set**. The validation set was also used to pick the best epoch, so the metrics below are slightly optimistic. External validation (e.g. Messidor-2, IDRiD) is **NOT DONE YET**.

**E12. Validation results** (best epoch 9, n = 733):

| | Value |
|---|---|
| Accuracy | 0.821 |
| Quadratic weighted kappa | **0.893** |
| Referable (≥2) AUC | 0.980 |
| Referable sensitivity / specificity (P(≥2) ≥ 0.5) | 0.893 / 0.956 |

Per-class recall: grade 0 **0.97** (350/361) · 1 **0.77** (57/74) · 2 **0.74** (147/200) · 3 **0.41** (16/39) · 4 **0.54** (32/59)

Confusion matrix (rows = true, columns = predicted):
```
        0    1    2    3    4
0     350    9    2    0    0
1       2   57   14    0    1
2       0   30  147   20    3
3       1    1   15   16    6
4       0    5   15    7   32
```
**Urgent cases (grades 3–4, n = 98):**
- 61 (62%) were predicted as 3 or 4.
- 91 (**93%**) were predicted ≥ 2, so they would still be referred.
- **7 (7%) were predicted 0–1**: these are the dangerous misses.
- Of the 85 images predicted 3–4, 61 (72%) were truly 3–4.

The model is weakest at telling grade 3 apart from grade 2.

**E13. Class imbalance.** Handled with class-weighted loss (sqrt inverse frequency), not resampling.
The model is **not biased toward grade 0**: it predicted grade 0 for 353 images vs 361 true, and only 3 non-zero images were called 0. It **over-predicts grade 1** (102 predicted vs 74 true), mostly by calling grade-2 images grade 1.

**E14. Ungradable class.** There is **none**. The model was trained only on APTOS images, which all have a grade from 0 to 4.
It will always output a grade, even for a bad image, so your quality check before the model is essential. The repo also has a heuristic check in `src/retinaai/quality.py`, which also rejects images that aren't fundus photos.

## F. Speed

**F15. CPU speed.** **NOT MEASURED YET.** Estimate for EfficientNet-B0 at 384 px on a laptop CPU: about 0.1–0.4 s for `predict` and 2–3× that for `gradcam`, so well under the "couple of seconds" target. The model is ~16 MB on disk and ~100–200 MB of RAM at runtime.
Measure it with: `python inference.py model_inference.pt some_image.png`, which prints the times.

**F16. TTA / ensemble.** **No ensemble.** `inference.py` runs a **single pass** by default, which is exactly how the validation metrics were computed (`train.py:51-60`).
`load_model(path, tta_flip=True)` adds a horizontal-flip average (2× time); it wasn't validated separately.
The repo's own `Predictor` additionally uses MC-dropout × flip (16 passes) for uncertainty estimates. Your triage doesn't need that.

## G. Licensing & demo images

**G17. License.** **NOT VERIFIED.** APTOS 2019 is Kaggle competition data, with its use governed by the competition rules (Rules tab, "Competition Data" section). Read that section before showing images in a **public** demo. Treat the data as non-commercial research and education only, and credit APTOS / Aravind Eye Hospital.

**G18. Demo images.** **NOT DONE YET.** Pick them from the validation set (images the model never trained on) with the Kaggle cell in §K3. Be transparent that these are hand-picked correct examples.

## H. Deliverables

**H19.** `export/inference.py` provides `load_model`, `predict` and `gradcam` with your signatures. It is CPU-only, runs `eval()` with `inference_mode` (a stricter `no_grad`), keeps no global mutable state, and returns PNG bytes of ≤512 px width.

**H20. Checklist (what you'll receive):**
- [ ] `model_inference.pt`, via the GitHub Release `v0.1-baseline` or Drive (never committed to git)
- [ ] `inference.py` (this repo: `export/inference.py`)
- [ ] Versions printed by §K1 (torch, timm, opencv, numpy)
- [ ] §K2 check passed (inference.py output matches the saved validation predictions)
- [ ] 5 demo image file names from §K3
- [ ] This document

---

## K. Kaggle cells (run in the training notebook)

**K1: Versions and file size**
```python
import torch, timm, cv2, numpy
print("torch", torch.__version__, "| timm", timm.__version__, "| opencv", cv2.__version__, "| numpy", numpy.__version__)
!ls -lh /kaggle/working/outputs/baseline/model_inference.pt
```

**K2: Check inference.py against the training-time validation predictions**
```python
import sys, numpy as np, pandas as pd, cv2
sys.path.insert(0, "export"); import inference as inf
DATA = "/kaggle/input/competitions/aptos2019-blindness-detection"; RUN = "/kaggle/working/outputs/baseline"
m   = inf.load_model(f"{RUN}/model_inference.pt")
val = pd.read_csv(f"{RUN}/val_split.csv", dtype={"id": str})
ref = np.load(f"{RUN}/val_probs.npy")                     # saved during training, same order as val_split.csv
diffs, agree = [], 0
for i in range(50):
    rgb = cv2.cvtColor(cv2.imread(f"{DATA}/train_images/{val.id[i]}.png"), cv2.COLOR_BGR2RGB)
    out = inf.predict(m, rgb)
    diffs.append(np.abs(np.array(out["probabilities"]) - ref[i]).max()); agree += out["grade"] == ref[i].argmax()
print(f"max prob difference {max(diffs):.4f} (expect < ~0.02: training used GPU half precision) | grade agreement {agree}/50")
open("gradcam_check.png", "wb").write(inf.gradcam(m, rgb))
```

**K3: Demo images (correct and confident, one per grade, validation set only)**
```python
val["pred"], val["conf"] = ref.argmax(1), ref.max(1)
picks = val[val.pred == val.label].sort_values("conf", ascending=False).groupby("label").head(1).sort_values("label")
print(picks[["id", "label", "conf"]].to_string(index=False))
```
