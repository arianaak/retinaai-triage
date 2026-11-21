# API contract (v1)

Frontend and backend can build against this **now** using mock mode — the real
model returns exactly the same structure.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `{status, version, mock, model}` — `mock: true` means fake predictions |
| `POST` | `/predict?heatmap=true` | multipart form, field **`file`** = fundus image (PNG/JPEG/TIFF, ≤ 20 MB) |

Errors: `400` unreadable/empty image, `413` too large, `415` wrong type.

```bash
curl -F "file=@fundus.png" "http://localhost:8000/predict?heatmap=true"
```

## Response

```json
{
  "api_version": "1",
  "package_version": "0.1.0",
  "mock": false,
  "dr": {
    "grade": 2,
    "grade_name": "Moderate NPDR",
    "probabilities": {
      "No DR": 0.05, "Mild NPDR": 0.12, "Moderate NPDR": 0.61,
      "Severe NPDR": 0.15, "Proliferative DR": 0.07
    },
    "expected_grade": 2.07
  },
  "uncertainty": {
    "entropy": 0.52,
    "p_referable_std": 0.06,
    "confidence": 0.61
  },
  "quality": {
    "label": "usable",
    "score": 0.71,
    "reasons": ["Some glare or reflections present."]
  },
  "triage": {
    "category": "referable",
    "priority": 3,
    "action": "Possible referable DR: refer for ophthalmology evaluation.",
    "p_referable": 0.83,
    "p_urgent": 0.22,
    "reasons": ["P(referable) = 0.83"]
  },
  "heatmap_png_base64": "iVBORw0KGgo...",
  "model": {"backbone": "efficientnet_b0", "img_size": 384, "trained_epoch": 8,
            "val_metrics": {"qwk": 0.88, "referable_sensitivity": 0.93}},
  "disclaimer": "Research and educational prototype only. ..."
}
```

## Field notes for the UI

- **`triage.category`** — one of `ungradable`, `urgent`, `human_review`, `referable`, `routine`.
  `priority` 0–4 (lower = show first) → sort the worklist by it.
  Suggested colours: ungradable grey, urgent red, human_review amber, referable orange, routine green.
- **`quality.label`** — `good` / `usable` / `reject`. On `reject`, show `quality.reasons` and ask to retake.
- **Heatmap** — display with `<img src="data:image/png;base64,{heatmap_png_base64}">`.
  Label it "model attention map", not "lesion map". It is `null` when `heatmap=false`.
- **Uncertainty** — `entropy` 0 (certain) → 1 (no idea). Good to show as a small bar.
- **Always display `disclaimer`.**

New fields may be **added** in later stages (e.g. `other_findings` for multi-disease);
existing fields will not be renamed within `api_version` 1.
