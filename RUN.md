# Run the temporal smoke model with checkpoints

This guide starts with a trained model and runs inference; you do not need to
train again. Commands use Bash and start in the repository root unless a block
explicitly changes directory. Replace `/absolute/path/...` with your own paths.

## 1. Understand which checkpoint you have

The model follows this pipeline:

```text
Ordered camera frames
  → YOLO smoke boxes
  → tubes (boxes linked across time)
  → stabilized, normalized crops
  → DINOv2 features + temporal transformer
  → tube logits
  → calibration + threshold
  → smoke decision
```

A tube follows one candidate region through several frames. For example, eight
images of one hillside can form a tube showing whether a suspicious region
develops like smoke. The classifier scores that tube, rather than each full
camera image independently.

| File | Contents | How to use it |
| --- | --- | --- |
| `model.zip` | YOLO weights, classifier weights, configuration, and usually a logistic calibrator | Load with `BboxTubeTemporalModel.from_package()` and supply images. Start here. |
| `best_checkpoint.pt` or training `best.ckpt` | Lightning checkpoint for the temporal classifier | Reconstruct the matching classifier, load its weights, and supply prepared tube crops; or package it for full inference. |
| `yolo_weights.pt` or detector `best.pt` | Detector weights | Proposes boxes; it does not contain the temporal classifier. |
| `model_onnx.zip` | Exported classifier, configuration, and calibrator | Load with `OnnxTemporalModel.from_package()` and supply images **and detections**. |

The filename extension does not identify the architecture. Training renames its
Lightning `best.ckpt` to `best_checkpoint.pt`; the contents remain a Lightning
checkpoint. Inside `model.zip`, those weights are called `classifier.ckpt`.

This repository implements the ViT/DINOv2 backbone with a transformer temporal
head. Older GRU, mean-pooling, ResNet, or ConvNeXt checkpoints are not supported
by the current classifier.

## 2. Set up Python and get a packaged checkpoint

Install `uv` if it is missing, then select Python 3.12. The packages require
Python 3.11–3.12; the system Python may be too new.

```bash
python3 -m pip install uv
uv python install 3.12
uv sync --project core --python 3.12 --locked
```

`core` is sufficient for Python inference. Its default development environment
includes the Torch and ONNX extras. The first install downloads substantial
dependencies. `make install` installs all seven packages if you need the entire
repository.

Download the release pinned by this checkout:

```bash
cat api/MODEL_VERSION
make fetch-model
```

This writes `api/models/model.zip`. The fetch command also prepares the `api`
Python environment through `uv run`; it downloads from the public Hugging Face
repository and checks the version recorded inside the archive. At the time this
guide was written, `api/MODEL_VERSION` contains `0.4.0`. Use the file as the
source of truth. To deliberately select that release:

```bash
make fetch-model MODEL_VERSION=0.4.0
```

If you already have `model.zip`, skip downloading it and use its path below.
The package contains the trained backbone weights, so inference does not need
to fetch DINOv2 pretrained weights separately.

Check the Torch environment before requesting a GPU:

```bash
uv run --project core python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

`device=None` selects CUDA, then Apple MPS, then CPU. `device="cpu"` forces
CPU. `device="cuda"` requires a CUDA-capable Torch installation and a working
NVIDIA driver. The `api` package uses CPU Torch wheels by default, so run the
Python examples using the `core` environment.

## 3. Run a sequence of your own images

Prepare frames from one camera in chronological order:

```text
/absolute/path/my_sequence/
  frame_000001.jpg
  frame_000002.jpg
  frame_000003.jpg
  ...
```

Zero-padding matters: lexical sorting puts `frame_10.jpg` before `frame_2.jpg`,
but correctly orders `frame_000002.jpg` before `frame_000010.jpg`. Timestamp
filenames also work when their lexical order matches time. Each filename stem
must identify its frame uniquely. No labels or detection files are needed for
this Torch example. Pass extracted images rather than a video file.

Run this complete example from the repository root:

```bash
uv run --project core python - api/models/model.zip /absolute/path/my_sequence <<'PY'
import json
import sys
from dataclasses import asdict
from pathlib import Path

from temporal_model.core.model import BboxTubeTemporalModel

model_path = Path(sys.argv[1])
sequence_dir = Path(sys.argv[2])
if not sequence_dir.is_dir():
    raise SystemExit(f"Sequence directory does not exist: {sequence_dir}")
paths = sorted(
    (p for p in sequence_dir.iterdir()
     if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
    key=lambda p: p.stem,
)
if not paths:
    raise SystemExit(f"No image frames found in {sequence_dir}")

model = BboxTubeTemporalModel.from_package(model_path, device=None)
frames = model.load_sequence(paths)
out = model.predict(frames, compute_trigger=True)

kept = out.details["tubes"]["kept"]
probabilities = [t["probability"] for t in kept if t["probability"] is not None]
probability = max(probabilities) if probabilities else (0.0 if not kept else None)

print("Device:", model.device)
print("Smoke detected:", out.is_positive)
print("Max calibrated tube probability:", probability)
print("Trigger frame index:", out.trigger_frame_index)
print("Kept tubes:", len(kept))
print("Decision:", out.details["decision"])
print("Preprocessing:", out.details["preprocessing"])

output_path = Path(".cache/run/sequence-output.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(asdict(out), indent=2), encoding="utf-8")
print("Saved:", output_path.resolve())
PY
```

Each invocation overwrites `.cache/run/sequence-output.json`; choose a different
output path if you want to keep several results.

The methods have distinct jobs:

| Method | What it does |
| --- | --- |
| `from_package(Path(...), device=...)` | Reconstructs the models and loads their trained weights, configuration, and calibrator. Call once and reuse the model for multiple sequences. |
| `load_sequence(list_of_paths)` | Builds ordered `Frame` records with IDs, paths, and optional timestamps. It does not sort the paths or run inference. |
| `predict(frames, compute_trigger=True)` | Detects boxes, builds tubes, scores them, and searches for the first threshold crossing. |
| `predict_sequence(list_of_paths)` | Convenience wrapper for `load_sequence()` followed by `predict()` with its defaults. Trigger search is off. |
| `detect(frames)` | Runs only the bundled YOLO. Useful when supplying or caching detections separately. |

If you only need the smoke decision, the shorter form is:

```python
# model and paths are the objects created in the example above.
out = model.predict_sequence(paths)
print(out.is_positive)
```

Interpret the output as follows:

- `is_positive` is the model's smoke decision. Use it as the decision rather
  than inventing a `0.5` cutoff; the archive carries its calibrated threshold.
- `probability` in each kept tube is a calibrated score. The maximum is a useful
  sequence summary, and is the convention used by the API. The raw `logit` is
  an unbounded classifier score, not a probability.
- `trigger_frame_index` is zero-based. An index of `3` means the fourth frame.
  With trigger search disabled, it is `None` even for a positive prediction.
  Short sequences can contain repeated frames from padding; inspect
  `padded_frame_indices` when interpreting their trigger positions.
- `details["preprocessing"]["num_truncated"]` counts discarded input frames.
  Inference keeps the **first** `classifier.max_frames` frames, currently 20 in
  training defaults. Choose the intended window yourself for a longer sequence.
- Short sequences are padded according to the archive configuration, currently
  to six frames in training defaults. Padding duplicates observations; it does
  not provide additional evidence.
- If no tubes survive detection and filtering, the result is negative and
  `kept` is empty. That means no candidate was classified as smoke; inspect the
  detector and tube details when debugging such a result.

Trigger search performs extra classifier work. It is useful for evaluation and
understanding detection delay; leave it off when you only need a decision.

## 4. Load a raw classifier checkpoint directly

Use this route to inspect classifier scores on already prepared tube crops.
A classifier checkpoint alone does not detect boxes, build tubes, or provide
the complete calibrated smoke decision.

Install the training environment and retrieve its DVC artifacts:

```bash
(
  cd train
  uv sync --python 3.12 --locked
  uv run dvc pull
)
```

The DVC remote is private S3 storage. This step requires your authorized AWS
credentials through the normal environment/profile setup. If you already have
the required artifacts locally, skip the pull. Do not commit credentials.

The example expects these files:

```text
train/data/06_models/vit_dinov2_finetune/best_checkpoint.pt
train/data/05_model_input/val/_index.json
train/data/05_model_input/val/<tube_id>/meta.json
train/data/05_model_input/val/<tube_id>/*.png
```

`TubePatchDataset` reads the index and metadata; an arbitrary directory of PNGs
does not satisfy its format. These patches come from the training
`build_model_input` stage. If raw datasets exist but prepared patches do not,
run the data-preparation target, which also reproduces its upstream stages:

```bash
(
  cd train
  uv run dvc repro build_model_input
)
```

Load one tube with the exact architecture used to train the checkpoint:

```bash
uv run --project train python - <<'PY'
from pathlib import Path

import torch
import yaml

from temporal_model.core.temporal_classifier import TemporalSmokeClassifier
from temporal_model.train.dataset import TubePatchDataset
from temporal_model.train.package import classifier_kwargs

params = yaml.safe_load(Path("train/params.yaml").read_text())
cfg = params["train_vit_dinov2_finetune"]
model = TemporalSmokeClassifier(**classifier_kwargs(cfg))

checkpoint = Path("train/data/06_models/vit_dinov2_finetune/best_checkpoint.pt")
# Load checkpoints you trust: this supports the full Lightning checkpoint.
blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
raw = blob["state_dict"] if "state_dict" in blob else blob
state_dict = {
    key.removeprefix("model."): value
    for key, value in raw.items()
    if key.startswith("model.")
} or raw
model.load_state_dict(state_dict, strict=True)
model.eval()

dataset = TubePatchDataset(
    Path("train/data/05_model_input/val"), max_frames=cfg["max_frames"]
)
if not len(dataset):
    raise SystemExit("The validation patch dataset is empty")
item = dataset[0]
patches = item["patches"].unsqueeze(0)  # [1, T, 3, 224, 224]
mask = item["mask"].unsqueeze(0)        # [1, T], True = valid frame
with torch.inference_mode():
    logit = model(patches, mask)
    score = torch.sigmoid(logit)

print("Tube:", item["sequence_id"])
print("Ground truth (0=fp, 1=smoke):", item["label"].item())
print("Logit:", logit.item())
print("Uncalibrated sigmoid score:", score.item())
PY
```

Why these steps matter:

- `classifier_kwargs()` reconstructs the backbone and transformer settings,
  with `pretrained=False` because the checkpoint supplies the weights. Use the
  **training-time** parameters for your checkpoint, even if today's
  `train/params.yaml` has changed.
- Lightning stores classifier weights under keys such as `model.backbone...`.
  The plain classifier expects `backbone...`, so the loader removes `model.`.
  `strict=True` checks for missing, unexpected, and incompatible weights.
- The dataset produces RGB crops normalized with ImageNet mean/std, padded to
  `T=max_frames`. A true mask entry marks a valid crop; false marks padding.
- `eval()` disables training behavior such as dropout. `inference_mode()`
  avoids building gradients. They serve different purposes.
- `sigmoid(logit)` maps a logit into `[0, 1]`, but is **uncalibrated**. The
  packaged model's logistic calibrator also uses tube features, so this score
  need not equal its reported probability.

## 5. Turn your raw checkpoint into a complete `model.zip`

Packaging loads existing weights, scores validation patches, fits a logistic
calibrator using the full pipeline on training sequences, and selects a
threshold on validation sequences. It can be expensive, but it does not
optimize the classifier weights again.

After the training artifacts in section 4 are available:

```bash
(
  cd train
  uv run python -m temporal_model.train.package \
    --variant vit_dinov2_finetune \
    --output data/06_models/vit_dinov2_finetune/model.zip
)
```

The default inputs are the trained checkpoint, the detector named in
`core/src/temporal_model/core/detector.yaml`, prepared validation patches, and
raw train/validation sequences. The current detector weights path is:

```text
train/data/06_models/detectors/yolo11s_swift-swallow_v8.2.0/yolo_weights.pt
```

If that detector file is missing, fetch the declared detector and verify its
SHA-256 using the existing helper:

```bash
uv run --project core python -m temporal_model.core.fetch_detector \
  --output train/data/06_models/detectors/yolo11s_swift-swallow_v8.2.0/yolo_weights.pt
```

For a checkpoint at a custom location, pass the matching inputs explicitly:

```bash
(
  cd train
  uv run python -m temporal_model.train.package \
    --variant vit_dinov2_finetune \
    --params-path /absolute/path/training-params.yaml \
    --checkpoint-path /absolute/path/best_checkpoint.pt \
    --yolo-weights-path data/06_models/detectors/yolo11s_swift-swallow_v8.2.0/yolo_weights.pt \
    --val-patches-dir /absolute/path/prepared-val-patches \
    --raw-train-dir /absolute/path/raw-train \
    --raw-val-dir /absolute/path/raw-val \
    --output /absolute/path/my-model.zip
)
```

The params file must contain the same sections as `train/params.yaml`, with
architecture and crop settings matching the checkpoint and prepared data. Raw
splits use `{wildfire,fp}/<sequence>/images/*.jpg`; data preparation also needs
the corresponding detection labels. The detector hash must match the declared
detector; an arbitrary YOLO checkpoint is rejected.

Run the resulting archive through section 3, replacing `api/models/model.zip`
with its path. Packaging does not publish anything. An unreleased local archive
can omit `model_version` and still load.

`uv run dvc repro package` from `train/` is the tracked pipeline alternative.
DVC can also rerun upstream stages if their inputs changed, including training.
Use the direct packaging command above when you specifically want to package
an existing checkpoint without invoking upstream training.

## 6. Evaluate a checkpoint on a labeled dataset

The evaluator consumes a **packaged** checkpoint and this layout:

```text
/absolute/path/test_split/
  wildfire/<sequence_id>/images/*.jpg
  fp/<sequence_id>/images/*.jpg
```

Ground truth comes from the parent category: `wildfire` means smoke and `fp`
means false-positive/non-smoke. Labels are not supplied to the model as
detections during this evaluation; the bundled YOLO runs on the images.

```bash
uv sync --project eval --python 3.12 --locked
(
  cd eval
  uv run temporal-eval \
    --model-zip ../api/models/model.zip \
    --sequences-dir /absolute/path/test_split \
    --output-dir data/08_reporting/custom/vit_dinov2_finetune \
    --model-name vit_dinov2_finetune-custom \
    --source custom \
    --device cpu
)
```

Use `--device cuda` when CUDA is available, or omit `--device` to auto-select.
Outputs include `metrics.json`, `predictions.json`, `dropped.json`, PR/ROC and
confusion-matrix plots, and viewer records in `results.json`, `details/`, and
`sequences/`. The evaluator enables trigger search for detection-delay metrics.
Empty image sequences are dropped; inference errors abort the run.

To evaluate on the repository's DVC test set using the released archive:

```bash
(
  cd eval
  uv run dvc pull data/01_raw/sequential_test.dvc
  uv run temporal-eval \
    --model-zip ../api/models/model.zip \
    --sequences-dir data/01_raw/sequential_test \
    --output-dir data/08_reporting/test/vit_dinov2_finetune \
    --model-name vit_dinov2_finetune-test \
    --source test
)
```

The test-set pull requires access to the private DVC remote. The tracked
`uv run dvc repro` evaluator instead reads
`eval/data/06_models/vit_dinov2_finetune/model.zip`. To refresh that imported
archive after packaging your own training checkpoint:

```bash
(
  cd eval
  make update-model
  uv run dvc repro
)
```

With reporting artifacts present, launch the read-only Streamlit viewer:

```bash
(
  cd eval
  make app
)
```

Choose the reporting source you evaluated in the viewer. See
[eval/README.md](eval/README.md) for metrics and the Next.js viewer.

## 7. Use the ONNX checkpoint with your own detections

The ONNX archive runs the temporal classifier without Torch. It contains no
YOLO detector, so every frame needs a supplied detection list, including an
empty list when your detector finds nothing.

From the repository root, download the pinned ONNX release, or derive an
archive from your existing Torch package:

```bash
make fetch-model-onnx
```

```bash
# Alternative: export from api/models/model.zip and check Torch/ONNX parity.
make export-onnx
```

Prepare `/absolute/path/detections.json` keyed by image filename stem. This
example illustrates the schema; substitute your detector's real boxes:

```json
{
  "frame_000001": [
    {"class_id": 0, "cx": 0.5, "cy": 0.4, "w": 0.05, "h": 0.03, "confidence": 0.6}
  ],
  "frame_000002": []
}
```

Coordinates are normalized center/width/height `(cx, cy, w, h)`, relative to the
original image, rather than pixel corners. For pixel corners `(x1,y1,x2,y2)` in
an image of width `W` and height `H`, use:

```text
cx = (x1 + x2) / (2W)    cy = (y1 + y2) / (2H)
w  = (x2 - x1) / W       h  = (y2 - y1) / H
```

Using the `core` environment already installed above:

```bash
uv run --project core python - /absolute/path/my_sequence /absolute/path/detections.json <<'PY'
import json
import sys
from pathlib import Path

from temporal_model.core import Detection, FrameDetections
from temporal_model.core.onnx_model import OnnxTemporalModel

model = OnnxTemporalModel.from_package(Path("api/models/model_onnx.zip"))
paths = sorted(Path(sys.argv[1]).glob("*.jpg"), key=lambda p: p.stem)
if not paths:
    raise SystemExit("No JPG frames found")
frames = model.load_sequence(paths)
boxes = json.loads(Path(sys.argv[2]).read_text())
detections = {
    frame.frame_id: FrameDetections(
        frame_idx=i,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        detections=[Detection(**box) for box in boxes[frame.frame_id]],
    )
    for i, frame in enumerate(frames)
}
out = model.predict(frames, frame_detections=detections)
print("Smoke detected:", out.is_positive)
print(json.dumps(out.details, indent=2))
PY
```

The dictionary lookup intentionally fails if a frame's detections are missing:
an empty detection list is different from having no detector result. Padding
can reuse a frame's supplied detections. ONNX defaults to CPU execution.
For a Torch-free deployment, install `temporal-model-core[onnx]` without the
development group; see [core/README.md](core/README.md).

The Torch backend also accepts the same `frame_detections` dictionary. You can
build it from `{fd.frame_id: fd for fd in model.detect(frames)}` to reuse YOLO
results across calls.

## 8. Serve the checkpoint through HTTP

If you want an HTTP service, use Docker Compose from the repository root:

```bash
make fetch-model
make serve
```

In another terminal:

```bash
curl -s http://localhost:8000/health
```

Check for `model_loaded: true`, then open `http://localhost:8000/docs` for the
interactive request schema. The default Compose service reads frames from
MinIO/S3. Files on your host are not automatically available to that service;
upload them to its configured bucket or configure local frame storage and its
mount as described in [api/README.md](api/README.md). The published image uses
CPU Torch; native GPU serving has separate setup in that README.

## Troubleshooting and validation

| Symptom | What to check |
| --- | --- |
| Python version error | Use `uv sync --python 3.12`; the supported range is `>=3.11,<3.13`. |
| `ModuleNotFoundError: temporal_model` or missing Torch | Use the package environment, such as `uv run --project core python`; sync it first. |
| Archive not found | Confirm the path from the current directory. `make fetch-model` writes `api/models/model.zip`. |
| ZIP/manifest error when loading `.pt` | `from_package()` expects `model.zip`. Use section 4 for a raw checkpoint. |
| `UncalibratedModelError` | Use a calibrated archive or package it with calibration. `allow_uncalibrated=True` is an explicit research opt-out; it does not create calibrated probabilities. |
| State-dict or tensor-size mismatch | Use the checkpoint's original architecture parameters and supported ViT transformer implementation. Renaming the checkpoint cannot fix it. |
| Missing `_index.json` | Restore or generate the prepared tube patch dataset rather than passing raw images to `TubePatchDataset`. |
| DVC authentication/access error | Check your authorized AWS profile and access to the package's configured S3 remote. Public released model downloads are a separate route. |
| CUDA unavailable | Run on CPU or fix the Torch/driver environment. Check CUDA in the same package environment used for inference. |
| `trigger_frame_index` is `None` | It is expected for negatives and whenever `compute_trigger` is off. |
| ONNX says it has no detector | Supply `FrameDetections` for every frame; use `[]` for a completed detection with no boxes. |

For further understanding, read [the illustrated pipeline guide](docs/README.md),
[the model API](core/src/temporal_model/core/model.py),
[the shared prediction pipeline](core/src/temporal_model/core/pipeline.py), and
[the packaging implementation](train/src/temporal_model/train/package.py).

Validation of this guide: commands and method signatures were checked against
this checkout, and Python/Bash examples were syntax-checked. Model inference,
GPU checks, downloads, private DVC pulls, and Docker serving were not executed
in the authoring workspace because it has no `uv` installation or checkpoint
artifacts and its system Python is 3.14.
