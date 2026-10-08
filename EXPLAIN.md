# Understanding this repository for your fire/smoke project

This guide explains the code in this checkout. The commands and examples below have been checked against the source; training and inference have not been executed as part of writing this guide.

## 1. What does this model actually do?

**Yes: if you already have a trained YOLO model, you can keep it and train the temporal classifier here. This training pipeline does not update YOLO.**

Think of two questions:

1. **YOLO:** “Where is a region that might contain smoke?”
2. **This classifier:** “Looking at that region over several frames, is it really smoke?”

For example, your YOLO segmentation model might mistake a cloud for smoke. A temporal classifier gets several crops of that cloud and learns to distinguish these false alarms from actual smoke. Whether it improves your detector must be measured on your own held-out sequences.

The pipeline is:

```text
Ordered images of the same scene
       ↓
YOLO proposes bounding boxes in each image
       ↓
Link overlapping boxes across time → “tubes”
       ↓
Crop the region across frames → 224 × 224 RGB patches
       ↓
DINOv2 extracts a feature vector from each patch
       ↓
Temporal transformer combines the sequence of feature vectors
       ↓
One smoke/non-smoke score per tube
       ↓
Decision rule → smoke/non-smoke for the sequence
```

A **tube** is a chain of boxes following roughly the same region over time. It is not a video file or a segmentation mask. Matching uses box overlap, called intersection over union (IoU), with some tolerance for missed detections.

The default crop is **stabilized**: it uses a common window covering the tube's observed boxes. This reduces artificial movement caused by a jittering crop. The window is enlarged for context and resized to 224 × 224.

### What is trained?

The current `train/params.yaml` specifies:

| Component | Updated during classifier training? |
|---|---|
| Your YOLO detector/segmenter | No |
| Most of pretrained DINOv2 | No; frozen |
| Last DINOv2 transformer block | Yes; `finetune_last_n_blocks: 1` |
| Temporal transformer and binary classification layer | Yes |

The DINOv2 backbone is `vit_small_patch14_dinov2.lvd142m`. It starts from pretrained weights, so you are not training DINOv2 from scratch. Setting `finetune: false` freezes the entire backbone and trains only the temporal head.

Packaging can additionally fit a small **logistic calibrator**: it combines the classifier score and tube/detection features into a final probability. This is separate from updating YOLO or DINOv2.

### How this relates to segmentation and fire

The implemented task is **binary smoke classification**. It does not predict fire as a separate class and does not output pixel masks. A fire-only image is not automatically a positive example for this smoke classifier.

Your YOLO segmentation model can supply its bounding boxes to this pipeline. Keep its masks in your own application if you need segmentation output. The temporal model currently uses only the boxes and RGB crops; it does not refine those masks.

A future application could use temporal scores to decide which smoke detections to retain, but associating its tube decisions back to your YOLO masks requires integration code. Training a fire verifier, or a joint fire/smoke verifier, also requires changes to the labels, model task, and decision logic.

## 2. Brief repository map

```text
temporal-model/
├── core/       Model architecture, boxes/tubes, crops, inference, loading
├── train/      Data preparation, classifier training, calibration, packaging
├── eval/       Evaluation of the full packaged pipeline
├── api/        HTTP service for predictions
├── viewer/     Browser interface for inspecting evaluation results
├── benchmark/  Speed and resource measurements
├── monitor/    Replay production decisions
├── triage/     Help sort annotation candidates
├── docs/       Illustrated explanations, specifications, runbooks
└── Makefile    Shortcuts for common commands
```

Start with **`train/` and `core/`**. You can train without starting the API or viewer.

Python code lives under `<package>/src/temporal_model/<package>/`. Each package has its own dependency settings in `pyproject.toml`, environment managed by `uv`, and `tests/` directory.

## 3. What is a Makefile?

A Makefile is a collection of named command shortcuts. `make` is the program that reads it.

For example, `train/Makefile` contains:

```makefile
install:
    uv sync

test:
    uv run pytest tests/ -v
```

The actual file uses tabs before commands. You do not need to edit it to train.

In the `train/` directory:

| Shortcut | Command it runs | Meaning |
|---|---|---|
| `make install` | `uv sync` | Install the package and its dependencies |
| `make test` | `uv run pytest tests/ -v` | Run automated tests; this does not train |
| `make lint` | `uv run ruff check .` | Check Python style and common mistakes |
| `make format` | `uv run ruff format .` | Format Python code |

`uv run` runs a command inside the package's managed Python environment. `python -m temporal_model.train.train` means “run that Python module,” allowing its package imports to resolve correctly.

**The directory matters.** Root `make install` installs all seven Python packages. `make -C train install` installs just the training package and its dependencies, including `core`. Root `make serve` starts Docker services. There is no `make train` target here.

## 4. Data for inference

### Prepare an ordered sequence of images

Use actual frames from the same camera/scene in chronological order:

```text
my_sequence/
├── frame_000001.jpg
├── frame_000002.jpg
├── frame_000003.jpg
└── ...
```

Use zero-padded numbering so sorting filenames gives the correct time order. Avoid joining unrelated still images or different camera views into one sequence. Keep the sampling interval similar between training and deployment: ten frames collected over one second convey different temporal behavior from ten frames collected over several minutes.

For inference, you need images and a trained temporal model. Ground-truth labels are unnecessary. With the bundled YOLO backend, images alone are enough; with your own external detector, also supply its per-frame boxes.

The current configuration uses at most **20 frames per call**, taking the first 20. Short sequences are padded to at least six frames by duplicating existing frames. Duplication allows computation but adds no new temporal information. For a long video, your application must extract frames and make repeated sequence/window calls; the model does not automatically scan an entire video.

### Try the existing released pipeline

From the repository root:

```bash
make -C core install
make -C api install
make fetch-model
```

`make fetch-model` downloads the released package pinned by `api/MODEL_VERSION` to `api/models/model.zip`. **It includes Pyronear's classifier and YOLO, not your own YOLO.** No training is performed by this command.

Save this example as `core/try_inference.py`, replacing the image directory:

```python
from pathlib import Path

from temporal_model.core.model import BboxTubeTemporalModel

model = BboxTubeTemporalModel.from_package(
    Path("../api/models/model.zip"),
    device=None,  # Automatically select CUDA, MPS, or CPU.
)
paths = sorted(Path("/absolute/path/to/my_sequence").glob("*.jpg"))
if not paths:
    raise ValueError("No input frames found")

output = model.predict_sequence(paths)
print("Smoke detected:", output.is_positive)
for tube in output.details["tubes"]["kept"]:
    print("Tube:", tube["tube_id"], "Probability:", tube["probability"])
```

Run from `core/` with `uv run python try_inference.py`. A negative result can mean either rejected candidates or no usable YOLO tube; inspect `output.details` to distinguish them.

### Supplying your own YOLO detections

The Python interface accepts `model.predict(frames, frame_detections=...)`, where:

- `frames` comes from `model.load_sequence(paths)`.
- `frame_detections` maps each image stem to a `FrameDetections` object.
- Each `Detection` contains `class_id`, normalized `cx, cy, w, h`, and confidence.
- Include every frame, with an empty detection list if YOLO found nothing.

The HTTP API offers the same idea through its `detections` request field, but uses normalized corner coordinates **`x_min, y_min, x_max, y_max`** instead. Keep these coordinate formats distinct.

**Filter your YOLO output to smoke boxes before supplying it.** The default inference helper takes all predicted classes, and tube matching does not enforce class identity. A mixed fire/smoke model otherwise sends both classes into the smoke pipeline. Replacing the detector can also change score distributions; the released package's calibration is not established for your detector.

Section 7 gives a local example using your own trained classifier and YOLO together.

## 5. Data for training

### You need sequences, labels, and both positive and negative examples

Your usual segmentation dataset of independent images and polygons is insufficient by itself. You need sequences of the same scene and boxes identifying the candidate region in each frame.

For this guide, put your own data here, separately from the repository's imported Pyronear dataset:

```text
train/data/01_raw/my_sequences/
├── train/
│   ├── wildfire/
│   │   └── smoke_camera01_event001/
│   │       ├── images/
│   │       │   ├── frame_000001.jpg
│   │       │   └── frame_000002.jpg
│   │       └── labels/
│   │           ├── frame_000001.txt
│   │           └── frame_000002.txt
│   └── fp/
│       └── cloud_camera02_event001/
│           ├── images/    # Ordered frames of a false alarm
│           └── labels/    # Candidate boxes predicted by YOLO
└── val/
    ├── wildfire/
    │   └── smoke_camera03_event002/
    │       ├── images/
    │       └── labels/
    └── fp/
        └── fog_camera04_event002/
            ├── images/
            └── labels/
```

The displayed two frames illustrate the layout; prepare longer examples. The default tube filter requires at least **four tube frames** and **two observed detections**.

Here `wildfire` is the directory name the code requires for **positive smoke examples**; `fp` means **false positive**, the negative class. The folder determines the target:

```text
wildfire/ → label 1 → real smoke
fp/       → label 0 → candidate region that is not real smoke
```

This label is not inferred from the numeric class ID in the `.txt` file. Consequently, a YOLO-predicted smoke box under `fp/` is a negative training example: YOLO guessed smoke, but a human knows it is cloud/fog/dust/etc.

Use globally unique sequence directory names across categories and splits. Several outputs are keyed only by the sequence name, so duplicate names can collide.

### Bounding-box label format

The reader supports exactly these two formats, one candidate per line:

```text
class_id cx cy width height
class_id cx cy width height confidence
```

Coordinates are normalized to the image dimensions. For example, if your smoke class is 81:

```text
81 0.50 0.40 0.20 0.10
```

This describes a box centered halfway across the image and 40% down it, with width 20% and height 10%. For a false alarm predicted by YOLO:

```text
81 0.50 0.40 0.20 0.10 0.73
```

Use your model's actual smoke class ID; check its class names rather than assuming 81. Five-column lines get confidence 1.0. Six-column lines preserve the supplied confidence. The intended existing dataset uses annotated smoke boxes for positives and YOLO predictions for negatives.

The training reader does **not** filter classes. Prepare smoke candidate boxes only. Empty/missing per-frame labels mean no detection in that frame, not a negative classification target. A sequence with no usable boxes cannot produce a crop and is dropped. Therefore, useful negatives are sequences where YOLO actually makes smoke-like false detections, not only clean backgrounds with empty labels.

### Converting your segmentation annotations

YOLO polygon labels such as:

```text
81 x1 y1 x2 y2 x3 y3 ...
```

are not accepted as polygons by this reader. Convert each smoke polygon to its enclosing box:

```text
x_min = minimum polygon x coordinate
x_max = maximum polygon x coordinate
y_min = minimum polygon y coordinate
y_max = maximum polygon y coordinate

cx = (x_min + x_max) / 2
cy = (y_min + y_max) / 2
width  = x_max - x_min
height = y_max - y_min
```

If the polygon coordinates are already normalized, these box coordinates are normalized too. Preserve the original polygons for your segmentation project; write converted boxes into this temporal dataset. For negative clips, extract boxes from your trained YOLO's predictions and have their negative status verified.

Use lowercase `.jpg` frames with matching `.txt` stems. The current training discovery only finds `*.jpg`; PNG patches are generated later. Merely renaming a PNG extension is not an image conversion.

### Choose coherent clips and independent splits

Make each positive clip center on a real smoke event and keep its candidate region consistent. Training retains **one longest tube per sequence** after merging fragments. It assigns the sequence's label to that tube. If a positive clip contains several unrelated candidate regions, it could select a non-smoke region and give it a positive label.

Split by original event/video, with camera separation where practical. Adjacent frames or overlapping windows of the same event should not occur in both training and validation. Keep a separate test set for the final pipeline evaluation. Include actual smoke and representative false alarms in both training and validation; negatives alone cannot measure smoke recall.

## 6. How to train on your own data

### Step A: Install the training environment

You need `uv`, Python 3.11 or 3.12, and preferably a CUDA-capable GPU. CPU training is supported but can be slow. Initial installation and pretrained DINOv2 loading can require downloads.

```bash
cd /home/tommy/Desktop/temporal-model/train
uv sync
```

This is equivalent to `make install` **inside `train/`**. The training script automatically selects an available accelerator and uses one device. It prints whether CUDA is available.

### Step B: Review `params.yaml`

The important current settings are:

| Setting | Current value | Meaning |
|---|---|---|
| `train_vit_dinov2_finetune.batch_size` | 16, inherited from `_vit_defaults` | Tubes in one training batch |
| `train_vit_dinov2_finetune.max_frames` | 20, inherited | Maximum patches per tube |
| `train_vit_dinov2_finetune.max_epochs` | 30, inherited | Maximum passes through training data |
| `train_vit_dinov2_finetune.early_stop_patience` | 5, inherited | Stop after validation F1 stops improving |
| `train_vit_dinov2_finetune.learning_rate` | 0.0001, inherited | Temporal head learning rate |
| `train_vit_dinov2_finetune.backbone_lr` | 0.00001 | Unfrozen DINOv2 learning rate |
| `train_vit_dinov2_finetune.finetune_last_n_blocks` | 1 | Number of DINOv2 blocks updated |
| `augment.enabled` | true | Training augmentation; validation remains unaugmented apart from normalization |

YAML entries such as `<<: *vit_defaults` inherit settings from another block. You can add an explicit `batch_size: 4` directly under `train_vit_dinov2_finetune` to override the inherited value if GPU memory is insufficient. Keep `patch_size` and `img_size` at 224 for this implementation; the dataset allocates 224 × 224 tensors.

### Step C: Build tubes and patches for both splits

Run this Bash block **from `train/`** after preparing section 5's dataset:

```bash
for split in train val; do
  uv run python -m temporal_model.train.truncate \
    --input-dir "data/01_raw/my_sequences/$split" \
    --output-dir "data/01_raw/my_sequences_truncated/$split" \
    --max-frames 20

  uv run python -m temporal_model.train.build_tubes \
    --input-dir "data/01_raw/my_sequences_truncated/$split" \
    --output-dir "data/03_primary/my_tubes/$split" \
    --iou-threshold 0.2 \
    --max-misses 2 \
    --min-tube-length 4 \
    --min-detected-entries 2 \
    --merge-iomin 0.3 \
    --merge-prox-factor 1.0 \
    --merge-max-gap 10

  uv run python -m temporal_model.train.build_model_input \
    --tubes-dir "data/03_primary/my_tubes/$split" \
    --raw-dir "data/01_raw/my_sequences_truncated/$split" \
    --output-dir "data/05_model_input/my_patches/$split" \
    --context-factor 1.5 \
    --patch-size 224 \
    --stabilize true
done
```

`for split in train val` repeats the three commands once for each split. The trailing `\` continues a command onto the next line.

These explicit preprocessing arguments match current `params.yaml` defaults. **They do not automatically read that file.** If you change preprocessing settings, update these arguments too. Packaging/inference configuration must use matching settings.

The steps do the following:

1. **Truncate:** copy the first 20 frames and matching labels. It skips already-existing destination sequences; source edits will not automatically refresh those copies. Use a fresh output directory when regenerating changed data. Ensure smoke is represented in the retained first 20 frames.
2. **Build tubes:** read label boxes, link and merge them, select the longest tube, and interpolate gaps. **It does not run YOLO.** `_summary.json` records counts and dropped examples. Use fresh tube output directories if removing/renaming sequences, since old JSON files are not automatically cleared.
3. **Build model input:** crop patches and write metadata. This step **recreates its output directory**, so use it only for generated patches.

Inspect the summaries, crop images, and `my_patches/<split>/_index.json` before training. Confirm both labels 0 and 1 survive in each split; check the printed crop error count. Wrong label formats can silently produce no detections.

Generated patch layout:

```text
train/data/05_model_input/my_patches/train/
├── _index.json                  # Tube list and binary targets
└── smoke_camera01_event001/
    ├── meta.json                # Ordered patch filenames and metadata
    ├── frame_00.png             # 224 × 224 RGB crop
    ├── frame_01.png
    └── ...
```

The training dataset loads a tube as `[20, 3, 224, 224]`: time, RGB channels, height, width. Shorter tubes are zero-padded and receive a validity mask, telling the transformer which entries are real. These patches and JSON files are generated; you do not create them manually.

### Step D: Train the classifier

Still from `train/`:

```bash
uv run python -m temporal_model.train.train \
  --train-dir data/05_model_input/my_patches/train \
  --val-dir data/05_model_input/my_patches/val \
  --output-dir data/06_models/my_smoke_model \
  --params-path params.yaml \
  --params-key train_vit_dinov2_finetune
```

This command reads `params.yaml`, loads pretrained DINOv2, and trains the binary temporal classifier on your generated patches. YOLO weights are not required for this step. The loss is binary cross-entropy: it encourages smoke tubes to score positive and false-alarm tubes to score negative.

The checkpoint is selected using the highest validation F1, calculated at a sigmoid score threshold of 0.5. Early stopping can finish before 30 epochs. Outputs include:

```text
train/data/06_models/my_smoke_model/
├── best_checkpoint.pt           # Best Lightning checkpoint, not YOLO weights
├── csv_logs/                    # Training and validation metrics
├── tb_logs/                     # TensorBoard logs
├── batch_samples/               # Example augmented training batches
└── plots/training_curves.png    # Curves, if plotting succeeds
```

Save the exact `params.yaml` used alongside your experiment records. Inference must reconstruct the classifier with the same architecture settings.

## 7. Try your checkpoint with your own YOLO

For an initial local experiment, you can instantiate the pipeline directly without building `model.zip`. Save the following as `train/try_my_model.py`. Replace the YOLO path, image directory, and smoke class ID.

This example uses repository helper functions, including the internal `_load_classifier_from_ckpt`; it is a local learning example rather than a standalone stable public API.

```python
from pathlib import Path

import yaml
from ultralytics import YOLO

from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.train.package import _load_classifier_from_ckpt, build_config

YOLO_WEIGHTS = Path("/absolute/path/to/your/best.pt")
SEQUENCE_DIR = Path("/absolute/path/to/my_sequence")
SMOKE_CLASS_ID = 81  # Replace with the ID in your YOLO model's class names.


class SmokeOnlyYOLO:
    def __init__(self, weights: Path):
        self.yolo = YOLO(str(weights))

    def predict(self, paths, **kwargs):
        return self.yolo.predict(paths, classes=[SMOKE_CLASS_ID], **kwargs)


params = yaml.safe_load(Path("params.yaml").read_text())
classifier = _load_classifier_from_ckpt(
    Path("data/06_models/my_smoke_model/best_checkpoint.pt"),
    params["train_vit_dinov2_finetune"],
)
config = build_config(
    params,
    params["train_vit_dinov2_finetune"],
    threshold=0.0,  # Raw logit 0 corresponds to sigmoid score 0.5.
    aggregation="max_logit",
    logistic_threshold=None,
)
model = BboxTubeTemporalModel(
    yolo_model=SmokeOnlyYOLO(YOLO_WEIGHTS),
    classifier=classifier,
    config=config,
)
paths = sorted(SEQUENCE_DIR.glob("*.jpg"))
if not paths:
    raise ValueError("No input frames found")

output = model.predict_sequence(paths)
print("Smoke detected:", output.is_positive)
for tube in output.details["tubes"]["kept"]:
    print("Tube:", tube["tube_id"], "Raw classifier logit:", tube["logit"])
```

From `train/`, run `uv run python try_my_model.py`.

This uses your frozen YOLO to find smoke candidates, then your new classifier to score them. Segmentation YOLO's `boxes` are used; its masks are not returned by this pipeline. The example deliberately uses an **uncalibrated** decision with a starting threshold of 0.0, not a fitted production probability. Evaluate the full pipeline on independent sequences and select a threshold using validation data. The `probability` field can be `None` for this uncalibrated mode.

## 8. DVC, packaging, and the existing automatic workflow

**DVC** manages large datasets/model files and records pipeline dependencies. `.dvc` files are pointers to external data, not the data itself. `dvc.yaml` describes the stages, and `dvc.lock` records their resolved versions/hashes.

The stock workflow, from `train/`, is:

```bash
uv run dvc repro
```

It executes the required stages in order and can reuse unchanged outputs:

```text
truncate → build_tubes → build_model_input → train → package
```

However, this checkout's stock pipeline expects **Pyronear's dataset**, under `data/01_raw/datasets_full/{train,val}`, and its pinned detector. The dataset pointers currently import `pyro-dataset` v4.3.0; the configured DVC remote uses S3. `dvc pull` retrieves those referenced artifacts and can require storage credentials. It does not discover your segmentation dataset.

Section 6's manual commands use your own paths and do not need the Pyronear DVC remote. After you understand the stages, you can adapt `train/dvc.yaml` and the data tracking to automate your experiment. Direct manual commands do not update the stock DVC experiment records.

### A checkpoint and a deployable package are different

`best_checkpoint.pt` contains the temporal classifier's trained state. `model.zip` bundles the classifier, detector weights, configuration, and usually a fitted calibrator so the complete model can be loaded consistently.

The default `package.py`:

1. Loads the trained classifier and the declared detector.
2. Chooses a classifier threshold using validation patches.
3. Runs the full detector/classifier pipeline on raw training sequences to fit a logistic calibrator.
4. Runs it on raw validation sequences to select a final threshold targeting recall 0.95.
5. Writes `model.zip`. The recall target is a validation operating-point choice, not a guarantee on new data.

**Your custom detector is not a drop-in replacement for stock packaging.** `core/src/temporal_model/core/detector.yaml` declares a particular HuggingFace detector, revision, and SHA-256 hash. `package.py` verifies that identity. Its `--yolo-weights-path` argument changes the location but does not bypass verification.

For a deployable custom package, adapt detector provenance and weight resolution honestly, update DVC dependencies/paths if using DVC, and ensure packaging/calibration uses the same smoke-class filtering as inference. The current detector source configuration expects a HuggingFace source and revision; supporting a purely local detector is an implementation change. Point packaging's checkpoint, patch, and raw-sequence arguments at your experiment outputs. Do not simply replace Pyronear's weights file while retaining its declared identity.

No such packaging changes are made by this guide. Section 7 is the immediate local route after training.

## 9. The most important files to understand

In this table, `train/example.py` means `train/src/temporal_model/train/example.py`, and `core/example.py` means `core/src/temporal_model/core/example.py`. The top-level configuration paths (`train/params.yaml` and `train/dvc.yaml`) are written in full. `core/detector.yaml` means `core/src/temporal_model/core/detector.yaml`.

| File | Why you need it |
|---|---|
| `train/params.yaml` | Main editable training and pipeline settings |
| `train/dvc.yaml` | Stock stage commands, dependencies, and output paths |
| `train/truncate.py` | Copy the first N frames and labels |
| `train/build_tubes.py` | Build a training tube from label boxes |
| `train/build_model_input.py` | Generate cropped patches and metadata |
| `train/train.py` | Main classifier training entry point |
| `train/dataset.py` | Load patch sequences, targets, and padding masks |
| `train/lit_temporal.py` | Training loss, optimizer, and validation metrics |
| `core/temporal_classifier.py` | DINOv2 backbone and temporal transformer |
| `train/package.py` | Calibration and deployable archive creation |
| `core/model.py`, `core/pipeline.py` | Run the full inference pipeline |
| `core/detector.yaml` | Stock detector identity; relevant when adapting packaging |

The data flow for your first experiment is:

```text
Your real smoke clips + YOLO false-alarm clips
    → smoke bounding-box labels
    → truncate.py
    → build_tubes.py
    → build_model_input.py
    → train.py
    → best_checkpoint.pt
    → local inference with your existing YOLO
    → independent full-pipeline evaluation
```

Validation F1 during training measures classification of prepared tubes. Final evaluation must also include your detector's missed candidates and false alarms. This repository's smoke scores do not establish segmentation mask quality; evaluate your YOLO masks separately with the segmentation metrics appropriate to your project.
