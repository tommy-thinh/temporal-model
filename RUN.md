# Run inference

You only need the `core` Python environment, a pretrained model, and a folder
of ordered images. The model includes YOLO and the temporal smoke classifier.

The previous guide covered several advanced workflows at once. The authors’
[core README](core/README.md) also supports a simple Python inference workflow;
the extra packaging, DVC, and server instructions are unnecessary for this task.

## 1. One-time setup

Run these commands from the repository root:

```bash
cd /home/tommy/Desktop/temporal-model
uv sync --project core --python 3.12 --locked
```

This installs the dependencies into `core/.venv`. The initial download can be
large. If `uv` is missing, install it using the official Linux installer, then
open a new terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## 2. Activate the environment

```bash
source core/.venv/bin/activate
```

This works like `conda activate`: `python` now uses this environment and its
installed packages. You can run normal Python commands without `uv run`.
Activate it once in each new terminal; you do not need to reinstall dependencies.
Use `deactivate` when finished. This is standard
[uv environment activation](https://docs.astral.sh/uv/pip/environments/).

## 3. Set your image folder

Open [run_inference.py](run_inference.py) and edit the setting near the top:

```python
IMAGE_DIR = Path("/absolute/path/to/my_sequence")
```

The folder should contain consecutive frames from one scene:

```text
my_sequence/
  frame_000001.jpg
  frame_000002.jpg
  frame_000003.jpg
  ...
```

The script sorts filenames. Use zero-padded numbers so they stay in time order.
JPG, JPEG, and PNG images are accepted. No label files are needed.

Leave `MODEL_PATH = None` to download the pretrained release specified in
`api/MODEL_VERSION` on the first run. The script saves it under
`.cache/models/` and reuses that file on subsequent runs. If you already have
a complete `model.zip`, set `MODEL_PATH` to its path instead.

This uses the YOLO bundled with that archive. Your own YOLO `.pt` file alone
cannot replace `model.zip`, which also contains the temporal classifier.

## 4. Run

```bash
python run_inference.py
```

The script prints the device, smoke decision, maximum calibrated tube
probability, and number of kept tubes. It saves full details to
`.cache/run/sequence-output.json`, replacing the previous result.

`DEVICE = None` automatically selects CUDA, Apple MPS, or CPU. Set it to
`"cpu"` to force CPU. A CUDA run requires compatible PyTorch and NVIDIA drivers.

The pinned model uses at most the first 20 frames of a sequence. The script
reports when frames were discarded. For a long recording, select the window
you want to inspect. The output is a smoke decision; segmentation masks and
a separate fire prediction are not produced.

## Next time

After editing `IMAGE_DIR`, these are the only commands you need:

```bash
cd /home/tommy/Desktop/temporal-model
source core/.venv/bin/activate
python run_inference.py
```

For training and custom-YOLO integration, see [EXPLAIN.md](EXPLAIN.md).
