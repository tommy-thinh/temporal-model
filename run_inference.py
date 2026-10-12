"""Run a packaged temporal smoke model; edit the settings below first."""

import json
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from temporal_model.core.model import BboxTubeTemporalModel

# Settings: paths can be absolute or relative to the repository root.
IMAGE_DIR = Path("/absolute/path/to/my_sequence")
MODEL_PATH = None  # None downloads the pinned release; or Path("my/model.zip").
DEVICE = None  # Automatically choose CUDA/MPS/CPU; or set "cpu" / "cuda".
OUTPUT_PATH = Path(".cache/run/sequence-output.json")

ROOT = Path(__file__).resolve().parent


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    image_dir = resolve_path(IMAGE_DIR)
    if not image_dir.is_dir():
        raise SystemExit(f"Edit IMAGE_DIR: directory does not exist: {image_dir}")
    paths = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        key=lambda path: path.stem,
    )
    if not paths:
        raise SystemExit(f"No JPG, JPEG, or PNG images found in {image_dir}")
    if len({path.stem for path in paths}) != len(paths):
        raise SystemExit("Frame filename stems must be unique within IMAGE_DIR")

    if MODEL_PATH is None:
        version = (ROOT / "api/MODEL_VERSION").read_text().strip()
        model_dir = ROOT / ".cache/models" / f"v{version}"
        model_path = model_dir / "model.zip"
        if not model_path.is_file():
            print(f"Downloading pretrained model v{version}...", flush=True)
            model_path = Path(
                hf_hub_download(
                    "pyronear/temporal-model",
                    "model.zip",
                    revision=f"v{version}",
                    local_dir=model_dir,
                )
            )
    else:
        model_path = resolve_path(MODEL_PATH)
    if not model_path.is_file():
        raise SystemExit(f"Model archive does not exist: {model_path}")

    print(f"Loading {model_path}...", flush=True)
    model = BboxTubeTemporalModel.from_package(model_path, device=DEVICE)
    print(f"Running on {model.device} with {len(paths)} input frames...", flush=True)
    output = model.predict_sequence(paths)
    kept = output.details["tubes"]["kept"]
    probabilities = [t["probability"] for t in kept if t["probability"] is not None]
    probability = max(probabilities) if probabilities else (0.0 if not kept else None)
    print("Smoke detected:", output.is_positive)
    print("Max calibrated tube probability:", probability)
    print("Kept tubes:", len(kept))
    print("Frames discarded:", output.details["preprocessing"]["num_truncated"])
    output_path = resolve_path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(output), indent=2), encoding="utf-8")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
