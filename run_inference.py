"""Run a packaged temporal smoke model; edit the settings below first."""

import json
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.package import load_model_package
from temporal_model.core.protocol import Frame
from temporal_model.core.types import Detection, FrameDetections

# Settings: paths can be absolute or relative to the repository root.
IMAGE_DIR = Path("/absolute/path/to/my_sequence")
MODEL_PATH = None  # None downloads the pinned release; or Path("my/model.zip").
YOLO_MODEL_PATH = None  # Set Path("/path/to/your/best.pt") to use your segmenter.
SMOKE_CLASS_ID = 81  # Class 80 (fire) is excluded from this smoke classifier.
YOLO_CONFIDENCE = 0.1
YOLO_IOU = 0.2
YOLO_IMAGE_SIZE = 1024
DEVICE = None  # Automatically choose CUDA/MPS/CPU; or set "cpu" / "cuda".
OUTPUT_PATH = Path(".cache/run/sequence-output.json")

ROOT = Path(__file__).resolve().parent


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def predict_smoke_boxes(
    detector: YOLO, frames: list[Frame], device: str
) -> dict[str, FrameDetections]:
    """Convert segmentation YOLO's smoke boxes into temporal-model inputs."""
    supplied = {}
    for idx, frame in enumerate(frames):
        # One image per call keeps segmentation mask memory bounded.
        result = detector.predict(
            str(frame.image_path),
            classes=[SMOKE_CLASS_ID],
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU,
            imgsz=YOLO_IMAGE_SIZE,
            device=device,
            verbose=False,
        )[0]
        detections = []
        if result.boxes is not None:
            for bbox, confidence, class_id in zip(
                result.boxes.xywhn.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
                strict=True,
            ):
                if int(class_id) != SMOKE_CLASS_ID:
                    continue
                detections.append(
                    Detection(
                        class_id=SMOKE_CLASS_ID,
                        cx=bbox[0],
                        cy=bbox[1],
                        w=bbox[2],
                        h=bbox[3],
                        confidence=confidence,
                    )
                )
        supplied[frame.frame_id] = FrameDetections(
            frame_idx=idx,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            detections=detections,
        )
        print(f"{frame.frame_id}: {len(detections)} smoke boxes", flush=True)
    return supplied


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
    if YOLO_MODEL_PATH is not None and not resolve_path(YOLO_MODEL_PATH).is_file():
        raise SystemExit(f"YOLO weights do not exist: {resolve_path(YOLO_MODEL_PATH)}")

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
    if YOLO_MODEL_PATH is None:
        model = BboxTubeTemporalModel.from_package(model_path, device=DEVICE)
        print("Using bundled YOLO.", flush=True)
        print(f"Running on {model.device} with {len(paths)} input frames...", flush=True)
        output = model.predict_sequence(paths)
    else:
        # Load the pretrained classifier/config/calibrator without bundled YOLO.
        package = load_model_package(model_path, with_detector=False)
        model = BboxTubeTemporalModel(
            yolo_model=None,
            classifier=package.classifier,
            config=package.config,
            calibrator=package.calibrator,
            device=DEVICE,
        )
        detector_path = resolve_path(YOLO_MODEL_PATH)
        detector = YOLO(str(detector_path))
        if SMOKE_CLASS_ID not in detector.names:
            raise SystemExit(
                f"Smoke class {SMOKE_CLASS_ID} is absent. Model classes: {detector.names}"
            )
        print(f"Using your YOLO: {detector_path}", flush=True)
        print(f"Smoke class: {SMOKE_CLASS_ID} ({detector.names[SMOKE_CLASS_ID]})")
        print(f"Running on {model.device} with {len(paths)} input frames...", flush=True)
        frames = model.load_sequence(paths)
        # The temporal model uses only the first max_frames images. Supplying
        # every used frame (including empty lists) also covers repeated padding.
        used_frames = frames[: package.config["classifier"]["max_frames"]]
        supplied = predict_smoke_boxes(detector, used_frames, str(model.device))
        output = model.predict(frames, frame_detections=supplied)
        print("Calibration is from the original detector; evaluate on your data.")
    kept = output.details["tubes"]["kept"]
    probabilities = [t["probability"] for t in kept if t["probability"] is not None]
    probability = max(probabilities) if probabilities else (0.0 if not kept else None)
    print("Smoke detected:", output.is_positive)
    print("Max calibrated tube probability:", probability)
    print("Kept tubes:", len(kept))
    print("Candidate tubes:", output.details["tubes"]["num_candidates"])
    print("Frames discarded:", output.details["preprocessing"]["num_truncated"])
    output_path = resolve_path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(output), indent=2), encoding="utf-8")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
