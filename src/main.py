"""Orchestrate sample report generation and attention visualization outputs."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import run_inference
from src.visualization.visualize_attention import build_attention_visualization


SAMPLE_DIR = ROOT / "data" / "sample"
OUTPUT_DIR = ROOT / "output"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_SAMPLES = 5
DEFAULT_IMAGE_SIZE = (224, 224)


def log(message: str) -> None:
    """Print a consistent orchestration log line."""
    print(f"[main] {message}")


def _discover_sample_images(sample_dir: Path) -> list[Path]:
    """Find up to five supported sample images in the sample directory."""
    if not sample_dir.exists():
        sample_dir.mkdir(parents=True, exist_ok=True)

    images = [
        path
        for path in sorted(sample_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return images[:MAX_SAMPLES]


def _create_placeholder_image(image_path: Path, index: int) -> Path:
    """Create a lightweight synthetic chest X-ray-like placeholder image."""
    height, width = DEFAULT_IMAGE_SIZE
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for row in range(height):
        intensity = int(25 + (row / max(height - 1, 1)) * 85)
        canvas[row, :, :] = intensity

    cv2.ellipse(canvas, (width // 2, height // 2 + 8), (58, 78), 0, 0, 360, (55, 55, 55), 2)
    cv2.line(canvas, (width // 2 - 22, 55), (width // 2 - 22, 165), (45, 45, 45), 3)
    cv2.line(canvas, (width // 2 + 22, 55), (width // 2 + 22, 165), (45, 45, 45), 3)
    cv2.putText(
        canvas,
        f"SAMPLE {index}",
        (20, 198),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), canvas):
        raise IOError(f"Failed to write placeholder image: {image_path}")
    return image_path


def _ensure_sample_images(sample_dir: Path) -> list[Path]:
    """Ensure five usable sample images exist for the pipeline."""
    images = _discover_sample_images(sample_dir)
    if len(images) >= MAX_SAMPLES:
        return images[:MAX_SAMPLES]

    next_index = len(images) + 1
    while len(images) < MAX_SAMPLES:
        placeholder_path = sample_dir / f"sample_auto_{next_index}.png"
        images.append(_create_placeholder_image(placeholder_path, next_index))
        next_index += 1

    return images


def _candidate_image_path(image_path: Path) -> Path:
    """Return a valid image path, raising a clear error if the file is missing."""
    if not image_path.exists():
        raise FileNotFoundError(f"Sample image not found: {image_path}")
    return image_path


def _run_sample_pipeline(sample_image: Path, sample_index: int) -> tuple[Path, Path]:
    """Run report generation and visualization for a single sample image."""
    report_path = OUTPUT_DIR / f"sample_{sample_index}_report.txt"
    viz_path = OUTPUT_DIR / f"sample_{sample_index}_viz.png"

    start_time = time.perf_counter()
    report = run_inference(
        image_path=sample_image,
        checkpoint=Path("checkpoints/best_model.pt"),
        max_length=64,
        device_arg="cpu",
        output_path=report_path,
        decode_strategy="beam",
        beam_size=3,
        temperature=1.0,
        top_k=0,
        return_attentions=True,
    )

    if not report.strip():
        raise ValueError(f"Empty report generated for {sample_image}")

    build_attention_visualization(
        image_path=sample_image,
        output_path=viz_path,
        output_dir=OUTPUT_DIR,
        alpha=0.6,
        show=False,
        save_intermediates=False,
    )

    elapsed = time.perf_counter() - start_time
    log(f"sample={sample_index}")
    log(f"image_path={sample_image}")
    log(f"report_path={report_path}")
    log(f"visualization_path={viz_path}")
    log(f"generation_time_sec={elapsed:.2f}")

    return report_path, viz_path


def run_sample_generation_pipeline(sample_dir: Path = SAMPLE_DIR) -> list[tuple[Path, Path]]:
    """Generate reports and visualizations for up to five sample images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_images = _ensure_sample_images(sample_dir)[:MAX_SAMPLES]
    if not sample_images:
        raise FileNotFoundError(f"No sample images found in {sample_dir}")

    outputs: list[tuple[Path, Path]] = []
    for index, image_path in enumerate(sample_images, start=1):
        try:
            outputs.append(_run_sample_pipeline(_candidate_image_path(image_path), index))
        except Exception as exc:
            log(f"sample={index} error={exc}")
            continue
    return outputs


def main() -> int:
    """Entry point for sample output generation."""
    try:
        run_sample_generation_pipeline()
        return 0
    except Exception as exc:
        log(f"fatal_error={exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
