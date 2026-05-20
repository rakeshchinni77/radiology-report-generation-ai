"""Production-style CLI for explainable AI attention visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.visualize_attention import (
    DEFAULT_SPATIAL_ATTENTION_PATH,
    build_attention_visualization,
)


DEFAULT_OUTPUT_PATH = Path("output/attention_overlay.png")


def log(message: str) -> None:
    print(f"[visualize] {message}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the visualization entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate an attention heatmap and overlay for a chest X-ray image."
    )
    parser.add_argument("--image_path", required=True, help="Path to the input chest X-ray image")
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path for the final overlay PNG; defaults to output/attention_overlay.png",
    )
    parser.add_argument(
        "--attention_path",
        default=str(DEFAULT_SPATIAL_ATTENTION_PATH),
        help="Path to the saved spatial attention tensor",
    )
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay blending factor")
    parser.add_argument("--show", action="store_true", help="Display the matplotlib visualization interactively")
    return parser.parse_args()


def _resolve_output_path(output_path: str | None) -> Path:
    """Resolve the final overlay save path."""
    if not output_path:
        return DEFAULT_OUTPUT_PATH

    resolved = Path(output_path)
    if resolved.suffix:
        return resolved
    return resolved / DEFAULT_OUTPUT_PATH.name


def main() -> int:
    """Run the visualization pipeline and save the requested outputs."""
    args = parse_args()
    try:
        resolved_output_path = _resolve_output_path(args.output_path)
        result = build_attention_visualization(
            image_path=args.image_path,
            attention_path=args.attention_path,
            output_dir=resolved_output_path.parent,
            output_path=resolved_output_path,
            alpha=args.alpha,
            show=args.show,
        )
        log(f"output_path={result['overlay_path']}")
        resolution = result.get("overlay_resolution")
        if resolution is not None:
            log(f"overlay_resolution={resolution[1]}x{resolution[0]}")
        return 0
    except Exception as exc:
        print(f"[visualize][error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
