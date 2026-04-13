from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.rug_visualizer.floor_detection import load_sam_predictor
from src.rug_visualizer.pipeline import (
    build_default_paths,
    process_assignment_batch,
    process_room3_enhanced,
    process_room_with_rug,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rug visualizer assignment runner")
    parser.add_argument(
        "--mode",
        choices=["batch", "single", "room3-enhanced"],
        default="batch",
        help="Run full assignment batch, one room/rug pair, or room3 enhanced mode.",
    )
    parser.add_argument("--room", type=str, help="Room image filename under rug-ai/images.")
    parser.add_argument("--rug", type=str, help="Rug image filename under rug-ai/images.")
    parser.add_argument("--color-tolerance", type=float, default=45)
    parser.add_argument("--rug-scale", type=float, default=0.85)
    parser.add_argument("--no-debug", action="store_true", help="Disable debug collage outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    paths = build_default_paths(project_root)

    if not paths.model_checkpoint.exists():
        print(f"Model checkpoint not found: {paths.model_checkpoint}")
        print("Download sam_vit_h_4b8939.pth into rug-ai/models before running.")
        return 1

    if args.mode == "batch":
        generated = process_assignment_batch(project_root, save_debug=not args.no_debug)
        print(f"Generated {len(generated)} outputs:")
        for name in generated:
            print(f"- {name}")
        return 0

    if not args.room or not args.rug:
        print("For single and room3-enhanced modes, provide --room and --rug.")
        return 1

    predictor = load_sam_predictor(paths.model_checkpoint)
    room_path = paths.images_dir / args.room
    rug_path = paths.images_dir / args.rug

    if args.mode == "single":
        result, _ = process_room_with_rug(
            room_path=room_path,
            rug_path=rug_path,
            predictor=predictor,
            output_dir=paths.output_dir,
            color_tolerance=args.color_tolerance,
            rug_scale=args.rug_scale,
            save_debug=not args.no_debug,
        )
        if result is None:
            print("No valid placement region found.")
            return 2
        print("Single run complete.")
        return 0

    result, _ = process_room3_enhanced(
        room_path=room_path,
        rug_path=rug_path,
        predictor=predictor,
        output_dir=paths.output_dir,
        rug_scale=args.rug_scale,
        save_debug=not args.no_debug,
    )
    if result is None:
        print("No valid enhanced placement region found.")
        return 2

    print("Room3 enhanced run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
