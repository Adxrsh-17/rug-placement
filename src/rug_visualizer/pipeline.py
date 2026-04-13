from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .floor_detection import get_floor_mask, load_sam_predictor
from .placement import (
    detect_furniture_regions,
    find_optimal_placement_region,
    find_optimal_placement_room3,
    place_rug_on_room,
)


@dataclass(frozen=True)
class PipelinePaths:
    base_dir: Path
    images_dir: Path
    model_checkpoint: Path
    output_dir: Path


DEFAULT_ROOM_CONFIG: dict[str, tuple[float, float]] = {
    "room1": (50, 0.9),
    "room2": (45, 0.85),
    "room3": (40, 0.9),
    "room4": (50, 0.85),
    "room5": (45, 0.8),
}


def build_default_paths(project_root: Path) -> PipelinePaths:
    base_dir = project_root / "rug-ai"
    return PipelinePaths(
        base_dir=base_dir,
        images_dir=base_dir / "images",
        model_checkpoint=base_dir / "models" / "sam_vit_h_4b8939.pth",
        output_dir=base_dir / "outputs",
    )


def _load_rgb_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def process_room_with_rug(
    room_path: Path,
    rug_path: Path,
    predictor,
    output_dir: Path,
    color_tolerance: float = 45,
    rug_scale: float = 1.0,
    save_debug: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    room_rgb = _load_rgb_image(room_path)
    rug_rgb = _load_rgb_image(rug_path)

    room_name = room_path.stem
    rug_name = rug_path.stem

    floor_mask = get_floor_mask(room_rgb, predictor, color_tolerance)

    rug_aspect = rug_rgb.shape[1] / rug_rgb.shape[0]
    placement_result = find_optimal_placement_region(floor_mask, rug_aspect)
    if placement_result is None:
        return None, floor_mask

    target_corners, dist_transform = placement_result

    center = target_corners.mean(axis=0)
    target_corners = center + (target_corners - center) * rug_scale

    result, warped_rug, rug_mask = place_rug_on_room(
        room_rgb,
        rug_rgb,
        floor_mask,
        target_corners,
        blend_mode="alpha",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_name = f"{room_name}_{rug_name}_result.jpg"
    cv2.imwrite(str(output_dir / result_name), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    if save_debug:
        _save_debug_collage(
            output_dir / f"{room_name}_{rug_name}_steps.jpg",
            room_rgb,
            rug_rgb,
            floor_mask,
            dist_transform,
            target_corners,
            result,
        )

    return result, floor_mask


def process_room3_enhanced(
    room_path: Path,
    rug_path: Path,
    predictor,
    output_dir: Path,
    rug_scale: float = 0.8,
    save_debug: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    room_rgb = _load_rgb_image(room_path)
    rug_rgb = _load_rgb_image(rug_path)

    room_name = room_path.stem
    rug_name = rug_path.stem

    floor_mask = get_floor_mask(room_rgb, predictor, color_tolerance=35)
    furniture_mask = detect_furniture_regions(room_rgb, floor_mask)

    rug_aspect = rug_rgb.shape[1] / rug_rgb.shape[0]
    target_corners, dist_transform = find_optimal_placement_room3(floor_mask, rug_aspect, furniture_mask)

    if target_corners is None:
        return None, floor_mask

    center = target_corners.mean(axis=0)
    target_corners = center + (target_corners - center) * rug_scale

    safe_floor_mask = floor_mask.copy()
    safe_floor_mask[furniture_mask > 0] = 0

    result, _, _ = place_rug_on_room(
        room_rgb,
        rug_rgb,
        safe_floor_mask,
        target_corners,
        blend_mode="alpha",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_name = f"{room_name}_{rug_name}_enhanced.jpg"
    cv2.imwrite(str(output_dir / result_name), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    if save_debug:
        _save_room3_debug_collage(
            output_dir / f"{room_name}_{rug_name}_enhanced_steps.jpg",
            room_rgb,
            rug_rgb,
            floor_mask,
            furniture_mask,
            safe_floor_mask,
            dist_transform,
            target_corners,
            result,
        )

    return result, safe_floor_mask


def process_assignment_batch(project_root: Path, save_debug: bool = True) -> list[str]:
    paths = build_default_paths(project_root)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    predictor = load_sam_predictor(paths.model_checkpoint)

    rooms = ["room1.jpeg", "room2.JPG", "room3.jpeg", "room4.png", "room5.JPG"]
    rugs = ["rug1.jpg", "rug3.jpg"]

    generated_files: list[str] = []

    for room_file in rooms:
        room_name = Path(room_file).stem
        for rug_file in rugs:
            color_tolerance, rug_scale = DEFAULT_ROOM_CONFIG.get(room_name, (45, 0.85))

            result, _ = process_room_with_rug(
                room_path=paths.images_dir / room_file,
                rug_path=paths.images_dir / rug_file,
                predictor=predictor,
                output_dir=paths.output_dir,
                color_tolerance=color_tolerance,
                rug_scale=rug_scale,
                save_debug=save_debug,
            )
            if result is not None:
                generated_files.append(f"{room_name}_{Path(rug_file).stem}_result.jpg")

            if room_name == "room3":
                enhanced_result, _ = process_room3_enhanced(
                    room_path=paths.images_dir / room_file,
                    rug_path=paths.images_dir / rug_file,
                    predictor=predictor,
                    output_dir=paths.output_dir,
                    rug_scale=0.8,
                    save_debug=save_debug,
                )
                if enhanced_result is not None:
                    generated_files.append(f"{room_name}_{Path(rug_file).stem}_enhanced.jpg")

    _create_gallery(paths.output_dir)
    generated_files.append("_gallery.jpg")
    return generated_files


def _save_debug_collage(
    output_path: Path,
    room_rgb: np.ndarray,
    rug_rgb: np.ndarray,
    floor_mask: np.ndarray,
    dist_transform: np.ndarray,
    target_corners: np.ndarray,
    result: np.ndarray,
) -> None:
    vis_placement = room_rgb.copy()
    pts = target_corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis_placement, [pts], True, (255, 0, 0), 3)

    center = target_corners.mean(axis=0)
    cv2.circle(vis_placement, (int(center[0]), int(center[1])), 8, (0, 255, 0), -1)

    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_HOT)

    floor_3ch = cv2.cvtColor(floor_mask, cv2.COLOR_GRAY2RGB)

    top = np.hstack((room_rgb, floor_3ch, cv2.cvtColor(dist_color, cv2.COLOR_BGR2RGB)))
    bottom = np.hstack((rug_rgb, vis_placement, result))

    top = cv2.resize(top, (1800, 600), interpolation=cv2.INTER_AREA)
    bottom = cv2.resize(bottom, (1800, 600), interpolation=cv2.INTER_AREA)
    grid = np.vstack((top, bottom))

    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def _save_room3_debug_collage(
    output_path: Path,
    room_rgb: np.ndarray,
    rug_rgb: np.ndarray,
    floor_mask: np.ndarray,
    furniture_mask: np.ndarray,
    safe_floor_mask: np.ndarray,
    dist_transform: np.ndarray,
    target_corners: np.ndarray,
    result: np.ndarray,
) -> None:
    vis_placement = room_rgb.copy()
    pts = target_corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis_placement, [pts], True, (0, 255, 0), 4)

    center = target_corners.mean(axis=0)
    cv2.circle(vis_placement, (int(center[0]), int(center[1])), 10, (255, 0, 0), -1)

    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_HOT)

    floor_3ch = cv2.cvtColor(floor_mask, cv2.COLOR_GRAY2RGB)
    furniture_3ch = cv2.cvtColor(furniture_mask, cv2.COLOR_GRAY2RGB)
    safe_3ch = cv2.cvtColor(safe_floor_mask, cv2.COLOR_GRAY2RGB)

    row1 = np.hstack((room_rgb, floor_3ch, furniture_3ch, safe_3ch))
    row2 = np.hstack((rug_rgb, cv2.cvtColor(dist_color, cv2.COLOR_BGR2RGB), vis_placement, result))

    row1 = cv2.resize(row1, (2200, 550), interpolation=cv2.INTER_AREA)
    row2 = cv2.resize(row2, (2200, 550), interpolation=cv2.INTER_AREA)
    grid = np.vstack((row1, row2))

    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def _create_gallery(output_dir: Path) -> None:
    rooms = ["room1", "room2", "room3", "room4", "room5"]
    rugs = ["rug1", "rug3"]

    tiles: list[np.ndarray] = []

    for room in rooms:
        for rug in rugs:
            result_path = output_dir / f"{room}_{rug}_result.jpg"
            if result_path.exists():
                img = cv2.cvtColor(cv2.imread(str(result_path)), cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros((500, 700, 3), dtype=np.uint8)
                cv2.putText(img, "Not generated", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            img = cv2.resize(img, (900, 500), interpolation=cv2.INTER_AREA)
            cv2.putText(
                img,
                f"{room} + {rug}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            tiles.append(img)

    rows = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
    gallery = np.vstack(rows)
    cv2.imwrite(str(output_dir / "_gallery.jpg"), cv2.cvtColor(gallery, cv2.COLOR_RGB2BGR))
