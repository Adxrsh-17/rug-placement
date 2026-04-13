from __future__ import annotations

import cv2
import numpy as np


def find_optimal_placement_region(
    floor_mask: np.ndarray,
    rug_aspect_ratio: float = 1.5,
    padding: float = 0.15,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find a perspective-aware rug region on the detected floor."""
    h, w = floor_mask.shape

    ys, xs = np.where(floor_mask == 255)
    if len(ys) < 100:
        return None

    dist_transform = cv2.distanceTransform(floor_mask, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
    center_x, center_y = max_loc

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    floor_width = x_max - x_min
    floor_height = y_max - y_min

    rug_width = floor_width * 0.5
    rug_height = rug_width / rug_aspect_ratio

    perspective_factor = 0.75
    half_w_top = (rug_width * perspective_factor) / 2
    half_w_bottom = rug_width / 2
    half_h = rug_height / 2

    center_y_adjusted = min(center_y + rug_height * 0.2, y_max - half_h - padding * floor_height)

    corners = np.array(
        [
            [center_x - half_w_top, center_y_adjusted - half_h],
            [center_x + half_w_top, center_y_adjusted - half_h],
            [center_x + half_w_bottom, center_y_adjusted + half_h],
            [center_x - half_w_bottom, center_y_adjusted + half_h],
        ],
        dtype=np.float32,
    )

    corners[:, 0] = np.clip(corners[:, 0], x_min + 10, x_max - 10)
    corners[:, 1] = np.clip(corners[:, 1], y_min + 10, y_max - 10)

    return corners, dist_transform


def place_rug_on_room(
    room_img: np.ndarray,
    rug_img: np.ndarray,
    floor_mask: np.ndarray,
    target_corners: np.ndarray,
    blend_mode: str = "alpha",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp and blend the rug into the room constrained by floor mask."""
    room_h, room_w = room_img.shape[:2]
    rug_h, rug_w = rug_img.shape[:2]

    src_corners = np.array(
        [[0, 0], [rug_w - 1, 0], [rug_w - 1, rug_h - 1], [0, rug_h - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src_corners, target_corners)
    warped_rug = cv2.warpPerspective(rug_img, matrix, (room_w, room_h))

    rug_mask_src = np.ones((rug_h, rug_w), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(rug_mask_src, matrix, (room_w, room_h))
    final_mask = cv2.bitwise_and(warped_mask, floor_mask)

    if blend_mode == "alpha":
        alpha = cv2.GaussianBlur(final_mask.astype(float), (5, 5), 0) / 255.0
        alpha = np.stack([alpha] * 3, axis=-1)
        result = (alpha * warped_rug + (1 - alpha) * room_img).astype(np.uint8)
    else:
        result = room_img.copy()
        mask_bool = final_mask > 127
        result[mask_bool] = warped_rug[mask_bool]

    return result, warped_rug, final_mask


def detect_furniture_regions(img_rgb: np.ndarray, floor_mask: np.ndarray) -> np.ndarray:
    """Detect likely furniture zones for exclusion in difficult scenes."""
    h, w = img_rgb.shape[:2]

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    edge_density = cv2.dilate(edges, kernel, iterations=3)

    vertical_weight = np.linspace(0.3, 1.0, h).reshape(-1, 1)
    vertical_weight = np.tile(vertical_weight, (1, w))

    saturation = hsv[:, :, 1]
    sat_floor = saturation[floor_mask > 0]
    if sat_floor.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    sat_threshold = np.percentile(sat_floor, 60)
    high_sat_mask = (saturation > sat_threshold).astype(np.uint8) * 255

    furniture_mask = np.zeros((h, w), dtype=np.uint8)
    furniture_mask[(edge_density > 100) & (vertical_weight < 0.7)] = 255
    furniture_mask[(high_sat_mask > 0) & (vertical_weight < 0.6)] = 255

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    furniture_mask = cv2.morphologyEx(furniture_mask, cv2.MORPH_CLOSE, kernel_close)
    furniture_mask = cv2.dilate(furniture_mask, kernel_close, iterations=2)

    return furniture_mask


def find_optimal_placement_room3(
    floor_mask: np.ndarray,
    rug_aspect_ratio: float = 1.5,
    furniture_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Room-3 optimized placement with furniture exclusion and bottom bias."""
    h, w = floor_mask.shape

    safe_floor = floor_mask.copy()
    if furniture_mask is not None:
        safe_floor[furniture_mask > 0] = 0

    bottom_mask = np.zeros_like(safe_floor)
    bottom_mask[int(h * 0.55) :, :] = 255
    safe_floor = cv2.bitwise_and(safe_floor, bottom_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    safe_floor = cv2.morphologyEx(safe_floor, cv2.MORPH_OPEN, kernel)
    safe_floor = cv2.morphologyEx(safe_floor, cv2.MORPH_CLOSE, kernel)

    ys, xs = np.where(safe_floor == 255)
    if len(ys) < 500:
        safe_floor = floor_mask.copy()
        safe_floor[: int(h * 0.5), :] = 0
        ys, xs = np.where(safe_floor == 255)
        if len(ys) < 100:
            return None, None

    dist_transform = cv2.distanceTransform(safe_floor, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
    center_x, center_y = max_loc

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    floor_width = x_max - x_min
    rug_width = min(floor_width * 0.45, max_val * 1.5)
    rug_height = rug_width / rug_aspect_ratio

    perspective_factor = 0.7
    half_w_top = (rug_width * perspective_factor) / 2
    half_w_bottom = rug_width / 2
    half_h = rug_height / 2

    center_y_adjusted = min(center_y + rug_height * 0.15, y_max - half_h - 20)

    corners = np.array(
        [
            [center_x - half_w_top, center_y_adjusted - half_h],
            [center_x + half_w_top, center_y_adjusted - half_h],
            [center_x + half_w_bottom, center_y_adjusted + half_h],
            [center_x - half_w_bottom, center_y_adjusted + half_h],
        ],
        dtype=np.float32,
    )

    corners[:, 0] = np.clip(corners[:, 0], x_min + 15, x_max - 15)
    corners[:, 1] = np.clip(corners[:, 1], y_min + 15, y_max - 15)

    return corners, dist_transform
