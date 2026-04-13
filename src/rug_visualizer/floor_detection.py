from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


def load_sam_predictor(checkpoint_path: Path | str, model_type: str = "vit_h") -> SamPredictor:
    """Load and return a SAM predictor on CPU or CUDA."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device)
    return SamPredictor(sam)


def compute_lbp_fast(image: np.ndarray) -> np.ndarray:
    """Fast LBP approximation using 8-neighborhood comparisons."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    gray = gray.astype(np.float32)

    lbp = np.zeros_like(gray, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

    for idx, (dy, dx) in enumerate(offsets):
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp += ((shifted >= gray).astype(np.uint8) << idx)

    return lbp


def resize_for_processing(img_rgb: np.ndarray, max_dim: int = 2000) -> tuple[np.ndarray, float]:
    """Resize large images for speed while keeping aspect ratio."""
    h, w = img_rgb.shape[:2]
    if max(h, w) <= max_dim:
        return img_rgb, 1.0

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def statistical_intensity_analysis(img_rgb: np.ndarray, mask: np.ndarray) -> Optional[dict[str, np.ndarray]]:
    """Compute robust floor color statistics with sampling for large regions."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    ys, xs = np.where(mask > 0)
    if len(ys) < 50:
        return None

    max_samples = 50000
    if len(ys) > max_samples:
        indices = np.random.choice(len(ys), max_samples, replace=False)
        ys = ys[indices]
        xs = xs[indices]

    floor_lab = lab[ys, xs]
    floor_hsv = hsv[ys, xs]

    return {
        "lab_mean": np.mean(floor_lab, axis=0),
        "lab_std": np.std(floor_lab, axis=0),
        "lab_median": np.median(floor_lab, axis=0),
        "hsv_mean": np.mean(floor_hsv, axis=0),
        "hsv_std": np.std(floor_hsv, axis=0),
    }


def canny_edge_floor_boundary(img_rgb: np.ndarray, floor_mask_255: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove likely non-floor edge areas from a candidate floor mask."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    median_val = np.median(gray_filtered)
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(gray_filtered, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    refined = floor_mask_255.copy()
    refined[edges_dilated > 0] = 0

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close)
    return refined, edges


def detect_floor_with_sam(img_rgb: np.ndarray, predictor: SamPredictor) -> np.ndarray:
    """Detect coarse floor region with SAM using bottom-biased prompts."""
    h, w = img_rgb.shape[:2]
    processed_img, scale = resize_for_processing(img_rgb, max_dim=1500)
    ph, pw = processed_img.shape[:2]

    predictor.set_image(processed_img)

    seed_points = [
        [pw * 0.25, ph * 0.85],
        [pw * 0.50, ph * 0.90],
        [pw * 0.75, ph * 0.85],
        [pw * 0.35, ph * 0.75],
        [pw * 0.65, ph * 0.75],
        [pw * 0.50, ph * 0.80],
        [pw * 0.40, ph * 0.70],
        [pw * 0.60, ph * 0.70],
    ]

    points_array = np.array(seed_points, dtype=np.float32)
    labels = np.ones(len(points_array), dtype=int)

    masks, scores, _ = predictor.predict(
        point_coords=points_array,
        point_labels=labels,
        multimask_output=True,
    )

    best_mask = None
    best_score = -1.0

    for mask, score in zip(masks, scores):
        mask_uint8 = mask.astype(np.uint8)
        ys, _ = np.where(mask_uint8 == 1)
        if len(ys) == 0:
            continue

        center_y = ys.mean() / ph
        mask_coverage = len(ys) / (ph * pw)

        if center_y > 0.5 and mask_coverage > 0.05:
            adjusted_score = score * (1 + center_y) * (1 + mask_coverage * 2)
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_mask = mask_uint8

    if best_mask is None:
        best_mask = masks[int(np.argmax(scores))].astype(np.uint8)

    if scale != 1.0:
        best_mask = cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return best_mask


def refine_floor_mask_enhanced(mask: np.ndarray, img_rgb: np.ndarray, color_tolerance: float = 45) -> np.ndarray:
    """Refine SAM mask using color, texture, geometry, and edge constraints."""
    h, w = mask.shape
    max_process_dim = 1500

    if max(h, w) > max_process_dim:
        scale = max_process_dim / max(h, w)
        small_h, small_w = int(h * scale), int(w * scale)
        mask_small = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        img_small = cv2.resize(img_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small_h, small_w = h, w
        mask_small = mask
        img_small = img_rgb

    stats = statistical_intensity_analysis(img_small, mask_small)
    if stats is None:
        return (mask * 255).astype(np.uint8)

    img_lab = cv2.cvtColor(img_small, cv2.COLOR_RGB2LAB)
    adaptive_tolerance = float(color_tolerance + stats["lab_std"][0] * 0.5)

    lab_diff = np.abs(img_lab.astype(float) - stats["lab_median"])
    weighted_diff = lab_diff[:, :, 0] * 0.5 + lab_diff[:, :, 1] * 0.25 + lab_diff[:, :, 2] * 0.25
    color_mask = (weighted_diff < adaptive_tolerance).astype(np.uint8)

    lbp = compute_lbp_fast(img_small)
    floor_ys, floor_xs = np.where(mask_small == 1)

    if len(floor_ys) > 100:
        sample_size = min(5000, len(floor_ys))
        sample_idx = np.random.choice(len(floor_ys), sample_size, replace=False)
        floor_lbp = lbp[floor_ys[sample_idx], floor_xs[sample_idx]]
        lbp_median = np.median(floor_lbp)
        lbp_std = np.std(floor_lbp)
        lbp_diff = np.abs(lbp.astype(float) - lbp_median)
        lbp_mask = (lbp_diff < lbp_std * 2.5).astype(np.uint8)
    else:
        lbp_mask = np.ones((small_h, small_w), dtype=np.uint8)

    geo_mask = np.zeros((small_h, small_w), dtype=np.uint8)
    geo_mask[int(small_h * 0.45) :, :] = 1

    combined = (mask_small * color_mask * lbp_mask * geo_mask).astype(np.uint8)
    combined_255 = combined * 255

    refined_by_edges, _ = canny_edge_floor_boundary(img_small, combined_255)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    refined_by_edges = cv2.morphologyEx(refined_by_edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    refined_by_edges = cv2.morphologyEx(refined_by_edges, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(refined_by_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours_sorted[0]

        result_small = np.zeros((small_h, small_w), dtype=np.uint8)
        cv2.drawContours(result_small, [largest_contour], -1, 255, -1)

        if len(contours_sorted) > 1:
            second_area = cv2.contourArea(contours_sorted[1])
            largest_area = cv2.contourArea(largest_contour)
            if second_area > largest_area * 0.3:
                cv2.drawContours(result_small, [contours_sorted[1]], -1, 255, -1)

        result_small = cv2.GaussianBlur(result_small, (15, 15), 0)
        _, result_small = cv2.threshold(result_small, 127, 255, cv2.THRESH_BINARY)

        if scale != 1.0:
            result = cv2.resize(result_small, (w, h), interpolation=cv2.INTER_LINEAR)
            _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
            return result

        return result_small

    if scale != 1.0:
        refined = cv2.resize(refined_by_edges, (w, h), interpolation=cv2.INTER_LINEAR)
        _, refined = cv2.threshold(refined, 127, 255, cv2.THRESH_BINARY)
        return refined

    return refined_by_edges


def get_floor_mask(img_rgb: np.ndarray, predictor: SamPredictor, color_tolerance: float = 45) -> np.ndarray:
    """Complete floor detection pipeline."""
    raw_mask = detect_floor_with_sam(img_rgb, predictor)
    return refine_floor_mask_enhanced(raw_mask, img_rgb, color_tolerance=color_tolerance)
