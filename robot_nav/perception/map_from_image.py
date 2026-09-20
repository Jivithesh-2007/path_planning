"""
OpenCV Computer Vision pipeline for converting top-down map images into occupancy grids with start/goal detection.
"""

from pathlib import Path
import numpy as np
import cv2
from typing import Tuple, Optional

from robot_nav.config import SimConfig, DEFAULT_CONFIG


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 contour corner points: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect


def warp_perspective_border(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Detect outer 4-point border contour and apply perspective transform.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image, False

    # Find largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    img_area = image.shape[0] * image.shape[1]

    # Must cover at least 30% of total image area to be considered an outer border
    if area < 0.30 * img_area:
        return image, False

    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        rect = order_points(pts)

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array(
            [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped, True

    return image, False


def detect_start_and_goal(
    bgr_image: np.ndarray, config: SimConfig = DEFAULT_CONFIG
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Detect start (green) and goal (red) centroid coordinates in pixels using HSV color masking.
    :return: (start_pixel (y, x), goal_pixel (y, x)) in floating point image coordinates.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Green mask for start
    green_mask = cv2.inRange(
        hsv, np.array(config.perception.green_lower), np.array(config.perception.green_upper)
    )

    # Red mask for goal (combining dual HSV red bands)
    red_mask1 = cv2.inRange(
        hsv, np.array(config.perception.red_lower1), np.array(config.perception.red_upper1)
    )
    red_mask2 = cv2.inRange(
        hsv, np.array(config.perception.red_lower2), np.array(config.perception.red_upper2)
    )
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    start_pt = None
    goal_pt = None

    # Centroid of green mask
    M_green = cv2.moments(green_mask)
    if M_green["m00"] > 10:
        cx = float(M_green["m10"] / M_green["m00"])
        cy = float(M_green["m01"] / M_green["m00"])
        start_pt = (cy, cx)  # (y, x)

    # Centroid of red mask
    M_red = cv2.moments(red_mask)
    if M_red["m00"] > 10:
        cx = float(M_red["m10"] / M_red["m00"])
        cy = float(M_red["m01"] / M_red["m00"])
        goal_pt = (cy, cx)  # (y, x)

    return start_pt, goal_pt


def load_map_from_image(
    image_path: str | Path,
    config: SimConfig = DEFAULT_CONFIG,
    save_debug_path: Optional[str | Path] = "maps/debug_overlay.png",
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Process map image -> occupancy grid, extract start/goal coordinates, save debug overlay.

    :param image_path: Path to RGB map image.
    :param config: System configuration.
    :param save_debug_path: Optional output filepath for multi-stage debug overlay.
    :return: (grid: uint8 np.ndarray, start: (r, c), goal: (r, c))
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Map image not found at {path}")

    bgr_img = cv2.imread(str(path))
    if bgr_img is None:
        raise ValueError(f"Could not decode image at {path}")

    # Stage 1: Detect start/goal prior to grayscale
    start_px, goal_px = detect_start_and_goal(bgr_img, config)

    # Stage 2: Border perspective warp
    warped_bgr, did_warp = warp_perspective_border(bgr_img)

    # Stage 3: Grayscale -> Adaptive Thresholding -> Morphological Close
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological closing to seal gaps in hand-drawn walls
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Downsample to NxM grid
    grid_h = config.map.grid_height
    grid_w = config.map.grid_width
    h, w = gray.shape

    cell_h = h / grid_h
    cell_w = w / grid_w

    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    for r in range(grid_h):
        for c in range(grid_w):
            r0, r1 = int(r * cell_h), int((r + 1) * cell_h)
            c0, c1 = int(c * cell_w), int((c + 1) * cell_w)

            cell_morph = morph[r0:r1, c0:c1]
            if cell_morph.size > 0:
                occupied_fraction = float(np.mean(cell_morph > 0))
                if occupied_fraction >= config.map.occupied_threshold:
                    grid[r, c] = 1

    # Convert start and goal pixel centroids to grid (r, c)
    if start_px is not None:
        start_r = int(np.clip(start_px[0] / cell_h, 0, grid_h - 1))
        start_c = int(np.clip(start_px[1] / cell_w, 0, grid_w - 1))
        start = (start_r, start_c)
    else:
        start = (1, 1)

    if goal_px is not None:
        goal_r = int(np.clip(goal_px[0] / cell_h, 0, grid_h - 1))
        goal_c = int(np.clip(goal_px[1] / cell_w, 0, grid_w - 1))
        goal = (goal_r, goal_c)
    else:
        goal = (grid_h - 2, grid_w - 2)

    # Ensure start and goal cells themselves are traversable in output grid
    grid[start[0], start[1]] = 0
    grid[goal[0], goal[1]] = 0

    # Create multi-panel debug overlay image
    if save_debug_path is not None:
        debug_out_path = Path(save_debug_path)
        debug_out_path.parent.mkdir(parents=True, exist_ok=True)

        # Panel 1: Original with detected start/goal
        p1 = bgr_img.copy()
        if start_px is not None:
            cv2.circle(p1, (int(start_px[1]), int(start_px[0])), 10, (0, 255, 0), -1)
        if goal_px is not None:
            cv2.circle(p1, (int(goal_px[1]), int(goal_px[0])), 10, (0, 0, 255), -1)

        # Panel 2: Morphological binary threshold
        p2 = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)

        # Panel 3: Downsampled grid visualizer
        grid_vis = np.full((grid_h * 10, grid_w * 10, 3), 240, dtype=np.uint8)
        for r in range(grid_h):
            for c in range(grid_w):
                if grid[r, c] == 1:
                    cv2.rectangle(
                        grid_vis, (c * 10, r * 10), ((c + 1) * 10, (r + 1) * 10), (50, 50, 50), -1
                    )
        cv2.rectangle(
            grid_vis,
            (start[1] * 10, start[0] * 10),
            ((start[1] + 1) * 10, (start[0] + 1) * 10),
            (0, 255, 0),
            -1,
        )
        cv2.rectangle(
            grid_vis,
            (goal[1] * 10, goal[0] * 10),
            ((goal[1] + 1) * 10, (goal[0] + 1) * 10),
            (0, 0, 255),
            -1,
        )

        # Resize all 3 panels to uniform 500x500 for clean side-by-side composite
        target_size = (500, 500)
        p1_r = cv2.resize(p1, target_size)
        p2_r = cv2.resize(p2, target_size)
        p3_r = cv2.resize(grid_vis, target_size)

        # Horizontal composite of 3 debug panels
        composite = np.hstack([p1_r, p2_r, p3_r])
        cv2.imwrite(str(debug_out_path), composite)


    return grid, start, goal
