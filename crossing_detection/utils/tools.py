"""
Tool functions for crossing detection.

This module contains utility tool functions that are used in the pipeline.
"""

import math

import cv2
import numpy as np
from sklearn.decomposition import PCA

from crossing_detection.utils.helper import normalize_line, normalize_lines


def perform_canny(
    img,
    canny_threshold_low: int = 50,
    canny_threshold_high: int = 75,
):
    """
    Perform Canny edge detection on the image.

    Arguments:
        img -- Input image.
        canny_threshold_low -- Low threshold for Canny (default 50).
        canny_threshold_high -- High threshold for Canny (default 75).

    Returns:
        Image with edges detected.
    """
    img = cv2.Canny(img, canny_threshold_low, canny_threshold_high)
    return img


def enhance_by_line_brightness(
    image,
    lines,
    percentile: int = 80,
):
    """
    Enhance image contrast based on detected line brightness.

    Samples pixels from detected lines, computes the percentile brightness,
    then applies sigmoid-based contrast enhancement.

    Arguments:
        image -- Grayscale image
        lines -- List of detected lines [[x1,y1,x2,y2], ...]
        percentile -- Brightness percentile to use as threshold (default 80)

    Returns:
        Contrast-enhanced image
    """
    if image is None or lines is None or len(lines) == 0:
        return image

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    line_pixels = []
    h, w = gray.shape[:2]

    for line in lines:
        nl = normalize_line(line)
        if nl is None:
            continue

        x1, y1, x2, y2 = nl[0].astype(int)

        num_samples = 50
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0
            px = int(x1 + t * (x2 - x1))
            py = int(y1 + t * (y2 - y1))
            if 0 <= px < w and 0 <= py < h:
                line_pixels.append(int(gray[py, px]))

    if len(line_pixels) == 0:
        return image

    threshold = np.percentile(line_pixels, percentile)

    img_float = gray.astype(np.float32)
    normalized = (img_float - threshold) / 50.0
    enhanced = 1.0 / (1.0 + np.exp(-normalized))
    enhanced = (enhanced * 255).astype(np.uint8)

    return enhanced


def fuse_similar_lines(
    lines,
    angle_tol_deg: float = 10.0,
    center_dist_tol: float = 30.0,
    require_min_lines: int = 1,
):
    """
    Merge lines that are nearly parallel and close to each other into.
    single representative segments.

    Algorithm (greedy clustering):
    - Normalize lines to numpy (1,4).
    - Compute each line's angle and center.
    - Group lines whose angle difference <= angle_tol_deg and whose
      center-to-center distance <= center_dist_tol.
    - For each group, collect all endpoints, run PCA to get the main
      axis, project endpoints on that axis and take the extreme
      projected points as the fused segment endpoints.

    Arguments:
        lines -- List of lines
        angle_tol_deg -- Angle tolerance for grouping (default 10.0)
        center_dist_tol -- Distance tolerance for grouping (default 30.0)
        require_min_lines -- Minimum lines to fuse (default 1)

    Returns:
        List of fused lines as numpy arrays, each shape (1,4) (float32).
    """
    normalized = normalize_lines(lines)
    if not normalized:
        return []

    n = len(normalized)
    angles = np.zeros(n, dtype=np.float32)
    centers = np.zeros((n, 2), dtype=np.float32)
    for i, ln in enumerate(normalized):
        x1, y1, x2, y2 = ln[0]
        dx = x2 - x1
        dy = y2 - y1
        angles[i] = (math.degrees(math.atan2(dy, dx)) + 360.0) % 180.0
        centers[i] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    visited = [False] * n
    fused = []

    for i in range(n):
        if visited[i]:
            continue

        group_idx = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if visited[j]:
                continue

            diff = abs(angles[i] - angles[j])
            diff = min(diff, 180.0 - diff)
            if diff <= angle_tol_deg:
                d = float(np.hypot(*(centers[i] - centers[j])))
                if d <= center_dist_tol:
                    group_idx.append(j)
                    visited[j] = True

        if len(group_idx) < require_min_lines:
            for idx in group_idx:
                fused.append(normalized[idx])
            continue

        pts = []
        for idx in group_idx:
            x1, y1, x2, y2 = normalized[idx][0]
            pts.append([x1, y1])
            pts.append([x2, y2])
        pts = np.array(pts, dtype=np.float32)

        if pts.shape[0] < 2:
            fused.append(normalized[group_idx[0]])
            continue

        # Use sklearn PCA to get the main axis
        pca = PCA(n_components=1)
        pca.fit(pts)
        axis = pca.components_[0]

        mean = pts.mean(axis=0)
        scalars = (pts - mean).dot(axis)
        min_s = scalars.min()
        max_s = scalars.max()
        p1 = mean + axis * min_s
        p2 = mean + axis * max_s

        fused.append(
            np.array(
                [[p1[0], p1[1], p2[0], p2[1]]],
                dtype=np.float32,
            )
        )

    return fused


def elongate_line(line, length: float = 200.0):
    """
    Elongate the given line to the specified length.

    Arguments:
        line -- Line as a pair of points (numpy array shape (1,4)).
        length -- Target length (default 200.0).

    Returns:
        Elongated line as a list [[x1, y1, x2, y2]].
    """
    x1, y1, x2, y2 = line[0]
    delta_x = x2 - x1
    delta_y = y2 - y1
    line_len = math.sqrt(delta_x**2 + delta_y**2)
    factor = length / line_len if line_len != 0 else 0
    new_delta_x = delta_x * factor
    new_delta_y = delta_y * factor
    line_center_x = (x1 + x2) / 2
    line_center_y = (y1 + y2) / 2

    new_x1 = int(line_center_x - new_delta_x / 2)
    new_y1 = int(line_center_y - new_delta_y / 2)
    new_x2 = int(line_center_x + new_delta_x / 2)
    new_y2 = int(line_center_y + new_delta_y / 2)

    return [[new_x1, new_y1, new_x2, new_y2]]


def clip_line_to_vertical_bounds(
    line,
    roi_bounds,
    min_rel: float = 0.5,
    max_rel: float = 0.75,
):
    """
    Clip a line segment to vertical boundaries defined by ROI fractions.

    Arguments:
        line -- Line as numpy array shape (1,4).
        roi_bounds -- Tuple of (roi_left, roi_right, roi_top, roi_bottom).
        min_rel -- Minimum relative position (0.0 to 1.0) within ROI width.
        max_rel -- Maximum relative position (0.0 to 1.0) within ROI width.

    Returns:
        Clipped line as numpy array [[x1,y1,x2,y2]] (float32) or None if
        outside vertical band.
    """
    if line is None or roi_bounds is None:
        return None

    nl = normalize_line(line)
    if nl is None:
        return None

    x1, y1, x2, y2 = nl[0].astype(float)
    roi_left, roi_right, roi_top, roi_bottom = roi_bounds
    roi_w = float(roi_right - roi_left)
    min_x = roi_left + roi_w * float(min_rel)
    max_x = roi_left + roi_w * float(max_rel)

    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-3:
        if x1 < min_x or x1 > max_x:
            return None
        return np.array([[x1, y1, x2, y2]], dtype=np.float32)

    t_min = (min_x - x1) / dx
    t_max = (max_x - x1) / dx
    t0 = max(0.0, min(t_min, t_max))
    t1 = min(1.0, max(t_min, t_max))
    if t1 <= t0:
        return None

    nx1 = x1 + t0 * dx
    ny1 = y1 + t0 * dy
    nx2 = x1 + t1 * dx
    ny2 = y1 + t1 * dy

    return np.array([[nx1, ny1, nx2, ny2]], dtype=np.float32)


def clip_opp_line_adaptive(
    line,
    roi_bounds,
    angle: float = None,
    min_rel_base: float = 0.15,
    max_rel_base: float = 0.4,
):
    """
    Clip opposite line to vertical bounds with adaptive X-range based on angle.

    Arguments:
        line -- Line to clip (numpy array shape (1,4)).
        roi_bounds -- Tuple of (roi_left, roi_right, roi_top, roi_bottom).
        angle -- Prominent angle (None defaults to 90°).
        min_rel_base -- Base minimum relative position.
        max_rel_base -- Base maximum relative position.

    Returns:
        Tuple of (clipped_line, (min_rel, max_rel)) where min_rel and max_rel
        are the normalized bounds used for clipping.

        At 90° (straight): min_rel=0.15, max_rel=0.4
        At 67° (right curve): min_rel=0.35, max_rel=0.65
        Interpolates between these points based on angle.
    """
    if line is None or roi_bounds is None:
        return None, None

    if angle is None:
        angle = 90.0

    if angle < 90.0:
        angle_factor = (90.0 - angle) / (90.0 - 70.0)
        angle_factor = min(1.0, max(0.0, angle_factor))

        min_rel = min_rel_base + angle_factor * (0.05 - min_rel_base)
        max_rel = max_rel_base + angle_factor * (0.25 - max_rel_base)

    else:
        angle_factor = (90.0 - angle) / (90.0 - 110.0)
        angle_factor = min(1.0, max(0.0, angle_factor))

        min_rel = min_rel_base + angle_factor * (0.48 - min_rel_base)
        max_rel = max_rel_base + angle_factor * (0.70 - max_rel_base)

    clipped_line = clip_line_to_vertical_bounds(
        line, roi_bounds, min_rel=min_rel, max_rel=max_rel
    )
    return clipped_line, (min_rel, max_rel)


def clip_ego_line_adaptive(
    line,
    roi_bounds,
    angle: float = None,
    min_rel_base: float = 0.5,
    max_rel_base: float = 0.75,
):
    """
    Clip ego line to vertical bounds with adaptive X-range based on angle.

    Arguments:
        line -- Line to clip (numpy array shape (1,4)).
        roi_bounds -- Tuple of (roi_left, roi_right, roi_top, roi_bottom).
        angle -- Prominent angle (None defaults to 90°).
        min_rel_base -- Base minimum relative position.
        max_rel_base -- Base maximum relative position.

    Returns:
        Tuple of (clipped_line, (min_rel, max_rel)) where min_rel and max_rel
        are the normalized bounds used for clipping.

        At 90° (straight): min_rel=0.5, max_rel=0.75
        At 113° (left curve): min_rel=0.25, max_rel=0.55
        Interpolates between these points based on angle.
    """
    if line is None or roi_bounds is None:
        return None, None

    if angle is None:
        angle = 90.0

    if angle <= 90.0:
        angle_factor = (90.0 - angle) / (90.0 - 70.0)
        angle_factor = min(1.0, max(0.0, angle_factor))

        min_rel = min_rel_base + angle_factor * (0.42 - min_rel_base)
        max_rel = max_rel_base + angle_factor * (0.60 - max_rel_base)

    else:
        angle_factor = (angle - 90.0) / (110.0 - 90.0)
        angle_factor = min(1.0, max(0.0, angle_factor))

        min_rel = min_rel_base - angle_factor * (min_rel_base - 0.55)
        max_rel = max_rel_base - angle_factor * (max_rel_base - 0.8)

    clipped_line = clip_line_to_vertical_bounds(
        line, roi_bounds, min_rel=min_rel, max_rel=max_rel
    )
    return clipped_line, (min_rel, max_rel)


def is_line_dotted_by_gap_detection(
    line,
    image,
    box_half_width: int = 22,
    length_extend: float = 1.2,
    min_gap_count: int = 2,
    gap_size_min: int = 3,
):
    """
    Gap-based dotted/solid detection.

    Procedure:
    - Extract rotated box around the line.
    - Binarize the box using Otsu's method (camera/light independent).
    - Find continuous white segments along the horizontal profile.
    - Count gaps (white -> black -> white transitions).
    - If gaps >= min_gap_count => dotted.
    Returns (is_dotted: bool, gap_count: int, white_ratio: float, _=1)
    """
    if line is None or image is None:
        return False, 0, 0.0, 0

    nl = normalize_line(line)
    if nl is None:
        return False, 0, 0.0, 0

    x1, y1, x2, y2 = nl[0].astype(float)
    dx = x2 - x1
    dy = y2 - y1
    line_len = math.hypot(dx, dy)
    if line_len < 1e-3:
        return False, 0, 0.0, 0

    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 180

    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0

    crop_w = int(max(10, line_len * float(length_extend))) + int(box_half_width * 2)
    crop_h = int(max(3, box_half_width * 2))
    h, w = image.shape[:2]

    M = cv2.getRotationMatrix2D((mid_x, mid_y), angle, 1.0)
    warped = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

    cx = M[0, 0] * mid_x + M[0, 1] * mid_y + M[0, 2]
    cy = M[1, 0] * mid_x + M[1, 1] * mid_y + M[1, 2]

    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))
    x1c = max(0, x0)
    y1c = max(0, y0)
    x2c = min(w, x0 + crop_w)
    y2c = min(h, y0 + crop_h)

    if x2c <= x1c or y2c <= y1c:
        return False, 0, 0.0, 0

    crop = warped[y1c:y2c, x1c:x2c]
    if crop is None or crop.size == 0:
        return False, 0, 0.0, 0

    gray = crop
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    try:
        canny_edges = cv2.Canny(gray, threshold1=50, threshold2=150)

        gray_float = gray.astype(np.float32)

        white_pixels = gray_float[canny_edges > 0]
        if len(white_pixels) > 0:
            median_intensity = np.median(white_pixels)
        else:
            median_intensity = np.median(gray_float)

        normalized = (gray_float - median_intensity) / 50.0
        sigmoid_enhanced = 1.0 / (1.0 + np.exp(-normalized))
        sigmoid_enhanced = (sigmoid_enhanced * 255).astype(np.uint8)

        adaptive_binary = cv2.adaptiveThreshold(
            sigmoid_enhanced,
            1,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=3,
        )

        binary = (gray >= 60).astype(np.uint8) & adaptive_binary
        binary = binary.astype(bool)

        white_per_col = np.sum(binary, axis=0)

        gaps = 0
        in_gap = False
        gap_length = 0

        for i in range(len(white_per_col)):
            if white_per_col[i] == 0:
                if not in_gap:
                    in_gap = True
                    gap_length = 1
                else:
                    gap_length += 1
            else:
                if in_gap and gap_length >= gap_size_min:
                    gaps += 1
                in_gap = False
                gap_length = 0

        if in_gap and gap_length >= gap_size_min:
            gaps += 1

        actual_crop_h = int(y2c - y1c)
        actual_crop_w = int(x2c - x1c)
        total_white_pixels = float(np.sum(white_per_col))
        total_pixels = float(actual_crop_h * actual_crop_w)
        if total_pixels > 0:
            white_ratio = (total_white_pixels / total_pixels) * 100.0
        else:
            white_ratio = 0.0

        is_dotted = gaps >= min_gap_count

        return bool(is_dotted), int(gaps), float(white_ratio), 1

    except Exception as e:
        return False, 0, 0.0, 0


def get_bev_black_corner_polygon(
    img_shape,
    corner="bottom_left",
    corner_height_rel=0.2,
    corner_width_rel=0.15,
):
    """
    Calculate the black area polygon created by BEV transform in image corners.

    The BEV (Bird's Eye View) transform creates black/invalid areas in the
    corners of the transformed image. This function calculates those regions
    as polygons so lines in those areas can be filtered out.

    Arguments:
        img_shape -- Tuple of (height, width) or full image shape
        corner -- Which corner to calculate: "bottom_left", "bottom_right",
                  "top_left", "top_right"
        corner_height_rel -- Height of corner area as relative portion of
                             image height (0.0-1.0)
        corner_width_rel -- Width of corner area as relative portion of
                            image width (0.0-1.0)

    Returns:
        Numpy array polygon (Nx2) with corner vertices in clockwise order,
        or None if invalid corner specified
    """
    if isinstance(img_shape, tuple) and len(img_shape) >= 2:
        height = img_shape[0]
        width = img_shape[1]
    else:
        height = img_shape.shape[0]
        width = img_shape.shape[1]

    corner_h = int(height * corner_height_rel)
    corner_w = int(width * corner_width_rel)

    if corner == "bottom_left":
        # Bottom-left corner polygon: (0, height) -> (corner_w, height)
        # -> (0, height - corner_h) -> (0, height)
        polygon = np.array(
            [
                [0, height],
                [corner_w, height],
                [0, height - corner_h],
            ],
            dtype=np.int32,
        )
    elif corner == "bottom_right":
        # Bottom-right corner
        polygon = np.array(
            [
                [width, height],
                [width - corner_w, height],
                [width, height - corner_h],
            ],
            dtype=np.int32,
        )
    elif corner == "top_left":
        # Top-left corner
        polygon = np.array(
            [
                [0, 0],
                [corner_w, 0],
                [0, corner_h],
            ],
            dtype=np.int32,
        )
    elif corner == "top_right":
        # Top-right corner
        polygon = np.array(
            [
                [width, 0],
                [width - corner_w, 0],
                [width, corner_h],
            ],
            dtype=np.int32,
        )
    else:
        return None

    return polygon


def find_corners_shi_tomasi(image, roi_bbox=None):
    """
    Detect corners using Shi-Tomasi corner detection.

    (cv2.goodFeaturesToTrack).

    This helps identify the edges/corners of the intersection by detecting
    strong corner features that typically appear at road line junctions.

    Arguments:
        image -- Grayscale or color image to detect corners in.
        roi_bbox -- Optional tuple (left, right, top, bottom) to limit
                    corner detection to a specific region. If None, uses
                    full image.

    Returns:
        List of corner coordinates as tuples (x, y), or empty list if no
        corners are found.
    """
    if image is None:
        return []

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if roi_bbox is not None:
        roi_left, roi_right, roi_top, roi_bottom = roi_bbox
        roi_region = gray[roi_top:roi_bottom, roi_left:roi_right]
    else:
        roi_left = 0
        roi_right = gray.shape[1]
        roi_top = 0
        roi_bottom = gray.shape[0]
        roi_region = gray

    try:
        corners = cv2.goodFeaturesToTrack(
            roi_region,
            maxCorners=4,
            qualityLevel=0.01,
            minDistance=200,
            blockSize=3,
            useHarrisDetector=False,
        )

        if corners is not None:
            corners_list = []
            for corner in corners:
                x, y = corner.ravel()
                # Add ROI offset if we extracted a region
                corners_list.append((int(x + roi_left), int(y + roi_top)))
            return corners_list
        else:
            return []

    except Exception as e:
        msg = f"Shi-Tomasi corner detection failed: {e}"
        self.get_logger().warning(msg)
        return []


def find_heading_angle(lines, logger=None):
    """
    Find the prominent angle from all detected lines using histogram.

    Builds a histogram of angles from all lines, keeping only those
    within ±45° of 90° (vertical), and finds the peak angle.

    Arguments:
        lines -- List of detected lines (each line is [[x1, y1, x2, y2]])
        image -- Input image (unused, kept for compatibility)

    Returns:
        Tuple of (prominent_angle_deg, line_count) or (None, 0) if invalid
    """
    log_stuff = logger is not None

    if lines is None or len(lines) == 0 and log_stuff:
        logger.debug("No lines provided to histogram")
        return None, 0

    valid_angles = []
    lines_filtered_by_angle = 0

    for line in lines:
        try:
            nl = normalize_line(line)
            if nl is None:
                continue

            x1, y1, x2, y2 = nl[0].astype(int)

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0 and dy == 0:
                continue

            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)

            angle_norm = (angle_deg + 360.0) % 180.0

            angle_error = abs(angle_norm - 90.0)
            if angle_error > 45.0:
                lines_filtered_by_angle += 1
                continue

            valid_angles.append(angle_norm)

        except Exception:
            continue

    if log_stuff:
        logger.debug(f"Total lines: {len(lines)}")
        logger.debug(f"Lines filtered by angle (>45 deg): {lines_filtered_by_angle}")
        logger.debug(f"Valid angles for histogram: {len(valid_angles)}")

    if len(valid_angles) == 0:
        return None, 0

    valid_angles_arr = np.array(valid_angles)
    hist, bin_edges = np.histogram(valid_angles_arr, bins=36, range=(0, 180))

    peak_bin = int(np.argmax(hist))
    prominent_angle = float((bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0)

    count_str = len(valid_angles)
    msg = f"Prominent angle: {prominent_angle:.2f} deg ({count_str} lines)"

    if log_stuff:
        logger.debug(msg)

    return prominent_angle, len(valid_angles)
