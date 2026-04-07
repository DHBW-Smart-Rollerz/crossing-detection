"""
Filter functions for crossing detection.

This module contains ONLY the filter_* functions.
"""

import math

import cv2
import numpy as np

from crossing_detection.utils.helper import normalize_line

FILTERING_ROI_REL_RLTB = (0.80, 0, 0, 0.815)  # left, right, top, bottom


def filter_by_angle(
    lines,
    tol_deg: float = 25.0,
    anchor_angle=None,
    anchor_tolerance=None,
):
    """
    Filter lines based on their angle.

    If anchor_angle is provided, filters lines into vertical and horizontal
    categories relative to the anchor angle (tilted coordinate system).
    Otherwise, filters into traditional vertical (90°) and horizontal (0°).

    Arguments:
        lines -- List of lines as pairs of points.
        tol_deg -- Tolerance for vertical/horizontal classification.
        anchor_angle -- Optional anchor angle to tilt the coordinate system.
        anchor_tolerance -- Tolerance around anchor (for vertical lines).

    Returns:
        Tuple of (vertical_lines, horizontal_lines)
    """
    vertical = []
    horizontal = []

    if anchor_angle:
        anchor_angle = (anchor_angle + 360) % 180.0

    if lines is None or len(lines) == 0:
        return vertical, horizontal

    for idx, line in enumerate(lines):
        arr = line[0]
        if arr.size < 4:
            continue
        x1 = float(arr[0])
        y1 = float(arr[1])
        x2 = float(arr[2])
        y2 = float(arr[3])

        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue

        angle = math.degrees(math.atan2(dy, dx))

        angle_norm = (angle + 360.0) % 180.0

        if anchor_angle is not None and anchor_tolerance is not None:
            angle_diff = abs(angle_norm - anchor_angle)
            if angle_diff > 90.0:
                angle_diff = 180.0 - angle_diff

            perpendicular = (anchor_angle + 90.0) % 180.0
            perp_diff = abs(angle_norm - perpendicular)
            if perp_diff > 90.0:
                perp_diff = 180.0 - perp_diff

            # Classify: choose the CLOSEST category
            if angle_diff < perp_diff:
                # Line is closer to anchor angle (parallel)
                if angle_diff <= anchor_tolerance:
                    vertical.append(line)
            else:
                # Line is closer to perpendicular angle
                if perp_diff <= anchor_tolerance:
                    horizontal.append(line)
        else:
            dist_h = min(abs(angle_norm - 0.0), abs(angle_norm - 180.0))
            dist_v = min(abs(angle_norm - 80.0), abs(angle_norm - 100.0))

            if dist_v <= tol_deg + 10 and dist_v < dist_h:
                vertical.append(line)
            elif dist_h <= tol_deg and dist_h < dist_v:
                horizontal.append(line)

    return vertical, horizontal


def filter_by_roi(lines, img_shape):
    """
    Filter lines based on a region of interest (ROI).

    Arguments:
        lines -- List of lines as pairs of points.
        img_shape -- Shape of the image.

    Returns:
        Filtered list of lines.
    """
    height = img_shape[0]
    width = img_shape[1]

    roi_top = int(height * FILTERING_ROI_REL_RLTB[2])
    roi_bottom = int(height * FILTERING_ROI_REL_RLTB[3])
    roi_left = int(width * FILTERING_ROI_REL_RLTB[1])
    roi_right = int(width * FILTERING_ROI_REL_RLTB[0])

    res = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (
            roi_left <= x1 <= roi_right
            and roi_left <= x2 <= roi_right
            and roi_top <= y1 <= roi_bottom
            and roi_top <= y2 <= roi_bottom
        ):
            res.append(line)

    return res


def filter_by_length(lines, min_length=70.0, max_length=10000.0):
    """
    Filter lines based on their length.

    Arguments:
        lines -- List of lines as pairs of points.
        min_length -- Minimum length of the line to be kept.
        max_length -- Maximum length of the line to be kept.

    Returns:
        Filtered list of lines.
    """
    res = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length >= min_length and length <= max_length:
            res.append(line)

    return res


def filter_lines_by_polygon(lines, polygon, require_full=True):
    """
    Filter lines to those inside a polygon region.

    Arguments:
        lines -- iterable of lines (any accepted format).
        polygon -- list of points defining the polygon.
        require_full -- if True (default) keep only lines where both
                        endpoints are inside. If False, keep lines with
                        at least one endpoint inside.

    Returns:
        List of normalized lines (each as numpy array shape (1,4)).
    """
    if not polygon or lines is None:
        return []

    try:
        poly = np.array(polygon, dtype=np.int32)
    except Exception:
        return []

    filtered = []
    for ln in lines:
        nl = normalize_line(ln)
        if nl is None:
            continue
        x1, y1, x2, y2 = nl[0].astype(int)

        d1 = cv2.pointPolygonTest(poly, (int(x1), int(y1)), False)
        d2 = cv2.pointPolygonTest(poly, (int(x2), int(y2)), False)

        if require_full:
            if d1 >= 0 and d2 >= 0:
                filtered.append(nl)
        else:
            if d1 >= 0 or d2 >= 0:
                filtered.append(nl)

    return filtered


def filter_by_bev_black_corner(lines, black_polygon):
    """
    Filter lines whose center point is in the BEV black corner area.

    Lines with their midpoint inside the black corner area (created by the
    BEV transform) are removed from the list.

    Arguments:
        lines -- List of lines as pairs of points
        black_polygon -- Polygon (Nx2) from get_bev_black_corner_polygon

    Returns:
        Filtered list of lines (excludes lines in black corner)
    """
    if lines is None or len(lines) == 0:
        return []

    # Get the black corner polygon
    if black_polygon is None:
        return lines

    # Ensure polygon is the correct type for OpenCV
    black_polygon = np.asarray(black_polygon, dtype=np.int32)

    filtered = []
    for line in lines:
        nl = normalize_line(line)
        x1, y1, x2, y2 = nl[0].astype(float)

        # Calculate line center
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        # Check if center is inside the black area
        is_inside = cv2.pointPolygonTest(black_polygon, (center_x, center_y), False)

        # Only keep lines whose center is NOT in the black area
        if is_inside < 0:
            filtered.append(line)

    return filtered
