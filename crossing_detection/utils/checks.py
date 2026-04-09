"""Stop line validation checks for crossing detection pipeline."""

import math

import numpy as np


def is_right_stop_line_valid(
    stop_line, crossing_center, min_x_inset, max_x_inset, min_x_right_stop
):
    """
    Check if right stop line is in valid position (y and x constraints).

    Validates:
    - Stop line is above crossing center (y position)
    - At least one endpoint above crossing
    - X position within valid bounds
    - X position not too close to left edge
    - X position not left of crossing center

    Arguments:
        stop_line -- Stop line as (1,4) array [x1, y1, x2, y2] or None
        crossing_center -- Tuple (x, y) of crossing center
        min_x_inset -- Minimum x position (left ROI bound with inset)
        max_x_inset -- Maximum x position (right ROI bound with inset)
        min_x_right_stop -- Minimum x for right stop line (60% from left)

    Returns:
        Tuple of (is_valid, rejection_reason) where rejection_reason
        is None if valid, or a string describing why it's invalid
    """
    if stop_line is None:
        return True, None

    x1, y1, x2, y2 = stop_line[0]
    stop_x = (x1 + x2) / 2.0
    stop_y = (y1 + y2) / 2.0

    # Check stop line is above crossing center (y position)
    if crossing_center is not None:
        crossing_y = crossing_center[1]
        if stop_y > crossing_y:
            return False, (f"y={stop_y:.1f} is below crossing y={crossing_y:.1f}")

        # At least one endpoint must be above crossing center
        min_y = min(y1, y2)
        if min_y > crossing_y:
            return False, (f"max_y={min_y:.1f} is below crossing y={crossing_y:.1f}")

    # Check horizontal bounds
    if stop_x < min_x_inset or stop_x > max_x_inset:
        return False, (
            f"x={stop_x:.1f} outside bounds " f"[{min_x_inset:.1f}, {max_x_inset:.1f}]"
        )

    # Check not too close to left edge
    if stop_x < min_x_right_stop:
        return False, (
            f"x={stop_x:.1f} too close to left edge " f"(min={min_x_right_stop:.1f})"
        )

    # Check not left of crossing center
    if crossing_center is not None:
        crossing_x = crossing_center[0]
        if stop_x < crossing_x:
            return False, (f"x={stop_x:.1f} is left of crossing x={crossing_x:.1f}")

    return True, None


def is_left_stop_line_valid(stop_line, crossing_center, max_x_left_stop):
    """
    Check if left stop line is in valid x and y position range.

    Arguments:
        stop_line -- Stop line as (1,4) array [x1, y1, x2, y2] or None
        crossing_center -- Tuple (x, y) of crossing center
        max_x_left_stop -- Maximum x for left stop line (40% from left)

    Returns:
        Tuple of (is_valid, rejection_reason) where rejection_reason
        is None if valid, or a string describing why it's invalid
    """
    if stop_line is None or crossing_center is None:
        return True, None

    x1, y1, x2, y2 = stop_line[0]
    stop_x = (x1 + x2) / 2.0
    crossing_x = crossing_center[0]
    crossing_y = crossing_center[1]

    # Check not right of crossing center
    if stop_x > crossing_x:
        return False, (f"x={stop_x:.1f} is right of crossing x={crossing_x:.1f}")

    # Check not too close to right edge
    if stop_x > max_x_left_stop:
        return False, (
            f"x={stop_x:.1f} too close to right edge " f"(max={max_x_left_stop:.1f})"
        )

    # Check endpoints relative to crossing
    max_y = max(y1, y2)
    if max_y < crossing_y:
        return False, (f"max_y={max_y:.1f} is below crossing y={crossing_y:.1f}")

    return True, None


def check_stop_line_crossing_openness(
    stop_line_right,
    stop_line_left,
    image_gray,
    logger,
    black_pixel_threshold: int = 40,
    black_pixel_pct_threshold: float = 55.0,
):
    """
    Check if crossing center area is open (not too dark).

    Finds the lowest point of right stop line and highest point of left
    stop line, then checks if these areas contain < 40% black pixels.
    This helps reject invalid stop line pairs that close off the
    crossing.

    Arguments:
        stop_line_right -- Right stop line as (1,4) array or None
        stop_line_left -- Left stop line as (1,4) array or None
        image_gray -- Grayscale image for pixel analysis

    Returns:
        Tuple (right_valid, left_valid) where each is True/False/None
    """
    if image_gray is None or image_gray.size == 0:
        return None, None

    right_valid = None
    left_valid = None

    if stop_line_right is not None:
        x1_r, y1_r, x2_r, y2_r = stop_line_right[0]
        bottom_y_r = max(y1_r, y2_r)
        bottom_x_r = x1_r if y1_r > y2_r else x2_r

        x_min = int(bottom_x_r - 12)
        x_max = int(bottom_x_r + 12)
        y_min = int(bottom_y_r)
        y_max = int(bottom_y_r + 70)

        x_min = max(0, x_min)
        x_max = min(image_gray.shape[1], x_max)
        y_min = max(0, y_min)
        y_max = min(image_gray.shape[0], y_max)

        if x_max > x_min and y_max > y_min:
            region = image_gray[y_min:y_max, x_min:x_max]
            total_pixels = region.size
            black_pixels = np.sum(region < black_pixel_threshold)
            black_pct = (black_pixels / total_pixels * 100) if total_pixels > 0 else 0

            right_valid = black_pct > black_pixel_pct_threshold
            logger.debug(
                f"RIGHT stop lowest point "
                f"(x={bottom_x_r:.1f}, y={bottom_y_r:.1f}): "
                f"{black_pct:.1f}%"
            )
        else:
            right_valid = False

    if stop_line_left is not None:
        x1_l, y1_l, x2_l, y2_l = stop_line_left[0]
        top_y_l = min(y1_l, y2_l)
        top_x_l = x1_l if y1_l < y2_l else x2_l

        x_min = int(top_x_l - 12)
        x_max = int(top_x_l + 12)
        y_min = int(top_y_l - 70)
        y_max = int(top_y_l)

        x_min = max(0, x_min)
        x_max = min(image_gray.shape[1], x_max)
        y_min = max(0, y_min)
        y_max = min(image_gray.shape[0], y_max)

        if x_max > x_min and y_max > y_min:
            region = image_gray[y_min:y_max, x_min:x_max]
            total_pixels = region.size
            black_pixels = np.sum(region < black_pixel_threshold)
            black_pct = (black_pixels / total_pixels * 100) if total_pixels > 0 else 0

            left_valid = black_pct > black_pixel_pct_threshold
            logger.debug(
                f"LEFT stop highest point "
                f"(x={top_x_l:.1f}, y={top_y_l:.1f}): "
                f"{black_pct:.1f}%"
            )
        else:
            left_valid = False

    return right_valid, left_valid


def measure_stop_line_thickness(stop_line_left, stop_line_right, image_gray, logger):
    """
    Measure the thickness of stop lines using orthogonal crosses.

    Creates an orthogonal line through the middle of each stop line and
    counts white pixels along that orthogonal to measure line thickness.

    Arguments:
        stop_line_right -- Right stop line as (1,4) array or None
        stop_line_left -- Left stop line as (1,4) array or None
        image_gray -- Grayscale image for pixel analysis

    Returns:
        Tuple (right_thickness, left_thickness) in pixels
    """
    try:
        if image_gray is None or image_gray.size == 0:
            return None, None

        right_thickness = None
        left_thickness = None

        if stop_line_right is not None:
            try:
                x1_r, y1_r, x2_r, y2_r = stop_line_right[0]

                mid_x_r = (x1_r + x2_r) / 2.0
                mid_y_r = (y1_r + y2_r) / 2.0

                dx = x2_r - x1_r
                dy = y2_r - y1_r
                line_length = np.sqrt(dx**2 + dy**2)

                if line_length > 0:
                    dx_norm = dx / line_length
                    dy_norm = dy / line_length

                    orth_dx = -dy_norm
                    orth_dy = dx_norm

                    max_dist = 25
                    thickness = 0
                    pixel_values = []

                    for dist in range(-max_dist, max_dist + 1):
                        px = int(mid_x_r + dist * orth_dx)
                        py = int(mid_y_r + dist * orth_dy)

                        if (
                            0 <= px < image_gray.shape[1]
                            and 0 <= py < image_gray.shape[0]
                        ):
                            try:
                                pv = image_gray[py, px]
                                if isinstance(pv, np.ndarray):
                                    pv = float(pv.flat[0])
                                else:
                                    pv = float(pv)
                                pixel_values.append(pv)
                                if pv > 100:
                                    thickness += 1
                            except (ValueError, IndexError):
                                pass

                    right_thickness = thickness
                    avg_pix = (
                        sum(pixel_values) / len(pixel_values) if pixel_values else 0
                    )
                    logger.debug(
                        f"RIGHT thickness: {right_thickness} px, "
                        f"avg={avg_pix:.0f} "
                        f"(mid x={mid_x_r:.1f}, y={mid_y_r:.1f})"
                    )
            except Exception as e:
                logger.error(f"Error measuring RIGHT thickness: {e}")

        if stop_line_left is not None:
            try:
                x1_l, y1_l, x2_l, y2_l = stop_line_left[0]

                mid_x_l = (x1_l + x2_l) / 2.0
                mid_y_l = (y1_l + y2_l) / 2.0

                dx = x2_l - x1_l
                dy = y2_l - y1_l
                line_length = np.sqrt(dx**2 + dy**2)

                if line_length > 0:
                    dx_norm = dx / line_length
                    dy_norm = dy / line_length

                    orth_dx = -dy_norm
                    orth_dy = dx_norm

                    max_dist = 25
                    thickness = 0
                    pixel_values = []

                    for dist in range(-max_dist, max_dist + 1):
                        px = int(mid_x_l + dist * orth_dx)
                        py = int(mid_y_l + dist * orth_dy)

                        if (
                            0 <= px < image_gray.shape[1]
                            and 0 <= py < image_gray.shape[0]
                        ):
                            try:
                                pv = image_gray[py, px]
                                if isinstance(pv, np.ndarray):
                                    pv = float(pv.flat[0])
                                else:
                                    pv = float(pv)
                                pixel_values.append(pv)
                                if pv > 100:
                                    thickness += 1
                            except (ValueError, IndexError):
                                pass

                    left_thickness = thickness
                    avg_pix = (
                        sum(pixel_values) / len(pixel_values) if pixel_values else 0
                    )
                    logger.debug(
                        f"LEFT thickness: {left_thickness} px, "
                        f"avg={avg_pix:.0f} "
                        f"(mid x={mid_x_l:.1f}, y={mid_y_l:.1f})"
                    )
            except Exception as e:
                logger.error(f"Error measuring LEFT thickness: {e}")

        return left_thickness, right_thickness

    except Exception as e:
        logger.error(f"Error in measure_stop_line_thickness: {e}")
        return None, None


def check_stop_line_pair_plausibility(
    stop_line_left,
    stop_line_right,
    max_y_diff: float = 30.0,
    min_y_diff: float = 0.0,
    max_x_separation: float = 200.0,
    min_x_separation: float = 50.0,
):
    """
    Validate stop lines: accept single lines or plausible pairs.

    If both stop lines exist, they should:
    - Be at roughly the same vertical position (y-coords close)
    - Have appropriate horizontal separation (not too close, not too far)

    If only one exists, it's accepted as valid.

    Arguments:
        stop_line_left -- Left stop line or None
        stop_line_right -- Right stop line or None
        max_y_diff -- Max vertical distance between left/right stop (pixels)
        min_y_diff -- Min vertical distance between left/right stop (pixels)
        max_x_separation -- Max horizontal separation between stops (pixels)
        min_x_separation -- Min horizontal separation between stops (pixels)

    Returns:
        Tuple of (validated_left, validated_right)
    """
    if stop_line_left is None and stop_line_right is None:
        return None, None

    if (stop_line_left is None) != (stop_line_right is None):
        return stop_line_left, stop_line_right

    if stop_line_left is not None and stop_line_right is not None:
        x1_l, y1_l, x2_l, y2_l = stop_line_left[0]
        x1_r, y1_r, x2_r, y2_r = stop_line_right[0]

        y_left = (y1_l + y2_l) / 2.0
        y_right = (y1_r + y2_r) / 2.0
        x_left = (x1_l + x2_l) / 2.0
        x_right = (x1_r + x2_r) / 2.0

        y_diff = abs(y_left - y_right)
        if y_diff > max_y_diff or y_diff < min_y_diff:
            # Not aligned vertically - likely false positives
            return None, None

        x_sep = abs(x_right - x_left)
        if x_sep < min_x_separation or x_sep > max_x_separation:
            return None, None

        return stop_line_left, stop_line_right

    return None, None


def is_ego_roi_and_distance_valid(clipped_line, crossing_center, image_shape, roi_box):
    """
    Check if ego line is within ROI bounds and distance.

    Validates:
    - Line endpoints within ROI x bounds
    - Line endpoints within ROI y bounds (with tolerance)
    - Line center within 250px of crossing center

    Arguments:
        clipped_line -- Line as (1,4) array or None
        crossing_center -- Tuple (x, y) of crossing center
        image_shape -- Shape of image for ROI calculation

    Returns:
        Boolean indicating if line passes all checks
    """
    if clipped_line is None or crossing_center is None:
        return False

    roi_left, roi_right, roi_top, roi_bottom = roi_box

    x1, y1, x2, y2 = (
        float(clipped_line[0][0]),
        float(clipped_line[0][1]),
        float(clipped_line[0][2]),
        float(clipped_line[0][3]),
    )

    # Check X bounds
    in_roi_x = roi_left <= x1 <= roi_right and roi_left <= x2 <= roi_right
    # Check Y bounds with tolerance
    y_tolerance = 10
    y_min_tol = roi_top - y_tolerance
    y_max_tol = roi_bottom + y_tolerance
    in_roi_y = (y_min_tol <= y1 <= y_max_tol) and (y_min_tol <= y2 <= y_max_tol)

    # Check distance to crossing center
    line_center_x = (x1 + x2) / 2.0
    line_center_y = (y1 + y2) / 2.0
    dist_to_center = math.sqrt(
        (line_center_x - crossing_center[0]) ** 2
        + (line_center_y - crossing_center[1]) ** 2
    )
    within_distance = dist_to_center <= 250

    return in_roi_x and in_roi_y and within_distance


def check_plausibility_horizontal_line_pair(
    opp_line,
    ego_line,
    intersection_point,
    line_horizontal_distance_threshold: float = 100.0,
    line_vertical_distance_threshold: float = 150.0,
    center_horizontal_distance_threshold: float = 100.0,
    negative_line_overlap_threshold: float = -80.0,
):
    """
    Check the plausibility of a pair of horizontal lines.

    Arguments:
        line1 -- First horizontal line as a pair of points.
        line2 -- Second horizontal line as a pair of points.
        intersection_point -- Intersection point as (x, y).

    Returns:
        True if the pair is plausible, False otherwise.
    """
    x1_1, y1_1, x2_1, y2_1 = ego_line[0]
    x1_2, y1_2, x2_2, y2_2 = opp_line[0]
    ego_line_leftmost = min(x1_1, x2_1)
    opp_line_rightmost = max(x1_2, x2_2)
    distance_between_lines_horizontal = ego_line_leftmost - opp_line_rightmost
    if distance_between_lines_horizontal > line_horizontal_distance_threshold:
        return False

    distance_between_lines_vertical = abs(((y1_1 + y2_1) / 2) - ((y1_2 + y2_2) / 2))
    if distance_between_lines_vertical < line_vertical_distance_threshold:
        return False

    return True


def check_line_right_y_pos(stop_line_right, opp_line_long):
    """
    Check if right stop line is above opp line (valid y position).

    Arguments:
        stop_line_right -- Right stop line as (1,4) array or None
        opp_line_long -- Opp line as (1,4) array or None

    Returns:
        Boolean indicating if check passes (True if stop_y >= opp_y)
    """
    if stop_line_right is None or opp_line_long is None:
        return True

    x1_o, y1_o, x2_o, y2_o = opp_line_long[0]
    opp_y = (y1_o + y2_o) / 2.0
    x1_r, y1_r, x2_r, y2_r = stop_line_right[0]
    stop_y_r = (y1_r + y2_r) / 2.0

    return stop_y_r >= opp_y


def check_line_left_y_pos(stop_line_left, ego_line_long, opp_line_long):
    """
    Check if left stop line is in valid y position relative to ego/opp.

    Validates:
    - Stop line below ego line (left_y < ego_y)
    - Stop line above opp line (left_y > opp_y)

    Arguments:
        stop_line_left -- Left stop line as (1,4) array or None
        ego_line_long -- Ego line as (1,4) array or None
        opp_line_long -- Opp line as (1,4) array or None

    Returns:
        Boolean indicating if both checks pass
    """
    if stop_line_left is None:
        return True

    x1_l, y1_l, x2_l, y2_l = stop_line_left[0]
    left_y = (y1_l + y2_l) / 2.0

    # Check with ego line
    if ego_line_long is not None:
        x1_e, y1_e, x2_e, y2_e = ego_line_long[0]
        ego_y = (y1_e + y2_e) / 2.0
        if left_y >= ego_y:
            return False

    # Check with opp line
    if opp_line_long is not None:
        x1_o, y1_o, x2_o, y2_o = opp_line_long[0]
        opp_y = (y1_o + y2_o) / 2.0
        if left_y <= opp_y:
            return False

    return True
