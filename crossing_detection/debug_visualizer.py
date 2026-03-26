"""Debug visualization utilities for crossing detection."""

import math

import cv2
import numpy as np

# Color constants (BGR format for OpenCV)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
LIME = (0, 255, 0)
PINK = (255, 192, 203)
VIOLET = (148, 0, 211)
TURQUOISE = (200, 230, 240)
GOLD = (255, 215, 0)


class CrossingDebugVisualizer:
    """Handles all debug visualization for crossing detection."""

    def __init__(self, node=None):
        """
        Initialize the debug visualizer.

        Arguments:
            node -- Optional reference to parent node for accessing methods
        """
        self.node = node
        self.debug_overlay_images = []

    def render_debug_overlays(
        self,
        image,
        # Lines and basic detection
        vert=None,
        horiz=None,
        joined_lines=None,
        ego_line_long=None,
        opp_line_long=None,
        pair_plausible=False,
        # Crossing center and corners
        crossing_center=None,
        detected_corners=None,
        # Ghost crossing centers
        ego_ghost_cc=None,
        opp_ghost_cc=None,
        left_stop_ghost_cc=None,
        right_stop_ghost_cc=None,
        # Stop lines
        stop_line_left=None,
        stop_line_right=None,
        stop_line_left_ext=None,
        stop_line_right_ext=None,
        label_stop_line_left=None,
        label_stop_line_right=None,
        # Labels and angles
        label=None,
        label2=None,
        closest_line_angle=None,
        # Quadrants
        q1=None,
        q2=None,
        q3=None,
        q4=None,
        # Crossing type and stability
        crossing_type=None,
        is_stable=False,
        buffer_levels=None,
        overall_confidence=0.0,
        # Enhanced image
        enhanced_image=None,
        # Unused for compatibility
        transformed_lines=None,
        filtered_lines=None,
        cone_left=None,
        cone_right=None,
        cl_vert=None,
        cl_vert_left=None,
        ego_clip_bounds=None,
        opp_clip_bounds=None,
    ):
        """
        COMPLETE visualization pipeline - single function for all debug visu.

        Pass all parameters needed and get a fully rendered debug image.

        Returns:
            Fully rendered debug image with all visualizations
        """
        result_image = image.copy()

        # ============================================================
        # PHASE 1: MAIN LINE DETECTION OVERLAYS
        # ============================================================
        if vert is not None:
            result_image = self._draw_lines(result_image, vert, color=GOLD)

        if joined_lines is not None:
            result_image = self._draw_lines(
                result_image, joined_lines, color=YELLOW, thickness=3
            )

        # ============================================================
        # PHASE 2: CROSSING CENTER AND GHOST CCs
        # ============================================================
        if crossing_center is not None:
            try:
                result_image = cv2.circle(result_image, crossing_center, 8, YELLOW)
                crossing_center_pulled = self._pull_point_to_roi_center(
                    crossing_center, result_image.shape
                )
                result_image = cv2.circle(
                    result_image,
                    (
                        int(crossing_center_pulled[0]),
                        int(crossing_center_pulled[1]),
                    ),
                    8,
                    TURQUOISE,
                )
            except Exception:
                pass

        # Draw all ghost crossing centers
        try:
            if ego_ghost_cc is not None:
                cv2.circle(result_image, ego_ghost_cc, 6, RED, 2)
                cv2.putText(
                    result_image,
                    "EGO_G",
                    (ego_ghost_cc[0] - 15, ego_ghost_cc[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    RED,
                    1,
                )

            if opp_ghost_cc is not None:
                cv2.circle(result_image, opp_ghost_cc, 6, BLUE, 2)
                cv2.putText(
                    result_image,
                    "OPP_G",
                    (opp_ghost_cc[0] - 15, opp_ghost_cc[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    BLUE,
                    1,
                )

            if left_stop_ghost_cc is not None:
                cv2.circle(result_image, left_stop_ghost_cc, 6, ORANGE, 2)
                cv2.putText(
                    result_image,
                    "LEFT_G",
                    (left_stop_ghost_cc[0] - 20, left_stop_ghost_cc[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    ORANGE,
                    1,
                )

            if right_stop_ghost_cc is not None:
                cv2.circle(result_image, right_stop_ghost_cc, 6, PINK, 2)
                cv2.putText(
                    result_image,
                    "RIGHT_G",
                    (right_stop_ghost_cc[0] - 25, right_stop_ghost_cc[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    PINK,
                    1,
                )
        except Exception:
            pass

        # ============================================================
        # PHASE 3: DETECTED CORNERS
        # ============================================================
        if detected_corners is not None and len(detected_corners) > 0:
            try:
                sorted_corners, interior_angles = self.node.compute_corner_angles(
                    detected_corners
                )

                for i, corner in enumerate(detected_corners):
                    x, y = corner
                    color = PINK if i % 2 == 0 else ORANGE
                    cross_size = 8
                    cv2.line(
                        result_image,
                        (x - cross_size, y),
                        (x + cross_size, y),
                        color,
                        2,
                    )
                    cv2.line(
                        result_image,
                        (x, y - cross_size),
                        (x, y + cross_size),
                        color,
                        2,
                    )

                if sorted_corners is not None and interior_angles is not None:
                    for idx, angle_deg in enumerate(interior_angles):
                        corner = sorted_corners[idx].astype(int)
                        cx, cy = corner[0], corner[1]
                        angle_text = f"{angle_deg:.0f}°"
                        cv2.putText(
                            result_image,
                            angle_text,
                            (cx - 15, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            GOLD,
                            1,
                        )

                corner_label = f"Corners: {len(detected_corners)}"
                cv2.putText(
                    result_image,
                    corner_label,
                    (0, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    PINK,
                )

                if sorted_corners is not None and interior_angles is not None:
                    is_rect = self.node.is_valid_rectangle(
                        interior_angles, angle_tolerance=20.0
                    )
                    rect_status = "✓ RECT" if is_rect else "✗ NOT RECT"
                    status_color = GREEN if is_rect else RED

                    angle_error = self.node.compute_angle_error(interior_angles)
                    error_text = f"Angle Error: {angle_error:.1f}°"

                    cv2.putText(
                        result_image,
                        rect_status,
                        (0, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        status_color,
                    )
                    cv2.putText(
                        result_image,
                        error_text,
                        (0, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        GOLD,
                    )

                    if self.node.active_crossing_center is not None:
                        frame_text = (
                            f"Hold Frame: " f"{self.node.crossing_center_frames}/4"
                        )
                        cv2.putText(
                            result_image,
                            frame_text,
                            (0, 165),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            CYAN,
                        )
            except Exception:
                pass

        # ============================================================
        # PHASE 4: EGO AND OPP LINES
        # ============================================================
        try:
            if ego_line_long is not None:
                result_image = self._draw_lines(
                    result_image,
                    [ego_line_long],
                    color=LIME,
                    thickness=6,
                )
            if opp_line_long is not None:
                result_image = self._draw_lines(
                    result_image,
                    [opp_line_long],
                    color=LIME if pair_plausible else ORANGE,
                    thickness=6,
                )
        except Exception:
            pass

        # ============================================================
        # PHASE 5: QUADRANTS
        # ============================================================
        if q1 is not None and q2 is not None and q3 is not None and q4 is not None:
            try:
                for quad in [q1, q2, q3, q4]:
                    pts = np.array(quad, np.int32)
                    cv2.polylines(result_image, [pts], True, (100, 100, 100), 1)
            except Exception:
                pass

        # ============================================================
        # PHASE 6: STOP LINES AND METRICS
        # ============================================================
        if stop_line_left is not None:
            x1, y1, x2, y2 = [int(v) for v in stop_line_left[0]]
            cv2.line(result_image, (x1, y1), (x2, y2), VIOLET, 2)

        if stop_line_right is not None:
            x1, y1, x2, y2 = [int(v) for v in stop_line_right[0]]
            cv2.line(result_image, (x1, y1), (x2, y2), VIOLET, 2)

        if stop_line_left_ext is not None:
            x1, y1, x2, y2 = [int(v) for v in stop_line_left_ext[0]]
            cv2.line(result_image, (x1, y1), (x2, y2), CYAN, 1)

        if stop_line_right_ext is not None:
            x1, y1, x2, y2 = [int(v) for v in stop_line_right_ext[0]]
            cv2.line(result_image, (x1, y1), (x2, y2), CYAN, 1)

        # Draw stop line metrics
        if stop_line_left is not None and stop_line_right is not None:
            try:
                x1_l, y1_l, x2_l, y2_l = stop_line_left[0]
                x1_r, y1_r, x2_r, y2_r = stop_line_right[0]

                y_left = (y1_l + y2_l) / 2.0
                y_right = (y1_r + y2_r) / 2.0
                x_left = (x1_l + x2_l) / 2.0
                x_right = (x1_r + x2_r) / 2.0

                y_diff = abs(y_left - y_right)
                x_sep = abs(x_right - x_left)

                y_mid = int((y_left + y_right) / 2.0)
                x_left_int = int(x_left)
                x_right_int = int(x_right)
                cv2.line(
                    result_image,
                    (x_left_int, y_mid),
                    (x_right_int, y_mid),
                    CYAN,
                    1,
                )

                diff_line_height = 20
                cv2.line(
                    result_image,
                    (x_left_int, int(y_left) - diff_line_height),
                    (x_left_int, int(y_left) + diff_line_height),
                    YELLOW,
                    1,
                )
                cv2.line(
                    result_image,
                    (x_right_int, int(y_right) - diff_line_height),
                    (x_right_int, int(y_right) + diff_line_height),
                    YELLOW,
                    1,
                )

                metrics_y = 100
                metrics_text_color = GREEN
                cv2.putText(
                    result_image,
                    f"y_diff={y_diff:.1f}px",
                    (result_image.shape[1] - 200, metrics_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    metrics_text_color,
                )
                cv2.putText(
                    result_image,
                    f"x_sep={x_sep:.1f}px",
                    (result_image.shape[1] - 200, metrics_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    metrics_text_color,
                )
            except Exception:
                pass

        # Draw stop line labels
        y_offset = 20
        if label_stop_line_left is not None:
            cv2.putText(
                result_image,
                label_stop_line_left,
                (0, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                GREEN,
            )
            y_offset += 25

        if label_stop_line_right is not None:
            cv2.putText(
                result_image,
                label_stop_line_right,
                (0, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                GREEN,
            )

        # Draw debug overlay images
        if len(self.debug_overlay_images) > 0:
            cv2.putText(
                result_image,
                "Line Detection Debug Overlays (warped images):",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            overlay_height = result_image.shape[0] // 3
            overlay_width = result_image.shape[1] // len(self.debug_overlay_images[:3])
            y_start = int(result_image.shape[0] * 0.6)
            for i, overlay_img in enumerate(self.debug_overlay_images[:3]):
                resized = cv2.resize(
                    overlay_img,
                    (overlay_width, overlay_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                x_start = i * overlay_width
                try:
                    result_image[
                        y_start : y_start + overlay_height,
                        x_start : x_start + overlay_width,
                    ] = resized
                except Exception:
                    pass
            self.debug_overlay_images.clear()

        # ============================================================
        # PHASE 7: LABELS AND ANGLE ARROW
        # ============================================================
        try:
            if label is not None:
                cv2.putText(
                    result_image,
                    label,
                    (0, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    GREEN,
                )
            if label2 is not None:
                cv2.putText(
                    result_image,
                    label2,
                    (0, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    GREEN,
                )
            if closest_line_angle is not None:
                self._draw_angle_arrow(result_image, closest_line_angle)
        except Exception:
            pass

        # ============================================================
        # PHASE 8: ROI BOX
        # ============================================================
        try:
            roi_left, roi_right, roi_top, roi_bottom = self.node.get_roi_bbox(
                result_image.shape
            )
            cv2.rectangle(
                result_image, (roi_left, roi_top), (roi_right, roi_bottom), RED, 2
            )
        except Exception:
            pass

        # ============================================================
        # PHASE 9: CROSSING TYPE VISUALIZATION
        # ============================================================
        if crossing_type is not None:
            self._draw_crossing_type_visualization(
                result_image,
                crossing_type,
                is_stable,
                buffer_levels,
                overall_confidence,
            )

        # ============================================================
        # PHASE 10: ENHANCED IMAGE CORNER
        # ============================================================
        if enhanced_image is not None:
            try:
                target_width = 250
                aspect_ratio = enhanced_image.shape[0] / enhanced_image.shape[1]
                target_height = int(target_width * aspect_ratio)

                resized_enhanced = cv2.resize(
                    enhanced_image,
                    (target_width, target_height),
                    interpolation=cv2.INTER_LINEAR,
                )

                if len(resized_enhanced.shape) == 2:
                    resized_enhanced = cv2.cvtColor(
                        resized_enhanced, cv2.COLOR_GRAY2BGR
                    )

                margin = 5
                x_pos = result_image.shape[1] - target_width - margin
                y_pos = result_image.shape[0] - target_height - margin

                if x_pos >= 0 and y_pos >= 0:
                    try:
                        result_image[
                            y_pos : y_pos + target_height,
                            x_pos : x_pos + target_width,
                        ] = resized_enhanced
                    except Exception:
                        pass
            except Exception:
                pass

        return result_image

    def _draw_crossing_type_visualization(
        self,
        image,
        crossing_type_str,
        is_stable=False,
        buffer_levels=None,
        overall_confidence=0.0,
    ):
        """Draw crossing type visualization in top right corner."""
        try:
            height, width = image.shape[:2]

            parts = crossing_type_str.split("-")
            if len(parts) not in [4, 5]:
                return

            ego_type = parts[0]
            opp_type = parts[1]
            stop_l_type = parts[2]
            stop_r_type = parts[3]

            panel_x = width - 90
            panel_y = 80
            square_size = 50

            overlay = image.copy()
            cv2.rectangle(
                overlay,
                (panel_x - 10, panel_y - 10),
                (panel_x + square_size + 10, panel_y + square_size + 10),
                (0, 0, 0),
                -1,
            )
            cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

            def get_color_and_style(line_type):
                """Get color and dotted flag from line type."""
                is_detected = line_type[1] != "n"
                is_dotted = line_type[1] == "d"
                color = GREEN if is_detected else RED
                return color, is_dotted

            def draw_styled_line(pt1, pt2, color, is_dotted, thickness=2):
                """Draw a line (solid or dotted)."""
                if is_dotted:
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    length = np.sqrt(dx**2 + dy**2)
                    if length == 0:
                        return

                    dash_length = 5
                    gap_length = 4
                    step = dash_length + gap_length

                    for i in range(0, int(length), step):
                        t1 = i / length
                        t2 = min((i + dash_length) / length, 1.0)

                        p1 = (
                            int(pt1[0] + t1 * dx),
                            int(pt1[1] + t1 * dy),
                        )
                        p2 = (
                            int(pt1[0] + t2 * dx),
                            int(pt1[1] + t2 * dy),
                        )
                        cv2.line(image, p1, p2, color, thickness)
                else:
                    cv2.line(image, pt1, pt2, color, thickness)

            ego_color, ego_dotted = get_color_and_style(ego_type)
            opp_color, opp_dotted = get_color_and_style(opp_type)
            left_color, left_dotted = get_color_and_style(stop_l_type)
            right_color, right_dotted = get_color_and_style(stop_r_type)

            top_left = (panel_x, panel_y)
            top_right = (panel_x + square_size, panel_y)
            bottom_left = (panel_x, panel_y + square_size)
            bottom_right = (panel_x + square_size, panel_y + square_size)

            draw_styled_line(top_left, top_right, opp_color, opp_dotted, thickness=2)
            draw_styled_line(
                bottom_left, bottom_right, ego_color, ego_dotted, thickness=2
            )
            draw_styled_line(
                top_left, bottom_left, left_color, left_dotted, thickness=2
            )
            draw_styled_line(
                top_right, bottom_right, right_color, right_dotted, thickness=2
            )

            if is_stable:
                purple = (128, 0, 128)
                dot_x = panel_x + square_size + 20
                dot_y = panel_y + square_size // 2
                cv2.circle(image, (dot_x, dot_y), 5, purple, -1)

            if buffer_levels is not None:
                buffer_y_start = panel_y + square_size + 50
                buffer_height = 60
                buffer_width = 180
                buffer_x = panel_x - 30

                overlay = image.copy()
                cv2.rectangle(
                    overlay,
                    (buffer_x, buffer_y_start),
                    (buffer_x + buffer_width, buffer_y_start + buffer_height),
                    (0, 0, 0),
                    -1,
                )
                cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

                line_height = 15
                buffers = [
                    ("EGO", buffer_levels.get("ego", 0.0)),
                    ("OPP", buffer_levels.get("opp", 0.0)),
                    ("LEFT", buffer_levels.get("stop_left", 0.0)),
                    ("RIGHT", buffer_levels.get("stop_right", 0.0)),
                ]

                for idx, (name, level) in enumerate(buffers):
                    y_offset = buffer_y_start + 18 + idx * line_height
                    value_str = f"{level:.2f}"

                    if level < 0.3:
                        color = (255, 255, 255)
                    elif level < 0.6:
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)

                    cv2.putText(
                        image,
                        f"{name}:{value_str}",
                        (buffer_x + 5, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                    )

            conf_y_start = (
                buffer_y_start + buffer_height + 15
                if buffer_levels is not None
                else panel_y + square_size + 50
            )
            conf_color = (50, 50, 50)
            overlay = image.copy()
            cv2.rectangle(
                overlay,
                (buffer_x if buffer_levels is not None else panel_x - 30, conf_y_start),
                (
                    (buffer_x if buffer_levels is not None else panel_x - 30) + 180,
                    conf_y_start + 25,
                ),
                conf_color,
                -1,
            )
            cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

            if overall_confidence < 0.3:
                conf_color = (0, 0, 255)
            elif overall_confidence < 0.6:
                conf_color = (0, 165, 255)
            else:
                conf_color = (0, 255, 0)

            cv2.putText(
                image,
                f"Overall: {overall_confidence:.2f}",
                (
                    (buffer_x if buffer_levels is not None else panel_x - 30) + 5,
                    conf_y_start + 18,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                conf_color,
                1,
            )

        except Exception:
            pass

    def _draw_lines(self, img, lines, color=(255, 0, 0), thickness=2):
        """Draw lines on the image."""
        img2 = img[::]
        if lines is None or (hasattr(lines, "__len__") and len(lines) == 0):
            return img2
        for line in lines:
            nl = self.node._normalize_line(line)
            if nl is None:
                continue
            x1, y1, x2, y2 = nl[0]
            cv2.line(
                img2,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness,
            )

        return img2

    def _draw_angle_arrow(self, image, angle_deg, arrow_length=25):
        """Draw an arrow showing the detected angle."""
        try:
            height, width = image.shape[:2]
            center_x = width - 45
            center_y = 40
            angle_rad = math.radians(float(angle_deg))
            end_x = center_x - arrow_length * math.cos(-angle_rad)
            end_y = center_y + arrow_length * math.sin(-angle_rad)
            cv2.arrowedLine(
                image,
                (int(center_x), int(center_y)),
                (int(end_x), int(end_y)),
                CYAN,
                thickness=2,
                tipLength=0.3,
            )
        except Exception:
            pass

    def _pull_point_to_roi_center(self, point, img_shape):
        """Pull a point towards the ROI center."""
        return self.node.pull_point_to_roi_center(point, img_shape)
