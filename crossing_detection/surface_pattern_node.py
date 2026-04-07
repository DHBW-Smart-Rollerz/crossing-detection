import math
from enum import IntEnum

import cv2
import cv_bridge
import numpy as np
import rclpy
import sensor_msgs.msg
import std_msgs.msg
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from smarty_utils.enums import NodeState
from smarty_utils.smarty_node import SmartyNode
from timing import timer

from crossing_detection.agreggator import IntersectionAggregator
from crossing_detection.debug_visualizer import CrossingDebugVisualizer

# Color constants (RGB tuples - will be converted to BGR via cv2.COLOR_RGB2BGR)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)

# Bright colors
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
LIME = (0, 255, 0)
PINK = (255, 192, 203)
VIOLET = (148, 0, 211)
TURQUOISE = (200, 230, 240)
GOLD = (255, 215, 0)


class LaneType(IntEnum):
    """Declare types of lines."""

    EGO_SOLID = 20
    EGO_DOTTED = 21
    OPP_SOLID = 22
    OPP_DOTTED = 23
    RIGHT_SOLID = 24
    RIGHT_DOTTED = 25
    LEFT_SOLID = 26
    LEFT_DOTTED = 27


FILTERING_ROI_REL_RLTB = (0.75, 0.15, 0, 0.815)  # left, right, top, bottom


lsd = cv2.createLineSegmentDetector(1)


class SurfacePatternDetector(SmartyNode):
    """
    A ROS2 node for surface pattern detection.

    Arguments:
        SmartyNode -- Base class for ROS2 nodes.

    Returns:
        None
    """

    def __init__(self):
        """Initialize the ROS2ExampleNode."""
        super().__init__(
            "surface_pattern_node",
            "crossing_detection",
            node_parameters={
                # Subscriber topics
                "image_subscriber": "/camera/birds_eye",
                # Publisher topics
                "debug_image_publisher": "/crossing_detection/debug/image",
                "result_publisher": "/crossing_detection/result",
                # Parameters
                "state": NodeState.INACTIVE.value,
                "image_path": "resources/img/example.png",
                "debug": False,
                "compute_crossing_center": False,
                # Sharpening parameters
                "sharpen_enabled": True,
                "sharpen_strength_top": 1.5,
                "sharpen_strength_bottom": 1.0,
                # Distortion enhancement parameters
                "enhance_distorted_roi_enabled": True,
                "enhance_distortion_kernel": 5,
                "enhance_distortion_dilations": 1,
                # Gap detection debug visualization
                "debug_line_gap_detection": False,
                "debug_logging": False,
            },
            subscribed_topics={
                "image_subscriber": (
                    sensor_msgs.msg.Image,
                    self.image_callback,
                    1,
                ),
            },
            published_topics={
                "debug_image_publisher": (sensor_msgs.msg.Image, 1),
                "result_publisher": (std_msgs.msg.Float32MultiArray, 1),
            },
        )

        self.cv_bridge = cv_bridge.CvBridge()
        try:
            self.compute_crossing_center = self.get_parameter(
                "compute_crossing_center"
            ).value
        except Exception:
            self.compute_crossing_center = True

        try:
            self.sharpen_enabled = self.get_parameter("sharpen_enabled").value
            self.sharpen_strength_top = self.get_parameter("sharpen_strength_top").value
            self.sharpen_strength_bottom = self.get_parameter(
                "sharpen_strength_bottom"
            ).value
        except Exception:
            # fallback defaults
            self.sharpen_enabled = True
            self.sharpen_strength_top = 1.5
            self.sharpen_strength_bottom = 1.0

        try:
            self.enhance_distorted_roi_enabled = self.get_parameter(
                "enhance_distorted_roi_enabled"
            ).value
            self.enhance_distortion_kernel = self.get_parameter(
                "enhance_distortion_kernel"
            ).value
            self.enhance_distortion_dilations = self.get_parameter(
                "enhance_distortion_dilations"
            ).value
        except Exception:
            self.enhance_distorted_roi_enabled = True
            self.enhance_distortion_kernel = 5
            self.enhance_distortion_dilations = 1

        try:
            self.debug_line_gap_detection = self.get_parameter(
                "debug_line_gap_detection"
            ).value
        except Exception:
            self.debug_line_gap_detection = False

        try:
            self.debug_logging = self.get_parameter("debug_logging").value
        except Exception:
            self.debug_logging = False

        # Configure logger level based on debug_logging parameter
        logger = self.get_logger()
        if self.debug_logging:
            logger.set_level(rclpy.logging.LoggingSeverity.DEBUG)
        else:
            logger.set_level(rclpy.logging.LoggingSeverity.INFO)

        self.debug_visualizer = CrossingDebugVisualizer(node=self)

        self.detected_crossing_center = None
        self.active_crossing_center = None
        self.crossing_center_frames = 0
        self.crossing_center_error = float("inf")

        self.intersection_aggregator = IntersectionAggregator(max_frames=7)

    @timer.Timer(name="image_callback", filter_strength=40)
    def image_callback(self, msg: sensor_msgs.msg.Image):
        """Executed by the ROS2 system whenever a new image is received."""
        if self.get_parameter("state").value != NodeState.ACTIVE.value:
            return

        try:
            try:
                img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception:
                img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            img_dbg, result_list = self.pipeline(img)

            try:
                msg_out = std_msgs.msg.Float32MultiArray(
                    data=[float(x) for x in result_list]
                )
                self.result_publisher.publish(msg_out)
            except Exception as e:
                self.get_logger().error(f"publishing result failed: {e}")

            img_dbg = cv2.cvtColor(img_dbg, cv2.COLOR_RGB2BGR)
            output_img = self.cv_bridge.cv2_to_imgmsg(img_dbg, encoding="bgr8")
            self.debug_image_publisher.publish(output_img)

        except Exception as e:
            self.get_logger().error(f"image_callback error: {e}")

    def perform_canny(self, img):
        """
        Perform Canny edge detection on the image.

        Arguments:
            img -- Input image.

        Returns:
            Image with edges detected.
        """
        img = cv2.Canny(img, 50, 75)
        return img

    def _enhance_by_line_brightness(self, image, lines, percentile=80):
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
            nl = self._normalize_line(line)
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

    def get_roi_bbox(self, img_shape):
        """
        Get the bounding box of the region of interest (ROI) based on relative coordinates.

        Arguments:
            img_shape -- Shape of the image.

        Returns:
            Tuple of (x_start, x_end, y_start, y_end) defining the ROI bounding box.
        """
        height = img_shape[0]
        width = img_shape[1]

        roi_left = int(width * FILTERING_ROI_REL_RLTB[1])
        roi_right = int(width * FILTERING_ROI_REL_RLTB[0])
        roi_top = int(height * FILTERING_ROI_REL_RLTB[2])
        roi_bottom = int(height * FILTERING_ROI_REL_RLTB[3])

        return roi_left, roi_right, roi_top, roi_bottom

    def get_rotated_roi_bbox(self, img_shape, angle_deg):
        """
        Get a rotated bounding box for the ROI aligned with prominent angle.

        Arguments:
            img_shape -- Shape of the image (height, width).
            angle_deg -- Rotation angle in degrees (0-180).

        Returns:
            Tuple of (center_x, center_y, width, height, angle_rad)
            for use with cv2.rotatedRectangle operations.
        """
        if angle_deg is None:
            return None

        height = img_shape[0]
        width = img_shape[1]

        # Get standard ROI bounds
        roi_top = int(height * FILTERING_ROI_REL_RLTB[2])
        roi_bottom = int(height * FILTERING_ROI_REL_RLTB[3])
        roi_left = int(width * FILTERING_ROI_REL_RLTB[1])
        roi_right = int(width * FILTERING_ROI_REL_RLTB[0])

        # Calculate center of ROI
        center_x = (roi_left + roi_right) / 2.0
        center_y = (roi_top + roi_bottom) / 2.0

        # Calculate dimensions
        roi_width = roi_right - roi_left
        roi_height = roi_bottom - roi_top

        # Normalize angle to [0, 180) range
        angle_normalized = float(angle_deg) % 180.0

        # Return rotated rectangle parameters
        # Note: cv2 expects angle in [-90, 0] for rotatedRect
        # We convert to [-90, 0] range for proper rotation
        angle_for_cv2 = angle_normalized - 90.0

        return center_x, center_y, roi_width, roi_height, angle_for_cv2

    def filter_by_rotated_roi(self, lines, img_shape, angle_deg):
        """
        Filter lines based on a rotated ROI aligned with the prominent angle.

        Arguments:
            lines -- List of lines as pairs of points.
            img_shape -- Shape of the image (height, width).
            angle_deg -- Rotation angle in degrees for ROI alignment.

        Returns:
            Filtered list of lines within the rotated ROI.
        """
        if angle_deg is None:
            # Fall back to standard ROI filtering
            return self.filter_by_roi(lines, img_shape)

        rotated_roi = self.get_rotated_roi_bbox(img_shape, angle_deg)
        if rotated_roi is None:
            return lines

        center_x, center_y, roi_width, roi_height, angle_cv2 = rotated_roi

        # Convert angle to radians for point transformation
        angle_rad = math.radians(float(angle_deg))
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        res = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Transform both endpoints to rotated coordinate system
            # Translate to origin
            p1_x = x1 - center_x
            p1_y = y1 - center_y

            p2_x = x2 - center_x
            p2_y = y2 - center_y

            # Rotate by -angle_deg (reverse rotation for coordinate system)
            p1_rot_x = p1_x * cos_a + p1_y * sin_a
            p1_rot_y = -p1_x * sin_a + p1_y * cos_a

            p2_rot_x = p2_x * cos_a + p2_y * sin_a
            p2_rot_y = -p2_x * sin_a + p2_y * cos_a

            # Check if both endpoints are within rotated ROI
            # ROI is centered at origin in rotated space
            if (
                abs(p1_rot_x) <= roi_width / 2.0
                and abs(p1_rot_y) <= roi_height / 2.0
                and abs(p2_rot_x) <= roi_width / 2.0
                and abs(p2_rot_y) <= roi_height / 2.0
            ):
                res.append(line)

        return res

    def _draw_angle_arrow(self, image, angle_deg, arrow_length=25):
        """
        Draw an arrow in the top right corner showing the detected angle.

        Arguments:
            image -- Image to draw on
            angle_deg -- Angle in degrees (0-180, where 90 is vertical)
            arrow_length -- Length of the arrow in pixels
        """
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

    def filter_by_angle(
        self,
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
            tol_deg -- Tolerance for vertical/horizontal classification (degrees)
            debug -- Debug flag
            anchor_angle -- Optional anchor angle to tilt the coordinate system
            anchor_tolerance -- Tolerance around anchor (for vertical lines)

        Returns:
            Tuple of (vertical_lines, horizontal_lines)
        """
        vertical = []
        horizontal = []

        if lines is None or len(lines) == 0:
            return vertical, horizontal

        for idx, line in enumerate(lines):
            arr = line[0]
            if arr.size < 4:
                continue
            x1, y1, x2, y2 = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])

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

                if angle_diff <= anchor_tolerance:
                    vertical.append(line)
                elif perp_diff <= anchor_tolerance:
                    horizontal.append(line)
            else:
                dist_h = min(abs(angle_norm - 0.0), abs(angle_norm - 180.0))
                dist_v = min(abs(angle_norm - 80.0), abs(angle_norm - 100.0))

                if dist_v <= tol_deg + 10 and dist_v < dist_h:
                    vertical.append(line)
                elif dist_h <= tol_deg and dist_h < dist_v:
                    horizontal.append(line)

        return vertical, horizontal

    def filter_by_diagonal_angle(
        self,
        lines,
        prominent_angle,
        tol_deg: float = 20.0,
    ):
        """
        Filter lines based on diagonal angles relative to the prominent angle.

        Uses the prominent angle to define diagonal directions:
        - diagonal_1 = prominent_angle + 63°
        - diagonal_2 = prominent_angle + 117° (perpendicular diagonal)

        Arguments:
            lines -- List of lines as pairs of points.
            prominent_angle -- The prominent angle from line analysis
            tol_deg -- Tolerance for diagonal angle classification (degrees)

        Returns:
            List of lines with diagonal orientation (within tolerance)
        """
        diagonal_lines = []

        if lines is None or len(lines) == 0:
            return diagonal_lines

        if prominent_angle is None:
            return diagonal_lines

        # Calculate diagonal angles based on prominent angle
        diagonal_1 = (63.0 - prominent_angle) % 180.0
        diagonal_2 = (117.0 - prominent_angle) % 180.0

        for line in lines:
            arr = line[0]
            if arr.size < 4:
                continue

            x1, y1, x2, y2 = (
                float(arr[0]),
                float(arr[1]),
                float(arr[2]),
                float(arr[3]),
            )

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0 and dy == 0:
                continue

            angle = math.degrees(math.atan2(dy, dx))
            # Normalize angle to 0-180 range
            angle_norm = (angle + 360.0) % 180.0

            # Check distance to first diagonal
            dist_to_diag1 = min(
                abs(angle_norm - diagonal_1),
                abs(angle_norm - (diagonal_1 + 180.0) % 180.0),
            )

            # Check distance to second diagonal
            dist_to_diag2 = min(
                abs(angle_norm - diagonal_2),
                abs(angle_norm - (diagonal_2 + 180.0) % 180.0),
            )

            # Accept if close to either diagonal
            min_dist = min(dist_to_diag1, dist_to_diag2)

            if min_dist <= tol_deg:
                diagonal_lines.append(line)

        return diagonal_lines

    def filter_by_roi(self, lines, img_shape):
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

    def filter_by_length(self, lines, min_length: float = 70.0, max_length=10000):
        """
        Filter lines based on their length.

        Arguments:
            lines -- List of lines as pairs of points.
            min_length -- Minimum length of the line to be kept.

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

    def line_segment_detector(self, img):
        """
        Detect line segments in the image using LSD.

        Arguments:
            img -- Input image.

        Returns:
            List of detected lines as pairs of points.
        """
        lines = lsd.detect(img)[0]

        return lines

    def _normalize_line(self, line):
        """
        Normalize a single line representation into a numpy array of.
        shape (1, 4). Accepts formats: ndarray (1,4) or (4,), nested
        lists [[x1,y1,x2,y2]] or tuples. Returns None for malformed
        entries.
        """
        if line is None:
            return None

        if isinstance(line, np.ndarray):
            arr = line.squeeze()
            if arr.ndim == 1 and arr.size >= 4:
                return arr.reshape(1, -1)[:, :4].astype(np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 4:
                return arr.reshape(1, -1)[:, :4].astype(np.float32)
            return None

        if isinstance(line, (list, tuple)):
            s = line
            while len(s) == 1 and isinstance(s[0], (list, tuple, np.ndarray)):
                s = s[0]
            try:
                flat = np.array(s).reshape(-1)
                if flat.size >= 4:
                    return flat[:4].astype(np.float32).reshape(1, 4)
            except Exception:
                return None

        return None

    def _normalize_lines(self, lines):
        """Normalize an iterable of lines to a list of numpy (1,4) arrays."""
        if lines is None or (hasattr(lines, "__len__") and len(lines) == 0):
            return []
        normalized = []
        for ln in lines:
            nl = self._normalize_line(ln)
            if nl is not None:
                normalized.append(nl)
        return normalized

    def fuse_similar_lines(
        self,
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

        Returns a list of numpy arrays, each shape (1,4) (float32).
        """
        normalized = self._normalize_lines(lines)
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

            mean = pts.mean(axis=0)
            U, S, Vt = np.linalg.svd(pts - mean)
            axis = Vt[0]

            scalars = (pts - mean).dot(axis)
            min_s = scalars.min()
            max_s = scalars.max()
            p1 = mean + axis * min_s
            p2 = mean + axis * max_s

            fused.append(np.array([[p1[0], p1[1], p2[0], p2[1]]], dtype=np.float32))

        return fused

    def find_heading_angle(self, lines):
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
        if lines is None or len(lines) == 0:
            self.get_logger().debug("No lines provided to histogram")
            return None, 0

        valid_angles = []
        lines_filtered_by_angle = 0

        for line in lines:
            try:
                nl = self._normalize_line(line)
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

        self.get_logger().debug(f"Total lines: {len(lines)}")
        self.get_logger().debug(
            f"Lines filtered by angle (>45 deg): {lines_filtered_by_angle}"
        )
        self.get_logger().debug(f"Valid angles for histogram: {len(valid_angles)}")

        if len(valid_angles) == 0:
            return None, 0

        valid_angles_arr = np.array(valid_angles)
        hist, bin_edges = np.histogram(valid_angles_arr, bins=36, range=(0, 180))

        peak_bin = int(np.argmax(hist))
        prominent_angle = float((bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0)

        count_str = len(valid_angles)
        msg = f"Prominent angle: {prominent_angle:.2f} deg ({count_str} lines)"
        self.get_logger().debug(msg)

        return prominent_angle, len(valid_angles)

    def preprocess_image(self, image):
        """
        Preprocess image with filtering, morphological operations, and cleanup.

        This method applies a series of image processing steps to prepare
        the image for line detection and crossing identification.

        Arguments:
            image -- Input image (BGR or grayscale)

        Returns:
            Tuple of:
                - image: Processed image after all preprocessing
                - enhanced_image: Brightness-enhanced image (if created)
        """
        image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        image = cv2.medianBlur(image, 7)

        image = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8), iterations=1
        )
        image = cv2.morphologyEx(
            image, cv2.MORPH_CLOSE, kernel=np.ones((8, 8), np.uint8), iterations=1
        )
        image = cv2.dilate(image, kernel=np.ones((4, 4), np.uint8), iterations=2)

        edges = self.perform_canny(image)
        transformed_lines = self.line_segment_detector(edges)
        transformed_lines = self._normalize_lines(transformed_lines)

        filtered_lines = self.filter_by_length(transformed_lines, min_length=20)
        filtered_lines = self.filter_by_roi(filtered_lines, image.shape)

        if filtered_lines and len(filtered_lines) > 0:
            image = self._enhance_by_line_brightness(
                image, filtered_lines, percentile=90
            )

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.uint8(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(30, 30))
        image = clahe.apply(image)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            image, connectivity=8
        )
        output_image = np.zeros_like(image)
        for label in range(1, num_labels):  # Skip background (label 0)
            if stats[label, cv2.CC_STAT_AREA] >= 60:
                output_image[labels == label] = image[labels == label]
        image = output_image

        return image

    def calculate_cluster_boxes(self, lines, line_clusters):
        """
        Calculate bounding boxes for clusters.

        Arguments:
            lines -- List of lines
            line_clusters -- Cluster labels from DBSCAN

        Returns:
            Dictionary with cluster_id as key and box (np.array)
        """
        cluster_boxes = {}

        if line_clusters is None:
            return cluster_boxes

        unique_clusters = set(line_clusters)
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue  # skip noise

            cluster_lines = [
                line for line, lbl in zip(lines, line_clusters) if lbl == cluster_id
            ]

            if len(cluster_lines) == 0:
                continue

            # Collect all points
            points = []
            for line in cluster_lines:
                x1, y1, x2, y2 = line[0]
                points.append((x1, y1))
                points.append((x2, y2))

            points = np.array(points, dtype=np.int32)

            # Calculate bounding box
            if len(points) >= 4:
                try:
                    rect = cv2.minAreaRect(points)
                    box = cv2.boxPoints(rect)
                    box = np.array(box, dtype=np.int32)
                    cluster_boxes[cluster_id] = box
                except Exception as e:
                    self.get_logger().debug(f"Could not fit rectangle: {e}")

        return cluster_boxes

    def draw_cluster_boxes(self, image, boxes):
        """
        Draw bounding boxes for clusters on image.

        Arguments:
            image -- Image to draw on
            lines -- List of lines
            line_clusters -- Cluster labels from DBSCAN
        """
        # Draw boxes
        for box in boxes:
            cv2.polylines(image, [box], True, (255, 0, 255), 2)

    def filter_pca(self, lines, line_clusters, min_variance_ratio: float = 0.85):
        """
        Filter clusters using PCA on their center points.

        Keeps only clusters where the center points lie on a line
        (high variance in 1D after PCA).

        Arguments:
            lines -- List of line segments
            line_clusters -- Cluster labels from DBSCAN
            min_variance_ratio -- Minimum explained variance to
                                 consider points as collinear

        Returns:
            List of valid cluster IDs
        """
        valid_clusters = []

        if line_clusters is None:
            return valid_clusters

        unique_clusters = set(line_clusters)

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue  # skip noise

            cluster_lines = [
                line for line, lbl in zip(lines, line_clusters) if lbl == cluster_id
            ]

            if len(cluster_lines) < 3:
                continue

            # Extract center points from lines
            center_points = []
            for line in cluster_lines:
                x1, y1, x2, y2 = line[0]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                center_points.append([cx, cy])

            # PCA on center points
            try:
                center_points_arr = np.array(center_points, dtype=np.float32)

                # Only do PCA if we have enough samples
                if len(center_points_arr) >= 2:
                    pca = PCA(n_components=1)
                    pca.fit(center_points_arr)
                    explained_var = pca.explained_variance_ratio_[0]

                    # If most variance in 1 component,
                    # points are on a line
                    if explained_var >= min_variance_ratio:
                        valid_clusters.append(cluster_id)

                        self.get_logger().debug(
                            f"Cluster {cluster_id}: " f"variance={explained_var:.4f}"
                        )
            except Exception as e:
                self.get_logger().debug(f"PCA failed for cluster {cluster_id}: {e}")

        return valid_clusters

    def build_dbscan_features(self, lines):
        """
        Build feature vectors for DBSCAN clustering.

        Arguments:
            lines -- List of line segments

        Returns:
            Numpy array of shape (n_samples, n_features)
        """
        features = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Convert to Hesse normal form (rho, theta)
            # rho: distance from origin to line
            # theta: angle of normal to line
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            dx = x2 - x1
            dy = y2 - y1

            # Line direction angle
            theta = math.atan2(dy, dx)

            # Normalize theta to [0, 2π)
            if theta < 0:
                theta += math.pi

            # Distance from origin (0,0) to line
            # Using point-to-line distance formula
            line_length = math.sqrt(dx**2 + dy**2)
            if line_length > 0:
                # Normal vector
                nx = -dy / line_length
                ny = dx / line_length
                # Distance from origin
                rho = abs(x1 * nx + y1 * ny)
            else:
                rho = 0

            features.append([rho, theta, center_x, center_y])

        features = StandardScaler().fit_transform(features)
        return np.array(features)

    def pipeline(self, image):
        """
        Complete processing pipeline for intersection detection.

        Arguments:
            img_path -- Path to the input image.
        """
        orig_image = image.copy()

        image = self.preprocess_image(image)

        lines = self.line_segment_detector(image)

        lines = self._normalize_lines(lines)
        lines = self.filter_by_length(lines, min_length=30, max_length=110)

        prominent_angle, _ = self.find_heading_angle(lines)

        self._draw_angle_arrow(orig_image, prominent_angle)

        xs, xe, ys, ye = self.get_roi_bbox(image.shape)
        cv2.rectangle(
            orig_image,
            (int(xs), int(ys)),
            (int(xe), int(ye)),
            (255, 0, 100),  # Dark gray
            2,
        )

        lines = self.filter_by_roi(lines, image.shape)

        lines = self.fuse_similar_lines(lines, angle_tol_deg=8, center_dist_tol=20)

        # filter angle for 45 and 135 deg
        lines = (
            self.filter_by_diagonal_angle(lines, abs(90 - prominent_angle), tol_deg=18)
            if prominent_angle is not None
            else []
        )

        lines = self.fuse_similar_lines(lines, angle_tol_deg=8, center_dist_tol=40)

        if lines is not None and len(lines) > 0:
            features = self.build_dbscan_features(lines)
            db = DBSCAN(eps=0.9, min_samples=3).fit(features)
            line_clusters = db.labels_

            # Filter clusters using PCA
            valid_clusters = self.filter_pca(
                lines, line_clusters, min_variance_ratio=0.98
            )

        else:
            line_clusters = None
            valid_clusters = []

        filtered_clusters = None
        cluster_boxes = None
        valid_boxes = []

        if line_clusters is not None and len(valid_clusters) > 0:
            filtered_clusters = np.full_like(line_clusters, -1, dtype=int)
            for i, lbl in enumerate(line_clusters):
                if lbl in valid_clusters:
                    filtered_clusters[i] = lbl

        if filtered_clusters is not None:
            cluster_boxes = self.calculate_cluster_boxes(lines, filtered_clusters)
            # get the width of the boxes
            valid_boxes = []
            for box in cluster_boxes.values():
                if box is not None and len(box) == 4:
                    edge1 = np.linalg.norm(box[0] - box[1])
                    edge2 = np.linalg.norm(box[1] - box[2])

                    if min(edge1, edge2) > 30 and min(edge1, edge2) < 120:
                        valid_boxes.append(box)

        if len(valid_boxes) > 0:
            self.draw_cluster_boxes(orig_image, valid_boxes)

        return orig_image, []


def main(args=None):
    """
    Main function to start the crossing detection node.

    Keyword Arguments:
        args -- Launch arguments (default: {None})
    """
    rclpy.init(args=args)
    node = SurfacePatternDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
