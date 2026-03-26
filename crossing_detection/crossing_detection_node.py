import math
from enum import IntEnum

import cv2
import cv_bridge
import numpy as np
import rclpy
import sensor_msgs.msg
import std_msgs.msg
from sklearn.cluster import DBSCAN
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


FILTERING_ROI_REL_RLTB = (0.80, 0, 0, 0.815)  # left, right, top, bottom


lsd = cv2.createLineSegmentDetector(1)


class IntersectionDetector(SmartyNode):
    """
    A ROS2 node for crossing detection.

    Arguments:
        SmartyNode -- Base class for ROS2 nodes.

    Returns:
        None
    """

    def __init__(self):
        """Initialize the ROS2ExampleNode."""
        super().__init__(
            "crossing_detection_node",
            "crossing_detection",
            node_parameters={
                # Subscriber topics
                "image_subscriber": "/camera/birds_eye",
                # Publisher topics
                "debug_image_publisher": "/crossing_detection/debug/image",
                "result_publisher": "/crossing_detection/result",
                # Parameters
                "state": NodeState.ACTIVE.value,
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

    def _sharpen_roi(self, image, strength: float = 1.5, do_roi_top: bool = True):
        """
        Apply sharpening to the ROI to enhance lines after blurring.

        Uses an unsharp mask technique: subtract a blurred copy from
        original to create high-pass filter effect.

        Arguments:
            image -- BGR image (numpy array)
            strength -- Sharpening intensity (1.0 = subtle, 2.0+ = strong)
            do_roi_top -- If True, sharpen top ROI; if False, sharpen
                          bottom ROI

        Returns:
            A copy of the image with ROI sharpened.
        """
        if image is None:
            return image

        h, w = image.shape[:2]
        roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)

        roi_left = max(0, min(w - 1, int(roi_left)))
        roi_right = max(0, min(w, int(roi_right)))
        roi_top = max(0, min(h - 1, int(roi_top)))
        roi_bottom = max(0, min(h, int(roi_bottom)))

        if do_roi_top:
            top_half_end = roi_top + roi_bottom
        else:
            top_half_end = roi_top + (roi_bottom - roi_top) // 2

        top = max(0, roi_top)
        bottom = max(top, min(h, top_half_end if do_roi_top else roi_bottom))
        left = max(0, roi_left)
        right = max(0, min(w, roi_right))

        out = image.copy()
        if bottom > top and right > left:
            patch = out[top:bottom, left:right].copy()

            blurred = cv2.GaussianBlur(patch, (5, 5), 0)

            sharpened = cv2.addWeighted(patch, 1.0 + strength, blurred, -strength, 0)

            sharpened = np.clip(sharpened, 0, 255).astype(patch.dtype)
            out[top:bottom, left:right] = sharpened

        return out

    def _enhance_distorted_roi(
        self,
        image,
        do_roi_top: bool = True,
        morph_kernel_size: int = 5,
        dilation_iterations: int = 1,
    ):
        """
        Enhance edges in distorted ROI areas (especially bird's eye top).

        Applies morphological operations (closing + dilation) to connect
        broken/distorted lines in the top ROI region where perspective
        distortion creates fragmented line segments.

        Arguments:
            image -- BGR image (numpy array)
            do_roi_top -- If True, enhance top ROI; False for bottom
            morph_kernel_size -- Kernel size for morphological ops (odd)
            dilation_iterations -- Number of dilation passes (1-3)

        Returns:
            Image with enhanced distorted areas.
        """
        if image is None:
            return image

        h, w = image.shape[:2]
        roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)

        roi_left = max(0, min(w - 1, int(roi_left)))
        roi_right = max(0, min(w, int(roi_right)))
        roi_top = max(0, min(h - 1, int(roi_top)))
        roi_bottom = max(0, min(h, int(roi_bottom)))

        if do_roi_top:
            top_half_end = roi_top + roi_bottom
        else:
            top_half_end = roi_top + (roi_bottom - roi_top) // 2

        top = max(0, roi_top)
        bottom = max(top, min(h, top_half_end if do_roi_top else roi_bottom))
        left = max(0, roi_left)
        right = max(0, min(w, roi_right))

        out = image.copy()
        if bottom > top and right > left:
            patch = out[top:bottom, left:right].copy()

            if morph_kernel_size % 2 == 0:
                morph_kernel_size += 1

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
            )

            closed = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, kernel, iterations=1)

            # 2. Dilation: strengthen and connect broken lines
            dilated = cv2.dilate(closed, kernel, iterations=dilation_iterations)

            out[top:bottom, left:right] = dilated

        return out

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

    def _draw_lines(self, img, lines, color=(255, 0, 0), thickness=2):
        """
        Show lines on the image.

        Arguments:
            img -- Input image.
            lines -- List of lines as pairs of points.
        """
        img2 = img[::]
        if lines is None or (hasattr(lines, "__len__") and len(lines) == 0):
            return img2
        for line in lines:
            nl = self._normalize_line(line)
            if nl is None:
                continue
            x1, y1, x2, y2 = nl[0]
            cv2.line(img2, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        return img2

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

    def fuse_nearby_lines(
        self, lines, distance_threshold: float = 20.0, angle_tolerance: float = 10.0
    ):
        """
        Fuse lines that are geometrically close to each other.

        Groups lines that are near each other and have similar angles,
        then merges them into single lines. This helps join segments
        belonging to the same stop line.

        Arguments:
            lines -- List of lines as [[x1, y1, x2, y2]]
            distance_threshold -- Max distance between line endpoints to fuse
            angle_tolerance -- Max angle difference (degrees) to fuse

        Returns:
            List of fused lines
        """
        if lines is None or len(lines) == 0:
            return []

        line_data = []
        for line in lines:
            nl = self._normalize_line(line)
            if nl is None:
                continue
            x1, y1, x2, y2 = nl[0].astype(float)
            dx = x2 - x1
            dy = y2 - y1
            line_len = math.hypot(dx, dy)
            if line_len < 1e-3:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            angle_norm = (angle + 360.0) % 180.0
            line_data.append(
                {
                    "line": line,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "angle": angle_norm,
                    "length": line_len,
                    "used": False,
                }
            )

        if not line_data:
            return []

        fused_lines = []

        for i, line_i in enumerate(line_data):
            if line_i["used"]:
                continue

            cluster = [line_i]
            line_i["used"] = True

            for j in range(i + 1, len(line_data)):
                line_j = line_data[j]
                if line_j["used"]:
                    continue

                angle_diff = abs(line_i["angle"] - line_j["angle"])
                angle_diff = min(angle_diff, 180.0 - angle_diff)
                if angle_diff > angle_tolerance:
                    continue

                is_close = False
                for line_c in cluster:
                    d_j1_to_i1 = math.hypot(
                        line_j["x1"] - line_c["x1"],
                        line_j["y1"] - line_c["y1"],
                    )
                    d_j1_to_i2 = math.hypot(
                        line_j["x1"] - line_c["x2"],
                        line_j["y1"] - line_c["y2"],
                    )
                    d_j2_to_i1 = math.hypot(
                        line_j["x2"] - line_c["x1"],
                        line_j["y2"] - line_c["y1"],
                    )
                    d_j2_to_i2 = math.hypot(
                        line_j["x2"] - line_c["x2"],
                        line_j["y2"] - line_c["y2"],
                    )
                    min_dist = min(d_j1_to_i1, d_j1_to_i2, d_j2_to_i1, d_j2_to_i2)
                    if min_dist <= distance_threshold:
                        is_close = True
                        break

                if is_close:
                    cluster.append(line_j)
                    line_j["used"] = True

            if cluster:
                all_x = []
                all_y = []
                for line_c in cluster:
                    all_x.extend([line_c["x1"], line_c["x2"]])
                    all_y.extend([line_c["y1"], line_c["y2"]])

                angle = cluster[0]["angle"]
                angle_rad = math.radians(angle)
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)

                projections = []
                for x, y in zip(all_x, all_y):
                    proj = x * cos_a + y * sin_a
                    projections.append((proj, x, y))

                projections.sort()
                min_proj = projections[0]
                max_proj = projections[-1]

                x1_fused = min_proj[1]
                y1_fused = min_proj[2]
                x2_fused = max_proj[1]
                y2_fused = max_proj[2]

                fused_line = np.array(
                    [[[x1_fused, y1_fused, x2_fused, y2_fused]]],
                    dtype=np.float32,
                )
                fused_lines.append(fused_line)

        return fused_lines

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

    def _is_line_in_region(self, line, region):
        """
        Check if a line (or its endpoints) overlaps with a region.

        Arguments:
            line -- Line as (1,4) numpy array: [x1, y1, x2, y2]
            region -- Tuple (x1, y1, x2, y2) defining the region bounds

        Returns:
            True if line has any point in the region, False otherwise
        """
        try:
            nl = self._normalize_line(line)
            if nl is None:
                return False

            x1, y1, x2, y2 = nl[0]
            region_x1, region_y1, region_x2, region_y2 = region

            p1_in = region_x1 <= x1 <= region_x2 and region_y1 <= y1 <= region_y2
            p2_in = region_x1 <= x2 <= region_x2 and region_y1 <= y2 <= region_y2

            if p1_in or p2_in:
                return True

            line_x_min = min(x1, x2)
            line_x_max = max(x1, x2)
            line_y_min = min(y1, y2)
            line_y_max = max(y1, y2)

            overlap = (
                line_x_min <= region_x2
                and line_x_max >= region_x1
                and line_y_min <= region_y2
                and line_y_max >= region_y1
            )

            return overlap

        except Exception:
            return False

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

    def is_line_solid(
        self,
        line,
        image,
        step: float = 0.05,
        sample_width: int = 7,
        sample_height: int = 15,
        white_pixel_thresh: int = 200,
        white_patch_ratio: float = 0.75,
        below_check_thresh: float = 0.10,
    ):
        """
        Determine whether a line is solid (durchgezogen) or dotted (unterbrochen).

        Procedure:
        - Sample points along the line at intervals of `step` (fraction of length).
        - For each sample point, extract a small rectangular patch located just
          above the point (towards image top) of size (sample_height x sample_width).
        - Consider a patch "white" if the fraction of pixels > white_pixel_thresh
          is >= white_patch_ratio.
        - If >= 85% (white_patch_ratio) of the samples are white, classify the
          whole line as solid.

        Returns (is_solid: bool, white_fraction: float, n_samples: int)
        """
        if line is None or image is None:
            return False, 0.0, 0

        nl = self._normalize_line(line)
        if nl is None:
            return False, 0.0, 0

        x1, y1, x2, y2 = nl[0].astype(float)

        gray = image
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        t_values = np.arange(0.0, 1.0 + 1e-6, step)
        valid_samples = 0
        white_samples = 0

        def _sample_patch(cx, cy, direction="above"):
            if direction == "above":
                top = cy - sample_height
                bottom = cy
            else:
                top = cy
                bottom = cy + sample_height

            left = cx - sample_width // 2
            right = cx + (sample_width - sample_width // 2)

            h, w = gray.shape[:2]
            top = max(0, top)
            bottom = min(h, bottom)
            left = max(0, left)
            right = min(w, right)

            if bottom <= top or right <= left:
                return None

            patch = gray[top:bottom, left:right]
            if patch.size == 0:
                return None

            try:
                _, binary_patch = cv2.threshold(
                    patch, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                )
                return float(np.mean(binary_patch))
            except Exception:
                thresh = np.percentile(patch, 85)
                return float(np.mean(patch > thresh))

        for t in t_values:
            px = (1.0 - t) * x1 + t * x2
            py = (1.0 - t) * y1 + t * y2

            cx = int(round(px))
            cy = int(round(py))

            white_ratio = _sample_patch(cx, cy, direction="above")
            if white_ratio is None:
                continue

            valid_samples += 1
            if white_ratio >= white_patch_ratio:
                white_samples += 1

        if valid_samples == 0:
            return False, 0.0, 0

        frac_above = float(white_samples) / float(valid_samples)

        if frac_above >= white_patch_ratio:
            return True, frac_above, valid_samples

        frac = frac_above
        if frac_above < below_check_thresh:
            valid_samples_b = 0
            white_samples_b = 0
            for t in t_values:
                px = (1.0 - t) * x1 + t * x2
                py = (1.0 - t) * y1 + t * y2
                cx = int(round(px))
                cy = int(round(py))

                white_ratio_b = _sample_patch(cx, cy, direction="below")
                if white_ratio_b is None:
                    continue
                valid_samples_b += 1
                if white_ratio_b >= white_patch_ratio:
                    white_samples_b += 1

            if valid_samples_b > 0:
                frac_below = float(white_samples_b) / float(valid_samples_b)
                frac = max(frac_above, frac_below)
            else:
                frac_below = 0.0

        is_solid = frac >= white_patch_ratio
        return bool(is_solid), frac, valid_samples

    def check_line_by_horizontal_extension(self, line, image, direction="right"):
        """
        Verify a line is a real lane line (not misclassified ego line).

        Extends the line by 40px padding + 50px test length in the specified
        direction and checks if it remains continuous (solid). Real lane lines
        fade/break when extended; misclassified lines stay continuous.

        Arguments:
            line -- Line to check (numpy array shape (1,4))
            image -- Image to test on
            direction -- "right" or "left" (extends rightmost or leftmost endpoint)

        Returns:
            Tuple of (is_valid, extended_line)
            - is_valid: False if line is valid, True if should be rejected
            - extended_line: The extended line array for visualization
        """
        if line is None:
            return False

        try:
            x1, y1, x2, y2 = line[0]

            m = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0

            if direction == "right":
                line_start = (x1, y1) if x1 > x2 else (x2, y2)

                x_padded = line_start[0] + 40.0
                y_padded = line_start[1] + m * 40.0

                x2_extended = x_padded + 50.0
                y2_extended = y_padded + m * 50.0

            else:  # direction == "left"
                line_start = (x1, y1) if x1 < x2 else (x2, y2)

                x_padded = line_start[0] - 40.0
                y_padded = line_start[1] - m * 40.0

                x2_extended = x_padded - 50.0
                y2_extended = y_padded - m * 50.0

            extended_line = np.array(
                [
                    [
                        x_padded,
                        y_padded,
                        x2_extended,
                        y2_extended,
                    ]
                ],
                dtype=np.float32,
            )

            (
                _,
                gaps_count,
                wr_extended,
                _,
            ) = self.is_line_dotted_by_gap_detection(
                extended_line,
                image,
                box_half_width=22,
                length_extend=1.1,
                min_gap_count=3,
            )

            self.get_logger().debug(
                f"EXTENSION TEST ({direction}): gaps={gaps_count} "
                f"wr={wr_extended:.1f}%"
            )

            if gaps_count == 0 and wr_extended > 20.0:
                self.get_logger().debug(
                    f"REJECTING: extended line ({direction}) is "
                    f"continuous (wr={wr_extended:.1f}% > 20%)"
                )
                return True, extended_line

            return False, extended_line

        except Exception as e:
            self.get_logger().error(f"Error in check_line_by_horizontal_extension: {e}")
            return False, None

    def check_line_by_vertical_extension(self, line, image, is_right=True):
        """
        Verify a stop line by extending from its endpoint.

        For right stop line: extends from the lowest point (max y) downward.
        For left stop line: extends from the highest point (min y) upward.

        Arguments:
            line -- Stop line to check (numpy array shape (1,4))
            image -- Image to test on
            is_right -- True for right stop line, False for left stop line

        Returns:
            Tuple of (gaps_count, wr_extended, extended_line)
            - gaps_count: Number of gaps detected
            - wr_extended: White ratio of extended line
            - extended_line: The extended line array for visualization
        """
        if line is None:
            return None, None

        try:
            x1, y1, x2, y2 = line[0]

            if is_right:
                if y1 > y2:
                    x_start, y_start = x1, y1
                    x_other, y_other = x2, y2
                else:
                    x_start, y_start = x2, y2
                    x_other, y_other = x1, y1

                if (y_start - y_other) != 0:
                    slope = (x_start - x_other) / (y_start - y_other)
                else:
                    slope = 0.0

                x_padded = x_start + slope * 40.0
                y_padded = y_start + 40.0

                x2_extended = x_padded + slope * 50.0
                y2_extended = y_padded + 50.0

            else:
                if y1 < y2:
                    x_start, y_start = x1, y1
                    x_other, y_other = x2, y2
                else:
                    x_start, y_start = x2, y2
                    x_other, y_other = x1, y1

                if (y_start - y_other) != 0:
                    slope = (x_start - x_other) / (y_start - y_other)
                else:
                    slope = 0.0

                x_padded = x_start - slope * 40.0
                y_padded = y_start - 40.0

                x2_extended = x_padded - slope * 50.0
                y2_extended = y_padded - 50.0

            extended_line = np.array(
                [
                    [
                        x_padded,
                        y_padded,
                        x2_extended,
                        y2_extended,
                    ]
                ],
                dtype=np.float32,
            )

            self.get_logger().debug(
                f"[DEBUG] Stop line {'RIGHT' if is_right else 'LEFT'}: "
                f"start=({x_start:.1f}, {y_start:.1f}), "
                f"padded=({x_padded:.1f}, {y_padded:.1f}), "
                f"extended=({x2_extended:.1f}, {y2_extended:.1f})"
            )

            (
                _,
                gaps_count,
                wr_extended,
                _,
            ) = self.is_line_dotted_by_gap_detection(
                extended_line,
                image,
                box_half_width=22,
                length_extend=1.1,
                min_gap_count=3,
            )

            stop_type = "RIGHT" if is_right else "LEFT"
            self.get_logger().debug(
                f"STOP LINE {stop_type} EXTENSION: gaps={gaps_count} "
                f"wr={wr_extended:.1f}%"
            )

            return gaps_count, wr_extended, extended_line

        except Exception as e:
            self.get_logger().error(f"Error in check_line_by_vertical_extension: {e}")
            return None, None, None

    def is_line_dotted_by_gap_detection(
        self,
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

        nl = self._normalize_line(line)
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

        if self.debug_line_gap_detection:
            debug_image = image.copy()
            cv2.line(
                debug_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                3,
            )
            if not hasattr(self, "debug_overlay_images"):
                self.debug_overlay_images = []
            self.debug_visualizer.debug_overlay_images.append(debug_image)

        crop_w = int(max(10, line_len * float(length_extend))) + int(box_half_width * 2)
        crop_h = int(max(3, box_half_width * 2))
        h, w = image.shape[:2]

        M = cv2.getRotationMatrix2D((mid_x, mid_y), angle, 1.0)
        warped = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

        cx = M[0, 0] * mid_x + M[0, 1] * mid_y + M[0, 2]
        cy = M[1, 0] * mid_x + M[1, 1] * mid_y + M[1, 2]

        if self.debug_line_gap_detection and abs(angle) > 10:
            self.get_logger().info(
                f"ROTATION DEBUG: angle={angle:.1f}° "
                f"mid=({mid_x:.1f},{mid_y:.1f}) "
                f"rotated_mid=({cx:.1f},{cy:.1f})"
            )

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

            if self.debug_line_gap_detection:
                vis_crop = (
                    crop.copy()
                    if crop.ndim == 3
                    else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                )
                vis_h, vis_w = vis_crop.shape[:2]

                x1_warped = M[0, 0] * x1 + M[0, 1] * y1 + M[0, 2]
                y1_warped = M[1, 0] * x1 + M[1, 1] * y1 + M[1, 2]
                x2_warped = M[0, 0] * x2 + M[0, 1] * y2 + M[0, 2]
                y2_warped = M[1, 0] * x2 + M[1, 1] * y2 + M[1, 2]

                line_x1_crop = int(x1_warped - x1c)
                line_y1_crop = int(y1_warped - y1c)
                line_x2_crop = int(x2_warped - x1c)
                line_y2_crop = int(y2_warped - y1c)

                line_x1_crop = max(0, min(vis_w - 1, line_x1_crop))
                line_y1_crop = max(0, min(vis_h - 1, line_y1_crop))
                line_x2_crop = max(0, min(vis_w - 1, line_x2_crop))
                line_y2_crop = max(0, min(vis_h - 1, line_y2_crop))

                cv2.line(
                    vis_crop,
                    (line_x1_crop, line_y1_crop),
                    (line_x2_crop, line_y2_crop),
                    (0, 0, 255),
                    2,
                )

                cv2.line(
                    vis_crop,
                    (0, vis_h // 2),
                    (vis_w, vis_h // 2),
                    (0, 255, 0),
                    1,
                )

                binary_vis = cv2.cvtColor(
                    (binary * 255).astype(np.uint8),
                    cv2.COLOR_GRAY2BGR,
                )

                profile_h = 50
                profile_w = len(white_per_col)
                profile_vis = np.zeros((profile_h, profile_w, 3), dtype=np.uint8)
                max_white = np.max(white_per_col) if np.max(white_per_col) > 0 else 1
                for i, count in enumerate(white_per_col):
                    h_bar = int((count / max_white) * profile_h)
                    if h_bar > 0:
                        profile_vis[profile_h - h_bar :, i] = [0, 255, 0]

                vis_stack = np.vstack([vis_crop, binary_vis, profile_vis])

                if not hasattr(self, "debug_overlay_images"):
                    self.debug_overlay_images = []
                self.debug_visualizer.debug_overlay_images.append(vis_stack)

            return bool(is_dotted), int(gaps), float(white_ratio), 1
        except Exception as e:
            self.get_logger().error(f"Error in is_line_dotted_by_gap_detection: {e}")
            return False, 0, 0.0, 0

    def is_line_dotted_ensemble(
        self,
        line,
        image,
        box_half_width: int = 22,
        length_extend: float = 1.2,
        voting_threshold: int = 2,
    ):
        """
        Simple wrapper around gap detection for dotted/solid detection.

        Returns (is_dotted: bool, gap_count: int, white_ratio: float, _=1)
        """
        try:
            is_dotted, gap_count, white_ratio, _ = self.is_line_dotted_by_gap_detection(
                line, image, box_half_width, length_extend
            )
            return bool(is_dotted), int(gap_count), float(white_ratio), 1
        except Exception:
            return False, 0, 0.0, 0

    def elongate_line(self, line, length: float = 450):
        """
        Elongate the given line to the specified length.

        Arguments:
            line -- Line as a pair of points.

        Keyword Arguments:
            length --  (default: {300})

        Returns:
            Elongated line as a pair of points.
        """
        aim_length = length

        x1, y1, x2, y2 = line[0]
        delta_x = x2 - x1
        delta_y = y2 - y1
        length = math.sqrt(delta_x**2 + delta_y**2)
        factor = aim_length / length if length != 0 else 0
        new_delta_x = delta_x * factor
        new_delta_y = delta_y * factor
        line_center_x = (x1 + x2) / 2
        line_center_y = (y1 + y2) / 2

        new_x1 = int(line_center_x - new_delta_x / 2)
        new_y1 = int(line_center_y - new_delta_y / 2)
        new_x2 = int(line_center_x + new_delta_x / 2)
        new_y2 = int(line_center_y + new_delta_y / 2)

        return [[new_x1, new_y1, new_x2, new_y2]]

    def is_line_within_front_roi(
        self, line, image, min_rel: float = 0.5, max_rel: float = 0.75
    ):
        """Return True if the line center lies within the horizontal ROI fraction [min_rel,max_rel]."""
        if line is None or image is None:
            return False
        nl = self._normalize_line(line)
        if nl is None:
            return False
        x1, y1, x2, y2 = nl[0].astype(float)
        roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)
        roi_w = float(roi_right - roi_left)
        min_x = roi_left + roi_w * float(min_rel)
        max_x = roi_left + roi_w * float(max_rel)
        cx = (x1 + x2) / 2.0
        return (cx >= min_x) and (cx <= max_x)

    def clip_line_to_vertical_bounds(
        self, line, image, min_rel: float = 0.5, max_rel: float = 0.75
    ):
        """
        Clip a line segment to vertical boundaries defined by ROI fractions.

        Returns a normalized numpy line [[x1,y1,x2,y2]] (float32) or None if
        the segment lies completely outside the vertical band.
        """
        if line is None or image is None:
            return None
        nl = self._normalize_line(line)
        if nl is None:
            return None
        x1, y1, x2, y2 = nl[0].astype(float)
        roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)
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
        self, line, image, angle=None, min_rel_base=0.15, max_rel_base=0.4
    ):
        """
        Clip opposite line to vertical bounds with adaptive X-range based.
        on prominent angle.

        Returns: (clipped_line, (min_rel, max_rel)) where min_rel and
        max_rel are the normalized bounds used for clipping.

        At 90° (straight): min_rel=0.15, max_rel=0.4
        At 67° (right curve): min_rel=0.35, max_rel=0.65
        Interpolates between these points based on angle.
        """
        if line is None or image is None:
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

        clipped_line = self.clip_line_to_vertical_bounds(
            line, image, min_rel=min_rel, max_rel=max_rel
        )
        return clipped_line, (min_rel, max_rel)

    def clip_ego_line_adaptive(
        self, line, image, angle=None, min_rel_base=0.5, max_rel_base=0.75
    ):
        """
        Clip ego line to vertical bounds with adaptive X-range based.
        on prominent angle (opposite direction to opp line).

        Returns: (clipped_line, (min_rel, max_rel)) where min_rel and
        max_rel are the normalized bounds used for clipping.

        At 90° (straight): min_rel=0.5, max_rel=0.75
        At 113° (left curve): min_rel=0.25, max_rel=0.55
        Interpolates between these points based on angle.
        """
        if line is None or image is None:
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

        clipped_line = self.clip_line_to_vertical_bounds(
            line, image, min_rel=min_rel, max_rel=max_rel
        )
        return clipped_line, (min_rel, max_rel)

    def find_intersections(self, lines1, lines2):
        """
        Find intersections between two sets of lines.

        Arguments:
            lines1 -- First set of lines as pairs of points.
            lines2 -- Second set of lines as pairs of points.

        Returns:
            List of intersection points.
        """
        intersections = []

        def line_from_segment(x1, y1, x2, y2):
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            return A, B, C

        for line1 in lines1:
            x1, y1, x2, y2 = line1[0]
            a1, b1, c1 = line_from_segment(x1, y1, x2, y2)

            for line2 in lines2:
                x1, y1, x2, y2 = line2[0]
                a2, b2, c2 = line_from_segment(x1, y1, x2, y2)

                denom = a1 * b2 - a2 * b1
                if denom == 0:
                    continue

                px = (b1 * c2 - b2 * c1) / denom
                py = (c1 * a2 - c2 * a1) / denom

                intersections.append((int(px), int(py)))

        return intersections

    def find_crossing_center(self, intersection_points):
        """
        Find center of crossing using robust clustering with fallbacks.

        Arguments:
            intersection_points -- List of (x, y) intersection points

        Returns:
            Tuple (x, y) of crossing center, or None if not determinable
        """
        if len(intersection_points) == 0:
            return None

        if len(intersection_points) < 2:
            return None

        points = np.array(intersection_points, dtype=float)

        try:
            clustering = DBSCAN(eps=40, min_samples=2).fit(points)
            labels = clustering.labels_

            valid_mask = labels != -1
            valid_labels = labels[valid_mask]

            if len(valid_labels) == 0:
                center = points.mean(axis=0)
                return (int(center[0]), int(center[1]))

            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            largest_label = unique_labels[np.argmax(counts)]

            cluster_points = points[labels == largest_label]
            center = cluster_points.mean(axis=0)

            return (int(center[0]), int(center[1]))

        except Exception as e:
            self.get_logger().warning(f"DBSCAN clustering failed ({e}), using mean")
            try:
                center = points.mean(axis=0)
                return (int(center[0]), int(center[1]))
            except Exception:
                return None

    def find_closest_line_to_roi_bottom(self, lines, roi_bbox, image=None):
        """
        Find closest lines on left and right of ROI center in quadrants q3/q4.

        Separates lines into left (x < roi_center_x) and right (x > roi_center_x)
        groups, finds closest line in each group, and validates angle consistency.

        Arguments:
            lines -- List of detected lines (each line is [[x1, y1, x2, y2]])
            roi_bbox -- Tuple (roi_left, roi_right, roi_top, roi_bottom)
            image -- Optional image for quadrant calculation

        Returns:
            Tuple of (closest_line, angle_degrees) or (None, None) if validation fails
        """
        if lines is None or len(lines) == 0:
            return None, None

        roi_left, roi_right, roi_top, roi_bottom = roi_bbox

        roi_center_x = (roi_left + roi_right) / 2.0
        roi_bottom_y = roi_bottom

        left_lines = []
        right_lines = []

        for line in lines:
            try:
                nl = self._normalize_line(line)
                if nl is None:
                    continue

                x1, y1, x2, y2 = nl[0]
                line_center_x = (x1 + x2) / 2.0

                if line_center_x < roi_center_x:
                    left_lines.append(nl)
                else:
                    right_lines.append(nl)
            except Exception:
                continue

        closest_left = None
        closest_left_dist = float("inf")
        closest_left_angle = None

        for line in left_lines:
            try:
                x1, y1, x2, y2 = line[0]
                line_center_x = (x1 + x2) / 2.0
                line_center_y = (y1 + y2) / 2.0

                distance = math.hypot(
                    line_center_x - roi_center_x, line_center_y - roi_bottom_y
                )

                if distance < closest_left_dist:
                    closest_left_dist = distance
                    closest_left = line

                    dx = x2 - x1
                    dy = y2 - y1
                    angle_rad = math.atan2(dy, dx)
                    angle_deg = math.degrees(angle_rad)
                    angle_norm = (angle_deg + 360.0) % 180.0
                    closest_left_angle = angle_norm
            except Exception:
                continue

        closest_right = None
        closest_right_dist = float("inf")
        closest_right_angle = None

        for line in right_lines:
            try:
                x1, y1, x2, y2 = line[0]
                line_center_x = (x1 + x2) / 2.0
                line_center_y = (y1 + y2) / 2.0

                distance = math.hypot(
                    line_center_x - roi_center_x, line_center_y - roi_bottom_y
                )

                if distance < closest_right_dist:
                    closest_right_dist = distance
                    closest_right = line

                    dx = x2 - x1
                    dy = y2 - y1
                    angle_rad = math.atan2(dy, dx)
                    angle_deg = math.degrees(angle_rad)
                    angle_norm = (angle_deg + 360.0) % 180.0
                    closest_right_angle = angle_norm
            except Exception:
                continue

        if closest_left is None or closest_right is None:
            return None, None

        if closest_left_angle is None or closest_right_angle is None:
            return None, None

        angle_diff = abs(closest_left_angle - closest_right_angle)
        if angle_diff > 10.0:
            return None, None

        avg_angle = (closest_left_angle + closest_right_angle) / 2.0
        return closest_left, avg_angle

    def find_prominent_angle_in_quadrants(self, lines, image):
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

    def find_corners_shi_tomasi(self, image, roi_bbox=None):
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

    def compute_crossing_center_from_corners(self, corners):
        """
        Compute crossing center from Shi-Tomasi corners if they form a rectangle.

        The algorithm:
        1. Sorts corners by angle around their centroid
        2. Calculates interior angles of the polygon
        3. If angle sum ≈ 360°, treats it as a valid rectangle
        4. Returns the centroid as crossing center

        Args:
            corners: List of (x, y) corner coordinates.

        Returns:
            Tuple (center_x, center_y) if valid rectangle, None otherwise.
        """
        if not corners or len(corners) < 3:
            return None

        try:
            corners_array = np.array(corners, dtype=np.float32)

            centroid = np.mean(corners_array, axis=0)

            angles = np.arctan2(
                corners_array[:, 1] - centroid[1], corners_array[:, 0] - centroid[0]
            )
            sorted_indices = np.argsort(angles)
            sorted_corners = corners_array[sorted_indices]

            n = len(sorted_corners)
            angle_sum = 0.0

            for i in range(n):
                p1 = sorted_corners[(i - 1) % n]
                p2 = sorted_corners[i]
                p3 = sorted_corners[(i + 1) % n]

                v1 = p1 - p2
                v2 = p3 - p2

                cos_angle = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
                )
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angle_sum += np.degrees(angle)

            expected_sum = (n - 2) * 180.0

            tolerance = 30.0
            if abs(angle_sum - expected_sum) <= tolerance:
                center = tuple(centroid.astype(int))
                return center
            else:
                return None

        except Exception as e:
            self.get_logger().warning(f"Crossing center from corners failed: {e}")
            return None

    def compute_corner_angles(self, corners):
        """
        Calculate the interior angles at each corner of a polygon.

        Args:
            corners: List of (x, y) corner coordinates.

        Returns:
            Tuple of (sorted_corners, angles_in_degrees) where:
            - sorted_corners: corners sorted by angle around centroid
            - angles_in_degrees: list of interior angles at each corner
        """
        if not corners or len(corners) < 3:
            return None, None

        try:
            corners_array = np.array(corners, dtype=np.float32)

            centroid = np.mean(corners_array, axis=0)

            angles_rad = np.arctan2(
                corners_array[:, 1] - centroid[1], corners_array[:, 0] - centroid[0]
            )
            sorted_indices = np.argsort(angles_rad)
            sorted_corners = corners_array[sorted_indices]

            n = len(sorted_corners)
            interior_angles = []

            for i in range(n):
                p1 = sorted_corners[(i - 1) % n]
                p2 = sorted_corners[i]
                p3 = sorted_corners[(i + 1) % n]

                v1 = p1 - p2
                v2 = p3 - p2

                cos_angle = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
                )
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                interior_angles.append(angle_deg)

            return sorted_corners, interior_angles

        except Exception as e:
            self.get_logger().warning(f"Corner angle calculation failed: {e}")
            return None, None

    def is_valid_rectangle(self, interior_angles, angle_tolerance=20.0):
        """
        Check if angles form a valid rectangle.

        A valid rectangle requires at least 3 out of 4 angles close to 90°.

        Args:
            interior_angles: List of interior angles in degrees.
            angle_tolerance: Allowed tolerance from 90° (default 20°).

        Returns:
            Boolean indicating if at least 3 angles are within tolerance of 90°.
        """
        if not interior_angles or len(interior_angles) < 3:
            return False

        target_angle = 90.0
        valid_angle_count = sum(
            1
            for angle in interior_angles
            if abs(angle - target_angle) <= angle_tolerance
        )

        return valid_angle_count >= 3

    def compute_angle_error(self, interior_angles):
        """
        Calculate mean deviation of interior angles from 90°.

        Args:
            interior_angles: List of interior angles in degrees.

        Returns:
            Float representing the mean absolute deviation from 90°.
            Returns inf if angles list is empty.
        """
        if not interior_angles:
            return float("inf")

        target_angle = 90.0
        deviations = [abs(angle - target_angle) for angle in interior_angles]
        return np.mean(deviations)

    def pull_point_to_roi_center(self, point, img_shape):
        """
        Pull a point towards the center of the ROI.

        Arguments:
            point -- The point to be pulled.
            img_shape -- Shape of the image.

        Returns:
            New point closer to the ROI center.
        """
        height = img_shape[0]
        width = img_shape[1]

        roi_left = int(width * FILTERING_ROI_REL_RLTB[1])
        roi_right = int(width * FILTERING_ROI_REL_RLTB[0])
        roi_top = int(height * FILTERING_ROI_REL_RLTB[2])
        roi_bottom = int(height * FILTERING_ROI_REL_RLTB[3])

        roi_center_x = (roi_left + roi_right) / 2
        roi_center_y = (roi_top + roi_bottom) / 2

        roi_diagonal_distance = (
            math.sqrt((roi_right - roi_left) ** 2 + (roi_bottom - roi_top) ** 2) / 2
        )

        distance_to_roi_center = math.sqrt(
            (point[0] - roi_center_x) ** 2 + (point[1] - roi_center_y) ** 2
        )
        pull_factor = min(1.0, distance_to_roi_center / roi_diagonal_distance)

        x, y = point
        new_x = (1 - pull_factor) * x + pull_factor * roi_center_x
        new_y = (1 - pull_factor) * y + pull_factor * roi_center_y

        return (new_x, new_y)

    def find_opp_line(self, horiz_lines, crossing_center, ghost_cc=None):
        """
        Find the opposite line from the list of lines.

        Uses ghost_cc if provided, otherwise falls back to crossing_center.

        Arguments:
            horiz_lines -- List of horizontal lines as pairs of points.
            crossing_center -- Main crossing center (x, y)
            ghost_cc -- Optional ghost crossing center for opp line search
        """
        max_distance = 10000
        nearest_line = None

        search_center = ghost_cc if ghost_cc is not None else crossing_center

        line_candidates = []

        for line in horiz_lines:
            x1, y1, x2, y2 = line[0]
            line_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            if line_center[1] < crossing_center[1]:
                line_candidates.append(line)

        for line in line_candidates:
            x1, y1, x2, y2 = line[0]
            line_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            distance_to_search_center = math.sqrt(
                (line_center[0] - search_center[0]) ** 2
                + (line_center[1] - search_center[1]) ** 2
            )

            if distance_to_search_center < max_distance:
                max_distance = distance_to_search_center
                nearest_line = line

        if nearest_line is not None:
            x1, y1, x2, y2 = nearest_line[0]
            if math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) < 90:
                return None
        return nearest_line

    def calculate_ghost_crossing_centers(
        self, crossing_center, prominent_angle, offset_distance: float = 85.0
    ):
        """
        Calculate ghost crossing centers offset from main crossing center.

        Projects two points from the main crossing center along the prominent
        angle direction:
        - ego_ghost_cc: 70px backward (opposite direction)
        - opp_ghost_cc: 70px forward (same direction)

        This helps find ego/opp lines that are not in the middle of the road.

        Arguments:
            crossing_center -- Main crossing center (x, y)
            prominent_angle -- Angle in degrees (0-180)
            offset_distance -- How far to offset (default 70px)

        Returns:
            Tuple of (ego_ghost_cc, opp_ghost_cc) as (x, y) tuples,
            or (None, None) if angle is not available
        """
        if crossing_center is None or prominent_angle is None:
            return None, None

        try:
            angle_rad = math.radians(float(prominent_angle))

            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)

            cx, cy = crossing_center

            opp_ghost_x = cx - dx * offset_distance
            opp_ghost_y = cy - dy * offset_distance
            opp_ghost_cc = (int(opp_ghost_x), int(opp_ghost_y))

            ego_ghost_x = cx + dx * offset_distance
            ego_ghost_y = cy + dy * offset_distance
            ego_ghost_cc = (int(ego_ghost_x + 20), int(ego_ghost_y))

            return ego_ghost_cc, opp_ghost_cc

        except Exception as e:
            self.get_logger().error(f"Error calculating ghost CCs: {e}")
            return None, None

    def calculate_stop_line_ghost_centers(
        self, crossing_center, prominent_angle, offset_distance: float = 100.0
    ):
        """
        Calculate ghost crossing centers for stop lines (left and right).

        Projects two points from the main crossing center along the orthogonal
        (perpendicular) direction to the prominent angle:
        - left_ghost_cc: offset to the left (perpendicular)
        - right_ghost_cc: offset to the right (perpendicular)

        This helps find left/right stop lines that are offset from center.

        Arguments:
            crossing_center -- Main crossing center (x, y)
            prominent_angle -- Angle in degrees (0-180)
            offset_distance -- How far to offset (default 100px)

        Returns:
            Tuple of (left_ghost_cc, right_ghost_cc) as (x, y) tuples,
            or (None, None) if angle is not available
        """
        if crossing_center is None or prominent_angle is None:
            return None, None

        try:
            angle_rad = math.radians(float(prominent_angle))

            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)

            orth_dx = -dy
            orth_dy = dx

            cx, cy = crossing_center

            left_ghost_x = cx + orth_dx * offset_distance
            left_ghost_y = cy + orth_dy * offset_distance
            left_ghost_cc = (int(left_ghost_x), int(left_ghost_y))

            right_ghost_x = cx - orth_dx * offset_distance
            right_ghost_y = cy - orth_dy * offset_distance
            right_ghost_cc = (int(right_ghost_x), int(right_ghost_y))

            return left_ghost_cc, right_ghost_cc

        except Exception as e:
            self.get_logger().error(f"Error calculating stop line ghost CCs: {e}")
            return None, None

    def find_ego_line(self, horiz_lines, crossing_center, ghost_cc=None):
        """
        Find the ego line from the list of lines.

        Uses ghost_cc if provided, otherwise falls back to crossing_center.

        Arguments:
            horiz_lines -- List of horizontal lines as pairs of points.
            crossing_center -- Main crossing center (x, y)
            ghost_cc -- Optional ghost crossing center for ego line search
        """
        max_distance = 10000
        nearest_line = None

        search_center = ghost_cc if ghost_cc is not None else crossing_center

        lines_candidates = []

        for line in horiz_lines:
            x1, y1, x2, y2 = line[0]
            line_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            if line_center[1] > crossing_center[1]:
                lines_candidates.append(line)

        for line in lines_candidates:
            x1, y1, x2, y2 = line[0]
            line_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            distance_to_search_center = math.sqrt(
                (line_center[0] - search_center[0]) ** 2
                + (line_center[1] - search_center[1]) ** 2
            )

            if distance_to_search_center < max_distance:
                max_distance = distance_to_search_center
                nearest_line = line

        if nearest_line is not None:
            x1, y1, x2, y2 = nearest_line[0]
            if math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) < 80:
                self.get_logger().debug("Nearest line too short, rejecting")
                return None
        return nearest_line

    def check_plausibility_horizontal_line_pair(
        self,
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

    def check_stop_line_pair_plausibility(
        self,
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

    def check_stop_line_endpoints_for_horizontals(self, stop_line, line_name, horiz):
        """
        Check if horizontal lines exist near both endpoints of stop line.

        Validates that a stop line connects to horizontal lines at both
        of its endpoints. Returns False if any endpoint has 0 horizontal
        lines in a 60x60 region around it.

        Arguments:
            stop_line -- Stop line as (1,4) numpy array [x1,y1,x2,y2]
            line_name -- Name for logging ("LEFT" or "RIGHT")
            horiz -- List of horizontal lines to check against

        Returns:
            True if valid (has horiz lines at both endpoints),
            False if should be rejected, None if stop_line is None.
        """
        if stop_line is None:
            return None

        x1, y1, x2, y2 = stop_line[0]

        if y1 < y2:
            top_pt = (x1, y1)
            bottom_pt = (x2, y2)
        else:
            top_pt = (x2, y2)
            bottom_pt = (x1, y1)

        endpoints = [
            ("top", top_pt),
            ("bottom", bottom_pt),
        ]

        endpoint_results = []

        for endpoint_name, (ep_x, ep_y) in endpoints:
            search_region = (
                int(ep_x - 30),
                int(ep_y - 30),
                int(ep_x + 30),
                int(ep_y + 30),
            )

            horiz_lines_in_region = [
                line
                for line in horiz
                if line is not None and self._is_line_in_region(line, search_region)
            ]

            num_lines = len(horiz_lines_in_region)
            self.get_logger().debug(
                f"{line_name} stop {endpoint_name} endpoint check "
                f"(y={ep_y:.1f}): found {num_lines} horizontal lines "
                f"in 60x60 region"
            )

            if num_lines == 0:
                endpoint_results.append(False)
            else:
                endpoint_results.append(True)

        return any(endpoint_results)

    def check_stop_line_crossing_openness(
        self, stop_line_right, stop_line_left, image_gray
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
                black_pixels = np.sum(region < 40)
                black_pct = (
                    (black_pixels / total_pixels * 100) if total_pixels > 0 else 0
                )

                right_valid = black_pct > 55
                self.get_logger().debug(
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
                black_pixels = np.sum(region < 40)
                black_pct = (
                    (black_pixels / total_pixels * 100) if total_pixels > 0 else 0
                )

                left_valid = black_pct > 55
                self.get_logger().debug(
                    f"LEFT stop highest point "
                    f"(x={top_x_l:.1f}, y={top_y_l:.1f}): "
                    f"{black_pct:.1f}%"
                )
            else:
                left_valid = False

        return right_valid, left_valid

    def measure_stop_line_thickness(self, stop_line_left, stop_line_right, image_gray):
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
                        self.get_logger().debug(
                            f"RIGHT thickness: {right_thickness} px, "
                            f"avg={avg_pix:.0f} "
                            f"(mid x={mid_x_r:.1f}, y={mid_y_r:.1f})"
                        )
                except Exception as e:
                    self.get_logger().error(f"Error measuring RIGHT thickness: {e}")

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
                        self.get_logger().debug(
                            f"LEFT thickness: {left_thickness} px, "
                            f"avg={avg_pix:.0f} "
                            f"(mid x={mid_x_l:.1f}, y={mid_y_l:.1f})"
                        )
                except Exception as e:
                    self.get_logger().error(f"Error measuring LEFT thickness: {e}")

            return left_thickness, right_thickness

        except Exception as e:
            self.get_logger().error(f"Error in measure_stop_line_thickness: {e}")
            return None, None

    def calculate_cat_eye(self, image):
        """
        Generate two cat's eye shapes (diamonds) for left and right stop line detection.

        The cat's eyes focus detection on two regions across the ROI width:
        - cat_eye_left (at ~0.3 of ROI width): detects left stop lines
        - cat_eye_right (at ~0.7 of ROI width): detects right stop lines

        Arguments:
            image -- Input image

        Returns:
            Tuple of (cat_eye_left, cat_eye_right), each as a list of 4 points:
                      [top_point, right_point, bottom_point, left_point]
        """
        [xs, xe, ys, ye] = self.get_roi_bbox(image.shape)

        roi_width = xe - xs
        roi_height = ye - ys
        roi_center_y = ys + roi_height / 2.0

        x_left = int(xs + roi_width * 0.25)
        top_point_left = (
            x_left - int(roi_width * 0.10),
            int(roi_height * 0.70 + ys),
        )
        right_point_left = (x_left, int(ys + roi_height * 0.35))
        bottom_point_left = (
            x_left + int(roi_width * 0.10),
            int(roi_height * 0.70 + ys),
        )
        left_point_left = (x_left, int(ys + roi_height * 0.98))
        cat_eye_left = [
            top_point_left,
            right_point_left,
            bottom_point_left,
            left_point_left,
        ]

        # Cat's eye right (at ~0.7 of ROI width) for right stop line
        x_right = int(xs + roi_width * 0.73)
        top_point_right = (x_right - int(roi_width * 0.10), int(roi_height * 0.3 + ys))
        right_point_right = (x_right, int(ys + roi_height * 0.02))
        bottom_point_right = (
            x_right + int(roi_width * 0.10),
            int(roi_height * 0.3 + ys),
        )
        left_point_right = (x_right, int(ys + roi_height * 0.65))
        cat_eye_right = [
            top_point_right,
            right_point_right,
            bottom_point_right,
            left_point_right,
        ]

        return cat_eye_left, cat_eye_right

    def calculate_roi_quadrants(self, image):
        """
        Split the ROI into 4 quadrants for line detection.

        Returns quadrants positioned as:
        [Q1(top-left), Q2(top-right), Q3(bottom-left), Q4(bottom-right)]

        Each quadrant is a polygon (list of 4 corner points).

        Arguments:
            image -- Input image

        Returns:
            Tuple of 4 quadrants, each as list of 4 points
        """
        [xs, xe, ys, ye] = self.get_roi_bbox(image.shape)

        x_center = (xs + xe) / 2.0
        y_center = (ys + ye) / 2.0

        q1 = [
            (int(xs), int(ys)),
            (int(x_center), int(ys)),
            (int(x_center), int(y_center)),
            (int(xs), int(y_center)),
        ]

        q2 = [
            (int(x_center), int(ys)),
            (int(xe), int(ys)),
            (int(xe), int(y_center)),
            (int(x_center), int(y_center)),
        ]

        q3 = [
            (int(xs), int(y_center)),
            (int(x_center), int(y_center)),
            (int(x_center), int(ye)),
            (int(xs), int(ye)),
        ]

        q4 = [
            (int(x_center), int(y_center)),
            (int(xe), int(y_center)),
            (int(xe), int(ye)),
            (int(x_center), int(ye)),
        ]

        return q1, q2, q3, q4

    def _draw_quadrants(
        self, image, q1, q2, q3, q4, color=(100, 100, 100), thickness=1
    ):
        """
        Draw the 4 ROI quadrants on the image.

        Arguments:
            image -- Image to draw on
            q1, q2, q3, q4 -- Quadrant polygons (lists of 4 points each)
            color -- Line color in BGR format
            thickness -- Line thickness

        Returns:
            Image with quadrants drawn
        """
        for quad in [q1, q2, q3, q4]:
            pts = np.array(quad, np.int32)
            cv2.polylines(image, [pts], True, color, thickness)
        return image

    def find_line_in_quadrant(
        self, lines, quadrant, min_length: float = 30.0, require_full: bool = False
    ):
        """
        Find the longest vertical line inside a quadrant.

        Arguments:
            lines -- List of detected line segments
            quadrant -- Quadrant polygon (list of 4 points)
            min_length -- Minimum line length to consider

        Returns:
            Best vertical line found in quadrant, or None
        """
        if lines is None or len(lines) == 0 or quadrant is None:
            return None

        quad_lines = self.filter_lines_by_polygon(
            lines, quadrant, require_full=require_full
        )

        if quad_lines is None or len(quad_lines) == 0:
            return None

        vert, horiz = self.filter_by_angle(quad_lines, tol_deg=10)

        if vert is None or len(vert) == 0:
            return None

        vert = self.filter_by_length(vert, min_length=min_length)

        if vert is None or len(vert) == 0:
            return None

        best_line = max(vert, key=lambda line: self._line_length(line))
        return self.elongate_line(best_line, length=200)

    def find_stop_line_in_cone(self, lines, cone, min_length: float = 30.0):
        """
        Find the longest vertical line inside a cone region.

        Vertical lines are perpendicular to the ego lane (stop lines).
        This is more robust than looking for multiple small lines.

        Arguments:
            lines -- List of detected line segments
            cone -- Cone region [start_point, arm_left_point, arm_right_point]
            min_length -- Minimum line length to consider

        Returns:
            Best vertical line found in cone, or None
        """
        if lines is None or len(lines) == 0 or cone is None:
            return None

        cone_lines = self.filter_lines_by_cone(lines, cone, require_full=True)

        if cone_lines is None or len(cone_lines) == 0:
            return None

        vert, horiz = self.filter_by_angle(cone_lines, tol_deg=10)

        if vert is None or len(vert) == 0:
            return None

        vert = self.filter_by_length(vert, min_length=min_length)

        if vert is None or len(vert) == 0:
            return None

        best_line = max(vert, key=lambda line: self._line_length(line))
        return self.elongate_line(best_line, length=180)

    def find_stop_line_in_cat_eye(self, lines, cat_eye, min_length: float = 30.0):
        """
        Find the longest vertical line inside the cat_eye diamond region.

        Vertical lines are perpendicular to the ego lane (stop lines).

        Arguments:
            lines -- List of detected line segments
            cat_eye -- Cat_eye region [top_point, right_point, bottom_point, left_point]
            min_length -- Minimum line length to consider

        Returns:
            Best vertical line found in cat_eye, or None
        """
        if lines is None or len(lines) == 0 or cat_eye is None:
            return None

        cat_eye_lines = self.filter_lines_by_polygon(lines, cat_eye, require_full=True)

        if cat_eye_lines is None or len(cat_eye_lines) == 0:
            return None

        vert, horiz = self.filter_by_angle(cat_eye_lines, tol_deg=10)

        if vert is None or len(vert) == 0:
            return None

        vert = self.filter_by_length(vert, min_length=min_length)

        if vert is None or len(vert) == 0:
            return None

        best_line = max(vert, key=lambda line: self._line_length(line))
        return self.elongate_line(best_line, length=180)

    def find_stop_line_by_ghost_cc(self, lines, ghost_cc, min_length: float = 60.0):
        """
        Find the closest vertical line to a ghost crossing center.

        Searches through all vertical lines and returns the one that is:
        1. Longer than min_length
        2. Closest to the ghost_cc point

        This is used to find left/right stop lines that may be offset
        from the center of the road.

        Arguments:
            lines -- List of detected line segments
            ghost_cc -- Ghost crossing center point (x, y) tuple
            min_length -- Minimum line length to consider (default 60px)

        Returns:
            Best vertical line closest to ghost_cc, or None
        """
        if lines is None or len(lines) == 0 or ghost_cc is None:
            return None

        try:
            long_lines = self.filter_by_length(lines, min_length=min_length)

            if long_lines is None or len(long_lines) == 0:
                return None

            vert, horiz = self.filter_by_angle(long_lines, tol_deg=15)

            if vert is None or len(vert) == 0:
                return None

            closest_line = None
            min_distance = float("inf")

            for line in vert:
                x1, y1, x2, y2 = line[0]
                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0

                distance = math.hypot(mid_x - ghost_cc[0], mid_y - ghost_cc[1])

                if distance < min_distance:
                    min_distance = distance
                    closest_line = line

            return closest_line

        except Exception as e:
            self.get_logger().error(f"Error finding stop line by ghost CC: {e}")
            return None

    def find_left_stop_line_by_ghost_cc(
        self, lines, left_ghost_cc, min_length: float = 60.0
    ):
        """
        Find the left stop line using the left ghost crossing center.

        Wrapper around find_stop_line_by_ghost_cc for the left side.

        Arguments:
            lines -- List of detected line segments
            left_ghost_cc -- Left ghost crossing center (x, y) tuple
            min_length -- Minimum line length to consider (default 60px)

        Returns:
            Left stop line closest to left_ghost_cc, or None
        """
        return self.find_stop_line_by_ghost_cc(lines, left_ghost_cc, min_length)

    def find_right_stop_line_by_ghost_cc(
        self, lines, right_ghost_cc, min_length: float = 60.0
    ):
        """
        Find the right stop line using the right ghost crossing center.

        Wrapper around find_stop_line_by_ghost_cc for the right side.

        Arguments:
            lines -- List of detected line segments
            right_ghost_cc -- Right ghost crossing center (x, y) tuple
            min_length -- Minimum line length to consider (default 60px)

        Returns:
            Right stop line closest to right_ghost_cc, or None
        """
        return self.find_stop_line_by_ghost_cc(lines, right_ghost_cc, min_length)

    def _line_length(self, line):
        """Calculate the Euclidean length of a line segment."""
        x1, y1, x2, y2 = line[0]
        return math.hypot(x2 - x1, y2 - y1)

    def _draw_cone(self, cone, image):
        [cone_start_point, cone_arm_left_point, cone_arm_right_point] = cone
        cv2.line(image, cone_arm_left_point, cone_start_point, (0, 255, 0), 1)
        cv2.line(image, cone_arm_right_point, cone_start_point, (0, 255, 0), 1)

        return image

    def _draw_cat_eye(self, cat_eye, image, color=(255, 165, 0), thickness=2):
        """
        Draw the cat_eye diamond shape on the image.

        Arguments:
            cat_eye -- List of 4 points [top, right, bottom, left]
            image -- Image to draw on
            color -- Color for the lines (BGR format, default orange)
            thickness -- Line thickness

        Returns:
            Image with cat_eye drawn
        """
        if cat_eye is None or len(cat_eye) < 4:
            return image

        top, right, bottom, left = cat_eye

        cv2.line(image, top, right, color, thickness)
        cv2.line(image, right, bottom, color, thickness)
        cv2.line(image, bottom, left, color, thickness)
        cv2.line(image, left, top, color, thickness)

        return image

    def filter_lines_by_cone(self, lines, cone, require_full=True):
        """
        Filter lines to those that lie inside the triangular cone.

        Arguments:
            lines -- iterable of lines (any accepted format by _normalize_line)
            cone -- list/tuple as returned by calculate_cone: [start, arm_left, arm_right]
            require_full -- if True (default) keep only lines where both endpoints are
                            inside the cone. If False, keep lines with at least one
                            endpoint inside.

        Returns:
            List of normalized lines (each as numpy array shape (1,4)).
        """
        if not cone or lines is None:
            return []

        try:
            cone_start, cone_left, cone_right = cone
        except Exception:
            # invalid cone format
            return []

        poly = np.array([cone_left, cone_right, cone_start], dtype=np.int32)

        filtered = []
        for ln in lines:
            nl = self._normalize_line(ln)
            if nl is None:
                continue
            x1, y1, x2, y2 = nl[0].astype(int)

            d2 = cv2.pointPolygonTest(poly, (int(x2), int(y2)), False)

            if require_full:
                if d1 >= 0 and d2 >= 0:
                    filtered.append(nl)
            else:
                if d1 >= 0 or d2 >= 0:
                    filtered.append(nl)

        return filtered

    def filter_lines_by_polygon(self, lines, polygon, require_full=True):
        """
        Filter lines to those inside a polygon region.

        Arguments:
            lines -- iterable of lines (any accepted format by _normalize_line)
            polygon -- list of points defining the polygon (e.g., cat_eye)
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
            nl = self._normalize_line(ln)
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

    def is_line_in_quadrant(self, line, quadrant):
        """
        Check if a line is contained within a quadrant polygon.

        A line is considered "in" a quadrant if both endpoints are inside
        or on the edge of the quadrant polygon.

        Arguments:
            line -- Line as numpy array [[x1, y1, x2, y2]]
            quadrant -- Quadrant as list of 4 corner points

        Returns:
            True if both line endpoints are in the quadrant, False otherwise
        """
        if line is None or quadrant is None:
            return False

        try:
            nl = self._normalize_line(line)
            if nl is None:
                return False

            x1, y1, x2, y2 = nl[0].astype(int)

            poly = np.array(quadrant, dtype=np.int32)

            d1 = cv2.pointPolygonTest(poly, (int(x1), int(y1)), False)
            d2 = cv2.pointPolygonTest(poly, (int(x2), int(y2)), False)

            return d1 >= 0 and d2 >= 0
        except Exception:
            return False

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

    def pipeline(self, image):
        """
        Complete processing pipeline for intersection detection.

        Arguments:
            img_path -- Path to the input image.
        """
        orig_image = image.copy()

        image = self.preprocess_image(image)
        enhanced_image = image

        q1, q2, q3, q4 = self.calculate_roi_quadrants(image)

        lines = self.line_segment_detector(image)
        lines = self._normalize_lines(lines)
        lines = self.filter_by_length(lines, min_length=20)
        filtered_lines = self.filter_by_roi(lines, image.shape)

        fused_lines = self.fuse_similar_lines(
            filtered_lines, angle_tol_deg=10, center_dist_tol=100
        )

        closest_line_angle = None
        if fused_lines is not None and len(fused_lines) > 0:
            closest_line_angle, line_count = self.find_prominent_angle_in_quadrants(
                fused_lines, orig_image
            )

        vert, horiz = self.filter_by_angle(
            fused_lines,
            anchor_angle=closest_line_angle,
            anchor_tolerance=10.0,
            tol_deg=5,
        )

        crossing_center = None
        if getattr(self, "compute_crossing_center", True):
            try:
                vert_filtered = self.filter_by_length(vert, min_length=100)
                horiz_filtered = self.filter_by_length(horiz, min_length=100)

                intersections = self.find_intersections(vert_filtered, horiz_filtered)
            except Exception as e:
                self.get_logger().error(f"find_intersections error: {e}")
                intersections = []
            try:
                crossing_center = self.find_crossing_center(intersections)
            except Exception as e:
                self.get_logger().error(f"find_crossing_center error: {e}")
                crossing_center = None
        else:
            roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)
            crossing_center = (
                int((roi_left + roi_right) / 2),
                int(roi_top + (roi_bottom - roi_top) * 0.4),
            )

        roi_bbox = self.get_roi_bbox(image.shape)
        detected_corners = self.find_corners_shi_tomasi(image, roi_bbox=roi_bbox)

        crossing_center = None
        if detected_corners and len(detected_corners) >= 3:
            sorted_corners, interior_angles = self.compute_corner_angles(
                detected_corners
            )

            if sorted_corners is not None and interior_angles is not None:
                corner_error = self.compute_angle_error(interior_angles)

                is_rect = self.is_valid_rectangle(interior_angles, angle_tolerance=20.0)

                if corner_error < 30.0 or is_rect:
                    center = np.mean(sorted_corners, axis=0).astype(int)
                    self.detected_crossing_center = tuple(center)
                    self.crossing_center_frames = 0
                    self.crossing_center_error = corner_error
                    self.active_crossing_center = self.detected_crossing_center

        if self.active_crossing_center is not None:
            roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)

            cx, cy = self.active_crossing_center
            shift_amount = 15 * (self.crossing_center_frames + 1)
            cy_shifted = min(cy + shift_amount, roi_bottom)

            crossing_center = (cx, int(cy_shifted))

            self.crossing_center_frames += 1

            if self.crossing_center_frames >= 4:
                self.active_crossing_center = None
                self.detected_crossing_center = None
                self.crossing_center_frames = 0
                roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(
                    image.shape
                )
                crossing_center = (
                    int((roi_left + roi_right) / 2),
                    int(roi_top + (roi_bottom - roi_top) * 0.4),
                )
        else:
            roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)
            crossing_center = (
                int((roi_left + roi_right) / 2),
                int(roi_top + (roi_bottom - roi_top) * 0.4),
            )

        lines = vert + horiz
        lines = self.filter_by_length(lines, min_length=70)
        fused_lines = lines

        if crossing_center is not None:
            (
                left_stop_ghost_cc,
                right_stop_ghost_cc,
            ) = self.calculate_stop_line_ghost_centers(
                crossing_center, closest_line_angle
            )

        if left_stop_ghost_cc is None and right_stop_ghost_cc is None:
            stop_line_left = self.find_line_in_quadrant(lines, q1, min_length=60)
            stop_line_right = self.find_line_in_quadrant(
                lines, q2, min_length=60, require_full=False
            )
        else:
            stop_line_left = self.find_left_stop_line_by_ghost_cc(
                lines, left_stop_ghost_cc, min_length=60
            )
            stop_line_right = self.find_right_stop_line_by_ghost_cc(
                lines, right_stop_ghost_cc, min_length=60
            )

        label_stop_line_left = None
        if stop_line_left is not None:
            (
                stop_dotted_left,
                gap_count_left,
                white_ratio_left,
                _,
            ) = self.is_line_dotted_by_gap_detection(
                stop_line_left,
                image,
                box_half_width=10,
                length_extend=1.2,
                min_gap_count=3,
            )
            min_wr = 25 if stop_dotted_left else 30

            (
                gaps_ext,
                wr_ext,
                stop_line_left_ext,
            ) = self.check_line_by_vertical_extension(
                stop_line_left, image, is_right=False
            )
            self.get_logger().debug(
                f"Left stop line: gaps_ext={gaps_ext}, wr_ext={wr_ext:.1f}%"
            )

            line_left_ext_passed = gaps_ext > 0 and wr_ext <= 10

            if white_ratio_left >= min_wr and line_left_ext_passed:
                label_stop_line_left = (
                    f"STOP_LEFT DOTTED (g={gap_count_left} wr={white_ratio_left:.1f}%)"
                    if stop_dotted_left
                    else f"STOP_LEFT SOLID (g={gap_count_left} wr={white_ratio_left:.1f}%)"
                )
            else:
                stop_line_left = None

        label_stop_line_right = None
        if stop_line_right is not None:
            (
                stop_dotted_right,
                gap_count_right,
                white_ratio_right,
                _,
            ) = self.is_line_dotted_by_gap_detection(
                stop_line_right,
                image,
                box_half_width=10,
                length_extend=1.2,
                min_gap_count=3,
            )
            min_wr = 25 if stop_dotted_right else 30

            (
                gaps_ext,
                wr_ext,
                stop_line_right_ext,
            ) = self.check_line_by_vertical_extension(
                stop_line_right, image, is_right=True
            )
            self.get_logger().debug(
                f"Right stop line: gaps_ext={gaps_ext}, wr_ext={wr_ext:.1f}%"
            )

            line_right_ext_passed = gaps_ext > 0 and wr_ext < 10

            if white_ratio_right >= min_wr and line_right_ext_passed:
                label_stop_line_right = (
                    f"STOP_RIGHT DOTTED "
                    f"(g={gap_count_right} wr={white_ratio_right:.1f}%)"
                    if stop_dotted_right
                    else (
                        f"STOP_RIGHT SOLID "
                        f"(g={gap_count_right} wr={white_ratio_right:.1f}%)"
                    )
                )
            else:
                stop_line_right = None

        if stop_line_right is not None and crossing_center is not None:
            x1_r, y1_r, x2_r, y2_r = stop_line_right[0]
            stop_y_r = (y1_r + y2_r) / 2.0
            crossing_y = crossing_center[1]
            if stop_y_r > crossing_y:
                stop_line_right = None
                label_stop_line_right = None
            else:
                min_y_r = min(y1_r, y2_r)
                if min_y_r > crossing_y:
                    stop_line_right = None
                    label_stop_line_right = None

        roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)
        roi_width = roi_right - roi_left
        min_x_inset = roi_left + roi_width * 0.1
        max_x_inset = roi_right - roi_width * 0.1
        min_x_right_stop = roi_left + roi_width * 0.6
        max_x_left_stop = roi_left + roi_width * 0.4

        if stop_line_right is not None:
            x1_r, y1_r, x2_r, y2_r = stop_line_right[0]
            stop_x_r = (x1_r + x2_r) / 2.0
            if stop_x_r < min_x_inset or stop_x_r > max_x_inset:
                stop_line_right = None
                label_stop_line_right = None
            elif stop_x_r < min_x_right_stop:
                self.get_logger().debug(
                    f"RIGHT stop rejected: x={stop_x_r:.1f} is too close to "
                    f"left edge (min={min_x_right_stop:.1f})"
                )
                stop_line_right = None
                label_stop_line_right = None
            elif crossing_center is not None:
                crossing_x = crossing_center[0]
                if stop_x_r < crossing_x:
                    self.get_logger().debug(
                        f"RIGHT stop rejected: x={stop_x_r:.1f} is left of "
                        f"crossing x={crossing_x:.1f}"
                    )
                    stop_line_right = None
                    label_stop_line_right = None

        if stop_line_left is not None and crossing_center is not None:
            x1_l, y1_l, x2_l, y2_l = stop_line_left[0]
            stop_y_l = (y1_l + y2_l) / 2.0
            stop_x_l = (x1_l + x2_l) / 2.0
            crossing_y = crossing_center[1]
            crossing_x = crossing_center[0]
            if stop_x_l > crossing_x:
                self.get_logger().debug(
                    f"LEFT stop rejected: x={stop_x_l:.1f} is right of "
                    f"crossing x={crossing_x:.1f}"
                )
                stop_line_left = None
                label_stop_line_left = None
            elif stop_x_l > max_x_left_stop:
                self.get_logger().debug(
                    f"LEFT stop rejected: x={stop_x_l:.1f} is too close to "
                    f"right edge (max={max_x_left_stop:.1f})"
                )
                stop_line_left = None
                label_stop_line_left = None
            else:
                max_y_l = max(y1_l, y2_l)
                if max_y_l < crossing_y:
                    stop_line_left = None
                    label_stop_line_left = None
                else:
                    self._stop_left_y = stop_y_l

        cross_right_open, cross_left_open = self.check_stop_line_crossing_openness(
            stop_line_right, stop_line_left, image
        )

        if not cross_right_open:
            self.get_logger().debug(
                "RIGHT stop rejected: crossing closing area too dark"
            )
            stop_line_right = None
            label_stop_line_right = None

        if not cross_left_open:
            self.get_logger().debug(
                "LEFT stop rejected: crossing closing area too dark"
            )
            stop_line_left = None
            label_stop_line_left = None

        left_thickness, right_thickness = self.measure_stop_line_thickness(
            stop_line_left, stop_line_right, image
        )

        if left_thickness is not None and left_thickness < 18 and not stop_dotted_left:
            self.get_logger().debug(
                f"LEFT stop rejected: thickness {left_thickness:.1f} is too thin"
            )
            stop_line_left = None
            label_stop_line_left = None

        if (
            right_thickness is not None
            and right_thickness < 18
            and not stop_dotted_right
        ):
            self.get_logger().debug(
                f"RIGHT stop rejected: thickness {right_thickness:.1f} is too thin"
            )
            stop_line_right = None
            label_stop_line_right = None

        stop_line_left, stop_line_right = self.check_stop_line_pair_plausibility(
            stop_line_left,
            stop_line_right,
            max_y_diff=300.0,
            min_y_diff=100,
            max_x_separation=380.0,
            min_x_separation=280.0,
        )
        if stop_line_left is None:
            label_stop_line_left = None
            self._stop_left_y = None
        if stop_line_right is None:
            label_stop_line_right = None

        lines = self.filter_by_length(lines, min_length=100)

        vert, horiz = self.filter_by_angle(
            fused_lines,
            anchor_angle=closest_line_angle,
            anchor_tolerance=10.0,
            tol_deg=5,
        )

        ego_line_long = None
        opp_line_long = None
        opp_line_extended = None
        pair_plausible = False
        label_ego = None
        label_opp = None
        ego_ghost_cc = None
        opp_ghost_cc = None
        ego_clip_bounds = None
        opp_clip_bounds = None
        clipped_ego = None
        clipped_opp = None

        if crossing_center is not None:
            ego_ghost_cc, opp_ghost_cc = self.calculate_ghost_crossing_centers(
                crossing_center, closest_line_angle
            )

            ego_line = self.find_ego_line(horiz, crossing_center, ghost_cc=ego_ghost_cc)
            opp_line = self.find_opp_line(horiz, crossing_center, ghost_cc=opp_ghost_cc)

            self.get_logger().debug(
                f"Initial ego line: {ego_line}, opp line: {opp_line}"
            )

            if ego_line is not None:
                ego_line_long = self.elongate_line(ego_line)
                clipped_ego, ego_clip_bounds = self.clip_ego_line_adaptive(
                    ego_line_long,
                    image,
                    angle=closest_line_angle,
                    min_rel_base=0.5,
                    max_rel_base=0.75,
                )

                (
                    ego_dotted,
                    ego_gap_count,
                    wr_ego,
                    _,
                ) = self.is_line_dotted_by_gap_detection(
                    clipped_ego,
                    image,
                    box_half_width=22,
                    length_extend=1.1,
                    min_gap_count=3,
                )

                angle_deviation = (
                    abs(90 - closest_line_angle)
                    if closest_line_angle is not None
                    else 0.0
                )
                is_angled_approach = angle_deviation > 15.0
                min_wr_ego_dotted = 20.0
                min_wr_ego_solid = 20.0 if is_angled_approach else 35.0
                min_wr_ego = min_wr_ego_dotted if ego_dotted else min_wr_ego_solid

                self.get_logger().debug(
                    f"EGO LINE: g={ego_gap_count} wr={wr_ego:.1f}% "
                    f"(angle_dev={angle_deviation:.1f}°, "
                    f"min_wr={min_wr_ego:.0f}%)"
                )

                (
                    ego_left_check_fail,
                    ego_line_ext_left,
                ) = (
                    self.check_line_by_horizontal_extension(
                        clipped_ego, image, direction="left"
                    )
                    if clipped_ego is not None
                    else (False, None)
                )

                ego_extension_check_passed = (
                    not ego_left_check_fail if clipped_ego is not None else False
                )

                ego_roi_and_distance_check = False
                if clipped_ego is not None and crossing_center is not None:
                    roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(
                        image.shape
                    )

                    x1, y1, x2, y2 = (
                        float(clipped_ego[0][0]),
                        float(clipped_ego[0][1]),
                        float(clipped_ego[0][2]),
                        float(clipped_ego[0][3]),
                    )

                    in_roi_x = (
                        roi_left <= x1 <= roi_right and roi_left <= x2 <= roi_right
                    )
                    y_tolerance = 10
                    y_min_tol = roi_top - y_tolerance
                    y_max_tol = roi_bottom + y_tolerance
                    in_roi_y = (y_min_tol <= y1 <= y_max_tol) and (
                        y_min_tol <= y2 <= y_max_tol
                    )

                    line_center_x = (x1 + x2) / 2.0
                    line_center_y = (y1 + y2) / 2.0
                    dist_to_center = math.sqrt(
                        (line_center_x - crossing_center[0]) ** 2
                        + (line_center_y - crossing_center[1]) ** 2
                    )
                    within_distance = dist_to_center <= 250

                    ego_roi_and_distance_check = (
                        in_roi_x and in_roi_y and within_distance
                    )

                if (
                    wr_ego >= min_wr_ego
                    and ego_extension_check_passed
                    and (ego_roi_and_distance_check)
                    and ego_gap_count <= 4
                ):
                    label_ego = (
                        f"EGO DOTTED (g={ego_gap_count} wr={wr_ego:.1f}%)"
                        if ego_dotted
                        else f"EGO SOLID (g={ego_gap_count} wr={wr_ego:.1f}%)"
                    )
                else:
                    label_ego = None
                    clipped_ego = None

                if clipped_ego is None:
                    ego_line_long = None
                else:
                    ego_line_long = clipped_ego

            if opp_line is not None:
                opp_line_long = self.elongate_line(opp_line)
                clipped_opp, opp_clip_bounds = self.clip_opp_line_adaptive(
                    opp_line_long, image, angle=closest_line_angle
                )

                (
                    opp_dotted,
                    opp_gap_count,
                    wr_opp,
                    _,
                ) = self.is_line_dotted_by_gap_detection(
                    clipped_opp,
                    image,
                    box_half_width=22,
                    length_extend=1.1,
                    min_gap_count=3,
                )

                angle_deviation = 0.0
                if closest_line_angle is not None:
                    normalized_angle = closest_line_angle
                    if normalized_angle > 90:
                        normalized_angle = 180 - normalized_angle
                    angle_deviation = abs(normalized_angle)

                is_angled_approach = angle_deviation > 15.0
                min_wr_opp = 28.0 if is_angled_approach else 35.0

                self.get_logger().debug(
                    f"OPP LINE: g={opp_gap_count} wr={wr_opp:.1f}% "
                    f"(angle_dev={angle_deviation:.1f}°, "
                    f"min_wr={min_wr_opp:.0f}%)"
                )

                q1q2 = [q1[0], q2[1], q2[2], q1[3]]
                opp_location_valid = self.is_line_in_quadrant(clipped_opp, q1q2)
                self.get_logger().debug(f"valid={opp_location_valid}")

                opp_line_extended = None
                (
                    opp_check_fail,
                    opp_line_ext_right,
                ) = (
                    self.check_line_by_horizontal_extension(
                        clipped_opp, image, direction="right"
                    )
                    if clipped_opp is not None
                    else (False, None)
                )

                opp_extension_check_passed = (
                    not opp_check_fail if clipped_opp is not None else False
                )

                if (
                    wr_opp >= min_wr_opp
                    and opp_extension_check_passed
                    and opp_location_valid
                ):
                    label_opp = (
                        f"OPP DOTTED (g={opp_gap_count} wr={wr_opp:.1f}%)"
                        if opp_dotted
                        else f"OPP SOLID (g={opp_gap_count} wr={wr_opp:.1f}%)"
                    )
                else:
                    label_opp = None
                    clipped_opp = None

            if clipped_opp is None:
                opp_line_long = None
            else:
                opp_line_long = clipped_opp

            self.get_logger().debug(
                f"Detected ego line: {label_ego}, opp line: {label_opp}"
            )

            if ego_line_long is not None and opp_line_long is not None:
                pair_plausible = self.check_plausibility_horizontal_line_pair(
                    opp_line_long, ego_line_long, crossing_center
                )

            if stop_line_right is not None and opp_line_long is not None:
                x1_o, y1_o, x2_o, y2_o = opp_line_long[0]
                opp_y = (y1_o + y2_o) / 2.0
                x1_r, y1_r, x2_r, y2_r = stop_line_right[0]
                stop_y_r = (y1_r + y2_r) / 2.0
                if stop_y_r < opp_y:
                    stop_line_right = None
                    label_stop_line_right = None

            if stop_line_left is not None and self._stop_left_y is not None:
                if ego_line_long is not None:
                    x1_e, y1_e, x2_e, y2_e = ego_line_long[0]
                    ego_y = (y1_e + y2_e) / 2.0
                    if self._stop_left_y >= ego_y:
                        stop_line_left = None
                        label_stop_line_left = None

                if stop_line_left is not None and opp_line_long is not None:
                    x1_o, y1_o, x2_o, y2_o = opp_line_long[0]
                    opp_y = (y1_o + y2_o) / 2.0
                    if self._stop_left_y <= opp_y:
                        stop_line_left = None
                        label_stop_line_left = None

        else:
            self.get_logger().debug(
                "pipeline: skipping ego/opp line search, no crossing center"
            )

        result_list = []

        def _push_entry(code, line, conf=1.0):
            if line is None:
                return
            nl = self._normalize_line(line)
            if nl is None:
                return
            x1p, y1p, x2p, y2p = nl[0].astype(float)
            try:
                code_int = int(code)
            except Exception:
                code_int = int(float(code))
            result_list.extend(
                [
                    float(code_int),
                    float(x1p),
                    float(y1p),
                    float(x2p),
                    float(y2p),
                    float(conf),
                ]
            )

        try:
            if "ego_line_long" in locals() and ego_line_long is not None:
                code = (
                    LaneType.EGO_DOTTED
                    if ("ego_dotted" in locals() and ego_dotted)
                    else LaneType.EGO_SOLID
                )
                _push_entry(code, ego_line_long, conf=1.0)
            if "opp_line_long" in locals() and opp_line_long is not None:
                code = (
                    LaneType.OPP_DOTTED
                    if ("opp_dotted" in locals() and opp_dotted)
                    else LaneType.OPP_SOLID
                )
                _push_entry(code, opp_line_long, conf=1.0)
        except Exception:
            result_list = []

        if "ego_line_long" in locals():
            ego_for_agg = ego_line_long if ego_line_long is not None else None
        else:
            ego_for_agg = None

        if "opp_line_long" in locals():
            opp_for_agg = opp_line_long if opp_line_long is not None else None
        else:
            opp_for_agg = None

        stop_left_for_agg = stop_line_left if stop_line_left is not None else None
        stop_right_for_agg = stop_line_right if stop_line_right is not None else None

        self.intersection_aggregator.add_detection(
            ego_line=ego_for_agg,
            opp_line=opp_for_agg,
            stop_line_left=stop_left_for_agg,
            stop_line_right=stop_right_for_agg,
            ego_dotted=(ego_dotted if ego_for_agg is not None else None),
            opp_dotted=(opp_dotted if opp_for_agg is not None else None),
            stop_dotted_left=(
                stop_dotted_left if stop_left_for_agg is not None else None
            ),
            stop_dotted_right=(
                stop_dotted_right if stop_right_for_agg is not None else None
            ),
            ego_angle=closest_line_angle,
            opp_angle=closest_line_angle,
        )

        crossing_type = self.intersection_aggregator.get_crossing_type()
        is_stable = self.intersection_aggregator.is_crossing_stable()
        buffer_levels = self.intersection_aggregator.get_buffer_levels()
        overall_confidence = self.intersection_aggregator.get_overall_confidence()
        stability_scores = self.intersection_aggregator.get_stability_score()

        self.get_logger().info(
            f"Aggregated crossing type: {crossing_type} | "
            f"Confidence: {overall_confidence:.2f} | "
            f"Stability: {stability_scores['overall']:.2f}"
        )

        debug_image = self.debug_visualizer.render_debug_overlays(
            image=orig_image,
            vert=vert,
            horiz=horiz,
            joined_lines=fused_lines,
            ego_line_long=ego_line_long,
            opp_line_long=opp_line_long,
            pair_plausible=pair_plausible,
            crossing_center=crossing_center,
            detected_corners=detected_corners,
            ego_ghost_cc=ego_ghost_cc,
            opp_ghost_cc=opp_ghost_cc,
            left_stop_ghost_cc=left_stop_ghost_cc,
            right_stop_ghost_cc=right_stop_ghost_cc,
            stop_line_left=stop_line_left,
            stop_line_right=stop_line_right,
            stop_line_left_ext=(
                stop_line_left_ext if "stop_line_left_ext" in locals() else None
            ),
            stop_line_right_ext=(
                stop_line_right_ext if "stop_line_right_ext" in locals() else None
            ),
            label_stop_line_left=label_stop_line_left,
            label_stop_line_right=label_stop_line_right,
            label=label_ego,
            label2=label_opp,
            closest_line_angle=closest_line_angle,
            q1=q1,
            q2=q2,
            q3=q3,
            q4=q4,
            crossing_type=crossing_type,
            is_stable=is_stable,
            buffer_levels=buffer_levels,
            overall_confidence=overall_confidence,
            enhanced_image=enhanced_image,
        )

        return debug_image, result_list


def main(args=None):
    """
    Main function to start the crossing detection node.

    Keyword Arguments:
        args -- Launch arguments (default: {None})
    """
    rclpy.init(args=args)
    node = IntersectionDetector()

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
