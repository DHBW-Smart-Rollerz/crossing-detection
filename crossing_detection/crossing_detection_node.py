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
from crossing_detection.utils.checks import (
    check_line_by_horizontal_extension,
    check_line_by_vertical_extension,
    check_line_left_y_pos,
    check_line_right_y_pos,
    check_plausibility_horizontal_line_pair,
    check_stop_line_crossing_openness,
    check_stop_line_pair_plausibility,
    is_ego_roi_and_distance_valid,
    is_left_stop_line_valid,
    is_line_in_quadrant,
    is_right_stop_line_valid,
    measure_stop_line_thickness,
)
from crossing_detection.utils.filter import (
    filter_by_angle,
    filter_by_bev_black_corner,
    filter_by_length,
    filter_by_roi,
    filter_lines_by_polygon,
)
from crossing_detection.utils.helper import normalize_line, normalize_lines
from crossing_detection.utils.models import TunableParamSet
from crossing_detection.utils.tools import (
    clip_ego_line_adaptive,
    clip_line_to_vertical_bounds,
    clip_opp_line_adaptive,
    elongate_line,
    enhance_by_line_brightness,
    find_corners_shi_tomasi,
    find_heading_angle,
    fuse_similar_lines,
    get_bev_black_corner_polygon,
    is_line_dotted_by_gap_detection,
    perform_canny,
)

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
                "state": NodeState.INACTIVE.value,
                "image_path": "resources/img/example.png",
                "debug": False,
                "compute_crossing_center": False,
                # Pipeline parameters
                "min_wr_dotted_ego": 20.0,
                "min_wr_solid_ego": 35.0,
                "min_wr_solid_ego_angled": 20.0,
                "min_wr_opp": 35.0,
                "min_wr_opp_angled": 28.0,
                "min_wr_stop_left_solid": 30,
                "min_wr_stop_left_dotted": 25,
                "min_wr_stop_right_solid": 30,
                "min_wr_stop_right_dotted": 25,
                "min_gap_count_dotted": 3,
                "clip_ego_adaptive_min_rel": 0.5,
                "clip_ego_adaptive_max_rel": 0.75,
                "clip_opp_adaptive_min_rel": 0.1,
                "clip_opp_adaptive_max_rel": 0.5,
                "allowed_corner_error_cc_rect": 30.0,
                "tolerance_angle_cc_rect": 20.0,
                "openness_black_pixel_pct_threshold": 55.0,
                "left_stop_line_min_thickness": 18,
                "right_stop_line_min_thickness": 18,
                "fuse_lines_distance_tolerance": 100,
                "bev_dead_area_corner_height_rel": 0.73,
                "bev_dead_area_corner_width_rel": 0.4,
                "heading_filter_angle_tolerance": 15.0,
                "default_cc_vertical_pos_relative": 0.5,
                "ego_valid_dist_to_cc_max": 250,
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
            self.debug = self.get_parameter("debug").value
        except Exception:
            self.debug = False

        # Configure logger level based on debug_logging parameter
        logger = self.get_logger()
        if self.debug:
            logger.set_level(rclpy.logging.LoggingSeverity.DEBUG)
        else:
            logger.set_level(rclpy.logging.LoggingSeverity.INFO)

        # Initialize tunable parameter set for use in pipeline
        self.tunable_params = TunableParamSet(self)

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
            nl = normalize_line(line)
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

    def calculate_ghost_crossing_centers(self, crossing_center, prominent_angle):
        """
        Calculate ghost crossing centers offset from main crossing center.

        Projects two points from the main crossing center along the prominent
        angle direction:
        - ego_ghost_cc: backward (opposite direction)
        - opp_ghost_cc: forward (same direction)

        This helps find ego/opp lines that are not in the middle of the road.

        Arguments:
            crossing_center -- Main crossing center (x, y)
            prominent_angle -- Angle in degrees (0-180)
            offset_distance -- How far to offset (uses parameter if None)

        Returns:
            Tuple of (ego_ghost_cc, opp_ghost_cc) as (x, y) tuples,
            or (None, None) if angle is not available
        """
        if crossing_center is None or prominent_angle is None:
            return None, None

        offset_distance = 85

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

    def calculate_stop_line_ghost_centers(self, crossing_center, prominent_angle):
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
            offset_distance -- How far to offset (uses parameter if None)

        Returns:
            Tuple of (left_ghost_cc, right_ghost_cc) as (x, y) tuples,
            or (None, None) if angle is not available
        """
        if crossing_center is None or prominent_angle is None:
            return None, None

        offset_distance = 100

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
            self.get_logger().error(f"Error calculating stop ghost CCs: {e}")
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

        print(len(horiz_lines))
        print(search_center)

        for line in horiz_lines:
            x1, y1, x2, y2 = line[0]
            line_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            if line_center[1] > search_center[1]:
                lines_candidates.append(line)

        print(len(lines_candidates))

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

        quad_lines = filter_lines_by_polygon(lines, quadrant, require_full=require_full)

        if quad_lines is None or len(quad_lines) == 0:
            return None

        vert, horiz = filter_by_angle(quad_lines, tol_deg=10)

        if vert is None or len(vert) == 0:
            return None

        vert = filter_by_length(vert, min_length=min_length)

        if vert is None or len(vert) == 0:
            return None

        best_line = max(vert, key=lambda line: self._line_length(line))
        return elongate_line(best_line, length=200)

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
            long_lines = filter_by_length(lines, min_length=min_length)

            if long_lines is None or len(long_lines) == 0:
                return None

            vert, horiz = filter_by_angle(long_lines, tol_deg=15)

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

        edges = perform_canny(image)
        transformed_lines = self.line_segment_detector(edges)
        transformed_lines = normalize_lines(transformed_lines)

        filtered_lines = filter_by_length(transformed_lines, min_length=30)
        filtered_lines = filter_by_roi(filtered_lines, image.shape)

        dead_area_bev = get_bev_black_corner_polygon(
            image.shape,
            corner_height_rel=self.tunable_params.bev_dead_area_corner_height_rel,
            corner_width_rel=self.tunable_params.bev_dead_area_corner_width_rel,
        )
        filtered_lines = filter_by_bev_black_corner(filtered_lines, dead_area_bev)

        filtered_lines = fuse_similar_lines(
            filtered_lines,
            angle_tol_deg=10,
            center_dist_tol=self.tunable_params.fuse_lines_distance_tolerance,
        )

        if filtered_lines and len(filtered_lines) > 0:
            image = enhance_by_line_brightness(
                image,
                filtered_lines,
                percentile=90,
            )

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.uint8(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(30, 30))
        image = clahe.apply(image)

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
        lines = normalize_lines(lines)
        lines = filter_by_length(lines, min_length=20)
        filtered_lines = filter_by_roi(lines, image.shape)

        dead_area_bev = get_bev_black_corner_polygon(
            image.shape,
            corner_height_rel=self.tunable_params.bev_dead_area_corner_height_rel,
            corner_width_rel=self.tunable_params.bev_dead_area_corner_width_rel,
        )
        cv2.drawContours(orig_image, [dead_area_bev], -1, (0, 255, 0), thickness=2)
        filtered_lines = filter_by_bev_black_corner(filtered_lines, dead_area_bev)

        fused_lines = fuse_similar_lines(
            filtered_lines,
            angle_tol_deg=10,
            center_dist_tol=self.tunable_params.fuse_lines_distance_tolerance,
        )

        heading_angle = None
        if fused_lines is not None and len(fused_lines) > 0:
            heading_angle, line_count = find_heading_angle(
                fused_lines, logger=self.get_logger()
            )

        fused_lines = filter_by_length(fused_lines, min_length=40)

        vert, horiz = filter_by_angle(
            fused_lines,
            anchor_angle=heading_angle,
            anchor_tolerance=self.tunable_params.heading_filter_angle_tolerance,
            tol_deg=5,
        )

        roi_bbox = self.get_roi_bbox(image.shape)
        detected_corners = find_corners_shi_tomasi(image, roi_bbox=roi_bbox)

        crossing_center = None
        if detected_corners and len(detected_corners) >= 3:
            sorted_corners, interior_angles = self.compute_corner_angles(
                detected_corners
            )

            if sorted_corners is not None and interior_angles is not None:
                corner_error = self.compute_angle_error(interior_angles)

                is_rect = self.is_valid_rectangle(
                    interior_angles,
                    angle_tolerance=self.tunable_params.tolerance_angle_cc_rect,
                )

                if (
                    corner_error < self.tunable_params.allowed_corner_error_cc_rect
                    or is_rect
                ):
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
                    int(
                        roi_top
                        + (roi_bottom - roi_top)
                        * self.tunable_params.default_cc_vertical_pos_relative
                    ),
                )
        else:
            roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)
            crossing_center = (
                int((roi_left + roi_right) / 2),
                int(roi_top + (roi_bottom - roi_top) * 0.5),
            )

        lines = vert + horiz
        lines = filter_by_length(lines, min_length=70)
        fused_lines = lines

        if crossing_center is not None:
            (
                left_stop_ghost_cc,
                right_stop_ghost_cc,
            ) = self.calculate_stop_line_ghost_centers(crossing_center, heading_angle)

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
            ) = is_line_dotted_by_gap_detection(
                stop_line_left,
                image,
                box_half_width=10,
                length_extend=1.2,
                min_gap_count=self.tunable_params.min_gap_count_dotted,
            )
            min_wr = (
                self.tunable_params.min_wr_stop_left_dotted
                if stop_dotted_left
                else self.tunable_params.min_wr_stop_left_solid
            )

            (
                gaps_ext,
                wr_ext,
                stop_line_left_ext,
            ) = check_line_by_vertical_extension(
                stop_line_left, image, self.get_logger(), is_right=False
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
            ) = is_line_dotted_by_gap_detection(
                stop_line_right,
                image,
                box_half_width=10,
                length_extend=1.2,
                min_gap_count=self.tunable_params.min_gap_count_dotted,
            )
            min_wr = (
                self.tunable_params.min_wr_stop_right_dotted
                if stop_dotted_right
                else self.tunable_params.min_wr_stop_right_solid
            )

            (
                gaps_ext,
                wr_ext,
                stop_line_right_ext,
            ) = check_line_by_vertical_extension(
                stop_line_right, image, self.get_logger(), is_right=True
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

        roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)
        roi_width = roi_right - roi_left
        min_x_inset = roi_left + roi_width * 0.1
        max_x_inset = roi_right - roi_width * 0.1
        min_x_right_stop = roi_left + roi_width * 0.6
        max_x_left_stop = roi_left + roi_width * 0.4

        if stop_line_right is not None:
            is_valid, reason = is_right_stop_line_valid(
                stop_line_right,
                crossing_center,
                min_x_inset,
                max_x_inset,
                min_x_right_stop,
            )
            if not is_valid:
                self.get_logger().debug(f"RIGHT stop rejected: {reason}")
                stop_line_right = None
                label_stop_line_right = None

        if stop_line_left is not None:
            is_valid, reason = is_left_stop_line_valid(
                stop_line_left, crossing_center, max_x_left_stop
            )
            if not is_valid:
                self.get_logger().debug(f"LEFT stop rejected: {reason}")
                stop_line_left = None
                label_stop_line_left = None
            else:
                # Store left stop line y for later use
                x1_l, y1_l, x2_l, y2_l = stop_line_left[0]
                self._stop_left_y = (y1_l + y2_l) / 2.0

        cross_right_open, cross_left_open = check_stop_line_crossing_openness(
            stop_line_right,
            stop_line_left,
            image,
            self.get_logger(),
            black_pixel_pct_threshold=self.tunable_params.openness_black_pixel_pct_threshold,
        )

        if not cross_right_open:
            self.get_logger().debug(
                "RIGHT stop rejected: lines in stop line closing area"
            )
            stop_line_right = None
            label_stop_line_right = None

        if not cross_left_open:
            self.get_logger().debug(
                "LEFT stop rejected: lines in stop line closing area too dark"
            )
            stop_line_left = None
            label_stop_line_left = None

        left_thickness, right_thickness = measure_stop_line_thickness(
            stop_line_left, stop_line_right, image, self.get_logger()
        )

        if (
            left_thickness is not None
            and left_thickness < self.tunable_params.left_stop_line_min_thickness
            and not stop_dotted_left
        ):
            self.get_logger().debug(
                f"LEFT stop rejected: thickness {left_thickness:.1f} is too thin"
            )
            stop_line_left = None
            label_stop_line_left = None

        if (
            right_thickness is not None
            and right_thickness < self.tunable_params.right_stop_line_min_thickness
            and not stop_dotted_right
        ):
            self.get_logger().debug(
                f"RIGHT stop rejected: thickness {right_thickness:.1f} is too thin"
            )
            stop_line_right = None
            label_stop_line_right = None

        stop_line_left, stop_line_right = check_stop_line_pair_plausibility(
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

        lines = filter_by_length(lines, min_length=100)

        vert, horiz = filter_by_angle(
            fused_lines,
            anchor_angle=heading_angle,
            anchor_tolerance=10.0,
            tol_deg=5,
        )

        ego_line_long = None
        opp_line_long = None
        pair_plausible = False
        label_ego = None
        label_opp = None
        ego_ghost_cc = None
        opp_ghost_cc = None
        clipped_ego = None
        clipped_opp = None

        if crossing_center is not None:
            ego_ghost_cc, opp_ghost_cc = self.calculate_ghost_crossing_centers(
                crossing_center, heading_angle
            )

            ego_line = self.find_ego_line(horiz, crossing_center, ghost_cc=ego_ghost_cc)
            opp_line = self.find_opp_line(horiz, crossing_center, ghost_cc=opp_ghost_cc)

            self.get_logger().debug(
                f"Initial ego line: {ego_line}, opp line: {opp_line}"
            )

            if ego_line is not None:
                ego_line_long = elongate_line(ego_line)
                clipped_ego, ego_clip_bounds = clip_ego_line_adaptive(
                    ego_line_long,
                    self.get_roi_bbox(image.shape),
                    angle=heading_angle,
                    min_rel_base=self.tunable_params.clip_ego_adaptive_min_rel,
                    max_rel_base=self.tunable_params.clip_ego_adaptive_max_rel,
                )
                (
                    ego_dotted,
                    ego_gap_count,
                    wr_ego,
                    _,
                ) = is_line_dotted_by_gap_detection(
                    clipped_ego,
                    image,
                    box_half_width=22,
                    length_extend=1.1,
                    min_gap_count=self.tunable_params.min_gap_count_dotted,
                )

                angle_deviation = (
                    abs(90 - heading_angle) if heading_angle is not None else 0.0
                )
                is_angled_approach = angle_deviation > 15.0
                min_wr_ego_dotted = self.tunable_params.min_wr_dotted_ego
                min_wr_ego_solid = (
                    self.tunable_params.min_wr_solid_ego
                    if not is_angled_approach
                    else self.tunable_params.min_wr_solid_ego_angled
                )
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
                    check_line_by_horizontal_extension(
                        clipped_ego,
                        image,
                        self.get_logger(),
                        direction="left",
                    )
                    if clipped_ego is not None
                    else (False, None)
                )

                ego_extension_check_passed = (
                    not ego_left_check_fail if clipped_ego is not None else False
                )

                ego_roi_and_distance_check = is_ego_roi_and_distance_valid(
                    clipped_ego,
                    crossing_center,
                    image.shape,
                    self.get_roi_bbox(image.shape),
                    valid_dist_to_cc_max=self.tunable_params.ego_valid_dist_to_cc_max,
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
                opp_line_long = elongate_line(opp_line)
                clipped_opp, opp_clip_bounds = clip_opp_line_adaptive(
                    opp_line_long,
                    self.get_roi_bbox(image.shape),
                    angle=heading_angle,
                    min_rel_base=self.tunable_params.clip_opp_adaptive_min_rel,
                    max_rel_base=self.tunable_params.clip_opp_adaptive_max_rel,
                )

                (
                    opp_dotted,
                    opp_gap_count,
                    wr_opp,
                    _,
                ) = is_line_dotted_by_gap_detection(
                    clipped_opp,
                    image,
                    box_half_width=22,
                    length_extend=1.1,
                    min_gap_count=self.tunable_params.min_gap_count_dotted,
                )

                angle_deviation = 0.0
                if heading_angle is not None:
                    normalized_angle = heading_angle
                    if normalized_angle > 90:
                        normalized_angle = 180 - normalized_angle
                    angle_deviation = abs(normalized_angle)

                is_angled_approach = angle_deviation > 15.0
                min_wr_opp = (
                    self.tunable_params.min_wr_opp_angled
                    if is_angled_approach
                    else self.tunable_params.min_wr_opp
                )

                self.get_logger().debug(
                    f"OPP LINE: g={opp_gap_count} wr={wr_opp:.1f}% "
                    f"(angle_dev={angle_deviation:.1f}°, "
                    f"min_wr={min_wr_opp:.0f}%)"
                )

                q1q2 = [q1[0], q2[1], q2[2], q1[3]]
                opp_location_valid = is_line_in_quadrant(clipped_opp, q1q2)
                self.get_logger().debug(f"valid={opp_location_valid}")

                (
                    opp_check_fail,
                    opp_line_ext_right,
                ) = (
                    check_line_by_horizontal_extension(
                        clipped_opp,
                        image,
                        self.get_logger(),
                        direction="right",
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
                pair_plausible = check_plausibility_horizontal_line_pair(
                    opp_line_long, ego_line_long, crossing_center
                )

            if stop_line_right is not None and opp_line_long is not None:
                if not check_line_right_y_pos(stop_line_right, opp_line_long):
                    stop_line_right = None
                    label_stop_line_right = None

            if stop_line_left is not None:
                if not check_line_left_y_pos(
                    stop_line_left, ego_line_long, opp_line_long
                ):
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
            nl = normalize_lines(line)
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
            ego_angle=heading_angle,
            opp_angle=heading_angle,
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
            heading_angle=heading_angle,
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
