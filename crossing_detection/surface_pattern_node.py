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
from crossing_detection.preprocess.pipe import preprocessing_pipeline
from crossing_detection.utils.filter import filter_by_length, filter_by_roi
from crossing_detection.utils.helper import normalize_lines
from crossing_detection.utils.models.tunable_param_set import TunableParamSet
from crossing_detection.utils.tools import (
    enhance_by_line_brightness,
    find_heading_angle,
    fuse_similar_lines,
)

try:
    from camera_preprocessing.transformation.coordinate_transform import (
        CoordinateTransform,
        Unit,
    )

    COORD_TRANSFORM_AVAILABLE = True
except ImportError:
    COORD_TRANSFORM_AVAILABLE = False


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


FILTERING_ROI_REL_RLTB = (0.75, 0.25, 0, 0.815)  # left, right, top, bottom


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
                "result_publisher": "/surface_pattern_detection/result",
                # Parameters
                "state": NodeState.INACTIVE.value,
                "debug": False,
                # Sharpening parameters
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
        self.tunable_params = TunableParamSet(self)
        self.cv_bridge = cv_bridge.CvBridge()
        # Configure logger level based on debug_logging parameter
        logger = self.get_logger()

        try:
            self.debug = self.get_parameter("debug").value
        except Exception:
            self.debug = False

        if self.debug:
            logger.set_level(rclpy.logging.LoggingSeverity.DEBUG)
        else:
            logger.set_level(rclpy.logging.LoggingSeverity.INFO)

        self.debug_visualizer = CrossingDebugVisualizer(node=self)

        self.intersection_aggregator = IntersectionAggregator(max_frames=7)

        if COORD_TRANSFORM_AVAILABLE:
            try:
                self.coord_transform = CoordinateTransform(debug=False)
            except Exception as e:
                self.coord_transform = None
        else:
            self.coord_transform = None

    @timer.Timer(name="image_callback", filter_strength=40)
    def image_callback(self, msg: sensor_msgs.msg.Image):
        """Executed by the ROS2 system whenever a new image is received."""
        if self.get_parameter("state").value != NodeState.ACTIVE.value:
            self.get_logger().info("Node is not active. Skipping processing.")
            msg_out = std_msgs.msg.Float32MultiArray()
            msg_out.data = []
            msg_out.layout.data_offset = 0
            self.result_publisher.publish(msg_out)
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
                if len(result_list) == 0:
                    self.get_logger().debug(
                        "No results detected, publishing empty message."
                    )
                    msg_out = std_msgs.msg.Float32MultiArray()
                    msg_out.data = []
                    msg_out.layout.data_offset = 0
                    self.result_publisher.publish(msg_out)
                else:
                    # Create Float32MultiArray message with proper layout
                    msg_out = std_msgs.msg.Float32MultiArray()
                    p1 = [result_list[0], result_list[1]]
                    p2 = [result_list[2], result_list[3]]
                    print(p1, p2)
                    p1_world = self.coord_transform.bird_to_world(p1)
                    p2_world = self.coord_transform.bird_to_world(p2)
                    msg_out.data = [
                        float(p1_world[0][0]),
                        float(p1_world[0][1]),
                        float(p2_world[0][0]),
                        float(p2_world[0][1]),
                    ]

                    # Set layout if we have results
                    # Layout: one dimension for the point data
                    msg_out.layout.dim.append(
                        std_msgs.msg.MultiArrayDimension(
                            label="data",
                            size=4,
                            stride=1,
                        )
                    )
                    self.get_logger().debug(f"Publishing result: {msg_out.data}")
                    msg_out.layout.data_offset = 0
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

        image = preprocessing_pipeline(image, tunable_params=self.tunable_params)

        lines = self.line_segment_detector(image)

        lines = normalize_lines(lines)
        lines = filter_by_length(lines, min_length=30, max_length=110)

        prominent_angle, _ = find_heading_angle(lines)

        self._draw_angle_arrow(orig_image, prominent_angle)

        xs, xe, ys, ye = self.get_roi_bbox(image.shape)
        cv2.rectangle(
            orig_image,
            (int(xs), int(ys)),
            (int(xe), int(ye)),
            (255, 0, 100),  # Dark gray
            2,
        )

        lines = filter_by_roi(lines, image.shape, roi=FILTERING_ROI_REL_RLTB)

        lines = fuse_similar_lines(lines, angle_tol_deg=8, center_dist_tol=20)

        # filter angle for 45 and 135 deg
        lines = (
            self.filter_by_diagonal_angle(lines, abs(90 - prominent_angle), tol_deg=18)
            if prominent_angle is not None
            else []
        )

        lines = fuse_similar_lines(lines, angle_tol_deg=8, center_dist_tol=40)

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

        largest_box = None
        for box in valid_boxes:
            if largest_box is None:
                largest_box = box
            else:
                largest_area = cv2.contourArea(largest_box)
                current_area = cv2.contourArea(box)
                if current_area > largest_area:
                    largest_box = box

        print(largest_box)

        top_point_box = None
        if largest_box is not None:
            top_point_box = min(largest_box, key=lambda p: p[1])
            cv2.circle(orig_image, tuple(top_point_box), 5, (255, 0, 255), -1)

            bottom_point_box = max(largest_box, key=lambda p: p[1])
            cv2.circle(orig_image, tuple(bottom_point_box), 5, (255, 0, 255), -1)

        result = []
        if top_point_box is not None:
            x1 = float(top_point_box[0])
            y1 = float(top_point_box[1])
            x2 = float(bottom_point_box[0])
            y2 = float(bottom_point_box[1])

            result = [x1, y1, x2, y2]

        return (orig_image, result)


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
