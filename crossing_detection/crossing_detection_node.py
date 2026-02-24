import math
import os
import time
from enum import IntEnum
from statistics import mode

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

DEBUG = True

# Color constants (BGR tuples)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)

# Bright colors
CYAN = (255, 255, 0)
ORANGE = (0, 128, 255)
LIME = (50, 255, 50)
PINK = (255, 0, 180)
VIOLET = (180, 50, 255)
TURQUOISE = (255, 255, 100)
GOLD = (0, 215, 255)


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


# FILTERING_ROI_REL_RLTB = (0.80, 0, 0.12, 0.45)  # left, right, top, bottom
FILTERING_ROI_REL_RLTB = (0.80, 0, 0, 0.8)  # left, right, top, bottom


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

        # Create required objects
        self.cv_bridge = cv_bridge.CvBridge()
        # read parameter into attribute for fast access
        try:
            self.compute_crossing_center = self.get_parameter(
                "compute_crossing_center"
            ).value
        except Exception:
            # fallback default
            self.compute_crossing_center = True

        # Debug overlay storage for line detection visualization
        self.debug_overlay_images = []

    @property
    def image_path(self) -> str:
        """Get the image path parameter."""
        return os.path.join(
            self.package_path,
            self.get_parameter("image_path").value,  # type: ignore
        )  # type: ignore # full path to the image

    def image_callback(self, msg: sensor_msgs.msg.Image):
        """Executed by the ROS2 system whenever a new image is received."""
        # Execute the prediction
        try:
            # try to get a BGR image first; fall back to passthrough then convert gray->BGR
            try:
                img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception:
                img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # run the pipeline on the cv2 image (returns image and result array)
            img_dbg, result_list = self.pipeline(img)

            # publish result_list as Float32MultiArray
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

    @timer.Timer(name="total", filter_strength=40)
    def execute_prediction(self, msg: sensor_msgs.msg.Image):
        """
        Execute the prediction.

        Arguments:
            msg -- The image message.
        """
        with timer.Timer(name="msg_transport", filter_strength=40):
            # The image has to be retrieved from the message
            image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="8UC1")

        res = "Example Result"

        with timer.Timer(name="publish", filter_strength=40):
            # Publish the result as the custom ROS2 message defined in this package
            try:
                msg = std_msgs.msg.Float32MultiArray(data=[0.0])
                self.result_publisher.publish(msg)
            except Exception:
                pass

        if self._debug:
            with timer.Timer(name="debug", filter_strength=40):
                # Create the debug image. Here it is just the image filled with
                # the example value.
                debug_image = image.copy()
                debug_image.fill(self.example_value)

                self.debug_image_publisher.publish(  # type: ignore
                    self.cv_bridge.cv2_to_imgmsg(image, encoding="8UC1")
                )

    # instrumentation printing removed to avoid console noise in production

    @staticmethod
    def load_img_grayscale(img_path: str) -> np.ndarray:
        """
        Load an image from a file and convert it to grayscale.

        Arguments:
            img_path -- Path to the image file.

        Returns:
            Grayscale image as a numpy ndarray.
        """
        img = cv2.imread(img_path)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _render_debug_overlays(
        self,
        image,
        transformed_lines=None,
        filtered_lines=None,
        vert=None,
        horiz=None,
        joined_lines=None,
        crossing_center=None,
        cone_left=None,
        cone_right=None,
        cl_vert=None,
        cl_vert_left=None,
        ego_line_long=None,
        opp_line_long=None,
        stop_line_left=None,
        stop_line_right=None,
        pair_plausible=False,
        label=None,
        label2=None,
    ):
        """
        Draw all debug overlays at once on a provided image.
        This keeps the detection logic separated from visualization.
        """
        # draw basic line sets
        if transformed_lines is not None:
            image = self._draw_lines(image, transformed_lines, color=MAGENTA)

        if vert is not None:
            image = self._draw_lines(image, vert, color=GOLD)

        # draw joined lines in yellow
        if joined_lines is not None:
            image = self._draw_lines(image, joined_lines, color=YELLOW, thickness=3)

        # crossing center markers
        if crossing_center is not None:
            try:
                image = cv2.circle(image, crossing_center, 8, YELLOW)
                crossing_center_pulled = self.pull_point_to_roi_center(
                    crossing_center, image.shape
                )
                image = cv2.circle(
                    image,
                    (int(crossing_center_pulled[0]), int(crossing_center_pulled[1])),
                    8,
                    TURQUOISE,
                )
            except Exception:
                pass

        # cones and small vertical clusters
        try:
            if cone_right is not None:
                self._draw_cone(cone_right, image)
            if cone_left is not None:
                self._draw_cone(cone_left, image)
            if cl_vert is not None:
                image = self._draw_lines(image, cl_vert)
            if cl_vert_left is not None:
                image = self._draw_lines(image, cl_vert_left)
        except Exception:
            pass

        # ego/opp lines and labels
        try:
            if ego_line_long is not None:
                image = self._draw_lines(
                    image,
                    [ego_line_long],
                    color=LIME,
                    thickness=6,
                )
            if opp_line_long is not None:
                image = self._draw_lines(
                    image,
                    [opp_line_long],
                    color=LIME if pair_plausible else ORANGE,
                    thickness=6,
                )
        except Exception:
            pass

        # stop lines (orthogonal to ego lane)
        try:
            if stop_line_right is not None:
                image = self._draw_lines(
                    image,
                    [stop_line_right],
                    color=VIOLET,
                    thickness=4,
                )
            if stop_line_left is not None:
                image = self._draw_lines(
                    image,
                    [stop_line_left],
                    color=VIOLET,
                    thickness=4,
                )
        except Exception:
            pass

        # put labels if provided
        try:
            if label is not None:
                cv2.putText(
                    image,
                    label,
                    (0, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    GREEN,
                )
            if label2 is not None:
                cv2.putText(
                    image,
                    label2,
                    (0, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    GREEN,
                )
        except Exception:
            pass

        # draw roi box and legend
        try:
            roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)
            cv2.rectangle(image, (roi_left, roi_top), (roi_right, roi_bottom), RED, 2)
            image = self._draw_legend(
                image,
                [
                    ("Transformed lines", MAGENTA),
                    ("Vertical lines", GOLD),
                    ("Ego line", LIME),
                    ("Opp plausible", LIME),
                    ("Opp implausible", ORANGE),
                    ("Stop lines", VIOLET),
                    ("Crossing center", YELLOW),
                    ("Center (ROI)", TURQUOISE),
                ],
            )
        except Exception:
            pass

        return image

    @staticmethod
    def show_image(img, title: str = "Graphic"):
        """
        Show an image using OpenCV.

        Arguments:
            img -- Image to be displayed.

        Keyword Arguments:
            title -- (default: {"Graphic"})
        """
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def perform_canny(self, img):
        """
        Perform Canny edge detection on the image.

        Arguments:
            img -- Input image.

        Returns:
            Image with edges detected.
        """
        img = cv2.Canny(img, 50, 50)
        # IntersectionDetector.show_image("Canny", img)
        return img

    def _blur_roi_top(
        self, image, ksize=(15, 15), sigmaX=0, do_close=True, close_kernel=(20, 3)
    ):
        """
        Apply a Gaussian blur to the upper half of the configured ROI.

        This helps the LSD detect distorted bird's-eye parts by smoothing
        high-frequency noise in the top ROI area while preserving the rest
        of the image.

        Arguments:
            image -- BGR image (numpy array)
            ksize -- Gaussian kernel size (must be odd numbers)
            sigmaX -- Gaussian sigma in X direction

        Returns:
            A copy of the image with the ROI top half blurred.
        """
        if image is None:
            return image

        h, w = image.shape[:2]
        roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)

        # clamp coords
        roi_left = max(0, min(w - 1, int(roi_left)))
        roi_right = max(0, min(w, int(roi_right)))
        roi_top = max(0, min(h - 1, int(roi_top)))
        roi_bottom = max(0, min(h, int(roi_bottom)))

        # top half of ROI
        # top_half_end = roi_top + (roi_bottom - roi_top) // 2
        top_half_end = roi_top + roi_bottom

        # ensure valid box
        top = max(0, roi_top)
        bottom = max(top, min(h, top_half_end))
        left = max(0, roi_left)
        right = max(0, min(w, roi_right))

        out = image.copy()
        if bottom > top and right > left:
            patch = out[top:bottom, left:right]
            # ensure odd kernel sizes for Gaussian
            kx, ky = ksize
            if kx % 2 == 0:
                kx += 1
            if ky % 2 == 0:
                ky += 1
            blurred = cv2.GaussianBlur(patch, (kx, ky), sigmaX)

            # optionally perform morphological closing to close small horizontal gaps
            if do_close:
                try:
                    ckx, cky = int(close_kernel[0]), int(close_kernel[1])
                except Exception:
                    ckx, cky = 20, 3
                # ensure kernel dimensions >=1
                ckx = max(1, ckx)
                cky = max(1, cky)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ckx, cky))
                processed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
                out[top:bottom, left:right] = processed
            else:
                out[top:bottom, left:right] = blurred

        return out

    def _blur_roi_bottom(
        self, image, ksize=(5, 5), sigmaX=0, do_close=True, close_kernel=(10, 3)
    ):
        """
        Apply a lighter Gaussian blur to the lower half of the configured ROI.

        This helps reduce noise in the lower ROI area while preserving
        edge details better than the top blur.

        Arguments:
            image -- BGR image (numpy array)
            ksize -- Gaussian kernel size (must be odd numbers)
            sigmaX -- Gaussian sigma in X direction
            do_close -- whether to perform morphological closing
            close_kernel -- kernel size for morphological closing

        Returns:
            A copy of the image with the ROI bottom half lightly blurred.
        """
        if image is None:
            return image

        h, w = image.shape[:2]
        roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)

        # clamp coords
        roi_left = max(0, min(w - 1, int(roi_left)))
        roi_right = max(0, min(w, int(roi_right)))
        roi_top = max(0, min(h - 1, int(roi_top)))
        roi_bottom = max(0, min(h, int(roi_bottom)))

        # bottom half of ROI
        bottom_half_start = roi_top + (roi_bottom - roi_top) // 2

        # ensure valid box
        top = max(0, bottom_half_start)
        bottom = max(top, min(h, roi_bottom))
        left = max(0, roi_left)
        right = max(0, min(w, roi_right))

        out = image.copy()
        if bottom > top and right > left:
            patch = out[top:bottom, left:right]
            # ensure odd kernel sizes for Gaussian
            kx, ky = ksize
            if kx % 2 == 0:
                kx += 1
            if ky % 2 == 0:
                ky += 1
            blurred = cv2.GaussianBlur(patch, (kx, ky), sigmaX)

            # optionally perform morphological closing
            if do_close:
                try:
                    ckx, cky = int(close_kernel[0]), int(close_kernel[1])
                except Exception:
                    ckx, cky = 10, 3
                # ensure kernel dimensions >=1
                ckx = max(1, ckx)
                cky = max(1, cky)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ckx, cky))
                processed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
                out[top:bottom, left:right] = processed
            else:
                out[top:bottom, left:right] = blurred

        return out

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
        # be explicit when checking collections: lines may be a numpy array
        if lines is None or (hasattr(lines, "__len__") and len(lines) == 0):
            return img2
        for line in lines:
            # normalize different possible representations
            nl = self._normalize_line(line)
            if nl is None:
                continue
            x1, y1, x2, y2 = nl[0]
            cv2.line(img2, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        return img2

    def hough_transformation(self, img, img_edges):
        """
        Perform Hough Transformation to detect lines in the image.

        Arguments:
            img -- Input image.
            img_edges -- Image with edges detected.

        Returns:
            List of detected lines as pairs of points.
        """
        img2 = img[::]
        transformed = []
        # lines = cv2.HoughLines(img_edges,1,np.pi/180,200,min_theta)
        lines = cv2.HoughLinesP(
            img_edges, 1, np.pi / 180, 30, minLineLength=90, maxLineGap=10
        )
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img2, (x1, y1), (x2, y2), (0, 50, 255), 2)
            transformed.append(((x1, y1), (x2, y2)))
        # cv2.imwrite("houghlines.jpg", img2)
        IntersectionDetector.show_image(img2, "houghlines")
        return transformed

    def filter_by_angle(self, lines, tol_deg: float = 25.0, debug=True):
        """
        Filter lines based on their angle.

        Arguments:
            hough_lines -- List of lines as pairs of points.

        Returns:
            Filtered list of lines.
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

            # angle in degrees in range (-180, 180]
            angle = math.degrees(math.atan2(dy, dx))
            # normalize to [0, 180)
            angle_norm = (angle + 360.0) % 180.0

            # distance to horizontal (0 or 180) and to vertical (90)
            dist_h = min(abs(angle_norm - 0.0), abs(angle_norm - 180.0))
            dist_v = min(abs(angle_norm - 80.0), abs(angle_norm - 100.0))

            # classify by the nearer of the two, but require within tolerance
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

        # Normalize and extract line data
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
            # Normalize angle to [0, 180)
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

        # Greedy clustering: start with each unused line
        for i, line_i in enumerate(line_data):
            if line_i["used"]:
                continue

            # Start a new cluster with this line
            cluster = [line_i]
            line_i["used"] = True

            # Find all nearby lines
            for j in range(i + 1, len(line_data)):
                line_j = line_data[j]
                if line_j["used"]:
                    continue

                # Check if angles are similar
                angle_diff = abs(line_i["angle"] - line_j["angle"])
                angle_diff = min(angle_diff, 180.0 - angle_diff)
                if angle_diff > angle_tolerance:
                    continue

                # Check if any endpoint of line_j is close to any endpoint
                # of any line in the cluster
                is_close = False
                for line_c in cluster:
                    # Distance from j's endpoints to i's endpoints
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

            # Fuse cluster into single line
            if cluster:
                all_x = []
                all_y = []
                for line_c in cluster:
                    all_x.extend([line_c["x1"], line_c["x2"]])
                    all_y.extend([line_c["y1"], line_c["y2"]])

                # Find extreme points
                # Project all points onto the line direction
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

    def _draw_legend(self, image, items, box_size=18, padding=8):
        """
        Draw a small legend on the image at the bottom-right.

        items: list of (label, bgr_color)
        """
        img = image
        height, width = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # compute width/height for background
        line_height = int(box_size * 1.4)
        legend_height = line_height * len(items) + padding

        # compute widest label to size background
        max_label_width = 0
        for label, _ in items:
            (w, _), _ = cv2.getTextSize(label, font, font_scale, thickness)
            if w > max_label_width:
                max_label_width = w
        legend_width = box_size + 6 + max_label_width + padding * 2

        # origin at bottom-right with 10px margin
        margin = 10
        origin_x = width - margin - legend_width
        origin_y = height - margin - legend_height

        # prevent drawing outside image (fallback to top-right if too large)
        if origin_y < 0:
            origin_y = margin

        bg_tl = (origin_x, origin_y)
        bg_br = (origin_x + legend_width, origin_y + legend_height)

        # background rectangle (semi-opaque look using filled dark rectangle)
        cv2.rectangle(img, bg_tl, bg_br, (30, 30, 30), -1)  # dark background
        # border
        cv2.rectangle(img, bg_tl, bg_br, (200, 200, 200), 1)

        # draw each item
        offset_y = origin_y + int(padding / 2) + box_size // 2
        for label, color in items:
            # color box
            box_tl = (origin_x + padding, offset_y - box_size // 2)
            box_br = (box_tl[0] + box_size, box_tl[1] + box_size)
            cv2.rectangle(img, box_tl, box_br, color, -1)
            cv2.rectangle(img, box_tl, box_br, (0, 0, 0), 1)

            # text next to box
            text_org = (box_br[0] + 6, offset_y + box_size // 4)
            cv2.putText(
                img,
                label,
                text_org,
                font,
                font_scale,
                (230, 230, 230),
                thickness,
                cv2.LINE_AA,
            )

            offset_y += line_height

        return img

    def detect_lane(self, lines, image):
        """
        Get the nearest line from the list of lines.

        Arguments:
            lines -- List of lines as pairs of points.

        Returns:
            Nearest line.
        """
        width = image.shape[1]
        min_distance = 10000
        nearest_line_right = None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            distance_to_left, _ = line_center
            distance_to_left = abs(distance_to_left - width / 2)
            if distance_to_left < min_distance:
                min_distance = distance_to_left
                nearest_line_right = line

        # mirror the right lane with center of image to get left lane
        if nearest_line_right is not None:
            x1, y1, x2, y2 = nearest_line_right[0]
            line_top_right = max(x1, x2)
            diff_to_center = abs(line_top_right - width / 2)
            new_x1 = width - x1 - diff_to_center * 2
            new_x2 = width - x2 - diff_to_center * 2
            nearest_line_left = [[new_x1, y1, new_x2, y2]]

        return nearest_line_left, nearest_line_right

    def get_nearest_line(self, lines):
        """
        Get the nearest line from the list of lines.

        Arguments:
            lines -- List of lines as pairs of points.

        Returns:
            Nearest line.
        """
        max_distance = 0
        nearest_line = None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            _, distance_to_top = line_center
            if distance_to_top > max_distance:
                max_distance = distance_to_top
                nearest_line = line

        return nearest_line

    def _normalize_line(self, line):
        """
        Normalize a single line representation into a numpy array of.
        shape (1, 4). Accepts formats: ndarray (1,4) or (4,), nested
        lists [[x1,y1,x2,y2]] or tuples. Returns None for malformed
        entries.
        """
        if line is None:
            return None
        # numpy array
        if isinstance(line, np.ndarray):
            arr = line.squeeze()
            if arr.ndim == 1 and arr.size >= 4:
                return arr.reshape(1, -1)[:, :4].astype(np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 4:
                return arr.reshape(1, -1)[:, :4].astype(np.float32)
            return None

        # list/tuple
        if isinstance(line, (list, tuple)):
            s = line
            # flatten nested one-element lists e.g. [[x1,y1,x2,y2]]
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
            # start new group
            group_idx = [i]
            visited[i] = True
            for j in range(i + 1, n):
                if visited[j]:
                    continue
                # angle difference (circular around 180)
                diff = abs(angles[i] - angles[j])
                diff = min(diff, 180.0 - diff)
                if diff <= angle_tol_deg:
                    # center distance
                    d = float(np.hypot(*(centers[i] - centers[j])))
                    if d <= center_dist_tol:
                        group_idx.append(j)
                        visited[j] = True

            # optionally discard tiny groups
            if len(group_idx) < require_min_lines:
                # keep originals for small groups
                for idx in group_idx:
                    fused.append(normalized[idx])
                continue

            # collect endpoints of group
            pts = []
            for idx in group_idx:
                x1, y1, x2, y2 = normalized[idx][0]
                pts.append([x1, y1])
                pts.append([x2, y2])
            pts = np.array(pts, dtype=np.float32)

            if pts.shape[0] < 2:
                # fallback: push single line
                fused.append(normalized[group_idx[0]])
                continue

            # PCA via SVD
            mean = pts.mean(axis=0)
            U, S, Vt = np.linalg.svd(pts - mean)
            axis = Vt[0]

            # project and find extremes
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

        # convert image to grayscale
        gray = image
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # prepare sampling positions
        t_values = np.arange(0.0, 1.0 + 1e-6, step)
        valid_samples = 0
        white_samples = 0

        # helper to sample patches with vertical offset direction
        def _sample_patch(cx, cy, direction="above"):
            if direction == "above":
                top = cy - sample_height
                bottom = cy
            else:
                top = cy
                bottom = cy + sample_height

            left = cx - sample_width // 2
            right = cx + (sample_width - sample_width // 2)

            # clip to image bounds
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
            return float(np.mean(patch > white_pixel_thresh))

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

        # If above fraction is already decisive, return
        if frac_above >= white_patch_ratio:
            return True, frac_above, valid_samples

        # If fraction is very low (< below_check_thresh) check below the line
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
                # use the larger fraction for final decision
                frac = max(frac_above, frac_below)
            else:
                frac_below = 0.0

        is_solid = frac >= white_patch_ratio
        return bool(is_solid), frac, valid_samples

    def is_line_dotted_by_gap_detection(
        self,
        line,
        image,
        box_half_width: int = 22,
        length_extend: float = 1.2,
        white_pixel_thresh: int = 180,
        min_gap_count: int = 2,
        gap_size_min: int = 3,
    ):
        """
        Gap-based dotted/solid detection.

        Procedure:
        - Extract rotated box around the line.
        - Binarize the box (white pixels = line pixels).
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
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0

        crop_w = int(max(10, line_len * float(length_extend))) + int(box_half_width * 2)
        crop_h = int(max(3, box_half_width * 2))
        h, w = image.shape[:2]

        M = cv2.getRotationMatrix2D((mid_x, mid_y), -angle, 1.0)
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
            # binarize: white pixels (line) vs black (background)
            binary = gray > white_pixel_thresh

            # compute horizontal profile (white pixel count per column)
            white_per_col = np.sum(binary, axis=0)  # white pixels per column

            # find segments of white pixels (gaps are where white_per_col == 0)
            gaps = 0
            in_gap = False
            gap_length = 0

            for i in range(len(white_per_col)):
                if white_per_col[i] == 0:  # black column (gap)
                    if not in_gap:
                        in_gap = True
                        gap_length = 1
                    else:
                        gap_length += 1
                else:  # white column (line)
                    if in_gap and gap_length >= gap_size_min:
                        gaps += 1
                    in_gap = False
                    gap_length = 0

            # check last gap if we end in a gap
            if in_gap and gap_length >= gap_size_min:
                gaps += 1

            # white ratio: percentage of white pixels in the bounding box
            # white_per_col is array of white pixel counts per column
            # crop_h is the height of the box
            # total white pixels = sum of white_per_col
            # total pixels in box = crop_h * len(white_per_col)
            total_white_pixels = float(np.sum(white_per_col))
            total_pixels = float(crop_h * len(white_per_col))
            if total_pixels > 0:
                white_ratio = (total_white_pixels / total_pixels) * 100.0
            else:
                white_ratio = 0.0

            is_dotted = gaps >= min_gap_count

            return bool(is_dotted), int(gaps), float(white_ratio), 1
        except Exception:
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
        # vertical segment
        if abs(dx) < 1e-3:
            if x1 < min_x or x1 > max_x:
                return None
            return np.array([[x1, y1, x2, y2]], dtype=np.float32)

        # compute param t where x = min_x and x = max_x
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
        Find center of crossing.

        Arguments:
            intersection_points -- desc

        Returns:
            returns
        """
        if len(intersection_points) == 0:
            return None
        points = np.array(intersection_points)
        clustering = DBSCAN(eps=20, min_samples=5).fit(points)
        labels = clustering.labels_
        largest_cluster = mode(labels[labels != -1])
        intersection_center = points[labels == largest_cluster].mean(axis=0)
        return (int(intersection_center[0]), int(intersection_center[1]))

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

    def find_opp_line(self, horiz_lines, crossing_center):
        """
        Find the ego line from the list of lines.

        Arguments:
            lines -- List of lines as pairs of points.
        """
        max_distance = 10000
        nearest_line = None

        line_candidates = []

        for line in horiz_lines:
            x1, y1, x2, y2 = line[0]
            line_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            if (
                line_center[1] < crossing_center[1]
                and line_center[0] < crossing_center[0]
            ):
                line_candidates.append(line)

        for line in line_candidates:
            x1, y1, x2, y2 = line[0]
            line_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            distance_to_crossing_center = math.sqrt(
                (line_center[0] - crossing_center[0]) ** 2
                + (line_center[1] - crossing_center[1]) ** 2
            )

            if distance_to_crossing_center < max_distance:
                max_distance = distance_to_crossing_center
                nearest_line = line

        # return none if line smaller than 80
        if nearest_line is not None:
            x1, y1, x2, y2 = nearest_line[0]
            if math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) < 90:
                return None
        return nearest_line

    def find_ego_line(self, horiz_lines, crossing_center):
        """
        Find the ego line from the list of lines.

        Arguments:
            lines -- List of lines as pairs of points.
        """
        max_distance = 10000
        nearest_line = None

        lines_candidates = []

        for line in horiz_lines:
            x1, y1, x2, y2 = line[0]
            line_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            if line_center[1] > crossing_center[1]:
                lines_candidates.append(line)

        for line in lines_candidates:
            x1, y1, x2, y2 = line[0]
            line_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            distance_to_crossing_center = math.sqrt(
                (line_center[0] - crossing_center[0]) ** 2
                + (line_center[1] - crossing_center[1]) ** 2
            )

            if distance_to_crossing_center < max_distance:
                max_distance = distance_to_crossing_center
                nearest_line = line

        if nearest_line is not None:
            x1, y1, x2, y2 = nearest_line[0]
            if math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) < 90:
                return None
        return nearest_line

    def check_plausibility_horizontal_line_pair(
        self,
        opp_line,
        ego_line,
        intersection_point,
        line_horizontal_distance_threshold: float = 100.0,
        line_vertical_distance_threshold: float = 80.0,
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
        # if distance_between_lines < negative_line_overlap_threshold:
        #    return False
        if distance_between_lines_horizontal > line_horizontal_distance_threshold:
            return False

        distance_between_lines_vertical = abs(((y1_1 + y2_1) / 2) - ((y1_2 + y2_2) / 2))
        if distance_between_lines_vertical < line_vertical_distance_threshold:
            return False

        # distance_to_center = abs(
        #    (ego_line_leftmost + opp_line_rightmost) / 2 - intersection_point[0]
        # )
        # if distance_to_center > center_horizontal_distance_threshold:
        #    return False

        return True

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

        # ROI dimensions
        roi_width = xe - xs
        roi_height = ye - ys
        roi_center_y = ys + roi_height / 2.0

        # Cat's eye left (at ~0.25 of ROI width) for left stop line
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

        # ROI center point
        x_center = (xs + xe) / 2.0
        y_center = (ys + ye) / 2.0

        # Q1: Top-left quadrant
        q1 = [
            (int(xs), int(ys)),  # top-left corner
            (int(x_center), int(ys)),  # top-center
            (int(x_center), int(y_center)),  # center
            (int(xs), int(y_center)),  # left-center
        ]

        # Q2: Top-right quadrant
        q2 = [
            (int(x_center), int(ys)),  # top-center
            (int(xe), int(ys)),  # top-right corner
            (int(xe), int(y_center)),  # right-center
            (int(x_center), int(y_center)),  # center
        ]

        # Q3: Bottom-left quadrant
        q3 = [
            (int(xs), int(y_center)),  # left-center
            (int(x_center), int(y_center)),  # center
            (int(x_center), int(ye)),  # bottom-center
            (int(xs), int(ye)),  # bottom-left corner
        ]

        # Q4: Bottom-right quadrant
        q4 = [
            (int(x_center), int(y_center)),  # center
            (int(xe), int(y_center)),  # right-center
            (int(xe), int(ye)),  # bottom-right corner
            (int(x_center), int(ye)),  # bottom-center
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

    def find_line_in_quadrant(self, lines, quadrant, min_length: float = 30.0):
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

        # Filter lines inside quadrant
        quad_lines = self.filter_lines_by_polygon(lines, quadrant, require_full=False)

        if quad_lines is None or len(quad_lines) == 0:
            return None

        # Use filter_by_angle to separate vertical and horizontal
        vert, horiz = self.filter_by_angle(quad_lines, tol_deg=10)

        # We want the vertical lines
        if vert is None or len(vert) == 0:
            return None

        # Filter by minimum length
        vert = self.filter_by_length(vert, min_length=min_length)

        if vert is None or len(vert) == 0:
            return None

        # Return the longest vertical line
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

        # Filter lines inside cone
        cone_lines = self.filter_lines_by_cone(lines, cone, require_full=True)

        if cone_lines is None or len(cone_lines) == 0:
            return None

        # Use existing filter_by_angle to separate vertical and horizontal
        vert, horiz = self.filter_by_angle(cone_lines, tol_deg=10)

        # We want the vertical lines (perpendicular to ego lane)
        if vert is None or len(vert) == 0:
            return None

        # Filter by minimum length
        vert = self.filter_by_length(vert, min_length=min_length)

        if vert is None or len(vert) == 0:
            return None

        # Return the longest vertical line (most likely the stop line)
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

        # Filter lines inside cat_eye polygon
        cat_eye_lines = self.filter_lines_by_polygon(lines, cat_eye, require_full=True)

        if cat_eye_lines is None or len(cat_eye_lines) == 0:
            return None

        # Use existing filter_by_angle to separate vertical and horizontal
        vert, horiz = self.filter_by_angle(cat_eye_lines, tol_deg=10)

        # We want the vertical lines (perpendicular to ego lane)
        if vert is None or len(vert) == 0:
            return None

        # Filter by minimum length
        vert = self.filter_by_length(vert, min_length=min_length)

        if vert is None or len(vert) == 0:
            return None

        # Return the longest vertical line (most likely the stop line)
        best_line = max(vert, key=lambda line: self._line_length(line))
        return self.elongate_line(best_line, length=180)

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

        # Draw the diamond: top -> right -> bottom -> left -> top
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

        # Expect cone as [cone_start_point, cone_arm_left_point, cone_arm_right_point]
        try:
            cone_start, cone_left, cone_right = cone
        except Exception:
            # invalid cone format
            return []

        # polygon: left-top, right-top, bottom (triangle)
        poly = np.array([cone_left, cone_right, cone_start], dtype=np.int32)

        filtered = []
        for ln in lines:
            nl = self._normalize_line(ln)
            if nl is None:
                continue
            x1, y1, x2, y2 = nl[0].astype(int)

            # pointPolygonTest returns >0 inside, 0 on edge, <0 outside
            d1 = cv2.pointPolygonTest(poly, (int(x1), int(y1)), False)
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

        # Convert polygon points to numpy array for cv2.pointPolygonTest
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

            # pointPolygonTest returns >0 inside, 0 on edge, <0 outside
            d1 = cv2.pointPolygonTest(poly, (int(x1), int(y1)), False)
            d2 = cv2.pointPolygonTest(poly, (int(x2), int(y2)), False)

            if require_full:
                if d1 >= 0 and d2 >= 0:
                    filtered.append(nl)
            else:
                if d1 >= 0 or d2 >= 0:
                    filtered.append(nl)

        return filtered

    def pipeline(self, image):
        """
        Complete processing pipeline for intersection detection.

        Arguments:
            img_path -- Path to the input image.
        """
        # image = IntersectionDetector.load_img_grayscale(img_path)
        # image = self.crop_image(image)

        # keep a copy of the original image for rendering overlays later
        orig_image = image.copy()

        # blur + optional closing on the upper half of the ROI to help LSD
        # detect distorted bird's-eye parts
        image = self._blur_roi_top(
            image, ksize=(22, 22), sigmaX=0, do_close=True, close_kernel=(25, 3)
        )

        # apply lighter blur to the lower half of the ROI
        image = self._blur_roi_bottom(
            image, ksize=(5, 5), sigmaX=0, do_close=True, close_kernel=(10, 3)
        )

        q1, q2, q3, q4 = self.calculate_roi_quadrants(image)
        edges = self.perform_canny(image)
        transformed_lines = self.line_segment_detector(edges)
        # normalize detected lines to canonical numpy (1,4) arrays
        transformed_lines = self._normalize_lines(transformed_lines)

        filtered_lines = self.filter_by_length(transformed_lines, min_length=20)
        # image = self._draw_lines(image, filtered_lines, color=BLUE)
        # image = self._draw_lines(image, filtered_lines, color=YELLOW)
        filtered_lines = self.filter_by_roi(filtered_lines, image.shape)
        # image = self._draw_lines(image, filtered_lines, color=GREEN)

        # Fuse similar lines that belong to the same stop line
        fused_lines = self.fuse_similar_lines(
            filtered_lines, angle_tol_deg=15, center_dist_tol=100
        )

        fused_lines = self.fuse_similar_lines(
            fused_lines, angle_tol_deg=5, center_dist_tol=120
        )

        fused_lines = self.fuse_similar_lines(
            fused_lines, angle_tol_deg=5, center_dist_tol=25
        )

        vert, horiz = self.filter_by_angle(fused_lines, tol_deg=5)

        # compute crossing center optionally; if disabled, use ROI center
        crossing_center = None
        if getattr(self, "compute_crossing_center", True):
            try:
                intersections = self.find_intersections(vert, horiz)
            except Exception as e:
                self.get_logger().error(f"find_intersections error: {e}")
                intersections = []
            try:
                crossing_center = self.find_crossing_center(intersections)
            except Exception as e:
                self.get_logger().error(f"find_crossing_center error: {e}")
                crossing_center = None
        else:
            # use ROI center as crossing center when computation is disabled
            roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)
            crossing_center = (
                int((roi_left + roi_right) / 2),
                int((roi_top + roi_bottom) / 2),
            )

        lines = vert + horiz
        lines = self.filter_by_length(lines, min_length=70)
        fused_lines = lines
        # save short lines for stop line detection in quadrants

        # Detect stop lines using ROI quadrants (Q1 for left, Q3 for right)
        stop_line_left = self.find_line_in_quadrant(lines, q1)
        stop_line_right = self.find_line_in_quadrant(lines, q2)

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
                box_half_width=22,
                length_extend=1.2,
                min_gap_count=3,
            )
            # Validate: Check if dotted or solid with appropriate thresholds
            # Dotted: wr >= 10%, Solid: wr >= 22% (percentage-based)
            min_wr = 5.5 if stop_dotted_left else 20
            if white_ratio_left >= min_wr:
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
                box_half_width=22,
                length_extend=1.2,
                min_gap_count=3,
            )
            # Validate: Check if dotted or solid with appropriate thresholds
            # Dotted: wr >= 10%, Solid: wr >= 22% (percentage-based)
            min_wr = 5.5 if stop_dotted_right else 20.0
            if white_ratio_right >= min_wr:
                label_stop_line_right = (
                    f"STOP_RIGHT DOTTED (g={gap_count_right} wr={white_ratio_right:.1f}%)"
                    if stop_dotted_right
                    else f"STOP_RIGHT SOLID (g={gap_count_right} wr={white_ratio_right:.1f}%)"
                )
            else:
                stop_line_right = None

        lines = self.filter_by_length(lines, min_length=100)

        vert, horiz = self.filter_by_angle(lines)

        # Initialize variables for ego/opp lines
        ego_line_long = None
        opp_line_long = None
        pair_plausible = False
        label_ego = None
        label_opp = None

        # Only attempt to find ego/opp lines if we have a valid crossing center.
        if crossing_center is not None:
            ego_line = self.find_ego_line(horiz, crossing_center)
            opp_line = self.find_opp_line(horiz, crossing_center)

            if ego_line is not None:
                # Elongate and clip ego line to ROI band
                ego_line_long = self.elongate_line(ego_line)
                clipped_ego = self.clip_line_to_vertical_bounds(
                    ego_line_long, image, min_rel=0.5, max_rel=0.75
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
                # Validate: Check if dotted or solid with appropriate thresholds
                # Dotted: wr >= 10%, Solid: wr >= 22% (percentage-based)
                min_wr = 10.0 if ego_dotted else 22.0
                if wr_ego >= min_wr:
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
                # Elongate and clip opposite line to ROI band
                opp_line_long = self.elongate_line(opp_line)
                clipped_opp = self.clip_line_to_vertical_bounds(
                    opp_line_long, image, min_rel=0.15, max_rel=0.4
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
                # Validate: Check if dotted or solid with appropriate thresholds
                # Dotted: wr >= 10%, Solid: wr >= 22% (percentage-based)
                min_wr = 10.0 if opp_dotted else 22.0
                if wr_opp >= min_wr:
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

            if ego_line_long is not None and opp_line_long is not None:
                pair_plausible = self.check_plausibility_horizontal_line_pair(
                    opp_line_long, ego_line_long, crossing_center
                )

        else:
            # no crossing center found for this frame; skip ego/opp identification
            self.get_logger().debug(
                "pipeline: skipping ego/opp line search, no crossing center"
            )

        # build result array for publisher: each record = [type_code, x1, y1, x2, y2, confidence]
        # type_code: 1=ego_solid, 2=ego_dotted, 3=opp_solid, 4=opp_dotted
        result_list = []

        def _push_entry(code, line, conf=1.0):
            if line is None:
                return
            nl = self._normalize_line(line)
            if nl is None:
                return
            x1p, y1p, x2p, y2p = nl[0].astype(float)
            # ensure enum values are converted to their integer codes
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
            # ego
            if "ego_line_long" in locals() and ego_line_long is not None:
                code = (
                    LaneType.EGO_DOTTED
                    if ("ego_dotted" in locals() and ego_dotted)
                    else LaneType.EGO_SOLID
                )
                _push_entry(code, ego_line_long, conf=1.0)
            # opp
            if "opp_line_long" in locals() and opp_line_long is not None:
                code = (
                    LaneType.OPP_DOTTED
                    if ("opp_dotted" in locals() and opp_dotted)
                    else LaneType.OPP_SOLID
                )
                _push_entry(code, opp_line_long, conf=1.0)
        except Exception:
            # on any error, leave result_list empty
            result_list = []

        # NOW: Render all debug overlays at the end, after detection is complete
        debug_image = self._render_debug_overlays(
            orig_image,
            transformed_lines=transformed_lines,
            filtered_lines=None,
            vert=vert,
            horiz=None,
            joined_lines=fused_lines,
            crossing_center=crossing_center,
            cone_left=None,
            cone_right=None,
            cl_vert=None,
            cl_vert_left=None,
            ego_line_long=ego_line_long,
            opp_line_long=opp_line_long,
            stop_line_left=None,
            stop_line_right=None,
            pair_plausible=pair_plausible,
            label=label_ego,
            label2=label_opp,
        )

        # Draw ROI quadrants
        debug_image = self._draw_quadrants(
            debug_image, q1, q2, q3, q4, color=(100, 100, 100), thickness=1
        )

        # Draw detected stop lines (left from Q1, right from Q3)
        if stop_line_left is not None:
            x1, y1, x2, y2 = [int(v) for v in stop_line_left[0]]
            cv2.line(debug_image, (x1, y1), (x2, y2), VIOLET, 2)

        if stop_line_right is not None:
            x1, y1, x2, y2 = [int(v) for v in stop_line_right[0]]
            cv2.line(debug_image, (x1, y1), (x2, y2), VIOLET, 2)

        # Draw stop line labels if present
        y_offset = 20
        if label_stop_line_left is not None:
            cv2.putText(
                debug_image,
                label_stop_line_left,
                (0, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                GREEN,
            )
            y_offset += 25

        if label_stop_line_right is not None:
            cv2.putText(
                debug_image,
                label_stop_line_right,
                (0, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                GREEN,
            )

        # Composite debug overlay images onto final debug image
        if True and len(self.debug_overlay_images) > 0:
            # Add a title to show these are detection debug overlays
            cv2.putText(
                debug_image,
                "Line Detection Debug Overlays (warped images):",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            # Stack overlay images side-by-side or in a grid at bottom of image
            overlay_height = debug_image.shape[0] // 3
            overlay_width = debug_image.shape[1] // len(self.debug_overlay_images[:3])
            y_start = int(debug_image.shape[0] * 0.6)
            for i, overlay_img in enumerate(self.debug_overlay_images[:3]):
                # Resize overlay to fit
                resized = cv2.resize(
                    overlay_img,
                    (overlay_width, overlay_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                x_start = i * overlay_width
                try:
                    debug_image[
                        y_start : y_start + overlay_height,
                        x_start : x_start + overlay_width,
                    ] = resized
                except Exception:
                    pass  # Skip if dimensions don't match
            # Clear the list for next frame
            self.debug_overlay_images.clear()

        # save debug image and return image + result list
        # IntersectionDetector.save_img_to_dir(
        #    debug_image, time.perf_counter_ns().__str__() + "_full.jpg"
        # )

        return debug_image, result_list


def main(args=None):
    """
    Main function to start the ROS2ExampleNode.

    Keyword Arguments:
        args -- Launch arguments (default: {None})
    """
    rclpy.init(args=args)
    node = IntersectionDetector()

    # We have 2 options on how to run the node:
    # 1. Let the node idle in the background with 'rclpy.spin(node)' if we want to let
    #   subscriber callback function handle the execution of our code.
    #   TODO: is it possible in this way, that our callback gets executed multiple times
    #       in parallel?
    # 2. Run the node in a while loop that waits for incoming messages and then executes
    #   our code. This makes sure that always the latest message is processed and never
    #   multiple messages in parallel. It should be used if the processing of the
    #   message/execution of our code takes longer than the time between incoming #
    #   messages.

    try:
        #
        use_wait_for_message = True
        if use_wait_for_message:
            while rclpy.ok():
                # node.wait_for_message_and_execute()
                # for img in os.listdir(IntersectionDetector.DBG_IMG_DIR):
                #    node.pipeline(os.path.join(IntersectionDetector.DBG_IMG_DIR, img))
                rclpy.spin_once(node)
        else:
            rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        # Shutdown if not already done by the ROS2 launch system
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
