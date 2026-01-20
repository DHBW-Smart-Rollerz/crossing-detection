import math
import os
import time
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

# Color constants (BGR tuples)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)

# Bright colors
CYAN = (255, 255, 0)
ORANGE = (0, 128, 255)
LIME = (50, 255, 50)
PINK = (255, 0, 180)
VIOLET = (180, 50, 255)
TURQUOISE = (255, 255, 100)
GOLD = (0, 215, 255)

RESULT_DIR_NAME = f"crossing_results/{time.strftime('%Y%m%d-%H%M%S')}"
# FILTERING_ROI_REL_RLTB = (0.80, 0, 0.12, 0.45)  # left, right, top, bottom
FILTERING_ROI_REL_RLTB = (0.80, 0, 0, 0.7)  # left, right, top, bottom


lsd = cv2.createLineSegmentDetector(1)


class IntersectionDetector(SmartyNode):
    """
    A ROS2 node for crossing detection.

    Arguments:
        SmartyNode -- Base class for ROS2 nodes.

    Returns:
        None
    """

    DBG_IMG_DIR = "/home/smartrollerz/Desktop/smartrollers/smarty_workspace/rosbag_images/rosbag2_2025_03_06-17_56_01"

    def __init__(self):
        """Initialize the ROS2ExampleNode."""
        super().__init__(
            "crossing_detection_node",
            "crossing_detection",
            node_parameters={
                # Subscriber topics
                "image_subscriber": "/camera/birds_eye",
                # Publisher topics
                "debug_image_publisher": "/example/birdseye_view",
                "result_publisher": "/example/result",
                # Parameters
                "state": NodeState.ACTIVE.value,
                "image_path": "resources/img/example.png",
                "example_value": 128,
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
                "result_publisher": (std_msgs.msg.Float32, 1),
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

    @property
    def image_path(self) -> str:
        """Get the image path parameter."""
        return os.path.join(
            self.package_path,
            self.get_parameter("image_path").value,  # type: ignore
        )  # type: ignore # full path to the image

    @property
    def example_value(self) -> int:
        """Get the example value parameter."""
        return self.get_parameter("example_value").value  # type: ignore

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

            # run the pipeline on the cv2 image
            self.pipeline(img)

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
            self.result_publisher.publish(  # type: ignore
                std_msgs.msg.Float32(
                    data=res,
                )
            )

        if self._debug:
            with timer.Timer(name="debug", filter_strength=40):
                # Create the debug image. Here it is just the image filled with
                # the example value.
                debug_image = image.copy()
                debug_image.fill(self.example_value)

                self.debug_image_publisher.publish(  # type: ignore
                    self.cv_bridge.cv2_to_imgmsg(image, encoding="8UC1")
                )

        timer.Timer(logger=self.get_logger().info).print()  # type: ignore

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

    @staticmethod
    def save_img_to_dir(img, img_name: str):
        """
        Save an image to a specified directory.

        Arguments:
            img -- Image to be saved.
            dir_path -- Directory path where the image will be saved.
            img_name -- Name of the image file.
        """
        if not os.path.exists(RESULT_DIR_NAME):
            os.makedirs(RESULT_DIR_NAME)
        cv2.imwrite(os.path.join(RESULT_DIR_NAME, img_name), img)

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
        img = cv2.GaussianBlur(img, (5, 5), 0)
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

    def filter_by_length(self, lines, min_length: float = 70.0):
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
            if length >= min_length:
                res.append(line)

        return res

    def crop_image_center(self, image):
        """
        Crop the image to a centered square.

        Arguments:
            image -- Input image.

        Returns:
            Cropped image.
        """
        # add docstring
        # how to crop the image (also with birdseye?)
        # also transform crop coordinates into real image coordinates later
        img = image[::]
        imsize = img.shape
        height = imsize[0]
        width = imsize[1]

        crop_height = 350
        crop_width = 350

        start_y = (height - crop_height) // 2
        start_x = (width - crop_width) // 2

        image = image[
            start_y : start_y + crop_height, start_x : start_x + crop_width, :
        ]
        return image

    def crop_image(self, image):
        """
        Crop the image to a centered rectangle.

        Arguments:
            image -- Input image.

        Returns:
            Cropped image.
        """
        # add docstring
        # how to crop the image (also with birdseye?)
        # also transform crop coordinates into real image coordinates later
        img = image[::]
        imsize = img.shape
        height = imsize[0]
        width = imsize[1]

        crop_height = 300
        crop_width = width - 100

        start_x = (width - crop_width) // 2

        image = image[crop_height:height, start_x : start_x + crop_width, :]
        return image

    def contours(self, edges, img):
        """
        Find and draw contours on the image.

        Arguments:
            edges -- Edge-detected
            img -- Input image.
        """
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Boxen auf das Farbbild zeichnen
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # Zeichne die Box auf das Farbbild
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

        # Speichere das Bild mit den Boxen
        IntersectionDetector.show_image(img, "contours")

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
        Draw a small legend on the image.

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

        # origin at bottom-left with 10px margin
        margin = 10
        origin_x = margin
        origin_y = height - margin - legend_height

        # prevent drawing outside image (fallback to top-left if too large)
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

    def elongate_line(self, line, length: float = 300):
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
        line_horizontal_distance_threshold: float = 60.0,
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
        x1_1, _, x2_1, _ = ego_line[0]
        x1_2, _, x2_2, _ = opp_line[0]
        ego_line_leftmost = min(x1_1, x2_1)
        opp_line_rightmost = max(x1_2, x2_2)
        distance_between_lines = ego_line_leftmost - opp_line_rightmost
        # if distance_between_lines < negative_line_overlap_threshold:
        #    return False
        if distance_between_lines > line_horizontal_distance_threshold:
            return False

        # distance_to_center = abs(
        #    (ego_line_leftmost + opp_line_rightmost) / 2 - intersection_point[0]
        # )
        # if distance_to_center > center_horizontal_distance_threshold:
        #    return False

        return True

    def pipeline(self, image):
        """
        Complete processing pipeline for intersection detection.

        Arguments:
            img_path -- Path to the input image.
        """
        # image = IntersectionDetector.load_img_grayscale(img_path)
        # image = self.crop_image(image)

        # blur + optional closing on the upper half of the ROI to help LSD
        # detect distorted bird's-eye parts
        image = self._blur_roi_top(
            image, ksize=(20, 20), sigmaX=0, do_close=True, close_kernel=(25, 3)
        )
        edges = self.perform_canny(image)
        transformed_lines = self.line_segment_detector(edges)
        # normalize detected lines to canonical numpy (1,4) arrays
        transformed_lines = self._normalize_lines(transformed_lines)
        image = self._draw_lines(image, transformed_lines, color=MAGENTA)
        filtered_lines = self.filter_by_length(transformed_lines, min_length=20)
        # image = self._draw_lines(image, filtered_lines, color=BLUE)
        # image = self._draw_lines(image, filtered_lines, color=YELLOW)
        filtered_lines = self.filter_by_roi(filtered_lines, image.shape)
        # image = self._draw_lines(image, filtered_lines, color=GREEN)

        vert, horiz = self.filter_by_angle(filtered_lines)

        # compute crossing center optionally; if disabled, use ROI center
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

        if crossing_center is not None:
            image = cv2.circle(image, crossing_center, 8, YELLOW)
            crossing_center = self.pull_point_to_roi_center(
                crossing_center, image.shape
            )
            image = cv2.circle(
                image, (int(crossing_center[0]), int(crossing_center[1])), 8, TURQUOISE
            )

        lines = vert + horiz
        lines = self.filter_by_length(lines, min_length=100)

        vert, horiz = self.filter_by_angle(lines)

        image = self._draw_lines(image, vert, color=RED)

        # Only attempt to find ego/opp lines if we have a valid crossing center.
        if crossing_center is not None:
            ego_line = self.find_ego_line(horiz, crossing_center)
            opp_line = self.find_opp_line(horiz, crossing_center)

            if ego_line is not None:
                ego_line = self.elongate_line(ego_line)
            if opp_line is not None:
                opp_line = self.elongate_line(opp_line)

            pair_plausible = False
            if ego_line is not None and opp_line is not None:
                pair_plausible = self.check_plausibility_horizontal_line_pair(
                    opp_line, ego_line, crossing_center
                )

            try:
                if ego_line is not None:
                    image = self._draw_lines(
                        image,
                        [ego_line],
                        color=GREEN if pair_plausible else RED,
                        thickness=6,
                    )
            except Exception as e:
                self.get_logger().error(f"find_ego_line error: {e}")

            try:
                if opp_line is not None:
                    image = self._draw_lines(
                        image,
                        [opp_line],
                        color=BLUE if pair_plausible else RED,
                        thickness=6,
                    )
            except Exception as e:
                self.get_logger().error(f"find_opp_line error: {e}")
        else:
            # no crossing center found for this frame; skip ego/opp identification
            self.get_logger().debug(
                "pipeline: skipping ego/opp line search, no crossing center"
            )

        """
         nearest_line = self.get_nearest_line(horiz)
        if nearest_line is not None:
            nearest_line = self.elongate_line(nearest_line)
            image = self._draw_lines(image, [nearest_line], color=RED, thickness=6)

        lane_left, lane_right = self.detect_lane(vert, image)

        if lane_right is not None:
            lane_right = self.elongate_line(lane_right, length=450)
            image = self._draw_lines(image, [lane_right], color=YELLOW, thickness=6)

        """
        # draw roi box
        roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)
        cv2.rectangle(
            image,
            (roi_left, roi_top),
            (roi_right, roi_bottom),
            RED,
            2,
        )

        image = self._draw_legend(
            image,
            [
                ("All lines (LSD)", MAGENTA),
                ("EGO LINE", GREEN),
                ("OPP LINE", BLUE),
                ("NON PLAUSIBLE LINEPAIR", RED),
                ("Calculated Intersection Center", YELLOW),
                ("Weighted Intersection Center", TURQUOISE),
                ("ROI box", RED),
            ],
        )

        IntersectionDetector.save_img_to_dir(
            image, time.perf_counter_ns().__str__() + "_full.jpg"
        )


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
