import math
import os
import time
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


# FILTERING_ROI_REL_RLTB = (0.80, 0, 0.12, 0.45)  # left, right, top, bottom
FILTERING_ROI_REL_RLTB = (0.80, 0, 0, 0.8)  # left, right, top, bottom


class IntersectionAggregator:
    """
    Aggregates intersection detection results over 10 frames.

    Different line types have different memory durations:
    - Stop lines (left/right): 3 frames
    - Ego line: 5 frames
    - Opp line: 3 frames

    Accumulates all detections and provides the current
    intersection configuration.
    """

    def __init__(self, max_frames=7):
        """
        Initialize the aggregator.

        Args:
            max_frames: Maximum number of frames to aggregate (7)
        """
        self.max_frames = max_frames
        self.frame_count = 0
        self.first_detection_frame = None

        # Storage for detections with their frame numbers
        # Format: (frame_num, line_data, is_dotted)
        self.ego_lines = []
        self.opp_lines = []
        self.stop_line_left = []
        self.stop_line_right = []

        # Current dotted flags for each line type
        self.ego_dotted = None
        self.opp_dotted = None
        self.stop_left_dotted = None
        self.stop_right_dotted = None

        # Memory durations for each line type
        self.memory_durations = {
            "ego": 3,
            "opp": 4,
            "stop_left": 6,
            "stop_right": 5,
        }

    def add_detection(
        self,
        ego_line=None,
        opp_line=None,
        stop_line_left=None,
        stop_line_right=None,
        ego_dotted=None,
        opp_dotted=None,
        stop_dotted_left=None,
        stop_dotted_right=None,
    ):
        """
        Add detection results for the current frame.

        Args:
            ego_line: Ego line data (numpy array) or None
            opp_line: Opp line data (numpy array) or None
            stop_line_left: Left stop line data or None
            stop_line_right: Right stop line data or None
            ego_dotted: True if ego is dotted, False if solid
            opp_dotted: True if opp is dotted, False if solid
            stop_dotted_left: True if left stop is dotted
            stop_dotted_right: True if right stop is dotted
        """
        # Initialize frame counter at first detection
        if self.first_detection_frame is None and any(
            [
                ego_line is not None,
                opp_line is not None,
                stop_line_left is not None,
                stop_line_right is not None,
            ]
        ):
            self.first_detection_frame = 0

        # Don't accumulate if no detections started yet
        if self.first_detection_frame is None:
            return

        current_frame = self.frame_count

        # Add to storage and update dotted flags
        if ego_line is not None:
            self.ego_lines.append((current_frame, ego_line.copy()))
            self.ego_dotted = ego_dotted

        if opp_line is not None:
            self.opp_lines.append((current_frame, opp_line.copy()))
            self.opp_dotted = opp_dotted

        if stop_line_left is not None:
            self.stop_line_left.append((current_frame, stop_line_left.copy()))
            self.stop_left_dotted = stop_dotted_left

        if stop_line_right is not None:
            self.stop_line_right.append((current_frame, stop_line_right.copy()))
            self.stop_right_dotted = stop_dotted_right

        self.frame_count += 1

    def _is_within_memory(self, detection_frame, current_frame, line_type):
        """
        Check if detection is within memory duration.

        Args:
            detection_frame: Frame number when detected
            current_frame: Current frame number
            line_type: Type of line

        Returns:
            True if within memory duration, False otherwise
        """
        frame_age = current_frame - detection_frame
        max_age = self.memory_durations.get(line_type, 1)
        return frame_age <= max_age

    def get_current_configuration(self):
        """
        Get current intersection configuration.

        Returns:
            Dictionary with 'ego_line', 'opp_line',
            'stop_line_left', 'stop_line_right', detection counts,
            frame_count, and is_complete flag.
        """
        # Check if we have exceeded max frames
        if self.first_detection_frame is not None and (
            self.frame_count - self.first_detection_frame >= self.max_frames
        ):
            return self._get_final_config()

        current_frame = self.frame_count - 1

        # Filter by memory duration
        valid_ego = [
            line
            for frame, line in self.ego_lines
            if self._is_within_memory(frame, current_frame, "ego")
        ]

        valid_opp = [
            line
            for frame, line in self.opp_lines
            if self._is_within_memory(frame, current_frame, "opp")
        ]

        valid_stop_left = [
            line
            for frame, line in self.stop_line_left
            if self._is_within_memory(frame, current_frame, "stop_left")
        ]

        valid_stop_right = [
            line
            for frame, line in self.stop_line_right
            if self._is_within_memory(frame, current_frame, "stop_right")
        ]

        config = {
            "ego_line": valid_ego[-1] if valid_ego else None,
            "opp_line": valid_opp[-1] if valid_opp else None,
            "stop_line_left": (valid_stop_left[-1] if valid_stop_left else None),
            "stop_line_right": (valid_stop_right[-1] if valid_stop_right else None),
            "ego_detections": len(self.ego_lines),
            "opp_detections": len(self.opp_lines),
            "stop_left_detections": len(self.stop_line_left),
            "stop_right_detections": len(self.stop_line_right),
            "frame_count": self.frame_count,
            "is_complete": all(
                [valid_ego, valid_opp, valid_stop_left, valid_stop_right]
            ),
        }

        return config

    def _get_final_config(self):
        """Get final configuration and reset aggregator."""
        current_frame = self.frame_count - 1

        valid_ego = [
            line
            for frame, line in self.ego_lines
            if self._is_within_memory(frame, current_frame, "ego")
        ]

        valid_opp = [
            line
            for frame, line in self.opp_lines
            if self._is_within_memory(frame, current_frame, "opp")
        ]

        valid_stop_left = [
            line
            for frame, line in self.stop_line_left
            if self._is_within_memory(frame, current_frame, "stop_left")
        ]

        valid_stop_right = [
            line
            for frame, line in self.stop_line_right
            if self._is_within_memory(frame, current_frame, "stop_right")
        ]

        config = {
            "ego_line": valid_ego[-1] if valid_ego else None,
            "opp_line": valid_opp[-1] if valid_opp else None,
            "stop_line_left": (valid_stop_left[-1] if valid_stop_left else None),
            "stop_line_right": (valid_stop_right[-1] if valid_stop_right else None),
            "ego_detections": len(self.ego_lines),
            "opp_detections": len(self.opp_lines),
            "stop_left_detections": len(self.stop_line_left),
            "stop_right_detections": len(self.stop_line_right),
            "frame_count": self.frame_count,
            "is_complete": all(
                [valid_ego, valid_opp, valid_stop_left, valid_stop_right]
            ),
        }

        # Reset for next aggregation cycle
        self._reset()

        return config

    def _reset(self):
        """Reset aggregator for next cycle."""
        self.frame_count = 0
        self.first_detection_frame = None
        self.ego_lines = []
        self.opp_lines = []
        self.stop_line_left = []
        self.stop_line_right = []

    def is_aggregation_complete(self):
        """Check if 10 frames have been aggregated."""
        if self.first_detection_frame is None:
            return False
        return self.frame_count - self.first_detection_frame >= self.max_frames

    def get_crossing_type(self):
        """
        Generate a crossing type string based on current state.

        Only considers lines that are still within memory duration.

        Format: es-od-ln-rn
        - e: ego (es=solid, ed=dotted, en=none)
        - o: opp (os=solid, od=dotted, on=none)
        - l: left stop (ls=solid, ld=dotted, ln=none)
        - r: right stop (rs=solid, rd=dotted, rn=none)

        Returns:
            String in format "es-od-ln-rn" representing the crossing
        """
        current_frame = self.frame_count - 1 if self.frame_count > 0 else 0

        # Check ego line within memory
        valid_ego = [
            line
            for frame, line in self.ego_lines
            if self._is_within_memory(frame, current_frame, "ego")
        ]
        if len(valid_ego) == 0:
            ego_type = "en"
        elif self.ego_dotted is True:
            ego_type = "ed"
        else:
            ego_type = "es"

        # Check opp line within memory
        valid_opp = [
            line
            for frame, line in self.opp_lines
            if self._is_within_memory(frame, current_frame, "opp")
        ]
        if len(valid_opp) == 0:
            opp_type = "on"
        elif self.opp_dotted is True:
            opp_type = "od"
        else:
            opp_type = "os"

        # Check left stop line within memory
        valid_stop_left = [
            line
            for frame, line in self.stop_line_left
            if self._is_within_memory(frame, current_frame, "stop_left")
        ]
        if len(valid_stop_left) == 0:
            left_stop_type = "ln"
        elif self.stop_left_dotted is True:
            left_stop_type = "ld"
        else:
            left_stop_type = "ls"

        # Check right stop line within memory
        valid_stop_right = [
            line
            for frame, line in self.stop_line_right
            if self._is_within_memory(frame, current_frame, "stop_right")
        ]
        if len(valid_stop_right) == 0:
            right_stop_type = "rn"
        elif self.stop_right_dotted is True:
            right_stop_type = "rd"
        else:
            right_stop_type = "rs"

        # Combine into final string
        crossing_type_str = (
            f"{ego_type}-{opp_type}-{left_stop_type}" f"-{right_stop_type}"
        )
        return crossing_type_str


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

        # Read sharpening parameters
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

        # Read distortion enhancement parameters
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
            # fallback defaults
            self.enhance_distorted_roi_enabled = True
            self.enhance_distortion_kernel = 5
            self.enhance_distortion_dilations = 1

        # Read gap detection debug visualization parameter
        try:
            self.debug_line_gap_detection = self.get_parameter(
                "debug_line_gap_detection"
            ).value
        except Exception:
            # fallback default
            self.debug_line_gap_detection = False

        # Debug overlay storage for line detection visualization
        self.debug_overlay_images = []

        # Crossing center with 4-frame hold and shift logic
        self.detected_crossing_center = None  # newly detected center
        self.active_crossing_center = None  # currently used center
        self.crossing_center_frames = 0  # frame counter (0-3)
        self.crossing_center_error = float("inf")  # error metric

        # Initialize intersection aggregator for result collection
        self.intersection_aggregator = IntersectionAggregator(max_frames=9)

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
        detected_corners=None,
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
        closest_line_angle=None,
    ):
        """
        Draw all debug overlays at once on a provided image.
        This keeps the detection logic separated from visualization.
        """
        # draw basic line sets
        # if transformed_lines is not None:
        #    image = self._draw_lines(image, transformed_lines, color=MAGENTA)

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
                    (
                        int(crossing_center_pulled[0]),
                        int(crossing_center_pulled[1]),
                    ),
                    8,
                    TURQUOISE,
                )
            except Exception:
                pass

        # draw detected corners as pink and orange crosses
        if detected_corners is not None and len(detected_corners) > 0:
            try:
                # Calculate angles at each corner
                sorted_corners, interior_angles = self.compute_corner_angles(
                    detected_corners
                )

                # Alternate between PINK and ORANGE for visual distinction
                for i, corner in enumerate(detected_corners):
                    x, y = corner
                    color = PINK if i % 2 == 0 else ORANGE
                    # Draw a cross at corner position
                    cross_size = 8
                    image = cv2.line(
                        image,
                        (x - cross_size, y),
                        (x + cross_size, y),
                        color,
                        2,
                    )
                    image = cv2.line(
                        image,
                        (x, y - cross_size),
                        (x, y + cross_size),
                        color,
                        2,
                    )

                # Draw interior angles at each corner if available
                if sorted_corners is not None and interior_angles is not None:
                    for idx, angle_deg in enumerate(interior_angles):
                        corner = sorted_corners[idx].astype(int)
                        cx, cy = corner[0], corner[1]
                        # Display angle near corner
                        angle_text = f"{angle_deg:.0f}°"
                        cv2.putText(
                            image,
                            angle_text,
                            (cx - 15, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            GOLD,
                            1,
                        )

                # Add corner count label
                corner_label = f"Corners: {len(detected_corners)}"
                cv2.putText(
                    image,
                    corner_label,
                    (0, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    PINK,
                )

                # Display angle metrics if angles are available
                if sorted_corners is not None and interior_angles is not None:
                    # Check if it's a valid rectangle
                    is_rect = self.is_valid_rectangle(
                        interior_angles, angle_tolerance=20.0
                    )
                    rect_status = "✓ RECT" if is_rect else "✗ NOT RECT"
                    status_color = GREEN if is_rect else RED

                    # Show angle error (mean deviation from 90°)
                    angle_error = self.compute_angle_error(interior_angles)
                    error_text = f"Angle Error: {angle_error:.1f}°"

                    cv2.putText(
                        image,
                        rect_status,
                        (0, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        status_color,
                    )
                    cv2.putText(
                        image,
                        error_text,
                        (0, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        GOLD,
                    )

                    # Show frame counter if in hold period
                    if self.active_crossing_center is not None:
                        frame_text = f"Hold Frame: " f"{self.crossing_center_frames}/4"
                        cv2.putText(
                            image,
                            frame_text,
                            (0, 165),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            CYAN,
                        )
            except Exception:
                pass

        # Draw crossing center (shifts 15px/frame if detected)
        try:
            if crossing_center is not None:
                cx, cy = crossing_center
                # Draw larger circle for crossing center
                cv2.circle(image, (cx, cy), 12, GREEN, 3)
                cv2.circle(image, (cx, cy), 8, CYAN, 2)

                # Label shows frame counter only if center was detected
                if self.active_crossing_center is not None:
                    frame_indicator = f"CENTER ({self.crossing_center_frames + 1}/4)"
                else:
                    frame_indicator = "CENTER (ROI)"

                # Add label
                cv2.putText(
                    image,
                    frame_indicator,
                    (cx + 15, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    GREEN,
                    1,
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
            if closest_line_angle is not None:
                self._draw_angle_arrow(image, closest_line_angle)
        except Exception:
            pass

        # draw roi box and legend
        try:
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
                    ("Transformed lines", MAGENTA),
                    ("Vertical lines", GOLD),
                    ("Ego line", LIME),
                    ("Opp plausible", LIME),
                    ("Opp implausible", ORANGE),
                    ("Stop lines", VIOLET),
                    ("Crossing center", YELLOW),
                    ("Center (ROI)", TURQUOISE),
                    ("Corners", CYAN),
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

        # Clamp coordinates
        roi_left = max(0, min(w - 1, int(roi_left)))
        roi_right = max(0, min(w, int(roi_right)))
        roi_top = max(0, min(h - 1, int(roi_top)))
        roi_bottom = max(0, min(h, int(roi_bottom)))

        # Determine which half to sharpen
        if do_roi_top:
            # Top half of ROI
            top_half_end = roi_top + roi_bottom
        else:
            # Bottom half of ROI
            top_half_end = roi_top + (roi_bottom - roi_top) // 2

        # Ensure valid box
        top = max(0, roi_top)
        bottom = max(top, min(h, top_half_end if do_roi_top else roi_bottom))
        left = max(0, roi_left)
        right = max(0, min(w, roi_right))

        out = image.copy()
        if bottom > top and right > left:
            patch = out[top:bottom, left:right].copy()

            # Create blurred version
            blurred = cv2.GaussianBlur(patch, (5, 5), 0)

            # Unsharp mask: original + strength * (original - blurred)
            sharpened = cv2.addWeighted(patch, 1.0 + strength, blurred, -strength, 0)

            # Clip to valid range [0, 255]
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

        # Clamp coordinates
        roi_left = max(0, min(w - 1, int(roi_left)))
        roi_right = max(0, min(w, int(roi_right)))
        roi_top = max(0, min(h - 1, int(roi_top)))
        roi_bottom = max(0, min(h, int(roi_bottom)))

        # Determine which half to enhance
        if do_roi_top:
            # Top half (most distorted)
            top_half_end = roi_top + roi_bottom
        else:
            # Bottom half
            top_half_end = roi_top + (roi_bottom - roi_top) // 2

        # Ensure valid box
        top = max(0, roi_top)
        bottom = max(top, min(h, top_half_end if do_roi_top else roi_bottom))
        left = max(0, roi_left)
        right = max(0, min(w, roi_right))

        out = image.copy()
        if bottom > top and right > left:
            patch = out[top:bottom, left:right].copy()

            # Ensure odd kernel size
            if morph_kernel_size % 2 == 0:
                morph_kernel_size += 1

            # Create morphological kernel
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
            )

            # 1. Morphological closing: fill small holes
            closed = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, kernel, iterations=1)

            # 2. Dilation: strengthen and connect broken lines
            dilated = cv2.dilate(closed, kernel, iterations=dilation_iterations)

            out[top:bottom, left:right] = dilated

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

    def _draw_crossing_type_visualization(self, image, crossing_type_str):
        """
        Draw a crossing type visualization in top right corner.

        Shows a square with 4 lines like a real intersection:
        - Top: OPP line
        - Bottom: EGO line
        - Left: LEFT STOP line
        - Right: RIGHT STOP line

        Green = detected, Red = not detected
        Solid = solid line, Dotted = dotted line

        Arguments:
            image -- Image to draw on
            crossing_type_str -- String like "es-od-ls-rn"
        """
        try:
            height, width = image.shape[:2]

            # Parse crossing type string
            parts = crossing_type_str.split("-")
            if len(parts) != 4:
                return

            ego_type = parts[0]  # en/es/ed
            opp_type = parts[1]  # on/os/od
            stop_l_type = parts[2]  # ln/ls/ld
            stop_r_type = parts[3]  # rn/rs/rd

            # Panel position and size
            panel_x = width - 90
            panel_y = 80
            square_size = 50

            # Draw semi-transparent background
            overlay = image.copy()
            cv2.rectangle(
                overlay,
                (panel_x - 10, panel_y - 10),
                (panel_x + square_size + 10, panel_y + square_size + 10),
                (0, 0, 0),
                -1,
            )
            cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

            # Helper function to get color and draw style
            def get_color_and_style(line_type):
                """Get color and dotted flag from line type."""
                is_detected = line_type[1] != "n"
                is_dotted = line_type[1] == "d"
                color = GREEN if is_detected else RED
                return color, is_dotted

            # Helper to draw a line (solid or dotted)
            def draw_styled_line(pt1, pt2, color, is_dotted, thickness=2):
                """Draw a line (solid or dotted)."""
                if is_dotted:
                    # Draw dotted line
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
                    # Draw solid line
                    cv2.line(image, pt1, pt2, color, thickness)

            # Get colors and styles for each line
            ego_color, ego_dotted = get_color_and_style(ego_type)
            opp_color, opp_dotted = get_color_and_style(opp_type)
            left_color, left_dotted = get_color_and_style(stop_l_type)
            right_color, right_dotted = get_color_and_style(stop_r_type)

            # Square corners
            top_left = (panel_x, panel_y)
            top_right = (panel_x + square_size, panel_y)
            bottom_left = (panel_x, panel_y + square_size)
            bottom_right = (
                panel_x + square_size,
                panel_y + square_size,
            )

            # Draw the 4 lines of the crossing square
            # Top line (OPP)
            draw_styled_line(top_left, top_right, opp_color, opp_dotted, thickness=2)

            # Bottom line (EGO)
            draw_styled_line(
                bottom_left,
                bottom_right,
                ego_color,
                ego_dotted,
                thickness=2,
            )

            # Left line (LEFT STOP)
            draw_styled_line(
                top_left,
                bottom_left,
                left_color,
                left_dotted,
                thickness=2,
            )

            # Right line (RIGHT STOP)
            draw_styled_line(
                top_right,
                bottom_right,
                right_color,
                right_dotted,
                thickness=2,
            )

        except Exception:
            pass

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

            # Check if either endpoint is in the region
            p1_in = region_x1 <= x1 <= region_x2 and region_y1 <= y1 <= region_y2
            p2_in = region_x1 <= x2 <= region_x2 and region_y1 <= y2 <= region_y2

            if p1_in or p2_in:
                return True

            # Check if line intersects region bounds
            # Simple bounding box check
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

            # Calculate slope
            m = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0

            if direction == "right":
                # Extend from rightmost endpoint
                line_start = (x1, y1) if x1 > x2 else (x2, y2)

                # Add 40px padding from line start
                x_padded = line_start[0] + 40.0
                y_padded = line_start[1] + m * 40.0

                # Extend 50px further to the right
                x2_extended = x_padded + 50.0
                y2_extended = y_padded + m * 50.0

            else:  # direction == "left"
                # Extend from leftmost endpoint
                line_start = (x1, y1) if x1 < x2 else (x2, y2)

                # Add 40px padding from line start (going left, subtract)
                x_padded = line_start[0] - 40.0
                y_padded = line_start[1] - m * 40.0

                # Extend 50px further to the left
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

            # Test extended line for gaps
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

            print(
                f"EXTENSION TEST ({direction}): gaps={gaps_count} "
                f"wr={wr_extended:.1f}%"
            )

            # Reject if no gaps and wr > 20% (continuous solid line)
            if gaps_count == 0 and wr_extended > 20.0:
                print(
                    f"REJECTING: extended line ({direction}) is "
                    f"continuous (wr={wr_extended:.1f}% > 20%)"
                )
                return True, extended_line  # Invalid, reject

            return False, extended_line  # Valid, keep

        except Exception as e:
            self.get_logger().error(f"Error in check_line_by_horizontal_extension: {e}")
            return False, None  # On error, assume valid

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
                # Right stop: extend from lowest point (max y) downward
                # Find the point with maximum y
                if y1 > y2:
                    x_start, y_start = x1, y1
                    x_other, y_other = x2, y2
                else:
                    x_start, y_start = x2, y2
                    x_other, y_other = x1, y1

                # Calculate slope along the line (for x-change)
                if (y_start - y_other) != 0:
                    slope = (x_start - x_other) / (y_start - y_other)
                else:
                    slope = 0.0

                # Add 40px padding downward (increasing y)
                x_padded = x_start + slope * 40.0
                y_padded = y_start + 40.0

                # Extend 50px further downward
                x2_extended = x_padded + slope * 50.0
                y2_extended = y_padded + 50.0

            else:  # is_left
                # Left stop: extend from highest point (min y) upward
                # Find the point with minimum y
                if y1 < y2:
                    x_start, y_start = x1, y1
                    x_other, y_other = x2, y2
                else:
                    x_start, y_start = x2, y2
                    x_other, y_other = x1, y1

                # Calculate slope along the line (for x-change)
                if (y_start - y_other) != 0:
                    slope = (x_start - x_other) / (y_start - y_other)
                else:
                    slope = 0.0

                # Add 40px padding upward (decreasing y)
                x_padded = x_start - slope * 40.0
                y_padded = y_start - 40.0

                # Extend 50px further upward
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

            print(
                f"[DEBUG] Stop line {'RIGHT' if is_right else 'LEFT'}: "
                f"start=({x_start:.1f}, {y_start:.1f}), "
                f"padded=({x_padded:.1f}, {y_padded:.1f}), "
                f"extended=({x2_extended:.1f}, {y2_extended:.1f})"
            )

            # Test extended line for gaps
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
            print(
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
        # Normalize angle to [0, 180) so opposite directions are same
        if angle < 0:
            angle += 180

        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0

        # DEBUG: Draw input line on image (controlled by node parameter)
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
            self.debug_overlay_images.append(debug_image)

        crop_w = int(max(10, line_len * float(length_extend))) + int(box_half_width * 2)
        crop_h = int(max(3, box_half_width * 2))
        h, w = image.shape[:2]

        M = cv2.getRotationMatrix2D((mid_x, mid_y), angle, 1.0)
        warped = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

        cx = M[0, 0] * mid_x + M[0, 1] * mid_y + M[0, 2]
        cy = M[1, 0] * mid_x + M[1, 1] * mid_y + M[1, 2]

        # DEBUG: Log rotation details for tilted lines
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
            # Adaptive thresholding: build histogram and find best threshold
            # Histogram of pixel intensities
            hist = np.histogram(gray.flatten(), bins=256, range=(45, 256))[0]

            # Find peaks in the histogram (local maxima)
            # We want the brightest peak (highest intensity with good count)
            peaks = []
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                    peaks.append((i, hist[i]))  # (intensity, count)

            # If we have peaks, use the brightest one (highest intensity)
            if peaks:
                brightest_peak = max(peaks, key=lambda x: x[0])[0]
                adaptive_thresh = min(255, brightest_peak - 20)
            else:
                # Fallback: use the overall max intensity - 20
                adaptive_thresh = max(50, int(np.max(gray)) - 20)

            # Use adaptive threshold if reasonable, else fall back
            thresh_reasonable = 45 <= adaptive_thresh <= 220
            thresh_to_use = adaptive_thresh if thresh_reasonable else 180

            # binarize: white pixels (line) vs black (background)
            binary = gray > thresh_to_use

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

            # white ratio: percentage of white pixels in ACTUAL bounding box
            # NOTE: Use actual crop dimensions after clipping to bounds
            # Tilted crops get clipped, reducing height; using intended
            # crop_h would artificially lower the white ratio for tilted lines
            actual_crop_h = int(y2c - y1c)  # actual height after clipping
            actual_crop_w = int(x2c - x1c)  # actual width after clipping
            total_white_pixels = float(np.sum(white_per_col))
            total_pixels = float(actual_crop_h * actual_crop_w)
            if total_pixels > 0:
                white_ratio = (total_white_pixels / total_pixels) * 100.0
            else:
                white_ratio = 0.0

            is_dotted = gaps >= min_gap_count

            # DEBUG: Render crop visualization (controlled by node param)
            if self.debug_line_gap_detection:
                # Show: original crop, grayscale, binary, and overlay
                vis_crop = (
                    crop.copy()
                    if crop.ndim == 3
                    else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                )
                vis_h, vis_w = vis_crop.shape[:2]

                # Transform line endpoints to warped image space
                # then to crop space
                x1_warped = M[0, 0] * x1 + M[0, 1] * y1 + M[0, 2]
                y1_warped = M[1, 0] * x1 + M[1, 1] * y1 + M[1, 2]
                x2_warped = M[0, 0] * x2 + M[0, 1] * y2 + M[0, 2]
                y2_warped = M[1, 0] * x2 + M[1, 1] * y2 + M[1, 2]

                # Convert to crop coordinates (relative to crop top-left)
                line_x1_crop = int(x1_warped - x1c)
                line_y1_crop = int(y1_warped - y1c)
                line_x2_crop = int(x2_warped - x1c)
                line_y2_crop = int(y2_warped - y1c)

                # Clamp to crop bounds for visibility
                line_x1_crop = max(0, min(vis_w - 1, line_x1_crop))
                line_y1_crop = max(0, min(vis_h - 1, line_y1_crop))
                line_x2_crop = max(0, min(vis_w - 1, line_x2_crop))
                line_y2_crop = max(0, min(vis_h - 1, line_y2_crop))

                # Draw the input line in red
                cv2.line(
                    vis_crop,
                    (line_x1_crop, line_y1_crop),
                    (line_x2_crop, line_y2_crop),
                    (0, 0, 255),
                    2,
                )

                # Draw horizontal analysis line in green
                cv2.line(
                    vis_crop,
                    (0, vis_h // 2),
                    (vis_w, vis_h // 2),
                    (0, 255, 0),
                    1,
                )

                # Create binary visualization
                binary_vis = cv2.cvtColor(
                    (binary * 255).astype(np.uint8),
                    cv2.COLOR_GRAY2BGR,
                )

                # Create horizontal profile visualization
                profile_h = 50
                profile_w = len(white_per_col)
                profile_vis = np.zeros((profile_h, profile_w, 3), dtype=np.uint8)
                max_white = np.max(white_per_col) if np.max(white_per_col) > 0 else 1
                for i, count in enumerate(white_per_col):
                    h_bar = int((count / max_white) * profile_h)
                    if h_bar > 0:
                        profile_vis[profile_h - h_bar :, i] = [0, 255, 0]

                # Stack visualizations vertically
                vis_stack = np.vstack([vis_crop, binary_vis, profile_vis])

                # Store for debug rendering
                if not hasattr(self, "debug_overlay_images"):
                    self.debug_overlay_images = []
                self.debug_overlay_images.append(vis_stack)

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
        Find center of crossing using robust clustering with fallbacks.

        Arguments:
            intersection_points -- List of (x, y) intersection points

        Returns:
            Tuple (x, y) of crossing center, or None if not determinable
        """
        if len(intersection_points) == 0:
            return None

        # Need at least 2 points to define a center
        if len(intersection_points) < 2:
            return None

        points = np.array(intersection_points, dtype=float)

        try:
            # Try DBSCAN clustering with lenient parameters
            # eps=40: allows points up to 40px apart (for larger crossings)
            # min_samples=2: just need 2 intersections to form a cluster
            clustering = DBSCAN(eps=40, min_samples=2).fit(points)
            labels = clustering.labels_

            # Get non-noise points (label != -1)
            valid_mask = labels != -1
            valid_labels = labels[valid_mask]

            # If no cluster found, fall back to mean
            if len(valid_labels) == 0:
                center = points.mean(axis=0)
                return (int(center[0]), int(center[1]))

            # Find largest cluster
            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            largest_label = unique_labels[np.argmax(counts)]

            # Calculate mean of largest cluster
            cluster_points = points[labels == largest_label]
            center = cluster_points.mean(axis=0)

            return (int(center[0]), int(center[1]))

        except Exception as e:
            self.get_logger().warning(f"DBSCAN clustering failed ({e}), using mean")
            # Fallback: simple mean of all points
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
            print("No lines provided to histogram")
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

        print(f"Total lines: {len(lines)}")
        print(f"Lines filtered by angle (>45 deg): {lines_filtered_by_angle}")
        print(f"Valid angles for histogram: {len(valid_angles)}")

        if len(valid_angles) == 0:
            return None, 0

        valid_angles_arr = np.array(valid_angles)
        hist, bin_edges = np.histogram(valid_angles_arr, bins=36, range=(0, 180))

        peak_bin = int(np.argmax(hist))
        prominent_angle = float((bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0)

        count_str = len(valid_angles)
        msg = f"Prominent angle: {prominent_angle:.2f} deg ({count_str} lines)"
        print(msg)

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

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # If ROI is specified, extract that region for corner detection
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
            # Shi Tomasi corner detection
            corners = cv2.goodFeaturesToTrack(
                roi_region,
                maxCorners=4,
                qualityLevel=0.01,
                minDistance=200,
                blockSize=3,
                useHarrisDetector=False,
            )

            if corners is not None:
                # Convert corner coordinates to original image space
                # corners is shape (N, 1, 2), convert to list of tuples
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

            # Compute centroid
            centroid = np.mean(corners_array, axis=0)

            # Sort corners by angle around centroid
            angles = np.arctan2(
                corners_array[:, 1] - centroid[1], corners_array[:, 0] - centroid[0]
            )
            sorted_indices = np.argsort(angles)
            sorted_corners = corners_array[sorted_indices]

            # Calculate interior angles at each corner
            n = len(sorted_corners)
            angle_sum = 0.0

            for i in range(n):
                p1 = sorted_corners[(i - 1) % n]
                p2 = sorted_corners[i]
                p3 = sorted_corners[(i + 1) % n]

                # Vectors from p2 to p1 and p2 to p3
                v1 = p1 - p2
                v2 = p3 - p2

                # Calculate angle between vectors
                cos_angle = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
                )
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angle_sum += np.degrees(angle)

            # Expected angle sum: (n-2) * 180 degrees
            expected_sum = (n - 2) * 180.0

            # Allow tolerance of ±30 degrees
            tolerance = 30.0
            if abs(angle_sum - expected_sum) <= tolerance:
                # Valid polygon (approximately a rectangle)
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

            # Compute centroid
            centroid = np.mean(corners_array, axis=0)

            # Sort corners by angle around centroid
            angles_rad = np.arctan2(
                corners_array[:, 1] - centroid[1], corners_array[:, 0] - centroid[0]
            )
            sorted_indices = np.argsort(angles_rad)
            sorted_corners = corners_array[sorted_indices]

            # Calculate interior angles at each corner
            n = len(sorted_corners)
            interior_angles = []

            for i in range(n):
                p1 = sorted_corners[(i - 1) % n]
                p2 = sorted_corners[i]
                p3 = sorted_corners[(i + 1) % n]

                # Vectors from p2 to p1 and p2 to p3
                v1 = p1 - p2
                v2 = p3 - p2

                # Calculate angle between vectors
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

        # Require at least 3 out of 4 angles to be valid
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
        # If both are None, that's okay
        if stop_line_left is None and stop_line_right is None:
            return None, None

        # If only one exists, accept it as valid (unpaired stop line)
        if (stop_line_left is None) != (stop_line_right is None):
            return stop_line_left, stop_line_right

        # Both exist: validate they form a plausible pair
        if stop_line_left is not None and stop_line_right is not None:
            x1_l, y1_l, x2_l, y2_l = stop_line_left[0]
            x1_r, y1_r, x2_r, y2_r = stop_line_right[0]

            y_left = (y1_l + y2_l) / 2.0
            y_right = (y1_r + y2_r) / 2.0
            x_left = (x1_l + x2_l) / 2.0
            x_right = (x1_r + x2_r) / 2.0

            # Check 1: Vertical alignment (y-coords should be similar)
            y_diff = abs(y_left - y_right)
            if y_diff > max_y_diff or y_diff < min_y_diff:
                # Not aligned vertically - likely false positives
                return None, None

            # Check 2: Horizontal separation (should be reasonable)
            x_sep = abs(x_right - x_left)
            if x_sep < min_x_separation or x_sep > max_x_separation:
                # Not plausible horizontal separation
                return None, None

            # Pair is plausible
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

        # Determine which is top (smaller y) and bottom (larger y)
        if y1 < y2:
            top_pt = (x1, y1)
            bottom_pt = (x2, y2)
        else:
            top_pt = (x2, y2)
            bottom_pt = (x1, y1)

        # Check both endpoints
        endpoints = [
            ("top", top_pt),
            ("bottom", bottom_pt),
        ]

        endpoint_results = []

        for endpoint_name, (ep_x, ep_y) in endpoints:
            # Create 60x60 search region around endpoint
            search_region = (
                int(ep_x - 30),
                int(ep_y - 30),
                int(ep_x + 30),
                int(ep_y + 30),
            )

            # Find horizontal lines in region
            horiz_lines_in_region = [
                line
                for line in horiz
                if line is not None and self._is_line_in_region(line, search_region)
            ]

            num_lines = len(horiz_lines_in_region)
            print(
                f"{line_name} stop {endpoint_name} endpoint check "
                f"(y={ep_y:.1f}): found {num_lines} horizontal lines "
                f"in 60x60 region"
            )

            # Reject if no horizontal lines found at this endpoint
            if num_lines == 0:
                endpoint_results.append(False)
            else:
                endpoint_results.append(True)

        return all(endpoint_results)

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

        # Filter lines inside quadrant
        quad_lines = self.filter_lines_by_polygon(
            lines, quadrant, require_full=require_full
        )

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

        # Enhance distorted ROI areas (bird's eye view distortion)
        image = self._enhance_distorted_roi(
            image,
            do_roi_top=True,
            morph_kernel_size=self.enhance_distortion_kernel,
            dilation_iterations=self.enhance_distortion_dilations,
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

        closest_line_angle = None
        if fused_lines is not None and len(fused_lines) > 0:
            closest_line_angle, line_count = self.find_prominent_angle_in_quadrants(
                fused_lines, orig_image
            )

        # Filter by angle: tilt vert/horiz reference if angle is found
        vert, horiz = self.filter_by_angle(
            fused_lines,
            anchor_angle=closest_line_angle,
            anchor_tolerance=10.0,
            tol_deg=5,
        )

        # compute crossing center optionally; if disabled, use ROI center
        crossing_center = None
        if getattr(self, "compute_crossing_center", True):
            try:
                # Pre-filter lines by length before finding intersections
                # This removes small noise lines that create spurious
                # intersections
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
            # use ROI center as crossing center when computation is disabled
            # positioned at 0.4 (upward) instead of 0.5 (center)
            roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)
            crossing_center = (
                int((roi_left + roi_right) / 2),
                int(roi_top + (roi_bottom - roi_top) * 0.4),
            )

        # Detect intersection corners using Shi-Tomasi
        roi_bbox = self.get_roi_bbox(image.shape)
        detected_corners = self.find_corners_shi_tomasi(image, roi_bbox=roi_bbox)

        # Try to compute crossing center from corner geometry
        crossing_center = None
        if detected_corners and len(detected_corners) >= 3:
            sorted_corners, interior_angles = self.compute_corner_angles(
                detected_corners
            )

            if sorted_corners is not None and interior_angles is not None:
                # Calculate error (mean deviation from 90°)
                corner_error = self.compute_angle_error(interior_angles)

                # Accept if error < 20° or valid rectangle
                is_rect = self.is_valid_rectangle(interior_angles, angle_tolerance=20.0)

                if corner_error < 20.0 or is_rect:
                    center = np.mean(sorted_corners, axis=0).astype(int)
                    # Start new 4-frame hold period
                    self.detected_crossing_center = tuple(center)
                    self.crossing_center_frames = 0
                    self.crossing_center_error = corner_error
                    self.active_crossing_center = self.detected_crossing_center

        # Use active crossing center if in hold period with shifting
        if self.active_crossing_center is not None:
            # Get ROI bounds for bottom shift calculation
            roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)

            # Shift 15px toward ROI bottom each frame
            cx, cy = self.active_crossing_center
            shift_amount = 15 * (self.crossing_center_frames + 1)
            cy_shifted = min(cy + shift_amount, roi_bottom)

            crossing_center = (cx, int(cy_shifted))

            # Increment frame counter
            self.crossing_center_frames += 1

            # After 4 frames, reset everything
            if self.crossing_center_frames >= 4:
                self.active_crossing_center = None
                self.detected_crossing_center = None
                self.crossing_center_frames = 0
                # Fallback to ROI center after hold period ends (at 0.4)
                roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(
                    image.shape
                )
                crossing_center = (
                    int((roi_left + roi_right) / 2),
                    int(roi_top + (roi_bottom - roi_top) * 0.4),
                )
        else:
            # No crossing center detected, use ROI center as default (at 0.4)
            roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)
            crossing_center = (
                int((roi_left + roi_right) / 2),
                int(roi_top + (roi_bottom - roi_top) * 0.4),
            )

        lines = vert + horiz
        lines = self.filter_by_length(lines, min_length=70)
        fused_lines = lines
        # save short lines for stop line detection in quadrants

        # Detect stop lines using ROI quadrants (Q1 for left, Q2 for right)
        stop_line_left = self.find_line_in_quadrant(lines, q1, min_length=60)
        stop_line_right = self.find_line_in_quadrant(
            lines, q2, min_length=60, require_full=True
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
            # Validate: Check if dotted or solid with appropriate thresholds
            # Dotted: wr >= 10%, Solid: wr >= 22% (percentage-based)
            min_wr = 15 if stop_dotted_left else 30

            # Check extension for left stop line
            (
                gaps_ext,
                wr_ext,
                stop_line_left_ext,
            ) = self.check_line_by_vertical_extension(
                stop_line_left, image, is_right=False
            )
            print(f"Left stop line: gaps_ext={gaps_ext}, wr_ext={wr_ext:.1f}%")

            # Invalidate if extension test shows continuous solid line (no gaps
            # and high white ratio)
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
            # Validate: Check if dotted or solid with appropriate thresholds
            # Dotted: wr >= 10%, Solid: wr >= 22% (percentage-based)
            min_wr = 15 if stop_dotted_right else 30

            # Check extension for right stop line
            (
                gaps_ext,
                wr_ext,
                stop_line_right_ext,
            ) = self.check_line_by_vertical_extension(
                stop_line_right, image, is_right=True
            )
            print(f"Right stop line: gaps_ext={gaps_ext}, " f"wr_ext={wr_ext:.1f}%")

            # Invalidate if extension test shows continuous solid line (no gaps
            # and high white ratio)
            line_right_ext_passed = gaps_ext > 0 and wr_ext < 10

            if white_ratio_right >= min_wr and line_right_ext_passed:
                label_stop_line_right = (
                    f"STOP_RIGHT DOTTED (g={gap_count_right} wr={white_ratio_right:.1f}%)"
                    if stop_dotted_right
                    else f"STOP_RIGHT SOLID (g={gap_count_right} wr={white_ratio_right:.1f}%)"
                )
            else:
                stop_line_right = None

        # Plausibility checks for stop lines
        # Right stop line: should be below opp line and above crossing center
        if stop_line_right is not None and crossing_center is not None:
            x1_r, y1_r, x2_r, y2_r = stop_line_right[0]
            stop_y_r = (y1_r + y2_r) / 2.0
            crossing_y = crossing_center[1]
            # Right stop should be above crossing center
            if stop_y_r > crossing_y:
                stop_line_right = None
                label_stop_line_right = None
            else:
                # Check that right stop's highest point (min y) is not below
                # crossing center
                min_y_r = min(y1_r, y2_r)
                if min_y_r > crossing_y:
                    stop_line_right = None
                    label_stop_line_right = None

        # Check ROI horizontal bounds for right and left stop lines
        # (at least 10% inset from ROI edges)
        roi_left, roi_right, roi_top, roi_bottom = self.get_roi_bbox(image.shape)
        roi_width = roi_right - roi_left
        min_x_inset = roi_left + roi_width * 0.1
        max_x_inset = roi_right - roi_width * 0.1
        # Right stop needs sufficient space on the left (at least 60% from roi_left)
        min_x_right_stop = roi_left + roi_width * 0.6
        # Left stop needs sufficient space on the right (at most 40% from roi_left)
        max_x_left_stop = roi_left + roi_width * 0.4

        # Right stop line should be within inset bounds and RIGHT of crossing center
        if stop_line_right is not None:
            x1_r, y1_r, x2_r, y2_r = stop_line_right[0]
            stop_x_r = (x1_r + x2_r) / 2.0
            # Right stop must be within 10%-90% of ROI width
            if stop_x_r < min_x_inset or stop_x_r > max_x_inset:
                stop_line_right = None
                label_stop_line_right = None
            # Right stop must have sufficient space on the left (at least 60%)
            elif stop_x_r < min_x_right_stop:
                print(
                    f"RIGHT stop rejected: x={stop_x_r:.1f} is too close to "
                    f"left edge (min={min_x_right_stop:.1f})"
                )
                stop_line_right = None
                label_stop_line_right = None
            # Right stop must be RIGHT of crossing center (x > crossing_x)
            elif crossing_center is not None:
                crossing_x = crossing_center[0]
                if stop_x_r < crossing_x:
                    print(
                        f"RIGHT stop rejected: x={stop_x_r:.1f} is left of "
                        f"crossing x={crossing_x:.1f}"
                    )
                    stop_line_right = None
                    label_stop_line_right = None

        # Left stop line: should be above ego and below opp
        if stop_line_left is not None and crossing_center is not None:
            x1_l, y1_l, x2_l, y2_l = stop_line_left[0]
            stop_y_l = (y1_l + y2_l) / 2.0
            stop_x_l = (x1_l + x2_l) / 2.0
            crossing_y = crossing_center[1]
            crossing_x = crossing_center[0]
            # Left stop must be LEFT of crossing center (x < crossing_x)
            if stop_x_l > crossing_x:
                print(
                    f"LEFT stop rejected: x={stop_x_l:.1f} is right of "
                    f"crossing x={crossing_x:.1f}"
                )
                stop_line_left = None
                label_stop_line_left = None
            # Left stop must have sufficient space on the right (at most 40%)
            elif stop_x_l > max_x_left_stop:
                print(
                    f"LEFT stop rejected: x={stop_x_l:.1f} is too close to "
                    f"right edge (max={max_x_left_stop:.1f})"
                )
                stop_line_left = None
                label_stop_line_left = None
            # Left stop's lowest point (max y) cannot be above crossing
            # center
            else:
                max_y_l = max(y1_l, y2_l)
                if max_y_l < crossing_y:
                    stop_line_left = None
                    label_stop_line_left = None
                else:
                    self._stop_left_y = stop_y_l

        # Check for horizontal lines at the endpoints of stop lines
        # If any endpoint has 0 horiz lines, reject the stop line
        right_valid = self.check_stop_line_endpoints_for_horizontals(
            stop_line_right, "RIGHT", horiz
        )
        left_valid = self.check_stop_line_endpoints_for_horizontals(
            stop_line_left, "LEFT", horiz
        )

        # Reject stop lines if endpoint check failed
        if right_valid is False:
            print("RIGHT stop rejected: no horizontal lines at endpoint")
            stop_line_right = None
            label_stop_line_right = None

        if left_valid is False:
            print("LEFT stop rejected: no horizontal lines at endpoint")
            stop_line_left = None
            label_stop_line_left = None

        # Validate stop line pair plausibility
        # Both stop lines should be present and well-aligned to be valid
        stop_line_left, stop_line_right = self.check_stop_line_pair_plausibility(
            stop_line_left,
            stop_line_right,
            max_y_diff=300.0,
            min_y_diff=100,
            max_x_separation=360.0,
            min_x_separation=280.0,
        )
        # Reset labels if pairs were invalidated
        if stop_line_left is None:
            label_stop_line_left = None
            self._stop_left_y = None
        if stop_line_right is None:
            label_stop_line_right = None

        lines = self.filter_by_length(lines, min_length=100)

        # vert, horiz = self.filter_by_angle(lines)
        vert, horiz = self.filter_by_angle(
            fused_lines,
            anchor_angle=closest_line_angle,
            anchor_tolerance=10.0,
            tol_deg=5,
        )

        # Initialize variables for ego/opp lines
        ego_line_long = None
        opp_line_long = None
        opp_line_extended = None
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
                # Dotted: wr >= 10%, Solid: wr >= 30% (percentage-based)
                min_wr = 13.0 if ego_dotted else 23.0
                print(f"EGO LINE: g={ego_gap_count} wr={wr_ego:.1f}%")

                # Check both left and right extensions
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

                if wr_ego >= min_wr and ego_extension_check_passed:
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
                # Dotted: wr >= 10%, Solid: wr >= 30% (percentage-based)
                min_wr = 13.0 if opp_dotted else 23.0
                print(f"OPP LINE: g={opp_gap_count} wr={wr_opp:.1f}%")

                opp_line_extended = None
                # Check if line passes validation and extension test
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

                if wr_opp >= min_wr and opp_extension_check_passed:
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

            print(f"Detected ego line: {label_ego}, opp line: {label_opp}")

            if ego_line_long is not None and opp_line_long is not None:
                pair_plausible = self.check_plausibility_horizontal_line_pair(
                    opp_line_long, ego_line_long, crossing_center
                )

            # Plausibility check for right stop line against opp line
            # Right stop should be below (larger y) opp line
            if stop_line_right is not None and opp_line_long is not None:
                x1_o, y1_o, x2_o, y2_o = opp_line_long[0]
                opp_y = (y1_o + y2_o) / 2.0
                x1_r, y1_r, x2_r, y2_r = stop_line_right[0]
                stop_y_r = (y1_r + y2_r) / 2.0
                # Right stop should not be above opp line
                if stop_y_r < opp_y:
                    stop_line_right = None
                    label_stop_line_right = None

            # Plausibility check for left stop line
            # Should be above ego line and below opp line
            if stop_line_left is not None and self._stop_left_y is not None:
                if ego_line_long is not None:
                    x1_e, y1_e, x2_e, y2_e = ego_line_long[0]
                    ego_y = (y1_e + y2_e) / 2.0
                    # Left stop should be above (smaller y) ego line
                    if self._stop_left_y >= ego_y:
                        stop_line_left = None
                        label_stop_line_left = None

                if stop_line_left is not None and opp_line_long is not None:
                    x1_o, y1_o, x2_o, y2_o = opp_line_long[0]
                    opp_y = (y1_o + y2_o) / 2.0
                    # Left stop should be below (larger y) opp line
                    if self._stop_left_y <= opp_y:
                        stop_line_left = None
                        label_stop_line_left = None

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
            detected_corners=detected_corners,
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
            closest_line_angle=closest_line_angle,
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

        # Draw stop line pair validation metrics when both lines are valid
        if stop_line_left is not None and stop_line_right is not None:
            x1_l, y1_l, x2_l, y2_l = stop_line_left[0]
            x1_r, y1_r, x2_r, y2_r = stop_line_right[0]

            y_left = (y1_l + y2_l) / 2.0
            y_right = (y1_r + y2_r) / 2.0
            x_left = (x1_l + x2_l) / 2.0
            x_right = (x1_r + x2_r) / 2.0

            # Calculate metrics
            y_diff = abs(y_left - y_right)
            x_sep = abs(x_right - x_left)

            # Draw vertical separation line between stop lines
            y_mid = int((y_left + y_right) / 2.0)
            x_left_int = int(x_left)
            x_right_int = int(x_right)
            cv2.line(
                debug_image,
                (x_left_int, y_mid),
                (x_right_int, y_mid),
                CYAN,
                1,
            )

            # Draw vertical diffs at each line (small vertical lines)
            diff_line_height = 20
            cv2.line(
                debug_image,
                (x_left_int, int(y_left) - diff_line_height),
                (x_left_int, int(y_left) + diff_line_height),
                YELLOW,
                1,
            )
            cv2.line(
                debug_image,
                (x_right_int, int(y_right) - diff_line_height),
                (x_right_int, int(y_right) + diff_line_height),
                YELLOW,
                1,
            )

            # Render the metrics as text
            metrics_y = 100
            metrics_text_color = GREEN
            cv2.putText(
                debug_image,
                f"y_diff={y_diff:.1f}px",
                (debug_image.shape[1] - 200, metrics_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                metrics_text_color,
            )
            cv2.putText(
                debug_image,
                f"x_sep={x_sep:.1f}px",
                (debug_image.shape[1] - 200, metrics_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                metrics_text_color,
            )

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

        # Add detections to aggregator
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
        )

        # Check if aggregation is complete
        crossing_type = self.intersection_aggregator.get_crossing_type()
        self.get_logger().info(f"Aggregated crossing type: {crossing_type}")

        # Draw crossing type visualization in top right corner
        self._draw_crossing_type_visualization(debug_image, crossing_type)

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
