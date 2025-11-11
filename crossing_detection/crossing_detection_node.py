import os

import cv2
import cv_bridge
import numpy as np
import rclpy
import sensor_msgs.msg
import std_msgs.msg
from smarty_utils.enums import NodeState
from smarty_utils.smarty_node import SmartyNode
from timing import timer


class IntersectionDetector(SmartyNode):
    """ROS2 Example Node."""

    DBG_IMG_DIR = "/home/smartrollerz/Desktop/smartrollers/smarty_workspace/rosbag_images/rosbag2_2025_03_06-17_56_01"

    def __init__(self):
        """Initialize the ROS2ExampleNode."""
        super().__init__(
            "crossing_detection_node",
            "crossing_detection",
            node_parameters={
                # Subscriber topics
                "image_subscriber": "/camera/image/undistorted",
                # Publisher topics
                "debug_image_publisher": "/example/debug_image",
                "result_publisher": "/example/result",
                # Parameters
                "state": NodeState.ACTIVE.value,
                "image_path": "resources/img/example.png",
                "example_value": 128,
                "debug": False,
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
        self.get_logger().info(f"Package Path: {self.package_path}")

        # Create required objects
        self.cv_bridge = cv_bridge.CvBridge()

        self.get_logger().info("Crossing Detector initialized.")

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
        self.execute_prediction(msg)

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
        img = cv2.Canny(img, 100, 100)
        # IntersectionDetector.show_image("Canny", img)
        return img

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
            img_edges, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=10
        )
        i = True
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0) if i else (0, 50, 255), 2)
            i = False
            transformed.append(((x1, y1), (x2, y2)))
        # cv2.imwrite("houghlines.jpg", img2)
        IntersectionDetector.show_image("Hough Lines", img2)
        return transformed

    def pipeline(self, img_path: str):
        """
        Complete processing pipeline for intersection detection.

        Arguments:
            img_path -- Path to the input image.
        """
        image = IntersectionDetector.load_img_grayscale(img_path)
        edges = self.perform_canny(image)
        transformed_lines = self.hough_transformation(image, edges)
        print(transformed_lines[0])


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
                for img in os.listdir(IntersectionDetector.DBG_IMG_DIR):
                    node.pipeline(os.path.join(IntersectionDetector.DBG_IMG_DIR, img))
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
