"""
Standard preprocessing pipeline for crossing detection and surface pattern detection.

For crossing and surface pattern.
"""

import cv2
import numpy as np

from crossing_detection.utils.filter import (
    filter_by_bev_black_corner,
    filter_by_length,
    filter_by_roi,
)
from crossing_detection.utils.helper import normalize_lines
from crossing_detection.utils.tools import (
    enhance_by_line_brightness,
    fuse_similar_lines,
    get_bev_black_corner_polygon,
    perform_canny,
)

lsd = cv2.createLineSegmentDetector(1)


def preprocessing_pipeline(image, tunable_params):
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
    transformed_lines = lsd.detect(edges)[0]
    transformed_lines = normalize_lines(transformed_lines)

    filtered_lines = filter_by_length(transformed_lines, min_length=30)
    filtered_lines = filter_by_roi(filtered_lines, image.shape)

    dead_area_bev = get_bev_black_corner_polygon(
        image.shape,
        corner_height_rel=tunable_params.bev_dead_area_corner_height_rel,
        corner_width_rel=tunable_params.bev_dead_area_corner_width_rel,
    )
    filtered_lines = filter_by_bev_black_corner(filtered_lines, dead_area_bev)

    filtered_lines = fuse_similar_lines(
        filtered_lines,
        angle_tol_deg=10,
        center_dist_tol=tunable_params.fuse_lines_distance_tolerance,
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
