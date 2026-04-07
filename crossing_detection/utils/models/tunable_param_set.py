"""Tunable parameter set for crossing detection pipeline."""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TunableParamSet:
    """
    Container for tunable pipeline parameters.

    This class retrieves and holds all configurable parameters used in the
    crossing detection pipeline, making them easily accessible. It handles
    parameter loading directly from the ROS2 node configuration.
    """

    def __init__(self, node):
        """
        Initialize tunable parameters by retrieving them from the node.

        Arguments:
            node -- The ROS2 node to retrieve parameters from
        """
        self._load_params(node)

    def _load_params(self, node):
        """
        Load all tunable parameters from the node configuration.

        Arguments:
            node -- The ROS2 node to retrieve parameters from
        """
        logger.info("Loading tunable parameters")

        def get_param(param_name, default):
            try:
                return node.get_parameter(param_name).value
            except Exception:
                return default

        self.fuse_lines_distance_tolerance = get_param(
            "fuse_lines_distance_tolerance", 80
        )
        self.min_wr_dotted_ego = get_param("min_wr_dotted_ego", 20.0)
        self.min_wr_solid_ego = get_param("min_wr_solid_ego", 35.0)
        self.min_wr_solid_ego_angled = get_param("min_wr_solid_ego_angled", 20.0)
        self.min_wr_opp = get_param("min_wr_opp", 35.0)
        self.min_wr_opp_angled = get_param("min_wr_opp_angled", 28.0)
        self.min_wr_stop_left_solid = get_param("min_wr_stop_left_solid", 30)
        self.min_wr_stop_left_dotted = get_param("min_wr_stop_left_dotted", 25)
        self.min_wr_stop_right_solid = get_param("min_wr_stop_right_solid", 30)
        self.min_wr_stop_right_dotted = get_param("min_wr_stop_right_dotted", 25)
        self.min_gap_count_dotted = get_param("min_gap_count_dotted", 3)
        self.clip_ego_adaptive_min_rel = get_param("clip_ego_adaptive_min_rel", 0.5)
        self.clip_ego_adaptive_max_rel = get_param("clip_ego_adaptive_max_rel", 0.75)
        self.clip_opp_adaptive_min_rel = get_param("clip_opp_adaptive_min_rel", 0.1)
        self.clip_opp_adaptive_max_rel = get_param("clip_opp_adaptive_max_rel", 0.5)
        self.allowed_corner_error_cc_rect = get_param(
            "allowed_corner_error_cc_rect", 30.0
        )
        self.tolerance_angle_cc_rect = get_param("tolerance_angle_cc_rect", 20.0)
        self.openness_black_pixel_pct_threshold = get_param(
            "openness_black_pixel_pct_threshold", 55.0
        )
        self.left_stop_line_min_thickness = get_param(
            "left_stop_line_min_thickness", 18
        )
        self.right_stop_line_min_thickness = get_param(
            "right_stop_line_min_thickness", 18
        )

        for param_name, param_value in self.__dict__.items():
            logger.info(f"Loaded parameter: {param_name} = {param_value}")
