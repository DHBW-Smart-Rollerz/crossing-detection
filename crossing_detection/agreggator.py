from collections import deque

import numpy as np


class IntersectionAggregator:
    """
    Aggregates intersection detection results using a confidence buffer system.

    Each line type has a confidence buffer (0.0-1.0) that:
    - Increments when detected (different increments per line type)
    - Decays when not detected (different decay rates per line type)
    - Becomes "valid" when buffer >= threshold

    Line types:
    - ego: increment 0.15, decay 0.05, threshold 0.4
    - opp: increment 0.25, decay 0.08, threshold 0.35
    - stop_left: increment 0.4, decay 0.12, threshold 0.3
    - stop_right: increment 0.4, decay 0.12, threshold 0.3
    """

    def __init__(self, max_frames=7):
        """
        Initialize the aggregator with buffer system.

        Args:
            max_frames: Maximum frames before reset (legacy, kept for compatibility)
        """
        self.max_frames = max_frames
        self.frame_count = 0
        self.first_detection_frame = None

        self.ego_buffer = 0.0
        self.opp_buffer = 0.0
        self.stop_left_buffer = 0.0
        self.stop_right_buffer = 0.0

        self.ego_line = None
        self.opp_line = None
        self.stop_line_left = None
        self.stop_line_right = None

        self.ego_dotted = None
        self.opp_dotted = None
        self.stop_left_dotted = None
        self.stop_right_dotted = None

        self.buffer_config = {
            "ego": {
                "increment": 0.18,
                "decay": 0.09,
                "threshold": 0.30,
            },
            "opp": {
                "increment": 0.25,
                "decay": 0.05,
                "threshold": 0.30,
            },
            "stop_left": {
                "increment": 0.4,
                "decay": 0.10,
                "threshold": 0.3,
            },
            "stop_right": {
                "increment": 0.4,
                "decay": 0.10,
                "threshold": 0.3,
            },
        }

        self.last_states = []

        self.history_size = 4
        self.ego_history = deque(maxlen=self.history_size)
        self.opp_history = deque(maxlen=self.history_size)
        self.stop_left_history = deque(maxlen=self.history_size)
        self.stop_right_history = deque(maxlen=self.history_size)

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
        ego_angle=None,
        opp_angle=None,
    ):
        """
        Add detection results and update buffers for current frame.

        Uses Exponential Moving Average (EMA) for smooth buffer updates
        and tracks multi-frame history for stability scoring.

        Buffers increment when line is detected, decay when not.
        Each line type has different increment/decay/threshold.

        If a line is not straight (angle != 0 or 90), add bonus.

        Args:
            ego_line: Ego line data (numpy array) or None
            opp_line: Opp line data (numpy array) or None
            stop_line_left: Left stop line data or None
            stop_line_right: Right stop line data or None
            ego_dotted: True if ego is dotted, False if solid
            opp_dotted: True if opp is dotted, False if solid
            stop_dotted_left: True if left stop is dotted
            stop_dotted_right: True if right stop is dotted
            ego_angle: Prominent angle of ego line (degrees) or None
            opp_angle: Prominent angle of opp line (degrees) or None
        """
        if self.first_detection_frame is None and any(
            [
                ego_line is not None,
                opp_line is not None,
                stop_line_left is not None,
                stop_line_right is not None,
            ]
        ):
            self.first_detection_frame = 0

        if ego_line is not None:
            self.ego_line = ego_line.copy()
            self.ego_dotted = ego_dotted
            cfg = self.buffer_config["ego"]
            increment = cfg["increment"]

            if ego_angle is not None:
                is_straight = ego_angle < 5 or ego_angle > 85
                if not is_straight:
                    increment = increment * 1.8  # 80% bonus

            self.ego_buffer = min(1.0, self.ego_buffer + increment)
        else:
            cfg = self.buffer_config["ego"]
            self.ego_buffer = max(0.0, self.ego_buffer - cfg["decay"])

        if opp_line is not None:
            self.opp_line = opp_line.copy()
            self.opp_dotted = opp_dotted
            cfg = self.buffer_config["opp"]
            increment = cfg["increment"]

            if opp_angle is not None:
                is_straight = opp_angle < 5 or opp_angle > 85
                if not is_straight:
                    increment = increment * 1.65  # 65% bonus

            self.opp_buffer = min(1.0, self.opp_buffer + increment)
        else:
            cfg = self.buffer_config["opp"]
            self.opp_buffer = max(0.0, self.opp_buffer - cfg["decay"])

        if stop_line_left is not None:
            self.stop_line_left = stop_line_left.copy()
            self.stop_left_dotted = stop_dotted_left
            cfg = self.buffer_config["stop_left"]
            increment = cfg["increment"]
            self.stop_left_buffer = min(1.0, self.stop_left_buffer + increment)
        else:
            cfg = self.buffer_config["stop_left"]
            self.stop_left_buffer = max(0.0, self.stop_left_buffer - cfg["decay"])

        if stop_line_right is not None:
            self.stop_line_right = stop_line_right.copy()
            self.stop_right_dotted = stop_dotted_right
            cfg = self.buffer_config["stop_right"]
            increment = cfg["increment"]
            self.stop_right_buffer = min(1.0, self.stop_right_buffer + increment)
        else:
            cfg = self.buffer_config["stop_right"]
            self.stop_right_buffer = max(0.0, self.stop_right_buffer - cfg["decay"])

        self.ego_history.append(self.ego_buffer)
        self.opp_history.append(self.opp_buffer)
        self.stop_left_history.append(self.stop_left_buffer)
        self.stop_right_history.append(self.stop_right_buffer)

        self.frame_count += 1

    def get_buffer_levels(self):
        """
        Get current confidence buffer levels for all line types.

        Useful for debugging and visualization.

        Returns:
            Dictionary with buffer levels (0.0-1.0) for each line type
        """
        return {
            "ego": self.ego_buffer,
            "opp": self.opp_buffer,
            "stop_left": self.stop_left_buffer,
            "stop_right": self.stop_right_buffer,
        }

    def get_current_configuration(self):
        """
        Get current intersection configuration based on buffers.

        Lines are valid if their buffer >= threshold.

        Returns:
            Dictionary with 'ego_line', 'opp_line',
            'stop_line_left', 'stop_line_right', buffer levels,
            frame_count, and is_complete flag.
        """
        config = {
            "ego_line": (
                self.ego_line
                if self.ego_buffer >= self.buffer_config["ego"]["threshold"]
                else None
            ),
            "opp_line": (
                self.opp_line
                if self.opp_buffer >= self.buffer_config["opp"]["threshold"]
                else None
            ),
            "stop_line_left": (
                self.stop_line_left
                if self.stop_left_buffer >= self.buffer_config["stop_left"]["threshold"]
                else None
            ),
            "stop_line_right": (
                self.stop_line_right
                if self.stop_right_buffer
                >= self.buffer_config["stop_right"]["threshold"]
                else None
            ),
            "frame_count": self.frame_count,
            "buffer_levels": self.get_buffer_levels(),
            "is_complete": all(
                [
                    self.ego_line is not None
                    and self.ego_buffer >= self.buffer_config["ego"]["threshold"],
                    self.opp_line is not None
                    and self.opp_buffer >= self.buffer_config["opp"]["threshold"],
                    self.stop_line_left is not None
                    and self.stop_left_buffer
                    >= self.buffer_config["stop_left"]["threshold"],
                    self.stop_line_right is not None
                    and self.stop_right_buffer
                    >= self.buffer_config["stop_right"]["threshold"],
                ]
            ),
        }

        return config

    def get_crossing_type(self):
        """
        Generate crossing type string based on current buffer states.

        Format: es-od-ln-rn
        - e: ego (es=solid, ed=dotted, en=none)
        - o: opp (os=solid, od=dotted, on=none)
        - l: left stop (ls=solid, ld=dotted, ln=none)
        - r: right stop (rs=solid, rd=dotted, rn=none)

        Returns:
            String in format "es-od-ln-rn" representing the crossing
        """
        ego_valid = self.ego_buffer >= self.buffer_config["ego"]["threshold"]
        opp_valid = self.opp_buffer >= self.buffer_config["opp"]["threshold"]
        stop_left_valid = (
            self.stop_left_buffer >= self.buffer_config["stop_left"]["threshold"]
        )
        stop_right_valid = (
            self.stop_right_buffer >= self.buffer_config["stop_right"]["threshold"]
        )

        if not ego_valid:
            ego_type = "en"
        elif self.ego_dotted is True:
            ego_type = "ed"
        else:
            ego_type = "es"

        if not opp_valid:
            opp_type = "on"
        elif self.opp_dotted is True:
            opp_type = "od"
        else:
            opp_type = "os"

        if not stop_left_valid:
            left_stop_type = "ln"
        elif self.stop_left_dotted is True:
            left_stop_type = "ld"
        else:
            left_stop_type = "ls"

        if not stop_right_valid:
            right_stop_type = "rn"
        elif self.stop_right_dotted is True:
            right_stop_type = "rd"
        else:
            right_stop_type = "rs"

        crossing_type_str = (
            f"{ego_type}-{opp_type}-{left_stop_type}" f"-{right_stop_type}"
        )

        self.last_states.append(crossing_type_str)

        return crossing_type_str

    def is_crossing_stable(self, lookback=2):
        """
        Check if the crossing state is stable (same for 2+ frames).

        Compares the current state with the last 'lookback' states.
        Returns True only if all lookback states match the current state.

        Arguments:
            lookback -- Number of previous states to compare with (default 2)

        Returns:
            True if current crossing state == all previous lookback states,
            False otherwise
        """
        if len(self.last_states) < lookback + 1:
            return False

        current_state = self.last_states[-1]
        for i in range(1, lookback + 1):
            if self.last_states[-i - 1] != current_state:
                return False
        return True

    def get_overall_confidence(self):
        """
        Calculate overall intersection confidence from buffer fill states.

        Confidence is the average of all 4 buffer levels (0.0-1.0),
        weighted by stability score.

        High stability means the buffers are consistent (less noise).
        Formula: confidence = average_buffer * stability_weight

        Returns:
            Float between 0.0-1.0 representing overall confidence
        """
        buffer_levels = self.get_buffer_levels()
        stability_score = self.get_stability_score()

        buffers = [
            buffer_levels["ego"],
            buffer_levels["opp"],
            buffer_levels["stop_left"],
            buffer_levels["stop_right"],
        ]
        buffers = [b for b in buffers if b > 0.0]
        base_confidence = sum(buffers) / len(buffers) if buffers else 0.0

        stability_weight = 0.85 + 0.15 * stability_score["overall"]

        overall_confidence = base_confidence * stability_weight
        return min(1.0, overall_confidence)

    def get_stability_score(self):
        """
        Calculate stability score based on multi-frame buffer history.

        Stability measures how consistent the buffers have been over the
        last 7 frames. A high stability score indicates low jitter/noise.

        Formula: stability = 1 - (std_dev / mean)
        - If std_dev is low relative to mean → stable (close to 1.0)
        - If std_dev is high relative to mean → unstable (close to 0.0)

        Returns:
            Dictionary with stability scores for each line type (0.0-1.0)
        """
        blank = 0

        def calculate_stability(history):
            """Calculate stability for a single buffer history."""
            if len(history) < 2:
                return 1.0

            history_array = np.array(list(history))
            mean_val = np.mean(history_array)

            if mean_val < 0.01:
                return 1.0

            std_dev = np.std(history_array)
            cv = std_dev / mean_val

            stability = max(0.0, min(1.0, 1.0 - cv))
            return stability

        ego_stability = calculate_stability(self.ego_history)
        opp_stability = calculate_stability(self.opp_history)
        stop_left_stability = calculate_stability(self.stop_left_history)
        stop_right_stability = calculate_stability(self.stop_right_history)

        return {
            "ego": ego_stability,
            "opp": opp_stability,
            "stop_left": stop_left_stability,
            "stop_right": stop_right_stability,
            "overall": (
                ego_stability
                + opp_stability
                + stop_left_stability
                + stop_right_stability
            )
            / 4,
        }
