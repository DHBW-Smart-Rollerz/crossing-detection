"""
Helper functions for crossing detection.

This module contains utility functions like normalize_line that are shared
across the crossing detection module.
"""

import cv2
import numpy as np


def normalize_line(line):
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


def normalize_lines(lines):
    """Normalize an iterable of lines to a list of numpy (1,4) arrays."""
    if lines is None or (hasattr(lines, "__len__") and len(lines) == 0):
        return []
    normalized = []
    for ln in lines:
        nl = normalize_line(ln)
        if nl is not None:
            normalized.append(nl)
    return normalized
