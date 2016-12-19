"""Detect lane boundaries in an image.
"""

from . import config
from camera import build_undistort_function
from line_detection import build_default_detect_lines_function
from transform import build_trapezoidal_bottom_roi_crop_function
from transform import build_default_warp_transform_function

import numpy as np

class LaneDetector(object):
	def __init__(self):
		pass
	def detect_lane(self, img):
		pass