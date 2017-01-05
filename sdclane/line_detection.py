"""
Module to detect lines in general, by using different models. See 
class `LineDetector` below for details.
Line detection will be the first step for many lane-estimation tasks.
"""


from .utility import make_pipeline, AND, OR

import cv2
import numpy as np

class LineDetector(object):
    """Detect pixels of lines in an image. Those pixels are usually part of
    lanes to be detected by downstream methods.
    """
    def __init__(self):
        pass
    def transform(self, img):
        """`img`: Original RGB image
        Return:
        `line_img`: boolean image with pixels on lines as 1 and others as 0
        """
        return img[:, :, 0] > 125
    def update(self, img):
        """incremental learning??
        """
        pass


def build_default_line_detector():
    return LineDetector()