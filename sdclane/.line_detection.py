"""
Module to detect lines in general, by using different models. See 
class `LineDetector` below for details.
Line detection will be the first step for many lane-estimation tasks.
"""


from .utility import make_pipeline, AND, OR

import cv2
import numpy as np

class LineDetector(object):
    """Detect lines (e.g. lane boundaries) in images using a combination 
    of different methods. It supports different line detection 
    algorithms and their combinations. For now the user needs 
    to pre-configure the methods beforehand, it may support 
    automated algorithm selection based on images in the future.
    
    There are three main steps in configuring a line detector:
        - convert color images to gray, e.g, by RGB->gray or HLS->S
        - line detection, e.g., canny, sobel_x, sobel_y, sobel_magnitude, sobel_dir
        - filtering, e.g., by threshold of pixel, line oritentation, etc.
    A pipeline can be built by setting different choices at each setp.
    A combination can be done by using AND/OR operations on the resulted line(binary) images.
    The combination also has a smoothing effect like Gaussian filtering. Using a larger
    `ksize` has the same effect.
    """
    def __init__(self):
        pass
    def gray_converter(self, gray_type):
        """Create gray_image_converter for line detecion pipeline.
        `gray_type`: {"gray", "saturation", "hue"}.
        Returns a gray image converter function with RGB image input
            and a gray image output 
        """
        gray_type = gray_type.lower()
        def f(img):
            if gray_type == "gray":
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif gray_type == "saturation":
                return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 2]
            elif gray_type == "hue":
                return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 0]
            elif gray_type == "lightness":
                return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 1]
            else:
                raise ValueError("Unknown gray_type %s" % gray_type)
        return f
    def sobel_detector(self, sobel_type, ksize=3):
        """Create sobel based line dector for line detection pipeline.
        `sobel_type`: {"x", "y", "magnitude", "direction"}.
        `ksize`: kernel size for Sobel filter, default to 3
        Returns a function with gray image input and a Sobel image output, 
            the pixels of Sobel image is within [-$\pi$/2, $\pi$/2] for 'direction'
            and [0, 255] for other sobel_types
        """
        sobel_type = sobel_type.lower()
        def f(gray):
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            if sobel_type == "x":
                sobel = np.absolute(sobelx)
            elif sobel_type == "y":
                sobel = np.absolute(sobely)
            elif sobel_type == "magnitude":
                sobel = np.sqrt(sobelx*sobelx + sobely*sobely)
            elif sobel_type == "direction":
                sobel = np.arctan(sobely / (sobelx + 1e-6))
            else:
                raise ValueError("Unknown sobel_type %s" % sobel_type)
            # normalize sobel as a normal gray image, so it can be visualized
            if sobel_type in ["x", "y", "magnitude"]:
                sobel = (sobel * 255. / sobel.max()).astype(np.uint8)
            return sobel
        return f
    def threshold_filter(self, lower, upper):
        """Create binary filter for line detection pipeline.
        `lower`, `upper` are bounds for pixel value (distributed between 0, 255).
        Returns a function that returns a binary image, where pixels in [lower, upper]
        are 1 and the rest are 0.
        """
        def f(gray):
            binary = (gray >= lower) & (gray <= upper)
            return binary
        return f

def build_default_detect_lines_function():
    ld = LineDetector()

    ksize = 11
    gradx_detector1 = make_pipeline([
            ld.gray_converter("lightness")
            , ld.sobel_detector("x", ksize=ksize)
            , ld.threshold_filter(25, 125)
    ])
    left_dir_detector1 = make_pipeline([
            ld.gray_converter("lightness")
            , ld.sobel_detector("direction", ksize=ksize)
            , ld.threshold_filter(0.3, 1.5)
            
    ])
    right_dir_detector1 = make_pipeline([
            ld.gray_converter("lightness")
            , ld.sobel_detector("direction", ksize=ksize)
            , ld.threshold_filter(-1.5, -0.3)    
    ])
    line_detector1 = AND( gradx_detector1, OR(left_dir_detector1, right_dir_detector1))

    gradx_detector2 = make_pipeline([
            ld.gray_converter("saturation")
            , ld.sobel_detector("x", ksize=ksize)
            , ld.threshold_filter(25, 125)
    ])
    left_dir_detector2 = make_pipeline([
            ld.gray_converter("saturation")
            , ld.sobel_detector("direction", ksize=ksize)
            , ld.threshold_filter(0.3, 1.5)
    ])
    right_dir_detector2 = make_pipeline([
            ld.gray_converter("saturation")
            , ld.sobel_detector("direction", ksize=ksize)
            , ld.threshold_filter(-1.5, -0.3)
            
    ])
    line_detector2 = AND( gradx_detector2, OR(left_dir_detector2, right_dir_detector2))

    detect_line = OR(line_detector1, line_detector2)

    return detect_line