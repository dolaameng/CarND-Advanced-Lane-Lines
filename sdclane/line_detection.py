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
    def __init__(self, ksize=11):
        """`ksize`: kernel size
        """
        self.ksize = ksize
    def get_sobel(self, gray_img, ksize):
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=ksize)
        sobeld = np.arctan(sobely / (sobelx + 1e-6))

        def normalize(sobel):
            sobel = np.absolute(sobel)
            return (sobel * 255. / sobel.max()).astype(np.uint8) 
        return normalize(sobelx), normalize(sobely), sobeld

    def filter_by(self, gray_img, lower, upper):
        return (gray_img >= lower) & (gray_img <= upper)

    def detect(self, img):
        """`img`: Original RGB image
        Return:
        `line_img`: boolean image with pixels on lines as 1 and others as 0
        """
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_img = hls_img[:, :, 1]
        s_img = hls_img[:, :, 2]
        # combine l and s channel
        gray = (l_img * 0.6 + s_img * 0.4).astype(np.uint8)


        s_x, s_y, s_d = self.get_sobel(gray, self.ksize)
        s_bin = (self.filter_by(s_x, 25, 125) 
                    & (self.filter_by(s_d, 0.5, 1.2) # left line
                        | self.filter_by(s_d, -1.5, -0.3))) # right line

        return s_bin


def build_line_detect_function():
    return LineDetector().detect