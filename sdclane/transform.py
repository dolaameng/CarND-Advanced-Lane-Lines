"""Transformations used in lane estimation, e.g.
- crop an ROI in the image
- warp an image into bird-eye view by perspective transform
"""

from . import config
from .utility import make_pipeline, read_rgb_imgs
from .camera import build_undistort_function
from .line_detection import build_default_detect_lines_function

import cv2
import numpy as np

from skimage.measure import LineModel, ransac
from skimage.morphology import binary_closing, disk
from skimage.measure import label, regionprops

from sklearn.cluster import KMeans

def build_roi_crop_function(roi, roi_value=1):
    """Return a function that can be used a step in pipleline
    """

    def crop_roi(bin_img):
        """Crop the roi region of an image, by masking out the other regions.
        - `bin_img`: a binary image, e.g. an image with detected lines.
        - `roi`: region of interest and other areas will be masked out
        - `roi_value`: pixel values to fill the ROI, 1 for binary image, and 255 for gray
        """
        mask = np.zeros_like(bin_img, dtype=np.int32)
        cv2.fillPoly(mask, roi, roi_value)
        masked_img = (bin_img & mask)
        return masked_img
    return crop_roi


def build_trapezoidal_bottom_roi_crop_function():
    """
    """
    test_img = read_rgb_imgs([config.warp_estimate_img])[0]
    H, W = test_img.shape[:2]
    trapezoidal_roi = np.array([[
        (40,H), 
        (W/2-40, H/2+80), 
        (W/2+40, H/2+80),
        (W-40,H)]], dtype=np.int32)
    return build_roi_crop_function(trapezoidal_roi, roi_value=1)


class PerspectiveTransformer(object):
    """Perspective Transformation, specially for bird-eye view.
    It fits on a binary image with lines detected, e.g., from result of 
    a LineDetector pipeline, and estimates the transform matrix.
    It transforms new images based on the estimated transform matrix.
    It also estimates/fix-codes the meter-per-pixel on x/y axis.
    """
    def __init__(self, forward_distance=None):
        """Constructor
        `forward_distance`: how far to look forward when estimate the tranform, 
            with the two lane boundaries, it should be roughly a trapzoid.
            if it is None, it will be estimated as half the image height plus a fixed bias.
        `PerspectiveTransformer.M`: transform matrix
        """
        self.M = None
        self.forward_distance = forward_distance
        # meters per pixel on y axis
        self.y_mpp = None
        # meters per pixel on x axis will be estimated later based on the image.
        self.x_mpp = None
    def fit(self, line_img):
        """Estimate the tranform matrix self.M based on a binary image
        with line detected.
        - `line_img`: image with two lines detected, representing the 
        left and right boundaries of lanes. In the transformed
        bird-eye view, the two boundaries should be roughly parallel.
        """
        # image shape
        H, W = line_img.shape[:2]
        # find line coordinates
        ys, xs = np.where(line_img > 0)
        # clustering of two lines
        cluster2 = KMeans(2)
        cluster2.fit(np.c_[xs, ys])
        # build robust linear model for each line
        linear_models = []
        for c in [0, 1]:
            i = (cluster2.labels_ == c)

            robust_model, inliers = ransac(np.c_[xs[i], ys[i]], LineModel, 
                                        min_samples=2, residual_threshold=1., max_trials=500)
            linear_models.append(robust_model)
        # get the vertices of a trapezoid as source points
        if self.forward_distance is None:
            middle_h = H/2 + 100#160
        else:
            middle_h = H - self.forward_distance
        line0 = [(linear_models[0].predict_x(H), H), (linear_models[0].predict_x(middle_h), middle_h)]
        line1 = [(linear_models[1].predict_x(H), H), (linear_models[1].predict_x(middle_h), middle_h)]
        src_pts = np.array(line0 + line1, dtype=np.float32)
        # get the vertices of destination points
        # here simply map it to a rect with same width/length from bottom
        bottom_x1, bottom_x2 = line0[0][0], line1[0][0]
        
        v = np.array(line0[1]) - np.array(line0[0])
        # it must be the same as source trapzoid length otherwise y_mpp will change
        L = H#int(np.sqrt(np.sum( v*v ))) #H
        dst_pts = np.array([(bottom_x1, H), (bottom_x1, H-L),
                           (bottom_x2, H), (bottom_x2, H-L)], 
                          dtype=np.float32)
        # estimate the transform matrix
        self.M =  cv2.getPerspectiveTransform(src_pts, dst_pts)
        # estimate meter-per-pixel in the transformed image
        self.x_mpp = 3 / np.abs(bottom_x1-bottom_x2)
        self.y_mpp = self.estimate_ympp(line_img)
        return self


    def transform(self, img):
        """`img`: color image with shape (H, W, C) and dtype np.uint8.
        Use `np.dstack` to stack binary/gray image before `transform` them.s
        """
        H, W = img.shape[:2]
        warped_img = cv2.warpPerspective(img, self.M, (W, H), flags=cv2.INTER_LINEAR)
        return warped_img

    def binary_transform(self, binary_img):
        """Transform a binary image (e.g., output of a line detection). 
        it seems `cv2.warpPerspective` doesn't work on that directly, so 
        convert it to a three-channel image first and convert it back
        after the transform
        """
        assert binary_img.ndim == 2
        # convert the binary to color image
        img = (np.dstack([binary_img, ]*3) * 255).astype(np.uint8)
        warped_img = self.transform(img)
        # convert it back to the binary image
        warped_img = (warped_img[:,:,0] > 0).astype(np.bool)
        return warped_img

    
    def estimate_ympp(self, line_img):
        """This implemenation of estimation meter-per-pixel on y axis
        is very hard-coded and image dependent. Actually I am not aware
        of a general way of doing it.
        """
        # warp line_img by the estimated perspective transform 
        warped_img = self.binary_transform(line_img)
        
        # close the gaps in image, focus on the right part that
        # has the dotted line segments
        right_lane_img = binary_closing(warped_img, selem=disk(5))
        right_lane_img = right_lane_img[:,1000:]

        # find the regions, find the length of dot-lane segment
        # as the max-axis-length of the larget region
        regions = regionprops(label(right_lane_img))
        max_len = max([r.major_axis_length for r in regions])
        
        # that segment is 3 meters in reality
        y_mpp = 3 / max_len
        return y_mpp



def build_default_warp_transform_function():
    test_img = read_rgb_imgs([config.warp_estimate_img])[0]
    img_pipe = make_pipeline([
        build_undistort_function(), 
        build_default_detect_lines_function(), 
        build_trapezoidal_bottom_roi_crop_function()])
    cropped_line_img = img_pipe(test_img)

    pt = PerspectiveTransformer().fit(cropped_line_img)

    return (pt.transform, pt.binary_transform, pt.x_mpp, pt.y_mpp)


