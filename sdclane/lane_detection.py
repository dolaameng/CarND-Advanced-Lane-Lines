"""Detect lane boundaries in an image.
"""

from . import config
from .camera import build_undistort_function
from .line_detection import build_default_detect_lines_function
from .transform import build_trapezoidal_bottom_roi_crop_function
from .transform import build_default_warp_transformer
from .utility import make_pipeline

import numpy as np
from skimage.morphology import remove_small_objects
import cv2

class LaneDetector(object):
    def __init__(self):
        # setup the pipe to get the line image and meter-per-pixel measurements
        # 1. undistort function 
        self.undistort = build_undistort_function()
        # 2. line detection function
        self.detect_lines = build_default_detect_lines_function()
        # 3. roi crop function
        self.roi_crop = build_trapezoidal_bottom_roi_crop_function()
        # 4. bird-eye transformer object - it's stateful
        self.transformer = build_default_warp_transformer()
    def detect_lane(self, img):
        """`img`: raw image from camera
        Returns:
            - `lane_params`: radius_of_curvature, offset from camera center
            - `marked_lane_img`: color img with lanes marked in different colors
        """
        # pipeline from raw to lane image
        undistorted_img = self.undistort(img)
        image_pipe = make_pipeline([
                    self.detect_lines,
                    self.roi_crop,
                    self.transformer.binary_transform])
        lane_img = image_pipe(undistorted_img)
        lane_params, _ = self.estimate_lane_params(lane_img, 
            self.transformer.x_mpp, self.transformer.y_mpp)
        _, models_in_pixel = self.estimate_lane_params(lane_img, 1, 1)
        marked_lane_img = self.draw_lanes(undistorted_img, models_in_pixel)
        return lane_params, marked_lane_img
    def estimate_lane_params(self, lane_img, x_mpp = 30/720, y_mpp = 3.7/700):
        """`lane_img`: binary lane image
            `x_mpp`: meter per pixel on x axis, default=1
            `y_mpp`: meter per pixel on y axis, default=1
        Returns: 
            - `l_curvature`: radius of curvature for left lane, as ${(1+(2Ay+B)^2)^{3/2}}\over{|2A|}$
            - `r_curvature`: radius of curvature for right lane
            - `center offset`: as $lane_center_x - image_center_x$ in pixels
            - `lmodel`, `mmodel`, `rmodel`: 2nd polynomial model for left, center and right lane.
        """
        H, W = lane_img.shape[:2]
        lane_img = remove_small_objects(lane_img.copy())
        window, stride = 100, 1
        lxs, lys = [], [] # lane left boundary coordinates
        mxs, mys = [], [] # lane middle coordinates
        rxs, rys = [], [] # lane right boundary coordinates
        for offset in range(0, H+1, stride):
            region = lane_img[offset:offset+window, :]
            ys, xs = np.where(region > 0)
            is_valid_region = (len(xs) > 0) and (W/3 <= (xs.max()-xs.min()) <= W*5/6)
            if is_valid_region:
                mx = (np.min(xs) + np.max(xs))/2
                my = np.mean(ys) + offset
                lx = np.min(xs)
                ly = my
                rx = np.max(xs)
                ry = my
                if len(mxs) == 0 or np.abs(mx - np.mean(mxs)) <= 100:
                    mxs.append(mx)
                    mys.append(my)
                    lxs.append(lx)
                    lys.append(ly)
                    rxs.append(rx)
                    rys.append(ry)

        lxs,lys,mxs,mys,rxs,rys = map(np.array, [lxs,lys,mxs,mys,rxs,rys])
        lmodel = np.polyfit(lys*y_mpp, lxs*x_mpp, 2)
        mmodel = np.polyfit(mys*y_mpp, mxs*x_mpp, 2)
        rmodel = np.polyfit(rys*y_mpp, rxs*x_mpp, 2)

        
        y = H*y_mpp

        A, B = lmodel[:2]
        l_curvature = (1 + (2*A*y+B)**2)**1.5 / np.abs(2*A)

        A, B = rmodel[:2]
        r_curvature = (1 + (2*A*y+B)**2)**1.5 / np.abs(2*A)

        offset = np.polyval(mmodel, y) - W*x_mpp/2
        return (l_curvature, r_curvature, offset), (lmodel, mmodel, rmodel)

    def draw_lanes(self, undistorted_img, models_in_pixel):
        lmodel, mmodel, rmodel = models_in_pixel
        y_span = range(0, undistorted_img.shape[0]+1)
        lxhat = np.polyval(lmodel, y_span).astype(np.int32)
        mxhat = np.polyval(mmodel, y_span).astype(np.int32)
        rxhat = np.polyval(rmodel, y_span).astype(np.int32)
        
        lane_layer = np.zeros_like(undistorted_img, dtype=np.uint8)
        for xs, col in zip([lxhat, mxhat, rxhat], 
                           [(255,0,0), (0,255,0), (0,0,255)]):
            pts = np.array([(x, y) for x, y in zip(xs, y_span)])
            lane_layer = cv2.polylines(lane_layer, [pts], isClosed=False, color=col, thickness=5)

        lane_layer = self.transformer.transform(lane_layer, inverse=True)
        lane_img = cv2.addWeighted(undistorted_img, 0.5, lane_layer, 1.5, 1)
        return lane_img