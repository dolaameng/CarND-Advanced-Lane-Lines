"""
Camera calibration based on a set of chessboard images
"""

from . import config
from . import utility

import numpy as np
import cv2

class CameraCalibrator(object):
    """Calibrate camera by estimating the distortion
    matrix and coefficients.
    """
    def __init__(self):
        """Class members:
        `self.M`: distortion matrix
        `self.d`: distortion coefficients
        """
        self.M = None
        self.d = None
    def fit(self, chessboard_images, nx=9, ny=6, img_size=None):
        """Estimate distortion matrix and coefficients
        by training on a set of chessboard images.
        `chessboard_images`: list of RGB images, assuming they
          are from the same chessboard and same camera, and of
          roughly the same size.
        `nx`,`ny`: number of corners in x and y directions
        """
        if img_size == None:
            h, w = chessboard_images[0].shape[:2]
            img_size = (w, h)
        objp = np.array([(x, y, 0) for y in range(ny) for x in range(nx)], dtype=np.float32)
        objpts, imgpts = [], []
        for img in chessboard_images:
            # convert to gray and find corner points
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret:
                objpts.append(objp)
                imgpts.append(corners)
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            else:
                # ignore images that cannot be used
                print("Cannot find corner points in image")
        ret, self.M, self.d, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, img_size, None, None)
        return self
    def save(self, model_file):
        model = {"M": self.M, "d": self.d}
        pickle.dump(model, open(model_file, "wb"))
        print("calibration model saved at %s" % model_file)
        return self
    def restore(self, model_file):
        model = pickle.load(open(model_file, "rb"))
        self.M = model["M"]
        self.d = model["d"]
        print("calibration model restored from %s" % model_file)
        return self
    def undistort(self, img):
        """img: original RGB img to be undistorted
        Return undistorted image.
        """
        undist = cv2.undistort(img, self.M, self.d, None, self.M)
        return undist

def build_undistort_function():
    """Build a undistort() function that 
    can be used in a pipeline, which takes 
    an original image and returns its undistorted version. 
    """
    cc = CameraCalibrator()
    cc.fit(utility.read_rgb_imgs(config.camera_calibr_img_files))
    return cc.undistort