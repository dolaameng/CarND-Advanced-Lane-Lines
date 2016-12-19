"""
Configuration of lane-detection module, 
e.g., images used to images, estimate perspective transform and etc.
"""

from glob import glob
import cv2

# images for camera calibration
camera_calibr_img_files = glob("camera_cal/*.jpg")

# images for visual check - samples from video
test_img_files = glob("test_images/*.jpg")

# image used to estimate the perspective transform
warp_estimate_img = test_img_files[3]
