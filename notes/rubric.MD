PROJECT SPECIFICATION
Advanced Lane Finding

Camera Calibration

CRITERIA
MEETS SPECIFICATIONS
Have the camera matrix and distortion coefficients been computed correctly and checked on the calibration test image?

OpenCV functions or other methods were used to calculate the correct camera matrix and distortion coefficients using the calibration chessboard images provided in the repository. The distortion matrix should be used to un-distort the test calibration image provided as a demonstration that the calibration is correct

Pipeline (single images)

CRITERIA
MEETS SPECIFICATIONS
Has the distortion correction been correctly applied to each image?

Distortion correction that was calculated via camera calibration has been correctly applied to each image.

Has a binary image been created using color transforms, gradients or other methods?

At least two methods (i.e., color transforms, gradients) have been combined to create a binary image containing likely lane pixels. There is no "ground truth" here, just visual verification that the pixels identified as part of the lane lines are, in fact, part of the lines.

Has a perspective transform been applied to rectify the image?

OpenCV function or other method has been used to correctly rectify each image to a "birds-eye view"

Have lane line pixels been identified in the rectified image and fit with a polynomial?

Methods have been used to identify lane line pixels in the rectified binary image. The left and right line have been identified and fit with a curved functional form (e.g., spine or polynomial).

Having identified the lane lines, has the radius of curvature of the road been estimated? And the position of the vehicle with respect to center in the lane?

Here the idea is to take the measurements of where the lane lines are and estimate how much the road is curving and where the vehicle is located with respect to the center of the lane. The radius of curvature may be given in meters assuming the curve of the road follows a circle and the position of the vehicle within the lane may be given as meters off of center.

Has the result from lane line detection been warped back to the original image space and displayed?

The fit from the rectified image has been warped back onto the original image and plotted to identify the lane boundaries. This should demonstrate that the lane boundaries were correctly identified.

Pipeline (video)

CRITERIA
MEETS SPECIFICATIONS
Does the pipeline established with the test images work to process the video?

The image processing pipeline that was established to find the lane lines in images successfully processes the video. The output here should be a new video where the lanes are identified in every frame, and outputs are generated regarding the radius of curvature of the lane and vehicle position within the lane. The identification and estimation don't need to be perfect, but they should not be wildly off in any case. The pipeline should correctly map out curved lines and not fail when shadows or pavement color changes are present.

Has some kind of search method been implemented to discover the position of the lines in the first images in the video stream?

In the first few frames of video, the algorithm should perform a search without prior assumptions about where the lines are (i.e., no hard coded values to start with). Once a high-confidence detection is achieved, that positional knowledge may be used in future iterations as a starting point to find the lines.

Has some form of tracking of the position of the lane lines been implemented?

As soon as a high confidence detection of the lane lines has been achieved, that information should be propagated to the detection step for the next frame of the video, both as a means of saving time on detection and in order to reject outliers (anomalous detections).

Readme

CRITERIA
MEETS SPECIFICATIONS
Has a Readme file been included that describes in detail the steps taken to construct the pipeline, techniques used, areas where improvements could be made?

The Readme file submitted with this project includes a detailed description of what steps were taken to achieve the result, what techniques were used to arrive at a successful result, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail.