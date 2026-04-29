# Image Matching and Retrieval (Computer Vision)
Computer Vision Assignment

## Overview
This project implements image matching and retrieval that identifies objects across images and retrieves visually similar images from a dataset.

## Approach
- Used OpenCV
- Feature-based pipeline:
  - ORB keypoint detection and descriptors
  - KNN matching with ratio test
  - RANSAC-based homography for geometric verification
- Ranked results using inlier matches

## Technologies Used
- Python
- OpenCV
- NumPy
