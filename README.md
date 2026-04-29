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

## Results
- Top-1 accuracy: 66%
- Top-10 accuracy: 77%
- Strong performance on structured objects (e.g. books), weaker on ambiguous scenes

## Key Learnings
- Feature-based methods are effective for object matching without deep learning
- RANSAC improves robustness by removing outliers
- Performance depends heavily on image quality and distinct features
