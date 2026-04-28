import cv2
import numpy as np
import os

def load_images(folder):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def compute_keypoints_and_descriptors(images, orb):
    kp_des = []
    for img in images:
        
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        
        kp_des.append((kp, des))
    return kp_des

def match_and_find_homography(kp1, des1, kp2, des2, ratio=0.8, ransac_thresh=5.0):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = [m for m, n in matches if m.distance < ratio * n.distance]

    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        
        
        if mask is not None:
            inliers = int(np.sum(mask))
        else:
            inliers = 0

        return H, inliers, good, mask
    else:
        return None, 0, good, None

