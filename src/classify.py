""" Classify Signatures """

import cv2
import numpy as np

def classify(test_img, DB, theta, neighbourhood, match, avg, idx):
    # Compute SURF Keypoints
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, extended=1)
    kp, des = surf.detectAndCompute(test_img, None)

    # Match Keypoints
    matched_points = 0
    avg_dist = 0
    for keypoint in des:
        min_dist = np.mean(np.sort(np.linalg.norm(DB - keypoint, axis=1))[:neighbourhood])
        avg_dist += min_dist
        if min_dist <= theta:
            matched_points += 1
    avg_dist = avg_dist / des.shape[0]

    match[idx] = matched_points
    avg[idx] = avg_dist
    return
