""" Classification Functions """
import os
import cv2
import threading
import numpy as np

def classify_matched(test_img, DB, threshold, match, idx):
    """ Classify matched points """

    # Compute SURF Keypoints
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 1000, extended = 1)
    kp, des = surf.detectAndCompute(test_img, None)

    # Match Keypoints
    matched_points = 0
    for keypoint in des:
        min_dist = np.min(np.linalg.norm(DB - keypoint, axis=1))
        if min_dist <= threshold:
            matched_points += 1

    # Return percentage of matched keypoints and avg distance
    match[idx] = matched_points / des.shape[0]
    return

def thread_helper(path, DB, threshold, test_size = None):
    """ Thread helper function for running classify_matched """

    # Load File names
    file_names = os.listdir(path)
    if test_size is not None:
        file_names = file_names[:test_size]

    # Variables
    idx = 0
    limit = 10
    threads = [None] * limit
    match = [None] * len(file_names)

    # Generate matched points percentage for each signature type
    for i, img_name in enumerate(file_names):
        # Run only limit number of threads
        if idx % limit == 0 and idx != 0:
            for j in range(limit):
                threads[j].join()
            idx = 0
        # Run thread
        threads[idx] = threading.Thread(target = classify_matched,
                       args = (cv2.imread(path + img_name, cv2.IMREAD_GRAYSCALE),
                       DB, threshold, match, i))
        threads[idx].start()
        idx += 1

    # Join remaining threads
    for j in range(idx):
        threads[j].join()
    return np.array(match)
