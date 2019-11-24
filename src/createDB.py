""" Create the Reference Database """
import os
import cv2
import numpy as np

def createDB(path):
    """ Create the Reference Database """

    # Load Images
    img = []
    file_names = os.listdir(path)
    for img_name in file_names:
        img.append(cv2.imread(path + img_name, cv2.IMREAD_GRAYSCALE))
    img = np.array(img)

    # Perform SURF for each Reference Image
    surf_db = []
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 1000, extended = 1)
    for i in range(img.shape[0]):
        kp, des = surf.detectAndCompute(img[i], None)
        surf_db.append(des)
    surf_db = np.array(surf_db)

    # Find stable keypoints
    finalDB = []
    for i in range(img.shape[0]):
        # Generate LOO database
        indexes = list(range(img.shape[0]))
        indexes.remove(i)
        tmp_db = np.concatenate(surf_db[indexes])

        # Find minimum distance for each keypoint
        sum_dist = 0
        distances = []
        for keypoint in surf_db[i]:
            min_dist = np.min(np.linalg.norm(tmp_db - keypoint, axis=1))
            sum_dist += min_dist
            distances.append(min_dist)
        avg_dist = sum_dist / surf_db[i].shape[0]

        # Append keypoints less than average distance
        indexes = np.where(distances <= avg_dist)
        finalDB.extend(list(surf_db[i][indexes]))

    finalDB = np.stack(finalDB)
    return finalDB
