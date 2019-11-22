import cv2
import matplotlib.pyplot as plt
import matplotlib
import PIL
import numpy as np
import os
import pickle
import threading

########################################### Helper Functions ##########################################
def display_images(image):
    cv2.imshow('Showing image', image)
    cv2.waitKey()


def storeData(dat, filename):
    dbfile = open(filename, 'ab')
    pickle.dump(dat, dbfile)
    dbfile.close()


def loadData(filename):
    dbfile = open(filename, 'rb')
    return pickle.load(dbfile)
#######################################################################################################


def createDB():
    # Load Images
    img = []
    file_names = os.listdir('../TrainingSet/Reference')
    for img_name in file_names:
        img.append(cv2.imread('../TrainingSet/Reference/' +
                              img_name, cv2.IMREAD_GRAYSCALE))
    img = np.array(img)

    # Perform SURF for each Reference Image
    surf_db = []
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, extended=1)
    for i in range(img.shape[0]):
        kp, des = surf.detectAndCompute(img[j], None)
        surf_db.append(des)
    surf_db = np.array(surf_db)

    # Find stable keypoints
    finalDB = []
    for i in range(img.shape[0]):
        # Generate LOO database
        indexes = list(range(img.shape[0]))
        indexes.remove(i)
        tmp_db = surf_db[indexes].flatten()

        # Find minimum distance for each keypoint
        sum_dist = 0
        distances = []
        des = surf_db[i]
        for keypoint in des:
            min_dist = np.min(np.linalg.norm(tmp_db - keypoint, axis = 1))
            sum_dist += min_dist
            distances.append(min_dist)
        avg_dist = sum_dist / des.shape[0]

        # Append keypoints less than average distance
        indexes = np.where(min_dist < avg_dist)
        finalDB.extend(list(des[indexes]))
    return np.array(finalDB)


def classify(test_img, theta):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, extended=1)
    kp, des = surf.detectAndCompute(test_img, None)
    matched_points = 0
    sum_dist = 0

    for keypoint in des:
        min_dist = np.min(np.linalg.norm(DB - keypoint, axis = 1))
        sum_dist += min_dist
        if min_dist <= theta:
            matched_points += 1

    avg_dist = sum_dist / des.shape[0]
    return "Matched Points: " + str(matched_points) + " Avg Dist: " + str(avg_dist) 

# DB = createDB()
# storeData(DB, 'database')


DB = loadData('database')
print(DB.shape)


""" Number of matched points for Genuine Signatures """
print("Genuine Signatures")
file_names = os.listdir('../TrainingSet/Genuine')
for img_name in file_names:
    print(classify(cv2.imread('../TrainingSet/Genuine/' +
                              img_name, cv2.IMREAD_GRAYSCALE), 0.09))

""" Number of matched points for Simulated Signatures """
print("Forged Signatures")
file_names = os.listdir('../TrainingSet/Simulated')
for img_name in file_names:
    print(classify(cv2.imread('../TrainingSet/Simulated/' +
                              img_name, cv2.IMREAD_GRAYSCALE), 0.09))

""" Number of matched points for Disguised Signatures """
print("Disguised Signatures")
file_names = os.listdir('../TrainingSet/Disguise')
for img_name in file_names:
    print(classify(cv2.imread('../TrainingSet/Disguise/' +
                              img_name, cv2.IMREAD_GRAYSCALE), 0.09))
