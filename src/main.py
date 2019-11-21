import cv2
import matplotlib.pyplot as plt
import matplotlib
import PIL
import numpy as np
import os

def display_images(image):
    cv2.imshow('Showing image', image)
    cv2.waitKey()

def preprocess(arr):
    res = []
    for i in range(arr.shape[0]):
        _, ret = cv2.threshold(arr[i], 127, 255, cv2.THRESH_BINARY)
        res.append(ret)
    res = np.array(res)
    return res

def createDB():
    img = []
    file_names = os.listdir('../TrainingSet/Reference')
    for img_name in file_names:
        img.append(cv2.imread('../TrainingSet/Reference/' + img_name, cv2.IMREAD_GRAYSCALE))
    img = np.array(img)
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 1000, extended=1)
    finalDB = []
    for i in range(img.shape[0]):
        tmp_db = []
        for j in range(img.shape[0]):
            if i == j:
                continue
            kp, des = surf.detectAndCompute(img[j], None)
            for keypoint in des:
                tmp_db.append(keypoint)
        tmp_db = np.array(tmp_db)

        kp, des = surf.detectAndCompute(img[i], None)
        sum_dist = 0
        count = 0
        for keypoint in des:
            res = 1e18
            s = np.zeros(128)
            for db_point in tmp_db:
                dist = np.linalg.norm(db_point - keypoint)
                if res > dist:
                    res = dist
                    s = db_point
            sum_dist = sum_dist + np.linalg.norm(keypoint - s)
            count = count + 1
        avg_dist = sum_dist / count
        count = 0
        for keypoint in des:
            for db_point in tmp_db:
                if np.linalg.norm(keypoint - db_point) <= avg_dist:
                    finalDB.append(keypoint)
                    count = count + 1
                    break
        print("Total number of points = ", des.shape[0])
        print("Green points = ", count)
        print("Average distance = ", avg_dist)
    return finalDB

DB = createDB()
print(DB.shape)