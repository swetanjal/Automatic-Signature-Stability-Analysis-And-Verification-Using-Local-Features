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
            min_dist = np.mean(np.linalg.norm(tmp_db - keypoint, axis=1))
            sum_dist += min_dist
            distances.append(min_dist)
        avg_dist = sum_dist / des.shape[0]

        # Append keypoints less than average distance
        indexes = np.where(min_dist < avg_dist)
        finalDB.extend(list(des[indexes]))
    return np.array(finalDB)


def classify(test_img, theta, neighbourhood):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, extended=1)
    kp, des = surf.detectAndCompute(test_img, None)
    matched_points = 0
    matched_avg = 0
    non_matched_avg = 0
    sum_dist = 0

    for keypoint in des:
        min_dist = np.mean(np.sort(np.linalg.norm(
            DB - keypoint, axis=1))[:neighbourhood])
        sum_dist += min_dist
        if min_dist <= theta:
            matched_points += 1
            matched_avg += min_dist
    # avg_dist = sum_dist / des.shape[0]
    non_matched_avg = (sum_dist - matched_avg) / \
        (des.shape[0] - matched_points)
    if matched_points != 0:
        matched_avg = matched_avg / matched_points
    else:
        matched_avg = 1
    # return "Matched Points: " + str(matched_points) + " Avg Dist: " + str(avg_dist)
    # return matched_points, avg_dist
    return matched_points, matched_avg, non_matched_avg

# DB = createDB()
# storeData(DB, 'database')


DB = loadData('database')
print(DB.shape)

test_size = 10
neighbourhood = 1
threshhold = 0.105

""" Number of matched points for Genuine Signatures """

print("Genuine Signatures")
genuine_match = []
genuine_mavg_dist = []
genuine_navg_dist = []

file_names = os.listdir('../TrainingSet/Genuine')[:test_size]
for img_name in file_names:
    matched_points, mavg_dist, navg_dist = classify(cv2.imread('../TrainingSet/Genuine/' +
                                                               img_name, cv2.IMREAD_GRAYSCALE), threshhold, neighbourhood)
    print(matched_points, mavg_dist, navg_dist)
    genuine_match.append(matched_points)
    genuine_mavg_dist.append(mavg_dist)
    genuine_navg_dist.append(navg_dist)

""" Number of matched points for Simulated Signatures """

simulated_match = []
simulated_mavg_dist = []
simulated_navg_dist = []

print("Forged Signatures")
file_names = os.listdir('../TrainingSet/Simulated')[:test_size]
for img_name in file_names:
    matched_points, mavg_dist, navg_dist = classify(cv2.imread('../TrainingSet/Simulated/' +
                                                               img_name, cv2.IMREAD_GRAYSCALE), threshhold, neighbourhood)
    print(matched_points, mavg_dist, navg_dist)
    simulated_match.append(matched_points)
    simulated_mavg_dist.append(mavg_dist)
    simulated_navg_dist.append(navg_dist)


""" Number of matched points for Disguised Signatures """

disguised_match = []
disguised_mavg_dist = []
disguised_navg_dist = []

print("Disguised Signatures")
file_names = os.listdir('../TrainingSet/Disguise')[:test_size]
for img_name in file_names:
    matched_points, mavg_dist, navg_dist = classify(cv2.imread('../TrainingSet/Disguise/' +
                                                               img_name, cv2.IMREAD_GRAYSCALE), threshhold, neighbourhood)
    print(matched_points, mavg_dist, navg_dist)
    disguised_match.append(matched_points)
    disguised_mavg_dist.append(mavg_dist)
    disguised_navg_dist.append(navg_dist)


# Results

plt.title("Matched Points ; Threshhold : " + str(threshhold))
plt.plot(range(10), genuine_match,  label="Genuine", color='r')
plt.plot(range(10), simulated_match,  label="Simulated", color='g')
plt.plot(range(10), disguised_match,  label="Disguised", color='b')
plt.legend()
plt.show()

plt.title("Matched Avg Distance ; Threshhold : " + str(threshhold))
plt.plot(range(10), genuine_mavg_dist,  label="Genuine", color='r')
plt.plot(range(10), simulated_mavg_dist,  label="Simulated", color='g')
plt.plot(range(10), disguised_mavg_dist,  label="Disguised", color='b')
plt.legend()
plt.show()

plt.title("Non Matched Avg Distance ; Threshhold : " + str(threshhold))
plt.plot(range(10), genuine_navg_dist,  label="Genuine", color='r')
plt.plot(range(10), simulated_navg_dist,  label="Simulated", color='g')
plt.plot(range(10), disguised_navg_dist,  label="Disguised", color='b')
plt.legend()
plt.show()
