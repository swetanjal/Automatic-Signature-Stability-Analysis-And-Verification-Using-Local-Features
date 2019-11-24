""" Demo to show working of Algorithm """

import os
import cv2
import pickle
import threading
import numpy as np
import matplotlib.pyplot as plt


############################### Helper Functions ##############################
def display_images(image):
    cv2.imshow('Showing image', image)
    cv2.waitKey()

def storeData(dat, filename):
    dbfile = open(filename, 'wb')
    pickle.dump(dat, dbfile)
    dbfile.close()

def loadData(filename):
    dbfile = open(filename, 'rb')
    return pickle.load(dbfile)
###############################################################################


def annotate(test_img, DB, threshold):
    """ Annotate keypoints """

    # Compute SURF Keypoints
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 1000, extended = 1)
    kp, des = surf.detectAndCompute(test_img, None)

    # Find matching and non-matching Keypoints
    kp1 = []
    for i in range(des.shape[0]):
        min_dist = np.min(np.linalg.norm(DB - des[i], axis=1))
        if min_dist <= threshold:
            kp1.append(kp[i])

    # Annotate image
    test_img = cv2.drawKeypoints(test_img, kp1, None, (255,0,0), 4)
    return test_img


# Load Pre-generated refernce database and match percentages
threshold = 0.11
DB = loadData('../Pickles/database.pkl')
genuine_match, disguise_match, simulated_match = loadData('../Pickles/vals' +
                                          str(threshold) + '.pkl')
# Find good, bad and average cases
gm = np.argsort(genuine_match)
dm = np.argsort(disguise_match)
sm = np.argsort(simulated_match)
gm = gm[[0, int(gm.shape[0] / 2), -1]]
dm = dm[[0, int(dm.shape[0] / 2), -1]]
sm = sm[[-1, int(sm.shape[0] / 2), 0]]
matches = np.concatenate((genuine_match[gm], disguise_match[dm], simulated_match[sm]))


# Annotate the image with SURF keypoints
img = []
for i, type in enumerate(['Genuine', 'Disguise', 'Simulated']):
    path = '../TrainingSet/' + type + '/'
    file_names = os.listdir(path)
    for j in [gm, dm, sm][i]:
        img.append(annotate(cv2.imread(path + file_names[j],
                             cv2.IMREAD_GRAYSCALE), DB, threshold))


# Display
for i, type in enumerate(['Genuine', 'Disguise', 'Simulated']):
    fig = plt.figure(figsize = (20, 20))
    fig.suptitle(type + ' Signatures', fontsize = 16)
    for j, case in enumerate(['Bad Case', 'Average Case', 'Good Case']):
        plt.subplot(2, 2, j + 1)
        plt.imshow(img[3 * i + j])
        plt.title(case + ', Score: ' + str(round(matches[3 * i + j], 4)))
    plt.savefig('../Plots/Demo' + str(i + 1) + '.jpg')
    plt.show()
