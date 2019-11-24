""" Generating Database and Graphs for Train set """

import os
import cv2
import pickle
import threading
import numpy as np
import matplotlib.pyplot as plt
# User defined Modules
from createDB import createDB
from plots import plot_matches, plot_EER
from classify import classify_matched, thread_helper


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


# Load/Save DB
DB = createDB('../TrainingSet/Reference/')
storeData(DB, '../Pickles/database.pkl')
DB = loadData('../Pickles/database.pkl')
threshold = 0.11


# Matched Points Percentage for each type of signature
print("Genuine Signatures")
genuine_match = thread_helper('../TrainingSet/Genuine/', DB, threshold)
print("Disguise Signatures")
disguise_match = thread_helper('../TrainingSet/Disguise/', DB, threshold)
print("Simulated Signatures")
simulated_match = thread_helper('../TrainingSet/Simulated/', DB, threshold)


# Save/Load percentages (for convenience)
storeData([genuine_match, disguise_match, simulated_match], '../Pickles/vals' +
                                            str(threshold) + '.pkl')
genuine_match, disguise_match, simulated_match = loadData('../Pickles/vals' +
                                          str(threshold) + '.pkl')
plot_matches(genuine_match, disguise_match, simulated_match, threshold)


# Calculate FAR and FRR
far = []
frr = []
theta_range = np.arange(0.01, 0.05, 0.005)
for theta in theta_range:
    ind1 = np.where(genuine_match < theta)
    ind2 = np.where(disguise_match < theta)
    ind3 = np.where(simulated_match > theta)
    frr.append((ind1[0].shape[0] + ind2[0].shape[0]) / (genuine_match.shape[0] + disguise_match.shape[0]) * 100)
    far.append(ind3[0].shape[0] / simulated_match.shape[0] * 100)
plot_EER(theta_range, far, frr, threshold)


# Compute Confidence Matrix
eer = 0.0275
ind = np.where(genuine_match < eer)
ind1 = np.where(disguise_match < eer)
ind2 = np.where(simulated_match > eer)
print('\t\tGenuine\tFake')
print('Genuine\t\t' + str(genuine_match.shape[0] - ind[0].shape[0]) + '\t' + str(ind[0].shape[0]))
print('Disguise\t' + str(disguise_match.shape[0] - ind1[0].shape[0]) + '\t' + str(ind1[0].shape[0]))
print('Simulated\t' + str(ind2[0].shape[0]) + '\t' + str(simulated_match.shape[0] - ind2[0].shape[0]))
