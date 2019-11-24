""" Generating Database and Graphs for Test set """

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
DB = createDB('../TestSet/Reference/')
storeData(DB, '../Pickles/database_test.pkl')
DB = loadData('../Pickles/database_test.pkl')
threshold = 0.13


# Matched Points Percentage for each type of signature
print("Questioned Signatures")
match = thread_helper('../TestSet/Questioned/', DB, threshold)
genuine_match = match[[48, 51, 65]]
disguise_match = match[[5, 14, 27, 28, 33, 86, 89]]
simulated_match = match[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17,
18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40,
41, 42, 43, 44, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
84, 85, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]


# Save/Load percentages (for convenience)
storeData([genuine_match, disguise_match, simulated_match], '../Pickles/vals_test.pkl')
genuine_match, disguise_match, simulated_match = loadData('../Pickles/vals_test.pkl')
plot_matches(genuine_match, disguise_match, simulated_match, '_test')


# Calculate FAR and FRR
far = []
frr = []
theta_range = np.arange(0.21, 0.24, 0.005)
for theta in theta_range:
    ind = np.where(genuine_match < theta)
    ind1 = np.where(disguise_match > theta)
    ind2 = np.where(simulated_match > theta)
    frr.append(ind[0].shape[0] / genuine_match.shape[0] * 100)
    far.append((ind1[0].shape[0] + ind2[0].shape[0]) / (disguise_match.shape[0] + simulated_match.shape[0]) * 100)
plot_EER(theta_range, far, frr, '_test')
