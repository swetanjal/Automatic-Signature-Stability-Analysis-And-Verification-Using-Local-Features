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
threshold = 0.13


# Matched Points Percentage for each type of signature
print("Genuine Signatures")
genuine_match = thread_helper('../TrainingSet/Genuine/', DB, threshold)
print("Disguise Signatures")
disguise_match = thread_helper('../TrainingSet/Disguise/', DB, threshold)
print("Simulated Signatures")
simulated_match = thread_helper('../TrainingSet/Simulated/', DB, threshold)
plot_matches(genuine_match, disguise_match, simulated_match, threshold)


# Save/Load percentages (for convenience)
storeData([genuine_match, disguise_match, simulated_match], '../Pickles/vals' +
                                            str(threshold) + '.pkl')
genuine_match, disguise_match, simulated_match = loadData('../Pickles/vals' +
                                          str(threshold) + '.pkl')


# Calculate FAR and FRR
far = []
frr = []
theta_range = np.arange(0.05, 0.15, 0.005)
for theta in theta_range:
    ind = np.where(genuine_match < theta)
    ind1 = np.where(disguise_match > theta)
    ind2 = np.where(simulated_match > theta)
    frr.append(ind[0].shape[0] / genuine_match.shape[0] * 100)
    far.append((ind1[0].shape[0] + ind2[0].shape[0]) / (disguise_match.shape[0] + simulated_match.shape[0]) * 100)
plot_EER(theta_range, far, frr, threshold)
