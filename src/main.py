import os
import cv2
import pickle
import threading
import numpy as np
import matplotlib.pyplot as plt
from classify import classify
from createDB import createDB


########################################### Helper Functions ##########################################
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

def gen_score(sig_type, test_size = None):
    # Load File names
    file_names = os.listdir('../TrainingSet/' + sig_type)
    # np.random.shuffle(file_names)
    if test_size is not None:
        file_names = file_names[:test_size]

    # Variables
    idx = 0
    limit = 20
    threads = [None] * limit
    avg = [None] * len(file_names)
    match = [None] * len(file_names)

    # Generate scores for each signature type
    for i, img_name in enumerate(file_names):
        # Run only limit number of threads
        if idx % limit == 0 and idx != 0:
            for j in range(limit):
                threads[j].join()
            idx = 0
        # Run thread
        threads[idx] = threading.Thread(target = classify, args = (cv2.imread('../TrainingSet/' + sig_type + '/' +
                              img_name, cv2.IMREAD_GRAYSCALE), DB, threshold, neighbourhood, match, avg, i))
        threads[idx].start()
        idx += 1

    for j in range(idx):
        threads[j].join()
    return match, avg
#######################################################################################################


DB = createDB()
storeData(DB, 'database')
DB = loadData('database')
# test_size = 10
test_size = None
neighbourhood = 1
threshold = 0.11


# Matched Points for each type of signature
print("Genuine Signatures")
genuine_match, genuine_avg = gen_score('Genuine', test_size)
print("Disguise Signatures")
disguise_match, disguise_avg = gen_score('Disguise', test_size)
print("Simulated Signatures")
simulated_match, simulated_avg = gen_score('Simulated', test_size)


# Results
plt.figure(figsize = (20, 8))
plt.subplot(1, 2, 1)
match = [genuine_match, disguise_match, simulated_match]
plt.hist(match, label=['Genuine', 'Disguise', 'Simulated'])
plt.legend()
plt.title('Matched Points')

plt.subplot(1, 2, 2)
avg = [genuine_avg, disguise_avg, simulated_avg]
plt.hist(avg, label=['Genuine', 'Disguise', 'Simulated'])
plt.legend()
plt.title('Average Distance')

plt.savefig('Matchpoints' + str(threshold) + '.jpg')
plt.show()
