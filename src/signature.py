""" Module to differentiate geniune, forged and duplicate signatures. """
import glob
import cv2
import numpy as np


def avg_distance(database, index):
    """ Find average distance. """
    size = len(database)
    keypoints_distance = 0
    number_keypoints = 0
    for iterator in range(size):
        print(iterator, index)
        if iterator != index:
            for key_point1 in database[index]:
                for key_point2 in database[iterator]:
                    keypoints_distance += np.linalg.norm(
                        key_point1 - key_point2)
                    number_keypoints += 1
    return keypoints_distance / number_keypoints


def train(dir_path):
    """ Method to train. """
    ref_path = dir_path + 'Reference/*'
    ref_file = glob.glob(ref_path)
    database = []
    for file in ref_file:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        surf = cv2.xfeatures2d.SURF_create(extended=1)
        _, descriptors = surf.detectAndCompute(img, None)
        database.append(descriptors)
        print(descriptors.shape)
    # database = np.array(database)
    for index in range(len(database)):
        print(avg_distance(database, index))
    # img = cv2.drawKeypoints(img, keypoints, None)
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # print(glob.glob(ref_path))

    gen_path = dir_path + 'Genuine/*'
    dis_path = dir_path + 'Disguise/*'
    for_path = dir_path + 'Simulated/*'


def main():
    """ Main Functon. """
    train('../Dataset_4NSigComp2010/Dataset_4NSigComp2010/TrainingSet/')


if __name__ == '__main__':
    main()
