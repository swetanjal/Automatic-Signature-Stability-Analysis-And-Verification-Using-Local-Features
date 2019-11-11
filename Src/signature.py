""" Module to differentiate geniune, forged and duplicate signatures. """
import cv2
# import numpy as np


def main():
    """ Main Functon. """
    dirpath = '../Dataset_4NSigComp2010/Dataset_4NSigComp2010/TrainingSet/Disguise/'
    img = cv2.imread(dirpath + 'D023.png', cv2.IMREAD_GRAYSCALE)
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, _ = surf.detectAndCompute(img, None)
    img = cv2.drawKeypoints(img, keypoints, None)
    cv2.imshow("Image", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
