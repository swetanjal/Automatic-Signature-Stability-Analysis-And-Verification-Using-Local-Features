""" Module to differentiate geniune, forged and duplicate signatures. """
import glob
import cv2


def train(dir_path):
    """ Method to train. """
    ref_path = dir_path + 'Reference/*'
    ref_file = glob.glob(ref_path)
    for file in ref_file:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        surf = cv2.xfeatures2d.SURF_create()
        keypoints, _ = surf.detectAndCompute(img, None)
        img = cv2.drawKeypoints(img, keypoints, None)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
    # print(glob.glob(ref_path))

    gen_path = dir_path + 'Genuine/*'
    dis_path = dir_path + 'Disguise/*'
    for_path = dir_path + 'Simulated/*'


def main():
    """ Main Functon. """
    # dirpath = '../Dataset_4NSigComp2010/Dataset_4NSigComp2010/TrainingSet/Disguise/'
    # img = cv2.imread(dirpath + 'D023.png', cv2.IMREAD_GRAYSCALE)
    # surf = cv2.xfeatures2d.SURF_create()
    # keypoints, _ = surf.detectAndCompute(img, None)
    # img = cv2.drawKeypoints(img, keypoints, None)
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    train('../Dataset_4NSigComp2010/Dataset_4NSigComp2010/TrainingSet/')


if __name__ == '__main__':
    main()
