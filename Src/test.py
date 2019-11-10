""" Module to differentiate geniune, forged and duplicate signatures. """
import cv2


def main():
    """ Main Functon. """
    dirpath = '../Dataset_4NSigComp2010/Dataset_4NSigComp2010/TrainingSet/Disguise/'
    img = cv2.imread(dirpath + 'D023.png', cv2.IMREAD_GRAYSCALE)
    surf = cv2.SURF(400)
    print(surf)


if __name__ == '__main__':
    main()
