""" Module to differentiate geniune, forged and duplicate signatures. """
import cv2

img = cv2.imread('fly.png', 0)
surf = cv2.SURF(400)
