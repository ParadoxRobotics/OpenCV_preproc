import numpy as np
import cv2
import math
import imutils
from imutils.video import WebcamVideoStream

cam = WebcamVideoStream(src=0).start()

def gaussian_pyramid_DoG(state):
    # compute first level of Difference of Gaussian with sigma = 1, 1 and √2
    GF1 = cv2.GaussianBlur(state, (7,7), 1)
    GF2 = cv2.GaussianBlur(GF1, (7,7), 1)
    GF3 = cv2.GaussianBlur(GF2, (7,7), np.sqrt(2))
    # DoG
    DIFF1 = GF1-GF2
    DIFF2 = GF2-GF3

    # dowsample the representation of the first layer
    DSGF3 = cv2.resize(GF3, (int(GF3.shape[1]/2), int(GF3.shape[0]/2)))

    # compute second level of Difference of Gaussian with sigma = 1 and √2
    GF4 = cv2.GaussianBlur(DSGF3, (5,5), 1)
    GF5 = cv2.GaussianBlur(GF4, (5,5), np.sqrt(2))
    # DoG
    DIFF3 = DSGF3-GF4
    DIFF4 = GF4-GF5

    # dowsample the representation of the first layer
    DSGF5 = cv2.resize(GF5, (int(GF5.shape[1]/2), int(GF5.shape[0]/2)))

    # compute thrid level of Difference of Gaussian with sigma = 1 and √2
    GF6 = cv2.GaussianBlur(DSGF5, (3,3), 1)
    GF7 = cv2.GaussianBlur(GF6, (3,3), np.sqrt(2))
    # DoG
    DIFF5 = DSGF5-GF6
    DIFF6 = GF6-GF7

    return DIFF1, DIFF2, DIFF3, DIFF4, DIFF5, DIFF6

while True:
    # get camera state, resize and grayscale it
    state = cam.read()
    state = cv2.resize(state, (640,480))
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

    # compute gradient in X and Y
    dx = cv2.Sobel(state, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(state, cv2.CV_32F, 0, 1)
    # avoid null division
    dx = dx + 0.00001
    dy = dy + 0.00001
    # compute magnitude and angle
    magnitude = cv2.magnitude(dx, dy)
    angle = cv2.phase(dx, dy, angleInDegrees=True)
    # compute thresolded mask of magnitude
    th = 80
    ret, mask = cv2.threshold(magnitude, th, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    # compute masked angle and magnitude
    magnitude = cv2.bitwise_and(magnitude, magnitude, mask=mask)
    angle = cv2.bitwise_and(angle, angle, mask=mask)

    # compute gaussian pyramid DoG
    DIFF1_M, DIFF2_M, DIFF3_M, DIFF4_M, DIFF5_M, DIFF6_M = gaussian_pyramid_DoG(magnitude)

    # plot things
    cv2.imshow('magnitude', cv2.applyColorMap(np.uint8(cv2.normalize(DIFF1_M, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)), cv2.COLORMAP_JET))

    # press escap to quit
    if cv2.waitKey(1) == 27:
        break
