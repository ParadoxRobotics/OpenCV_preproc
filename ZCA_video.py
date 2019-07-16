import numpy as np
import matplotlib.pyplot as plt
import math

import cv2
import imutils
from imutils.video import WebcamVideoStream

print("Camera init -> DONE")
cam = WebcamVideoStream(src=0).start()
print("Starting...")

while True:
    # Capture state
    img = cam.read()
    # grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize
    img = cv2.resize(img, (640,480))
    # centered image
    img_mean = np.mean(img)
    img_std = np.std(img)
    img = (img - img_mean)/img_std
    # calculate covariance matrix
    img_cov = np.dot(img.T, img) / img.shape[0]
    # eigenvalue decomposition
    d, V = np.linalg.eigh(img_cov)
    # downsampling
    D = np.diag(1. / np.sqrt(d + 1e-7))
    # calculate whitening matrix
    W = np.dot(np.dot(V, D), V.T)
    # image whitening
    img_white = np.dot(img,W)
    # plot
    plt.imshow(img_white)
    plt.pause(0.0001)
    plt.draw()
