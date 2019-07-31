import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
from imutils.video import WebcamVideoStream

print("Camera init -> DONE")
cam = WebcamVideoStream(src=0).start()
print("Start Laplacian...")

while True:
    # Capture state Xt
    img = cam.read()
    img = cv2.resize(img, (65,65))
    #src = cv2.GaussianBlur(img, (3, 3), 0)
    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(src, cv2.CV_16S, ksize=5)
    abs_dst = cv2.normalize(dst, dst=dst, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    plt.imshow(abs_dst)
    plt.pause(0.000001)
    plt.draw()
