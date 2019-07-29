import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
from imutils.video import WebcamVideoStream



print("Camera init -> DONE")
cam = WebcamVideoStream(src=0).start()
print("Starting MECHA_CORTEX_V5...")

# Main function from :
# https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/

def build_filters():
    filters = []
    ksize = 30
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 3, theta, 8, 10, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)
    return accum

filters = build_filters()
while True:
    # Capture state Xt
    img = cam.read()
    # grayscale -> 0 to 255
    img = cv2.resize(img, (65,65))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res1 = process(gray, filters)

    plt.imshow(res1)
    plt.pause(0.000001)
    plt.draw()
