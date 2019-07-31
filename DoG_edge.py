import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
from imutils.video import WebcamVideoStream

# difference of gaussian DoG for edge detection

# create 2D gaussian kernel
def create_gaussian_kernel(kernel_size_X, kernel_size_Y, sigma_X, sigma_Y):
    gx = cv2.getGaussianKernel(kernel_size_X, sigma_X, cv2.CV_32F)
    gy = cv2.getGaussianKernel(kernel_size_Y, sigma_Y, cv2.CV_32F)
    return gx*gy.T

# init DoG kernel
g1 = create_gaussian_kernel(3,3,50,50)
g2 = create_gaussian_kernel(3,3,0,0)
DoG_kernel = g1 - g2

print("Camera init -> DONE")
cam = WebcamVideoStream(src=0).start()
print("Start DoG")

while True:
    # get current sensory state
    frame = cam.read()
    # grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # resize
    frame = cv2.resize(frame, (65,65))
    # convolve
    frame = cv2.filter2D(frame, -1 , DoG_kernel)
    # plot
    plt.imshow(frame)
    plt.pause(0.0000001)
    plt.draw()
