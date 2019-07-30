import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
from imutils.video import WebcamVideoStream


print("Camera init -> DONE")
cam = WebcamVideoStream(src=0).start()
print("Starting MECHA_CORTEX_V5...")

# Local Contrast Normalization LCN

while True:
    # Capture state Xt
    img = cam.read()
    # grayscale -> 0 to 255
    img = cv2.resize(img, (65,65))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    float_gray = gray.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(float_gray, (0, 0), sigmaX=2, sigmaY=2)
    num = float_gray - blur
    blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=20, sigmaY=20)
    den = cv2.pow(blur, 0.5)
    gray = num / den
    cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

    plt.imshow(gray)
    plt.pause(0.000001)
    plt.draw()
