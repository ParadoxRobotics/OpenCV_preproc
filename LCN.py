import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('C:/Users/quentin.munch/Desktop/scene.png')
img = cv2.resize(img, (65,65))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
float_gray = img.astype(np.float32) / 255.0
blur = cv2.GaussianBlur(float_gray, (0, 0), sigmaX=2, sigmaY=2)
num = float_gray - blur
blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=20, sigmaY=20)
den = cv2.pow(blur, 0.5)
gray = num / den
gray = cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

plt.matshow(gray)
plt.show()
