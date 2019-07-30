import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('C:/Users/quentin.munch/Desktop/scene.png')
img = cv2.resize(img, (129,129))
src = cv2.GaussianBlur(img, (3, 3), 0)
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
dst = cv2.Laplacian(src, cv2.CV_16S, ksize=3)
#abs_dst = cv2.convertScaleAbs(dst)
abs_dst = cv2.normalize(dst, dst=dst, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

plt.matshow(abs_dst)
plt.show()
