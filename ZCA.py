import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

img = cv2.imread("C:/Users/quentin.munch/Desktop/img.jpg", cv2.IMREAD_GRAYSCALE)
# centered image
img_mean = np.mean(img)
img_std = np.std(img)
img = (img - img_mean)/img_std
# calculate covariance matrix
img_cov = np.dot(img.T, img) / img.shape[0]
# eigenvalue decomposition
d, V = np.linalg.eigh(img_cov)
# downsampling
D = np.diag(1. / np.sqrt(d + 1e-8))
# calculate whitening matrix
W = np.dot(np.dot(V, D), V.T)
# image whitening
img_white = np.dot(img,W)

plt.matshow(img)
plt.show()
plt.matshow(img_cov)
plt.show()
plt.matshow(V)
plt.show()
plt.matshow(W)
plt.show()
plt.matshow(img_white)
plt.show()
