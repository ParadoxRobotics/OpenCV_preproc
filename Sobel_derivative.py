import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('C:/Users/quentin.munch/Desktop/scene.png')
img = cv2.resize(img, (129,129))
src = cv2.GaussianBlur(img, (3, 3), 0)
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

grad_x = cv2.Sobel(src, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(src, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
#abs_grad_x = cv2.convertScaleAbs(grad_x)
#abs_grad_y = cv2.convertScaleAbs(grad_y)

#grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
grad = cv2.normalize(grad, dst=grad, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

plt.matshow(grad)
plt.show()
