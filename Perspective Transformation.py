import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'test3'

img = cv2.imread(image_path + '.jpg')
rows, cols, ch = img.shape

pts1 = np.float32([[0, 0], [225, 125], [125, 275], [300, 300]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img, M, (300, 300))

cv2.imwrite(image_path + '_output.jpg', dst)

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()
