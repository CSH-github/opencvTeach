import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('/home/cunrui/1.jpg')
img2 = cv2.imread('/home/cunrui/2.jpg')
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
e1 = cv2.getTickCount()
dst = cv2.addWeighted(img,0.7,img2,0.3,0)#cv2.WINDOW_AUTOSIZE cv2.WINDOW_NORMAL
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print time
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# # plt.xticks([])
# # plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()
# px = img[100,100]
# print px
# print img.shape
# ball = img[280:340, 330:390]
# img[273:333, 100:160] = ball
# cv2.imshow('image',img)
# cv2.imshow('image1',ball)
cv2.imshow('image',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()