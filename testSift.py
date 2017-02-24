import cv2
import numpy as np
import sys

img = cv2.imread('/home/cunrui/1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
print cv2.__version__


detector = cv2.xfeatures2d.SIFT_create()
kp1, desc1 = detector.detectAndCompute(gray, None)

img = cv2.drawKeypoints(image=img,outImage=img,keypoints= kp1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(51,163,236))
print type(kp1[0])
print dir(kp1[0])

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
