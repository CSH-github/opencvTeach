import cv2
import numpy as np
import sys
import dlib
import matplotlib.pyplot as plt
import glob
from skimage import io
import os
import random
import math
from collections import Counter

def isInsidePolygon(pt, poly):
    c = False
    i = -1
    l = len(poly)
    j = l - 1
    while i < l-1:
        i += 1
        #print i,poly[i], j,poly[j]
        if ((poly[i][0] <= pt[0] and pt[0] < poly[j][0]) or (poly[j][0] <= pt[0] and pt[0] < poly[i][0])):
            if (pt[1] < (poly[j][1] - poly[i][1]) * (pt[0] - poly[i][0]) / (poly[j][0] - poly[i][0]) + poly[i][1]):
                c = not c
        j = i
    return c


#print cv2.__version__

predictor_path = '/home/cunrui/tmp/shape_predictor_68_face_landmarks.dat'
# faces_path = '/home/cunrui/tmp/face/3/s4_1.JPG'
faces_folder_path = '/home/cunrui/tmp/face/13/'

# img = cv2.imread(faces_path)
# imgtmp = cv2.imread(faces_path)
# gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# siftpic = cv2.imread(faces_path)
    # np.zeros((300,300,3), np.uint8)

siftdetector = cv2.xfeatures2d.SIFT_create()
w=0;

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
ls =[]



for f in glob.glob(os.path.join(faces_folder_path, "*.*")):
    if w==0:
        siftpic = cv2.imread(f)
        w=w+1
    img = cv2.imread(f)
    siftpic = cv2.addWeighted(siftpic, 0.9, img, 0.1, 0)


for f in glob.glob(os.path.join(faces_folder_path, "*.*")):
    print("Processing file: {}".format(f))

    #img = io.imread(f)
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp1, desc1 = siftdetector.detectAndCompute(gray, None)
    dets = detector(img, 1)
    # cv2.drawKeypoints(image=siftpic, outImage=siftpic, keypoints=kp1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 200, 236))

    #print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        facecolor =[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)]
        for i in kp1:
            # print i.response,i.size
            if i.size>50.0:
                cv2.circle(siftpic,(int(i.pt[0]),int(i.pt[1])),1,tuple(facecolor),-1)
                jiaodu = i.angle
                r  = i.size*0.5
                a = list(i.pt)
                # print [int(a[0]),int(a[1])]
                ls.append((int(a[0]), int(a[1]), i.size, i.angle))
                # print type(i.pt)
                cv2.line(siftpic,(int(i.pt[0]),int(i.pt[1])),(int(i.pt[0]+r*math.cos(jiaodu)),int(i.pt[1]+r*math.sin(jiaodu))), facecolor,1)
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #     k, d.left(), d.top(), d.right(), d.bottom()))
        # # Get the landmarks/parts for the face in box d.
        # shape = predictor(img, d)
        # print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
        #                                           shape.part(1)))
        # # Draw the face landmarks on the screen.


win = cv2.namedWindow('image', cv2.WINDOW_GUI_EXPANDED)
cv2.imshow('image',siftpic)
cv2.imwrite("./result/1-line-size50.jpg", siftpic, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
ld = Counter(ls)
print ld
test_case ='81.jpg'
img4 = cv2.imread(faces_folder_path + test_case)
circlecolor = [0, 255, 255]
facecolor = [0, 0, 255]
for d,x in ld.items():
    lm = list(d)
    if x>=30:
        img4 = cv2.line(img4,(lm[0],lm[1]),(int(lm[0]+lm[2]*0.8*math.cos(lm[3])),int(lm[1]+lm[2]*0.8*math.sin(lm[3]))),facecolor,1)
        cv2.circle(img4, (lm[0],lm[1]), 1, circlecolor, 2)
        cv2.putText(img4,str(x),(lm[0],lm[1]+random.randint(0, 20)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 255),1)

cv2.imshow('image2', img4)
cv2.imwrite("../result-sface/" +str(test_case)+'_'+'13'+".jpg", img4, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
cv2.waitKey(0)
cv2.destroyAllWindows()