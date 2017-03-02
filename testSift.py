# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
import os
import glob
import math


# class KeyPoint
#     {
#         Point2f    pt; // 坐标
#         float    size; // 特征点邻域直径
#         float    angle; // 特征点的方向，值为[0, 360)，负值表示不使用
#         float    response; //
#         int    octave; // 特征点所在的图像金字塔的组
#         int    class_id; // 用于聚类的id
#     }

print '你好'


foldnum =13
sizeDoorMin =0
sizeDoorMax=10

faces_folder_path = '/home/cunrui/tmp/face/' + str(foldnum) + '/'

for f in glob.glob(os.path.join(faces_folder_path, "*.*")):
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # print cv2.__version__

    detector = cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = detector.detectAndCompute(gray, None)

    for i in kp1:
        print i.response,i.size,i.octave,math.sqrt(i.octave)
        if i.size < sizeDoorMin or i.size > sizeDoorMax:
            # print i
            kp1.remove(i)

    img = cv2.drawKeypoints(image=img, outImage=img, keypoints=kp1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # img = cv2.drawKeypoints(image=img, outImage=img, keypoints=kp1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(0, 0, 0))
    # print type(kp1[0])
    # print dir(kp1[0])

    cv2.imshow('image', img)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
        break

