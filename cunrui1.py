import cv2
import numpy as np
import sys
import dlib
import matplotlib.pyplot as plt

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

predictor_path = '/home/cunrui/tmp/dlib-19.2/python_examples/shape_predictor_68_face_landmarks.dat'
faces_path = '/home/cunrui/tmp/dlib-19.2/examples/faces/naicha.jpg'


img = cv2.imread(faces_path)
imgtmp = cv2.imread(faces_path)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

detector = cv2.xfeatures2d.SIFT_create()
kp1, desc1 = detector.detectAndCompute(gray, None)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


dets = detector(img, 1)
for k, d in enumerate(dets):
    #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
    # Get the landmarks/parts for the face in box d.

    shape = predictor(img, d)
    num = 78
    ls = [[shape.part(0).x, shape.part(0).y], [shape.part(8).x, shape.part(8).y], [shape.part(16).x, shape.part(16).y]]
    pts = np.array(ls, np.int32)
    cv2.polylines(imgtmp, [pts], True, (0, 255, 255))

    onekey = [kp1[num].pt[0],kp1[num].pt[1]]
    cv2.circle(imgtmp, (int(kp1[num].pt[0]),int(kp1[num].pt[1])), 10, (0, 0,255), 2)

    print k,isInsidePolygon(onekey,ls)

    #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))

    #Draw the face landmarks on the screen.
    cv2.rectangle(imgtmp, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 1)

    for i in range(68):
        pt = shape.part(i)
        cv2.circle(imgtmp, (pt.x, pt.y), 1, (55, 255, 155), -1)


    #pts = np.array([[10, 5], [34, 23], [231, 54], [76, 98]], np.uint8)
    #pts = pts.reshape((-1, 1, 2))
    # plt.imshow(rgbImg)
    # plt.show()


imgtmp = cv2.drawKeypoints(image=imgtmp,outImage=imgtmp,keypoints= kp1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(0,200,236))
# for i in kp1:
#     cv2.circle(img,(int(i.pt[0]),int(i.pt[1])),int(i.size*0.5),(55,255,155),1)
#win = cv2.namedWindow('image', flags=0) cv2.WINDOW_NORMAL
win = cv2.namedWindow('image', cv2.WINDOW_GUI_EXPANDED)
cv2.imshow('image',imgtmp)
cv2.imwrite("./cat3.jpg", imgtmp, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
cv2.waitKey(0)
cv2.destroyAllWindows()

#pt is one point