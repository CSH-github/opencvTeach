import  cv2
import glob
import os
import math
import random
from collections import Counter



def getDistance(k1,k2):
    # print k1,k2
    dis=float(int(k1[0])-int(k2[0]))
    dis =dis*dis
    dis= dis +(int(k1[1])-int(k2[1]))*(int(k1[1])-int(k2[1]))
    dis = math.sqrt(dis)

    return dis


test_case='86.jpg'
foldnum=8

faces_folder_path = '/home/cunrui/tmp/face/'+str(foldnum)+'/'

test_faces_folder_path = '/home/cunrui/tmp/face/14/'

detector = cv2.xfeatures2d.SIFT_create()
ls = []
pics = glob.glob(os.path.join(faces_folder_path, "*.*"))
pics_test = glob.glob(os.path.join(test_faces_folder_path, "*.*"))

img1 = cv2.imread(test_faces_folder_path+test_case)
for f in range(0,len(pics),1):
    # img1 = cv2.imread(pics_test[test_case])
    img1 = cv2.imread(test_faces_folder_path + test_case)
    img2 = cv2.imread(pics[f])

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    kp1,desc1 = detector.detectAndCompute(img1_gray,None)
    kp2, desc2 = detector.detectAndCompute(img2_gray, None)
    bf =cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)

    # print type(matches[0].queryIdx)
    # print len(matches)

    mttmp = list(matches)
    # print len(mttmp)


    for m,n in enumerate(matches):
        # print m,n.queryIdx,n.trainIdx,kp1[n.queryIdx].pt,kp2[n.trainIdx].pt

        if getDistance(kp1[n.queryIdx].pt,kp2[n.trainIdx].pt)>=60.0:
            # print type(matches)
            # print n.queryIdx, n.trainIdx, getDistance(kp1[n.queryIdx].pt, kp2[n.trainIdx].pt), n.distance
            # print matches.count(n)
            mttmp.remove(n)
            # kp1.remove(kp1[n.queryIdx])
            # kp2.remove(kp2[n.trainIdx])
            # print matches.count(n)
            # print 'A',m, n.queryIdx, n.trainIdx, getDistance(kp1[n.queryIdx].pt, kp2[n.trainIdx].pt)

        else:
            # print 'B',m,n.queryIdx, n.trainIdx,getDistance(kp1[n.queryIdx].pt,kp2[n.trainIdx].pt)
            # img1 = cv2.circle(img1, (int(kp1[n.queryIdx].pt[0]), int(kp1[n.queryIdx].pt[1])),int(kp1[n.queryIdx].size * 0.5), (0, 0, 155), 1)
            # img2 = cv2.circle(img2, (int(kp2[n.trainIdx].pt[0]), int(kp2[n.trainIdx].pt[1])),int(kp2[n.trainIdx].size * 0.5), (55, 255, 155), 1)
            facecolor = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            a =list(kp1[n.queryIdx].pt)
            # print [int(a[0]),int(a[1])]
            ls.append((int(a[0]),int(a[1]),kp1[n.queryIdx].size,kp1[n.queryIdx].angle))

            cv2.line(img1, (int(kp1[n.queryIdx].pt[0]), int(kp1[n.queryIdx].pt[1])), (
            int(kp1[n.queryIdx].pt[0] + kp1[n.queryIdx].size * 0.8 * math.cos(kp1[n.queryIdx].angle)),
            int(kp1[n.queryIdx].pt[1] + kp1[n.queryIdx].size * 0.8 * math.sin(kp1[n.queryIdx].angle))), facecolor, 1)

            cv2.line(img2, (int(kp2[n.trainIdx].pt[0]), int(kp2[n.trainIdx].pt[1])), (
            int(kp2[n.trainIdx].pt[0] + kp2[n.trainIdx].size * 0.8 * math.cos(kp2[n.trainIdx].angle)),
            int(kp2[n.trainIdx].pt[1] + kp2[n.trainIdx].size * 0.8 * math.sin(kp2[n.trainIdx].angle))), facecolor, 1)

    # print len(mttmp)

    mttmp = sorted(mttmp, key=lambda x: x.distance)

    # print len(matches)




    img3 =cv2.drawMatches(img1, kp1, img2,kp2, mttmp,img2.copy(),flags=2)

    # cv2.imshow('image', img3)
    # cv2.imwrite("../result/"+str(f)+"_"+str(f+1)+".jpg", img3, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    # if cv2.waitKey(0) == 27:
    #     break
    # else:
    #     continue
print  Counter(ls),len(ls)
ld = Counter(ls)
img4 = cv2.imread(test_faces_folder_path + test_case)
# font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)
circlecolor = [0, 255, 255]
facecolor = [0, 0, 255]
for d,x in ld.items():
    lm = list(d)
    if x>=40:
        img4 = cv2.line(img4,(lm[0],lm[1]),(int(lm[0]+lm[2]*0.8*math.cos(lm[3])),int(lm[1]+lm[2]*0.8*math.sin(lm[3]))),facecolor,1)
        cv2.circle(img4, (lm[0],lm[1]), 1, circlecolor, 2)
        cv2.putText(img4,str(x),(lm[0],lm[1]+random.randint(0, 20)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 255),1)

cv2.imshow('image', img4)
cv2.imwrite("../result-sface/" +str(test_case)+'_'+str(foldnum)+".jpg", img4, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    # cv2.line(img4, (d(0),d(1)), ((d(0) + d(2) * 0.8 * math.cos(d(3))),d(1) + d(2) * 0.8 * math.sin(d(3)))), facecolor, 0.01*x)
if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
