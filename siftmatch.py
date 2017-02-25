import  cv2
import glob
import os
import math
import random


def getDistance(k1,k2):
    # print k1,k2
    dis=float(int(k1[0])-int(k2[0]))
    dis =dis*dis
    dis= dis +(int(k1[1])-int(k2[1]))*(int(k1[1])-int(k2[1]))
    dis = math.sqrt(dis)

    return dis



faces_folder_path = '/home/cunrui/tmp/face/8/'

detector = cv2.xfeatures2d.SIFT_create()

pics = glob.glob(os.path.join(faces_folder_path, "*.*"))
for f in range(0,len(pics),1):
    img1 = cv2.imread(pics[0])
    img2 = cv2.imread(pics[f])

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    kp1,desc1 = detector.detectAndCompute(img1_gray,None)
    kp2, desc2 = detector.detectAndCompute(img2_gray, None)
    bf =cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)

    # print type(matches[0].queryIdx)
    print len(matches)

    mttmp = list(matches)
    print len(mttmp)
    ls  =[]

    for m,n in enumerate(matches):
        # print m,n.queryIdx,n.trainIdx,kp1[n.queryIdx].pt,kp2[n.trainIdx].pt

        if getDistance(kp1[n.queryIdx].pt,kp2[n.trainIdx].pt)>=80.0:
            # print type(matches)
            # print n.queryIdx, n.trainIdx, getDistance(kp1[n.queryIdx].pt, kp2[n.trainIdx].pt), n.distance
            # print matches.count(n)
            mttmp.remove(n)
            # kp1.remove(kp1[n.queryIdx])
            # kp2.remove(kp2[n.trainIdx])
            # print matches.count(n)
            print 'A',m, n.queryIdx, n.trainIdx, getDistance(kp1[n.queryIdx].pt, kp2[n.trainIdx].pt)

        else:
            print 'B',m,n.queryIdx, n.trainIdx,getDistance(kp1[n.queryIdx].pt,kp2[n.trainIdx].pt)
            # img1 = cv2.circle(img1, (int(kp1[n.queryIdx].pt[0]), int(kp1[n.queryIdx].pt[1])),int(kp1[n.queryIdx].size * 0.5), (0, 0, 155), 1)
            # img2 = cv2.circle(img2, (int(kp2[n.trainIdx].pt[0]), int(kp2[n.trainIdx].pt[1])),int(kp2[n.trainIdx].size * 0.5), (55, 255, 155), 1)
            facecolor = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

            ls.append(kp1[n.queryIdx])

            cv2.line(img1, (int(kp1[n.queryIdx].pt[0]), int(kp1[n.queryIdx].pt[1])), (
            int(kp1[n.queryIdx].pt[0] + kp1[n.queryIdx].size * 0.8 * math.cos(kp1[n.queryIdx].angle)),
            int(kp1[n.queryIdx].pt[1] + kp1[n.queryIdx].size * 0.8 * math.sin(kp1[n.queryIdx].angle))), facecolor, 1)

            cv2.line(img2, (int(kp2[n.trainIdx].pt[0]), int(kp2[n.trainIdx].pt[1])), (
            int(kp2[n.trainIdx].pt[0] + kp2[n.trainIdx].size * 0.8 * math.cos(kp2[n.trainIdx].angle)),
            int(kp2[n.trainIdx].pt[1] + kp2[n.trainIdx].size * 0.8 * math.sin(kp2[n.trainIdx].angle))), facecolor, 1)

    print len(mttmp)

    mttmp = sorted(mttmp, key=lambda x: x.distance)
    print len(matches)




    img3 =cv2.drawMatches(img1, kp1, img2,kp2, mttmp,img2.copy(),flags=2)

    cv2.imshow('image', img3)
    cv2.imwrite("../result/"+str(f)+"_"+str(f+1)+".jpg", img3, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if cv2.waitKey(0) == 27:
        break
    else:
        continue

cv2.destroyAllWindows()
