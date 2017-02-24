import  cv2
import  matplotlib.pyplot as plt
import glob
import os
import time

faces_folder_path = '/home/cunrui/tmp/face/7/'

detector = cv2.xfeatures2d.SIFT_create()

pics = glob.glob(os.path.join(faces_folder_path, "*.*"))
for f in range(0,len(pics),2):
    img1 = cv2.imread(pics[f])
    img2 = cv2.imread(pics[f+1])

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    kp1,desc1 = detector.detectAndCompute(img1_gray,None)
    kp2, desc2 = detector.detectAndCompute(img2_gray, None)
    bf =cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)

    print type(matches[0].queryIdx)
    print dir(matches[0])

    for m,n in enumerate(matches):
        print m,n.queryIdx,n.trainIdx
        img1 = cv2.circle(img1,(int(kp1[n.queryIdx].pt[0]),int(kp1[n.queryIdx].pt[1])),int(kp1[n.queryIdx].size*0.5),(0,0,155),1)
        img2 = cv2.circle(img2, (int(kp2[n.trainIdx].pt[0]), int(kp2[n.trainIdx].pt[1])),int(kp2[n.trainIdx].size * 0.5), (55, 255, 155), 1)



    matches = sorted(matches, key=lambda x: x.distance)

    img3 =cv2.drawMatches(img1, kp1, img2,kp2, matches[:40],img2.copy(),flags=2)

    cv2.imshow('image', img3)
    cv2.imwrite("../result/"+str(f)+"_"+str(f+1)+".jpg", img3, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if cv2.waitKey(0) == 27:
        break
    else:
        continue

cv2.destroyAllWindows()
