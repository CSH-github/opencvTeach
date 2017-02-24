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
    print type(matches[0])
    print dir(matches[0])

    for m,n in enumerate(matches):
        print m
        print dir(n)


    matches = sorted(matches, key=lambda x: x.distance)
    img3 =cv2.drawMatches(img1, kp1, img2,kp2, matches[:40],img2.copy(),flags=2)

    plt.imshow(img3)
    plt.show()
    time.sleep(1)
    plt.close()
    # if cv2.waitKey(10) == 27:
    #     break
    # else:
    #     plt.close()
    #     continue