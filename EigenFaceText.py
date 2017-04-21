#coding:utf-8
from numpy import *
import numpy as np
import cv2
import os
from decimal import *
from PIL import Image
import matplotlib.pyplot as plt
'''
1.获取包含n张人脸图像的集合。我们这里使用400张xxx.bmp来作为人脸训练图像，所以这里n=400.我们把导入的图像拉平，
本来是92*112的矩阵，我们拉平也就是一个1*10304的矩阵，然后n张放在一个大矩阵下，该矩阵为n*10304。
'''
def lengthFile(filename):
    length=0
    imgs=[]
    for dirname, dirnames, filenames in os.walk(filename):
        for subfilename in filenames:
            if subfilename.split('.')[1] == 'bmp':
              subject_path = os.path.join(dirname, subfilename)
              length=length+1
              imgs.append(subject_path)

    return length,imgs

#加载图片
def loadImageSet(filename):
    width=92
    heigh=112
    n,imags=lengthFile(filename)
    FaceMat = mat(zeros((n,width*heigh)))#初始化一个全为0的矩阵，大小为n*width*heigh
    for i in range(n):#加载数据
            try:
                img = cv2.imread(imags[i],0)#加载图片 add+i是路径
            except:
                print 'load %s failed'%i
            FaceMat[i,:] = mat(img).flatten()#拉成一维矩阵
            i += 1
    return FaceMat

def ReconginitionVector(selecthr = 0.8):
    # step1: 加载图片,获得所有图像合并出来的大矩阵，注意转置了  10340*10
    #fileDir = "E:" + os.sep + "picture"
    fileDir = '/home/cunrui/PycharmProjects/ORL'
    FaceMat = loadImageSet(fileDir).T
    #step2:平均图像也就把每一行的10个元素平均计算
    avgImg = mean(FaceMat,1)
    # step3:偏差矩阵，及每张人脸都减去这个平均图像
    diffTrain = FaceMat-avgImg
    #step4：计算协方差矩阵的特征向量
    #计算矩阵特征向量linalg.eig；array.T是转置矩阵，AT*A,eigvals,eigVects分别对应特征值与特征向量组成的向量
    eigvals,eigVects = linalg.eig(mat(diffTrain.T*diffTrain))#argsort函数返回的是数组值从小到大的索引值
    eigSortIndex = argsort(-eigvals)#按特征值从大到小排列，先要最大的
    #4.主成分分析,当排序后的特征值的一部分相加大于该阈值时，我们选择这部分特征值对应的特征向量
    # 此时我们剩下的矩阵是11368*M',M'根据情况在变化。
    # 这样我们不仅减少了计算量，而且保留了主成分，减少了噪声的干扰。
    rowlength = shape(FaceMat)[1]
    for i in xrange(rowlength):#shape()[1]函数读取每列长度
        if (eigvals[eigSortIndex[:i]]/eigvals.sum()).sum() >= selecthr:
            eigSortIndex = eigSortIndex[:i]
            break
    covVects = diffTrain * eigVects[:,eigSortIndex]
    # avgImg 是均值图像，covVects是协方差矩阵的特征向量，diffTrain是偏差矩阵
    return avgImg,covVects,diffTrain

#这一步就是开始进行人脸识别了。此时我们导入一个新的人脸，我们使用上面主成分分析后得到的特征向量
# ，来求得一个每一个特征向量对于导入人脸的权重向量。
def judgeFace(judgeImg,FaceVector,avgImg,diffTrain):
    diff = judgeImg.T - avgImg#协方差
    #每一个特征向量对于导入人脸的权重向量
    # 特征向量其实就是训练集合的图像与均值图像在该方向上的偏差，通过未知人脸在特征向量的投影，
    # 我们就可以知道未知人脸与平均图像在不同方向上的差距。
    weiVec = FaceVector.T* diff#每一个特征向量对于导入人脸的权重向量。
    res = 0
    n,images=lengthFile('/home/cunrui/PycharmProjects/ORL')#图片数量
    dislist=[]
    resVal = inf#numpy中的inf表示一个无限大的正数
    for i in range(n):
        TrainVec = FaceVector.T*diffTrain[:,i]#第k张训练人脸的权重向量
        disdata = (array(weiVec-TrainVec)**2).sum()#欧式距离来判断未知人脸与第k张训练人脸之间的差距。
        dislist.append(disdata)#获得拒每个图片的距离
    disarray=np.array(dislist)
    #从小到大排序,用快排算法
    sortarray=np.sort(disarray,axis=0, kind='quicksort', order=None)
    indexlist=[]
    for element in sortarray.flat:
              itemindex = dislist.index(element)
              indexlist.append(itemindex)
              if(len(indexlist)>=20):
                  break
    return indexlist,dislist


def TextEigenFace():
    avgImg,FaceVector,diffTrain = ReconginitionVector(selecthr = 0.9)
    #该代码用于显示特征脸
    '''
    FaceVector =FaceVector.T
    data=FaceVector[1].reshape(112,92)
    new_im = Image.fromarray(data)
    #plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    new_im.show()
    FaceVector =FaceVector.T
    '''
    piclength,imags=lengthFile('/home/cunrui/PycharmProjects/ORL')#图片数量
    n=piclength/10 #s1---s40文件夹
    textlist=[]
    putlist=[]
    dislist=[]
    count = 0
    for i in range(5):#可以填n，测试40张，填1测试一张
         # 这里的loadname就是我们要识别的未知人脸图，我们通过40张未知人脸找出的对应训练人脸进行对比来求出正确率
        loadname = '/home/cunrui/PycharmProjects/ORL/s'+str(i+1)+'/3.bmp'
        #print i+1
        textlist.append(loadname)
        judgeImg = cv2.imread(loadname,0)
        indexlist,dislistall = judgeFace(mat(judgeImg).flatten(),FaceVector,avgImg,diffTrain)#前20下标
        #print indexlist
        for j in range(20):
            temp = imags[indexlist[j]]
            disdata=dislistall[indexlist[j]]/(10**15)
            dislist.append(round(disdata,4))
            putlist.append(temp)

    #print putlist
    #print textlist
    #print 'accuracy  is %f'%(float(count)/n)  # 求出正确率
    return textlist,putlist,dislist

