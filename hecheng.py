import cv2
import numpy as np
import sys
import dlib
import os
import glob
from skimage import io,data

predictor_path ='/home/cunrui/tmp/dlib-19.2/python_examples/shape_predictor_68_face_landmarks.dat'
faces_path = '/home/cunrui/girl.jpg'
imgglob=glob.glob(faces_path)
img = io.imread(imgglob[0])
detector = dlib.get_frontal_face_detector()#class
predictor = dlib.shape_predictor(predictor_path)#class
win = dlib.image_window()
win.clear_overlay()
win.set_image(img)
dets = detector(img, 1)
shape = predictor(img, dets[0])
win.add_overlay(shape)
win.add_overlay(dets[0])