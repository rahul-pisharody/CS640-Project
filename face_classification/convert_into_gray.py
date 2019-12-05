#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec	4 21:05:25 2019

@author: shivamsatwah
"""

import cv2
import numpy as np
import os

from utils.inference import apply_offsets

from utils.preprocessor import preprocess_input


face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

emotion_offsets = (20, 40)
path = "./datasets/presi/training/"
images = os.listdir("./datasets/presi/training/")
for file in images:
	img = cv2.imread(str(path + file))
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
	faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=13,
				minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

	for face_coordinates in faces:
		x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
		gray_face = gray_image[y1:y2, x1:x2]
		try:
			gray_face = cv2.resize(gray_face, (64,64))
		except:
			continue
		
		# gray_face = preprocess_input(gray_face, True)
		# cv2.imshow("ff",gray_face)

		# print(gray_face)
		cv2.imwrite(str("./datasets/presi/gray_dataset/" + file),gray_face)
		# cv2.waitKey(0)