import cv2
import os
import numpy as np
import pandas as pd
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

from confusion import getConfusionMatrix, getPerformanceScores


def updateFrames(neg,net,pos,emotion,prob):
	threshold=0.7
	if emotion == 0:
		if prob>=threshold:
			neg+=1
	elif emotion == 2:
		if prob>=threshold:
			pos+=1
	elif emotion == 1:
		if prob>=threshold:
			net+=1
	
	return neg,net,pos


def heuristic_maj(neg,net,pos):
	maj_emo = 1
	total_face_frames=neg+net+pos
	if total_face_frames==0:
		return 1
	# print(neg,net,pos,total_face_frames)
	if pos>net:
		if pos>=neg:
			maj_emo = 2
		else:
			maj_emo = 0
	elif neg>net:
		if neg>pos:
			maj_emo = 0
		else:
			maj_emo = 2
	
	return maj_emo

def heuristic_1(neg,net,pos):
	maj_emo = 1
	total_face_frames=neg+net+pos
	# print(neg,net,pos,total_face_frames)
	if net>=0.7*total_face_frames:
		maj_emo = 1
	elif pos>=neg:
		maj_emo = 2
	else:
		maj_emo = 0
	
	return maj_emo

def heuristic_2(neg,net,pos):
	total_face_frames=neg+net+pos
	if total_face_frames==0:
		return 1
	# print(neg,net,pos,total_face_frames)
	if neg>=0.3*total_face_frames or pos>=0.3*total_face_frames:
		maj_emo = 0 if neg>pos else 2
	else:
		maj_emo = 1
	
	return maj_emo


# parameters for loading data and images
emotion_model_path_1 = './models/presi_CNN.197.hdf5'
emotion_model_path_2 = './models/presi_big_XCEPTION.108.hdf5'
emotion_labels = get_labels('presi')

# hyper-parameters for bounding boxes shape
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

emotion_classifier_1 = load_model(emotion_model_path_1)
emotion_classifier_2 = load_model(emotion_model_path_2)

# getting input model shapes for inference
emotion_target_size = emotion_classifier_1.input_shape[1:3]

dataset_path = "./test/"


video_capture = cv2.VideoCapture(0)

cap = None

labels = pd.read_csv(dataset_path+'Labels.csv')


max_test=300
true_neg = true_net = true_pos = 0
hit_neg = hit_net = hit_pos = 0
count=1

mapp = {'Negative':0,'Neutral':1,'Positive':2}

y_true=[]

y_pred_m1_h0 = []
y_pred_m1_h1 = []
y_pred_m1_h2 = []

y_pred_m2_h0 = []
y_pred_m2_h1 = []
y_pred_m2_h2 = []

files = os.listdir(dataset_path+'test_vids/')
c=0
for file in files:
	c+=1
	if c>max_test:
		break
	cur_label = mapp[labels.loc[labels['Filename']==file].iloc[0]['Expression Sentiment']]
	# cur_label=mapp[labels['Expression Sentiment'][ind]]
	
	cap = cv2.VideoCapture(dataset_path+'test_vids/'+file)
	total_frames=0
	neg_1 = net_1 = pos_1 = 0
	neg_2 = net_2 = pos_2 = 0
	while cap.isOpened(): # True:
		ret, bgr_image = cap.read()

		#bgr_image = video_capture.read()[1]
		if not(ret):
			break
		total_frames+=1
		gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
		rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

		faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=13,
				minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

		for face_coordinates in faces:

			x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
			gray_face = gray_image[y1:y2, x1:x2]
			try:
				gray_face = cv2.resize(gray_face, (emotion_target_size))
			except:
				continue

			gray_face = preprocess_input(gray_face, True)
			gray_face = np.expand_dims(gray_face, 0)
			gray_face = np.expand_dims(gray_face, -1)
			
			
			#-------Model 1 Prediction-------
			emotion_prediction_1 = emotion_classifier_1.predict(gray_face)
			emotion_probability_1 = np.max(emotion_prediction_1)
			emotion_label_arg_1 = np.argmax(emotion_prediction_1)
			
			#-------Model 2 Prediction-------
			emotion_prediction_2 = emotion_classifier_2.predict(gray_face)
			emotion_probability_2 = np.max(emotion_prediction_2)
			emotion_label_arg_2 = np.argmax(emotion_prediction_2)
			
			neg_1, net_1, pos_1 = updateFrames(neg_1, net_1, pos_1, emotion_label_arg_1, emotion_probability_1)
			neg_2, net_2, pos_2 = updateFrames(neg_2, net_2, pos_2, emotion_label_arg_2, emotion_probability_2)

		for i in range(5):
			cap.grab()
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	#Heuristics on Model 1
	pred_emo_m1_h0 = heuristic_maj(neg_1,net_1,pos_1)
	pred_emo_m1_h1 = heuristic_1(neg_1,net_1,pos_1)
	pred_emo_m1_h2 = heuristic_2(neg_1,net_1,pos_1)
	
	#Heuristics on Model 2
	pred_emo_m2_h0 = heuristic_maj(neg_2,net_2,pos_2)
	pred_emo_m2_h1 = heuristic_1(neg_2,net_2,pos_2)
	pred_emo_m2_h2 = heuristic_2(neg_2,net_2,pos_2)
	
	print("--------Video "+str(count)+"--------")
	print("Filename: ",file)
	y_true.append(cur_label)
	
	y_pred_m1_h0.append(pred_emo_m1_h0)
	y_pred_m1_h1.append(pred_emo_m1_h1)
	y_pred_m1_h2.append(pred_emo_m1_h2)
	
	y_pred_m2_h0.append(pred_emo_m2_h0)
	y_pred_m2_h1.append(pred_emo_m2_h1)
	y_pred_m2_h2.append(pred_emo_m2_h2)
	
	print(pred_emo_m1_h0,cur_label)
	print(pred_emo_m2_h0,cur_label)
	# print(pred_emo_m1_h1,cur_label)
	# print(pred_emo_m1_h2,cur_label)
	
	cap.release()
	count+=1
	# cv2.destroyAllWindows()

print("--------------Model 1 Majority Heuristic--------------")
print(getPerformanceScores(y_true,y_pred_m1_h0))

print("--------------Model 1 Heuristic 1--------------")
print(getPerformanceScores(y_true,y_pred_m1_h1))

print("--------------Model 1 Heuristic 2--------------")
print(getPerformanceScores(y_true,y_pred_m1_h2))

print("--------------Model 2 Majority Heuristic--------------")
print(getPerformanceScores(y_true,y_pred_m2_h0))

print("--------------Model 2 Heuristic 1--------------")
print(getPerformanceScores(y_true,y_pred_m2_h1))

print("--------------Model 2 Heuristic 2--------------")
print(getPerformanceScores(y_true,y_pred_m2_h2))
