import cv2
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


USE_WEBCAM = False # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/presi_simple_CNN.99.hdf5'
emotion_labels = get_labels('presi')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

threshold = 0.7
dataset_path = "./test/"

# starting lists for calculating modes
emotion_window = []

# starting video streaming

# cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
# cap = None
# if (USE_WEBCAM == True):
	# cap = cv2.VideoCapture(0) # Webcam source
# else:
	# cap = cv2.VideoCapture('./demo/joe.SFAHcLh5Dl0.15.mp4') # Video file source
cap = None
labels = pd.read_csv(dataset_path+'Labels.csv')


total_test=200
true_neg = true_net = true_pos = 0
hit_neg = hit_net = hit_pos = 0
count=1

mapp = {'Negative':0,'Neutral':1,'Positive':2}

y_true=[]
y_pred=[]

for ind in (np.random.choice(len(labels), total_test, False)):
	cur_label=mapp[labels['Expression Sentiment'][ind]]
	# if cur_label=="Negative":
		# true_neg+=1
	# elif cur_label=="Neutral":
		# true_net+=1
	# else:
		# true_pos+=1
	cap = cv2.VideoCapture(dataset_path+labels['Filename'][ind])
	total_frames=0
	no_frames_neg = no_frames_pos = no_frames_net =0
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
			emotion_prediction = emotion_classifier.predict(gray_face)
			emotion_probability = np.max(emotion_prediction)
			emotion_label_arg = np.argmax(emotion_prediction)
			emotion_text = emotion_labels[emotion_label_arg]
			emotion_window.append(emotion_text+" "+str(round(emotion_probability, 2)))
			
			
			
			if len(emotion_window) > frame_window:
				emotion_window.pop(0)
			try:
				emotion_mode = mode(emotion_window)
			except:
				continue

			if emotion_text == 'Negative':
				color = emotion_probability * np.asarray((255, 0, 0))
				if emotion_probability>=threshold:
					no_frames_neg+=1
			elif emotion_text == 'Positive':
				if emotion_probability>=threshold:
					no_frames_pos+=1
				color = emotion_probability * np.asarray((255, 255, 0))
			elif emotion_text == 'Neutral':
				if emotion_probability>=threshold:
					no_frames_net+=1
				color = emotion_probability * np.asarray((255, 255, 0))
			else:
				color = emotion_probability * np.asarray((0, 255, 255))

			color = color.astype(int)
			color = color.tolist()

			draw_bounding_box(face_coordinates, rgb_image, color)
			draw_text(face_coordinates, rgb_image, emotion_mode,
					  color, 0, -45, 1, 1)

		bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
		# cv2.imshow('window_frame', bgr_image)
		
		# cv2.waitKey(100)
		for i in range(5):
			cap.grab()
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	maj_emo = 1
	total_face_frames=no_frames_neg+no_frames_pos+no_frames_net
	# print(no_frames_pos,no_frames_neg,no_frames_net,total_face_frames)
	if no_frames_net>=0.7*total_face_frames:
		maj_emo = 1
	elif no_frames_pos>=no_frames_net:
		maj_emo = 2
	else:
		maj_emo = 0
	print("--------Video "+str(count)+"--------")
	y_true.append(cur_label)
	y_pred.append(maj_emo)
	# print(maj_emo,labels['Expression Sentiment'][ind])
	# if maj_emo==cur_label:
		# if cur_label=="Negative":
			# hit_neg+=1
		# elif cur_label=="Neutral":
			# hit_net+=1
		# else:
			# hit_pos+=1
	# print("Total Frames: ",total_frames)
	cap.release()
	count+=1
	# cv2.destroyAllWindows()

print(getPerformanceScores(y_true,y_pred))
# print("Negative Acc.: ",(hit_neg/true_neg))	
# print("Neutral Acc.: ",(hit_net/true_net))	
# print("Positive Acc.: ",(hit_pos/true_pos))
# print("Accuracy: ",((hit_neg+hit_net+hit_pos)/total_test))
