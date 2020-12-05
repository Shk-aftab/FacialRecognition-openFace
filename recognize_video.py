#python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import datetime
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output", type=str, default="",
                help="path to (optional) output video file")
ap.add_argument("-u", "--use-gpu", type=bool, default=False,help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
 
# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

if args["use_gpu"]:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
 
# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture('final.mp4')
time.sleep(2.0)

writer = None
 
# start the FPS throughput estimator
fps = FPS().start()
 
# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# writer = cv2.VideoWriter(args["output"], fourcc, 30,(frame.shape[1], frame.shape[0]), True)

steve = 0
tony = 0 
jimmy = 0
date = datetime.datetime.now()
minuteU = date.strftime("%M")

# loop over frames from the video file stream
while(vs.isOpened()):
	date = datetime.datetime.now()
	minute = date.strftime("%M")

	# grab the frame from the threaded video stream
	ret, frame = vs.read()
 
	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]
 
	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
 
	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]
 
		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
 
			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
 
			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()
 
			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			if minute == minuteU:
				if name == 'Steve Rogers':
					steve+=1
				if name == 'Tony Stark':
					tony+=1
				if name == 'Jimmy':
					jimmy+=1
			else:
				minuteU = minute
				tony = 0
				steve = 0 
				jimmy = 0
				
				if name == 'Steve Rogers':
					steve+=1
				if name == 'Tony Stark':
					tony+=1
				if name == 'Jimmy':
					jimmy+=1

 
			# draw the bounding box of the face along with the
			# associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			text2 = "Steve Roger:{} ,Tony Stark:{} ,Jimmy:{} per minute".format(steve,tony,jimmy)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			cv2.putText(frame,text2, (25,25),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
 
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF


	if args["output"] != "" and writer is None:

		fourcc = cv2.VideoWriter_fourcc(*"MPEG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,(frame.shape[1], frame.shape[0]), True)

	writer.write(frame)

	if key == ord("q"):
		break

 	# update the FPS counter
	fps.update()
	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()