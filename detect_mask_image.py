# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import pandas as pd
# Calculate the distance between each centre

from scipy.spatial import distance
def compute_distance(midpoints,num):
    dist = np.zeros((num,num))
    for i in range(num):
        for j in range(i+1,num):
            if i!=j:
                dst = distance.euclidean(midpoints[i], midpoints[j])
                dist[i][j]=dst
    return dist

def find_closest(dist,num,thresh):
    p1=[]
    p2=[]
    d=[]
    for i in range(num):
        for j in range(i,num):
            if( (i!=j) & (dist[i][j]<=thresh)):
                p1.append(i)
                p2.append(j)
                d.append(dist[i][j])
        return p1,p2,d

def change_2_red(img,list_of_center,p1,p2):
    risky = np.unique(p1+p2)
    for i in risky:
        list_of_center[i]
#         _ = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)  
        cv2.circle(img, list_of_center[i], 5, (0, 0, 255), 2)
    return img

def mask_image():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=False,
		help="path to input image")
	ap.add_argument("-f", "--face", type=str,
		default="face_detector",
		help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str,
		default="mask_detector.model",
		help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
	weightsPath = os.path.sep.join([args["face"],
		"res10_300x300_ssd_iter_140000.caffemodel"])
	net = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	model = load_model(args["model"])

	# load the input image from disk, clone it, and grab the image spatial
	# dimensions
	image = cv2.imread(".\\images\\pic2.jpg")
	orig = image.copy()
	(h, w) = image.shape[:2]
	# show the output image
	cv2.imshow("input", image)
	cv2.waitKey(0)

	# construct a blob from the image
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	print("[INFO] computing face detections...")
	net.setInput(blob)
	detections = net.forward()

	list_of_center = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# pass the face through the model to determine if the face
			# has a mask or not
			(mask, withoutMask) = model.predict(face)[0]

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(image, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

			# Draw centre
			#compute center 
			x_center = int((startX+endX)/2)
			y_center = int((startY+endY)/2)

			center = (x_center, y_center)
			list_of_center.append(center)
	
	dist= compute_distance(list_of_center,len(list_of_center))
	num = len(list_of_center)
	thresh=400
	p1,p2,d=find_closest(dist,num,thresh)
	df = pd.DataFrame({"p1":p1,"p2":p2,"dist":d})
	img = change_2_red(image,list_of_center,p1,p2)

	# show the output image
	cv2.imshow("Output", img)
	cv2.waitKey(0)
	
if __name__ == "__main__":
	mask_image()
