import numpy as np
import argparse
import time
import cv2
import os

# yolo on python
# https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

# opencv dnn
# https://docs.opencv.org/3.4/d6/d0f/group__dnn.html#gafde362956af949cce087f3f25c6aff0d

# opencv net
# https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html


# 應該做成抽象對象，被繼承
class CVModel:

	def __init__(self):
		self.net
		self.detectMethod
		self.result
		self.imageSize

        self.configPath
        self.weightsPath

	# may be not useful
	def initResult(self):
		pass

	def loadModel(self, model):
		self.model = model

    # load model.txt
    def loadNames(self, path):
        namesPath = os.path.sep.join([args['yolo'], 'lp.names'])
        labels = open(namesPath).read().strip().split('\n')

	def load(self):
		self.configPath = os.path.sep.join([args["yolo"], "lp.cfg"])
		self.weightsPath = os.path.sep.join([args["yolo"], "lp.weights"])
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
    
    # this should not be placed here
	def frame(num):
		return <image>

	def detectImage(image):
		blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB = True, crop = False)
		self.net.setInput(blob)
		outputs = self.forward()
		return 

	def detectVideo(video):
		return	
        


class ModelResult:
	def __init__(self):
		