import numpy as np
import argparse
import time
import cv2
import os
import easyocr

from .YoloModelError import YoloModelErrors
from ..CVModel.CVModel import CVModel

# yolo on python
# https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

# opencv dnn
# https://docs.opencv.org/3.4/d6/d0f/group__dnn.html#gafde362956af949cce087f3f25c6aff0d

# opencv net
# https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html


# 應該做成抽象對象，被繼承
class YoloModel(CVModel):

	def __init__(self, namesPath = '', configPath = '', weightsPath = ''):
		self.net
		self.detectMethod
		self.result
		self.imageSize
		self.LPNum = ''

		self.namesPath = namesPath
		self.configPath = configPath
		self.weightsPath = weightsPath
		self.labels = []
		self.outputLayerNames
		self.threshold = 0.3
		self.confidence = 0.5
		self.colors = []
		self.minProbability = 0.5

	# 解析.names文件
	def loadNames(self):
		# TODO error handling: when the file can not be loaded
		self.labels = open(self.namesPath).read().strip().split('\n')
		self.colors = np.random.randint(0, 255, size = (len(self.labels), 3), dtype = "uint8")
		
	def load(self):
		print("loading YOLO Model...")
		## TODO error handling
		# self.namesPath = os.path.sep.join("", "lp.names")
		# self.configPath = os.path.sep.join("", "lp.cfg")
		# self.weightsPath = os.path.sep.join("", "lp.weights")
		self.loadNames()
		self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
		self.outputLayerNames = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

	# (cx, cy, w, h) -> (p1x, p1y, p2x, p2y)
	@staticmethod
	def yoloFormatToTwoPoint(centerX, centerY, width, height):
		# 計算角點坐標
		p1x = int(centerX - (width / 2))
		p1y = int(centerY - (height / 2))
		p2x = p1x + width
		p2y = p1y + height
		return [p1x, p1y, p2x, p2y]

	def detectImage(self, image: cv2.Mat):
		result = self.DetectResult()
		( H, W ) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB = True, crop = False)
		self.net.setInput(blob)
		layerOutputs = self.net.forward(self.outputLayerNames)
		
		for output in layerOutputs:
			for detection in output:
				# detection 前4個是box中心點坐標(x, y)和寬高(w, h)，按比例，像txt標注檔數據
				scores = detection[5:]
				# 找出得分最大的 index
				classID = np.argmax(scores)
				confidence = scores[classID]
				if confidence > self.minProbability:
					box = detection[0 : 4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					result.add(self.yoloFormatToTwoPoint(centerX, centerY, width, height), float(confidence), classID)
		return result

	def drawBoxes(self, image, detectResults):
		idxs = cv2.dnn.NMSBoxes(detectResults.boxes, detectResults.confidences, self.confidence, self.threshold)
		if len(idxs) > 0:
			for i in idxs.flatten():
				p1x = detectResults.boxes[i][0]
				p1y = detectResults.boxes[i][1]
				p2x = detectResults.boxes[i][2]
				p2y = detectResults.boxes[i][3]

		# 修改image是原地？
		color = self.colors[detectResults.classIDs[i]] ###
		cv2.rectangle(image, (p1x, p1y), (p2x, p2y), color, 2)
		text = "{}: {:.4f}".format(self.labels[detectResults.classIDs[i]], detectResults.confidences[i])
		cv2.putText(image, text, (p1x, p1y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		return image
	
	# 矯正
	@staticmethod
	def correct():
		pass

	# 獲得車牌號碼
	@staticmethod
	def getLPNum(LPimage):
		reader = easyocr.Reader(['en']) # need to run only once to load model into memory
		detectLPnum = reader.readtext(LPimage, detail = 0)
		return detectLPnum

	#
	def compareLPNum(self):
		pass

