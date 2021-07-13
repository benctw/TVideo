import numpy as np
import argparse
import time
import cv2
import os
import easyocr
import Levenshtein
from TrafficPolice.YoloModel.YoloModelError import YoloModelErrors
from TrafficPolice.CVModel.CVModel import CVModel

# yolo on python
# https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

# opencv dnn
# https://docs.opencv.org/3.4/d6/d0f/group__dnn.html#gafde362956af949cce087f3f25c6aff0d

# opencv net
# https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html


# 應該做成抽象對象，被繼承
class YoloModel(CVModel):

	def __init__(self, namesPath = '', configPath = '', weightsPath = ''):
		self.LPNumber = ''
		self.namesPath = namesPath
		self.configPath = configPath
		self.weightsPath = weightsPath
		self.labels = []
		self.threshold = 0.3
		self.confidence = 0.5
		self.colors = []
		self.minConfidence = 0.2

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

	def detectImage(self, image):
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
				if confidence > self.minConfidence:
					box = detection[0 : 4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					result.add(self.yoloFormatToTwoPoint(centerX, centerY, width, height), float(confidence), classID)
		return result

	def drawBoxes(self, image, detectResults):
		newImage = image.copy()
		idxs = cv2.dnn.NMSBoxes(detectResults.boxes, detectResults.confidences, self.confidence, self.threshold)
		if len(idxs) > 0:
			for i in idxs.flatten():
				p1x = detectResults.boxes[i][0]
				p1y = detectResults.boxes[i][1]
				p2x = detectResults.boxes[i][2]
				p2y = detectResults.boxes[i][3]

				color = self.colors[detectResults.classIDs[i]]
				cv2.rectangle(newImage, (p1x, p1y), (p2x, p2y), color, 2)
				text = "{}: {:.4f}".format(self.labels[detectResults.classIDs[i]], detectResults.confidences[i])
				cv2.putText(newImage, text, (p1x, p1y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		return newImage
	
	# 矯正
	@staticmethod
	def correct(image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (11, 11), 0)
		edged = cv2.Canny(blurred, 20, 160)          # 边缘检测

		cnts,_ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测

		docCnt = None
		if len(cnts) > 0:
			cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 根据轮廓面积从大到小排序
			for c in cnts:
				peri = cv2.arcLength(c, True)                         # 计算轮廓周长
				approx = cv2.approxPolyDP(c, 0.02*peri, True)         # 轮廓多边形拟合
				# 轮廓为4个点表示找到纸张
				if len(approx) == 4:
					docCnt = approx
					break
		for peak in docCnt:
			peak = peak[0]
			cv2.circle(image, tuple(peak), 10, (255, 0, 0))
			
		H, W = image.shape[:2]

		point_set_0 = np.float32([docCnt[1,0],docCnt[2,0],docCnt[3,0],docCnt[0,0]])
		point_set_1 = np.float32([[0, 0],[0, 140],[440, 140],[440, 0]])

		# 变换矩阵
		mat = cv2.getPerspectiveTransform(point_set_0, point_set_1)
		# 投影变换
		lic = cv2.warpPerspective(image, mat, (440, 140))

	# 獲得車牌號碼
	@staticmethod
	def getLPNumber(LPImage):
		reader = easyocr.Reader(['en']) # need to run only once to load model into memory
		return reader.readtext(LPImage, detail = 0)

	# 比較車牌號碼
	def compareLPNum(self, detectLPNumber):
		return 1 if self.LPNumber == detectLPNumber else Levenshtein.ratio(detectLPNumber, self.LPNumber)
	
		
