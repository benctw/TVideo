import numpy as np
import argparse
import time
import cv2
import os
import easyocr
import Levenshtein
from TrafficPolice.YoloModel.YoloModelError import YoloModelErrors
from TrafficPolice.CVModel.CVModel import CVModel


# 應該做成抽象對象，被繼承
class YoloModel(CVModel):

	def __init__(self):
		self.LPNumber = ''

	# (cx, cy, w, h) -> (p1x, p1y, p2x, p2y)
	@staticmethod
	def yoloFormatToTwoPoint(centerX, centerY, width, height):
		# 計算角點坐標
		p1x = int(centerX - (width / 2))
		p1y = int(centerY - (height / 2))
		p2x = p1x + width
		p2y = p1y + height
		return [p1x, p1y, p2x, p2y]

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

	def load(self, namesPath, configPath, weightsPath, threshold = 0.2, confidence = 0.5, minConfidence = 0.2):
		print("Loading YOLO Model...")
		self.namesPath = os.path.join(*namesPath.split('\\'))
		self.configPath = os.path.join(*configPath.split('\\'))
		self.weightsPath = os.path.join(*weightsPath.split('\\'))
		self.threshold = threshold
		self.confidence = confidence
		self.minConfidence = minConfidence
		## TODO error handling
		self.labels = open(self.namesPath).read().strip().split('\n')
		self.colors = np.random.randint(0, 255, size = (len(self.labels), 3), dtype = "uint8")
		self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
		self.outputLayerNames = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
		print("Loaded YOLO Model!")

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
					result.add(classID, self.yoloFormatToTwoPoint(centerX, centerY, width, height), float(confidence))
		return result

	def drawBoxes(self, image, detectResult):
		newImage = image.copy()
		idxs = cv2.dnn.NMSBoxes(detectResult.boxes, detectResult.confidences, self.confidence, self.threshold)
		if len(idxs) > 0:
			for i in idxs.flatten():
				p1x, p1y, p2x, p2y = detectResult.boxes[i]
				color = self.colors[detectResult.classIDs[i]]
				cv2.rectangle(newImage, (p1x, p1y), (p2x, p2y), color, 2)
				text = "{}: {:.4f}".format(self.labels[detectResult.classIDs[i]], detectResult.confidences[i])
				cv2.putText(newImage, text, (p1x, p1y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		return newImage
	
	# 比較車牌號碼 返回相似度 在0~1之間
	def compareLPNumber(self, detectLPNumber):
		return 1 if self.LPNumber == detectLPNumber else Levenshtein.ratio(detectLPNumber, self.LPNumber)
	
		
