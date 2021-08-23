import numpy as np
import cv2
from abc import ABC, abstractmethod
from .CVModelError import CVModelErrors, DetectResultErrors
from rich.progress import track

class CVModel(ABC):
	def __init__(self):
		self.images = []
		self.labels = []

	@staticmethod
	def getFrames(videoCapture):
		needRelease = False
		if type(videoCapture) is str:
			videoCapture = cv2.VideoCapture(videoCapture)
			needRelease = True
		frames = []
		frame = None
		rval = False
		if videoCapture.isOpened(): 
			rval, frame = videoCapture.read()
		while rval:
			frames.append(frame)
			rval, frame = videoCapture.read()
		if needRelease:
			videoCapture.release()
		return frames

	@abstractmethod
	def detectImage(self, image):
		raise NotImplemented

	# 根據 interval 的間隔遍歷一遍影片的幀
	def detectVideo(self, videoCapture, interval = 1):
		results = DetectResults(self.labels)
		self.images = self.getFrames(videoCapture)
		for image in track(self.images[::interval], "detecting"):
			results.add(self.detectImage(image))
		return results
	
	# 計算任意點數的質心點位置
	# points: [point]
	@staticmethod
	def getCenterPosition(points):
		l = len(points)
		cp = []
		for i in range(0, len(points[0])):
			sum = 0
			for point in points:
				sum += point[i]
			cp.append(sum)
		cp = [c / l for c in cp]
		return cp
	
	# 計算兩個矩形的IoU
	# rect: [p1, p2]
	# p1: [x, y]
	@staticmethod
	def IoU(rect1, rect2):
		iouP1 = [max(rect1[0][0], rect2[0][0]), max(rect1[0][1], rect2[0][1])]
		iouP2 = [min(rect1[1][0], rect2[1][0]), min(rect1[1][1], rect2[1][1])]
		# 交集面積
		iouArea = (iouP2[0] - iouP1[0]) * (iouP2[1] - iouP1[1])
		# 聯集面積 = rect1面積 + rect2面積 - iou面積
		allArea = (rect1[1][0] - rect1[0][0]) * (rect1[1][1] - rect1[0][1]) + (rect2[1][0]-rect2[0][0]) * (rect2[1][1]-rect2[0][1]) - iouArea
		# IoU =  面積交集 / 面積聯集
		return iouArea / allArea
	
	@staticmethod
	def crop(image, box):
		croppedImage = image.copy()
		return croppedImage[box[1]:box[3], box[0]:box[2]]

### 改成只針對yolo的結果
class DetectResult:
	def __init__(self, image, labels = [], threshold = 0.2, confidence = 0.2, colors = None):
		self.image = image
		self.labels = labels
		self.threshold = threshold
		self.confidence = confidence
		self.boxes = []
		self.confidences = []
		self.classIDs = []
		self.colors = colors
		self.NMSIndexs = []

	@staticmethod
	def checkColor(color):
		if isinstance(color, (list, tuple)):
			raise TypeError()
		if len(color) != 3:
			raise ValueError("color 長度不為 3")

	def getAutoSelectColors(self):
		return np.random.randint(0, 255, size = (len(self.labels), 3), dtype = "uint8")

	def setColors(self, colors):
		for color in colors:
			self.checkColor(color)
			color = [max(0, min(round(c), 255)) for c in color]
		self.colors = colors
		return self

	def setColor(self, index, color):
		if not isinstance(index, (int, str)):
			raise TypeError
		self.checkColor(color)
		if type(index) is str:
			try:
				index = self.labels.index(index)
			except ValueError:
				raise ValueError("沒有此 label")
		# color 在 0 到 255 範圍
		color = [max(0, min(round(c), 255)) for c in color]
		self.colors[index] = color
		return self

	# 添加結果
	def add(self, classID, box, confidence):
		box = [abs(p) for p in box]
		self.boxes.append(box)
		self.confidences.append(confidence)
		self.classIDs.append(classID)
		return self
	
	@property
	def count(self):
		return len(self.classIDs)
	
	def hasResult(self):
		return len(self.classIDs) > 0

	def getNMSDetectResult(self):
		NMSDetectResult = DetectResult(self.image, self.labels, self.threshold, self.confidence, self.colors)
		for i in self.NMSIndexs:
			NMSDetectResult.add(self.classIDs[i], self.boxes[i], self.confidences[i])
		return NMSDetectResult

	def calcNMS(self):
		idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.confidence, self.threshold)
		if len(idxs) > 0:	
			self.NMSIndexs = idxs.flatten()

	@property
	def AllIndex(self):
		return [i for i in range(0, self.count)]

	def crop(self, boxIndex):
		croppedImage = self.image.copy()
		p1x, p1y, p2x, p2y = self.boxes[boxIndex]
		return croppedImage[p1y:p2y, p1x:p2x]

	def cropAll(self, classID, indexs):
		croppedImages = []
		# 參數接受 label: str 和 classID: int 類型
		if type(classID) == str:
			classID = self.labels.index(classID)
		elif type(classID) != int:
			raise ValueError('{} 有誤，每個參數都必須是 int 或 str 類型'.format(classID))

		for i in range(0, self.count):
			if (self.classIDs[i] == classID) and (i in indexs):
				croppedImages.append(self.crop(i))
			else:
				croppedImages.append(None)
		
		return croppedImages

	def table(self):
		header = ['Index', 'Label', 'ClassID', 'Box', 'Confidence']
		rowFormat = '{!s:15} {!s:20} {!s:10} {!s:30} {!s:20}'
		print(rowFormat.format(*header))
		for i in range(0, self.count):
			print(rowFormat.format(i, self.labels[self.classIDs[i]], self.classIDs[i], self.boxes[i], self.confidences[i]))

	def msg(self, classID, _, confidence, i):
		percentage = round(confidence * 100)
		return "{}: ({}%)".format(self.labels[classID], percentage)

	def drawBoxes(self, indexs, callbackReturnText = None):
		self.colors = self.colors if not self.colors is None else self.getAutoSelectColors()
		resultImage = self.image.copy()
		for i in indexs:
			p1x, p1y, p2x, p2y = self.boxes[i]
			color = [int(c) for c in self.colors[self.classIDs[i]]]
			# 框
			cv2.rectangle(resultImage, (p1x, p1y), (p2x, p2y), color, 2)
			# 如果沒有定義函數不繪畫字
			if callbackReturnText is None:
				return resultImage
			# 附帶的字
			text = callbackReturnText(self.classIDs[i], self.boxes[i], self.confidences[i], i)
			# 如果沒有信息不繪畫字
			if text == None:
				return resultImage
			# 字型設定
			font = cv2.FONT_HERSHEY_COMPLEX
			fontScale = 1
			fontThickness = 1
			# 顏色反相
			textColor = [255 - c for c in color]
			# 對比色
			# textColor = [color[1], color[2], color[0]]
			# 獲取字型尺寸
			(textW, textH), _ = cv2.getTextSize(text, font, fontScale, fontThickness)
			# 添加字的背景
			cv2.rectangle(resultImage, (p1x, p1y - textH), (p1x + textW, p1y), color, -1)
			# 添加字
			cv2.putText(resultImage, text, (p1x, p1y), font, fontScale, textColor, fontThickness, cv2.LINE_AA)
		return resultImage

	def draw(self, indexs, callbackCroppedImage):
		resultImage = self.image.copy()
		for i in indexs:
			image = callbackCroppedImage(self.crop(i), i)
			p1x, p1y, p2x, p2y = self.boxes[i]
			resultImage[p1y:p2y, p1x:p2x] = image
		return resultImage

class DetectResults:
	#! 改名
	AllIndex = -1
	NMSIndexs = 0

	def __init__(self, labels = [], colors = None):
		self.detectResults = []
		self.labels = labels
		self.colors = colors if colors != None else np.random.randint(0, 255, size = (len(self.labels), 3), dtype = "uint8")

	def add(self, detectResult):
		if not isinstance(detectResult, DetectResult):
			raise TypeError("參數必須是 {} 類型".format(DetectResult))
		if not self.colors is None:
			detectResult.setColors(self.colors)
		self.detectResults.append(detectResult)
		return self

	def setColors(self, colors):
		for detectResult in self.detectResults:
			detectResult.setColors(colors)
		self.colors = colors
		return self
	#!! callbackReturnCustomText 返回一個 str
	def drawBoxes(self, indexs, callbackReturnTexts = None):
		results = []
		# 對所有的結果繪畫框
		if indexs == self.AllIndex:
			for frameIndex, detectResult in enumerate(track(self.detectResults, 'drawing')):
				results.append(detectResult.drawBoxes([int(j) for j in range(0, detectResult.count)], lambda classID, box, confidence, j: callbackReturnTexts(detectResult, frameIndex, classID, box, confidence, j)))
		
		elif indexs == self.NMSIndexs:
			for frameIndex, detectResult in enumerate(track(self.detectResults, 'drawing')):
				results.append(detectResult.drawBoxes(detectResult.NMSIndexs, lambda classID, box, confidence, j: callbackReturnTexts(detectResult, frameIndex, classID, box, confidence, j)))
		
		return results
	
	def loop(self, indexs, callback):
		results = []
		for frameIndex, detectResult in enumerate(self.detectResults):
			for objIndex in range(0, detectResult.count):
				if objIndex in indexs:
					results.append(callback(detectResult, frameIndex, detectResult.classIDs[objIndex], detectResult.boxes[objIndex], detectResult.confidences[objIndex], objIndex))
		return results

	def draw(self, indexs, callbackCroppedImage):
		results = []
		if indexs == self.AllIndex:
			for frameIndex, detectResult in enumerate(track(self.detectResults, 'drawing')):
				results.append(detectResult.draw([int(j) for j in range(0, detectResult.count)], lambda croppedImage, i: callbackCroppedImage(detectResult, frameIndex, croppedImage, i)))
		
		elif indexs == self.NMSIndexs:
			for frameIndex, detectResult in enumerate(track(self.detectResults, 'drawing')):
				results.append(detectResult.draw(detectResult.NMSIndexs, lambda croppedImage, i: callbackCroppedImage(detectResult, frameIndex, croppedImage, i)))
		
		return results


	def table(self):
		for i, detectResult in enumerate(self.detectResults):
			print(f'Frame Index: {i}')
			detectResult.table()