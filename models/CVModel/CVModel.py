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
	def getFrames(self, videoCapture):
		if type(videoCapture) is str:
			videoCapture = cv2.VideoCapture(videoCapture)
		frames = []
		rval = False
		if videoCapture.isOpened(): rval, frame = videoCapture.read() #判斷是否開啟影片
		while rval:	#擷取視頻至結束

			frames.append(frame)

			rval, frame = videoCapture.read()
		videoCapture.release()
		return frames

	@abstractmethod
	def detectImage(self, image):
		raise NotImplemented

	# 根據 interval 的間隔遍歷一遍影片的幀
	def detectVideo(self, videoCapture, interval = 1):
		if type(videoCapture) is str:
			videoCapture = cv2.VideoCapture(videoCapture)
		results = DetectResults(self.labels)
		self.images = self.getFrames(videoCapture)
		for image in track(self.images[::interval], "detecting"):
			results.add(self.detectImage(image))
		return results


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

	@staticmethod
	def checkColor(color):
		if isinstance(color, (list, tuple)):
			raise TypeError
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
		self.boxes.append(box)
		self.confidences.append(confidence)
		self.classIDs.append(classID)
		return self
	
	def hasResult(self):
		return len(self.classIDs) > 0
	
	def crop(self, image, boxIndex = 0):
		croppedImage = image.copy()
		p1x, p1y, p2x, p2y = self.boxes[boxIndex]
		return croppedImage[p1y:p2y, p1x:p2x]
	
	def display(self):
		header = ['Index', 'Label', 'ClassID', 'Box', 'Confidence']
		rowFormat = '{!s:15} {!s:20} {!s:10} {!s:30} {!s:20}'
		print(rowFormat.format(*header))
		for i in range(0, len(self.classIDs)):
			print(rowFormat.format(i, self.labels[self.classIDs[i]], self.classIDs[i], self.boxes[i], self.confidences[i]))

	def drawBoxes(self):
		self.colors = self.colors if not self.colors is None else self.getAutoSelectColors()
		resultImage = self.image.copy()
		idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.confidence, self.threshold)
		if len(idxs) > 0:
			for i in idxs.flatten():
				p1x, p1y, p2x, p2y = self.boxes[i]
				color = [int(c) for c in self.colors[self.classIDs[i]]]
				cv2.rectangle(resultImage, (p1x, p1y), (p2x, p2y), color, 2)
				text = "{}: {:.4f}".format(self.labels[self.classIDs[i]], self.confidences[i])
				cv2.putText(resultImage, text, (p1x, p1y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		return resultImage


class DetectResults:
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

	def drawBoxes(self):
		results = []
		for detectResult in self.detectResults:
			results.append(detectResult.drawBoxes())
		return results
		
