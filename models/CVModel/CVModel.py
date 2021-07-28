import numpy as np
import cv2
from abc import ABC, ABCMeta, abstractmethod
from .CVModelError import CVModelErrors, DetectResultErrors
from rich.progress import track

class CVModel(ABC):
	def __init__(self):
		###!!!
		self.images = []

	# @staticmethod
	def getImagesFromVideo(self, videoCapture):
		rval = False
		if videoCapture.isOpened(): rval, frame = videoCapture.read() #判斷是否開啟影片
		while rval:	#擷取視頻至結束
			self.images.append(frame)
			rval, frame = videoCapture.read()
    ### 在這釋放？
		videoCapture.release()

	@abstractmethod
	def detectImage(image):
		raise NotImplemented

	# 根據 interval 的間隔遍歷一遍影片的幀
	def detectVideo(self, videoCapture, interval = 1):
		results = []
		# videoImages = self.getImagesFromVideo(videoCapture)
		self.getImagesFromVideo(videoCapture)
		for image in track(self.images[::interval], "detecting"):
			results.append(self.detectImage(image))
		return results

	# def detectVideo2(self, videoCapture, interval = 1):
	# 	results = []
	# 	rval = False
	# 	# 判斷是否開啟影片
	# 	if videoCapture.isOpened(): rval, frame = videoCapture.read()
	# 	frameLength = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
	# 	while rval:
  #   for i in track(frameLength, "detecting"):
	# 		results.append(self.detectImage(frame))
	# 		rval, frame = videoCapture.read()
	# 	### 在這釋放？
	# 	videoCapture.release()
	# 	return results

### 改成只針對yolo的結果
class DetectResult:
	def __init__(self, image, labels = [], threshold = 0.2, confidence = 0.2):
		# if not(isinstance(boxes, list) and isinstance(confidences, list) and isinstance(classIDs, list)):
		# 	raise DetectResultErrors.ArgumentTypeError(classIDs, boxes, confidences, list)
		self.image = image
		self.labels = labels
		self.threshold = threshold
		self.confidence = confidence
		self.boxes = []
		self.confidences = []
		self.classIDs = []
		self.autoSelectColors()

	def autoSelectColors(self):
		self.colors = np.random.randint(0, 255, size = (len(self.labels), 3), dtype = "uint8")

	def setColor(self, index, color):
		if not isinstance(index, (int, str)) and isinstance(color, (list, tuple)):
			raise TypeError
		if len(color) != 3:
			raise ValueError("color 長度不為 3")
		if type(index) is str:
			try:
				index = self.labels.index(index)
			except ValueError:
				raise ValueError("沒有此 label")
		for c in color:
			if c < 0:
				c = 0
			elif c > 255:
				c = 255
		self.colors[index] = color

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