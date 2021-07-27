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
	def __init__(self, image, classIDs = [], boxes = [], confidences = []):
		if not(isinstance(boxes, list) and isinstance(confidences, list) and isinstance(classIDs, list)):
			raise DetectResultErrors.ArgumentTypeError(classIDs, boxes, confidences, list)
		self.image = image
		self.boxes = boxes
		self.confidences = confidences
		self.classIDs = classIDs
	
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
		header = ['Index', 'ClassID', 'Box', 'Confidence']
		rowFormat = '{!s:15} {!s:15} {!s:30} {!s:15}'
		print(rowFormat.format(*header))
		for i in range(0, len(self.classIDs)):
			print(rowFormat.format(i, self.classIDs[i], self.boxes[i], self.confidences[i]))

	def drawBoxes(self, threshold = 0.2, confidence = 0.2):
		resultImage = self.image.copy()
		idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, confidence, threshold)
		if len(idxs) > 0:
			for i in idxs.flatten():
				p1x, p1y, p2x, p2y = self.boxes[i]
				color = [int(c) for c in self.colors[self.classIDs[i]]]
				cv2.rectangle(resultImage, (p1x, p1y), (p2x, p2y), color, 2)
				text = "{}: {:.4f}".format(self.labels[self.classIDs[i]], self.confidences[i])
				cv2.putText(resultImage, text, (p1x, p1y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		return resultImage