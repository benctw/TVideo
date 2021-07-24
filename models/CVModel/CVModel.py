import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
from .CVModelError import CVModelErrors, DetectResultErrors


class CVModel(ABCMeta):
	__metaclass__ = ABCMeta
	def __init__(self):
		pass

	class DetectResult:
		def __init__(self, classIDs = [], boxes = [], confidences = []):
			if not(isinstance(boxes, list) and isinstance(confidences, list) and isinstance(classIDs, list)):
				raise DetectResultErrors.ArgumentTypeError(classIDs, boxes, confidences, list)
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

	@staticmethod
	def getImagesFromVideo(videoCapture):
		videoImages = []
		rval = False
		if videoCapture.isOpened(): rval, videoFrame = videoCapture.read() #判斷是否開啟影片
		while rval:	#擷取視頻至結束
			videoImages.append(videoFrame)
			rval, videoFrame = videoCapture.read()
		videoCapture.release()
		return videoImages

	@abstractmethod
	def detectImage(image):
		raise NotImplemented

	# 根據 interval 的間隔遍歷一遍影片的幀
	def detectVideo(self, videoCapture, interval):
		results = []
		videoImages = self.getImagesFromVideo(videoCapture)
		for image in videoImages[::interval]:
			results.append(self.detectImage(image))
		return results
