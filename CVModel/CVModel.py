import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
from .CVModelError import CVModelErrors, DetectResultErrors


class CVModel(ABCMeta):
	__metaclass__ = ABCMeta
	def __init__(self):
		pass

	class DetectResult:
		def __init__(self, boxes = [], confidences = [], classIDs = []):
			if not(isinstance(boxes, list) and isinstance(confidences, list) and isinstance(classIDs, list)):
				raise DetectResultErrors.ArgumentTypeError(boxes, confidences, classIDs, list)
			self.boxes = boxes
			self.confidences = confidences
			self.classIDs = classIDs
		
		# 添加結果
		def add(self, box, confidence, classID):
			self.boxes.append(box)
			self.confidences.append(confidence)
			self.classIDs.append(classID)
			return self

	@abstractmethod
	def detectImage(image):
		raise NotImplemented

	# 根據 interval 的間隔遍歷一遍影片的幀
	def detectVideo(self, vc, interval):
		results = []
		videoImages = self.getImagesFromVideo(vc)
		for image in videoImages[::interval]:
			results.append(self.detectImage(image))
		return results

	@staticmethod
	def getImagesFromVideo(vc):
		videoImages = []
		rval = False
		if vc.isOpened(): rval, videoFrame = vc.read() #判斷是否開啟影片
		while rval:	#擷取視頻至結束
			videoImages.append(videoFrame)
			rval, videoFrame = vc.read()
		vc.release()
		return videoImages
