import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
from .CVModelError import CVModelErrors, DetectResultErrors


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


class CVModel(ABCMeta):
    __metaclass__ = ABCMeta
    def __init__(self):
        pass

    @abstractmethod
    def detectImage(image):
        raise NotImplemented

    # 根據 interval 的間隔遍歷一遍影片的幀
    def detectVideo(self, video, interval):
        # TODO 假設 self.detectImage 是 YoloModel 的 detectImage，用 self.detectImage 去辨識幀
        pass