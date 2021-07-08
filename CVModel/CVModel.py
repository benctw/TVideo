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
		video_images=self.get_images_from_video(video_name)
		for i in range(0, len(video_images))
			detectVideo=DetectResult(video_images[i])
    @staticmethod
    def get_images_from_video(video_name):
    	video_images = []
        vc = cv2.VideoCapture(video_name)
        c = 0

        if vc.isOpened():	#判斷是否開啟影片
		rval, video_frame = vc.read()
	else:
		rval = False

	while rval:	#擷取視頻至結束
		rval, video_frame = vc.read()

		if(c % 30 == 0):	#每隔30幀進行擷取
			video_images.append(video_frame)     
		c = c + 1
	vc.release()
    
	return video_images
