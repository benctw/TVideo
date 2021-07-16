import math
import cv2
import os
import sys

from TrafficPolice.models.YoloModel.YoloModel import YoloModel
from TrafficPolice.models.Timeline.Timeline import Timeline

# sys.path.append("models")
# from YoloModel.YoloModel import YoloModel
# from Timeline.Timeline import Timeline

class TrafficPolice:
	# 實例化模型但不加載 net
	LPModel = YoloModel()

	def __init__(self):
		pass

	def loadImage(self):
		pass

	def loadVideo(self):
		pass

	# 更新裁剪的時間點
	def updataMarkPoint(self, start, end):
		pass

	# 辨識車牌
	def LPProcess(self, path):
		if path.lower().endswith(('.jpg', '.jpeg', '.png')):
			image = cv2.imread(path)
			detectResult = self.LPModel.detectImage(image)
			if detectResult.hasResult():
				LPImage = detectResult.crop(image)
				# correctedLPImage = self.LPModel.correct(LPImage)
				LPNumber = self.LPModel.getLPNumber(LPImage)
				print(LPNumber)
				similarity = self.LPModel.compareLPNumber(LPNumber)
				if similarity == 1:
					print('找到對應車牌號碼: {}'.format(LPNumber))
		elif path.lower().endswith(('.mp4', '.avi')):
			videoCapture = cv2.VideoCapture(path)
			detectResult = self.LPModel.detectVideo(videoCapture)
		else:
			print('不支持的文件格式！')
			return
		

	# 辨識車輛
	def detectCar(self):
		# 返回時間序列
		return Timeline()

	# 辨識紅綠燈
	def detectTrafficLight(self):
		# 返回時間序列
		return Timeline()

	# 辨識交通標誌 
	def detectTrafficSigns():
		return Timeline()

	# 判斷車輛行駛方向
	def drivingDirection(p1, p2):
		vector = (p1[i] - p2[i] for i in range(0, len(p1)))
		norm = math.sqrt(sum([v ** 2 for v in vector]))
		unitVector = (v / norm for v in vector)
		return unitVector

	# 裁剪影片從 start 到 end
	def cropVideo(self, start, end):
		pass

	# 儲存片段到指定路徑
	def saveVideo(self, video, path):
		pass

	# 生成報告
	def createReport():
    	# TODO 先定義 report 的格式和内容
		pass
	
	# 版本更新
	@staticmethod
	def versionUpdate():
		pass