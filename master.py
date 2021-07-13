import math
import cv2
import os

from TrafficPolice.CVModel.CVModel import CVModel
from TrafficPolice.YoloModel.YoloModel import YoloModel
from TrafficPolice.Timeline.Timeline import Timeline

class TrafficPolice:
	# 實例化模型但不加載
	LPModel = YoloModel()

	def __init__(self):
		pass

	def loadLicensePlateModel(self):
		# 調用 <class YoloModel> 父類型 <class CVModel> 的 load 方法加載模型
		self.LPModel.load()

	def loadImage(self):
		pass

	def loadVideo(self):
		pass

	# 更新裁剪的時間點
	def updataMarkPoint(self, start, end):
		pass

	# 辨識車牌
	def licensePlateProcess(self, path):
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
		vector = (p2[0] - p1[0], p2[1] - p1[1])
		# norm = math.sqrt(vector[0]**2 + vector[1]**2)
		# unitVector = (v / norm for v in vector)
		slope = vector[1] / vector[0]
		return slope

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