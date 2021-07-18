import math
import cv2
import os
import sys
import argparse

from TrafficPolice.models.YoloModel.YoloModel import YoloModel
from TrafficPolice.models.Timeline.Timeline import Timeline

# sys.path.append("models")
# from YoloModel.YoloModel import YoloModel
# from Timeline.Timeline import Timeline

def buildArgparser():
    parser = argparse.ArgumentParser()
    #在此增加共用參數

    subparsers = parser.add_subparsers(help='choose model')

    # create the parser for the "detect" command
    parser_detect = subparsers.add_parser('detect', help='detect')
    #在此增加 detect 參數

    # create the parser for the "yolo" command
    parser_yolo = subparsers.add_parser('yolo', help='Yolo model')
    #在此增加 yolo 參數
    parser_yolo.add_argument('-o', '--option', type=int, help='yolo option')

    # create the parser for the "resa" command
    parser_resa = subparsers.add_parser('resa', help='Resa model')
    #在此增加 resa 參數
    parser_resa.add_argument('-o', '--option', type=str, help='resa option')

    # create the parser for the "smoke" command
    parser_smoke = subparsers.add_parser('smoke', help='smoke model')
    #在此增加 smoke 參數
    parser_smoke.add_argument('-o', '--option', type=float, help='smoke option')

    parser_detect.set_defaults(func=detect)
    parser_yolo.set_defaults(func=yolo)
    parser_resa.set_defaults(func=resa)
    parser_smoke.set_defaults(func=smoke)
    args = parser.parse_args()
    args.func(args)
    return

#執行 detect
def detect(args):
    pass

#執行 yolo
def yolo(args):
    pass

#執行 resa
def resa(args):
    pass

#執行 smoke
def smoke(args):
    pass

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