from typing import List, OrderedDict, Tuple, Any, Union, Callable
import math
import numpy as np
import cv2
import os
import sys
import argparse
import Levenshtein
from rich.progress import track

import config as glo
from models.helper import *
from models.TVideo.TVideo import *
from models.TVideo.Process import *


def buildArgparser():
	parser = argparse.ArgumentParser()
	#在此增加共用參數

	subparsers = parser.add_subparsers(help='choose model')

	# create the parser for the "detect" command
	parser_detect = subparsers.add_parser('detect', help='detect')
	#在此增加 detect 參數
	parser_detect.add_argument("path", nargs='*', help='enter your path of video and PL')

	# create the parser for the "yolo" command
	parser_yolo = subparsers.add_parser('yolo', help='Yolo model')
	#在此增加 yolo 參數
	parser_yolo.add_argument("-i", "--image", type=str, help="image path")
	parser_yolo.add_argument("-v", "--video", type=str, help="video path")
	parser_yolo.add_argument("-s", "--save", type=str, help="save path + image name")

	# create the parser for the "resa" command
	parser_resa = subparsers.add_parser('resa', help='Resa model')
	#在此增加 resa 參數

	# create the parser for the "smoke" command
	parser_smoke = subparsers.add_parser('smoke', help='smoke model')
	#在此增加 smoke 參數

	parser_detect.set_defaults(func=detect)
	parser_yolo.set_defaults(func=yolo)
	parser_resa.set_defaults(func=resa)
	parser_smoke.set_defaults(func=smoke)
	args = parser.parse_args()
	args.func(args)
	return

# 執行 detect
def detect(args):
	print("detect")
	TrafficPolice()
	os._exit(0)

#!
# 執行 yolo
def yolo(args):
	TP = TrafficPolice()

	if not args.image is None:
		detectResult = TP.LPModel.detectImage(args.image)
		detectResult.table()
		if not args.save is None:
			resultImage = detectResult.drawBoxes(detectResult.NMSIndexs)
			cv2.imwrite(args.save, resultImage)
			print("saved")

	# video 一定要有--save
	if not args.video is None:
		detectResults = TP.LPModel.detectVideo(args.video)
		saveVideo(detectResults.drawBoxes(indexs=detectResults.NMSIndexs), args.save)
	os._exit(0)

# 執行 resa
def resa(args):
	print("resa")
	os._exit(0)

# 執行 smoke
def smoke(args):
	print("smoke")
	os._exit(0)

# 單例模式
class TrafficPolice:
	instance = None
	def __new__(cls, *args, **kwargs):
		if cls.instance is None:
			cls.instance = super().__new__(cls)
		return cls.instance
	
	def __init__(self):
		self.LPModel = glo.LPModel
		self.targetLPNumber = ""

	# 更新裁剪的時間點
	def updataMarkPoint(self, start, end):
		pass

	# 比較車牌號碼 返回相似度 在0~1之間
	#! 提升速度，可能會直接對比
	@staticmethod
	def compareLPNumber(targetLPNumber, detectLPNumber):
		return 1 if targetLPNumber == detectLPNumber else Levenshtein.ratio(detectLPNumber, targetLPNumber)

	# 判斷車輛行駛方向
	@staticmethod
	def drivingDirection(p1, p2):
		vector = (p1[i] - p2[i] for i in range(0, len(p1)))
		norm = math.sqrt(sum([v ** 2 for v in vector]))
		unitVector = (v / norm for v in vector)
		return unitVector

	# 生成報告
	def createReport(self):
		# TODO 先定義 report 的格式和内容
		pass
	
	# 版本更新
	@staticmethod
	def versionUpdate():
		pass


def main():
	if len(sys.argv) > 1:
		buildArgparser()
	# tp.targetLPNumber = "825BHW"
	# imageOrVideoPath = "/content/gdrive/MyDrive/LP/detectImage/11.jpg"
	# tp.LPProcess(imageOrVideoPath)

  #yolov3 coco model
	# yoloModel = YoloModel(
	# 	namesPath = "/content/gdrive/MyDrive/yolo3/coco.names",
	# 	configPath = "/content/gdrive/MyDrive/yolo3/yolov3.cfg",
	# 	weightsPath = "/content/gdrive/MyDrive/yolo3/yolov3.weights",
	# 	## 至少要有的信心
	# 	confidence=0.2,
	# 	## 可重疊程度
	# 	threshold=0.7
	# )

	# TP = TrafficPolice()
	# image = cv2.imread("D:/chiziSave/image/U20151119083338.jpg")
	# imshow(image)
	# detectResult = TP.LPModel.detectImage(image)
	# detectResult.table()
	# croppedImages = detectResult.cropAll('license plate', indexs=detectResult.NMSIndexs)
	# print(croppedImages)

	# def callbackReturnLPNumber(classID, box, confidence, i):
	# 	return TrafficPolice.getLPNumber(croppedImages[i])

	# detectImage = detectResult.drawBoxes(detectResult.NMSIndexs, callbackReturnLPNumber)
	# imshow(detectImage)

	""""""""""""""""""""""""""""""""""""
	# video 車牌
	# video = cv2.VideoCapture("D:/下載/違規影片-20210820T200841Z-001/違規影片/04-紅燈越線/越線01-(006-PNG，123403-123406).mp4")
	# interval = 3
	# detectResults = TP.LPModel.detectVideo(video, interval)
	# detectResults.table()
	# def callbackReturnTexts(detectResult, frameIndex, classID, box, confidence, i):
	# 	if detectResult.classIDs[i] == 1:
	# 		number = TrafficPolice.getLPNumber(detectResult.crop(i))
	# 		print(f'number: {number}')
	# 		return number
	# 	return None
	# resultImages = detectResults.drawBoxes(detectResults.NMSIndexs, callbackReturnTexts)
	# fps = video.get(cv2.CAP_PROP_FPS)
	# print('fps: ', fps)
	# TP.saveVideo(resultImages, "D:/下載/result/越線01-(006-PNG，123403-123406).mp4", fps / interval)
	

	""""""""""""""""""""""""""""""""""""
	# # video 車牌x
	# video = cv2.VideoCapture("D:/下載/違規影片-20210820T200841Z-001/違規影片/04-紅燈越線/越線06-(AQF-3736，074106-074111).mp4")
	# interval = 8
	# detectResults = TP.LPModel.detectVideo(video, interval)
	# detectResults.table()
	# detectResults.setColors([np.array([0, 0, 255]), np.array([255, 0, 0])])

	# # detector = cv2.SIFT_create()
	# # def callbackKeypoints(detectResult, frameIndex, croppedImage, i):
	# # 	keypoints = detector.detect(croppedImage)
	# # 	img_keypoints = np.empty((croppedImage.shape[0], croppedImage.shape[1], 3), dtype=np.uint8)
	# # 	cv2.drawKeypoints(croppedImage, keypoints, img_keypoints)
	# # 	return  img_keypoints
	# # resultImages = detectResults.draw(detectResults.NMSIndexs, callbackKeypoints)

	# def callbackReturnTexts(detectResult, frameIndex, classID, box, confidence, i):
	# 	# if detectResult.classIDs[i] == 1:
	# 		# lp = LicensePlateData(detectResult.crop(i), detectResult.boxes[i], detectResult.confidences[i])
	# 		# print(f'number: {lp.number}')
	# 		# return lp.number
	# 	if detectResult.classIDs[i] == 0:
	# 		tl = TrafficLightData(detectResult.crop(i), detectResult.boxes[i], detectResult.confidences[i])
	# 		print(f'state: {tl.state.name}')
	# 		return tl.state.name
	# 	return None
	# resultImages = detectResults.drawBoxes(detectResults.NMSIndexs, callbackReturnTexts)
	
	# for i in range(0, len(detectResults.detectResults)):
	# 	detectResults.detectResults[i].image = resultImages[i]

	# def callbackCroppedImage(detectResult, frameIndex, croppedImage, i):
	# 	if detectResult.classIDs[i] == 1:
	# 		# correctedImage, p1, p2, p3, p4 = TP.correct(croppedImage)
	# 		# number = TrafficPolice.getLPNumber(CVModel.crop(correctedImage, detectResult.boxes[i]))
	# 		# print(f'number: {number}')

	# 		cornerPoints = LicensePlateData.getCornerPoints(croppedImage)
	# 		if len(cornerPoints) != 0:
	# 			cornerPoints = np.array(cornerPoints)
	# 			cv2.polylines(croppedImage, [cornerPoints], True, (0, 0, 255), 2, cv2.LINE_AA)
	# 			cv2.line(croppedImage, cornerPoints[0], cornerPoints[2], (0, 255, 0), 2, cv2.LINE_AA)
	# 			cv2.line(croppedImage, cornerPoints[1], cornerPoints[3], (0, 255, 0), 2, cv2.LINE_AA)
	# 	return croppedImage
	# resultImages = detectResults.draw(detectResults.NMSIndexs, callbackCroppedImage)
	# fps = video.get(cv2.CAP_PROP_FPS)
	# print('fps: ', fps)
	# saveVideo(resultImages, "D:/下載/result/越線06-(AQF-3736，074106-074111)1.mp4", fps / interval)
	
	""""""""""""""""""""""""""""""""""""
	tVideo = TVideo('D:/chiziSave/違規影片/03-紅燈直行/直行08-(XS5-327，182607-182609).mp4')
	tVideo.runProcess(TVideoSchedule.forEach, Process.yolo)
	tVideo.runProcess(TVideoSchedule.forEach, Process.findCorrespondingLicensePlate)
	tVideo.runProcess(TVideoSchedule.forEach, Process.drawBoxesLicensePlate)
	tVideo.save('D:/chiziSave/detect-result/直行08-(XS5-327，182607-182609)13.mp4')


if __name__ == '__main__':
	main()
