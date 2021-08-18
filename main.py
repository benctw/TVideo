import math
import numpy as np
import cv2
# 以下這一句為了解決vscode報錯問題，運行時沒必要
# from cv2 import cv2
import os
import sys
import argparse
import easyocr
import Levenshtein
from rich.progress import track

from models.YoloModel.YoloModel import YoloModel
from models.Timeline.Timeline import Timeline
from models.helper import *

#!!
__dirname = os.path.dirname(os.path.abspath(__file__))


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
	TrafficPolice().process(args.detect)
	os._exit(0)

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
		TP.saveVideo(detectResults.drawBoxes(), args.save)
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
		#!!
		__dirname = os.path.dirname(os.path.abspath(__file__))

		self.LPModel = YoloModel(
			namesPath   = __dirname + "/static/model/lp.names",
			configPath  = __dirname + "/static/model/lp_yolov4.cfg",
			# weightsPath = __dirname + "/static/model/lp_yolov4_final.weights"
			weightsPath = "D:/chiziSave/TrafficPoliceYoloModel/model/lp_yolov4_final.weights"
		)
		self.targetLPNumber = ""

	def process():
		pass

	#//////// 共同 ////////#

	def loadImage(self):
		pass

	def loadVideo(self):
		pass

	# 更新裁剪的時間點
	def updataMarkPoint(self, start, end):
		pass

	#//////// 車牌 ////////#

	# 獲得車牌號碼
	@staticmethod
	def getLPNumber(LPImage):
		reader = easyocr.Reader(['en']) # need to run only once to load model into memory
		return reader.readtext(LPImage, detail = 0)
	
	# 矯正
	@staticmethod
	def correct(image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (11, 11), 0)
		edged = cv2.Canny(blurred, 20, 160)	# 边缘检测

		cnts,_ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)	# 轮廓检测

		docCnt = None
		if len(cnts) > 0:
			cnts = sorted(cnts, key=cv2.contourArea, reverse=True)	# 根据轮廓面积从大到小排序
			for c in cnts:
				peri = cv2.arcLength(c, True)	# 计算轮廓周长
				approx = cv2.approxPolyDP(c, 0.02*peri, True)	# 轮廓多边形拟合
				# 轮廓为4个点表示找到纸张
				if len(approx) == 4:
					docCnt = approx
					break
		for peak in docCnt:
			peak = peak[0]
			cv2.circle(image, tuple(peak), 10, (255, 0, 0))
			
		H, W = image.shape[:2]

		point_set_0 = np.float32([docCnt[1,0],docCnt[2,0],docCnt[3,0],docCnt[0,0]])
		point_set_1 = np.float32([[0, 0],[0, 140],[440, 140],[440, 0]])

		# 变换矩阵
		mat = cv2.getPerspectiveTransform(point_set_0, point_set_1)
		# 投影变换
		lic = cv2.warpPerspective(image, mat, (440, 140))

	
	# 比較車牌號碼 返回相似度 在0~1之間
	def compareLPNumber(self, detectLPNumber):
		return 1 if self.targetLPNumber == detectLPNumber else Levenshtein.ratio(detectLPNumber, self.targetLPNumber)

	# 辨識車牌流程
	def LPProcess(self, path, targetLPNumber):
		if path.lower().endswith(('.jpg', '.jpeg', '.png')):
			image = cv2.imread(path)
			detectResult = self.LPModel.detectImage(image)
			if detectResult.hasResult():
				LPImage = detectResult.crop(image)
				# correctedLPImage = self.correct(LPImage)
				LPNumber = self.getLPNumber(LPImage)
				print('[INFO] ', LPNumber)
				similarity = self.compareLPNumber(''.join(LPNumber))
				if similarity == 1:
					print('[INFO] 找到對應車牌號碼: {}'.format(LPNumber))
		elif path.lower().endswith(('.mp4', '.avi')):
			videoCapture = cv2.VideoCapture(path)
			detectResults = self.LPModel.detectVideo(videoCapture)
			self.saveVideo(detectResults.drawBoxes(), "/content/gdrive/MyDrive/video/result.mp4")

			###!!!
			# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

			# out = cv2.VideoWriter('/content/TrafficPolice/store/output/output.mp4', fourcc, 20.0, (640, 360))
			# for i in track(range(0, len(self.LPModel.images)), "[INFO] 寫入影片"):
			# 	out.write(self.LPModel.drawBoxes(self.LPModel.images[i], detectResults[i]))
			# 	print("[INFO] 處理中 {}".format(i))
			# 	print(detectResults[i].table())
			# out.release()
			# print("[INFO] 完成")

		else:
			print('[INFO] 不支持的文件格式！')
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

	def markVehicle(box):
		pass
	
	def guessPositionOfNextMoment():
		pass

	# 儲存片段到指定路徑 ###只支持MP4
	@staticmethod
	def saveVideo(images, path):
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		height, width = images[0].shape[:2]
		out = cv2.VideoWriter(path, fourcc, 20.0, (int(width), int(height)))
		for image in track(images, "saving video"):
			out.write(image)
		out.release()

	# 生成報告
	def createReport():
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

	TP = TrafficPolice()
	image = cv2.imread("D:/chiziSave/image/U20151119083338.jpg")
	imshow(image)
	detectResult = TP.LPModel.detectImage(image)
	detectResult.table()
	croppedImages = detectResult.cropAll2('license plate', indexs=detectResult.NMSIndexs)
	print(croppedImages)

	def callbackReturnLPNumber(classID, box, confidence, i):
		return ''.join(TrafficPolice.getLPNumber(croppedImages[i]))

	detectImage = detectResult.drawBoxes(detectResult.NMSIndexs, callbackReturnLPNumber)
	imshow(detectImage)

	# detectResults = TP.LPModel.detectVideo("videoplayback.mp4")
	# def customText(detectResult):
	# 	detectResult
	
	# def callbackDetectResultReturnCustomTexts(detectResult):
	# 	customTexts = []
	# 	croppedImages = detectResult.cropAll(1)
	# 	for croppedImage in croppedImages:
	# 		customTexts.append(''.join(TrafficPolice.getLPNumber(croppedImage)))
	# 	return customTexts

	# detectResults.drawBoxes(callbackDetectResultReturnCustomTexts)
	# TP.saveVideo(detectResults.drawBoxes(), "videoplayback_result_車牌分析.mp4")

	# resultImage = detectResult.drawBoxes()

	# cv2.imwrite("/content/TrafficPolice/store/output/2.jpg", resultImage)


if __name__ == '__main__':
	main()
