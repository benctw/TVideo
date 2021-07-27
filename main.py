import math
import numpy as np
import cv2
import os
import sys
import argparse
import easyocr
import Levenshtein
from rich.progress import track

from models.YoloModel.YoloModel import YoloModel
from models.Timeline.Timeline import Timeline

__dirname = os.path.dirname(os.path.abspath(__file__))

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

	def __init__(self, LPModel):
		###!!!
		self.LPModel = LPModel
		self.LPNumber = ""

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
		edged = cv2.Canny(blurred, 20, 160)          # 边缘检测

		cnts,_ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测

		docCnt = None
		if len(cnts) > 0:
			cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 根据轮廓面积从大到小排序
			for c in cnts:
				peri = cv2.arcLength(c, True)                         # 计算轮廓周长
				approx = cv2.approxPolyDP(c, 0.02*peri, True)         # 轮廓多边形拟合
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
		return 1 if self.LPNumber == detectLPNumber else Levenshtein.ratio(detectLPNumber, self.LPNumber)

	# 辨識車牌
	def LPProcess(self, path):
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
			self.saveVideo(self.LPModel.images, "store/output/output.mp4")

			###!!!
			# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

			# out = cv2.VideoWriter('/content/TrafficPolice/store/output/output.mp4', fourcc, 20.0, (640, 360))
			# for i in track(range(0, len(self.LPModel.images)), "[INFO] 寫入影片"):
			# 	out.write(self.LPModel.drawBoxes(self.LPModel.images[i], detectResults[i]))
			# 	print("[INFO] 處理中 {}".format(i))
			# 	print(detectResults[i].display())
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
	def saveVideo(self, frames, path):
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out = cv2.VideoWriter(path, fourcc, 20.0, (640, 360))
		for frame in track(frames, "[INFO] save video"):
			out.write(frame)
		out.release()

	# 生成報告
	def createReport():
		# TODO 先定義 report 的格式和内容
		pass
	
	# 版本更新
	@staticmethod
	def versionUpdate():
		pass


if __name__ == '__main__':
	LPModel = YoloModel(
		namesPath = __dirname + "/static/model/lp.names",
		configPath = __dirname + "/static/model/lp.cfg",
		weightsPath = __dirname + "/static/model/lp.weights"
		# threshold = ,
		# confidence = ,
		# minConfidence = 
	)
	tp = TrafficPolice(LPModel)
	tp.LPNumber = "825BHW"
	imageOrVideoPath = "/content/直行01-(825-BHW，060333-060335).mp4"
	tp.LPProcess(imageOrVideoPath)


