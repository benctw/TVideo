import math
import time
import os
import sys
import argparse

import config as cfg
from models.helper import *
from models.TVideo.TVideo import *
from models.TVideo.Process import *
from models.communicate import *
from models.TVideo.Record import Record

def buildArgparser():
	parser = argparse.ArgumentParser()
	#在此增加共用參數

	subparsers = parser.add_subparsers(help='choose model')

	# create the parser for the "detect" command
	parser_detect = subparsers.add_parser('detect', help='detect')
	#在此增加 detect 參數
	parser_detect.add_argument("path", nargs='*', help='enter your path of video and PL')



	parser_findNumber = subparsers.add_parser('findNumber', help='findNumber')
	parser_findNumber.add_argument("data", nargs='*', help='')


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

	parser_findNumber.set_defaults(func=findNumber)

	parser_yolo.set_defaults(func=yolo)
	parser_resa.set_defaults(func=resa)
	parser_smoke.set_defaults(func=smoke)
	args = parser.parse_args()
	args.func(args)
	return


# 執行 detect
def detect(args):
	print("detect")
	os._exit(0)

# 執行 findNumber
def findNumber(args):
	INFO('run findNumber')
	paths   = args.data[::2]
	numbers = args.data[1::2]
	INFO(paths)
	INFO(numbers)
	INFO.progress(0)
	tVideo = TVideo(paths[0], numbers[0])
	tVideo.runProcess(
		TVideoSchedule.forEachStepAll(60), 
		Process.showIndex, 
		Process.yolo, 
		# Process.cocoDetect, 
		Process.calcLicensePlateData, 
		Process.findCorresponding(), 
		Process.findTargetNumber()
	)
	INFO.progress(0.25)
	currentIndex = tVideo.currentIndex
	tVideo.runProcess(
		TVideoSchedule.forward(currentIndex + 1, 5), 
		Process.showIndex, 
		Process.yolo, 
		# Process.cocoDetect,
		# Process.calcLicensePlateData, 
		Process.calcCenterPosition,
		Process.findCorresponding(), 
		Process.hasCorrespondingTargetLicensePlate
	)
	INFO.progress(0.50)
	tVideo.runProcess(
		TVideoSchedule.backward(currentIndex - 1, 5), 
		Process.showIndex, 
		Process.yolo, 
		# Process.cocoDetect,
		# Process.calcLicensePlateData, 
		Process.calcCenterPosition,
		Process.findCorresponding(reverse=True), 
		Process.hasCorrespondingTargetLicensePlate
	)
	INFO.progress(0.75)
	tVideo.runProcess(
		TVideoSchedule.forEach, 
		Process.drawBoxes,
		Process.correspondingTrafficLights,
		Process.drawCurrentTrafficLightState,
		Process.updateRangeOfTargetLicensePlate,
	)
	tVideo.runProcess(
		TVideoSchedule.once,
		Process.calcPathDirection
	)
	tVideo.runProcess(
		TVideoSchedule.once,
		Process.intersectionOfLPAndTL
	)
	record = Record()
	print('dddd:', f'{cfg.outputDir}/result-video_Record_{record.getLastRecordId()}_{tVideo.fileName}.mp4')
	tVideo.save(f'{cfg.outputDir}/result-video_Record_{record.getLastRecordId() + 1}_{tVideo.fileName}.mp4')
	record.save(tVideo)
	INFO.progress(1)
	os._exit(0)

#!
# 執行 yolo
def yolo(args):
	if not args.image is None:
		boxes, classIDs, confidences = cfg.LPModel.detect(args.image)

		

	# 	if not args.save is None:
	# 		resultImage = detectResult.drawBoxes(detectResult.NMSIndexs)
	# 		cv2.imwrite(args.save, resultImage)
	# 		print("saved")

	# # video 一定要有--save
	# if not args.video is None:
	# 	detectResults = TP.LPModel.detectVideo(args.video)
	# 	saveVideo(detectResults.drawBoxes(indexs=detectResults.NMSIndexs), args.save)
	os._exit(0)

# 執行 resa
def resa(args):
	print("resa")
	os._exit(0)

# 執行 smoke
def smoke(args):
	print("smoke")
	os._exit(0)


# 比較車牌號碼 返回相似度 在0~1之間
#! 提升速度，可能會直接對比
# def compareLPNumber(targetLPNumber, detectLPNumber):
# 	return 1 if targetLPNumber == detectLPNumber else Levenshtein.ratio(detectLPNumber, targetLPNumber)

# 判斷車輛行駛方向
def drivingDirection(p1, p2):
	vector     = (p1[i] - p2[i] for i in range(0, len(p1)))
	norm       = math.sqrt(sum([v ** 2 for v in vector]))
	unitVector = (v / norm for v in vector)
	return unitVector


def main():
	if len(sys.argv) > 1: buildArgparser()

	start = time.process_time()
	INFO.progress(0)
	tVideo = TVideo('C:/Users/zT3Tz/Documents/違規影片/04-紅燈越線/越線04(267-MAE，095248-095254).mp4', '267MAE')
	tVideo.runProcess(
		TVideoSchedule.forEachStepAll(30), 
		Process.showIndex, 
		Process.yolo, 
		# Process.cocoDetect, 
		Process.calcLicensePlateData, 
		Process.findCorresponding(), 
		Process.findTargetNumber()
	)
	INFO.progress(0.25)
	currentIndex = tVideo.currentIndex
	tVideo.runProcess(
		TVideoSchedule.forward(currentIndex + 1, 10), 
		Process.showIndex, 
		Process.yolo, 
		# Process.cocoDetect,
		# Process.calcLicensePlateData, 
		Process.calcCenterPosition,
		Process.findCorresponding(), 
		Process.hasCorrespondingTargetLicensePlate
	)
	INFO.progress(0.50)
	tVideo.runProcess(
		TVideoSchedule.backward(currentIndex - 1, 10), 
		Process.showIndex, 
		Process.yolo, 
		# Process.cocoDetect,
		# Process.calcLicensePlateData, 
		Process.calcCenterPosition,
		Process.findCorresponding(reverse=True), 
		Process.hasCorrespondingTargetLicensePlate
	)
	INFO.progress(0.75)
	tVideo.runProcess(
		TVideoSchedule.forEach, 
		Process.drawBoxes,
		Process.correspondingTrafficLights,
		Process.drawCurrentTrafficLightState,
		Process.updateRangeOfTargetLicensePlate,
	)
	tVideo.runProcess(
		TVideoSchedule.once,
		Process.calcPathDirection
	)
	tVideo.runProcess(
		TVideoSchedule.once,
		Process.intersectionOfLPAndTL
	)
	tVideo.save('C:/Users/zT3Tz/Documents/detect-result/越線04(267-MAE，095248-095254)8.mp4')
	end = time.process_time()
	print(end - start)
	print(tVideo.directs)
	record = Record()
	record.save(tVideo)
	INFO.progress(1)

if __name__ == '__main__':
	main()
