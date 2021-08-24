import re
from typing import List, Tuple, Any, Union, Callable
from enum import Enum
import easyocr
import numpy as np
import cv2
from rich.progress import track

from models.CVModel.CVModel import CVModel, DetectResult, DetectResults

Point = Tuple[int, int]
Box = Tuple[Point, Point]

# 一載具的數據
class VehicleData:
	def __init__(
		self, 
		image: np.ndarray, 
		box  : List,
		confidence: float
	):
		self.image = image
		self.box = box
		self.confidence = confidence
		self.label = 'Vehicle'
		# self.calc()

	# 生成之後計算的數據
	def calc(self):
		self.cornerPoints = ...
		# 載具質心點位置
		self.centerPosition: List = CVModel.getCenterPosition(self.cornerPoints)
		# 方向
		# self.direction = direction
		# 下一時刻可能的位置
		# self.possiblePositionAtTheNextMoment: List = possiblePositionAtTheNextMoment
		# 對應的車牌
		#!


# 一車牌的數據
class LicensePlateData:

	# 車牌的長寬比
	ratioOfLicensePlate = 2.375

	def __init__(
		self, 
		image: np.ndarray, 
		box  : List,
		confidence: float
	):
		self.image = image
		self.box = box
		self.confidence = confidence
		self.label = 'LicensePlate'
		# self.calc()
	
	# 生成之後計算的數據
	def calc(self):
		# 車牌的四個角點
		self.cornerPoints  : List = self.getCornerPoints(self.image)
		# 矯正的圖像
		self.correctImage  : np.ndarray = self.correct(self.image, self.cornerPoints, int(150 * self.ratioOfLicensePlate), 150)
		# 車牌質心點位置
		self.centerPosition: List = CVModel.getCenterPosition(self.cornerPoints)
		# 車牌號碼
		self.number        : str = self.getNumber(self.correctImage)
	
	@staticmethod
	def getCornerPoints(image: np.ndarray) -> List:
		img = image.copy()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.medianBlur(img, 3)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
		close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
		normed = cv2.normalize(close, None, 0, 255, cv2.NORM_MINMAX)
		img = normed
		_, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
		cnts, _ = cv2.findContours(img ,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cntsArea = [cv2.contourArea(c) for c in cnts]
		maxIndex = np.argmax(cntsArea)
		maxCnt = cnts[maxIndex]
		# 用凸包把內部細節去除
		maxCnt = cv2.convexHull(maxCnt, True, False)
		# 找出4個點來趨近它
		l = cv2.arcLength(maxCnt, True)
		# e從0開始增加0.01，至到能用4個點逼近為止
		e = 0
		approx = cv2.approxPolyDP(maxCnt, e * l, True)
		while True:
			e += 0.001
			approx = cv2.approxPolyDP(maxCnt, e * l, True)
			if len(approx) == 4:
				print(f"e = {e}")
				break
			if len(approx) < 4:
				print(f"e = {e} 擬合不到 4 個點")
				# 沒矯正結果
				return []
		# 重新排序4個角點的順序
		# approxPolyDP 返回角點的順序每次都不一樣
		# 所以排序成從左上角開始逆時針排序
		p1, p2, p3, p4 = [tuple(p.flatten()) for p in approx]
		# 質心點
		cx = int((p1[0] + p2[0] + p3[0] + p4[0]) / 4)
		# 沒用到 cy
		# cy = int((p1[1] + p2[1] + p3[1] + p4[1]) / 4)
		# 左邊的點 和 右邊的點
		lp = []
		rp = []
		# 分割左右點
		for p in [p1, p2, p3, p4]:
			lp.append(p) if p[0] < cx else rp.append(p)
		lp = sorted(lp, key = lambda s: s[1])
		rp = sorted(rp, key = lambda s: s[1], reverse=True)
		return lp + rp
	
	@staticmethod
	def correct(image: np.ndarray, cornerPoints, w: int, h: int) -> np.ndarray:
		cornerPoints = np.float32(cornerPoints)
		dst = np.float32(np.array([[0, 0], [0, h], [w, h], [w, 0]]))
		mat = cv2.getPerspectiveTransform(cornerPoints, dst)
		return cv2.warpPerspective(image.copy(), mat, (w, h))

	@staticmethod
	def getNumber(image: np.ndarray) -> str:
		h, w = image.shape[:2]
		imageCenterPoint = (w / 2, h / 2)
		reader = easyocr.Reader(['en'])
		easyocrResult: List[Any] = reader.readtext(image)
		if len(easyocrResult) == 0:
			return ''
		elif len(easyocrResult) == 1:
			return re.sub(r'[^\dA-Z]+', '', easyocrResult[0][1].upper())
		# 只提取最接近中心點的文字
		distanceFromCenterPoints = []
		for box, _, _ in easyocrResult:
			centerPoint = CVModel.getCenterPosition(box)
			distanceFromCenterPoints.append(abs(centerPoint[0] - imageCenterPoint[0]) + abs(centerPoint[1] - imageCenterPoint[1]))
		minIndex =  np.argmin(distanceFromCenterPoints)
		return re.sub(r'[^\dA-Z]+', '', easyocrResult[minIndex][1].upper())


# 紅綠燈狀態
class TrafficLightState(Enum):
	unknow = 0
	red    = 1
	yellow = 2
	green  = 3

class TrafficLightData:
	def __init__(
		self,
		image: np.ndarray,
		box  : List,
		confidence: float
	):
		self.image = image
		self.box = box
		self.confidence = confidence
		self.label = 'TrafficLight'
		# self.calc()
	
	def calc(self):
		# 紅綠燈狀態
		self.state: TrafficLightState = self.ColorDectect(self.image, *self.getTrafficLightColor(self.image))
	
	@staticmethod
	def getTrafficLightColor(image: np.ndarray) -> List[int]:
		hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		# min and max HSV values
		redMin  = np.array([0, 5, 150])
		redMax  = np.array([8, 255, 255])
		redMin2 = np.array([175, 5, 150])
		redMax2 = np.array([180, 255, 255])

		yellowMin = np.array([20, 5, 150])
		yellowMax = np.array([30, 255, 255])

		greenMin = np.array([49, 79, 137])
		greenMax = np.array([90, 255, 255])

		redThresh    = cv2.inRange(hsvImg, redMin, redMax) + cv2.inRange(hsvImg, redMin2, redMax2)
		yellowThresh = cv2.inRange(hsvImg, yellowMin, yellowMax)
		greenThresh  = cv2.inRange(hsvImg, greenMin, greenMax)

		redBlur    = cv2.medianBlur(redThresh, 5)
		yellowBlur = cv2.medianBlur(yellowThresh, 5)
		greenBlur  = cv2.medianBlur(greenThresh, 5)

		red    = cv2.countNonZero(redBlur)
		yellow = cv2.countNonZero(yellowBlur)
		green  = cv2.countNonZero(greenBlur)	
		
		return [red, yellow, green]
		
		# if lightColor > 60:
    	# 		if lightColor == red: return TrafficLightState.red
		# 	elif lightColor == yellow: return TrafficLightState.yellow
		# 	elif lightColor == green: return TrafficLightState.green
		# else:
		# 	image = CVModel.OverexPose(image)
		
		# return TrafficLightState.unknow

	@staticmethod
	def ColorDectect(image: np.ndarray, red: int, yellow: int, green: int) -> TrafficLightState:
		lightColor = max(red, yellow, green)
		if lightColor >= 20:
			if lightColor == red: return TrafficLightState.red
			elif lightColor == yellow: return TrafficLightState.yellow
			elif lightColor == green: return TrafficLightState.green
		elif lightColor < 20:
			image = CVModel.OverexPose(image)
			overexpose = image.copy()
			equalizeOver = np.zeros(overexpose.shape, overexpose.dtype)
			equalizeOver[:, :, 0] = cv2.equalizeHist(overexpose[:, :, 0])
			equalizeOver[:, :, 1] = cv2.equalizeHist(overexpose[:, :, 1])
			equalizeOver[:, :, 2] = cv2.equalizeHist(overexpose[:, :, 2])
			overexposeColor = TrafficLightData.getTrafficLightColor(overexpose)
			lightColor = max(*overexposeColor)
			if lightColor >= 20:
				if overexposeColor[0] == red: return TrafficLightState.red
				elif overexposeColor[1] == yellow: return TrafficLightState.yellow
				elif overexposeColor[2] == green: return TrafficLightState.green
		
		return TrafficLightState.unknow

# 一幀的數據
class TFrameData:
	def __init__(
		self, 
		frame                 : np.ndarray, 
		# vehicles              : List[VehicleData], 
		# licensePlates         : List[LicensePlateData], 
		# trafficLights         : List[TrafficLightData], 
		# hasTrafficLight       : bool, 
		# hasLicensePlate       : bool, 
		# hasMatchTargetLPNumber: bool, 
	):
		# 圖像
		self.frame                  = frame
		# self.editedFrame: np.ndarray = frame.copy()
		# self.temp: Any = ''
		# self.objs: List[List[Any]] = [
		# 	# 載具數據
		# 	[],
		# 	# 車牌數據
		# 	[],
		# 	# 紅綠燈數據
		# 	[]
		# ]
		# 載具數據
		self.vehicles: List[VehicleData] = []
		# 車牌數據
		self.licensePlates: List[LicensePlateData] = []
		# 紅綠燈數據
		self.trafficLights: List[TrafficLightData] = [] 
		# # 有沒有紅綠燈
		# self.hasTrafficLight        = hasTrafficLight
		# # 有沒有車牌
		# self.hasLicensePlate        = hasLicensePlate
		# 是否匹配到車牌號碼
		# self.hasMatchTargetLPNumber = hasMatchTargetLPNumber

	def addObj(self, className: str, data: Any):
		if hasattr(self, className):
			# setattr(self, className, getattr(self, className, [data]))
			attr = getattr(self, className)
			#! 不知道為什麼會有 None
			if attr is None:
				attr = []
			setattr(self, className, attr.append(data))
		else:
			setattr(self, className, [data])


class ProcessState(Enum):
	next = 1
	stop = 2

ForEachFrameData = Callable[[TFrameData, int, Any], ProcessState]

# 一影片的數據
class TVideo:
	def __init__(
		self, 
		path : str,
		lastCodename: int = 0
	):
		self.path = path

		videoDetails = self.__getVideoDetails(path)
		self.frames    : List[np.ndarray] = videoDetails[0]
		self.width     : float = videoDetails[1]
		self.height    : float = videoDetails[2]
		self.fps       : float = videoDetails[3]
		self.frameCount: int   = videoDetails[4]

		# 多幀數據（從 video 初始化）
		self.framesData: List[TFrameData] = [TFrameData(frame) for frame in track(self.frames, '初始化每幀數據')]
		# 最後使用的代號
		self.lastCodename = lastCodename

		# 裁剪影片的開始
		self.start: int = 0
		# 裁剪影片的結束，-1 是最後
		self.end: int = -1
		# # 每幀會處理的流程 #! 沒有用到
		# self.process: OrderedDict[str, ForEachFrameData] = collections.OrderedDict()
		# # 上一個流程返回的結果 #! 沒有用到
		# self.previousProcessResult: Any = None

	@staticmethod
	def __getVideoDetails(path: str) -> Any:
		videoCapture = cv2.VideoCapture(path)
		frames: List[np.ndarray] = []
		rval = False
		frame: np.ndarray = np.array([])
		frameCount: int = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
		if videoCapture.isOpened():
			for _ in track(range(0, frameCount), '讀取影片'):
				rval, frame = videoCapture.read()
				frames.append(frame)

		# while rval:
		# 	frames.append(frame)
		# 	rval, frame = videoCapture.read()
		
		width     : float = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
		height    : float = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
		fps       : float = videoCapture.get(cv2.CAP_PROP_FPS)
		videoCapture.release()
		return [frames, width, height, fps, frameCount]

	def forEach(self, callback: ForEachFrameData):
		for i in range(0, self.frameCount):
			callback(self.framesData[i], i, self)

	def runProcess(self, schedule: Callable[[int, int, int], int], process: ForEachFrameData, maxTimes: int = None):
		'''
		if callbackReturnIntex() return a negative number, will stop this process.
		'''
		frameIndex = -1
		# 有限
		if not maxTimes is None:
			for times in track(range(0, maxTimes), "run process"):
				frameIndex = schedule(frameIndex, self.frameCount, times)
				print('Frame Index:', frameIndex)
				if frameIndex < 0:
					print('break')
					break
				isBreak = process(self.framesData[frameIndex], frameIndex, self)
				if isBreak == ProcessState.stop:
					break
		# 無限
		else:
			times = 0
			while True:
				frameIndex = schedule(frameIndex, self.frameCount, times)
				print('Frame Index:', frameIndex)
				if frameIndex < 0:
					print('break')
					break
				isBreak = process(self.framesData[frameIndex], frameIndex, self)
				if isBreak == ProcessState.stop:
					break
				times += 1
		return self

	#!
	def calc(self):
		for typeName in ['vehicles', 'licensePlates', 'trafficLights']:
			self.findCorresponding(typeName, len(self.framesData))

	#! 應該要寫成staticmethod
	def findCorresponding(self, typeName: str, frameIndex: int, threshold: float = 0.1):
		# 跟前一幀比
		frameData2 = self.framesData[frameIndex]
		objs2 = getattr(frameData2, typeName)
		objs1 = []
		if frameIndex != 0:
			frameData1 = self.framesData[frameIndex - 1]
			objs1 = getattr(frameData1, typeName)
		
		for i, obj2 in enumerate(objs2):
			IoUs = []
			for obj1 in objs1:
				IoUs.append(CVModel.IoU(obj2.box, obj1.box))
			
			# 完全沒對應
			if len(IoUs) == 0:
				objs2[i].codename = self.newCodename()
			else:
				# 對應前一幀的 box
				maxIndex = np.argmax(IoUs)
				# 小於閥值 或 對應的沒有 codename，給新 codename
				if IoUs[maxIndex] <= threshold or not hasattr(objs1[maxIndex], 'codename'):
					objs2[i].codename = self.newCodename()
				else:
					objs2[i].codename = objs1[maxIndex].codename

	def newCodename(self) -> int:
		self.lastCodename += 1
		return self.lastCodename
	
	#!
	def getVehicleCorrespondingToTheLicensePlate(self, licensePlateData: LicensePlateData) -> VehicleData:
		...
	
	#!
	def guessPositionOfNextMoment(self, typeName: str) -> List:
		objs = getattr(self.framesData, typeName)
		# if hasattr(objs[i], 'possiblePositionAtTheNextMoment'):
		return []

	def save(self, path: str, fps: float = 30, fourccType: str = 'mp4v'):
		fourcc = cv2.VideoWriter_fourcc(*fourccType)
		out = cv2.VideoWriter(path, fourcc, fps, (int(self.width), int(self.height)))
		for frameData in track(self.framesData, "saving video"):
			out.write(frameData.frame)
		out.release()


class TVideoSchedule:
	
	#檢查返回值有沒有超出frame的範圍，有的話返回-1跳出
	@staticmethod
	def checkIndexOutOfRange(resultIndex: int, frameCount: int) -> int:
		if resultIndex >= frameCount:
			return -1
		return resultIndex

	@staticmethod
	def forEach(previousIndex: int, frameCount: int, times: int) -> int:
		return TVideoSchedule.checkIndexOutOfRange(previousIndex + 1, frameCount)
	
	@staticmethod
	def forEachInterval(interval: int):
		def forEach(previousIndex: int, frameCount: int, times: int) -> int:
			return TVideoSchedule.checkIndexOutOfRange(previousIndex + interval, frameCount)
		return forEach
	
	# 一開始
	# 上一次的index
	# previousIndex = -1
	# 執行了幾次
	# times = 0 每次自動加　1
	# frameCount = 整條影片有多少個frame，恆定
	@staticmethod
	def xxx(previousIndex: int, frameCount: int, times: int) -> int:
		return -1

#!
def detectResultToTFrameDatas(detectResult: DetectResult) -> Union[List[LicensePlateData], List[TrafficLightData]]:
	licensePlates = []
	for i in range(0, detectResult.count):
		licensePlates.append(LicensePlateData(detectResult.image, detectResult.boxes[i], detectResult.confidence[i]))
	return licensePlates

#!
def detectResultsToTVideoData(detectResults: DetectResults) -> List[TFrameData]:
    	...

