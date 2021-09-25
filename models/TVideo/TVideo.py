import re
from typing import List, Tuple, Any, Union, Callable
from enum import Enum, auto
import easyocr
import numpy as np
import cv2
from rich.progress import track
import random
import pickle

from models.CVModel.CVModel import CVModel, DetectResult, DetectResults

Point = Tuple[int, int]
Box = Tuple[Point, Point]

class TObj(Enum):
	Undefined = 'Undefined'
	Vehicle = 'Vehicle'
	LicensePlate = 'LicensePlate'
	TrafficLight = 'TrafficLight'
	Lane = 'Lane'


from abc import ABC, abstractmethod
class TObject(ABC):
	def __init__(
		self,
		image: np.ndarray, 
		box  : List,
		confidence: float
	):
		self.image = image
		self.box = box
		self.confidence = confidence
		self.label: TObj = TObj.Undefined

# 一載具的數據
class VehicleData:
	def __init__(
		self, 
		image: np.ndarray, 
		box  : List,
		confidence: float,
		type : str
	):
		self.image = image
		self.box = box
		self.confidence = confidence
		self.type = type
		self.label: TObj = TObj.Vehicle
		# self.calc()

	# 生成之後計算的數據
	def calc(self):
		self.cornerPoints = ...
		# 載具質心點位置
		self.centerPosition: List = []
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
		self.label: TObj = TObj.LicensePlate

		self.number = ''
		# self.calc()
	
	# 生成之後計算的數據
	def calc(self):
		# 車牌的四個角點
		self.cornerPoints  : List = self.getCornerPoints(self.image)
		# 矯正的圖像
		self.correctImage  : np.ndarray = self.correct(self.image, self.cornerPoints, int(150 * self.ratioOfLicensePlate), 150)
		# 車牌質心點位置
		self.centerPosition: List[int] = CVModel.getCenterPosition(self.cornerPoints)
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
			e += 0.0001
			approx = cv2.approxPolyDP(maxCnt, e * l, True)
			if len(approx) == 4:
				# print(f"e = {e}")
				break
			if len(approx) < 4:
				# print(f"e = {e} 擬合不到 4 個點")
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
		
		easyocrResult: List[Any] = reader.readtext(
			image, 
			allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
			batch_size = 5
		)
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
	unknow = 'unknow'
	red    = 'red'
	yellow = 'yellow'
	green  = 'green'

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
		self.label: TObj = TObj.TrafficLight

		self.calc()

	def calc(self):
		# 紅綠燈狀態
		self.state: TrafficLightState = self.ColorDectect(self.image)

	@staticmethod
	def getTrafficLightColor(image: np.ndarray) -> Tuple[TrafficLightState, np.ndarray]:
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

		red = cv2.countNonZero(redBlur)
		yellow = cv2.countNonZero(yellowBlur)
		green  = cv2.countNonZero(greenBlur)

		maxIndex = np.argmax([red, yellow, green])

		if maxIndex == 0: return TrafficLightState.red, redBlur
		elif maxIndex == 1: return TrafficLightState.yellow, yellowBlur
		elif maxIndex == 2: return TrafficLightState.green, greenBlur
		else: return TrafficLightState.unknow, np.zeros((0, 0), np.uint8)

	#將Blur圖片分三等份
	@staticmethod
	def threePartOfTrafficLight(image: np.ndarray) -> List[np.ndarray]:

		h, w = image.shape[:2]
		w /= 3
		partOne = [0, 0, int(w), int(h)]
		partTwo = [int(w), 0, int(2*w), int(h)]
		partThree = [int(2*w), 0, int(3*w), int(h)]

		partOneImg = CVModel.crop(image, partOne)
		partTwoImg = CVModel.crop(image, partTwo)
		partThreeImg = CVModel.crop(image, partThree)

		threePartImgs = [partOneImg, partTwoImg, partThreeImg]
		
		return threePartImgs

	@staticmethod
	def cntsOfeachPart(threePartImgs: List[np.ndarray]) -> TrafficLightState:

		partOneCnts = cv2.countNonZero(threePartImgs[0])
		partTwoCnts = cv2.countNonZero(threePartImgs[1])
		partThreeCnts = cv2.countNonZero(threePartImgs[2])

		cnts = max(partOneCnts, partTwoCnts, partThreeCnts)

		if cnts == partOneCnts : return TrafficLightState.red
		elif cnts == partTwoCnts : return TrafficLightState.yellow
		elif cnts == partThreeCnts : return TrafficLightState.green
		else: return TrafficLightState.unknow

	@staticmethod
	def ColorDectect(image: np.ndarray) -> TrafficLightState:
		
		lightColorState, blur = TrafficLightData.getTrafficLightColor(image)
		threePartImgs = TrafficLightData.threePartOfTrafficLight(blur)
		colorOfPartState = TrafficLightData.cntsOfeachPart(threePartImgs)
		
		if lightColorState == TrafficLightState.red and colorOfPartState == TrafficLightState.red: return TrafficLightState.red
		elif lightColorState == TrafficLightState.yellow and colorOfPartState == TrafficLightState.yellow: return TrafficLightState.yellow
		elif lightColorState == TrafficLightState.green and colorOfPartState == TrafficLightState.green: return TrafficLightState.green
		else: return TrafficLightState.unknow


# 車道線數據
class LaneData:
	def __init__(
		self,
		image: np.ndarray,
		lane: List[List[int]],
		# box  : List,
		confidence: float,
	):
		self.image = image,
		self.lane = lane
		self.confidence = confidence
		self.label: TObj = TObj.Lane
	
	def calc(self):
		self.vanishingPoint = self.getVanishingPoint()
	
	def getVanishingPoint(self):
		...


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
		# 載具數據
		self.vehicles: List[VehicleData] = []
		# 車牌數據
		self.licensePlates: List[LicensePlateData] = []
		# 紅綠燈數據
		self.trafficLights: List[TrafficLightData] = []
		self.allClass: List[List[Any]] = [self.vehicles, self.licensePlates, self.trafficLights]
		self.numOfClass: int = len(self.allClass)
		self.currentTrafficLightState: TrafficLightState = TrafficLightState.unknow
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
	
	def getTargetLicensePlatePosition(self, targetLicensePlateCodename) -> Union[List[int], None]:
		for lp in self.licensePlates:
			if lp.codename == targetLicensePlateCodename and hasattr(lp, 'centerPosition'):
				return lp.centerPosition
		return None
	
class Direct(Enum):
    right = auto()
    left = auto()
    straight = auto()

class ProcessState(Enum):
	next = auto()
	nextLoop = auto()
	stop = auto()

ForEachFrameData = Callable[[TFrameData, int, Any], ProcessState]
indexType = Union[int, List[int]]

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
		self.width     : int = videoDetails[1]
		self.height    : int = videoDetails[2]
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
		# 剛處理完的index
		self.currentIndex: int = -1
		# 目標車牌的codename
		self.targetLicensePlateCodename: int = -1

		# 路徑方向
		self.directs: List[Direct] = []

		# 紅燈加車牌的幀位置
		self.trafficLightStateIsRedFrameIndexs = []

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
		
		width     : int = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
		height    : int = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps       : float = videoCapture.get(cv2.CAP_PROP_FPS)
		videoCapture.release()
		return [frames, width, height, fps, frameCount]

	def forEach(self, callback: ForEachFrameData):
		for i in range(0, self.frameCount):
			callback(self.framesData[i], i, self)

	def runProcess(self, schedule: Callable[[List[indexType], int], indexType], *processes: ForEachFrameData, maxTimes: int = None):
		'''
		if callbackReturnIntex() return a negative number, will stop this process.
		'''
		indexs: List[indexType] = []
		frameIndex: indexType = -1
		# 無限
		if maxTimes is None:
			while True:
				frameIndex = schedule(indexs, self.frameCount)
				indexs.append(frameIndex)

				idxs: List[int] = []
				if type(frameIndex) is int: idxs.append(frameIndex)
				elif type(frameIndex) is list: idxs = frameIndex

				for i in idxs:
					if i < 0:
						return
					for process in processes:
						state = process(self.framesData[i], i, self)
						self.currentIndex = i
						if state == ProcessState.next: pass
						elif state == ProcessState.nextLoop: break
						elif state == ProcessState.stop: return
		# 有限
		else:
			for _ in track(range(0, maxTimes), "run process"):
				frameIndex = schedule(indexs, self.frameCount)
				indexs.append(frameIndex)
				
				idxs: List[int] = []
				if type(frameIndex) is int: idxs.append(frameIndex)
				elif type(frameIndex) is list: idxs = frameIndex
				
				for i in idxs:
					if i < 0:
						return
					for process in processes:
						state = process(self.framesData[i], i, self)
						self.currentIndex = i
						if state == ProcessState.next: pass
						elif state == ProcessState.nextLoop: break
						elif state == ProcessState.stop: return
		return self

	#!
	def calc(self):
		...

	def findCorresponding(self, frameData1: TFrameData, frameData2: TFrameData, threshold: float = 0.1):
		# 跟前一幀比
		for classIndex in range(0, frameData1.numOfClass):
			objs1 = frameData1.allClass[classIndex]
			objs2 = frameData2.allClass[classIndex]
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
	
	def getTargetLicensePlatePath(self) -> List[List[int]]:
		path = []
		for frameData in self.framesData:
			hasTargetLicensePlate = False
			for lp in frameData.licensePlates:
				if lp.codename == self.targetLicensePlateCodename and hasattr(lp, 'centerPosition'):
					path.append(lp.centerPosition)
					hasTargetLicensePlate = True
			if not hasTargetLicensePlate:
				path.append(None)
		return path
	
	def getVaildTargetLicensePlatePath(self) -> List[List[int]]:
		paths = []
		path = []
		#! 把none刪掉，回傳多段路徑
		for frameData in self.framesData:
			hasTargetLicensePlate = False
			for lp in frameData.licensePlates:
				if lp.codename == self.targetLicensePlateCodename and hasattr(lp, 'centerPosition'):
					path.append(lp.centerPosition)
					hasTargetLicensePlate = True
			# if not hasTargetLicensePlate:
			# 	# if len(path) == 0:
			# 		paths.append(path)
			# 		# path.clear()
		paths.append(path)
		return paths

	
	#!
	def getVehicleCorrespondingToTheLicensePlate(self, licensePlateData: LicensePlateData) -> VehicleData:
		...
	
	#!
	def guessPositionOfNextMoment(self, typeName: str) -> List:
		objs = getattr(self.framesData, typeName)
		# if hasattr(objs[i], 'possiblePositionAtTheNextMoment'):
		return []

	#! end = -1?
	def save(self, path: str, start: int = None, end: int = None, fps: float = None, fourccType: str = 'avc1'):
		if fps is None: fps = self.fps
		fourcc = cv2.VideoWriter_fourcc(*fourccType)
		out = cv2.VideoWriter(path, fourcc, fps, (int(self.width), int(self.height)))
		for frameData in track(self.framesData[start: end], "saving video"):
			out.write(frameData.frame)
		out.release()
	
	#!
	def saveData(self, path: str):
		with open(path, 'wb') as f:
			pickle.dump(self, f)
			
	# def loadData(self, path: str):
		



class TVideoSchedule:
    
	#檢查返回值有沒有超出frame的範圍，有的話返回-1跳出
	@staticmethod
	def checkIndexOutOfRange(resultIndex: int, frameCount: int) -> int:
		return -1 if resultIndex >= frameCount else resultIndex
	
	@staticmethod
	def oneTimes(resultIndex: int, indexs: List[indexType]) -> indexType:
		return -1 if len(indexs) > 0 else resultIndex
	
	@staticmethod
	def sample(indexs: List[indexType], frameCount: int) -> int:
		...

	@staticmethod
	def index(i: int, times: int = 1):
		def __index(indexs: List[indexType], frameCount: int) -> indexType:
			if len(indexs) > 0: return -1
			return [i] * times
			# return TVideoSchedule.checkIndexOutOfRange(i, frameCount)
		return __index

	@staticmethod
	def forEach(indexs: List[indexType], frameCount: int) -> indexType:
		if len(indexs) > 0: return -1
		return list(range(0, frameCount))
		# return TVideoSchedule.checkIndexOutOfRange(indexs[-1] + 1, frameCount)
	
	@staticmethod
	def forEachStep(step: int):
		def __forEach(indexs: List[indexType], frameCount: int) -> indexType:
			if len(indexs) > 0: return -1
			return list(range(0, frameCount, step))
			# return TVideoSchedule.checkIndexOutOfRange(indexs[-1] + step, frameCount)
		return __forEach
	
	@staticmethod
	def range(start: int, end: int, step):
		def __range(indexs: List[indexType], frameCount: int) -> indexType:
			return list(range(start, end, step))
		return __range
	
	@staticmethod
	def forEachStepAll(step: int):
		def __forEachStepAll(indexs: List[indexType], frameCount: int) -> indexType:
			if len(indexs) > 0: return -1
			resultIndexs = []
			for i in range(0, step):
				resultIndexs += list(range(i, frameCount, step))
			return resultIndexs
			# return TVideoSchedule.checkIndexOutOfRange(indexs[-1] + step, frameCount)
		return __forEachStepAll

	@staticmethod
	def random(indexs: List[indexType], frameCount: int) -> indexType:
		if len(indexs) > 0: return -1
		li = list(range(0, frameCount))
		random.shuffle(li)
		return li
	
	@staticmethod
	def randomIndex(indexs: List[indexType], frameCount: int) -> indexType:
		if len(indexs) > 0: return -1
		return random.randint(0, frameCount - 1)
	
	@staticmethod
	def forward(start: int, length: int = None, step: int = 1):
		def __forward(indexs: List[indexType], frameCount: int) -> indexType:
			if len(indexs) > 0: return -1
			if not length is None: frameCount = start + length
			return list(range(start, frameCount, step))
		return __forward
	
	@staticmethod
	def backward(start: int, length: int = None, step: int = -1):
		def __backward(indexs: List[indexType], frameCount: int) -> indexType:
			if len(indexs) > 0: return -1
			frameCount = 0 if length is None else start - length
			return list(range(start, frameCount, step))
		return __backward