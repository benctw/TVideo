from typing import List, Set, Tuple, Dict, Any, Union
from enum import IntFlag
import easyocr
import numpy as np
import cv2

from models.CVModel.CVModel import CVModel
from models.TPFrames.TFType import Box, Point2D


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
		self.calc()

	# 生成之後計算的數據
	def calc(self):
		# 載具質心點位置
		self.centerPosition: List = CVModel.getCenterPosition(self.box)
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
		self.calc()
	
	# 生成之後計算的數據
	def calc(self):
		# 車牌質心點位置
		self.centerPosition: List = CVModel.getCenterPosition(self.box)
		# 車牌的四個角點
		self.cornerPoints  : List = self.getCornerPoints(self.image)
		# 矯正的圖像
		self.correctImage  : np.ndarray = self.correct(self.image, self.cornerPoints, int(150 * self.ratioOfLicensePlate), 150)
		# 車牌號碼
		self.number        : str = self.getNumber(self.image)
	
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
	def correct(image: np.ndarray, cornerPoints,  w: int, h: int) -> np.ndarray:
		cornerPoints = np.float32(cornerPoints)
		dst = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
		mat = cv2.getPerspectiveTransform(cornerPoints, dst)
		return cv2.warpPerspective(image.copy(), mat, (w, h))

	@staticmethod
	def getNumber(image: np.ndarray) -> str:
		reader = easyocr.Reader(['en'])
		text: List[Any] = reader.readtext(image, detail = 0)
		number: str = ''.join(text)
		return number.strip(' ').upper()
 
 
# 紅綠燈狀態
class TrafficLightState(IntFlag):
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
	
	def calc(self):
		# 紅綠燈狀態
		self.state: int = self.getState(self.image)
	
	@staticmethod
	def getState(image: np.ndarray) -> TrafficLightState:
		return TrafficLightState.unknow


# 一幀的數據
class TPFrameData:
	def __init__(
		self, 
		frame                 : Union[str, np.ndarray], 
		vehicles              : List[VehicleData], 
		licensePlates         : List[LicensePlateData], 
		trafficLights         : List[TrafficLightData], 
		hasTrafficLight       : bool, 
		hasLicensePlate       : bool, 
		hasMatchTargetLPNumber: bool, 
	):
		# 圖像
		self.frame                  = frame
		# 載具數據
		self.vehicles               = vehicles
		# 車牌數據
		self.licensePlates          = licensePlates
		# 紅綠燈數據
		self.trafficLights          = trafficLights
		# 有沒有紅綠燈
		self.hasTrafficLight        = hasTrafficLight
		# 有沒有車牌
		self.hasLicensePlate        = hasLicensePlate
		# 是否匹配到車牌號碼
		self.hasMatchTargetLPNumber = hasMatchTargetLPNumber


# 一影片的數據
class TPFrames:
	def __init__(
		self, 
		video: Union[str, cv2.VideoCapture], 
		framesData: List[TPFrameData] = []
	):
		self.video = video
		# 多幀數據
		self.framesData = framesData
		# 最後使用的代號
		self.lastCodename: int = 0

	def add(self, frameData: TPFrameData):
		self.framesData.append(frameData)

	def calc(self):
		for typeName in ['vehicles', 'licensePlates', 'trafficLights']:
			self.findCorresponding(typeName, len(self.framesData))

	def findCorresponding(self, typeName: str, frameIndex: int, threshold: float = 0.9):
		# 跟前一幀比
		frameData1, frameData2 = self.framesData[frameIndex - 1 : frameIndex]
		objs1 = getattr(frameData1, typeName)
		objs2 = getattr(frameData2, typeName)
		for i, obj2 in enumerate(objs2):
			IoUs = []
			for obj1 in objs1:
				IoUs.append(CVModel.IoU(obj2.box, obj1.box))
			# 對應前一幀的box
			maxIndex = np.argmax(IoUs)
			# 小於閥值 或 對應的沒有 codename，給新 codename
			if IoUs[maxIndex] < threshold or not hasattr(objs1[maxIndex], 'codename'):
				objs2[maxIndex].codename = self.newCodename()
			else:
				objs2[i].codename = objs1[maxIndex].codename

	def newCodename(self) -> int:
		self.lastCodename += 1
		return self.lastCodename
	
	#!
	def getVehicleCorrespondingToTheLicensePlate(self, licensePlateData: LicensePlateData) -> VehicleData:
		...
	
	#!
	def getPossiblePositionAtTheNextMoment(self, typeName: str) -> List:
		objs = getattr(self.framesData, typeName)
		# if hasattr(objs[i], 'possiblePositionAtTheNextMoment'):
		return []


# vehicleDatas = [VehicleData(), VehicleData(), ...] # 一幀
# frameData = TPFrameData(image , vehicles=vehicleDatas)
# TPFrames().add(frameData)
# TPFrames().done()