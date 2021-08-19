from typing import List, Set, Tuple, Dict, Any, Union
from models.CVModel.CVModel import CVModel
import easyocr
import numpy as np
import cv2


# 一載具的數據
class VehicleData:
	def __init__(
		self, 
		# 框
		box: List = [],
		# 方向
		direction: List = [],
		# 下一時刻可能的位置
		possiblePositionAtTheNextMoment: List = [],
		# 對應的車牌
		#!
		
	):
		self.box = box
		# 計算出來的
		self.direction = direction
		self.possiblePositionAtTheNextMoment = possiblePositionAtTheNextMoment

		self.calc()

	# 生成之後計算的數據
	def calc(self):
		# 載具質心點位置: list
		self.centerPosition = CVModel.getCenterPosition(self.box)


# 一車牌的數據
class LicensePlateData:

	# 車牌的長寬比
	ratioOfLicensePlate = 2.375

	def __init__(
		self, 
		# 車牌圖像
		image: np.ndarray, 
		# 框
		box: List = []
	):
		self.image = image
		self.box = box

		self.calc()
	
	# 生成之後計算的數據
	def calc(self):
		# 車牌質心點位置 : list
		self.centerPosition = CVModel.getCenterPosition(self.box)
		# 車牌的四個角點
		self.cornerPoints = self.getCornerPoints(self.image)
		#! 看看要不要把correct搬到這
		self.correctImage = self.correct(self.image, self.cornerPoints, 150 * self.ratioOfLicensePlate, 150)
		# 車牌號碼: str
		self.number = self.getNumber(self.image)
	
	@staticmethod
	def getCornerPoints(image) -> List:
		img = image.copy()
		# img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
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
				return None
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
	def correct(image, cornerPoints: List,  w: int, h: int) -> np.ndarray:
		cornerPoints = np.float32(cornerPoints)
		dst = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
		mat = cv2.getPerspectiveTransform(cornerPoints, dst)
		return cv2.warpPerspective(image.copy(), mat, (w, h))

	#! 看看要不要寫到這裡
	@staticmethod
	def getNumber(image) -> str:
		reader = easyocr.Reader(['en'])
		text = ''.join(reader.readtext(image, detail = 0))
		return text.strip(' ').upper()



class TrafficLightData:

	# 紅綠燈狀態
	unknow = 0
	red    = 1
	yellow = 2
	green  = 3

	def __init__(
		self,
		# 紅綠燈圖像
		image,
	):
		self.image = image
	
	def calc(self):
		# 紅綠燈狀態: int
		self.state = self.getState(self.image)
	
	def getState(image) -> int:
		pass


# 一幀的數據
class TPFrameData:
	def __init__(
		self, 
		# 圖像: cv2.Mat / path
		frame: np.ndarray, 
		# 載具數據
		vehicles: List[VehicleData], 
		# 車牌數據
		licensePlates: List[LicensePlateData], 
		# 紅綠燈數據
		trafficLights: List[TrafficLightData], 
		# 有沒有紅綠燈
		hasTrafficLight: bool, 
		# 有沒有車牌
		hasLicensePlate: bool, 
		# 是否匹配到車牌號碼
		hasMatchTargetLPNumber: bool, 
	):
		self.frame = frame
		self.vehicles = vehicles
		self.licensePlates = licensePlates
		self.trafficLights = trafficLights
		self.hasTrafficLight = hasTrafficLight
		self.hasLicensePlate = hasLicensePlate
		self.hasMatchTargetLPNumber = hasMatchTargetLPNumber


# 一影片的數據
class TPFrames:
	def __init__(
		self, video: Union[str, cv2.VideoCapture], 
		framesData: List[TPFrameData] = []
	):
		# 影像: cv2.VideoCapture / path
		self.video = video
		# 多幀數據
		self.framesData = framesData
		self.lastCodename: int = 0

	# frameData: TPFrameData
	def add(self, frameData: TPFrameData):
		self.framesData.append(frameData)

	def calc(self):
		pass

	#!
	def findCorresponding(self, type, frameIndex, ):
		# 跟前一幀比
		frameData1, frameData2 = self.framesData[frameIndex - 1 : frameIndex]
		for licensePlateData2 in frameData2.licensePlates:
			IoUs = []
			for licensePlateData1 in frameData1.licensePlates:
				IoUs.append(CVModel.IoU(licensePlateData2.box, licensePlateData1.box))
			# 對應前一幀的box
			maxIndex = np.argmax(IoUs)
			threshold = 0.9
			# 小於閥值 或 對應的沒有 codename，給新 codename
			if IoUs[maxIndex] < threshold or not hasattr(frameData1.licensePlates[maxIndex], 'codename'):
				licensePlateData2.licensePlates[maxIndex].codename = self.newCodename()
				continue
			else:
				licensePlateData2.codename = frameData1.licensePlates[maxIndex].codename

	def newCodename(self) -> int:
		self.lastCodename += 1
		return self.lastCodename
	
	def getPossiblePositionAtTheNextMoment(self) -> List:
		if hasattr(..., 'possiblePositionAtTheNextMoment'):
			pass
		pass



# vehicleDatas = [VehicleData(), VehicleData(), ...] # 一幀
# frameData = TPFrameData(image , vehicles=vehicleDatas)
# TPFrames().add(frameData)
# TPFrames().done()