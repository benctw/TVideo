from models.CVModel.CVModel import CVModel
from .TimelineError import TimelineErrors

class TPFrameData:
	def __init__(
		self, 
		# 影像: cv2.Mat
		frame, 
		# 載具數據: [VehicleData]
		vehicles,
		# 車牌數據: [LPData]
		LPs,
		# 有沒有紅綠燈: bool
		hasTrafficLight, 
		# 紅綠燈狀態: ''
		TrafficLightState, 
		# 有沒有車牌: bool
		hasLicensePlate, 
		# 是否匹配到車牌號碼: bool
		hasMatchTargetLPNumber,
	):
		self.frame = frame
		self.vehicles = vehicles
		self.LPs = LPs
		self.hasTrafficLight = hasTrafficLight
		self.TrafficLightState = TrafficLightState
		self.hasLicensePlate = hasLicensePlate
		self.hasMatchTargetLPNumber = hasMatchTargetLPNumber
		
class LPData:
	def __init__(
		self, 
		# 車牌圖像
		image,
		# 車牌號碼: str
		number = '', 
		# 框: [p1x, p1y, p2x, p2y]
		box = [],
		# 車牌的四個角點: 

	):
		self.image = image
		self.number = number
		self.box = box
	
	# 生成之後計算的數據
	def calc(self):
		# 車牌質心點位置 : list
		self.centerPosition = CVModel.getCenterPosition(self.box)

# 載具的數據
class VehicleData:
	def __init__(
		self, 
		# 框: [p1x, p1y, p2x, p2y]
		box = [],
		# 代號: int , 0 是不知道
		codename = 0,
		# 方向: list
		direction = [],
		# 下一時刻可能的位置: list
		possiblePositionAtTheNextMoment = [],
		# 對應的車牌
		
	):
		self.codename = codename
		self.box = box
		self.direction = direction
		self.possiblePositionAtTheNextMoment = possiblePositionAtTheNextMoment

	# 生成之後計算的數據
	def calc(self):
		# 載具質心點位置: list
		self.centerPosition = CVModel.getCenterPosition(self.box)


class TPFrames:
	def __init__(self, video, framesData = []):
		self.video = video
		self.framesData = framesData

	def add(self, frameData):
		self.framesData.append(frameData)
		



vehicleDatas = [VehicleData(), VehicleData(), ...] # 一幀
frameData = TPFrameData(image , vehicles=vehicleDatas)

TPFrames().add(frameData)
TPFrames().done()