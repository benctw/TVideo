from .YoloModel import YoloModel
from .Timeline import Timeline

class TrafficPolice:
	# 實例化模型但不加載
	LPModel = YoloModel()

	def __init__(self):
		self.LPModel
		self.processVideo

	def loadLicensePlateModel(self):
		# 調用 <class YoloModel> 父類型 <class CVModel> 的 load 方法加載模型
		self.LPModel.load()

	def loadImage(self):
		pass

	def loadVideo(self):
		pass

	# 更新裁剪的時間點
	def updataMarkPoint(self, start, end):
		pass

	# 辨識車牌
	def detectLicensePlate(self, video):
		# 返回時間序列
		return Timeline()

	# 辨識車輛
	def detectCar(self):
		# 返回時間序列
		return Timeline()

	# 辨識紅綠燈
	def detectTrafficLight(self):
		# 返回時間序列
		return Timeline()

	def detectTrafficSigns():
		return Timeline()

	# 裁剪影片從 start 到 end
	def cropVideo(self, start, end):
		pass

	# 儲存片段到指定路徑
	def saveVideo(self, video, path):
		pass

	def createReport():
    	# TODO 先定義 report 的格式和内容
		pass
	
	def versionUpdate():
		pass