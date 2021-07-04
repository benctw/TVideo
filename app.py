
class App:
	# 實例化模型但不加載
	LPModel = LicensePlateModel()

	def __init__(self):
		self.LPModel
		self.processVideo

	def loadLicensePlateModel(self):
		# 調用 <class LPModel> 的 load 方法加載模型
		LPModel.load()

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
		return <list>

	# 辨識車輛
	def detectCar(self):
		# 返回時間序列
		return <list>

	# 辨識紅綠燈
	def detectTrafficLight(self):
		# 返回時間序列
		return <list>

    def detectTrafficSigns():
        return <list>

	# 裁剪影片從 start 到 end
	def cropVideo(self, start, end):
		pass

	# 根據interval的間隔遍歷一遍影片的幀
	def forVideo(self, interval):
		pass

	# 儲存片段到指定路徑
	def saveVideo(self, video, path):
		pass


class Timeline:

    def __init__(self):
        self.timestamps = []

    def timestamp(self, time):
        if not isinstance(time, <time>):
            raise TimelineErrors.ArgumentTypeError(time)
        self.timestamps.append(time)
