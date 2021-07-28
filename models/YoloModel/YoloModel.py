import numpy as np
import cv2
from .YoloModelError import YoloModelErrors
from ..CVModel.CVModel import CVModel, DetectResult

# 應該做成抽象對象，被繼承
class YoloModel(CVModel):
	def __init__(self, namesPath, configPath, weightsPath, confidence = 0.2, threshold = 0.7, minConfidence = 0.2):
		super(YoloModel, self).__init__()
		print("[INFO] Loading YOLO Model...")
		self.namesPath = namesPath
		self.configPath = configPath
		self.weightsPath = weightsPath
		# 至少要在此信心以上
		self.confidence = confidence
		# 可重疊程度
		self.threshold = threshold
		self.minConfidence = minConfidence
		self.labels = open(self.namesPath).read().strip().split('\n')
		self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
		self.outputLayerNames = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
		print("[INFO] Loaded YOLO Model!")

	# (cx, cy, w, h) -> (p1x, p1y, p2x, p2y)
	@staticmethod
	def yoloFormatToTwoPoint(centerX, centerY, width, height):
		# 計算角點坐標
		p1x = int(centerX - (width / 2))
		p1y = int(centerY - (height / 2))
		p2x = int(p1x + width)
		p2y = int(p1y + height)
		return [p1x, p1y, p2x, p2y]

	def detectImage(self, image):
		result = DetectResult(image, self.labels, self.threshold, self.confidence)
		( H, W ) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB = True, crop = False)
		self.net.setInput(blob)
		layerOutputs = self.net.forward(self.outputLayerNames)
		
		for output in layerOutputs:
			for detection in output:
				# detection 前4個是box中心點坐標(x, y)和寬高(w, h)，按比例，像txt標注檔數據
				scores = detection[5:]
				# 找出得分最大的 index
				classID = np.argmax(scores)
				confidence = scores[classID]
				if confidence > self.minConfidence:
					box = detection[0 : 4] * np.array([W, H, W, H])
					result.add(classID, self.yoloFormatToTwoPoint(*box.astype(int)), float(confidence))
		return result