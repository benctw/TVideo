from typing import List, Tuple
import numpy as np
import cv2
from ..CVModel.CVModel import CVModel, DetectResult
# import config as cfg
from ..communicate import *

# 應該做成抽象對象，被繼承
class YoloModel(CVModel):
	def __init__(
		self, 
		namesPath    : str, 
		configPath   : str, 
		weightsPath  : str, 
		inputWidth   : int,
		inputHeight  : int,
		confidence   : float = 0.2, 
		threshold    : float = 0.7, 
		minConfidence: float = 0.2
	):
		super(YoloModel, self).__init__()
		INFO("Loading YOLO Model")
		self.namesPath     = namesPath
		self.configPath    = configPath
		self.weightsPath   = weightsPath
		self.inputWidth    = inputWidth
		self.inputHeight   = inputHeight
		# 至少要在此信心以上
		self.confidence    = confidence
		# 可重疊程度
		self.threshold     = threshold
		self.minConfidence = minConfidence

		self.labels: List[str] = open(self.namesPath).read().strip().split('\n')
		self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
		self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		self.outputLayerNames = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
		INFO("Loaded YOLO Model!")

	# (cx, cy, w, h) -> (p1x, p1y, p2x, p2y)
	@staticmethod
	def yoloFormatToTwoPoint(centerX: int, centerY: int, width: int, height: int) -> List[int]:
		# 計算角點坐標
		p1x = int(centerX - (width / 2))
		p1y = int(centerY - (height / 2))
		p2x = int(p1x + width)
		p2y = int(p1y + height)
		return [p1x, p1y, p2x, p2y]
	
	def detect(self, image: np.ndarray) -> Tuple[List[List[int]], List[int], List[float]]:
		H, W = image.shape[:2]
		blob = cv2.dnn.blobFromImage(image, 1 / 255, (self.inputWidth, self.inputHeight), swapRB = True, crop = False)
		self.net.setInput(blob)
		layerOutputs = self.net.forward(self.outputLayerNames)

		boxes      : List[List[int]] = []
		classIDs   : List[int] = []
		confidences: List[float] = []
		NMSIndexs  : List[int] = []

		for output in layerOutputs:
			for detection in output:
				# detection 前4個是box中心點坐標(x, y)和寬高(w, h)，按比例，像txt標注檔數據
				scores = detection[5:]
				# 找出得分最大的 index
				classID = np.argmax(scores)
				confidence = scores[classID]
				if confidence > self.minConfidence:
					box = detection[0:4] * np.array([W, H, W, H])
					boxes.append(self.yoloFormatToTwoPoint(*box.astype(int)))
					classIDs.append(int(classID))
					confidences.append(float(confidence))

		idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
		if len(idxs) > 0:
			NMSIndexs = idxs.flatten()
			boxes = [boxes[i] for i in NMSIndexs]
			classIDs = [classIDs[i] for i in NMSIndexs]
			confidences = [confidences[i] for i in NMSIndexs]

		# 不知道為什麼 box 的點會負數，消掉負數
		boxes = [[max(0, b) for b in box] for box in boxes]

		return boxes, classIDs, confidences

	#!
	def detectImage(self, image: np.ndarray) -> DetectResult:
		if type(image) is str:
			image = cv2.imread(image)
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
		result.calcNMS()
		return result
	
	def showConfig(self):
		...