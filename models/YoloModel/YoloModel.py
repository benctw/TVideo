import numpy as np
import cv2
from .YoloModelError import YoloModelErrors
from ..CVModel.CVModel import CVModel

# 應該做成抽象對象，被繼承
class YoloModel(CVModel):

	def __init__(self, namesPath, configPath, weightsPath, threshold = 0.2, confidence = 0.2, minConfidence = 0.2):
		print("[INFO] Loading YOLO Model...")
		self.namesPath = namesPath
		self.configPath = configPath
		self.weightsPath = weightsPath
		self.threshold = threshold
		self.confidence = confidence
		self.minConfidence = minConfidence
		## TODO error handling
		self.labels = open(self.namesPath).read().strip().split('\n')
		self.colors = np.random.randint(0, 255, size = (len(self.labels), 3), dtype = "uint8")
		self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
		self.outputLayerNames = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
		print("[INFO] Loaded YOLO Model!")

	# (cx, cy, w, h) -> (p1x, p1y, p2x, p2y)
	@staticmethod
	def yoloFormatToTwoPoint(centerX, centerY, width, height):
		# 計算角點坐標
		p1x = int(centerX - (width / 2))
		p1y = int(centerY - (height / 2))
		p2x = p1x + width
		p2y = p1y + height
		return [p1x, p1y, p2x, p2y]

	def detectImage(self, image):
		result = self.DetectResult()
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
					(centerX, centerY, width, height) = box.astype("int")
					result.add(classID, self.yoloFormatToTwoPoint(centerX, centerY, width, height), float(confidence))
		return result

	def drawBoxes(self, image, detectResult):
		newImage = image.copy()
		idxs = cv2.dnn.NMSBoxes(detectResult.boxes, detectResult.confidences, self.confidence, self.threshold)
		if len(idxs) > 0:
			for i in idxs.flatten():
				p1x, p1y, p2x, p2y = detectResult.boxes[i]
				color = self.colors[detectResult.classIDs[i]]
				cv2.rectangle(newImage, (p1x, p1y), (p2x, p2y), color, 2)
				text = "{}: {:.4f}".format(self.labels[detectResult.classIDs[i]], detectResult.confidences[i])
				cv2.putText(newImage, text, (p1x, p1y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		return newImage