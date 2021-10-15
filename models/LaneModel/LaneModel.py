from typing import List
from ..CVModel.CVModel import CVModel
from PIL import Image
import numpy as np
import cv2
from tensorflow import keras

class LaneModel():
    def __init__(
        self,
        modelPath   : str,
        inputWidth  : int,
        inputHeight : int
    ):
        self.modelPath   : str  = modelPath
        self.inputWidth  : int  = inputWidth
        self.inputHeight : int  = inputHeight

        self.model = keras.models.load_model(modelPath)
    
    def resizeImage(self, image: np.ndarray) -> np.ndarray:
        return np.array(Image.fromarray(image).resize(size=(self.inputHeight, self.inputWidth), resample=Image.BICUBIC))
        
    def detect(self, image : np.ndarray) -> np.ndarray:
        resizedImage = self.resizeImage(image.copy())

        recent_fit  : List = []
        avg_fit     : List = []

        h, w = image.shape[:2]
        image = image.astype(np.uint8)

        resizedImage = resizedImage[None,:,:,:]
        model_result = self.model.predict(resizedImage)
        prediction = model_result[0] * 255
        recent_fit.append(prediction)

        if len(recent_fit) > 5:
            recent_fit = recent_fit[1:]

        avg_fit = np.mean(np.array([i for i in recent_fit]), axis = 0)
        _, thresh = cv2.threshold(avg_fit, 200, 255, cv2.THRESH_BINARY)

        lane_drawn = cv2.resize(thresh, (w, h)).astype(np.uint8) 

        # blanks = np.zeros_like(lane_drawn).astype(np.uint8)
        lane_drawn = np.dstack((lane_drawn, lane_drawn, lane_drawn))
        # cv2.imshow('result', lane_drawn)
        # cv2.waitKey(0)
        # result = cv2.add(image, lane_drawn)

        return np.array(lane_drawn)