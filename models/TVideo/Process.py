from enum import Enum
from os import stat
import re
from typing import NamedTuple
from models.TVideo.TVideo import *
from models.YoloModel.YoloModel import *
from models.CVModel.CVModel import *
from models.TVideo.TVideo import *
from config import *


class Process:

    @staticmethod
    def yolo(frameData: TFrameData, frameIndex: int) -> ProcessState:
        labels = [re.sub(r' ', '', label) for label in LPModel.labels]

        boxes, classIDs, confidences = LPModel.detect(frameData.frame)

        for objIndex, classID in enumerate(classIDs):
            box = boxes[objIndex]
            confidence = confidences[objIndex]
            label = labels[objIndex]
            # 紅綠燈
            if classID == 0:
                # 因為label名稱有空格，不能成為class的屬性名稱
                frameData.addObj(label, TrafficLightData(CVModel.crop(frameData.frame, box), box, confidence))
            # 車牌
            elif classID == 1:
                frameData.addObj(label, LicensePlateData(CVModel.crop(frameData.frame, box), box, confidence))
        
        return ProcessState.next

    @staticmethod
    def drawBoxes(frameData: TFrameData, frameIndex: int) -> ProcessState:
        # self.colors = self.colors if not self.colors is None else self.getAutoSelectColors()
        resultImage = frameData.editedFrame
        # for i in indexs:
            p1x, p1y, p2x, p2y = self.boxes[i]
            color = [int(c) for c in self.colors[self.classIDs[i]]]
            # 框
            cv2.rectangle(resultImage, (p1x, p1y), (p2x, p2y), color, 2)
            # 如果沒有定義函數不繪畫字
            if callbackReturnText is None:
                return resultImage
            # 附帶的字
            text = callbackReturnText(self.classIDs[i], self.boxes[i], self.confidences[i], i)
            # 如果沒有信息不繪畫字
            if text == None:
                return resultImage
            # 字型設定
            font = cv2.FONT_HERSHEY_COMPLEX
            fontScale = 1
            fontThickness = 1
            # 顏色反相
            textColor = [255 - c for c in color]
            # 對比色
            # textColor = [color[1], color[2], color[0]]
            # 獲取字型尺寸
            (textW, textH), _ = cv2.getTextSize(text, font, fontScale, fontThickness)
            # 添加字的背景
            cv2.rectangle(resultImage, (p1x, p1y - textH), (p1x + textW, p1y), color, -1)
            # 添加字
            cv2.putText(resultImage, text, (p1x, p1y), font, fontScale, textColor, fontThickness, cv2.LINE_AA)

        return ProcessState.next
