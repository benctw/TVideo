from models.YoloModel.YoloModel import *
from models.CVModel.CVModel import *
from models.TVideo.TVideo import *
from models.helper import *
from config import *
from ..communicate import *
from rich.progress import track
import time
import math
import numpy as np
import cv2

#!
colors = None
def getColors(lastCodename):
    global colors
    if colors is None:
        colors = np.random.randint(0, 255, size = (lastCodename + 1, 3), dtype = "uint8")
        colors = [[int(c) for c in color] for color in colors]
    return colors

class Process:
    @staticmethod
    def showIndex(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        INFO(f'index: {frameIndex}')
        return ProcessState.next

    @staticmethod
    def yolo(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        boxes, classIDs, confidences = LPModel.detect(frameData.frame)

        for objIndex, classID in enumerate(classIDs):
            box = boxes[objIndex]
            confidence = confidences[objIndex]
            # 紅綠燈
            if classID == 0:
                frameData.trafficLights.append(TrafficLightData(CVModel.crop(frameData.frame, box), box, confidence))
            # 車牌
            elif classID == 1:
                frameData.licensePlates.append(LicensePlateData(CVModel.crop(frameData.frame, box), box, confidence))
        
        return ProcessState.next
    
    @staticmethod
    def calcLicensePlateData(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        for lp in frameData.licensePlates:
            lp.calc()
        return ProcessState.next

    @staticmethod
    def findCorresponding(reverse: bool = False):
        def __findCorresponding(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
            # 定義邊界和方向
            edge = tvideo.frameCount if reverse else 0
            direction = 1 if reverse else -1

            if frameIndex == edge:
                frameImage = np.zeros((tvideo.height, tvideo.width, 3), np.uint8)
                frameImage.fill(255)
                tvideo.findCorresponding(TFrameData(frameImage), frameData)
            else:
                tvideo.findCorresponding(tvideo.framesData[frameIndex + direction], frameData)
            return ProcessState.next
        return __findCorresponding
    
    @staticmethod
    def hasCorrespondingTargetLicensePlate(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        for lp in frameData.licensePlates:
            if lp.codename == tvideo.targetLicensePlateCodename:
                return ProcessState.next
        return ProcessState.stop

    @staticmethod
    def drawBoxes(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        colors = getColors(tvideo.lastCodename)

        resultImage = frameData.frame
        for classIndex in range(0, frameData.numOfClass):
            for obj in frameData.allClass[classIndex]:
                #!
                if obj.label is TObj.LicensePlate and obj.codename != tvideo.targetLicensePlateCodename: continue
                
                color = colors[obj.codename]
                p1x, p1y, p2x, p2y = obj.box
                # 框
                cv2.rectangle(resultImage, (p1x, p1y), (p2x, p2y), color, 2)

                if obj.label is TObj.LicensePlate:
                    text = f'{obj.codename} [{obj.number}]'
                elif obj.label is TObj.TrafficLight:
                    text = f'{obj.codename} {obj.state}'
                else:
                    text = '{} {}({:.0f}%)'.format(obj.codename, obj.label, obj.confidence * 100)

                # 字型設定
                font = cv2.FONT_HERSHEY_COMPLEX
                fontScale = 1
                fontThickness = 1
                # 顏色反相
                textColor = [255 - c for c in color]
                # 獲取字型尺寸
                (textW, textH), _ = cv2.getTextSize(text, font, fontScale, fontThickness)
                # 添加字的背景
                cv2.rectangle(resultImage, (p1x, p1y - textH), (p1x + textW, p1y), color, -1)
                # 添加字
                cv2.putText(resultImage, text, (p1x, p1y), font, fontScale, textColor, fontThickness, cv2.LINE_AA)

        return ProcessState.next

    @staticmethod
    def drawPath(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        colors = getColors(tvideo.lastCodename)
        windowStartIndex = frameIndex - 30 if frameIndex - 30 >= 0 else 0
        framesData: List[TFrameData] = tvideo.framesData[windowStartIndex: frameIndex + 1]
        for windowIndex, fd in enumerate(framesData):
            for lp in fd.licensePlates:
                cv2.circle(frameData.frame, (lp.box[0] + lp.centerPosition[0], lp.box[1] + lp.centerPosition[1]), int((windowIndex + 1) / 5), colors[lp.codename], -1)
        return ProcessState.next
    
    @staticmethod
    def findTargetNumber(number: str):
        def __findNumber(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
            for lp in frameData.licensePlates:
                if lp.number == number:
                    INFO(f'found')
                    INFO(f'在 {frameIndex} 幀, {frameIndex / tvideo.fps} 秒')
                    tvideo.targetLicensePlateCodename = lp.codename
                    return ProcessState.stop
            return ProcessState.next
        return __findNumber
    
    # 判斷當前的紅綠燈
    @staticmethod
    def correspondingTrafficLights(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        tlsArea = []
        for tl in frameData.trafficLights:
            if tl.state is TrafficLightState.unknow: tlsArea.append(0)
            else: tlsArea.append(CVModel.boxArea(tl.box))
        if len(tlsArea) != 0:
            maxAreaIndex = np.argmax(tlsArea)
            frameData.currentTrafficLightState = frameData.trafficLights[maxAreaIndex].state
        return ProcessState.next
    
    @staticmethod
    def drawCurrentTrafficLightState(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        resultImage = frameData.frame
        text = frameData.currentTrafficLightState.value
        # 字型設定
        font = cv2.FONT_HERSHEY_COMPLEX
        fontScale = 1
        fontThickness = 1
        
        color = (0, 0, 0)
        if frameData.currentTrafficLightState is TrafficLightState.red: textColor = (0, 0, 255)
        elif frameData.currentTrafficLightState is TrafficLightState.yellow: textColor = (0, 255, 255)
        elif frameData.currentTrafficLightState is TrafficLightState.green: textColor = (0, 255, 0)
        else: textColor = (255, 255, 255)

        # 獲取字型尺寸
        (textW, textH), _ = cv2.getTextSize(text, font, fontScale, fontThickness)
        # 添加字的背景
        cv2.rectangle(resultImage, (0, 0 + textH), (0 + textW, 0), color, -1)
        # 添加字
        cv2.putText(resultImage, text, (0, 0 + textH), font, fontScale, textColor, fontThickness, cv2.LINE_AA)
        return ProcessState.next
    
    @staticmethod
    def cocoDetect(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        boxes, classIDs, confidences = coco.detect(frameData.frame)
        for objIndex, classID in enumerate(classIDs):
            box = boxes[objIndex]
            confidence = confidences[objIndex]
            frameData.vehicles.append(VehicleData(CVModel.crop(frameData.frame, box), box, confidence, coco.labels[classID]))
        return ProcessState.next
    
    @staticmethod
    def updateRangeOfTargetLicensePlate(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        for lp in frameData.licensePlates:
            if lp.codename == tvideo.targetLicensePlateCodename:
                tvideo.start = frameIndex if tvideo.start == 0 else min(tvideo.start, frameIndex)
                tvideo.end = max(tvideo.end, frameIndex)
        return ProcessState.next

    @staticmethod
    def test(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        time.sleep(5)
        return ProcessState.next

    #!
    @staticmethod
    def sift(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:

        if frameIndex - 1 < 0: return ProcessState.next

        queryImage: np.ndarray = np.array([])
        trainImage: np.ndarray = np.array([])

        hasTargetLicensePlate = False
        for lp in tvideo.framesData[frameIndex - 1].licensePlates:
            if lp.codename == tvideo.targetLicensePlateCodename:
                
                hasTargetLicensePlate = True

                #! 根據前面的路徑去推算要offset的量
                queryImage = CVModel.crop(frameData.frame, CVModel.offset(CVModel.expand(lp.box, 10), x=0, y=0))
                queryImage = cv2.cvtColor(queryImage, cv2.COLOR_RGB2GRAY)
                #! 要確認上一幀已經做了矯正
                trainImage = cv2.cvtColor(lp.correctImage, cv2.COLOR_RGB2GRAY)
                break

        if not hasTargetLicensePlate: return ProcessState.next

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6, # 12
                        key_size = 12,     # 20
                        multi_probe_level = 1) #2

        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(queryImage, None)
        kp2, des2 = sift.detectAndCompute(trainImage, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(queryImage,kp1,trainImage,kp2,matches,None,**draw_params)

        h, w = img3.shape[:2]

        frameData.frame[0:h, 0:w] = img3
        
        return ProcessState.next
    
    #!
    @staticmethod
    def calcPathDirection(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:

        def drivingDirection(p1: List[int], p2: List[int]) -> Tuple[float, ...]:
            vector: Tuple[int, ...] = tuple(p1[i] - p2[i] for i in range(0, len(p1)))
            norm = math.sqrt(sum([v ** 2 for v in vector]))
            unitVector = tuple(v / norm if norm != 0 else 0 for v in vector)
            return unitVector
        
        if frameIndex + 1 >= tvideo.frameCount: return ProcessState.next

        position1 = frameData.getTargetLicensePlatePosition(tvideo.targetLicensePlateCodename)
        position2 = tvideo.framesData[frameIndex + 1].getTargetLicensePlatePosition(tvideo.targetLicensePlateCodename)

        if position1 is None or position2 is None: return ProcessState.next

        shift = (tvideo.width - 100, 100)

        unitVector = drivingDirection(position1, position2)
        unitVector = tuple(int(unitVector[i] * 15 + shift[i]) for i in range(0, len(unitVector)))
        frameData.frame = cv2.arrowedLine(frameData.frame, shift, unitVector, (0, 0, 255), 3) 

        return ProcessState.next