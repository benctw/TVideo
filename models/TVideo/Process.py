from models.YoloModel.YoloModel import *
from models.CVModel.CVModel import *
from models.TVideo.TVideo import *
from config import *
from rich.progress import track
import time
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
        print('index:', frameIndex)
        return ProcessState.next

    @staticmethod
    def yolo(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        print('--> Yolo Process')
        boxes, classIDs, confidences = LPModel.detect(frameData.frame)

        for objIndex, classID in enumerate(track(classIDs, 'Add data')):
            box = boxes[objIndex]
            confidence = confidences[objIndex]
            # 紅綠燈
            if classID == 0:
                # 因為label名稱有空格，不能成為class的屬性名稱
                frameData.trafficLights.append(TrafficLightData(CVModel.crop(frameData.frame, box), box, confidence))
                # frameData.addObj(label, TrafficLightData(CVModel.crop(frameData.frame, box), box, confidence))
            # 車牌
            elif classID == 1:
                frameData.licensePlates.append(LicensePlateData(CVModel.crop(frameData.frame, box), box, confidence))
                # frameData.addObj(label, LicensePlateData(CVModel.crop(frameData.frame, box), box, confidence))
        
        return ProcessState.next

    @staticmethod
    def findCorresponding(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        print('--> Find Corresponding License Plate Process')
        if frameIndex == 0:
            frameImage = np.zeros((tvideo.height, tvideo.width, 3), np.uint8)
            frameImage.fill(255)
            tvideo.findCorresponding(TFrameData(frameImage), tvideo.framesData[frameIndex])
        else:
            tvideo.findCorresponding(tvideo.framesData[frameIndex - 1], tvideo.framesData[frameIndex])
        return ProcessState.next

    @staticmethod
    def drawBoxes(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        # print('--> Draw Boxes License Plate')
        colors = getColors(tvideo.lastCodename)

        resultImage = frameData.frame
        for classIndex in range(0, frameData.numOfClass):
            for obj in frameData.allClass[classIndex]:
                color = colors[obj.codename]
                p1x, p1y, p2x, p2y = obj.box
                # 框
                cv2.rectangle(resultImage, (p1x, p1y), (p2x, p2y), color, 2)
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
    def findNumber(number: str):
        def __findNumber(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
            for lp in frameData.licensePlates:
                if lp.number == number:
                    print(f'找到對應車牌號碼: {number}')
                    print(f'在 {frameIndex} 幀, {frameIndex / tvideo.fps} 秒')
                    return ProcessState.stop
            return ProcessState.next
        return __findNumber
    
    @staticmethod
    def test(frameData: TFrameData, frameIndex: int, tvideo: TVideo) -> ProcessState:
        print('sleep 5s')
        time.sleep(5)
        return ProcessState.next