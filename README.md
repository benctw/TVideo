# Traffic-police

```
TrafficPolice
├─ CVModel
│	├─ CVModel.py
│	├─ CVModelError.py
│	└─ __init__.py
├─ README.md
├─ RESAModel
│	├─ RESAModel.py
│	└─ __init__.py
├─ Timeline
│	├─ Timeline.py
│	├─ TimelineError.py
│	└─ __init__.py
├─ View
│	├─ Application.py
│	├─ __init__.py
│	├─ defaultSetting.json
│	├─ lang
│	│	├─ standard.json
│	│	└─ zh-TW.json
│	├─ settings.json
│	└─ src
├─ YoloModel
│	├─ YoloModel.py
│	├─ YoloModelError.py
│	├─ __init__.py
│	└─ src
├─ __init__.py
├─ helper.py
└─ master.py
```

***

### class CVModel

用於輸入為影像的模型的抽象對象

#### @staticmethod　getImagesFromVideo(videoCapture: cv.VideoCapture) -> [cv.Mat]

獲取 `videoCapture` 中每一幀的影像

#### detectImage(image: cv.Mat) -> DetectResult

必須定義 `detectImage` 方法

#### detectVideo(self, videoCapture: cv.VideoCapture, interval: int) -> [DetectResult]

根據 `interval` 的間隔採樣辨識 `videoCapture` 中的幀

***

### class DetectResult

#### \_\_init\_\_(classIDs = [], boxes = [], confidences = [])

`classIDs` : 辨識到的多個分類

`boxes` : 多個分類的邊框位置

`confidences` : 多個分類的可信程度

`boxes` 中每個位置應該儲存的的資料：

該分類在圖上的 `[p1x, p1y, p2x, p2y]` ，分別為左上角 `p1x, p1y` 和右下角　`p2x, p2y` 點的位置

三個參數中同 `index` 的值為同一個結果

#### add(self, classID: , box: list, confidence: float) -> self

用於模型的辨識結果

#### hasResult(self) -> bool

是否有至少一個結果

#### crop(self, image: cv.Mat, boxIndex: int = 0) -> cv.Mat

以 `boxes` 中的第 `boxIndex` 個裁剪 `image` 圖片

#### display(self) -> void

打印出列表

***

### helper

### class dotdict(dict)

#### @staticmethod deep(d: dict) -> dict

深度讓 `dict` 可以以 `.` 取值賦值

***

### class Timeline

#### \_\_init\_\_(self)

初始化

#### stamp(self, time: int) -> self

添加一次時間戳

***

### class YoloModel(CVModel)

#### \_\_init\_\_(self, namesPath, configPath, weightsPath)

`namesPath` : .names 文件的路徑

`configPath` : .cfg 文件的路徑

`weightsPath` : .weights 文件的路徑

預設：

`self.threshold = 0.2`

`self.confidence = 0.5`

`self.minConfidence = 0.2`

#### @staticmethod yoloFormatToTwoPoint(centerX: int, centerY: int, width: int, height: int) -> [int, int, int, int]

把中心點坐標和長寬轉換成左上角和右下角點的坐標

`p1` : 左上角點

`p2` : 右下角點

返回的 `list` 對應 `[p1x, p1y, p2x, p2y]`

#### @staticmethod correct(image: cv.Mat) -> cv.Mat

矯正圖像中的矩形成正面

#### @staticmethod getLPNumber(LPImage: cv.Mat) -> str

從圖像中辨識車牌號碼

#### load(self) -> void

加載模型

> 加載 `.names` 文件中的 `label`　和定義各 `label` 的顔色

#### detectImage(self, image: cv.Mat) -> DetectResult

使用模型辨識圖像

#### drawBoxes(self, image: cv.Mat, detectResult: DetectResult) -> cv.Mat

在圖像上畫上辨識框，返回新的圖像

#### compareLPNumber(self, detectLPNumber: str) -> float

比較號碼的相似度，返回大於 0 到 1 的值

當返回 1 時代表號碼完全一樣

***

### class TrafficPolice
整體的處理

#### drivingDirection(p1: (int, int), p2: (int, int)) -> float 

判斷車輛行駛方向

返回從 `p1` 到 `p2` 位置的單位向量

### 待續 . . .
