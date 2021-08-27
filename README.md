# TrafficPolice（未更新）

```
TrafficPolice
├─ README.md
├─ config.py
├─ controllers
│    └─ setting.py
├─ main.py
├─ models
│    ├─ CVModel
│    │    ├─ CVModel.py
│    │    └─ __init__.py
│    ├─ RESAModel
│    │    ├─ RESAModel.py
│    │    └─ __init__.py
│    ├─ TVideo
│    │    ├─ Process.py
│    │    ├─ TFType.py
│    │    ├─ TVideo.py
│    │    └─ __init__.py
│    ├─ YoloModel
│    │    ├─ YoloModel.py
│    │    └─ __init__.py
│    ├─ __init__.py
│    └─ helper.py
├─ requirements.txt
├─ static
│    ├─ ico
│    ├─ img
│    ├─ model
│    │    ├─ lp.names
│    │    └─ lp_yolov4.cfg
│    └─ preset
│           └─ defaultSetting.json
├─ store
│    ├─ output
│    │    └─ 1.jpg
│    └─ settings.json
└─ views
       └─ lang
              ├─ standard.json
              └─ zh-TW.json
```

***

### class CVModel

用於輸入為影像的模型的抽象對象

#### @staticmethod getImagesFromVideo(videoCapture: cv.VideoCapture) -> [cv.Mat]

獲取 `videoCapture` 中每一幀的影像

#### @abstractmethod detectImage(self, image: cv.Mat) -> DetectResult

必須重新定義該方法

定義：利用模型辨識傳入之圖像並回傳 `DetectResult` 的類型

#### detectVideo(self, videoCapture: cv.VideoCapture, interval: int) -> [DetectResult]

根據 `interval` 的間隔採樣辨識 `videoCapture` 中的幀

***

### class DetectResult

#### \_\_init\_\_(self, classIDs = [], boxes = [], confidences = [])

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

#### \_\_init\_\_(self, namesPath: str, configPath: str, weightsPath: str, threshold: float = 0.2, confidence: float = 0.5, minConfidence: float = 0.2) -> void

加載模型

`namesPath` : .names 文件的路徑

`configPath` : .cfg 文件的路徑

`weightsPath` : .weights 文件的路徑

預設：

`threshold` : 閥值

`confidence` : 信心

`minConfidence` : 最小信心

> 加載 `.names` 文件中的 `label`　和定義各 `label` 的顔色

#### @staticmethod yoloFormatToTwoPoint(centerX: int, centerY: int, width: int, height: int) -> [int, int, int, int]

把中心點坐標和長寬轉換成左上角和右下角點的坐標

`p1` : 左上角點

`p2` : 右下角點

返回的 `list` 對應 `[p1x, p1y, p2x, p2y]`

#### detectImage(self, image: cv.Mat) -> DetectResult

使用模型辨識圖像

#### drawBoxes(self, image: cv.Mat, detectResult: DetectResult) -> cv.Mat

在圖像上畫上辨識框，返回新的圖像

***

### class TrafficPolice
整體的處理

#### drivingDirection(p1: (int, int), p2: (int, int)) -> float 

判斷車輛行駛方向

返回從 `p1` 到 `p2` 位置的單位向量

#### @staticmethod getLPNumber(LPImage: cv.Mat) -> str

從圖像中辨識車牌號碼

#### @staticmethod correct(image: cv.Mat) -> cv.Mat

矯正圖像中的矩形成正面

#### compareLPNumber(self, detectLPNumber: str) -> float

比較號碼的相似度，返回大於 0 到 1 的值

當返回 1 時代表號碼完全一樣

### 待續 . . .


``` python
class TObject:
```

`image` : `np.ndarray`
`box` : `List`
`confidence` : `float`

`label` = `'Undefined'`

```python
class VehicleData
```

`centerPosition`

---

```python
class LicensePlateData
```

`cornerPoints`
`correctImage`
`centerPosition`
`number`

`@staticmethod`

```python
def getCornerPoints(image: np.ndarray) -> List
```

```python
def correct(image: np.ndarray, cornerPoints, w: int, h: int) -> np.ndarray
```

```python
def getNumber(image: np.ndarray) -> str
```

---

```python
class TrafficLightData
```

`state`

`@staticmethod`

```python
def getTrafficLightColor(image: np.ndarray) -> List[int]
```

```python
def ColorDectect(image: np.ndarray, red: int, yellow: int, green: int) -> TrafficLightState
```

---

```python
class TFrameData
```

```frame : np.ndarray, ```

`vehicles: List[VehicleData] = []`

載具數據

`licensePlates: List[LicensePlateData] = []`

車牌數據

`trafficLights: List[TrafficLightData] = []`

紅綠燈數據

---

```python
class TVideo
```

`frames`
`width`
`height`
`fps`
`frameCount`
`framesData`
`lastCodename`
`start`
`end`

```python
def forEach(self, callback: ForEachFrameData)
```

```python
def runProcess(self, schedule: Callable[[int, int, int], int], process: ForEachFrameData, maxTimes: int = None)
```

```python
def findCorresponding(self, typeName: str, frameIndex: int, threshold: float = 0.1)
```

```python
def newCodename(self) -> int
```

```python
def save(self, path: str, fps: float = 30, fourccType: str = 'mp4v')
```
