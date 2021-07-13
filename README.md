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

### CVModel

```python
class CVModel()
```

用於輸入為影像的模型的抽象對象

必須重新定義：

```python
def detectImage(image)
```
