from models.YoloModel.YoloModel import *
import os

__dirname = os.path.dirname(os.path.abspath(__file__))

LPModel = YoloModel(
    namesPath   = __dirname + "/static/model/lp.names",
    configPath  = __dirname + "/static/model/lp_yolov4.cfg",
    # weightsPath = __dirname + "/static/model/lp_yolov4_final.weights"
    weightsPath = "C:/Users/zT3Tz/Documents/TrafficPoliceYoloModel/model/lp_yolov4_final.weights",
    inputWidth  = 608,
    inputHeight = 608
)


outputDir = __dirname + '/store/output/'

# coco = YoloModel(
#     namesPath   = "D:/chiziSave/yolov3coco/coco.names",
#     configPath  = "D:/chiziSave/yolov3coco/yolov3.cfg",
#     weightsPath = "D:/chiziSave/yolov3coco/yolov3.weights",
#     inputWidth  = 416,
#     inputHeight = 416
# )

saveRecordPath = __dirname + '/store/records'