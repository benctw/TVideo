from models.LaneModel.LaneModel import LaneModel
from models.YoloModel.YoloModel import *
import os

__dirname = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

LPModelFolderPath = __dirname + '/static/model/lp'

LPModel = YoloModel(
    namesPath   = LPModelFolderPath + '/lp.names',
    configPath  = LPModelFolderPath + '/lp_yolov4.cfg',
    weightsPath = LPModelFolderPath + '/lp_yolov4_final.weights',
    inputWidth  = 608,
    inputHeight = 608
)

laneModelFolderPath = __dirname + '/static/model/lane'

laneModel = LaneModel(
    modelPath   = laneModelFolderPath + '/model.h5',
    inputWidth  = 80,
    inputHeight = 160
)

outputDir = __dirname + '/store/output'

# coco = YoloModel(
#     namesPath   = "D:/chiziSave/yolov3coco/coco.names",
#     configPath  = "D:/chiziSave/yolov3coco/yolov3.cfg",
#     weightsPath = "D:/chiziSave/yolov3coco/yolov3.weights",
#     inputWidth  = 416,
#     inputHeight = 416
# )

saveRecordPath = __dirname + '/store/records'
outputNameFormat = 'result-video_Record_'