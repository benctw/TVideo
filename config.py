from models.YoloModel.YoloModel import *
import os

__dirname = os.path.dirname(os.path.abspath(__file__))

LPModel = YoloModel(
    namesPath   = __dirname + "/static/model/lp.names",
    configPath  = __dirname + "/static/model/lp_yolov4.cfg",
    # weightsPath = __dirname + "/static/model/lp_yolov4_final.weights"
    weightsPath = "D:/chiziSave/TrafficPoliceYoloModel/model/lp_yolov4_final.weights"
)

yoloImageWidth = 608
yoloImageHeight = 608

outputDir = ''