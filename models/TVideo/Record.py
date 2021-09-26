from enum import Enum
from typing import Any, Dict, List, Union
import numpy as np

from models.TVideo.TVideo import TVideo, VehicleData, LicensePlateData, TrafficLightData, LaneData
import json

class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        if   isinstance(obj, np.integer) : return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray) : return obj.tolist()
        elif isinstance(obj, Enum)       : return obj.name
        else                             : return super(MyEncoder, self).default(obj)

class Record:

    savePath: str = 'C:/Users/zT3Tz/Documents/TrafficPolice/store/records'

    def __init__(
        self, 
        # tvideo                   : TVideo,
        # vehicles                 : List[VehicleData],
        # licensePlates            : List[LicensePlateData],
        # trafficLights            : List[TrafficLightData],
        # lanes                    : List[LaneData],
        # currentTrafficLightState : TrafficLightState

    ) -> None:
        self.lastRecordId: int = self.getLastRecordId()
    
    @staticmethod
    def getLastRecordId() -> int:
        id = 0
        try: 
            with open(f'{Record.savePath}/lastRecordId.txt', 'r') as f:
                id = f.read()
        except:
            print(f'無法讀取{Record.savePath}/lastRecordId.txt')
        print('id:', id)
        return int(id)

    @staticmethod
    def saveLastRecordId(id: Union[int, str]):
        with open(f'{Record.savePath}/lastRecordId.txt', 'w') as f:
            f.write(str(id))

    def getNextFileName(self) -> str:
        return f'Record_{self.lastRecordId + 1}'

    @staticmethod
    def __attrDefault(obj, attr: str, value) -> Any:
        value = getattr(obj, attr) if hasattr(obj, attr) else value
        return value

    @staticmethod
    def basicObjData(obj):
        return {
            "box"       : Record.__attrDefault(obj, 'box', None),
            'confidence': Record.__attrDefault(obj, 'confidence', None),
            'label'     : Record.__attrDefault(obj, 'label', None),
            'codename'  : Record.__attrDefault(obj, 'codename', None)
        }

    @staticmethod
    def vehiclesData(vehicles: List[VehicleData]) -> Dict[str, Any]:
        vehiclesData = []
        for v in vehicles:
            objData = Record.basicObjData(v)
            objData.update({
                'type': Record.__attrDefault(v, 'type', None)
            })
            vehiclesData.append(objData)
        return vehiclesData

    @staticmethod
    def licensePlatesData(licensePlates: List[LicensePlateData]) -> Dict[str, Any]:
        licensePlatesData = []
        for lp in licensePlates:
            objData = Record.basicObjData(lp)
            objData.update({
                'number'        : Record.__attrDefault(lp, 'number', None),
                'cornerPoints'  : Record.__attrDefault(lp, 'cornerPoints', None),
                'centerPosition': Record.__attrDefault(lp, 'centerPosition', None),
            })
            licensePlatesData.append(objData)
        return licensePlatesData

    @staticmethod
    def trafficLightsData(trafficLights: List[TrafficLightData]) -> Dict[str, Any]:
        trafficLightsData = []
        for tl in trafficLights:
            objData = Record.basicObjData(tl)
            objData.update({
                'state': Record.__attrDefault(tl, 'state', None)
            })
            trafficLightsData.append(objData)
        return trafficLightsData

    @staticmethod
    def lanesData(lanes: List[LaneData]) -> Dict[str, Any]:
        lanesData = []
        for lane in lanes:
            objData = Record.basicObjData(lane)
            objData.update({
                'vanishingPoint': Record.__attrDefault(lane, 'vanishingPoint', None),
            })
            lanesData.append(objData)
        return lanesData

    @staticmethod
    def createData(tvideo: TVideo) -> Dict[str, Any]:
        data = {
            'path'                              : tvideo.path,
            'lastCodename'                      : tvideo.lastCodename,
            'start'                             : tvideo.start,
            'end'                               : tvideo.end,
            'currentIndex'                      : tvideo.currentIndex,
            'targetLicensePlateCodename'        : tvideo.targetLicensePlateCodename,
            'trafficLightStateIsRedFrameIndexs' : tvideo.trafficLightStateIsRedFrameIndexs,
            'directs'                           : [direct.name for direct in tvideo.directs],
            'videoVehicles'                     : [Record.vehiclesData(frameData.vehicles) for frameData in tvideo.framesData],
            'videoLicensePlates'                : [Record.licensePlatesData(frameData.licensePlates) for frameData in tvideo.framesData],
            'videoTrafficLights'                : [Record.trafficLightsData(frameData.trafficLights) for frameData in tvideo.framesData],
            'videoLanes'                        : [Record.lanesData(frameData.lanes) for frameData in tvideo.framesData],
        }
        return data

    @classmethod
    def setSavePath(cls, path: str):
        cls.savePath = path
    
    def save(self, tvideo: TVideo):
        with open(f'{self.savePath}/{self.getNextFileName()}.json', 'w') as f:
            data = self.createData(tvideo)
            data.update({'id': self.lastRecordId + 1})
            json.dump(data, f, indent=4, cls=MyEncoder)
        self.lastRecordId = self.lastRecordId + 1
        self.saveLastRecordId(self.lastRecordId)
        print('saved record')
    
    def load(id: int) -> TVideo:
        ...