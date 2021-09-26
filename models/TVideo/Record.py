from typing import Any, Dict, List
from models.TVideo.TVideo import TVideo, TrafficLightState, VehicleData, LicensePlateData, TrafficLightData, LaneData, Direct
from helper import dotdict
import json

class Record:

    savePath: str = 'C:/Users/zT3Tz/Documents/TrafficPolice/store/records'

    def __init__(
        self, 
        tvideo                  : TVideo,
        vehicles                : List[VehicleData],
        licensePlates           : List[LicensePlateData],
        trafficLights           : List[TrafficLightData],
        lanes                   : List[LaneData],
        currentTrafficLightState: TrafficLightState

    ) -> None:
        self.path                              = tvideo.path
        self.lastCodename                      = tvideo.lastCodename
        self.start                             = tvideo.start
        self.end                               = tvideo.end
        self.currentIndex                      = tvideo.currentIndex
        self.targetLicensePlateCodename        = tvideo.targetLicensePlateCodename
        self.directs                           = tvideo.directs
        self.trafficLightStateIsRedFrameIndexs = tvideo.trafficLightStateIsRedFrameIndexs
        self.vehicles                          = self.vehiclesData(vehicles)
        self.licensePlates                     = self.licensePlatesData(licensePlates)
        self.trafficLights                     = self.trafficLightsData(trafficLights)
        self.lanes                             = self.lanesData(lanes)
        self.currentTrafficLightState          = currentTrafficLightState

    @staticmethod
    def __default(obj, attr: str, value) -> Any:
        return getattr(obj, attr) if hasattr(obj, attr) else value

    @staticmethod
    def basicObjData(obj):
        return {
            'box'       : Record.__default(obj, 'box', None),
            'confidence': Record.__default(obj, 'confidence', None),
            'label'     : Record.__default(obj, 'label', None),
            'codename'  : Record.__default(obj, 'codename', None)
        }

    @staticmethod
    def vehiclesData(vehicles: List[VehicleData]) -> Dict[str, Any]:
        vehiclesData = []
        for v in vehicles:
            vehiclesData.append(Record.basicObjData(v).update({
                'type': Record.__default(v, 'type', None)
            }))
        return dotdict(vehiclesData)

    @staticmethod
    def licensePlatesData(licensePlates: List[LicensePlateData]) -> Dict[str, Any]:
        licensePlatesData = []
        for lp in licensePlates:
            licensePlatesData.append(Record.basicObjData(lp).update({
                'number'        : Record.__default(lp, 'number', None),
                'cornerPoints'  : Record.__default(lp, 'cornerPoints', None),
                'centerPosition': Record.__default(lp, 'centerPosition', None),
            }))
        return dotdict(licensePlatesData)

    @staticmethod
    def trafficLightsData(trafficLights: List[TrafficLightData]) -> Dict[str, Any]:
        trafficLightsData = []
        for tl in trafficLights:
            trafficLightsData.append(Record.basicObjData(tl).update({
                'state': Record.__default(tl, 'state', None)
            }))
        return dotdict(trafficLightsData)

    @staticmethod
    def lanesData(lanes: List[LaneData]) -> Dict[str, Any]:
        lanesData = []
        for lane in lanes:
            lanesData.append(Record.basicObjData(lane).update({
                'vanishingPoint': Record.__default(lane, 'vanishingPoint', None),
            }))
        return dotdict(lanesData)

    @classmethod
    def setSavePath(path):
        ...

    def save(self):
        with open(self.savePath, 'w') as f:
            f.write(json.dump())