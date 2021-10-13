from enum import Enum
from typing import Any, Dict, List, Union
import numpy as np
import time
import json

from models.TVideo.TVideo import TVideo
import config as cfg

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if   isinstance(obj, Enum)        : return obj.name
        elif hasattr(obj, '__dict__')     : return vars(obj)
        elif isinstance(obj, np.integer)  : return int(obj)
        elif isinstance(obj, np.floating) : return float(obj)
        elif isinstance(obj, np.ndarray)  : return None
        else                              : return super(MyEncoder, self).default(obj)

class Record:

    savePath: str = cfg.saveRecordPath

    def __init__(self):
        self.lastRecordId: int = self.getLastRecordId()
    
    @staticmethod
    def getLastRecordId() -> int:
        id = 0
        try: 
            with open(f'{Record.savePath}/lastRecordId.txt', 'r') as f:
                id = f.read()
        except:
            print(f'無法讀取{Record.savePath}/lastRecordId.txt')
        return int(id)

    @staticmethod
    def saveLastRecordId(id: Union[int, str]):
        with open(f'{Record.savePath}/lastRecordId.txt', 'w') as f:
            f.write(str(id))

    def getNextFileName(self) -> str:
        return f'Record_{self.lastRecordId + 1}'

    @staticmethod
    def createData(tvideo: TVideo, id: int, date: str = None) -> Dict[str, Any]:
        tvideo.id = id
        tvideo.date = time.strftime("%m-%d-%Y", time.localtime()) if date is None else date
        return tvideo

    @classmethod
    def setSavePath(cls, path: str):
        cls.savePath = path
    
    def save(self, tvideo: TVideo):
        with open(f'{self.savePath}/{self.getNextFileName()}.json', 'w') as f:
            data = self.createData(tvideo, self.lastRecordId + 1)
            json.dump(data, f, indent=4, cls=MyEncoder)
        self.lastRecordId += 1
        self.saveLastRecordId(self.lastRecordId)
        print('id:', self.lastRecordId, 'saved record')
    
    def load(self, id: int) -> TVideo:
        tvideo = TVideo()
        print('未完成 load 功能')
        try:
            with open(f'{self.savePath}/Record_{id}.json') as f:
                ...
        except:
            print(f'找不到{self.savePath}/Record_{id}.json')
        return tvideo