import json

class INFO:
    head: str = '[INFO]'
    
    def __init__(self, msg) -> None:
        print(self.head, msg)

    @staticmethod
    def percent(value: float):
        print(INFO.head, json.dumps({'percent': value}))
    