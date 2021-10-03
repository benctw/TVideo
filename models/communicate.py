import json
import sys

class INFO:
    head: str = '[INFO]'

    def __init__(self, msg) -> None:
        jsonstr = json.dumps({'msg': msg})
        print(f'{INFO.head} {jsonstr}')

    @staticmethod
    def progress(value: float):
        jsonstr = json.dumps({'progress': value})
        print(f'{INFO.head} {jsonstr}')
    
    # def __init__(self, msg) -> None:
    #     sys.stdout.flush()
    #     jsonstr = json.dumps({'msg': msg})
    #     sys.stdout.write(f'{self.head} {jsonstr}')
    #     sys.stdout.flush()

    # @staticmethod
    # def progress(value: float):
    #     sys.stdout.flush()
    #     jsonstr = json.dumps({'progress': value})
    #     sys.stdout.write(f'{INFO.head} {jsonstr}')
    #     sys.stdout.flush()
    