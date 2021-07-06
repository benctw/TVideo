from abc import ABCMeta, abstractmethod

class CVModel():
    __metaclass__ = ABCMeta
    def __init__():
        pass

    @abstractmethod
    def detectImage(image):
        pass

    # 根據interval的間隔遍歷一遍影片的幀
    def loopVideo(self, video, interval):
        # TODO get video frame
        frame = None
        self.detectImage(frame)