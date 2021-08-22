# 讓 dict 可以用 . 取值
class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def deep(d):
        d = dotdict(d)
        for key in d.keys():
            if isinstance(d[key], dict):
                d[key] = dotdict.deep(d[key])
        return d

import cv2
from rich.progress import track
def imshow(img, title = ''):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def saveVideo(images, path, fps = 30, fourccType = 'mp4v'):
    fourcc = cv2.VideoWriter_fourcc(*fourccType)
    height, width = images[0].shape[:2]
    out = cv2.VideoWriter(path, fourcc, fps, (int(width), int(height)))
    for image in track(images, "saving video"):
        out.write(image)
    out.release()