import cv2
from rich.progress import track
import numpy as np

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

def imshow(img: np.ndarray, title = ''):
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

# 求三點曲率
def curvature(x, y):
    t_a = np.linalg.norm([x[1]-x[0],y[1]-y[0]])
    t_b = np.linalg.norm([x[2]-x[1],y[2]-y[1]])
    
    M = np.array([
        [1, -t_a, t_a**2],
        [1, 0,    0     ],
        [1,  t_b, t_b**2]
    ])

    a = np.matmul(np.linalg.inv(M), x)
    b = np.matmul(np.linalg.inv(M), y)

    kappa = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)
    return kappa, [b[1],-a[1]]/np.sqrt(a[1]**2.+b[1]**2.)