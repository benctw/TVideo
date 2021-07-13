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