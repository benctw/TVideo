import tkinter as tk
import json
import os
from ..helper import dotdict


class Application(tk.Frame):
    def __init__(self, main = None):
        super().__init__(main)
        self.main = main
        self.pack()

        self.settings = {}
        self.contents = {}

        self.loadSettings()
        self.loadContents(self.settings.lang)
        self.createWindow()

    # 加載設定文件
    def loadSettings(self):
        # TODO 讀取 settings.json，把内容解析並儲存到 self.settings
        pass

    # 加載界面的文字内容
    def loadContents(self, lang):
        # TODO 根據參數 lang 名稱讀取文件夾 lang 中的 json 文件解析並儲存到 self.contents
        pass

    # 修改設定
    def modifySettings(self, settings):
        for (key, val) in settings.items:
            self.settings[key] = val
        with os.open('settings.json', 'w') as f:
            json.dump(self.settings, f)

    def moveWindow(event):
        pass

    def createWindow(self):
        self.main.overrideredirect(True)
        self.main.title(self.contents.window.Application_Title)
        self.main.geometry('{}x{}+{}+{}'.format(500, 250, 50, 50))
        self.main.resizable(0, 0)
        # self.main.iconbitmap("## logo paht ##")
        self.main.config(bg = "black")


def center(toplevel):
    toplevel.update_idletasks()

    screen_width = toplevel.winfo_screenwidth()
    screen_height = toplevel.winfo_screenheight()

    size = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))
    x = screen_width / 2 - size[0] / 2
    y = screen_height / 2 - size[1] / 2

    toplevel.geometry('+{}+{}'.format(x, y))



main = tk.Tk()
app = Application(main = main)
app.mainloop()