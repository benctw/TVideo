import tkinter as tk
import json
import os
from TrafficPolice.helper import dotdict


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
        with open("settings.json", mode = 'r') as file:
            self.settings = dotdict.deep(json.load(file))

    # 加載界面的文字内容
    def loadContents(self, lang):
        with open(os.path.join("lang", "{}.json".format(lang)), mode = 'r') as file:
            self.contents = dotdict.deep(json.load(file))

    # 修改設定
    def modifySettings(self, settings):
        for (key, val) in settings.items():
            if key in self.settings:
                self.settings[key] = val
            else:
                raise KeyError
        with open('settings.json', 'w') as file:
            json.dump(self.settings, file)

    def moveWindow(event):
        pass

    def createWindow(self):
        # self.main.overrideredirect(True)
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