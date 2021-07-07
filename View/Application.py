from _typeshed import FileDescriptorLike
import tkinter as tk
import json
import os
from typing_extensions import final


class Application(tk.Frame):
    def __init__(self, main = None):
        super().__init__(main)
        self.main = main
        self.pack()

        self.settings = {}
        self.contents = {}

        self.loadSettings()
        self.loadContents(self.settings["lang"])
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

    def createWindow(self):
        # TODO 使用 self.contents 取代固定的内容
        # TODO 使用 self.settings 取代固定的設定
        self.main.title(self.contents['window']['application_title'])
        self.main.geometry("500x250")
        self.main.resizable(0, 0)
        # self.main.iconbitmap("###path###")
        self.main.config(bg = "black")


main = tk.Tk()
app = Application(main = main)
app.mainloop()