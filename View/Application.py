import tkinter as tk


class Application(tk.Frame):
    def __init__(self, main = None):
        super().__init__(main)
        self.main = main
        self.pack()

        self.settings = {}
        self.contents = {}

        self.createWindow()

    # 加載設定文件
    def loadSettings(self):
        # TODO 讀取 settings.txt，把設定參數解析儲存到 self.settings，要先定義文件中參數的儲存格式
        pass

    # 加載界面的文字内容
    def loadContents(self, lang):
        # TODO 根據參數 lang 讀取 lang 文件夾中的 txt 文件儲存到self.contents
        pass

    # 修改設定
    def modifySettings(self):
        # TODO 修改 settings 文件中的資料
        # 可能要重新渲染
        pass

    def createWindow(self):
        # TODO 使用 self.contents 取代固定的内容
        # TODO 使用 self.settings 取代固定的設定
        self.main.title("Traffic Police")
        self.main.geometry("500x250")
        self.main.resizable(0, 0)
        # self.main.iconbitmap("###path###")
        self.main.config(bg = "black")


main = tk.Tk()
app = Application(main = main)
app.mainloop()