import tkinter as tk


class Application(tk.Frame):
    def __init__(self, main = None):
        super().__init__(main)
        self.main = main
        self.pack()

        self.settings = {}
        self.texts = {}

        self.createWindow()

    # 加載設定文件
    def loadSettings(self):
        # TODO 讀取指定路徑的txt文件，把設定參數加載並解析儲存到self.settings，要先定義文件中參數的儲存格式
        pass

    # 加載界面的文字内容
    def loadText(self):
        # TODO 同loadSettings()，但換了一個txt文件
        pass

    def createWindow(self):
        # TODO 使用 self.texts 取代固定的内容
        # TODO 使用 self.settings 取代固定的設定
        self.main.title("Traffic Police")
        self.main.geometry("500x250")
        self.main.resizable(0, 0)
        self.main.iconbitmap("###path###")
        self.main.config(bg = "black")


main = tk.Tk()
app = Application(main = main)
app.mainloop()