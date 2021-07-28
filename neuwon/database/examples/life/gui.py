from tkinter import *
from tkinter import ttk

class GUI:
    def __init__(self):
        self.root = root = Tk()
        root.title("Feet to Meters")

        mainframe = ttk.Frame(root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

    def mainloop(self):
        self.root.mainloop()
