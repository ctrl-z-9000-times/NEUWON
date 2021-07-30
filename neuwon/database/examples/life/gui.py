from tkinter import *
from tkinter import ttk
import model

class GUI:
    def __init__(self):
        self.model = model.GameOfLife()

        self.root = root = Tk()
        root.title("Feet to Meters")

        mainframe = ttk.Frame(root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)



        canvas = Canvas(parent, width=500, height=400, background='gray75')

        bitmap = np.empty(self.shape)

    @property
    def shape(self):
        return self.model.grid.shape


    def mainloop(self):
        self.root.mainloop()
