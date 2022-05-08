from .control_panels import *
from .themes import ThemedTk, set_theme, pick_theme
from .model_container import ModelContainer
from neuwon import Model
import json

class RunControl:
    def __init__(self, filename):
        self.model = ModelContainer(filename)
        self.root = ThemedTk()
        set_theme(self.root)



        self.run = ttk.Button(self.root, text = 'Run',
                command = self.run_callback)
        self.run.pack(side = 'top')

    def instantiate_model(self):
        self.model = Model()
        1/0


    def run_callback(self):
        print("Run!")



if __name__ == '__main__':
    Main().root.mainloop()
