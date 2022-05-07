from .control_panels import *
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

matplotlib.use('TkAgg')

class MatplotlibEmbed:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.figure = Figure()
        self.canvas = FigureCanvasTkAgg(self.figure, self.frame)
        NavigationToolbar2Tk(self.canvas, self.frame)
        self.axes = self.figure.add_subplot()
        self.canvas.get_tk_widget().pack()

    def update(self, timeseries):
        timestamps = timeseries.get_timestamps()
        data       = timeseries.get_data()
        self.axes.clear()
        self.axes.plot(timestamps, data)
        self.axes.set_xlabel('ms')
        self.canvas.draw()
