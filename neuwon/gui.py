""" Graphical User Interface

The purpose of this program is to render the internal state of a model. It
should be suitable for debugging models and gathering experimential data.
"""
"""
NOTES:
* The HH propagation example should be a 20-minute tutorial w/o programming?

* Use the tools which I'm familiar with: tk & matplotlib.
  How to make 3d? I really really do not want to deal with any of that.
  Implementing 3d rendering is obviously a non-starter, even using a framework
  or engine is overkill...

* No initialization. Initializing a model is too complicated for a GUI. User
  must write that python code.
  Start the GUI by calling the "model.gui()" method.

* The GUI APP is a collection of APPLETS. Each applet is a single independent
  window-frame which performs a single function.
  The menu bar (file,edit,view,tools,settings,help,etc) will contain buttons
  for opening all of the different applets.
  By coding each applet as an independent entity, I should be able to organize
  them easily, and maybe even allow the user to rearange them to their liking.

* 3D Viewports: There are four windows with pictures of the model: a true 3D
  rendering of it and 3 orthogonal projections along the XYZ axes which are
  centered on the camera. I want these windows to look and feel like a normal
  3D editor.
    -> Each viewport (or set of 4 viewports) has a rendering control window.
        + Select database component to view
        + Select a color scale
        + Enter min & max (if not already known). On error the GUI will
          saturate or something instead of crash...
        + Enter background color & fade distance
    -> Right-click on the viewport will select a neuron segment and open a menu
       to inspect it, plot data at that point.

* Plot Time Series:
    + Any database component.
    + Use database.TimeSeriesBuffer class.
    + Animate grapic at run time w/ latest data?

* Raster plots of AP activity?

* Walk through the database, entry by entry. This is something I currently do
  with print statements.

* Capture: experiments should be able to directly use the output of this GUI in
  their paper's results.
    - No replay. Model can never be "rewound". Too much data to be feasible. Not useful anyways.



* Model execution: how does the user step through an experiment? setup IO?

THREADS: Should the GUI run in its own thread?
-> Single threaded pros/cons:
    + simpler.
    + guarenteed to collect all datapoints exactly once.
    - TK demands flow control.
-> Multithreaded pros/cons:
    - complex
    + User needs to keep control of the program to run the model, until I get
      execution control implemented...


* Execution Control Applet
    -> "Owns" the model.
        + Will take over updating gui stuff from the model (Animations & DataPlots)
    -> Displays the elapsed time since last reset.
    -> Control buttons:
        * Standard start/pause/stop buttons


* Input Control Applet
    -> Inject electric charge or chemicals.
    -> User selects entity & component.
    -> Library of waveform types, user parameterizes them w/ duration & magnitude
    -> Button to show data plot of this entity & component.
    * Note: This applet needs to automatically connect to the execution control applet.




"""

from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import itertools
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tkinter
import neuwon.database as db

class PlotData():
    def __init__(self, entity, component, tk_frame=None):
        """
        Argument frame is a tkinter window frame.
        """
        self.entity     = entity
        self.component  = str(component)
        assert(isinstance(entity, db.Entity))
        if tk_frame is None:
            tk_frame = tkinter.Tk()
            tk_frame.wm_title("Embedding Animation in Tk")
        self.tk_frame   = tk_frame
        self.time_step  = entity.database.access("time_step")
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.grid()
        self.reset()
        self.animation = animation.FuncAnimation(self.fig, self.run, init_func=self.reset,
                interval=200,)
        canvas = FigureCanvasTkAgg(self.fig, master=self.tk_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        tkinter.mainloop()

    def reset(self):
        self.ticks = 0
        self.xdata = []
        self.ydata = []
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(0, 10)
        self.line.set_data(self.xdata, self.ydata)
        return self.line,

    def _advance(self):
        self.ticks += 1
        self.xdata.append(self.ticks * self.time_step)
        self.ydata.append(self.entity.read(self.component))
        xmin, xmax = self.ax.get_xlim()
        if t >= xmax:
            self.ax.set_xlim(xmin, 2*xmax)
            self.ax.figure.canvas.draw()
        self.line.set_data(self.xdata, self.ydata)

    def run(self, data):
        return self.line,

if __name__ == '__main__':
    root = tkinter.Tk()
    root.wm_title("Embedding Animation in Tk")
    tkinter.mainloop()
