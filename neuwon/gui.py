""" Graphical User Interface

The purpose of this program is to render the internal state of a model. It
should be suitable for debugging models and gathering experimential data.
"""
"""
NOTES:
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

* 2D Plots: Show time course of a database component.
    + Matplotlib can embed into tk apps! And it can update dynamically at run time?
    + Select component
    + Select location on membrane

* Raster plots of AP activity?

* Walk through the database, entry by entry. This is something I currently do
  with print statements.

* Model execution: how does the user step through an experiment? setup IO?

* Capture: experiments should be able to directly use the output of this GUI in
  their paper's results.

"""
