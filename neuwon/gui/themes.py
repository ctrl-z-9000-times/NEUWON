from .control_panels import *
from ttkthemes import ThemedTk

__all__ = ('ThemedTk', 'set_theme', 'pick_theme')

# TODO: This should save the selected theme to a hidden file in the users home folder.

default_theme = 'black'

def set_theme(root, theme=None):
    if theme is None:
        # TODO: load theme from file here.
        theme = default_theme
    else:
        # TODO: save theme to file here.
        pass
    root.set_theme(theme)

def pick_theme(root):
    window, frame = Toplevel('Select a Theme')
    themes = sorted(root.get_themes())
    rows   = int(len(themes) ** .5)
    cols   = int(np.ceil(len(themes) / rows))
    for idx, name in enumerate(themes):
        def make_closure():
            current_name = name
            return lambda: set_theme(root, current_name)
        button = ttk.Button(frame, text=name.title(), command=make_closure())
        button.grid(row=idx//cols, column=idx%cols, padx=padx, pady=pady)
    for row in range(rows):
        frame.rowconfigure(row, weight=1)
    for col in range(cols):
        frame.columnconfigure(col, weight=1)
    window.bind('<Escape>', lambda event: window.destroy())
