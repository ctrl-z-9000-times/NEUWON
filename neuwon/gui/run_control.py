from .control_panels import *
from .themes import ThemedTk, set_theme, pick_theme
from .model_container import ModelContainer
from .signal_editor import SignalEditor
from neuwon import Model
from .viewport import Viewport

class RunControl:
    def __init__(self, filename):
        self.model = ModelContainer(filename)
        self.root = ThemedTk()
        set_theme(self.root)
        self.root.rowconfigure(   0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.title("NEUWON: " + self.model.short_name)
        self._init_menu(self.root)
        self._init_main_panel(self.root)
        self.parameters = self.model.load()
        print(self.model.export()) # DEBUGGING!
        self.model.model = Model(**self.model.export())
        self.viewport = Viewport()
        self.viewport.set_scene(self.model.model.get_database())


    def _init_menu(self, parent):
        self.menubar = tk.Menu(parent)
        parent.config(menu = self.menubar)
        self.filemenu = self._init_file_menu(self.menubar)

        self.menubar.add_command(label="Themes", command=lambda: pick_theme(self.root))
        self.menubar.add_command(label="Edit Model", command=self.switch_to_model_editor)

    def _init_file_menu(self, parent_menu):
        # TODO: I want to have all of the buttons, but they don't really make
        # sense in this context? Like new_model, open etc...
        filemenu = tk.Menu(parent_menu, tearoff=False)
        parent_menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Save",      underline=0, command=self.save,    accelerator="Ctrl+S")
        filemenu.add_command(label="Save As",   underline=5, command=self.save_as, accelerator="Ctrl+Shift+S")
        filemenu.add_command(label="Quit",      underline=0, command=self.close)
        self.root.bind_all("<Control-s>", self.save)
        self.root.bind_all("<Control-S>", self.save_as)
        return filemenu

    def _init_main_panel(self, parent):
        self.panel = OrganizerPanel(parent)
        frame = self.panel.get_widget()
        frame.grid(sticky='nesw')
        # self.panel.add_tab("Run Control", )
        self.panel.add_tab("Signal Editor", SignalEditor(frame))
        # self.panel.add_tab("Probes", )

    def switch_to_model_editor(self):
        self.save()
        if self.model.filename is None:
            return
        self.close()
        from .model_editor import ModelEditor
        ModelEditor(self.model.filename)

    def save(self, event=None):
        self.model.save(self.parameters)

    def save_as(self, event=None):
        1/0

    def close(self, event=None):
        self.root.destroy()


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    RunControl(filename).root.mainloop()
