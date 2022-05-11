from .control_panels import *
from .themes import ThemedTk, set_theme, pick_theme
from .model_container import ModelContainer
from .signal_editor import SignalEditor
from neuwon import Model
from neuwon.database import data_components
from .viewport import Viewport

class ExperimentControl(OrganizerPanel):
    def __init__(self, filename):
        self.model = ModelContainer(filename)
        self.parameters = self.model.load()
        self.viewport = None
        self._initialize_model()
        self.root = ThemedTk()
        set_theme(self.root)
        self.root.rowconfigure(   0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.title("NEUWON: " + self.model.short_name)
        self._init_menu(self.root)
        self._init_main_panel(self.root)
        self._open_viewport()

    def _initialize_model(self):
        self.instance = Model(**self.model.export())
        self.instance.get_database().sort()
        if self.viewport is not None:
            self._open_viewport()

    def _open_viewport(self):
        if self.viewport is not None:
            self.viewport.close()
        self.viewport = Viewport()
        self.viewport.set_scene(self.instance)
        self.root.after(0, self._tick)

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
        super().__init__(parent)
        frame = self.get_widget()
        self.add_tab("run_control", RunControl(frame, self))
        self.add_tab("signal_editor", SignalEditor(frame))
        # self.add_tab("probes", )
        self.add_tab("view_control", ViewControl(frame, self)) # DEBUGGING!
        self.add_tab("color_control", ColorControl(frame, self)) # DEBUGGING!
        frame.grid(sticky='nesw')

    def switch_to_model_editor(self):
        self.save()
        if self.model.filename is None:
            return
        self.close()
        self.viewport.close()
        from .model_editor import ModelEditor
        ModelEditor(self.model.filename)

    def save(self, event=None):
        self.parameters.update(self.get_parameters())
        self.model.save(self.parameters)

    def save_as(self, event=None):
        # Does it even make sense to save-as for a running model?
        1/0

    def close(self, event=None):
        self.root.destroy()
        self.viewport.close()

    def _tick(self):
        self.viewport.tick()
        if self.viewport.alive:
            self.root.after(1, self._tick)

class RunControl(SettingsPanel):
    def __init__(self, parent, experiment):
        super().__init__(parent)
        self.add_entry('run_for',
                valid_range = (0, inf),
                default     = inf,)
        self.add_entry('clock',
                valid_range = (-max_float, max_float),
                default     = 0,)

# THOUGHT: instead of implementing a special widget for lists of checkboxes,
#           Make a scrollbar option for the SettingsPanel?
#           Then its much more general purpose.
#           I can add the custom buttons too, but at the application level?

class ViewControl(Panel):
    def __init__(self, parent, experiment):
        self.frame = ttk.Frame(parent)

        self._neuron_panel = SettingsPanel(self.frame)
        self._neuron_panel.get_widget().grid(row=1, column=0, sticky='nesw')
        self._segment_panel = SettingsPanel(self.frame)
        self._segment_panel.get_widget().grid(row=1, column=1, sticky='nesw')

        database = experiment.instance.get_database()
        Neuron   = database.get_instance_type('Neuron')
        Segment  = database.get_instance_type('Segment')
        for neuron_type in sorted(Neuron.neuron_types_list):
            self._neuron_panel.add_checkbox(neuron_type, default=True)
        for segment_type in sorted(Segment.segment_types_list):
            self._segment_panel.add_checkbox(segment_type, default=True)

    def _select_all(self):
        1/0

    def _deselect_all(self):
        1/0



class ColorControl(SettingsPanel):
    def __init__(self, parent, experiment):
        super().__init__(parent)
        available_components = [
                'voltage'
                # TODO: All of the species concentrations.
        ]
        self.add_dropdown('component', available_components)
        self.add_dropdown('colormap', ['red/blue'])
        self.add_checkbox('show_scale')
        self.add_checkbox('show_time')
        self.add_radio_buttons('background', ['Black', 'White'], default='Black')

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    ExperimentControl(filename).root.mainloop()
