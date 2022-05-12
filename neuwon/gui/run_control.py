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
        self.add_tab("signal_editor", SignalEditor(frame))
        # self.add_tab("probes", )
        self.add_tab("run_control", RunControl(frame, self))
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


class WorkerThread:
    def __init__(self, instance, messages):
        self.instance = instance
        self.messages = messages
        self.state = 'stopped'
        1/0

    def __call__(self):
        1/0


class RunControl(Panel):
    def __init__(self, parent, experiment):

        database = experiment.instance.get_database()
        Neuron   = database.get_instance_type('Neuron')
        Segment  = database.get_instance_type('Segment')

        self.frame    = ttk.Frame(parent)
        self.settings = SettingsPanel(self.frame)
        self.neurons  = ListSelector(self.frame, Neuron.neuron_types_list, default=True)
        self.segments = ListSelector(self.frame, Segment.segment_types_list, default=True)

        neuron_label  = ttk.Label(self.frame, text='Visible Neurons')
        segment_label = ttk.Label(self.frame, text='Visible Segments')

        self.settings.get_widget().grid(row=1, column=0, sticky='nesw')
        self.neurons .get_widget().grid(row=1, column=1, sticky='nesw')
        neuron_label              .grid(row=0, column=1)
        self.segments.get_widget().grid(row=1, column=2, sticky='nesw')
        segment_label             .grid(row=0, column=2)
        self.frame.grid_rowconfigure(1, weight=1)


        self.settings.add_entry('run_for',
                valid_range = (0, inf),
                default     = inf,)
        self.settings.add_entry('clock',
                valid_range = (-max_float, max_float),
                default     = 0,)

        self.settings.add_section('Video Settings')
        # TODO: Resolution
        # TODO: Target Framerate, sim-to-irl (in case it runs too fast, lol like that will happen)
        available_components = [
                'voltage'
                # TODO: All of the species concentrations.
        ]
        self.settings.add_dropdown('component', available_components)
        self.settings.add_dropdown('colormap', ['red/blue'])
        self.settings.add_checkbox('show_scale')
        self.settings.add_checkbox('show_time')
        self.settings.add_checkbox('show_type')
        self.settings.add_radio_buttons('background', ['Black', 'White'], default='Black')


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    ExperimentControl(filename).root.mainloop()
