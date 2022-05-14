from ..control_panels import *
from ..project_container import ProjectContainer
from ..themes import ThemedTk, set_theme, pick_theme
from .model_thread import ModelThread, Message
from .signal_editor import SignalEditor
from .viewport.viewport import Viewport
from neuwon import Model
from neuwon.database import data_components
from tkinter import messagebox
import queue
import time

class ModelRunner(OrganizerPanel):
    def __init__(self, filename):
        self.project    = ProjectContainer(filename)
        self.parameters = self.project.load()
        self.exported   = self.project.export()
        self.root = ThemedTk()
        self.instance = None
        self.viewport = None
        self.runner   = ModelThread()
        self.root.bind("<Destroy>", lambda e: self.runner.control_queue.put(Message.QUIT))
        self._initialize_model()
        set_theme(self.root)
        self.root.rowconfigure(   0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.title('NEUWON: ' + self.project.short_name)
        self._init_menu(self.root)
        self._init_main_panel(self.root)
        self._open_viewport()

    def _initialize_model(self):
        self.instance = Model(**self.exported)
        self.instance.get_database().sort()
        self.runner.control_queue.put((Message.INSTANCE, self.instance))
        if self.viewport is not None:
            self._open_viewport()

    def _open_viewport(self):
        if self.viewport is None:
            self.viewport = Viewport()
            self.root.after(0, self._viewport_tick)
        self.viewport.set_scene(self.instance)

    def _init_menu(self, parent):
        menubar = tk.Menu(parent)
        parent.config(menu=menubar)
        self._init_file_menu(menubar)
        menubar.add_command(label='Themes', command=lambda: pick_theme(self.root))
        self._init_model_menu(menubar)

    def _init_file_menu(self, parent_menu):
        # TODO: I want to have all of the buttons, but they don't really make
        #       sense in this context? Like new_model, open etc...
        #       What if New & Open also switched back to the model editor?
        #           Then they would fit more naturally into my imagined workflow.
        file_menu = tk.Menu(parent_menu, tearoff=False)
        parent_menu.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label='Save',    underline=0, command=self.save,    accelerator='Ctrl+S')
        file_menu.add_command(label='Save As', underline=5, command=self.save_as, accelerator='Ctrl+Shift+S')
        file_menu.add_command(label='Quit',    underline=0, command=self.close)
        self.root.bind_all('<Control-s>', self.save)
        self.root.bind_all('<Control-S>', self.save_as)

    def _init_model_menu(self, parent_menu):
        model_menu = tk.Menu(parent_menu, tearoff=False)
        parent_menu.add_cascade(label='Model', menu=model_menu)
        model_menu.add_command(label='Info', command=self.model_info)
        model_menu.add_command(label='Edit', command=self.switch_to_model_editor)
        model_menu.add_command(label='Rebuild', command=self.rebuild_model)

    def _init_main_panel(self, parent):
        super().__init__(parent)
        frame = self.get_widget()
        self.add_tab('run_control', RunControl(frame, self))
        self.add_tab('signal_editor', SignalEditor(frame))
        # self.add_tab('probes', )
        frame.grid(sticky='nesw')

    def model_info(self):
        window, frame = Toplevel('Model Information')
        # TODO: Display a bunch of random stats about the model.
        #       Show the number of instances of various things.
        #               (neurons, segments, and synapses)
        #           Total number and also break down by type.
        #       Num instances of each mechanism type.
        # 
        # The purpose of this is to show the user what the procedural generation
        # actually created, as opposed to things that they could find out by
        # switching to the model editor.
        info = ttk.Label(frame, text='TODO', justify='left', padding=padx)
        info.grid(row=0, column=0, padx=padx, pady=pady)

    def switch_to_model_editor(self):
        self.save()
        if self.project.filename is None:
            return
        self.close()
        self.viewport.close()
        from .model_editor import ModelEditor
        ModelEditor(self.project.filename)

    def rebuild_model(self):
        confirmation = messagebox.askyesno(f'Confirm Rebuild Model',
                f'Are you sure you want to rebuild the model?',
                parent=self.get_widget())
        if not confirmation:
            return
        # TODO: Clear out all of the associated data: probes and recorded data.
        self._initialize_model()

    def save(self, event=None):
        self.parameters.update(self.get_parameters())
        self.project.save(self.parameters)

    def save_as(self, event=None):
        # Does it even make sense to save-as for a running model?
        1/0

    def close(self, event=None):
        self.root.destroy()
        if self.viewport is not None:
            self.viewport.close()

    def _viewport_tick(self):
        start_time = time.time()
        self.viewport.tick()
        render_time = 1000 * (time.time() - start_time)
        if self.viewport.alive:
            max_fps = 30
            self.root.after(round(1000 / max_fps - render_time), self._viewport_tick)
        else:
            self.viewport = None
            self.runner.control_queue.put(Message.HEADLESS)

    def _collect_results(self):
        try:
            render_data = self.runner.results_queue.get_nowait()
        except queue.Empty:
            self.root.after(1, self._collect_results)
            return
        self.root.after(1, self._collect_results)

class RunControl(Panel):
    def __init__(self, parent, experiment):

        database = experiment.instance.get_database()
        Neuron   = database.get_instance_type('Neuron')
        Segment  = database.get_instance_type('Segment')

        self.frame    = ttk.Frame(parent)
        self.settings = SettingsPanel(self.frame)
        self.visible  = FilterVisible(self.frame, experiment.exported)

        self.settings.get_widget().grid(row=1, column=1, sticky='nesw', padx=padx, pady=pady)
        self.visible .get_widget().grid(row=1, column=2, sticky='nesw', padx=padx, pady=pady)
        self.frame.grid_rowconfigure(1, weight=1)


        # TODO: start/pause button.
        start_callback = lambda: experiment.runner.control_queue.put((Message.RUN, None))
        pause_callback = lambda: experiment.runner.control_queue.put((Message.PAUSE, None))

        run_for = self.settings.add_entry('run_for',
                valid_range = (0, inf),
                default     = inf,)
        clock = self.settings.add_entry('clock',
                valid_range = (-max_float, max_float),
                default     = 0,)
        self.disable_while_running = [run_for, clock]

        self.settings.add_section('Video Settings')
        self.settings.add_empty_space()
        # TODO: Resolution
        self.settings.add_entry('Slowdown',
                default = 1000,
                units   = 'Real-Time : Model-Time')
        available_components = [
                'voltage'
                # TODO: All of the species concentrations.
        ]
        self.settings.add_dropdown('component', available_components)
        self.settings.add_dropdown('colormap', ['red/blue'])
        self.settings.add_checkbox('show_scale')
        self.settings.add_checkbox('show_type')
        self.settings.add_checkbox('show_time')
        self.settings.add_radio_buttons('background', ['Black', 'White'], default='Black')

class FilterVisible(Panel):
    def __init__(self, parent, parameters):
        neuron_types_list  = sorted(parameters['neurons'].keys())
        segment_types_list = sorted(parameters['segments'].keys())
        # Make the widgets.
        self.frame    = ttk.Frame(parent)
        neuron_label  = ttk.Label(self.frame, text='Visible Neurons')
        segment_label = ttk.Label(self.frame, text='Visible Segments')
        self.neurons  = ListSelector(self.frame, neuron_types_list, default=True)
        self.segments = ListSelector(self.frame, segment_types_list, default=True)
        # Arrange the widgets.
        neuron_label              .grid(row=1, column=1, padx=padx, pady=pady)
        segment_label             .grid(row=1, column=2, padx=padx, pady=pady)
        self.neurons .get_widget().grid(row=2, column=1, padx=padx, pady=pady, sticky='nesw')
        self.segments.get_widget().grid(row=2, column=2, padx=padx, pady=pady, sticky='nesw')
        self.frame.grid_rowconfigure(2, weight=1) # Resize vertically.

    def get_parameters(self) -> dict:
        return {'visible_neurons':  self.neurons.get_parameters()
                'visible_segments': self.segments.get_parameters()}

    def set_parameters(self, parameters: dict):
        self.neurons .set_parameters(parameters['visible_neurons'])
        self.segments.set_parameters(parameters['visible_segments'])

    def export(self):
        neuron_types  = [nt for nt, v in self.neurons.get_parameters().items() if v]
        segment_types = [st for st, v in self.segments.get_parameters().items() if v]
        return {'neuron_types': neuron_types, 'segment_types': segment_types}


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    ModelRunner(filename).root.mainloop()
