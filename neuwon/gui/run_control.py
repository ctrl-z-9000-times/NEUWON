from .control_panels import *
from tkinter import messagebox
from .themes import ThemedTk, set_theme, pick_theme
from .project_container import ProjectContainer
from .signal_editor import SignalEditor
from neuwon import Model
from neuwon.database import data_components
from .viewport.viewport import Viewport
from .model_runner import ModelRunner, Message
import queue

class ExperimentControl(OrganizerPanel):
    def __init__(self, filename):
        self.model = ProjectContainer(filename)
        self.parameters = self.model.load()
        self.root = ThemedTk()
        self.instance = None
        self.viewport = None
        self.runner   = ModelRunner()
        self.root.bind("<Destroy>", lambda event: self.runner.quit())
        self._initialize_model()
        set_theme(self.root)
        self.root.rowconfigure(   0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.title('NEUWON: ' + self.model.short_name)
        self._init_menu(self.root)
        self._init_main_panel(self.root)
        self._open_viewport()

    def _initialize_model(self):
        self.instance = Model(**self.model.export())
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
        if self.model.filename is None:
            return
        self.close()
        self.viewport.close()
        from .model_editor import ModelEditor
        ModelEditor(self.model.filename)

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
        self.model.save(self.parameters)

    def save_as(self, event=None):
        # Does it even make sense to save-as for a running model?
        1/0

    def close(self, event=None):
        self.root.destroy()
        if self.viewport is not None:
            self.viewport.close()

    def _viewport_tick(self):
        try:
            render_data = self.runner.results_queue.get_nowait()
        except queue.Empty:
            pass

        self.viewport.tick()
        if self.viewport.alive:
            self.root.after(1, self._viewport_tick)
        else:
            self.viewport = None

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


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    ExperimentControl(filename).root.mainloop()
