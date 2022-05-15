from ..control_panels import *
from ..project_container import ProjectContainer
from ..themes import ThemedTk, set_theme, pick_theme
from .model_thread import ModelThread, Message
from .signal_editor import SignalEditor
from .viewport.viewport import Viewport, Coloration
from neuwon import Model
from tkinter import messagebox
import queue
import time

class ModelRunner(OrganizerPanel):
    def __init__(self, filename):
        self.project    = ProjectContainer(filename)
        self.parameters = self.project.export()
        self.runner     = ModelThread()
        self.viewport   = Viewport()
        self._initialize_model()
        self.root       = ThemedTk()
        set_theme(self.root)
        self.root.rowconfigure(   0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.title('NEUWON: ' + self.project.short_name)
        self.root.bind("<Destroy>", self.close)
        self._init_menu(self.root)
        self._init_main_panel(self.root)
        self.set_parameters(self.get_parameters())
        self.root.after(0, self._collect_results)
        self.root.after(0, self._viewport_tick)

    def _initialize_model(self):
        self.model = Model(**self.parameters)
        self.model.get_database().sort()
        self.runner.control_queue.put((Message.INSTANCE, self.model))
        self.runner.control_queue.put((Message.HEADLESS, not self.viewport.is_open()))
        if self.viewport.is_open():
            self.viewport.set_scene(self.model)

    def _open_viewport(self):
        if not self.viewport.is_open():
            self.viewport.open()
            self.root.after(0, self._viewport_tick)
        self.viewport.set_scene(self.model)
        self.runner.control_queue.put((Message.HEADLESS, False))

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
        self.add_tab('run_control', MainControl(frame, self))
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
        from ..model_editor.model_editor import ModelEditor
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
        parameters = self.project.load()
        parameters.update(self.get_parameters())
        self.project.save(parameters)

    def save_as(self, event=None):
        # Does it even make sense to save-as for a running model?
        1/0

    def close(self, event=None):
        self.runner.control_queue.put(Message.QUIT)
        self.viewport.close()
        if event is None or event.type != tk.EventType.Destroy:
            self.root.destroy()

    def _viewport_tick(self):
        if not self.viewport.is_open():
            self.runner.control_queue.put((Message.HEADLESS, True))
            return
        start_time = time.time()
        rclick_segment = self.viewport.tick()
        render_time = 1000 * (time.time() - start_time)
        max_fps = 30
        self.root.after(round(1000 / max_fps - render_time), self._viewport_tick)

    def _collect_results(self):
        while True:
            if self.runner.exception is not None:
                # TODO: Display the text of the exception to the user in a modal
                # pop-up dialog. Allow the user to keep running this program,
                # but warn them that its effectively dead.
                # -> User might want to save recorded data, or save recording of video.
                1/0
            try:
                results = self.runner.results_queue.get_nowait()
            except queue.Empty:
                self.root.after(10, self._collect_results)
                return
            if results == Message.PAUSE:
                self.run_control.run_ctrl.pause()
            else:
                timestamp, remaining, render_data = results
                timestamp = round(timestamp, 6) # Round to nearest nanosecond

                self.run_control.run_ctrl.set_parameters({
                        'run_for': remaining,
                        'clock':   timestamp,
                })

                if self.viewport.is_open():
                    if self.run_control.video.get_parameters()['show_time']:
                        text = f'Clock: {timestamp} ms'
                    else:
                        text = ''
                    self.viewport.get_text_overlay().set_text(text)
                    # Normalize the render_data into the range [0,1]
                    vmin = -100
                    vmax = +100
                    render_data -= vmin
                    render_data /= (vmax - vmin)
                    self.viewport.get_coloration().set_segment_values(render_data)
                    # TODO: Read these values from the parameters & model.clock!
                    slowdown = 1000
                    dt = .1
                    self.root.after(round(slowdown * dt), self._collect_results)
                    return


class MainControl(Panel):
    def __init__(self, parent, experiment):
        self.frame    = ttk.Frame(parent)
        self.run_ctrl = RunControl(self.frame, experiment.runner)
        self.video    = VideoSettings(self.frame, experiment.runner, experiment.viewport)
        self.visible  = FilterVisible(self.frame, experiment.parameters)

        self.run_ctrl.get_widget().grid(row=1, column=1, sticky='nw', padx=padx, pady=pady)
        self.video   .get_widget().grid(row=2, column=1, sticky='nw', padx=padx, pady=pady)
        self.visible .get_widget().grid(row=1, column=2, rowspan=2, sticky='nesw', padx=padx, pady=pady)
        self.frame.grid_rowconfigure(2, weight=1)

    def get_parameters(self) -> dict:
        return {'run_ctrl': self.run_ctrl.get_parameters(),
                'video':    self.video   .get_parameters(),
                'visible':  self.visible .get_parameters(),}

    def set_parameters(self, parameters:dict):
        self.run_ctrl.set_parameters(parameters['run_ctrl'])
        self.video   .set_parameters(parameters['video'])
        self.visible .set_parameters(parameters['visible'])


class RunControl(Panel):
    def __init__(self, parent, runner):
        self.runner   = runner
        self.frame    = ttk.Frame(parent)
        self.settings = SettingsPanel(self.frame)

        self.running = False

        start_button = ttk.Button(self.frame, text='Start/Pause', command=self.toggle)

        start_button.grid(row=1)
        self.settings.get_widget().grid(row=2)

        run_for = self.settings.add_entry('run_for',
                valid_range = (0, inf),
                default     = inf,)
        clock = self.settings.add_entry('clock',
                valid_range = (-max_float, max_float),
                default     = 0,)
        self.disable_while_running = [run_for, clock]

    def toggle(self):
        if self.running:
            self.pause()
        else:
            self.start()

    def pause(self):
        self.runner.control_queue.put(Message.PAUSE)
        for entry in self.disable_while_running:
            entry.configure(state='enabled')
        self.running = False

    def start(self):
        parameters = self.get_parameters()
        self.runner.control_queue.put((Message.SET_TIME, parameters['clock']))
        self.runner.control_queue.put((Message.DURATION, parameters['run_for']))
        self.runner.control_queue.put(Message.RUN)
        for entry in self.disable_while_running:
            entry.configure(state='readonly')
        self.running = True

    def get_parameters(self):
        return self.settings.get_parameters()
    def set_parameters(self, parameters):
        return self.settings.set_parameters(parameters)


class VideoSettings(Panel):
    def __init__(self, parent, runner, viewport):
        self.runner   = runner
        self.viewport = viewport
        self.frame    = ttk.Frame(parent)
        self.settings = SettingsPanel(self.frame)
        self.settings.get_widget().grid(row=1, column=1, sticky='nesw', padx=padx, pady=pady)

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
        self.settings.add_dropdown('component', available_components,
                                    default=available_components[0])
        self.settings.add_dropdown('colormap', Coloration.get_all_colormaps(),
                                    default='turbo')
        self.settings.add_checkbox('show_scale')
        self.settings.add_checkbox('show_type', default=True)
        self.settings.add_checkbox('show_time', default=True)
        self.settings.add_radio_buttons('background', ['Black', 'White'], default='Black')
        self.settings.add_callback(self.settings_changed)

    def get_parameters(self):
        return self.settings.get_parameters()
    def set_parameters(self, parameters):
        return self.settings.set_parameters(parameters)

    def settings_changed(self):
        parameters = self.get_parameters()
        component  = 'Segment.' + parameters['component']
        self.runner.control_queue.put((Message.COMPONENT, component))
        self.viewport.coloration.set_colormap(parameters['colormap'])
        show_type = parameters['show_type']
        self.viewport.text_overlay.set_neuron_type(show_type)
        self.viewport.text_overlay.set_segment_type(show_type)
        self.viewport.set_background_color(parameters['background'])


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
        return {'visible_neurons':  self.neurons.get_parameters(),
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
