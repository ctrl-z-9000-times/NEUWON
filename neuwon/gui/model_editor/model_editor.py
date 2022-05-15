from ..control_panels import *
from ..project_container import ProjectContainer
from ..model_runner.model_runner import ModelRunner
from ..themes import ThemedTk, set_theme, pick_theme
from .mechanism_editor import MechanismManager
from .neuron_editor import SegmentEditor, NeuronEditor
from .region_editor import RegionEditor
from .species_editor import SpeciesEditor
from tkinter import filedialog
import json

# TODO: The rename & delete buttons need callbacks to apply the changes through
# the whole program.


class ModelEditor(OrganizerPanel):
    def __init__(self, filename=None):
        self.root = ThemedTk()
        set_theme(self.root)
        self.root.rowconfigure(   0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self._init_menu(self.root)
        self._init_main_panel(self.root)
        self.project = ProjectContainer(filename)
        if self.project.filename is not None:
            self.set_parameters(self.project.load())
            self._set_title()
        else:
            self.new_model()

    def _init_menu(self, parent):
        self.menubar = tk.Menu(parent)
        parent.config(menu = self.menubar)
        self.filemenu = self._init_file_menu(self.menubar)

        self.menubar.add_command(label='Themes', command=lambda: pick_theme(self.root))
        self.menubar.add_command(label='Run', command=self.switch_to_run_control)

    def _init_file_menu(self, parent_menu):
        filemenu = tk.Menu(parent_menu, tearoff=False)
        parent_menu.add_cascade(label='File', menu=filemenu)
        filemenu.add_command(label='New Model', underline=0, command=self.new_model)
        filemenu.add_command(label='Open',      underline=0, command=self.open,    accelerator='Ctrl+O')
        filemenu.add_command(label='Save',      underline=0, command=self.save,    accelerator='Ctrl+S')
        filemenu.add_command(label='Save As',   underline=5, command=self.save_as, accelerator='Ctrl+Shift+S')
        filemenu.add_command(label='Export',    underline=1, command=self.export)
        filemenu.add_command(label='Quit',      underline=0, command=self.close)
        self.root.bind_all('<Control-o>', self.open)
        self.root.bind_all('<Control-s>', self.save)
        self.root.bind_all('<Control-S>', self.save_as)
        return filemenu

    def _init_main_panel(self, parent):
        super().__init__(parent)
        frame = self.get_widget()
        self.add_tab('simulation', SimulationSettings(frame))
        self.add_tab('mechanisms', MechanismManager(frame))
        self.add_tab('species',    SpeciesEditor(frame))
        self.add_tab('regions',    RegionEditor(frame))
        self.add_tab('segments',   SegmentEditor(frame, self))
        self.add_tab('neurons',    NeuronEditor(frame, self))
        frame.grid(sticky='nesw')

    def _set_title(self):
        title = 'NEUWON Model Editor'
        if self.project.short_name is not None:
            title += ': ' + self.project.short_name
        self.root.title(title)

    def new_model(self, event=None):
        self.project.set_file(None)
        self._set_title()
        self.set_parameters({})

    def open(self, event=None):
        open_filename = filedialog.askopenfilename(title='Open Model',
                        filetypes=[('Model File', '.json')])
        if not open_filename:
            return
        self.project.set_file(open_filename)
        self.set_parameters(self.project.load())
        self._set_title()

    def save(self, event=None):
        if not self.project.filename:
            self.save_as()
        else:
            self.root.focus_set() # Unfocusing triggers input validation.
            parameters = self.get_parameters() # Successfully get the parameters before truncating the output file.
            self.project.save(parameters)

    def save_as(self, event=None):
        save_as_filename = filedialog.asksaveasfilename(defaultextension='.json')
        if not save_as_filename:
            return
        self.project.set_file(save_as_filename)
        self.save()
        self._set_title()

    def export(self):
        self.project.save(self.get_parameters())
        parameters = self.project.export()
        export_filename = filedialog.asksaveasfilename(defaultextension='.py')
        with open(export_filename, 'wt') as f:
            json.dump(parameters, f, indent=4)
            f.flush()

    def close(self, event=None):
        self.root.destroy()

    def switch_to_run_control(self, event=None):
        self.save()
        if self.project.filename is None:
            return
        # TODO: Consider hiding this window instead of closing it? Because then
        # the user can quickly switch back to the model editor without
        # re-loading everything. The mechanisms in particular can take a long
        # time to load.
        self.close()
        ModelRunner(self.project.filename)

    def run(self):
        ''' Blocks calling thread until the ModelEditor is closed. '''
        self.root.mainloop()


def SimulationSettings(root):
    self = SettingsPanel(root)

    self.add_entry('time_step',
            valid_range = (greater_than_zero, max_float),
            default     = 0.1,
            units       = 'ms')

    self.add_entry('temperature',
            valid_range = (0, 100),
            default     = 37.0,
            units       = '°C')

    self.add_entry('initial_voltage',
            valid_range = (-max_float, max_float),
            default     = -70.0,
            units       = 'mV')

    self.add_entry('cytoplasmic_resistance',
            valid_range = (greater_than_zero, max_float),
            default     = 100.0,
            units       = '')

    self.add_entry('membrane_capacitance',
            valid_range = (greater_than_zero, max_float),
            default     = 1.0,
            units       = 'μf/cm^2')

    return self

