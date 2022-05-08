from .control_panels import *
from .region_editor import RegionEditor
from .species_editor import SpeciesEditor
from .mechanism_editor import MechanismManager, MechanismSelector
from .model_container import ModelContainer
from .run_control import RunControl
from .themes import ThemedTk, set_theme, pick_theme
from tkinter import filedialog, simpledialog
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
        self.model = ModelContainer(filename)
        if self.model.filename is not None:
            self.set_parameters(self.model.load())
            self._set_title()
        else:
            self.new_model()

    def _init_menu(self, parent):
        self.menubar = tk.Menu(parent)
        parent.config(menu = self.menubar)
        self.filemenu = self._init_file_menu(self.menubar)

        self.menubar.add_command(label="Themes", command=lambda: pick_theme(self.root))
        self.menubar.add_command(label="Run", command=self.switch_to_run_control)

    def _init_file_menu(self, parent_menu):
        filemenu = tk.Menu(parent_menu, tearoff=False)
        parent_menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="New Model", underline=0, command=self.new_model)
        filemenu.add_command(label="Open",      underline=0, command=self.open,    accelerator="Ctrl+O")
        filemenu.add_command(label="Save",      underline=0, command=self.save,    accelerator="Ctrl+S")
        filemenu.add_command(label="Save As",   underline=5, command=self.save_as, accelerator="Ctrl+Shift+S")
        filemenu.add_command(label="Export",    underline=1, command=self.export)
        filemenu.add_command(label="Quit",      underline=0, command=self.close)
        self.root.bind_all("<Control-o>", self.open)
        self.root.bind_all("<Control-s>", self.save)
        self.root.bind_all("<Control-S>", self.save_as)
        return filemenu

    def _init_main_panel(self, parent):
        super().__init__(parent)
        frame = self.get_widget()
        self.add_tab('simulation', Simulation(frame))
        self.add_tab('mechanisms', MechanismManager(frame))
        self.add_tab('species',    SpeciesEditor(frame))
        self.add_tab('regions',    RegionEditor(frame))
        self.add_tab('segments',   SegmentEditor(frame, self))
        self.add_tab('neurons',    Neurons(frame, self))
        frame.grid(sticky='nesw')

    def _set_title(self):
        title = "NEUWON Model Editor"
        if self.model.short_name is not None:
            title += ": " + self.model.short_name
        self.root.title(title)

    def new_model(self, event=None):
        self.model.set_file(None)
        self._set_title()
        self.set_parameters({})

    def open(self, event=None):
        open_filename = filedialog.askopenfilename(title="Open Model",
                        filetypes=[('Model File', '.json')])
        if not open_filename:
            return
        self.model.set_file(open_filename)
        self.set_parameters(self.model.load())
        self._set_title()

    def save(self, event=None):
        if not self.model.filename:
            self.save_as()
        else:
            self.root.focus_set() # Unfocusing triggers input validation.
            parameters = self.get_parameters() # Successfully get the parameters before truncating the output file.
            self.model.save(parameters)

    def save_as(self, event=None):
        save_as_filename = filedialog.asksaveasfilename(defaultextension='.json')
        if not save_as_filename:
            return
        self.model.set_file(save_as_filename)
        self.save()
        self._set_title()

    def export(self):
        parameters = self.model.export(self.get_parameters())
        export_filename = filedialog.asksaveasfilename(defaultextension='.py')
        with open(export_filename, 'wt') as f:
            json.dump(parameters, f, indent=4)
            f.flush()

    def close(self, event=None):
        self.root.destroy()

    def switch_to_run_control(self, event=None):
        self.save()
        if self.model.filename is None:
            return
        self.close()
        RunControl(self.model.filename)

    def run(self):
        """ Blocks calling thread until the ModelEditor is closed. """
        self.root.mainloop()


class Simulation(SettingsPanel):
    def __init__(self, root):
        super().__init__(root)

        self.add_entry("time_step",
                valid_range = (greater_than_zero, max_float),
                default     = 0.1,
                units       = 'ms')

        self.add_entry("temperature",
                valid_range = (0, 100),
                default     = 37.0,
                units       = '°C')

        self.add_entry("initial_voltage",
                valid_range = (-max_float, max_float),
                default     = -70.0,
                units       = 'mV')

        self.add_entry("cytoplasmic_resistance",
                valid_range = (greater_than_zero, max_float),
                default     = 100.0,
                units       = '')

        self.add_entry("membrane_capacitance",
                valid_range = (greater_than_zero, max_float),
                default     = 1.0,
                units       = 'μf/cm^2')


class SegmentEditor(ManagementPanel):
    def __init__(self, root, model_editor):
        super().__init__(root, "Segment", controlled_panel="OrganizerPanel")

        self.add_button_create()
        self.add_button_delete()
        self.add_button_rename(row=1)
        self.add_button_duplicate(row=1)

        self.morphology = Morphology(self.controlled.get_widget(), model_editor)
        self.controlled.add_tab('morphology', self.morphology)

        self.mechanisms = MechanismSelector(self.controlled.get_widget(), model_editor.tabs["mechanisms"])
        self.controlled.add_tab('mechanisms', self.mechanisms)


class Neurons(ManagementPanel):
    def __init__(self, root, model_editor):
        self.segment_editor = model_editor.tabs["segments"]
        super().__init__(root, "Neuron", controlled_panel=("ManagementPanel", [],
                    {"title": "Segment", "keep_sorted": False, "controlled_panel": "OrganizerPanel"}))
        self.add_button_create()
        self.add_button_delete()
        self.add_button_rename(row=1)
        self.add_button_duplicate(row=1)

        self.segments = self.controlled
        self.segments.selector.add_button("Add", self._add_segment_to_neuron)
        self.segments.add_button_delete("Remove")
        self.segments.add_buttons_up_down(row=1)

        tab_ctrl = self.segments.controlled

        tab_ctrl.add_tab('soma', self._init_soma_settings(tab_ctrl.get_widget()))

        self.morphology = Morphology(tab_ctrl.get_widget(), model_editor, override_mode=True)
        tab_ctrl.add_tab('morphology', self.morphology)

        self.mechanisms = MechanismSelector(tab_ctrl.get_widget(), model_editor.tabs["mechanisms"])
        tab_ctrl.add_tab('mechanisms', self.mechanisms)

    def _init_soma_settings(self, parent):
        settings = SettingsPanel(parent)
        settings.add_entry("Number", tk.IntVar(),
                valid_range = (0, max_int),
                units       = 'cells')
        return settings

    def _add_segment_to_neuron(self, selected):
        seg_types = sorted(self.segment_editor.get_parameters().keys())
        dialog    = _AddSegmentToNeuron(self.segments.get_widget(), seg_types)
        selected  = dialog.selected
        if selected is None:
            return
        if selected in self.segments.parameters:
            return
        self.segments.parameters[selected] = {}
        self.segments.selector.insert(selected)

    def _set_defaults(self,):
        selected = self.segments.selector.get()
        if selected is None:
            return
        defaults = self.segment_editor.get_parameters()[selected]
        self.morphology.set_defaults(defaults["morphology"])
        self.mechanisms.set_defaults(defaults["mechanisms"])

class _AddSegmentToNeuron(simpledialog.Dialog):
    def __init__(self, parent, segment_types):
        self.selected = None
        self.segment_types = segment_types
        super().__init__(parent, "Select Segment")

    def body(self, parent):
        parent = ttk.Frame(parent)
        parent.grid(sticky='nesw')
        label = ttk.Label(parent, text="Select a segment type to\nadd to the neuron type:")
        label.grid(row=0)
        self.listbox = tk.Listbox(parent, selectmode='browse', exportselection=True)
        self.listbox.insert(0, *self.segment_types)
        self.listbox.grid(row=1, padx=padx, pady=pad_top)
        self.listbox.bind("<Double-Button-1>", self.ok)
        return self.listbox

    def validate(self):
        idx = self.listbox.curselection()
        if not idx:
            return False
        self.selected = self.segment_types[idx[0]]
        return True


class Morphology(SettingsPanel):
    def __init__(self, root, model_editor, override_mode=False):
        super().__init__(root, override_mode=override_mode)

        self.add_radio_buttons("extend_before_bifurcate", ["Dendrite", "Axon"],
                tk.BooleanVar(),
                title="")

        self.add_checkbox("competitive",
                title   = "Competitive Growth",
                default = True)

        self.add_slider("balancing_factor",
                valid_range = (0, 1))

        self.add_entry("carrier_point_density",
                valid_range = (0, max_float),
                units       = "")

        self.add_entry("maximum_segment_length",
                valid_range = (greater_than_zero, inf),
                default     = 10,
                units       = 'μm')

        self.add_dropdown("global_region", lambda: model_editor.tabs["regions"].get_parameters().keys())

        self.add_dropdown("neuron_region",
                lambda: ["None"] + list(model_editor.tabs["regions"].get_parameters().keys()),
                default = "None")

        self.add_entry("diameter",
                valid_range = (greater_than_zero, max_float),
                default     = 3,
                units       = 'μm')

        self.add_slider("extension_angle",
                title       = "Maximum Extension Angle",
                valid_range = (0, 180),
                default     = 60,
                units       = '°')

        self.add_entry("extension_distance",
                title       = "Maximum Extension Distance",
                valid_range = (0, inf),
                default     = 100,
                units       = 'μm')

        self.add_slider("bifurcation_angle",
                title       = "Maximum Branch Angle",
                valid_range = (0, 180),
                default     = 60,
                units       = '°')

        self.add_entry("bifurcation_distance",
                title       = "Maximum Branch Distance",
                valid_range = (0, inf),
                default     = 100,
                units       = 'μm')

        # grow_from (combo-list of segment types)
        # exclude_from (combo-list of segment types)

        # SOMA OPTIONS:
        # region
        # diameter
        # number to grow


if __name__ == '__main__':
    ModelEditor().run()
