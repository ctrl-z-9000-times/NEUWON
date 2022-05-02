from .control_panels import *
from .region_editor import RegionEditor
from .species_editor import SpeciesEditor
from .mechanism_editor import MechanismManager, MechanismSelector
from ttkthemes import ThemedTk
from tkinter import filedialog, simpledialog
import os.path
import json

highest_negative = np.nextafter(0, -1)
inf = np.inf

# TODO: The rename & delete buttons need callbacks to apply the changes through
# the whole program.

# TODO: the segments need an associated region.

class ModelEditor(OrganizerPanel):
    def __init__(self):
        self.root = ThemedTk(theme='blue')
        self.root.rowconfigure(   0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self._init_menu(self.root)
        self._init_main_panel(self.root)
        self.new_model()

    def _init_menu(self, parent):
        self.menubar = tk.Menu(parent)
        parent.config(menu = self.menubar)
        self.filemenu = self._init_file_menu(self.menubar)

        self.menubar.add_command(label="Themes", command=lambda: pick_theme(self.root))

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
        self.add_tab('segments',   SegmentEditor(frame, self.tabs['mechanisms']))
        self.add_tab('neurons',    Neurons( frame, self.tabs['segments'], self.tabs['mechanisms']))
        frame.grid(sticky='nesw')

    def _set_title(self):
        title = "NEUWON Model Editor"
        if self.filename:
            filename = os.path.abspath(self.filename)
            home     = os.path.expanduser('~')
            if filename.startswith(home):
                filename = os.path.relpath(filename, home)
                filename = os.path.join('~', filename)
            title += ": " + filename
        self.root.title(title)

    def new_model(self, event=None):
        self.filename = None
        self._set_title()
        self.set_parameters({})

    def open(self, event=None):
        open_filename = filedialog.askopenfilename(title="Open Model",
                        filetypes=[('Model File', '.json')])
        if not open_filename:
            return
        with open(open_filename, 'rt') as f:
            parameters = json.load(f)
        self.set_parameters(parameters)
        self.filename = open_filename
        self._set_title()

    def save(self, event=None):
        if not self.filename:
            self.save_as()
        else:
            self.root.focus_set() # Unfocusing triggers input validation.
            parameters = self.get_parameters() # Successfully get the parameters before truncating the output file.
            with open(self.filename, 'wt') as f:
                json.dump(parameters, f, indent=4)
                f.flush()

    def save_as(self, event=None):
        save_as_filename = filedialog.asksaveasfilename(defaultextension='.json')
        if not save_as_filename:
            return
        self.filename = save_as_filename
        self.save()
        self._set_title()

    def export(self):
        parameters = self.get_NEUWON_parameters()
        export_filename = filedialog.asksaveasfilename(defaultextension='.py')
        with open(export_filename, 'wt') as f:
            json.dump(parameters, f, indent=4)
            f.flush()

    def get_NEUWON_parameters(self):
        """ Fixup the programs internal parameters into NEUWON's parameter structure. """
        return {
            'simulation':   self.tabs["simulation"].get_parameters(),
            'mechanisms':   self.tabs["mechanisms"].export(),
            'species':      self.tabs["species"].export(),
            'regions':      self.tabs["regions"].export(),
            # segments
            # neurons
        }

    def close(self, event=None):
        self.root.destroy()

    def run(self):
        """ Blocks calling thread until the ModelEditor is closed. """
        self.root.mainloop()


def pick_theme(root):
    # TODO: This should save the selected theme to a hidden file in the users
    # home folder, and automatically load the saved theme at start up.
    window, frame = Toplevel("Select a Theme")
    themes = sorted(root.get_themes())
    rows   = int(len(themes) ** .5)
    cols   = int(np.ceil(len(themes) / rows))
    for idx, name in enumerate(themes):
        def make_closure():
            current_name = name
            return lambda: root.set_theme(current_name)
        button = ttk.Button(frame, text=name, command=make_closure())
        button.grid(row=idx//cols, column=idx%cols, padx=padx, pady=pady)
    for row in range(rows):
        frame.rowconfigure(row, weight=1)
    for col in range(cols):
        frame.columnconfigure(col, weight=1)
    window.bind("<Escape>", lambda event: window.destroy())


class Simulation(SettingsPanel):
    def __init__(self, root):
        super().__init__(root)

        self.add_entry("time_step",
                valid_range = (0, inf),
                default     = 0.1,
                units       = 'ms')

        self.add_entry("temperature",
                valid_range = (0, 100),
                default     = 37.0,
                units       = '°C')

        self.add_entry("initial_voltage",
                default     = -70.0,
                units       = 'mV')

        self.add_entry("cytoplasmic_resistance",
                valid_range = (0, inf),
                default     = 100.0,
                units       = '')

        self.add_entry("membrane_capacitance",
                valid_range = (0, inf),
                default     = 1.0,
                units       = 'μf/cm^2')


class SegmentEditor(ManagementPanel):
    def __init__(self, root, mechanism_manager):
        super().__init__(root, "Segment", init_settings_panel=False)

        self.add_button_create()
        self.add_button_delete()
        self.add_button_rename(row=1)
        self.add_button_duplicate(row=1)

        tab_ctrl = OrganizerPanel(self.frame)
        self.set_settings_panel(tab_ctrl)

        self.morphology = Morphology(tab_ctrl.get_widget())
        tab_ctrl.add_tab('morphology', self.morphology)

        self.mechanisms = MechanismSelector(tab_ctrl.get_widget(), mechanism_manager)
        tab_ctrl.add_tab('mechanisms', self.mechanisms)


class Neurons(ManagementPanel):
    def __init__(self, root, segment_editor, mechanism_manager):
        self.segment_editor = segment_editor
        super().__init__(root, "Neuron", init_settings_panel=False)
        self.add_button_create()
        self.add_button_delete()
        self.add_button_rename(row=1)
        self.add_button_duplicate(row=1)

        self.segments = ManagementPanel(self.frame, "Segment", keep_sorted=False, init_settings_panel=False)
        self.set_settings_panel(self.segments)
        self.segments.selector.add_button("Add", self._add_segment_to_neuron)
        self.segments.add_button_delete("Remove")
        self.segments.add_buttons_up_down(row=1)

        tab_ctrl = OrganizerPanel(self.segments.frame)
        self.segments.set_settings_panel(tab_ctrl)

        tab_ctrl.add_tab('soma', self._init_settings_panel(tab_ctrl.get_widget()))

        self.morphology = Morphology(tab_ctrl.get_widget(), override_mode=True)
        tab_ctrl.add_tab('morphology', self.morphology)

        self.mechanisms = MechanismSelector(tab_ctrl.get_widget(), mechanism_manager)
        tab_ctrl.add_tab('mechanisms', self.mechanisms)

    def _init_settings_panel(self, parent):
        settings = SettingsPanel(parent)
        settings.add_entry("Number", tk.IntVar(),
                valid_range = (-1, inf),
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
    def __init__(self, root, override_mode=False):
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
                valid_range = (highest_negative, None),
                units       = "")

        self.add_entry("maximum_segment_length",
                valid_range = (0, None),
                default     = 10,
                units       = 'μm')

        self.add_slider("extension_angle",
                title       = "Maximum Extension Angle",
                valid_range = (0, 180),
                default     = 60,
                units       = '°')

        self.add_entry("extension_distance",
                title       = "Maximum Extension Distance",
                valid_range = (highest_negative, None),
                default     = 100,
                units       = 'μm')

        self.add_slider("bifurcation_angle",
                title       = "Maximum Branch Angle",
                valid_range = (0, 180),
                default     = 60,
                units       = '°')

        self.add_entry("bifurcation_distance",
                title       = "Maximum Branch Distance",
                valid_range = (highest_negative, None),
                default     = 100,
                units       = 'μm')

        self.add_entry("diameter",
                valid_range = (0, None),
                default     = 3,
                units       = 'μm')

        # neuron region (drop down menu?)
        # global region
        # grow_from (combo-list of segment types)
        # exclude_from (combo-list of segment types)


        # SOMA OPTIONS:
        # region
        # diameter
        # number to grow


if __name__ == '__main__':
    ModelEditor().run()
