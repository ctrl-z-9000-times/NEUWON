from .control_panels import *
from .region_editor import RegionEditor
from .species_editor import SpeciesEditor
from neuwon.rxd.nmodl.parser import NmodlParser
import tkinter as tk
from ttkthemes import ThemedTk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog
import os.path
import json
import numpy as np

highest_negative = np.nextafter(0, -1)
inf = np.inf

class ModelEditor(OrganizerPanel):
    def __init__(self):
        self.root = ThemedTk()
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


class MechanismManager(ManagementPanel):
    def __init__(self, root):
        super().__init__(root, "Mechanism", init_settings_panel=False)
        self.selector.add_button("Import", self.import_mechanisms)
        self.add_button_delete("Remove")
        self.add_button_rename()
        self.selector.add_button("Info", self.info_on_mechanism,require_selection=True)
        self.set_settings_panel(CustomSettingsPanel(self.get_widget(), "filename"))
        self.documentation = {}

    def import_mechanisms(self, selected):
        files = filedialog.askopenfilenames(
                title="Import Mechanisms",
                filetypes=[('NMODL', '.mod')])
        for abspath in files:
            name = os.path.splitext(os.path.basename(abspath))[0]
            if name in self.parameters:
                continue
            self.parameters[name] = {'filename': abspath}
            self._make_nmodl_settings_panel(abspath)
            self.selector.insert(name)

    def set_parameters(self, parameters):
        for mech_name, mech_parameters in parameters.items():
            self._make_nmodl_settings_panel(mech_parameters["filename"])
        super().set_parameters(parameters)

    def _make_nmodl_settings_panel(self, filename):
        try:
            self.settings.get_panel(filename)
            return
        except KeyError:
            pass
        settings_panel = self.settings.add_custom_settings_panel(filename, override_mode=True)
        parser = NmodlParser(filename)
        for name, (value, units) in parser.gather_parameters().items():
            settings_panel.add_entry(name, title=name, default=value, units=units)
        name, point_process, title, description = parser.gather_documentation()
        self.documentation[filename] = title + "\n\n" + description

    def info_on_mechanism(self, selected):
        window, frame = Toplevel(selected + " Documentation")
        # Display filename in a raised box.
        filename = self.parameters[selected]["filename"]
        fn = ttk.Label(frame, text=filename, padding=padx, relief='raised')
        fn.grid(row=0, column=0, padx=padx, pady=pad_top, sticky='e')
        # Button to copy the filename to the clipboard.
        def copy_filename():
            window.clipboard_clear()
            window.clipboard_append(filename)
        copy = ttk.Button(frame, text="Copy", command=copy_filename)
        copy.grid(row=0, column=1, padx=padx, pady=pad_top, sticky='w')
        # Show documentation scraped from the NMODL file.
        docs = ttk.Label(frame, text=self.documentation[filename], justify='left', padding=padx)
        docs.grid(row=1, column=0, columnspan=2, padx=padx, pady=pady)

    def export(self):
        sim = {}
        for name, gui in self.get_parameters().items():
            gui = dict(gui)
            sim[name] = (gui.pop("filename"), gui)
        return sim


class SegmentEditor(ManagementPanel):
    def __init__(self, root, mechanism_manager):
        super().__init__(root, "Segment", init_settings_panel=False)

        self.add_button_create()
        self.add_button_delete()
        self.add_button_rename()
        self.add_button_duplicate()

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
        self.add_button_rename()
        self.add_button_duplicate()

        self.segments = ManagementPanel(self.frame, "Segment", keep_sorted=False, init_settings_panel=False)
        self.set_settings_panel(self.segments)
        self.segments.selector.add_button("Add", self._add_segment_to_neuron)
        self.segments.add_button_delete("Remove")
        self.segments.add_buttons_up_down()

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


class MechanismSelector(ManagementPanel):
    def __init__(self, root, mechanism_manager):
        self.mechanisms = mechanism_manager
        super().__init__(root, "Mechanism")
        self.selector.add_button("Insert", self.insert_mechanism)
        self.add_button_delete("Remove", require_confirmation=False)
        self.selector.add_button("Info", self.mechanisms.info_on_mechanism, require_selection=True)
        self.settings.add_empty_space()
        self.settings.add_entry('magnitude', default=1.0)

    def insert_mechanism(self, selected):
        window, frame = Toplevel("Select Mechanisms to Insert")
        mechanisms = sorted(self.mechanisms.parameters)
        listbox = tk.Listbox(frame, selectmode='extended', exportselection=True)
        listbox.grid(row=0, column=0, columnspan=2, padx=padx, pady=pad_top)
        listbox.insert(0, *mechanisms)
        selection = []
        def ok_callback():
            for idx in listbox.curselection():
                selection.append(mechanisms[idx])
            window.destroy()
        ok = ttk.Button(frame, text="Ok",     command=ok_callback,)
        no = ttk.Button(frame, text="Cancel", command=window.destroy,)
        ok.grid(row=1, column=0, padx=2*padx, pady=pad_top)
        no.grid(row=1, column=1, padx=2*padx, pady=pad_top)
        # 
        listbox.focus_set()
        listbox.bind("<Double-Button-1>", lambda event: ok_callback())
        window .bind("<Escape>", lambda event: window.destroy())
        # Make the dialog window modal. This prevents user interaction with
        # any other application window until this dialog is resolved.
        window.grab_set()
        window.transient(self.frame)
        window.wait_window(window)
        # 
        for x in selection:
            if x in self.parameters:
                continue
            self.parameters[x] = {}
            self.selector.insert(x)


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
