from .control_panels import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog
import os.path
import json
import pprint
import numpy as np

highest_negative = np.nextafter(0, -1)
inf = np.inf

class ModelEditor(OrganizerPanel):
    def __init__(self):
        self.root = tk.Tk()
        self._init_menu(self.root)
        self._init_main_panel(self.root)
        self.new_model()

    def _init_menu(self, parent):
        self.menubar = tk.Menu(parent)
        parent.config(menu = self.menubar)
        self.filemenu = self._init_file_menu(self.menubar)

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
        self.add_tab('species',    Species(frame))
        self.add_tab('segments',   Segments(frame, self.tabs['mechanisms']))
        self.add_tab('neurons',    Neurons( frame, self.tabs['mechanisms']))
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
        parameters = pprint.pformat(parameters)
        export_filename = filedialog.asksaveasfilename(defaultextension='.py')
        with open(export_filename, 'wt') as f:
            f.write(parameters)
            f.write('\n')
            f.flush()

    def get_NEUWON_parameters(self):
        """ Fixup the programs internal parameters into NEUWON's parameter structure. """
        parameters = self.get_parameters()
        1/0

    def close(self, event=None):
        self.root.destroy()


class Simulation(SettingsPanel):
    def __init__(self, root):
        super().__init__(root)

        self.add_entry("time_step", tk.DoubleVar(),
                valid_range = (0, inf),
                default     = 0.1,
                units       = 'ms')

        self.add_entry("temperature", tk.DoubleVar(),
                valid_range = (0, 100),
                default     = 37.0,
                units       = '°C')

        self.add_entry("initial_voltage", tk.DoubleVar(),
                default     = -70.0,
                units       = 'mV')

        self.add_entry("cytoplasmic_resistance", tk.DoubleVar(),
                valid_range = (0, inf),
                default     = 100.0,
                units       = '')

        self.add_entry("membrane_capacitance", tk.DoubleVar(),
                valid_range = (0, inf),
                default     = 1.0,
                units       = 'μf/cm^2')


class Species(ManagementPanel):
    def __init__(self, root):
        super().__init__(root, "Species", lambda: None)

        self.add_button_create()
        self.add_button_delete()
        self.add_button_rename()

        self.settings.add_empty_space()

        self.settings.add_entry("charge", tk.IntVar(),
                valid_range = (-inf, inf),
                units       = 'e')

        self.settings.add_entry('diffusivity', tk.DoubleVar(),
                valid_range = (highest_negative, inf),
                units       = '')

        self.settings.add_entry('decay_period', tk.DoubleVar(),
                valid_range = (0, None),
                default     = inf,
                units       = 'ms')

        reversal_type_var = tk.StringVar()
        self.settings.add_radio_buttons("reversal_potential", reversal_type_var,
                ["Const", "Nerst", "GHK"],
                default = "Const")
        reversal_entrybox = self.settings.add_entry("const_reversal_potential", tk.DoubleVar(),
                title       = "",
                valid_range = (-inf, inf),
                units       = 'mV')
        def const_entrybox_control(*args):
            if reversal_type_var.get() == "Const":
                reversal_entrybox.configure(state='enabled')
            else:
                reversal_entrybox.configure(state='readonly')
        reversal_type_var.trace_add("write", const_entrybox_control)

        self.settings.add_section("Intracellular")
        self.settings.add_checkbox('inside_constant', tk.BooleanVar(),
                title       = "Global Constant")
        self.settings.add_entry('inside_initial_concentration', tk.DoubleVar(),
                title       = "Initial Concentration",
                valid_range = (highest_negative, inf),
                units       = 'mmol')

        self.settings.add_section("Extracellular")
        self.settings.add_checkbox('outside_constant', tk.BooleanVar(),
                title       = "Global Constant")
        self.settings.add_entry('outside_initial_concentration', tk.DoubleVar(),
                title       = "Initial Concentration",
                valid_range = (highest_negative, inf),
                units       = 'mmol')


class MechanismManager(ManagementPanel):
    def __init__(self, root):
        super().__init__(root, "Mechanism")
        self.selector.add_button("Import", self.import_mechanisms)
        self.add_button_delete("Remove")
        self.add_button_rename()
        self.selector.add_button("Info",   self.info_on_mechanism,require_selection=True)

    def import_mechanisms(self, selected):
        files = filedialog.askopenfilenames(
                title="Import Mechanisms",
                filetypes=[('NMODL', '.mod')])
        for abspath in files:
            name = os.path.splitext(os.path.basename(abspath))[0]
            if name in self.parameters:
                continue
            self.parameters[name] = {'filename': abspath, 'parameters': {}}
            self.selector.insert(name)

    def info_on_mechanism(self, selected):
        info = tk.Toplevel()
        info.title(selected)

        # Hacks to make this selectable so that the user can copy-paste it.
        filename = self.parameters[selected][0]
        v = tk.StringVar(info, value=filename)
        tk.Entry(info, textvar=v, state='readonly',
        ).grid(row=0, column=0, padx=padx, pady=pady, sticky='new')

        docs = "TODO:\nScrape title & comments from NMODL file \n and display them here."
        tk.Message(info, text=docs, justify='left',
        ).grid(row=1, column=0, padx=padx, pady=pady, sticky='nw')
        v.set(filename)


class MechanismSelector(ManagementPanel):
    def __init__(self, root, mechanism_manager):
        self.mechanisms = mechanism_manager
        super().__init__(root, "Mechanism")
        self.selector.add_button("Insert", self.insert_mechanism)
        self.add_button_delete("Remove", require_confirmation=False)
        self.selector.add_button("Info", self.mechanisms.info_on_mechanism, require_selection=True)
        self.settings.add_empty_space()
        self.settings.add_entry('magnitude', tk.DoubleVar(), default=1.0)

    def insert_mechanism(self, selected):
        dialog = tk.Toplevel()
        dialog.title("Select Mechanisms to Insert")
        mechanisms = sorted(self.mechanisms.parameters)
        listbox = tk.Listbox(dialog, selectmode='extended', exportselection=True)
        listbox.grid(row=0, column=0, columnspan=2, padx=padx, pady=pad_top)
        listbox.insert(0, *mechanisms)
        selection = []
        def ok_callback():
            for idx in listbox.curselection():
                selection.append(mechanisms[idx])
            dialog.destroy()
        ok = ttk.Button(dialog, text="Ok",     command=ok_callback,)
        no = ttk.Button(dialog, text="Cancel", command=dialog.destroy,)
        ok.grid(row=1, column=0, padx=2*padx, pady=pad_top)
        no.grid(row=1, column=1, padx=2*padx, pady=pad_top)
        dialog.bind("<Escape>", lambda event: dialog.destroy())
        listbox.bind("<Double-Button-1>", lambda event: ok_callback())
        # Make the dialog window modal. This prevents user interaction with
        # any other application window until this dialog is resolved.
        dialog.focus_set()
        dialog.grab_set()
        dialog.transient(self.frame)
        dialog.wait_window(dialog)
        # 
        for x in selection:
            if x in self.parameters:
                continue
            self.parameters[x] = {}
            self.selector.insert(x)


class Segments(ManagementPanel):
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
    def __init__(self, root, mechanism_manager):
        super().__init__(root, "Neurons", init_settings_panel=False)

        self.segment_list = ManagementPanel(self.frame, "Segments")
        self.segment_list.frame.grid(row=1, column=2)

        tab_ctrl = OrganizerPanel(self.segment_list.frame)
        tab_ctrl.get_widget().grid(row=1, column=2)

        tab_ctrl.add_tab('soma', self._init_settings_panel(tab_ctrl.get_widget()))

        self.morphology = Morphology(tab_ctrl.get_widget())
        tab_ctrl.add_tab('morphology', self.morphology)

        self.mechanisms = MechanismSelector(tab_ctrl.get_widget(), mechanism_manager)
        tab_ctrl.add_tab('mechanisms', self.mechanisms)

    def _init_settings_panel(self, parent):
        settings = SettingsPanel(parent)
        return settings


class Morphology(SettingsPanel):
    def __init__(self, root):
        super().__init__(root)

        self.add_radio_buttons("extend_before_bifurcate", tk.BooleanVar(),
                ["Dendrite", "Axon"],
                title="")

        self.add_checkbox("competitive", tk.BooleanVar(),
                title   = "Competitive Growth",
                default = True)

        self.add_slider("balancing_factor", tk.DoubleVar(),
                valid_range = (0, 1))

        self.add_entry("carrier_point_density", tk.DoubleVar(),
                valid_range = (highest_negative, None),
                units       = "")

        self.add_entry("maximum_segment_length", tk.DoubleVar(),
                valid_range = (0, None),
                default     = 10,
                units       = 'μm')

        self.add_slider("extension_angle", tk.DoubleVar(),
                title       = "Maximum Extension Angle",
                valid_range = (0, 180),
                default     = 60,
                units       = '°')

        self.add_entry("extension_distance", tk.DoubleVar(),
                title       = "Maximum Extension Distance",
                valid_range = (highest_negative, None),
                default     = 100,
                units       = 'μm')

        self.add_slider("bifurcation_angle", tk.DoubleVar(),
                title       = "Maximum Branch Angle",
                valid_range = (0, 180),
                default     = 60,
                units       = '°')

        self.add_entry("bifurcation_distance", tk.DoubleVar(),
                title       = "Maximum Branch Distance",
                valid_range = (highest_negative, None),
                default     = 100,
                units       = 'μm')

        self.add_entry("diameter", tk.DoubleVar(),
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


class Regions:
    def __init__(self, root):
        1/0
        # The problem with this is that using numbers is a terrible way to
        # specify the regions. They're spatial coordinate, I should have some
        # way to visualize where they are & what they look like. But that's
        # really complicated to implement.


if __name__ == '__main__':
    ModelEditor().root.mainloop()
