from .gui_widgets import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog
import os.path
import json

class ModelEditor(OrganizerPanel):
    def __init__(self):
        self.root = tk.Tk()
        self.menubar = tk.Menu(self.root)
        self.root.config(menu = self.menubar)
        self.filemenu = self._init_file_menu(self.menubar)
        self._init_main_panel(self.root)
        self.new_model()

    def _init_file_menu(self, parent_menu):
        filemenu = tk.Menu(parent_menu, tearoff=False)
        parent_menu.add_cascade(label="File", menu=filemenu)

        filemenu.add_command(label="New Model", underline=0, command=self.new_model)

        filemenu.add_command(label="Open", underline=0, accelerator="Ctrl+O", command=self.open)
        self.root.bind_all("<Control-o>", self.open)

        filemenu.add_command(label="Save", underline=0, accelerator="Ctrl+S", command=self.save)
        self.root.bind_all("<Control-s>", self.save)

        filemenu.add_command(label="Save As", underline=5, accelerator="Ctrl+Shift+S", command=self.save_as)
        self.root.bind_all("<Control-S>", self.save_as)

        filemenu.add_command(label="Quit", underline=0, command=self.close)
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

    def set_title(self):
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
        self.set_title()
        self.set_parameters({})

    def open(self, event=None):
        open_filename = filedialog.askopenfilename(filetypes=[('Model File', '.json')])
        if not open_filename:
            return
        with open(open_filename, 'rt') as f:
            parameters = json.load(f)
        self.set_parameters(parameters)
        self.filename = open_filename
        self.set_title()

    def save(self, event=None):
        if not self.filename:
            self.save_as()
        else:
            self.root.focus_set() # Unfocusing triggers input validation.
            parameters = self.get_parameters() # Successfully get the parameters before truncating the output file.
            with open(self.filename, 'wt') as f:
                json.dump(parameters, f, indent=4)

    def save_as(self, event=None):
        save_as_filename = filedialog.asksaveasfilename(defaultextension='.json')
        if not save_as_filename:
            return
        self.filename = save_as_filename
        self.save()
        self.set_title()

    def close(self, event=None):
        self.root.destroy()

    def export(self):
        """ Fixup the programs internal parameters into NEUWON's parameter structure. """
        parameters = self.get_parameters()
        1/0
        return parameters


class Simulation(SettingsPanel):
    def __init__(self, root):
        super().__init__(root, True)

        self.add_entry("Time Step",
                tk.DoubleVar(self.frame, name="time_step", value=0.1),
                units='ms',)

        self.add_entry("Temperature",
                tk.DoubleVar(self.frame, name="temperature", value=37.0),
                units='°C',)

        self.add_entry("Initial Voltage",
                tk.DoubleVar(self.frame, name="initial_voltage", value=-70.0),
                units='mV',)

        self.add_entry("Cytoplasmic Resistance",
                tk.DoubleVar(self.frame, name="cytoplasmic_resistance", value=100.0),
                units='',)

        self.add_entry("Membrane Capacitance",
                tk.DoubleVar(self.frame, name="membrane_capacitance", value=1.0),
                units='μf/cm^2',)

        self.add_radio_buttons("TESTING", ["A", "B"],
                tk.StringVar(value="B"))

        self.add_checkbox("TESTING BOX", tk.BooleanVar(value=True))

        self.add_slider("TEST SLIDER", tk.DoubleVar(value=3.3), 0, 100, )


class Species(ManagementPanel):
    def __init__(self, root):
        super().__init__(root, "Species", lambda: None)

        self.add_button_create()
        self.add_button_delete()
        self.add_button_rename()

        settings = self.settings
        frame    = settings.frame

        settings.add_empty_space()

        settings.add_entry("Diffusivity",
                tk.DoubleVar(frame, name='diffusivity'),
                units='')
        settings.add_entry("Decay Period",
                tk.DoubleVar(frame, name='decay_period', value=float('inf')),
                units='ms')
        settings.add_entry("Charge",
                tk.IntVar(frame, name="charge"),
                units='e')
        settings.add_radio_buttons("Reversal Potential", 
                ["Const", "Nerst", "GHK"],
                tk.StringVar(frame, name="reversal_potential", value="Const"))
        settings.add_entry("",
                tk.DoubleVar(frame, name='const_reversal_potential'),
                units='mV')

        settings.add_empty_space()

        settings.add_checkbox("Intracellular",
                tk.BooleanVar(frame, name='inside'))
        settings.add_checkbox("Global Constant",
                tk.BooleanVar(frame, name='inside_constant'))
        settings.add_entry("Initial Concentration",
                tk.DoubleVar(frame, name='inside_initial_concentration'),
                units='mmol')

        settings.add_empty_space()

        settings.add_checkbox("Extracellular",
                tk.BooleanVar(frame, name='outside'))
        settings.add_checkbox("Global Constant",
                tk.BooleanVar(frame, name='outside_constant'))
        settings.add_entry("Initial Concentration",
                tk.DoubleVar(frame, name='outside_initial_concentration'),
                units='mmol')


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
            self.selector.insert_sorted(name)

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
        self.settings.add_entry("Magnitude", tk.DoubleVar(name='magnitude', value=1.0))

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
        dialog.bind("<Escape>", lambda event: dialog.destroy)
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
            self.parameters[x] = 1.0
            self.selector.insert_sorted(x)


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

        self.soma = SettingsPanel(tab_ctrl.get_widget())
        tab_ctrl.add_tab('soma', self.soma)

        self.morphology = Morphology(tab_ctrl.get_widget())
        tab_ctrl.add_tab('morphology', self.morphology)

        self.mechanisms = MechanismSelector(tab_ctrl.get_widget(), mechanism_manager)
        tab_ctrl.add_tab('mechanisms', self.mechanisms)


class Morphology(SettingsPanel):
    def __init__(self, root):
        super().__init__(root)

        self.add_radio_buttons("", ["Dendrite", "Axon"],
                tk.BooleanVar(self.frame, name="extend_before_bifurcate"))

        self.add_checkbox("Competitive Growth",
                tk.BooleanVar(self.frame,
                    value=True,
                    name="competitive"))

        self.add_slider("Balancing Factor",
                tk.DoubleVar(self.frame,
                    value=False,
                    name="balancing_factor"),
                0, 1)

        self.add_entry("Carrier Point Density",
                tk.DoubleVar(self.frame,
                    value=0,
                    name="carrier_point_density"))

        self.add_entry("Maximum Segment Length",
                tk.DoubleVar(self.frame,
                    value=10,
                    name="maximum_segment_length"),
                units='μm')

        self.add_slider("Maximum Extension Angle ",
                tk.DoubleVar(self.frame,
                    value=60,
                    name="extension_angle"),
                0, 180,
                units='°')

        self.add_entry("Maximum Extension Distance",
                tk.DoubleVar(self.frame,
                    value=100,
                    name="extension_distance"),
                units='μm')

        self.add_slider("Maximum Branch Angle ",
                tk.DoubleVar(self.frame,
                    value=60,
                    name="bifurcation_angle"),
                0, 180,
                units='°')

        self.add_entry("Maximum Branch Distance",
                tk.DoubleVar(self.frame,
                    value=100,
                    name="bifurcation_distance"),
                units='μm')

        self.add_entry("Diameter",
                tk.DoubleVar(self.frame,
                    value=3,
                    name="diameter"),
                units='μm')

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
