from .gui_widgets import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog, font
import bisect
import math
import os.path
import pprint

padx = 5
pady = 1
pad_top = 10

class ModelEditor:
    def __init__(self):
        self.root = tk.Tk()

        self.menubar = tk.Menu(self.root)
        self.root.config(menu = self.menubar)
        self.filemenu = tk.Menu(self.menubar, tearoff=False)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.filemenu.add_command(label="New Model", command=self.new_model)
        self.filemenu.add_command(label="Open", command=self.open)
        self.filemenu.add_command(label="Save", command=self.save)
        self.filemenu.add_command(label="Save As", command=self.save_as)
        self.filemenu.add_command(label="Close", command=self.close)

        self.tab_ctrl = ttk.Notebook(self.root)
        self.tab_ctrl.grid(sticky='nesw')

        def add_tab(frame, text):
            self.tab_ctrl.add(frame, text=text,
                    sticky='nesw',
                    padding=(padx, pad_top))

        self.simulation = Simulation(self.tab_ctrl)
        add_tab(self.simulation.frame, 'Simulation')

        self.species = Species(self.tab_ctrl)
        add_tab(self.species.frame, 'Species')

        self.mechanisms = Mechanisms(self.tab_ctrl)
        add_tab(self.mechanisms.frame, 'Mechanisms')

        add_tab(tk.Frame(self.tab_ctrl), 'Regions')

        self.segments = Segments(self.tab_ctrl)
        add_tab(self.segments.frame, 'Segments')

        self.neurons = Neurons(self.tab_ctrl)
        add_tab(self.neurons.frame, 'Neurons')

        add_tab(tk.Frame(self.tab_ctrl), 'Synapses')

        add_tab(tk.Frame(self.tab_ctrl), 'Preview')

        self.new_model()

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

    def new_model(self):
        self.filename = None
        self.set_title()
        self.set_parameters({
                'simulation': {
                    "time_step": 0.1,
                    "temperature": 37.0,
                    "initial_voltage": -70.0,
                    "cytoplasmic_resistance": 100.0,
                    "membrane_capacitance": 1.0,
                },
                'species': {},
                'mechanisms': {},
        })

    def open(self):
        open_filename = filedialog.askopenfilename()
        if not open_filename:
            return
        with open(open_filename, 'rt') as f:
            parameters = f.read()
        parameters = eval(parameters)
        self.set_parameters(parameters)
        self.filename = open_filename
        self.set_title()

    def save(self):
        if not self.filename:
            self.save_as()
        parameters = self.get_parameters()
        parameters = pprint.pformat(parameters)
        with open(self.filename, 'wt') as f:
            f.write(parameters)
            f.write('\n')

    def save_as(self):
        save_as_filename = filedialog.asksaveasfilename(defaultextension='.py')
        if not save_as_filename:
            return
        self.filename = save_as_filename
        self.set_title()
        self.save()

    def close(self):
        self.root.destroy()

    def get_parameters(self):
        return {
            "simulation":   self.simulation.get_parameters(),
            "species":      self.species.get_parameters(),
            "mechanisms":   self.mechanisms.get_parameters(),
            "segments":     self.segments.get_parameters(),
            "neurons":      self.neurons.get_parameters(),
        }

    def set_parameters(self, parameters):
        parameters = dict(parameters)
        self.simulation.set_parameters(     parameters.pop("simulation", {}))
        self.species.set_parameters(        parameters.pop("species", {}))
        self.mechanisms.set_parameters(     parameters.pop("mechanisms", {}))
        self.segments.set_parameters(       parameters.pop("segments", {}))
        self.neurons.set_parameters(        parameters.pop("neurons", {}))
        assert not parameters


class Simulation(SettingsPanel):
    def __init__(self, root):
        super().__init__(root)

        self.add_entry("Time Step",
                tk.DoubleVar(self.frame, name="time_step"),
                units='ms',)

        self.add_entry("Temperature",
                tk.DoubleVar(self.frame, name="temperature"),
                units='°C',)

        self.add_entry("Initial Voltage",
                tk.DoubleVar(self.frame, name="initial_voltage"),
                units='mV',)

        self.add_entry("Cytoplasmic Resistance",
                tk.DoubleVar(self.frame, name="cytoplasmic_resistance"),
                units='',)

        self.add_entry("Membrane Capacitance",
                tk.DoubleVar(self.frame, name="membrane_capacitance"),
                units='μf/cm^2',)


class Species:
    def __init__(self, root):
        self.parameters = {}
        self.mgmt_panel = ManagementPanel(root, "Species", self.select_species)
        self.frame = self.mgmt_panel.frame

        self.mgmt_panel.selector.add_button("New", self.create_species)
        self.mgmt_panel.selector.add_button("Delete", self.destroy_species, require_selection=True)
        self.mgmt_panel.selector.add_button("Rename", self.rename_species, require_selection=True)

        self.init_species_control_panel()
        self._default_parameters = {str(v): v.get() for v in self.mgmt_panel.settings.variables}
        self._default_parameters["decay_period"] = math.inf
        self._default_parameters["reversal_potential"] = "Const"
        self.mgmt_panel.selector.touch()

    def init_species_control_panel(self):
        settings = self.mgmt_panel.settings
        frame    = settings.frame

        settings.add_empty_space()

        settings.add_entry("Diffusivity",
                tk.DoubleVar(frame, name='diffusivity'),
                units='')
        settings.add_entry("Decay Period",
                tk.DoubleVar(frame, name='decay_period'),
                units='ms')
        settings.add_entry("Charge",
                tk.IntVar(frame, name="charge"),
                units='e')
        settings.add_radio_buttons("Reversal Potential", 
                ["Const", "Nerst", "GHK"],
                tk.StringVar(frame, name="reversal_potential"))
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

    def select_species(self, old_species, new_species):
        # Save the current parameters from the SettingsPanel.
        if old_species is not None:
            self.parameters[old_species] = self.mgmt_panel.settings.get_parameters()
        # Load the newly selected species parameters into the SettingsPanel.
        if new_species is not None:
            self.mgmt_panel.settings.set_parameters(self.parameters[new_species])
        else:
            self.mgmt_panel.settings.set_parameters(self._default_parameters)

    def create_species(self, selected_species):
        species_name = simpledialog.askstring("Create Species", "Enter Species Name:")
        if species_name is None:
            return
        species_name = species_name.strip()
        if not species_name:
            return
        if species_name in self.parameters:
            self._duplicate_species_name_error(species_name)
            return
        self.parameters[species_name] = dict(self._default_parameters)
        self.mgmt_panel.selector.insert_sorted(species_name)

    def _duplicate_species_name_error(self, species_name):
        messagebox.showerror("Species Name Error",
                f'Species "{species_name}" is already defined!')

    def destroy_species(self, species_name):
        confirmation = messagebox.askyesno("Confirm Delete Species",
                f"Are you sure you want to delete species '{species_name}'?")
        if not confirmation:
            return
        self.mgmt_panel.selector.delete(species_name)
        self.parameters.pop(species_name)

    def rename_species(self, species_name):
        new_name = simpledialog.askstring("Rename Species",
                f'Rename Species "{species_name}" to')
        if new_name is None:
            return
        new_name = new_name.strip()
        if not new_name:
            return
        elif new_name == species_name:
            return
        elif new_name in self.parameters:
            self._duplicate_species_name_error(new_name)
            return
        self.parameters[new_name] = self.parameters[species_name]
        self.mgmt_panel.selector.rename(species_name, new_name)
        self.parameters.pop(species_name)

    def get_parameters(self):
        self.mgmt_panel.selector.touch()
        return self.parameters

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.mgmt_panel.selector.set(sorted(self.parameters.keys()))


class Mechanisms:
    def __init__(self, root):
        self.parameters = {}
        self.frame = ttk.Frame(root)

        self.mech_list = SelectorPanel(self.frame, self.select_mechanism)
        self.mech_ctrl = SettingsPanel(self.frame)
        self.mech_list.frame.grid(row=0, column=0, padx=padx, pady=pady, sticky='nsw')
        self.mech_ctrl.frame.grid(row=0, column=1, padx=padx, pady=pady, sticky='nw')

        self.mech_list.add_button("Import", self.import_mechanisms)
        self.mech_list.add_button("Remove", self.remove_mechanism, require_selection=True)

        self.mech_ctrl.add_label(
                textvariable=tk.StringVar(self.frame, name="current_mechanism_title"),
                font=font.BOLD,
                relief='raised')

        self.mech_ctrl.add_empty_space()

    def import_mechanisms(self, selected):
        files = filedialog.askopenfilenames(
                title="Import Mechanisms",
                filetypes=[('NMODL', '.mod')])
        for abspath in files:
            name = os.path.splitext(os.path.basename(abspath))[0]
            if name in self.parameters:
                continue
            self.parameters[name] = (abspath, {})
            self.mech_list.insert_sorted(name)

    def remove_mechanism(self, selected):
        confirmation = messagebox.askyesno("Confirm Remove Mechanism",
                f"Are you sure you want to remove mechanism '{selected}'?")
        if not confirmation:
            return
        self.mech_list.delete(selected)
        self.parameters.pop(selected)

    def select_mechanism(self, old_item, new_item):
        # Save the current parameters from the SettingsPanel.
        if old_item is not None:
            self.parameters[old_item] = self.mech_ctrl.get_parameters()
        # Load the selected mechanism into the SettingsPanel.
        if new_item is not None:
            self.frame.setvar("current_mechanism_title", f"Mechanism: {new_item}")
            self.mech_ctrl.set_parameters(self.parameters[new_item][1])
        else:
            self.frame.setvar("current_mechanism_title", f"Mechanism: None")

    def get_parameters(self):
        self.mech_list.touch()
        return self.parameters

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.mech_list.set(sorted(self.parameters.keys()))


class MechanismSelector:
    def __init__(self, root, mechanisms):
        self.parameters = {}
        self.mechanisms = mechanisms
        self.frame = ttk.Frame(root)
        self.list = SelectorPanel(self.frame, self.select_mechanism)
        self.list.add_button("Insert", self.insert_mechanism)
        self.list.add_button("Remove", self.remove_mechanism)

    def select_mechanism(self, old_item, new_item):
        pass

    def insert_mechanism(self, selected):
        pass

    def remove_mechanism(self, selected):
        pass




class Segments:
    def __init__(self, root):
        self.parameters = {}
        self.frame = ttk.Frame(root)
        self.frame.grid()
        self.segments_list = SelectorPanel(self.frame, (lambda event: None))
        self.segments_list.frame.grid(row=0, column=1)

        tab_ctrl = ttk.Notebook(self.frame)
        tab_ctrl.grid(row=0, column=2)

        tab_ctrl.add(ttk.Frame(tab_ctrl), text='Soma')

        self.morphology = Morphology(tab_ctrl)
        tab_ctrl.add(self.morphology.frame, text='Morphology')

        self.mechanisms = MechanismSelector(tab_ctrl, {})
        tab_ctrl.add(self.mechanisms.frame, text='Mechanisms')

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, parameters):
        self.parameters = parameters



class Neurons:
    def __init__(self, root):
        self.parameters = {}
        self.frame = ttk.Frame(root)
        self.frame.grid()
        self.neuron_list = SelectorPanel(self.frame, (lambda event: None))
        self.neuron_list.frame.grid(row=0, column=0)
        self.segment_list = SelectorPanel(self.frame, (lambda event: None))
        self.segment_list.frame.grid(row=0, column=1)

        tab_ctrl = ttk.Notebook(self.frame)
        tab_ctrl.grid(row=0, column=2)

        tab_ctrl.add(ttk.Frame(tab_ctrl), text='Soma')

        self.morphology = Morphology(tab_ctrl)
        tab_ctrl.add(self.morphology.frame, text='Morphology')

        tab_ctrl.add(ttk.Frame(tab_ctrl), text='Mechanisms')

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, parameters):
        self.parameters = parameters


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
        self.frame = ttk.Frame(root)
        self.regions_list = SelectorPanel(self.frame)
        self.regions_ctrl = SettingsPanel(self.frame)
        self.regions_list.frame.grid(row=0, column=0)
        self.regions_ctrl.frame.grid(row=0, column=1)
        # The problem with this is that using numbers is a terrible way to
        # specify the regions. They're spatial coordinate, I should have some
        # way to visualize where they are & what they look like. But that's
        # really complicated to implement.
        1/0

    def get_parameters(self):
        1/0

    def set_parameters(self, parameters):
        1/0


if __name__ == '__main__':
    ModelEditor().root.mainloop()
