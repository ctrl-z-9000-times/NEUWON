import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog, font
import bisect
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

        self.cell_builder = CellBuilder(self.tab_ctrl)
        add_tab(self.cell_builder.frame, 'Neurons')

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
        save_as_filename = filedialog.asksaveasfilename()
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
        }

    def set_parameters(self, parameters):
        self.simulation.set_parameters( parameters["simulation"])
        self.species.set_parameters(    parameters["species"])


class ControlPanel:
    """ GUI element for editing a table of parameters. """
    def __init__(self, root):
        self.frame = ttk.Frame(root)
        self.frame.grid(sticky='nesw')
        self.row_idx = 0
        self.variables = []

    def get_parameters(self):
        return {str(v): v.get() for v in self.variables}

    def set_parameters(self, parameters):
        for k,v in parameters.items():
            self.frame.setvar(k, v)

    def add_label(self, **kwargs):
        label = ttk.Label(self.frame, **kwargs)
        label.grid(row=self.row_idx, column=0, columnspan=3, sticky='nw',
                    ipadx=padx, ipady=pady)
        self.row_idx += 1

    def add_empty_space(self, size=pad_top):
        self.frame.rowconfigure(self.row_idx, minsize=size)
        self.row_idx += 1

    def add_radio_buttons(self, text, options, variable):
        self.variables.append(variable)
        label   = ttk.Label(self.frame, text=text)
        btn_row = tk.Frame(self.frame)
        for column, x in enumerate(options):
            if isinstance(variable, tk.StringVar):
                value = x
            else:
                value = column
            button = ttk.Radiobutton(btn_row, text=x, variable=variable, value=value)
            button.grid(row=0, column=column)
        label  .grid(row=self.row_idx, column=0, sticky='w', padx=padx, pady=pady)
        btn_row.grid(row=self.row_idx, column=1, sticky='w', padx=padx, pady=pady,
                columnspan=2) # No units so allow expansion into the units column.
        self.row_idx += 1

    def add_checkbox(self, text, variable):
        self.variables.append(variable)
        label  = ttk.Label(self.frame, text=text)
        button = ttk.Checkbutton(self.frame, variable=variable,)
        label .grid(row=self.row_idx, column=0, sticky='w', padx=padx, pady=pady)
        button.grid(row=self.row_idx, column=1, sticky='w', padx=padx, pady=pady)
        self.row_idx += 1

    def add_slider(self, text, variable, from_, to, units=""):
        self.variables.append(variable)
        label = ttk.Label(self.frame, text=text)
        value = ttk.Label(self.frame)
        def value_changed_callback(v):
            v = float(v)
            v = round(v, 3)
            v = str(v) + " " + units
            value.configure(text=v.ljust(5))
        scale = ttk.Scale(self.frame, variable=variable,
                from_=from_, to=to,
                command = value_changed_callback,
                orient = 'horizontal',)
        value_changed_callback(scale.get())
        # 
        label.grid(row=self.row_idx, column=0, sticky='w', padx=padx, pady=pady)
        scale.grid(row=self.row_idx, column=1, sticky='ew',pady=pady)
        value.grid(row=self.row_idx, column=2, sticky='w', padx=padx, pady=pady)
        self.row_idx += 1

    def add_entry(self, text, variable, units=""):
        self.variables.append(variable)
        label = ttk.Label(self.frame, text=text)
        entry = ttk.Entry(self.frame, textvar = variable, justify='right')
        units = ttk.Label(self.frame, text=units, justify='left')
        # 
        if isinstance(variable, tk.BooleanVar):
            validate_type = bool
        elif isinstance(variable, tk.IntVar):
            validate_type = int
        elif isinstance(variable, tk.DoubleVar):
            validate_type = float
        else:
            validate_type = lambda x: x
        value = None
        def focus_in(event):
            nonlocal value
            value = variable.get()
        def focus_out(event):
            entry.selection_clear()
            text = entry.get()
            try:
                v = validate_type(text)
            except ValueError:
                variable.set(value)
            else:
                entry.delete(0, tk.END)
                entry.insert(0, str(v))
        entry.bind('<FocusIn>', focus_in)
        entry.bind('<FocusOut>', focus_out)
        # 
        label.grid(row=self.row_idx, column=0, sticky='w', padx=padx, pady=pady)
        entry.grid(row=self.row_idx, column=1, sticky='w', pady=pady)
        units.grid(row=self.row_idx, column=2, sticky='w', padx=padx, pady=pady)
        self.row_idx += 1

class SelectionPanel:
    """ GUI element for managing lists. """
    def __init__(self, root, command, title=""):
        self.frame = ttk.Frame(root)
        self.frame.grid(sticky='nesw')
        self._on_select = command
        self._current_selection = ()
        # 
        if title:
            title = ttk.Label(self.frame, text=title)
            title.grid(row=0, column=0)
        # The button_panel is a row of buttons.
        self.button_panel = ttk.Frame(self.frame)
        self.button_panel.grid(row=1, column=0, sticky='nesw')
        self.column_idx = 0 # Index for appending buttons.
        # 
        self.listbox = tk.Listbox(self.frame, selectmode='single', exportselection=True)
        self.listbox.bind('<<ListboxSelect>>', self._callback)
        self.listbox.grid(row=2, column=0, sticky='nesw')

    def _callback(self, _, deselect=False):
        indices = self.listbox.curselection()
        items   = tuple(self.listbox.get(idx) for idx in indices)
        if items == self._current_selection:
            return
        if not items and not deselect:
            return
        self._on_select(self._current_selection, items)
        self._current_selection = items

    def touch(self):
        """ Issue an event as though the user just selected the current items. """
        self._on_select(self._current_selection, self._current_selection)

    def add_button(self, text, command):
        button = ttk.Button(self.button_panel, text=text, command=command)
        button.grid(row=1, column=self.column_idx, sticky='w', padx=padx, pady=pady)
        self.column_idx += 1

    def set(self, items):
        """ Replace the current contents of this Listbox with the given list of items. """
        self.listbox.delete(0, tk.END)
        self.listbox.insert(0, *items)
        self._callback(None, deselect=True)

    def get(self):
        return self._current_selection

    def get_all(self):
        return self.listbox.get(0, tk.END)

    def select(self, item):
        idx = self.get_all().index(item)
        self.listbox.selection_set(idx)
        self._callback(None)

    def clear(self):
        """ Deselect everything. """
        self.listbox.selection_clear(0, tk.END)
        self._callback(None, deselect=True)

    def insert_sorted(self, item):
        idx = bisect.bisect(self.get_all(), item)
        self.listbox.insert(idx, item)
        self.listbox.selection_set(idx)
        self._callback(None)

    def rename(self, old_item, new_item):
        idx = self.get_all().index(old_item)
        self.listbox.delete(idx)
        self.listbox.insert(idx, new_item)
        self.listbox.selection_set(idx)
        self._callback(None)

    def delete(self, item):
        idx = self.get_all().index(item)
        self.listbox.delete(idx)
        self._callback(None, deselect=True)


class Simulation(ControlPanel):
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
        self.frame = ttk.Frame(root)

        self.species_list = SelectionPanel(self.frame, self.select_species)
        self.species_ctrl = ControlPanel(self.frame)
        self.species_list.frame.grid(row=0, column=0, padx=padx, pady=pady, sticky='nsw')
        self.species_ctrl.frame.grid(row=0, column=1, padx=padx, pady=pady, sticky='nw')

        self.species_list.add_button("New", self.create_species)
        self.species_list.add_button("Delete", self.destroy_species)
        self.species_list.add_button("Rename", self.rename_species)

        self.init_species_control_panel()
        self._default_parameters = {str(v): v.get() for v in self.species_ctrl.variables}
        self.species_list.touch()

    def init_species_control_panel(self):

        self.species_ctrl.add_label(
                textvariable=tk.StringVar(self.frame, name="current_species_title"),
                font=font.BOLD,
                relief='raised')
        self.species_ctrl.add_empty_space()

        self.species_ctrl.add_entry("Charge", tk.IntVar(self.frame, name="charge"))
        self.species_ctrl.add_radio_buttons("Reversal Potential", 
                ["Const", "Nerst", "GHK"],
                tk.StringVar(self.frame, name="reversal_potential"))
        # TODO: constant reversal_potential

        self.species_ctrl.add_empty_space()
        self.species_ctrl.add_checkbox("Intracellular",
                tk.BooleanVar(self.frame, name='inside'))
        self.species_ctrl.add_checkbox("Global Constant",
                tk.BooleanVar(self.frame, name='inside_constant'))
        self.species_ctrl.add_entry("Initial Concentration",
                tk.DoubleVar(self.frame, name='inside_initial_concentration'),
                units='mmol')
        self.species_ctrl.add_entry("Diffusivity",
                tk.DoubleVar(self.frame, name='inside_diffusivity'))
        self.species_ctrl.add_entry("Decay Period",
                tk.DoubleVar(self.frame, name='inside_decay_period'),
                units='ms')

        self.species_ctrl.add_empty_space()
        self.species_ctrl.add_checkbox("Extracellular",
                tk.BooleanVar(self.frame, name='outside'))
        self.species_ctrl.add_checkbox("Global Constant",
                tk.BooleanVar(self.frame, name='outside_constant'))
        self.species_ctrl.add_entry("Initial Concentration",
                tk.DoubleVar(self.frame, name='outside_initial_concentration'),
                units='mmol')
        self.species_ctrl.add_entry("Diffusivity",
                tk.DoubleVar(self.frame, name='outside_diffusivity'))
        self.species_ctrl.add_entry("Decay Period",
                tk.DoubleVar(self.frame, name='outside_decay_period'),
                units='ms')

    def select_species(self, old_species, new_species):
        # Save the current parameters from the ControlPanel.
        if old_species:
            (old_species,) = old_species
            self.parameters[old_species] = self.species_ctrl.get_parameters()
        # Load the newly selected species parameters into the ControlPanel.
        if new_species:
            (new_species,) = new_species
            self.frame.setvar("current_species_title", f"Species: {new_species}")
            self.species_ctrl.set_parameters(self.parameters[new_species])
        else:
            self.frame.setvar("current_species_title", f"Species: None")
            self.species_ctrl.set_parameters(self._default_parameters)

    def create_species(self):
        species_name = simpledialog.askstring("Create Species", "Enter Species Name:")
        species_name = species_name.strip()
        if not species_name:
            return
        if species_name in self.parameters:
            self._duplicate_species_name_error(species_name)
            return
        self.parameters[species_name] = dict(self._default_parameters)
        self.species_list.insert_sorted(species_name)

    def _duplicate_species_name_error(self, species_name):
        messagebox.showerror("Species Name Error",
                f'Species "{species_name}" is already defined!')

    def destroy_species(self):
        selected = self.species_list.get()
        if not selected:
            return
        (species_name,) = selected
        confirmation = messagebox.askyesno("Confirm Delete Species",
                f"Are you sure you want to delete species '{species_name}'?")
        if not confirmation:
            return
        self.species_list.delete(species_name)
        self.parameters.pop(species_name)

    def rename_species(self):
        selected = self.species_list.get()
        if not selected:
            return
        (species_name,) = selected
        new_name = simpledialog.askstring("Rename Species",
                f'Rename Species "{species_name}" to')
        new_name = new_name.strip()
        if not new_name:
            return
        elif new_name == species_name:
            return
        elif new_name in self.parameters:
            self._duplicate_species_name_error(new_name)
            return
        self.parameters[new_name] = self.parameters[species_name]
        self.species_list.rename(species_name, new_name)
        self.parameters.pop(species_name)

    def get_parameters(self):
        self.species_list.touch()
        return self.parameters

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.species_list.set(sorted(self.parameters.keys()))


class Mechanisms:
    def __init__(self, root):
        self.parameters = {}
        self.frame = ttk.Frame(root)

        self.mech_list = SelectionPanel(self.frame, self.select_mech)
        self.mech_ctrl = ControlPanel(self.frame)
        self.mech_list.frame.grid(row=0, column=0, padx=padx, pady=pady, sticky='nsw')
        self.mech_ctrl.frame.grid(row=0, column=1, padx=padx, pady=pady, sticky='nw')

    def select_mech(self, event):
        1/0

    def get_parameters(self):
        1/0
    def set_parameters(self, parameters):
        1/0


class Regions:
    def __init__(self, root):
        self.frame = ttk.Frame(root)
        self.regions_list = SelectionPanel(self.frame)
        self.regions_ctrl = ControlPanel(self.frame)
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


class CellBuilder:
    def __init__(self, root):
        self.frame = ttk.Frame(root)
        self.frame.grid()
        self.neuron_list = SelectionPanel(self.frame, (lambda event: None), "Neuron Types")
        self.neuron_list.frame.grid(row=0, column=0)
        self.segment_list = SelectionPanel(self.frame, (lambda event: None), "Segment Types")
        self.segment_list.frame.grid(row=0, column=1)

        tab_ctrl = ttk.Notebook(self.frame)
        tab_ctrl.grid(row=0, column=2)

        tab_ctrl.add(ttk.Frame(tab_ctrl), text='Soma')

        self.morphology = Morphology(tab_ctrl)
        tab_ctrl.add(self.morphology.frame, text='Morphology')

        tab_ctrl.add(ttk.Frame(tab_ctrl), text='Mechanisms')


class Morphology(ControlPanel):
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


if __name__ == '__main__':
    ModelEditor().root.mainloop()
