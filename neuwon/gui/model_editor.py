import tkinter as tk
from tkinter import ttk


class ModelEditor:
    def __init__(self):
        self.parameters = {}

        self.root = tk.Tk()
        self.root.title("NEUWON Model Editor")

        self.menubar = tk.Menu(self.root)
        self.root.config(menu = self.menubar)
        self.filemenu = tk.Menu(self.menubar, tearoff=False)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.filemenu.add_command(label="New Model", command=lambda: 1/0)
        self.filemenu.add_command(label="Open", command=lambda: 1/0)
        self.filemenu.add_command(label="Save", command=lambda: 1/0)
        self.filemenu.add_command(label="Save as", command=lambda: 1/0)
        self.filemenu.add_command(label="Close", command=lambda: 1/0)

        self.tab_ctrl = ttk.Notebook(self.root)
        self.tab_ctrl.grid(sticky='nesw')

        self.simulation = Simulation(self.tab_ctrl)
        self.tab_ctrl.add(self.simulation.frame, text='Simulation')

        self.species = Species(self.tab_ctrl)
        self.tab_ctrl.add(self.species.frame, text='Species')

        self.tab_ctrl.add(tk.Frame(self.tab_ctrl), text='Regions')

        self.cell_builder = CellBuilder(self.tab_ctrl)
        self.tab_ctrl.add(self.cell_builder.frame, text='Neurons')

        self.tab_ctrl.add(tk.Frame(self.tab_ctrl), text='Synapses')

        self.tab_ctrl.add(tk.Frame(self.tab_ctrl), text='Preview')




class ControlPanel:
    def __init__(self, root):
        self.frame = ttk.Frame(root)
        self.padx = 5
        self.pady = 1
        self.frame.grid(sticky='nesw')
        self.row_idx = 0

    def add_empty_space(self, size=10):
        self.frame.rowconfigure(self.row_idx, minsize=size)
        self.row_idx += 1

    def add_checkbox(self, text, variable):
        label  = ttk.Label(self.frame, text=text)
        button = ttk.Checkbutton(self.frame, variable=variable,)
        label .grid(row=self.row_idx, column=0, sticky='w', padx=self.padx, pady=self.pady)
        button.grid(row=self.row_idx, column=1, sticky='w', padx=self.padx, pady=self.pady)
        self.row_idx += 1

    def add_slider(self, text, variable, from_, to, units=""):
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
        label.grid(row=self.row_idx, column=0, sticky='w', padx=self.padx, pady=self.pady)
        scale.grid(row=self.row_idx, column=1, sticky='ew',pady=self.pady)
        value.grid(row=self.row_idx, column=2, sticky='w', padx=self.padx, pady=self.pady)
        self.row_idx += 1

    def add_number_entry(self, text, variable, units=""):
        label = ttk.Label(self.frame, text=text)
        entry = tk.Entry(self.frame, textvar = variable, justify='right')
        units = ttk.Label(self.frame, text=units, justify='left')
        # 
        label.grid(row=self.row_idx, column=0, sticky='w', padx=self.padx, pady=self.pady)
        entry.grid(row=self.row_idx, column=1, sticky='w', pady=self.pady)
        units.grid(row=self.row_idx, column=2, sticky='w', padx=self.padx, pady=self.pady)
        self.row_idx += 1


class Simulation(ControlPanel):
    def __init__(self, root):
        super().__init__(root)
        self.add_empty_space()

        self.time_step = tk.DoubleVar(self.frame, 0.1)
        self.add_number_entry("Time Step", self.time_step, units='ms')

        self.temperature = tk.DoubleVar(self.frame, 37.0)
        self.add_number_entry("Temperature", self.temperature, units='°C')

        self.initial_voltage = tk.DoubleVar(self.frame, -70.0)
        self.add_number_entry("Initial Voltage", self.initial_voltage, units='mV')

        self.resistance = tk.DoubleVar(self.frame, 100.0)
        self.add_number_entry("Cytoplasmic Resistance", self.resistance, units='')

        self.capacitance = tk.DoubleVar(self.frame, 1.0)
        self.add_number_entry("Membrane Capacitance", self.capacitance, units='μf/cm^2')

        # TODO: Load mechanisms from file?


class Species:
    def __init__(self, root):
        self.frame = ttk.Frame(root)

        self.species_list = tk.Listbox(self.frame)
        self.species_list.grid(row=0, column=0)

        self.species_ctrl = ControlPanel(self.frame)
        self.species_ctrl.frame.grid(row=0, column=1)

        self.reversal = tk.DoubleVar(self.frame)
        self.species_ctrl.add_number_entry("Reversal Potential", self.reversal)
        # name
        # charge
        # decay_period

        # inside diffusivity
        # inside initial_concentration
        # inside constant

        # outside diffusivity
        # outside initial_concentration
        # outside constant


class NeuriteSelector:
    def __init__(self, root):
        self.frame = ttk.Frame(root)
        self.frame.grid()

        self.neurons = tk.Listbox(self.frame)
        ttk.Label(self.frame, text="Neuron Types").grid(row=1, column=0)
        self.neurons.grid(row=2, column=0)

        self.segments = tk.Listbox(self.frame)
        ttk.Label(self.frame, text="Segment Types").grid(row=1, column=1)
        self.segments.grid(row=2, column=1)




class CellBuilder:
    def __init__(self, root):
        self.frame = ttk.Frame(root)
        self.frame.grid()
        self.selector = NeuriteSelector(self.frame)
        self.selector.frame.grid(row=0, column=0)

        tab_ctrl = ttk.Notebook(self.frame)
        tab_ctrl.grid(row=0, column=1)

        tab_ctrl.add(ttk.Frame(tab_ctrl), text='Soma')

        self.morphology = Morphology(tab_ctrl)
        tab_ctrl.add(self.morphology.frame, text='Morphology')

        tab_ctrl.add(ttk.Frame(tab_ctrl), text='Mechanisms')



class Morphology(ControlPanel):
    def __init__(self, root):
        super().__init__(root)

        neurite_type_panel = tk.Frame(self.frame)
        neurite_type_panel.grid(row=self.row_idx, column=0)
        self.extend_before_bifurcate = tk.BooleanVar(self.frame, False)
        ttk.Radiobutton(neurite_type_panel, text="Dendrite",
                variable=self.extend_before_bifurcate,
                value=False).grid(row=0, column=0)
        ttk.Radiobutton(neurite_type_panel, text="Axon",
                variable=self.extend_before_bifurcate,
                value=True).grid(row=0, column=1)
        self.row_idx += 1

        self.competitive = tk.BooleanVar(self.frame, True)
        self.add_checkbox("Competitive Growth", self.competitive)

        self.balancing_factor = tk.DoubleVar(self.frame, False)
        self.add_slider("Balancing Factor", self.balancing_factor, 0, 1)

        self.carrier_point_density = tk.DoubleVar(self.frame, 0)
        self.add_number_entry("Carrier Point Density", self.carrier_point_density)

        self.max_segment_length = tk.DoubleVar(self.frame, 10)
        self.add_number_entry("Maximum Segment Length", self.max_segment_length, units='μm')

        self.extension_angle = tk.DoubleVar(self.frame, 60)
        self.add_slider("Maximum Extension Angle ", self.extension_angle, 0, 180, units='°')

        self.extension_distance = tk.DoubleVar(self.frame, 100)
        self.add_number_entry("Maximum Extension Distance", self.extension_distance, units='μm')

        self.branch_angle = tk.DoubleVar(self.frame, 60)
        self.add_slider("Maximum Branch Angle ", self.branch_angle, 0, 180, units='°')

        self.branch_distance = tk.DoubleVar(self.frame, 100)
        self.add_number_entry("Maximum Branch Distance", self.branch_distance, units='μm')

        self.diameter = tk.DoubleVar(self.frame, 3)
        self.add_number_entry("Diameter", self.diameter, units='μm')

        # neuron region (drop down menu?)
        # global region
        # grow_from (combo-list of segment types)
        # exclude_from (combo-list of segment types)


        # SOMA OPTIONS:
        # region
        # diameter
        # number to grow

    def set_parameters(self, parameters):
        1/0

    def get_parameters(self):
        return {
            'diameter': self.diameter.get(),
            'morphology': {
                'extend_before_bifurcate':  self.extend_before_bifurcate.get(),
                'balancing_factor':         self.balancing_factor.get(),
                'carrier_point_density':    self.carrier_point_density.get(),
                'maximum_segment_length':   self.max_segment_length.get(),
                'extension_angle':          self.extension_angle.get(),
                'extension_distance':       self.extension_distance.get(),
                'bifurcation_angle':        self.branch_angle.get(),
                'bifurcation_distance':     self.branch_distance.get(),
            },
        }


if __name__ == '__main__':
    ModelEditor().root.mainloop()
