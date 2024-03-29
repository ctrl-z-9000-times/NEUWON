from ..control_panels import *

def SpeciesEditor(root):
    self = ManagementPanel(root, 'Species')

    self.add_button_create()
    self.add_button_delete()
    # self.add_button_rename(row=1)

    self.controlled.add_empty_space()

    self.controlled.add_entry('charge', tk.IntVar(),
            valid_range = (-1000, 1000),
            units       = 'e')

    self.controlled.add_entry('diffusivity',
            valid_range = (0, max_float),
            units       = '')

    self.controlled.add_entry('decay_period',
            valid_range = (greater_than_zero, np.inf),
            default     = np.inf,
            units       = 'ms')

    reversal_type_var = tk.StringVar()
    self.controlled.add_radio_buttons('reversal_potential', ['Const', 'Nerst', 'GHK'],
            reversal_type_var,
            default = 'Const')
    reversal_entrybox = self.controlled.add_entry('const_reversal_potential',
            title       = '',
            valid_range = (-max_float, max_float),
            units       = 'mV')
    def const_entrybox_control(*args):
        if reversal_type_var.get() == 'Const':
            reversal_entrybox.configure(state='enabled')
            reversal_entrybox.configure(show='')
        else:
            reversal_entrybox.configure(state='readonly')
            reversal_entrybox.configure(show='*')
    reversal_type_var.trace_add('write', const_entrybox_control)

    self.controlled.add_section('Intracellular')
    self.controlled.add_checkbox('inside_global_constant',
            title       = 'Global Constant',
            default     = True)
    self.controlled.add_entry('inside_initial_concentration',
            title       = 'Initial Concentration',
            valid_range = (0, max_float),
            units       = 'mmol')

    self.controlled.add_section('Extracellular')
    self.controlled.add_checkbox('outside_global_constant',
            title       = 'Global Constant',
            default     = True)
    self.controlled.add_entry('outside_initial_concentration',
            title       = 'Initial Concentration',
            valid_range = (0, max_float),
            units       = 'mmol')

    return self

def export(parameters):
    for name, species in parameters.items():
        const_reversal_potential = species.pop('const_reversal_potential')
        if species['reversal_potential'] == 'Const':
            species['reversal_potential'] = const_reversal_potential
    return parameters
