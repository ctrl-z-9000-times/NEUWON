from .control_panels import *

zero_plus = np.nextafter(0, 1)
inf_minus = 99999999 # TODO: This should really be the max float?

class SpeciesEditor(ManagementPanel):
    def __init__(self, root):
        super().__init__(root, "Species")

        self.add_button_create()
        self.add_button_delete()
        self.add_button_rename(row=1)

        self.controlled.add_empty_space()

        self.controlled.add_entry("charge", tk.IntVar(),
                valid_range = (-1000, 1000),
                units       = 'e')

        self.controlled.add_entry('diffusivity',
                valid_range = (0, inf_minus),
                units       = '')

        self.controlled.add_entry('decay_period',
                valid_range = (zero_plus, None),
                default     = np.inf,
                units       = 'ms')

        reversal_type_var = tk.StringVar()
        self.controlled.add_radio_buttons("reversal_potential", ["Const", "Nerst", "GHK"],
                reversal_type_var,
                default = "Const")
        reversal_entrybox = self.controlled.add_entry("const_reversal_potential",
                title       = "",
                valid_range = (-inf_minus, inf_minus),
                units       = 'mV')
        def const_entrybox_control(*args):
            if reversal_type_var.get() == "Const":
                reversal_entrybox.configure(state='enabled')
            else:
                reversal_entrybox.configure(state='readonly')
        reversal_type_var.trace_add("write", const_entrybox_control)

        self.controlled.add_section("Intracellular")
        self.controlled.add_checkbox('inside_constant',
                title       = "Global Constant",
                default     = True)
        self.controlled.add_entry('inside_initial_concentration',
                title       = "Initial Concentration",
                valid_range = (0, inf_minus),
                units       = 'mmol')

        self.controlled.add_section("Extracellular")
        self.controlled.add_checkbox('outside_constant',
                title       = "Global Constant",
                default     = True)
        self.controlled.add_entry('outside_initial_concentration',
                title       = "Initial Concentration",
                valid_range = (0, inf_minus),
                units       = 'mmol')

    def export(self):
        sim = {}
        for name, gui in self.get_parameters().items():
            sim[name] = {}
        return sim

if __name__ == "__main__":
    root = tk.Tk()
    root.title("SpeciesEditor Test")
    SpeciesEditor(root).get_widget().grid()
    root.mainloop()
