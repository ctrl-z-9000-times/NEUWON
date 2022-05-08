from .control_panels import *
import sys

maximum_float     = sys.float_info.max
greater_than_zero = np.nextafter(0, 1)

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
                valid_range = (0, maximum_float),
                units       = '')

        self.controlled.add_entry('decay_period',
                valid_range = (greater_than_zero, np.inf),
                default     = np.inf,
                units       = 'ms')

        reversal_type_var = tk.StringVar()
        self.controlled.add_radio_buttons("reversal_potential", ["Const", "Nerst", "GHK"],
                reversal_type_var,
                default = "Const")
        reversal_entrybox = self.controlled.add_entry("const_reversal_potential",
                title       = "",
                valid_range = (-maximum_float, maximum_float),
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
                valid_range = (0, maximum_float),
                units       = 'mmol')

        self.controlled.add_section("Extracellular")
        self.controlled.add_checkbox('outside_constant',
                title       = "Global Constant",
                default     = True)
        self.controlled.add_entry('outside_initial_concentration',
                title       = "Initial Concentration",
                valid_range = (0, maximum_float),
                units       = 'mmol')

    @classmethod
    def export(cls, parameters):
        sim = {}
        for name, gui in parameters.items():
            sim[name] = {}
        return sim

if __name__ == "__main__":
    root = tk.Tk()
    root.title("SpeciesEditor Test")
    SpeciesEditor(root).get_widget().grid()
    root.mainloop()
