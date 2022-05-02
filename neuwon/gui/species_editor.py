from .control_panels import *
import numpy as np

highest_negative = np.nextafter(0, -1)
inf = np.inf

class SpeciesEditor(ManagementPanel):
    def __init__(self, root):
        super().__init__(root, "Species")

        self.add_button_create()
        self.add_button_delete()
        self.add_button_rename(row=1)

        self.settings.add_empty_space()

        self.settings.add_entry("charge", tk.IntVar(),
                valid_range = (-inf, inf),
                units       = 'e')

        self.settings.add_entry('diffusivity',
                valid_range = (highest_negative, inf),
                units       = '')

        self.settings.add_entry('decay_period',
                valid_range = (0, None),
                default     = inf,
                units       = 'ms')

        reversal_type_var = tk.StringVar()
        self.settings.add_radio_buttons("reversal_potential", ["Const", "Nerst", "GHK"],
                reversal_type_var,
                default = "Const")
        reversal_entrybox = self.settings.add_entry("const_reversal_potential",
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
        self.settings.add_checkbox('inside_constant',
                title       = "Global Constant",
                default     = True)
        self.settings.add_entry('inside_initial_concentration',
                title       = "Initial Concentration",
                valid_range = (highest_negative, inf),
                units       = 'mmol')

        self.settings.add_section("Extracellular")
        self.settings.add_checkbox('outside_constant',
                title       = "Global Constant",
                default     = True)
        self.settings.add_entry('outside_initial_concentration',
                title       = "Initial Concentration",
                valid_range = (highest_negative, inf),
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
