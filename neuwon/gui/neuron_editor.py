from .control_panels import *
from tkinter import simpledialog

class SegmentEditor(ManagementPanel):
    def __init__(self, root, model_editor):
        super().__init__(root, "Segment", controlled_panel="OrganizerPanel")

        self.add_button_create()
        self.add_button_delete()
        self.add_button_rename(row=1)
        self.add_button_duplicate(row=1)

        self.morphology = Morphology(self.controlled.get_widget(), model_editor)
        self.controlled.add_tab('morphology', self.morphology)

        self.mechanisms = MechanismSelector(self.controlled.get_widget(), model_editor.tabs["mechanisms"])
        self.controlled.add_tab('mechanisms', self.mechanisms)

    @classmethod
    def export(cls, parameters):
        return parameters

class NeuronEditor(ManagementPanel):
    def __init__(self, root, model_editor):
        self.segment_editor = model_editor.tabs["segments"]
        super().__init__(root, "Neuron", controlled_panel=("ManagementPanel", [],
                    {"title": "Segment", "keep_sorted": False, "controlled_panel": "OrganizerPanel"}))
        self.add_button_create()
        self.add_button_delete()
        self.add_button_rename(row=1)
        self.add_button_duplicate(row=1)

        self.segments = self.controlled
        self.segments.selector.add_button("Add", self._add_segment_to_neuron)
        self.segments.add_button_delete("Remove")
        self.segments.add_buttons_up_down(row=1)

        tab_ctrl = self.segments.controlled

        tab_ctrl.add_tab('soma', self._init_soma_settings(tab_ctrl.get_widget()))

        self.morphology = Morphology(tab_ctrl.get_widget(), model_editor, override_mode=True)
        tab_ctrl.add_tab('morphology', self.morphology)

        self.mechanisms = MechanismSelector(tab_ctrl.get_widget(), model_editor.tabs["mechanisms"])
        tab_ctrl.add_tab('mechanisms', self.mechanisms)

    def _init_soma_settings(self, parent):
        settings = SettingsPanel(parent)
        settings.add_entry("Number", tk.IntVar(),
                valid_range = (0, max_int),
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

    def _set_defaults(self):
        selected = self.segments.selector.get()
        if selected is None:
            return
        defaults = self.segment_editor.get_parameters()[selected]
        self.morphology.set_defaults(defaults["morphology"])
        self.mechanisms.set_defaults(defaults["mechanisms"])

    @classmethod
    def export(cls, gui_parameters:dict) -> dict:
        sim_parameters = {}
        for neuron_type, neuron_parameters in gui_parameters.items():
            instructions = []
            for segment_type, segment_parameters in neuron_parameters.items():
                segment_parameters = dict(segment_parameters)
                segment_parameters["segment_type"] = segment_type
                instructions.append(segment_parameters)
            sim_parameters[neuron_type] = instructions
        return sim_parameters

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


class Morphology(SettingsPanel):
    def __init__(self, root, model_editor, override_mode=False):
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
                valid_range = (0, max_float),
                units       = "")

        self.add_entry("maximum_segment_length",
                valid_range = (greater_than_zero, inf),
                default     = 10,
                units       = 'μm')

        self.add_dropdown("global_region", lambda: model_editor.tabs["regions"].get_parameters().keys())

        self.add_dropdown("neuron_region",
                lambda: ["None"] + list(model_editor.tabs["regions"].get_parameters().keys()),
                default = "None")

        self.add_entry("diameter",
                valid_range = (greater_than_zero, max_float),
                default     = 3,
                units       = 'μm')

        self.add_slider("extension_angle",
                title       = "Maximum Extension Angle",
                valid_range = (0, 180),
                default     = 60,
                units       = '°')

        self.add_entry("extension_distance",
                title       = "Maximum Extension Distance",
                valid_range = (0, inf),
                default     = 100,
                units       = 'μm')

        self.add_slider("bifurcation_angle",
                title       = "Maximum Branch Angle",
                valid_range = (0, 180),
                default     = 60,
                units       = '°')

        self.add_entry("bifurcation_distance",
                title       = "Maximum Branch Distance",
                valid_range = (0, inf),
                default     = 100,
                units       = 'μm')

        # grow_from (combo-list of segment types)
        # exclude_from (combo-list of segment types)

        # SOMA OPTIONS:
        # region
        # diameter
        # number to grow
