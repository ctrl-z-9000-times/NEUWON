from .control_panels import *
from .mechanism_editor import MechanismSelector
from tkinter import simpledialog

class SegmentEditor(ManagementPanel):
    def __init__(self, parent, model_editor):
        options_grid = {"morphology_type": [
                "Soma",
                "Dendrite",
                "Axon",
        ]}
        super().__init__(parent, "Segment", panel=(SegmentSettings, (model_editor,)))

        self.add_button_create(options_grid)
        self.add_button_delete()
        self.add_button_rename(row=1)
        self.add_button_duplicate(row=1)


class NeuronEditor(ManagementPanel):
    def __init__(self, parent, model_editor):
        self.segment_editor = model_editor.segments
        super().__init__(parent, "Neuron", panel=("ManagementPanel",
                    {"title": "Segment", "keep_sorted": False,
                    "panel": (SegmentSettings, (model_editor,), {"override_mode": True})}))
        self.add_button_create()
        self.add_button_delete()
        self.add_button_rename(row=1)
        self.add_button_duplicate(row=1)

        self.segments = self.controlled
        self.segments.selector.add_button("Add", self._add_segment_to_neuron)
        self.segments.add_button_delete("Remove")
        self.segments.add_buttons_up_down(row=1)

    def _add_segment_to_neuron(self, selected):
        seg_types = sorted(self.segment_editor.get_parameters().keys())
        dialog    = _AddSegmentToNeuron(self.segments.get_widget(), seg_types)
        selected  = dialog.selected
        if selected is None:
            return
        if selected in self.segments.parameters:
            return
        parameters = self.segment_editor.get_parameters()[selected]
        morphology = parameters["morphology"]["morphology_type"]
        self.segments.parameters[selected] = {"morphology_type": morphology}
        self.segments.selector.insert(selected)

    def _set_defaults(self):
        # TODO: Where should this be called from?
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


class SegmentSettings(OrganizerPanel):
    def __init__(self, parent, model_editor, override_mode=False):
        super().__init__(parent)
        frame = self.get_widget()
        self.add_tab('morphology', Morphology(frame, model_editor, override_mode=override_mode))
        self.add_tab('mechanisms', MechanismSelector(frame, model_editor.mechanisms,
                override_mode=override_mode))

    def set_parameters(self, parameters):
        if "morphology_type" in parameters:
            morphology = parameters.setdefault("morphology", {})
            morphology["morphology_type"] = parameters.pop("morphology_type")
        super().set_parameters(parameters)


class Morphology(CustomSettingsPanel):
    def __init__(self, parent, model_editor, override_mode=False):
        super().__init__(parent, "morphology_type")
        self.model_editor = model_editor
        soma_settings     = self.add_settings_panel("Soma",     override_mode=override_mode)
        dendrite_settings = self.add_settings_panel("Dendrite", override_mode=override_mode)
        self.add_panel("Axon", self.get_panel("Dendrite"))
        self._init_soma_settings(soma_settings)
        self._init_dendrite_settings(dendrite_settings)

    def _init_soma_settings(self, settings_panel):
        settings_panel.add_entry("Number", tk.IntVar(),
                valid_range = (0, max_int),
                units       = 'cells')

        settings_panel.add_entry("diameter",
                valid_range = (greater_than_zero, max_float),
                default     = 30,
                units       = 'μm')

        settings_panel.add_dropdown("region",
                lambda: self.model_editor.regions.get_parameters().keys())

    def _init_dendrite_settings(self, settings_panel):
        settings_panel.add_checkbox("competitive",
                title   = "Competitive Growth",
                default = True)

        settings_panel.add_slider("balancing_factor",
                valid_range = (0, 1))

        settings_panel.add_entry("carrier_point_density",
                valid_range = (0, max_float),
                units       = "")

        settings_panel.add_entry("maximum_segment_length",
                valid_range = (greater_than_zero, inf),
                default     = 10,
                units       = 'μm')

        settings_panel.add_dropdown("global_region",
                lambda: self.model_editor.regions.get_parameters().keys())

        settings_panel.add_dropdown("neuron_region",
                lambda: ["None"] + list(self.model_editor.regions.get_parameters().keys()),
                default = "None")

        settings_panel.add_entry("diameter",
                valid_range = (greater_than_zero, max_float),
                default     = 3,
                units       = 'μm')

        settings_panel.add_slider("extension_angle",
                title       = "Maximum Extension Angle",
                valid_range = (0, 180),
                default     = 60,
                units       = '°')

        settings_panel.add_entry("extension_distance",
                title       = "Maximum Extension Distance",
                valid_range = (0, inf),
                default     = 100,
                units       = 'μm')

        settings_panel.add_slider("bifurcation_angle",
                title       = "Maximum Branch Angle",
                valid_range = (0, 180),
                default     = 60,
                units       = '°')

        settings_panel.add_entry("bifurcation_distance",
                title       = "Maximum Branch Distance",
                valid_range = (0, inf),
                default     = 100,
                units       = 'μm')

        # TODO: grow_from (combo-list of segment types)
        # TODO: exclude_from (combo-list of segment types)

