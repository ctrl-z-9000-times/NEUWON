from .control_panels import *
from .mechanism_editor import MechanismSelector


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
        segment_types = sorted(self.segment_editor.get_parameters().keys())
        selected = None
        def ok_callback(event=None):
            nonlocal selected
            idx = listbox.curselection()
            if idx:
                selected = segment_types[idx[0]]
                window.destroy()
            else:
                listbox.focus_set()
                window.bell()
        # Create the widgets.
        window, frame = Toplevel("Select Segment")
        label = ttk.Label(frame, text="Select a segment type to\nadd to the neuron type:")
        listbox = tk.Listbox(frame, selectmode='browse', exportselection=True)
        listbox.insert(0, *segment_types)
        ok = ttk.Button(frame, text="Ok",     command=ok_callback,)
        no = ttk.Button(frame, text="Cancel", command=window.destroy,)
        # Arrange the widgets.
        label.grid(row=0, columnspan=2)
        listbox.grid(row=1, columnspan=2, padx=padx, pady=pad_top)
        ok.grid(row=2, column=0, padx=2*padx, pady=pad_top, sticky='ew')
        no.grid(row=2, column=1, padx=2*padx, pady=pad_top, sticky='ew')
        # 
        listbox.bind("<Double-Button-1>", ok_callback)
        listbox.bind("<Escape>", lambda event: window.destroy())
        listbox.focus_set()
        # Make the dialog window modal. This prevents user interaction with
        # any other application window until this dialog is resolved.
        window.grab_set()
        window.transient(self.get_widget())
        window.wait_window(window)
        # 
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
    def export(cls, parameters:dict) -> dict:
        for neuron_type, neuron_parameters in parameters.items():
            instructions = []
            for segment_type, segment_parameters in neuron_parameters.items():
                segment_parameters["segment_type"] = segment_type
                segment_parameters = SegmentSettings.export(segment_parameters)
                instructions.append(segment_parameters)
            parameters[neuron_type] = instructions
        return parameters


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

    @classmethod
    def export(cls, parameters):
        for segment_type, segment_parameters in parameters.items():
            parameters[segment_type] = SegmentSettings.export(segment_parameters)
        return parameters


class SegmentSettings(OrganizerPanel):
    def __init__(self, parent, model_editor, override_mode=False):
        super().__init__(parent)
        frame = self.get_widget()
        self.add_tab('morphology', MorphologyEditor(frame, model_editor,
                override_mode=override_mode))
        self.add_tab('mechanisms', MechanismSelector(frame, model_editor.mechanisms,
                override_mode=override_mode))

    def set_parameters(self, parameters):
        if "morphology_type" in parameters:
            morphology = parameters.setdefault("morphology", {})
            morphology["morphology_type"] = parameters.pop("morphology_type")
        super().set_parameters(parameters)

    @classmethod
    def export(cls, parameters):
        morphology      = parameters['morphology']
        morphology_type = morphology.pop('morphology_type')
        if morphology_type == 'Soma':
            parameters.update(parameters.pop('morphology'))
        elif morphology_type == 'Dendrite' or morphology_type == 'Axon':
            if 'diameter' in morphology:
                parameters['diameter'] = morphology.pop('diameter')
            if 'global_region' in morphology:
                parameters['region'] = morphology.pop('global_region')
            if morphology.get('neuron_region', 123) == 'None':
                morphology.pop('neuron_region')
            if 'extension_angle' in morphology:
                morphology['extension_angle'] *= np.pi / 180
            if 'bifurcation_angle' in morphology:
                morphology['bifurcation_angle'] *= np.pi / 180
        mechanisms = parameters['mechanisms']
        for k, v in mechanisms.items():
            mechanisms[k] = v['magnitude']
        return parameters


class MorphologyEditor(CustomSettingsPanel):
    def __init__(self, parent, model_editor, override_mode=False):
        super().__init__(parent, "morphology_type")
        self.model_editor = model_editor
        soma_settings = self.add_settings_panel("Soma",     override_mode=override_mode)
        dend_settings = self.add_settings_panel("Dendrite", override_mode=override_mode)
        axon_settings = self.add_settings_panel("Axon",     override_mode=override_mode)
        self._init_soma_settings(soma_settings)
        self._init_dendrite_settings(dend_settings)
        self._init_dendrite_settings(axon_settings)

    def _init_soma_settings(self, settings_panel):
        settings_panel.add_entry("number", tk.IntVar(),
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

