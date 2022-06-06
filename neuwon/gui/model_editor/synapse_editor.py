from ..control_panels import *
from .mechanism_editor import MechanismSelector

class SynapseEditor(ManagementPanel):
    def __init__(self, root, model):
        super().__init__(root, 'Synapse', panel=OrganizerPanel)

        self.add_button_create()
        self.add_button_delete()

        settings_panel = SettingsPanel(self.get_widget())
        self.controlled.add_tab('settings', settings_panel)

        settings_panel.add_entry('cleft_volume',
                valid_range = (0, inf),
                default     = 0,
                units       = 'μm^3')
        settings_panel.add_entry('cleft_spillover_area',
                valid_range = (0, inf),
                default     = 0,
                units       = 'μm^2')
        settings_panel.add_entry('maximum_distance',
                valid_range = (greater_than_zero, inf),
                default     = 10,
                units       = 'μm')

        self.controlled.add_tab('presynapse',  AttachmentPoint(self.get_widget(), model))
        self.controlled.add_tab('postsynapse', AttachmentPoint(self.get_widget(), model))

    @classmethod
    def export(cls, parameters):
        for synapse in parameters.values():
            synapse.update(synapse.pop('settings'))
            synapse['attachment_points'] = (
                    AttachmentPoint.export(synapse.pop('presynapse')),
                    AttachmentPoint.export(synapse.pop('postsynapse')))
        return parameters

class AttachmentPoint(Panel):
    def __init__(self, root, model):
        self.model          = model
        self.frame          = ttk.Frame(root)
        self.mechanisms     = MechanismSelector(self.frame, model.mechanisms)
        self.neurons        = ListSelector(self.get_widget(), model.neurons.get_parameters())
        self.segments       = ListSelector(self.get_widget(), model.segments.get_parameters())
        mechanisms_label    = ttk.Label(self.get_widget(), text='Mechanisms')
        neurons_label       = ttk.Label(self.get_widget(), text='Neurons')
        segments_label      = ttk.Label(self.get_widget(), text='Segments')

        mechanisms_label                .grid(row=0, column=0)
        self.mechanisms.get_widget()    .grid(row=1, column=0, sticky='ns')
        neurons_label                   .grid(row=0, column=1)
        self.neurons.get_widget()       .grid(row=1, column=1, sticky='ns')
        segments_label                  .grid(row=0, column=2)
        self.segments.get_widget()      .grid(row=1, column=2, sticky='ns')
        self.frame.rowconfigure(1, weight=1)

    def get_parameters(self):
        return {
            "mechanisms": self.mechanisms.get_parameters(),
            "neurons": self.neurons.get_parameters(),
            "segments": self.segments.get_parameters(),
        }

    def set_parameters(self, parameters):
        neurons  = {n: False for n in self.model.neurons.get_parameters()}
        segments = {n: False for n in self.model.segments.get_parameters()}
        neurons .update(parameters.get("neurons", {}))
        segments.update(parameters.get("segments", {}))
        self.mechanisms.set_parameters(parameters.get("mechanisms", {}))
        self.neurons.set_parameters(neurons)
        self.segments.set_parameters(segments)

    @classmethod
    def export(cls, parameters):
        return {
            'mechanisms': MechanismSelector.export(parameters['mechanisms']),
            'constraints': {
                'neuron_types':  [k for k,v in parameters['neurons'] .items() if v],
                'segment_types': [k for k,v in parameters['segments'].items() if v],
                'maximum_share': 1
            }
        }
