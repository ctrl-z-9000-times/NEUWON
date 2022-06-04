from ..control_panels import *

def SynapseEditor(root, model):
    self = ManagementPanel(root, 'Synapse')

    self.add_button_create()
    self.add_button_delete()

    self.controlled.add_entry('cleft_volume',
            valid_range = (0, inf),
            default     = 0,
            units       = 'μm^3')
    self.controlled.add_entry('cleft_spillover_area',
            valid_range = (0, inf),
            default     = 0,
            units       = 'μm^2')
    self.controlled.add_entry('maximum_distance',
            valid_range = (greater_than_zero, inf),
            default     = 10,
            units       = 'μm')
    self.controlled.add_dropdown('region',
            lambda: sorted(model.regions.get_parameters()))

    return self

def export(parameters):
    parameters['cleft'] = {
            'volume':           parameters.pop('cleft_volume'),
            'spillover_area':   parameters.pop('cleft_spillover_area'),}
    return parameters
