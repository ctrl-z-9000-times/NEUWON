from .model_editor import species_editor
from .model_editor import region_editor
from .model_editor.mechanism_editor  import MechanismManager
from .model_editor.neuron_editor     import SegmentEditor, NeuronEditor
from .model_editor.synapse_editor    import SynapseEditor
import os.path
import json

class ProjectContainer:
    def __init__(self, filename=None):
        self.set_file(filename)

    def set_file(self, filename):
        if filename is None:
            self.filename   = None
            self.short_name = None
        else:
            self.filename = os.path.abspath(filename)
            home = os.path.expanduser('~')
            if self.filename.startswith(home):
                self.short_name = os.path.relpath(self.filename, home)
                self.short_name = os.path.join('~', self.short_name)
            else:
                self.short_name = self.filename

    def save(self, parameters: dict):
        with open(self.filename, 'wt') as f:
            json.dump(parameters, f, indent=4)
            f.flush()

    def load(self) -> dict:
        with open(self.filename, 'rt') as f:
            return json.load(f)

    def export(self) -> dict:
        """ Fixup the programs internal parameters into NEUWON's parameter structure. """
        parameters = self.load()
        return {
            'simulation':   parameters["simulation"],
            'mechanisms':   MechanismManager.export(    parameters["mechanisms"]),
            'species':      species_editor.export(      parameters["species"]),
            'regions':      region_editor.export(       parameters["regions"]),
            'segments':     SegmentEditor.export(       parameters["segments"]),
            'neurons':      NeuronEditor.export(        parameters["neurons"]),
            'synapses':     SynapseEditor.export(       parameters["synapses"]),
        }
