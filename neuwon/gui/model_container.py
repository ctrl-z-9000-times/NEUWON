from . import species_editor
from .mechanism_editor  import MechanismManager
from .neuron_editor     import SegmentEditor, NeuronEditor
from .region_editor     import RegionEditor
import os.path
import json

# TODO: Consider renaming this class & file to "ProjectContainer"?

class ModelContainer:
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
            'regions':      RegionEditor.export(        parameters["regions"]),
            'segments':     SegmentEditor.export(       parameters["segments"]),
            'neurons':      NeuronEditor.export(        parameters["neurons"]),
        }
