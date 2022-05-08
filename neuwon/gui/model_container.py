from .species_editor    import SpeciesEditor
from .mechanism_editor  import MechanismManager
from .region_editor     import RegionEditor
import os.path
import json

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

    def export(self, parameters) -> dict:
        """ Fixup the programs internal parameters into NEUWON's parameter structure. """
        return {
            'simulation':   parameters["simulation"],
            'mechanisms':   MechanismManager.export(    parameters["mechanisms"]),
            'species':      SpeciesEditor.export(       parameters["species"]),
            'regions':      RegionEditor.export(        parameters["regions"]),
            'segments':     parameters["segments"],
            'neurons':      parameters["neurons"],
        }
