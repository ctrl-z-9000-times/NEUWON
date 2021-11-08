from collections.abc import Callable, Iterable, Mapping
from neuwon.database import Database, epsilon
from neuwon.database.time import Clock
from neuwon.parameters import Parameters
from neuwon.segment import SegmentMethods
from neuwon.mechanisms import MechanismsFactory
from neuwon.species import SpeciesFactory
import cupy as cp
import numpy as np

default_simulation_parameters = {
    'time_step': 0.1,
    'celsius': 37,
    'fh_space': 300e-10, # Frankenhaeuser Hodgkin Space, in Angstroms
    'max_outside_radius':20e-6,
    'outside_volume_fraction':.20,
    'outside_tortuosity':1.55,
    'cytoplasmic_resistance': 100,
    'membrane_capacitance': 1, # uf/cm^2
    'initial_voltage': -70,
}

class Model:
    def __init__(self, time_step,
            celsius = 37,
            fh_space = 300e-10, # Frankenhaeuser Hodgkin Space, in Angstroms
            max_outside_radius=20e-6,
            outside_volume_fraction=.20,
            outside_tortuosity=1.55,
            cytoplasmic_resistance = 100,
            membrane_capacitance = 1, # uf/cm^2
            initial_voltage = -70,):
        """
        Argument cytoplasmic_resistance

        Argument outside_volume_fraction

        Argument outside_tortuosity

        Argument max_outside_radius

        Argument membrane_capacitance

        Argument initial_voltage, units: millivolts
        """
        self.species = {}
        self.reactions = {}
        self.database = db = Database()
        self.clock = Clock(time_step, units="ms")
        self.database.add_clock(self.clock)
        self.time_step = self.clock.get_tick_period()
        self.input_clock = Clock(0.5 * self.time_step, units="ms")
        self.celsius = float(celsius)
        self.Segment = SegmentMethods._initialize(db,
                initial_voltage=initial_voltage,
                cytoplasmic_resistance=cytoplasmic_resistance,
                membrane_capacitance=membrane_capacitance,)
        self.Segment._model = self
        # self.Section = ditto
        # self.Inside = ditto
        # self.Outside = ditto

        self.fh_space = float(fh_space)
        self.max_outside_radius = float(max_outside_radius)
        self.outside_tortuosity = float(outside_tortuosity)
        self.outside_volume_fraction = float(outside_volume_fraction)
        assert(self.fh_space >= epsilon * 1e-10)
        assert(self.max_outside_radius >= epsilon * 1e-6)
        assert(self.outside_tortuosity >= 1.0)
        assert(1.0 >= self.outside_volume_fraction >= 0.0)

    def __len__(self):
        return len(self.Segment.get_database_class())

    def __repr__(self):
        return repr(self.database)

    def get_database(self):
        return self.database

    def check(self):
        self.database.check()

    def add_species(self, species, *args, **kwargs) -> Species:
        """
        Accepts either a Species object or the arguments to create one.
                See 'neuwon.species.Species.__init__' for function signature.
        """
        if not isinstance(species, Species):
            species = Species(species, *args, **kwargs)
        assert species.name not in self.species
        self.species[species.name] = species
        species._initialize(self.database, self.time_step, self.celsius, self.input_clock)
        return species

    def get_species(self, species_name:str) -> Species:
        return self.species[str(species_name)]

    def get_all_species(self) -> [Species]:
        return list(self.species.values())

    def add_reaction(self, reaction: Reaction) -> Reaction:
        r = reaction
        if isinstance(r, str):
            if r.endswith(".mod"):
                from neuwon.nmodl import NmodlMechanism
                r = NmodlMechanism(r)
            else:
                raise ValueError("File extension not understood")
        if hasattr(r, "initialize"):
            retval = r.initialize(self.database,
                    time_step=self.time_step,
                    celsius=self.celsius,)
            if retval is not None: r = retval
        name = str(r.get_name())
        assert name not in self.reactions
        self.reactions[name] = r
        return r

    def get_reaction(self, reaction_name:str) -> Reaction:
        return self.reactions[str(reaction_name)]

    def get_all_reactions(self) -> [Reaction]:
        return list(self.reactions.values())

    def advance(self):
        """
        All systems (reactions & mechanisms, diffusions & electrics) are
        integrated using input values from halfway through their time step.
        Tracing through the exact sequence of operations is difficult because
        both systems see the other system as staggered halfway through their
        time step.

        For more information see: The NEURON Book, 2003.
        Chapter 4, Section: Efficient handling of nonlinearity.
        """
        self.database.sort()
        with self.database.using_memory_space('cuda'):
            self._advance_species()
            self._advance_reactions()
            self._advance_species()
        self.clock.tick()

    def _advance_lockstep(self):
        """ Naive integration strategy, for reference only. """
        self.database.sort()
        self._advance_species()
        self._advance_species()
        self._advance_reactions()
        self.clock.tick()

    def _advance_species(self):
        """ Note: Each call to this method integrates over half a time step. """
        dt = self.input_clock.get_tick_period()
        sum_conductance = self.database.get_data("Segment.sum_conductance")
        driving_voltage = self.database.get_data("Segment.driving_voltage")
        sum_conductance.fill(0.0) # Zero accumulator.
        driving_voltage.fill(0.0) # Zero accumulator.
        self.input_clock.tick()
        driving_voltage /= sum_conductance
        xp = cp.get_array_module(driving_voltage)
        driving_voltage[:] = xp.nan_to_num(driving_voltage)

        self.Segment._electric_advance(dt)

        for s in self.species.values():
            s._advance(dt)

    def _advance_reactions(self):
        for name, species in self.species.items():
            species._zero_accumulators(self.database)
        for name, r in self.reactions.items():
            try: r.advance(self)
            except Exception: raise RuntimeError("in reaction " + name)
