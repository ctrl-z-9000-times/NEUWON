from collections.abc import Callable, Iterable, Mapping
from neuwon.database import *
from neuwon.database.time import Clock
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm
import copy
import cupy as cp
import math
import neuwon.segment
import neuwon.voronoi
import numba.cuda
import numpy as np

_ITERATIONS_PER_TIMESTEP = 2 # model._advance calls model._advance_species this many times.

class Reaction:
    """ Abstract class for specifying reactions and mechanisms. """
    __slots__ = ()
    @classmethod
    def get_name(self):
        """ A unique name for this reaction and all of its instances. """
        name = getattr(self, "name", False)
        if name:
            return name
        return type(self).__name__

    @classmethod
    def initialize(self, database, time_step, celsius):
        """
        Optional method.
        This method is called after the Model has been created.
        This method is called on a deep copy of each Reaction object.

        (Optional) Returns a new Reaction object to use in place of this one. """
        pass

    @classmethod
    def advance(self):
        """ Advance all instances of this reaction. """
        raise TypeError("Abstract method called by %s."%repr(self))

class Model:
    def __init__(self, time_step,
            celsius = 37,
            fh_space = 300e-10, # Frankenhaeuser Hodgkin Space, in Angstroms
            max_outside_radius=20e-6,
            outside_volume_fraction=.20,
            outside_tortuosity=1.55,
            cytoplasmic_resistance = 1,
            # TODO: Consider switching membrane_capacitance to use NEURON's units: uf/cm^2
            membrane_capacitance = 1e-2,
            initial_voltage = -70,):
        """
        Argument cytoplasmic_resistance

        Argument outside_volume_fraction

        Argument outside_tortuosity

        Argument max_outside_radius

        Argument membrane_capacitance, units: Farads / Meter^2

        Argument initial_voltage, units: millivolts
        """
        self.species = {}
        self.reactions = {}
        self.database = db = Database()
        self.clock = Clock(time_step, units="ms")
        self.time_step = self.clock.get_tick_period()
        self.input_clock = Clock(0.5 * self.time_step, units="ms")
        self.celsius = float(celsius)
        self.T = self.celsius + 273.15 # Temperature, in Kelvins.
        self.Segment = neuwon.segment.SegmentMethods._initialize(db)
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

    def add_species(self, species):
        """
        Argument species is one of:
          * An instance of the Species class,
          * A dictionary of arguments for initializing a new instance of the Species class,
          * The species name, to be filled in from a standard library.
        """
        if isinstance(species, Mapping):
            species = Species(**species)
        elif isinstance(species, str):
            if species in Species._library: species = Species(species, **Species._library[species])
            else: raise ValueError("Unrecognized species: %s."%species)
        else: assert(isinstance(species, Species))
        assert(species.name not in self.species)
        self.species[species.name] = species
        species._initialize(self.database)

    def get_species(self, species_name):
        return self.species[str(species_name)]

    def add_reaction(self, reaction: Reaction) -> Reaction:
        r = reaction
        if hasattr(r, "initialize"):
            r = copy.deepcopy(r)
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
        self._advance_species()
        self._advance_reactions()
        self._advance_species()
        self._advance_clock()

    def _advance_lockstep(self):
        """ Naive integration strategy, for reference only. """
        self._advance_species()
        self._advance_species()
        self._advance_reactions()
        self._advance_clock()

    def _advance_species(self):
        """ Note: Each call to this method integrates over half a time step. """
        self.input_clock.tick()

        conductances        = access("membrane/conductances")
        driving_voltages    = access("membrane/driving_voltages")
        conductances.fill(0.0) # Zero accumulator.
        driving_voltages.fill(0.0) # Zero accumulator.
        for s in self.species.values():
            s._accumulate_conductances(self.database)
        driving_voltages /= conductances
        driving_voltages[:] = cp.nan_to_num(driving_voltages)

        self.Segment._electric_advance(self.time_step)

        for s in self.species.values():
            s._advance(self.time_step)

    def _advance_reactions(self):
        def zero(component_name):
            self.database.get_component(component_name).fill(0.0)
        for name, species in self.species.items():
            if species.transmembrane:
                zero("Segment.conductances_"+name)
            if species.inside_diffusivity != 0.0:
                zero(species.inside_archetype + "/delta_concentrations/"+name)
            if species.outside_diffusivity != 0.0:
                zero("outside/delta_concentrations/"+name)
        for name, r in self.reactions.items():
            try: r.advance(self)
            except Exception: raise RuntimeError("in reaction " + name)
