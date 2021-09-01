import copy
import cupy as cp
import math
import neuwon.voronoi
import numba.cuda
import numpy as np
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping
from neuwon.database import *
from neuwon.database.time import Clock
import neuwon.segment
from PIL import Image, ImageFont, ImageDraw
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm

F = 96485.3321233100184 # Faraday's constant, Coulombs per Mole of electrons
R = 8.31446261815324 # Universal gas constant

_ITERATIONS_PER_TIMESTEP = 2 # model._advance calls model._advance_species this many times.

class Reaction:
    """ Abstract class for specifying reactions and mechanisms. """
    @classmethod
    def get_name(self):
        """ A unique name for this reaction and all of its instances. """
        return type(self).__name__

    @classmethod
    def initialize(self, model):
        """
        Optional method.
        This method is called after the Model has been created.
        This method is called on a deep copy of each Reaction object.

        (Optional) Returns a new Reaction object to use in place of this one. """
        pass

    @classmethod
    def advance(self, model):
        """ Advance all instances of this reaction. """
        raise TypeError("Abstract method called by %s."%repr(self))

    # TODO: Don't provide a standard library.
    #       Distribute these statements into the examples which use them.

    # _library = {
    #     "hh": ("nmodl_library/hh.mod",
    #         dict(parameter_overrides = {"celsius": 6.3})),

        # "na11a": ("neuwon/nmodl_library/Balbi2017/Nav11_a.mod", {}),

        # "Kv11_13States_temperature2": ("neuwon/nmodl_library/Kv-kinetic-models/hbp-00009_Kv1.1/hbp-00009_Kv1.1__13States_temperature2/hbp-00009_Kv1.1__13States_temperature2_Kv11.mod", {}),

        # "AMPA5": ("neuwon/nmodl_library/Destexhe1994/ampa5.mod",
        #     dict(pointers={"C": AccessHandle("Glu", outside_concentration=True)})),

        # "caL": ("neuwon/nmodl_library/Destexhe1994/caL3d.mod",
        #     dict(pointers={"g": AccessHandle("ca", conductance=True)})),
    # }

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
        self._injected_currents = _InjectedCurrents()


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

    def __str__(self):
        return str(self.database)

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

    def add_reaction(self, reaction: Reaction):
        r = reaction
        if hasattr(r, "initialize"):
            r = copy.deepcopy(r)
            retval = r.initialize(self.database, celsius=self.celsius, time_step=self.time_step)
            if retval is not None: r = retval
        name = str(r.get_name())
        assert(name not in self.reactions)
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
        dt                  = access("time_step") / 1000 / _ITERATIONS_PER_TIMESTEP
        conductances        = access("membrane/conductances")
        driving_voltages    = access("membrane/driving_voltages")
        voltages            = access("membrane/voltages")
        capacitances        = access("membrane/capacitances")
        # Accumulate the net conductances and driving voltages from each ion species' data.
        conductances.fill(0.0) # Zero accumulator.
        driving_voltages.fill(0.0) # Zero accumulator.
        for s in self.species.values():
            if not s.transmembrane: continue
            reversal_potential = s._reversal_potential(access)
            g = access("membrane/conductances/"+s.name)
            conductances += g
            driving_voltages += g * reversal_potential
        driving_voltages /= conductances
        driving_voltages[:] = cp.nan_to_num(driving_voltages)
        # Update voltages.
        exponent    = -dt * conductances / capacitances
        alpha       = cp.exp(exponent)
        diff_v      = driving_voltages - voltages
        irm         = access("membrane/diffusion")
        voltages[:] = irm.dot(driving_voltages - diff_v * alpha)
        integral_v  = dt * driving_voltages - exponent * diff_v * alpha
        # Calculate the transmembrane ion flows.
        for s in self.species.values():
            if not (s.transmembrane and s.charge != 0): continue
            if s.inside_global_const and s.outside_global_const: continue
            reversal_potential = access("membrane/reversal_potentials/"+s.name)
            g = access("membrane/conductances/"+s.name)
            millimoles = g * (dt * reversal_potential - integral_v) / (s.charge * F)
            if s.inside_diffusivity != 0:
                if s.use_shells:
                    1/0
                else:
                    volumes        = access("membrane/inside/volumes")
                    concentrations = access("membrane/inside/concentrations/"+s.name)
                    concentrations += millimoles / volumes
            if s.outside_diffusivity != 0:
                volumes = access("outside/volumes")
                s.outside.concentrations -= millimoles / self.geometry.outside_volumes
        # Update chemical concentrations with local changes and diffusion.
        for s in self.species.values():
            if not s.inside_global_const:
                x    = access("membrane/inside/concentrations/"+s.name)
                rr   = access("membrane/inside/delta_concentrations/"+s.name)
                irm  = access("membrane/inside/diffusions/"+s.name)
                x[:] = irm.dot(cp.maximum(0, x + rr * 0.5))
            if not s.outside_global_const:
                x    = access("outside/concentrations/"+s.name)
                rr   = access("outside/delta_concentrations/"+s.name)
                irm  = access("outside/diffusions/"+s.name)
                x[:] = irm.dot(cp.maximum(0, x + rr * 0.5))

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
