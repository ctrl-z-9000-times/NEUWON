"""
NEUWON is a simulation framework for neuroscience and artificial intelligence
specializing in conductance based models. This software is a modern remake of
the NEURON simulator. It is accurate, efficient, and easy to use.
"""

class Species:
    """ """
    def __init__(self, name,
            charge = 0,
            transmembrane = False,
            reversal_potential = "nerst",
            intra_concentration = 0.0,
            extra_concentration = 0.0,
            intra_diffusivity = None,
            extra_diffusivity = None,
            intra_decay_period = float("inf"),
            extra_decay_period = float("inf"),
            intra_shells = True,
            extra_grid = None,):
        """
        If diffusivity is not given, then the concentration is constant.
        Argument reversal_potential is one of: number, "nerst", "goldman_hodgkin_katz"
        """
        self.name = str(name)
        self.charge = int(charge)
        self.transmembrane = bool(transmembrane)
        self.intra_concentration = float(intra_concentration)
        self.extra_concentration = float(extra_concentration)
        self.intra_diffusivity = float(intra_diffusivity) if intra_diffusivity is not None else None
        self.extra_diffusivity = float(extra_diffusivity) if extra_diffusivity is not None else None
        self.intra_decay_period = float(intra_decay_period)
        self.extra_decay_period = float(extra_decay_period)
        self.intra_shells = bool(intra_shells)
        self.extra_grid = None if extra_grid is None else tuple(float(x) for x in extra_grid)
        assert(self.intra_concentration >= 0.0)
        assert(self.extra_concentration >= 0.0)
        assert(self.intra_diffusivity is None or self.intra_diffusivity >= 0)
        assert(self.extra_diffusivity is None or self.extra_diffusivity >= 0)
        assert(self.intra_decay_period > 0.0)
        assert(self.extra_decay_period > 0.0)
        assert(len(self.extra_grid) == 3 and all(x > 0 for x in self.extra_grid))
        if reversal_potential == "nerst":
            self.reversal_potential = str(reversal_potential)
            self._reversal_potential_method = lambda T, i, o, v: nerst_potential(self.charge, T, i, o)
        elif reversal_potential == "goldman_hodgkin_katz":
            self.reversal_potential = str(reversal_potential)
            self._reversal_potential_method = self.goldman_hodgkin_katz
        else:
            self.reversal_potential = float(reversal_potential)
            self._reversal_potential_method = lambda T, i, o, v: self.reversal_potential

class Reaction:
    """ Abstract class for specifying reactions and mechanisms. """
    @classmethod
    def name(self):
        """ A unique name for this reaction and all of its instances. """
        raise TypeError("Abstract method called by %s."%repr(self))

    @classmethod
    def initialize(self, database):
        """ (Optional) This method is called after the Model has been created.
        This method is called on a deep copy of each Reaction object.

        Argument database is a function(name) -> value

        (Optional) Returns a new Reaction object to use in place of this one. """
        pass

    @classmethod
    def new_instances(self, database_access, locations, *args, **kw_args):
        """ """
        pass

    @classmethod
    def advance(self, database_access):
        """ Advance all instances of this reaction.

        Argument database_access is function: f(component_name) -> value
        """
        raise TypeError("Abstract method called by %s."%repr(self))

species_library = {
    "na": {
        "charge": 1,
        "transmembrane": True,
        "reversal_potential": "nerst",
        "intra_concentration":  15e-3,
        "extra_concentration": 145e-3,
    },
    "k": {
        "charge": 1,
        "transmembrane": True,
        "reversal_potential": "nerst",
        "intra_concentration": 150e-3,
        "extra_concentration":   4e-3,
    },
    "ca": {
        "charge": 2,
        "transmembrane": True,
        "reversal_potential": "goldman_hodgkin_katz",
        "intra_concentration": 70e-9,
        "extra_concentration": 2e-3,
    },
    "cl": {
        "charge": -1,
        "transmembrane": True,
        "reversal_potential": "nerst",
        "intra_concentration":  10e-3,
        "extra_concentration": 110e-3,
    },
    "glu": {
        # "extra_concentration": 1/0, # TODO!
        "extra_diffusivity": 1e-6, # TODO!
        # "extra_decay_period": 1/0, # TODO!
    },
}

reactions_library = {
    # "hh": ("neuwon/nmodl_library/hh.mod",
    #     dict(pointers={"gl": AccessHandle("L", conductance=True)},
    #          parameter_overrides = {"celsius": 6.3})),

    # "na11a": ("neuwon/nmodl_library/Balbi2017/Nav11_a.mod", {}),

    # "Kv11_13States_temperature2": ("neuwon/nmodl_library/Kv-kinetic-models/hbp-00009_Kv1.1/hbp-00009_Kv1.1__13States_temperature2/hbp-00009_Kv1.1__13States_temperature2_Kv11.mod", {}),

    # "AMPA5": ("neuwon/nmodl_library/Destexhe1994/ampa5.mod",
    #     dict(pointers={"C": AccessHandle("Glu", extra_concentration=True)})),

    # "caL": ("neuwon/nmodl_library/Destexhe1994/caL3d.mod",
    #     dict(pointers={"g": AccessHandle("ca", conductance=True)})),
}
