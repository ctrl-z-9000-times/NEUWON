"""

I want to do a full workup of the Destexhe mechanisms, just to show what it
takes to go from neuroscience specification to runnable experiment. And then
analyse how well it works.

Do a very simple & controlled single synapse example.
    axons and dendrites run perpendiclar, with a single near-touching point.



"""

from neuwon.api import *
from neuwon.nmodl import NmodlMechanism

hh = NmodlMechanism("examples/Destexhe/nmodl_files/HH2.mod",
        pointers={"gl": Pointer("L", conductance=True)})

AMPA5 = NmodlMechanism("examples/Destexhe/nmodl_files/ampa5.mod",
        pointers={"C": Pointer("Glu", extra_concentration=True)})

caL = NmodlMechanism("examples/Destexhe/nmodl_files/caL3d.mod",
        pointers={"g": Pointer("ca", conductance=True)}),



# Copy a lot of stuff from the petri dish example?

1/0



