"""
Synthesis of Models for Excitable Membranes,
Synaptic Transmission and Neuromodulation
Using a Common Kinetic Formalism

ALAIN DESTEXHE AND ZACHARY F. MAINEN
    The Howard Hughes Medical Institute and The Salk Institute for Biological
    Studies, Computational Neurobiology Laboratory, 10010 North Torrey Pines
    Road, La JotIa, CA 92037, USA

TERRENCE J. SEJNOWSKI
    The Howard Hughes Medical Institute and The Salk Institute for Biological
    Studies, Computational NeurobioIogy Laboratory, 10010 North Torrey Pines
    Road, La Jolla, CA 92037, USA and Dept. of Biology, University of
    California-San Diego, La JoIla, CA 92037

Journal of Computational Neuroscience, 1, 195-230 (1994)
9 1994 Kluwer Academic Publishers. Manufactured in The Netherlands.
"""
import numpy as np
import numba
from neuwon import Mechanism, Species, Real
from neuwon.mechanisms import KineticModel

um2_per_msec = (1e-6)**2 / (1e-3)
glutamate = Species("glutamate", extra_diffusivity = .1 *um2_per_msec)
rev0 = Species("rev0", transmembrane=True, reversal_potential=0)

class AMPA5_model(Mechanism):
    """ Detailed model of glutamate AMPA receptors

          Kinetic model of AMPA receptors
          ===============================

          6-state gating model:
          similar to that suggested by
          Patneau and Mayer, Neuron 6:785 (1991)
          Patneau et al, J Neurosci 13:3496 (1993)

          C ---- C1 -- C2 -- O
                 |     |
                 D1    D2

    Based on voltage-clamp recordings of AMPA receptor-mediated currents in rat
    hippocampal slices (Xiang et al., J. Neurophysiol. 71: 2552-2556, 1994), this
    model was fit directly to experimental recordings in order to obtain the
    optimal values for the parameters (see Destexhe, Mainen and Sejnowski, 1996).

    See details in:
        Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic models of
        synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition;
        edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1998, pp. 1-25.
        (electronic copy available at http://cns.iaf.cnrs-gif.fr)

    Alain Destexhe and Zach Mainen, 1995
    """
    def required_species(self):
        return [glutamate, rev0]

    def instance_dtype(self):
        return (Real, 6)

    def __init__(self,
            gmax = .5e-9,   # maximal conductance
            Rb   = 13,      # Units: 1/mM /ms   Rate of binding, diffusion limited (DO NOT ADJUST).
            Ru1  = 0.0059,  # Units: 1/ms       Rate of unbinding (1st site).
            Ru2  = 86,      # Units: 1/ms       Rate of unbinding (2nd site).
            Rd   = 0.9,     # Units: 1/ms       Rate of desensitization.
            Rr   = 0.064,   # Units: 1/ms       Rate of resensitization.
            Ro   = 2.7,     # Units: 1/ms       Rate of opening.
            Rc   = 0.2,     # Units: 1/ms       Rate of closing.
        ):
        """ Use global variable AMPA5 for instance with default values. """
        self.gmax = float(gmax)
        self.Rb   = float(Rb)
        self.Ru1  = float(Ru1)
        self.Ru2  = float(Ru2)
        self.Rd   = float(Rd)
        self.Rr   = float(Rr)
        self.Ro   = float(Ro)
        self.Rc   = float(Rc)

    def set_time_step(self, time_step):
        self.kinetics = KineticModel(
                time_step = time_step * 1e3,
                input_ranges = [(0, 100e3)],
                states = [
                    "C0",   # unbound
                    "C1",   # single glutamate bound
                    "C2",   # double glutamate bound
                    "D1",   # single glutamate bound, desensitized
                    "D2",   # double glutamate bound, desensitized
                    "O",    # open state
                ],
                initial_state = "C0",
                kinetics = [
                    ("C0", "C1", lambda glutamate: glutamate * 1e-6 * self.Rb, self.Ru1),
                    ("C1", "C2", lambda glutamate: glutamate * 1e-6 * self.Rb, self.Ru2),
                    ("C1", "D1", self.Rd, self.Rr),
                    ("C2", "D2", self.Rd, self.Rr),
                    ("C2", "O",  self.Ro, self.Rc),],
                conserve_sum = 1,)

    def new_instance(self, time_step, location, geometry, *args):
        return self.kinetics.initial_state

    def advance(self, locations, instances, time_step, reaction_inputs, reaction_outputs):
        glutamate = reaction_inputs.extra.glutamate[locations]
        instances = numba.cuda.cudadrv.devicearray.DeviceNDArray(
            instances.shape, instances.strides, np.float64,
            gpu_data=instances)
        self.kinetics.advance((glutamate,), instances)
        threads = 128
        blocks = (instances.shape[0] + (threads - 1)) // threads
        _AMPA5_output[blocks,threads](locations, instances,
                self.kinetics.states.index("O"),
                self.gmax, reaction_outputs.conductances.rev0)
@numba.cuda.jit()
def _AMPA5_output(locations, instances, open_state, gmax, conductances):
    index = numba.cuda.grid(1)
    if index >= instances.shape[0]:
        return
    location = locations[index]
    instance = instances[index]
    conductances[location] += gmax * instance[open_state]

AMPA5 = AMPA5_model()

class NMDA5(Mechanism):
    """ Detailed model of glutamate NMDA receptors

        Kinetic model of NMDA receptors
        ===============================

        5-state gating model:
        Clements & Westbrook 1991. Neuron 7: 605.
        Lester & Jahr 1992. J Neurosci 12: 635.
        Edmonds & Colquhoun 1992. Proc. R. Soc. Lond. B 250: 279.
        Hessler, Shirke & Malinow. 1993. Nature 366: 569.
        Clements et al. 1992. Science 258: 1498.

        C -- C1 -- C2 -- O
                   |
                   D

    Voltage dependence of Mg2+ block:
    Jahr & Stevens 1990. J Neurosci 10: 1830.
    Jahr & Stevens 1990. J Neurosci 10: 3178.

    Based on voltage-clamp recordings of NMDA receptor-mediated currents in rat
    hippocampal slices (Hessler et al., Nature 366: 569-572, 1993), this model 
    was fit directly to experimental recordings in order to obtain the optimal
    values for the parameters (see Destexhe, Mainen and Sejnowski, 1996).

    See details in:
        Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic models of 
        synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition; 
        edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1998, pp 1-25.
        (electronic copy available at http://cns.iaf.cnrs-gif.fr)

    Written by Alain Destexhe and Zach Mainen, 1995
    """
    species = [glutamate, rev0]
    Erev    = 0    # Units: mV      reversal potential
    gmax    = 500  # Units: pS      maximal conductance
    # Rates
    # Destexhe, Mainen & Sejnowski, 1996
    Rb  = 5e-3     # Units: 1/uM /ms    binding       
    Ru  = 12.9e-3  # Units: 1/ms        unbinding     
    Rd  = 8.4e-3   # Units: 1/ms        desensitization
    Rr  = 6.8e-3   # Units: 1/ms        resensitization 
    Ro  = 46.5e-3  # Units: 1/ms        opening
    Rc  = 73.8e-3  # Units: 1/ms        closing
    def __init__(self, time_step, location, geometry, *args):
        self.location = location
        self.kinetics = make_kinetic_table(
            name = "NMDA5",
            time_step = time_step * 1e3,
            inputs = "glutamate",
            states = [
                "C0",      # unbound
                "C1",      # single bound
                "C2",      # double bound
                "D",       # desensitized
                "O",],     # open
            initial_state = "C0",
            kinetics = [
                ("C0", "C1", lambda glut: self.Rb * (1e3) * glut, self.Ru),
                ("C1", "C2", lambda glut: self.Rb * (1e3) * glut, self.Ru),
                ("C2", "D", self.Rd, self.Rr),
                ("C2", "O", self.Ro, self.Rc),
            ],
            conserve_sum = 1,)
        self.state = self.kinetics.initial_state

    def advance(self, reaction_inputs, reaction_outputs):
        idx = self.location
        v = reaction_inputs.v[idx]
        x = reaction_inputs.extra.glutamate[idx]
        mg = 3e-6
        # B is the fraction free of Mg2+ block, from Jahr & Stevens
        B = 1 / (1 + np.exp(0.062e-3 * -v) * (mg / 3.57e-6))
        self.state = self.kinetics.advance((x,), self.state)
        reaction_outputs.conductances.rev0[idx] += self.gmax * self.state.O * B

class GABAa5(Mechanism):
    """ Detailed model of GABA-A receptors

    Kinetic model of GABA-A receptors
    =================================

    5-state gating model from Busch and Sakmann (Cold Spring Harbor
    Symp. Quant. Biol. 55: 69-80, 1990)

        C -- C1 -- C2
             |     |
             O1    O2

    Based on voltage-clamp recordings of GABAA receptor-mediated currents in rat
    hippocampal slices (Otis and Mody, Neuroscience 49: 13-32, 1992), this model
    was fit directly to experimental recordings in order to obtain the optimal
    values for the parameters (see Destexhe, Mainen and Sejnowski, 1996).

    See details in:
        Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic models of 
        synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition; 
        edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1998, pp. 1-25.
        (electronic copy available at http://cns.iaf.cnrs-gif.fr)

    Written by Alain Destexhe, Laval University, 1995
    """
    Erev    = -80 #   (mV)   # reversal potential
    gmax    = 500 # (pS) # maximal conductance
    # Rates
    # from Destexhe, Mainen and Sejnowski, 1996
    kf1 = 0.02  #   (/uM /ms)  # binding
    kf2 = 0.01  #   (/uM /ms)  # binding
    kb1 = 4.6   #  (/ms)   # unbinding
    kb2 = 9.2   #  (/ms)   # unbinding
    a1  = 3.3   #  (/ms)   # opening
    b1  = 9.8   #  (/ms)   # closing
    a2  = 10.6  #  (/ms)   # opening
    b2  = 0.41  #  (/ms)   # closing

    def __init__(self, time_step, location, geometry, *args):
        self.location = location
        self.kinetics = make_kinetic_table(
            name = "GABAa5",
            time_step = time_step * 1e3,
            inputs = "gaba",
            states = [
                "C0",      # unbound
                "C1",      # single bound
                "C2",      # double bound
                "O1",      # open
                "O2",],    # open
            initial_state = "C0",
            kinetics = [
                # f1 = kf1 * (1e3) * C 
                # f2 = kf2 * (1e3) * C 
                # ~ C0 <-> C1 (f1,kb1)
                # ~ C1 <-> C2 (f2,kb2)
                # ~ C1 <-> O1 (a1,b1)
                # ~ C2 <-> O2 (a2,b2)
            ],
            conserve_sum = 1,)
        self.state = self.kinetics.initial_state

    def advance(self, reaction_inputs, reaction_outputs):
        g = gmax * (O1+O2)
        i = (1e-6) * g * (v - Erev)

class GABAb3(Mechanism):
    """ Detailed model of GABAB receptors

    Kinetic model of GABA-B receptors
    =================================

    Detailed model of GABAB currents including nonlinear stiumulus
    dependency (fundamental to take into account for GABAB receptors)
    and precise fit to experimentally-recorded currents.

    Features:
      - peak at 100 ms; time course fit to Tom Otis' PSC
      - NONLINEAR SUMMATION (psc is much stronger with bursts)
        due to cooperativity of G-protein binding on K+ channels

    Approximations:
        - single binding site on receptor 
        - desensitization of the receptor
        - model of alpha G-protein activation (direct) of K+ channel
        - G-protein dynamics is second-order; simplified as follows:
            - saturating receptor
            - Michaelis-Menten of receptor for G-protein production
            - "resting" G-protein is in excess
            - Quasi-stat of intermediate enzymatic forms
        - binding on K+ channel is fast

    Kinetic Equations:

      dR/dt = K1 * T * (1-R-D) - K2 * R + d2 * D

      dD/dt = d1 * R - d2 * D

      dG/dt = K3 * R - K4 * G

      R : activated receptor
      T : transmitter
      G : activated G-protein
      K1,K2,K3,K4,d1,d2 = kinetic rate cst

    n activated G-protein bind to a K+ channel:

    n G + C <-> O       (Alpha,Beta)

    If the binding is fast, the fraction of open channels is given by:

    O = G^n / ( G^n + KD )

    where KD = Beta / Alpha is the dissociation constant


    Based on voltage-clamp recordings of GABAB receptor-mediated currents in rat
    hippocampal slices (Otis et al, J. Physiol. 463: 391-407, 1993), this model 
    was fit directly to experimental recordings in order to obtain the optimal
    values for the parameters (see Destexhe and Sejnowski, 1995).

    See details in:
        Destexhe, A. and Sejnowski, T.J.  G-protein activation kinetics and
        spill-over of GABA may account for differences between inhibitory
        responses in the hippocampus and thalamus.  Proc. Natl. Acad. Sci. USA 
        92: 9515-9519, 1995.

    See also: 
        Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic models of 
        synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition; 
        edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1998, pp. 1-25.

    (electronic copy available at http://cns.iaf.cnrs-gif.fr)


    Written by Alain Destexhe, Laval University, 1995
    """
    # From simplex fitting to experimental data
    # (Destexhe and Sejnowski, 1995)
    K1  = 0.66     #   (/ms mM)    # forward binding rate to receptor
    K2  = 0.020    #   (/ms)       # backward (unbinding) rate of receptor
    K3  = 0.083    #   (/ms)       # rate of G-protein production
    K4  = 0.0079   #   (/ms)      # rate of G-protein decay
    d1  = 0.017    #   (/ms)       # rate of desensitization
    d2  = 0.0053   #   (/ms)      # rate of re-sensitization
    KD  = 100           # dissociation constant of K+ channel
    n   = 4         # nb of binding sites of G-protein on K+
    Erev = -95   # (mV)        # reversal potential (E_K)
    def __init__(self, time_step, location, geometry, *args):
        self.R = 0 # fraction of activated receptor
        self.D = 0 # fraction of desensitized receptor
        self.G = 0 # fraction of activated G-protein

    def advance(self, reaction_inputs, reaction_outputs):
        # SOLVE bindkin METHOD cnexp
        # R' = K1 * C * (1-R-D) - K2 * R + d2 * D
        # D' = d1 * R - d2 * D
        # G' = K3 * R - K4 * G
        # Gn = G^n
        # g = gmax * Gn / (Gn+KD)
        # i = g*(v - Erev)
        pass

class caL(Mechanism):
    """ High threshold Ca2+ channel

    2-state kinetics with sigmoidal voltage-dependence

        C<->O

    Reference:
        Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. Synthesis of models for
        excitable membranes, synaptic transmission and neuromodulation using a 
        common kinetic formalism, Journal of Computational Neuroscience 1: 
        195-230, 1994.
        (electronic copy available at http://cns.iaf.cnrs-gif.fr)
    """
    species = []
    conductances = [rev0]
    p    = 0.2e-3  #  (cm/s)          # max permeability
    th   = 5       #  (mV)            # v 1/2 for on/off
    q    = 13      #  (mV)            # voltage dependence
    # max rates
    Ra   = 1.6     #  (/ms)           # open (v)
    Rb   = 0.2     #  (/ms)           # close (v)
    temp = 22      #  (degC)          # original temp
    q10  = 3                          # temperature sensitivity
    def __init__(self, time_step, location, geometry, *args):
        self.location = location
        self.strength = self.p * (1e-2) * geometry.surface_areas[location]
        celsius = 37
        tadj = self.q10 ** ((celsius - self.temp) / 10)
        self.kinetics = make_kinetic_table(
                name = "caL",
                time_step = time_step * 1e3,
                inputs = ["v"],
                states = ["C", "O",],
                initial_state = "C",
                kinetics = [["C", "O",
                    lambda v: self.Ra / (1 + np.exp(-(v - self.th) / self.q)) * tadj,
                    lambda v: self.Rb / (1 + np.exp((v - self.th) / self.q)) * tadj]],
                conserve_sum = 1,)
        self.state = self.kinetics.initial_state

    def advance(self, reaction_inputs, reaction_outputs):
        v  = reaction_inputs.v[self.location] * 1e3
        self.state = self.kinetics.advance((v,), self.state)
        reaction_outputs.conductances.rev0[self.location] += self.strength * self.state.O

class CaPump(Mechanism):
    """ Decay of internal calcium concentration
    
    Internal calcium concentration due to calcium currents and pump.
    Differential equations.

    Simple model of ATPase pump with 3 kinetic constants (Destexhe 92)
     Cai + P <-> CaP -> Cao + P  (k1,k2,k3)
    A Michaelis-Menten approximation is assumed, which reduces the complexity
    of the system to 2 parameters: 
       kt = <tot enzyme concentration> * k3  -> TIME CONSTANT OF THE PUMP
    kd = k2/k1 (dissociation constant)    -> EQUILIBRIUM CALCIUM VALUE
    The values of these parameters are chosen assuming a high affinity of 
    the pump to calcium and a low transport capacity (cfr. Blaustein, 
    TINS, 11: 438, 1988, and references therein).  

    Units checked using "modlunit" -> factor 10000 needed in ca entry

    VERSION OF PUMP + DECAY (decay can be viewed as simplified buffering)

    This mechanism was published in:  Destexhe, A. Babloyantz, A. and 
    Sejnowski, TJ.  Ionic mechanisms for intrinsic slow oscillations in
    thalamic relay neurons. Biophys. J. 65: 1538-1552, 1993)
    (electronic copy available at http://cns.iaf.cnrs-gif.fr)

    Written by Alain Destexhe, Salk Institute, Nov 12, 1992
    """
    depth   = .1    # Units: um         : depth of shell
    taur    = 700   # Units: ms         : rate of calcium removal
    cainf   = 1e-8  # Units: mM
    kt      = 1     # Units: mM/ms      : estimated from k3=.5, tot=.001
    kd      = 5e-4  # Units: mM         : estimated from k2=250, k1=5e5
    def __init__(self, time_step, location, geometry, *args):
        self.location = location
        1/0

    def advance(self, reaction_inputs, reaction_outputs):
        loc = self.location
        drive_channel =  - (10000) * ica / (2 * FARADAY * self.depth)
        drive_channel = max(0, drive_channel) # cannot pump inward
        cai = reaction_inputs.intra.ca[loc]
        drive_pump = -self.kt * cai / (cai + self.kd) # Michaelis-Menten
        reaction_outputs.intra.ca[loc] += drive_channel + drive_pump + (cainf-cai)/taur
        1/0

if __name__ == '__main__':
    import numpy as np
    dt = .1e-3
    k = AMPA5(dt, 0, None)
    r_in = {"glutamate": [0]}
    t = []
    g = []
    for tick in range(int(1000e-3 / dt)):
        r_out = {"rev0": [0]}
        r_in["glutamate"][0] *= np.exp(-dt / 10e-3)
        if 100e-3 < tick * dt < 100.5e-3:
            r_in["glutamate"][0] = 1e-2
        if 120e-3 < tick * dt < 120.5e-3:
            r_in["glutamate"][0] = 1e-2
        if 140e-3 < tick * dt < 140.5e-3:
            r_in["glutamate"][0] = 1e-2
        if 160e-3 < tick * dt < 160.5e-3:
            r_in["glutamate"][0] = 1e-2
        if 180e-3 < tick * dt < 180.5e-3:
            r_in["glutamate"][0] = 1e-2
        if 300e-3 < tick * dt < 300.5e-3:
            r_in["glutamate"][0] = 1e-2
        if 400e-3 < tick * dt < 400.5e-3:
            r_in["glutamate"][0] = 1e-2
        if 970e-3 < tick * dt < 970.5e-3:
            r_in["glutamate"][0] = 1e-2
        k.advance(r_in, r_out)
        t.append(tick * dt)
        g.append(r_out["rev0"][0])
        # g.append(r_in["glutamate"][0])
    print(k.state)
    import matplotlib.pyplot as plt
    plt.plot(t, g)
    plt.show()
