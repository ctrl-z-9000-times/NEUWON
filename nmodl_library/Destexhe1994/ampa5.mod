TITLE detailed model of glutamate AMPA receptors

COMMENT
-----------------------------------------------------------------------------

        Kinetic model of AMPA receptors
        ===============================

        6-state gating model:
        similar to that suggested by
        Patneau and Mayer, Neuron 6:785 (1991)
        Patneau et al, J Neurosci 13:3496 (1993)
  
        C ---- C1 -- C2 -- O
               |     |
               D1    D2

-----------------------------------------------------------------------------

  Based on voltage-clamp recordings of AMPA receptor-mediated currents in rat
  hippocampal slices (Xiang et al., J. Neurophysiol. 71: 2552-2556, 1994), this
  model was fit directly to experimental recordings in order to obtain the
  optimal values for the parameters (see Destexhe, Mainen and Sejnowski, 1996).

-----------------------------------------------------------------------------

  This mod file does not include mechanisms for the release and time course
  of transmitter; it is to be used in conjunction with a sepearate mechanism
  to describe the release of transmitter and that provides the concentration
  of transmitter in the synaptic cleft (to be connected to pointer C here).

-----------------------------------------------------------------------------

  See details in:

  Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic models of 
  synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition; 
  edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1998, pp. 1-25.

  (electronic copy available at http://cns.iaf.cnrs-gif.fr)



  Alain Destexhe and Zach Mainen, 1995

-----------------------------------------------------------------------------
ENDCOMMENT

NEURON {
        POINT_PROCESS AMPA5
        USEION glu READ extra_glu
}

UNITS {
        (nA) = (nanoamp)
        (mV) = (millivolt)
        (pS) = (picosiemens)
        (umho) = (micromho)
        (mM) = (milli/liter)
        (uM) = (micro/liter)
}

PARAMETER {
        gmax    = 500 (pS/AMPA5)    : maximal conductance

        Rb      = 13   (/mM /ms): binding 
                                : diffusion limited (DO NOT ADJUST)
        Ru1     = 0.0059  (/ms) : unbinding (1st site)
        Ru2     = 86  (/ms)     : unbinding (2nd site)          
        Rd      = 0.9   (/ms)   : desensitization
        Rr      = 0.064 (/ms)   : resensitization 
        Ro      = 2.7    (/ms)  : opening
        Rc      = 0.2    (/ms)  : closing
}

STATE {
        : Channel states (all fractions)
        C0              : unbound
        C1              : single glu bound
        C2              : double glu bound
        D1              : single glu bound, desensitized
        D2              : double glu bound, desensitized
        O               : open state 2
}

INITIAL {
        C0=1
        C1=0
        C2=0
        D1=0
        D2=0
        O=0
}

BREAKPOINT {
        SOLVE kstates METHOD sparse

        g = (1e-6) * gmax * O
        gna = 0.5 * g
        gk  = 0.5 * g
}

KINETIC kstates {
        rb = Rb * extra_glu 

        ~ C0 <-> C1     (rb,Ru1)
        ~ C1 <-> C2     (rb,Ru2)
        ~ C1 <-> D1     (Rd,Rr)
        ~ C2 <-> D2     (Rd,Rr)
        ~ C2 <-> O      (Ro,Rc)

        CONSERVE C0+C1+C2+D1+D2+O = 1
}

