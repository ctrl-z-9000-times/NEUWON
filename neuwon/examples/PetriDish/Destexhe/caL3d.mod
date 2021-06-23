COMMENT

High threshold Ca2+ channel

2-state kinetics with sigmoidal voltage-dependence

  C<->O

MODEL
    MODEL AUTHOR  : D.A. McCormick & J. Huguenard
    MODEL DATE    : 1992
    MODEL REF     : A model of the electrophysiological properties of 
                    thalamocortical relay neurons.
                    J Neurophysiol, 1992 Oct, 68(4):1384-400.

EXPERIMENT
    EXP AUTHOR    : Kay AR; Wong RK
    EXP DATE      : 1987
    EXP REF       : Journal of Physiology, 1987 Nov, 392:603-16.
    ANIMAL        : guinea-pig
    BRAIN REGION  : hippocampus
    CELL TYPE     : Ca1 pyramidal
    TECHNIQUE     : slices, whole-cell
    RECORDING METHOD  : voltage-clamp
    TEMPERATURE   : 20-22
 
Reference:
   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. Synthesis of models for
   excitable membranes, synaptic transmission and neuromodulation using a 
   common kinetic formalism, Journal of Computational Neuroscience 1: 
   195-230, 1994.
  (electronic copy available at http://cns.iaf.cnrs-gif.fr)

ENDCOMMENT

NEURON {
    SUFFIX caL
    USEION ca READ cai, cao WRITE gca
}

UNITS {
    F = (faraday) (coulomb)
    R = (k-mole) (joule/degC)
    (mA) = (milliamp)
    (mV) = (millivolt)
    (pS) = (picosiemens)
    (um) = (micron)
    (mM) = (milli/liter)
} 

PARAMETER {
    p    = 0.2e-6   (S/cm2)      : max permeability

    v           (mV)
    th   = 5    (mV)        : v 1/2 for on/off
    q    = 13   (mV)        : voltage dependence

    : max rates

    Ra   = 1.6  (/ms)       : open (v)
    Rb   = 0.2  (/ms)       : close (v)

    celsius     (degC)
    temp = 22   (degC)      : original temp
    q10  = 3                : temperature sensitivity
}

STATE { C O }

INITIAL {
    C = 1
}

BREAKPOINT {
    rates(v)
    SOLVE kstates METHOD cnexp
    gca = O * p
} 

KINETIC kstates {
    ~ C <-> O   (a,b)
    CONSERVE C + O = 1
}   

PROCEDURE rates(v(mV)) {
    tadj = q10 ^ ((celsius - temp)/10 (degC))

    a = Ra / (1 + exp(-(v-th)/q)) * tadj
    b = Rb / (1 + exp((v-th)/q)) * tadj
}
