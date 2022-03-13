TITLE Hippocampal HH channels

COMMENT

Fast Na+ and K+ currents responsible for action potentials
Iterative equations

Equations modified by Traub, for Hippocampal Pyramidal cells, in:
Traub & Miles, Neuronal Networks of the Hippocampus, Cambridge, 1991

range variable vtraub adjust threshold

Written by Alain Destexhe, Salk Institute, Aug 1992

Modified Oct 96 for compatibility with Windows: trap low values of arguments

ENDCOMMENT

NEURON {
        SUFFIX hh
        USEION na READ ena WRITE gna
        USEION k READ ek WRITE gk
}

UNITS {
        (mV) = (millivolt)
}

PARAMETER {
        gnabar  = .003  (mho/cm2)
        gkbar   = .005  (mho/cm2)

        ena     = 50    (mV)
        ek      = -90   (mV)
        celsius = 36    (degC)
        dt              (ms)
        v               (mV)
        vtraub  = -63   (mV)
}

STATE {
        m h n
}

BREAKPOINT {
        SOLVE states METHOD cnexp
        gna = gnabar * m*m*m*h
        gk  = gkbar * n*n*n*n
}


DERIVATIVE states {   : exact Hodgkin-Huxley equations
       evaluate_fct(v)
       m' = (m_inf - m) / tau_m
       h' = (h_inf - h) / tau_h
       n' = (n_inf - n) / tau_n
}

UNITSOFF
INITIAL {
:
:  Q10 was assumed to be 3 for both currents
:
        tadj = 3.0 ^ ((celsius-36)/ 10 )

        m = 0
        h = 0
        n = 0
}

PROCEDURE evaluate_fct(v(mV)) {

        v2 = v - vtraub : convert to traub convention

        a = 0.32 * vtrap(13-v2, 4)
        b = 0.28 * vtrap(v2-40, 5)
        tau_m = 1 / (a + b) / tadj
        m_inf = a / (a + b)

        a = 0.128 * exp((17-v2)/18)
        b = 4 / ( 1 + exp((40-v2)/5) )
        tau_h = 1 / (a + b) / tadj
        h_inf = a / (a + b)

        a = 0.032 * vtrap(15-v2, 5)
        b = 0.5 * exp((10-v2)/40)
        tau_n = 1 / (a + b) / tadj
        n_inf = a / (a + b)

        m_exp = 1 - exp(-dt/tau_m)
        h_exp = 1 - exp(-dt/tau_h)
        n_exp = 1 - exp(-dt/tau_n)
}

FUNCTION vtrap(x,y) {
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        } else {
                vtrap = x/(exp(x/y)-1)
        }
}
