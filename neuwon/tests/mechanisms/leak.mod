TITLE Leak mechanism.

NEURON {
    SUFFIX leak
    USEION k WRITE gk
}

PARAMETER {
    k_conductance = .002 (S/cm2)
}

BREAKPOINT {
    gk = k_conductance
}
