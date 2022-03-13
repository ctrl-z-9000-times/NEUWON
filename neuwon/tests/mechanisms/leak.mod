TITLE Leak mechanism.

NEURON {
    SUFFIX leak
    USEION leak WRITE gleak
}

PARAMETER {
    leak_conductance = .002 (S/cm2)
}

BREAKPOINT {
    gleak = leak_conductance
}
