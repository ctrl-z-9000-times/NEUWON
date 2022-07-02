TITLE Test detecting nonlinear kinetic models.

NEURON {
    SUFFIX nonlin
}

STATE { A B }

INITIAL {
    A = 0
    B = 1
}

BREAKPOINT {
    SOLVE kin METHOD sparse
}

KINETIC kin {
    ~ A <-> B + B (42, 3.14159)
}
