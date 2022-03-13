TITLE A very simple glutamate postsynaptic receptor.

NEURON {
    POINT_PROCESS ampa
    USEION zero WRITE gzero
    USEION glu READ gluo
}

PARAMETER {
    max_conductance = .001 (S/mM/ampa)
}

BREAKPOINT {
    gzero = gluo * max_conductance
}
