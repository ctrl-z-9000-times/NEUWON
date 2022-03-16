TITLE A very simple glutamate presynapse.

NEURON {
    POINT_PROCESS glu_presyn
    USEION glu WRITE gluo
}

PARAMETER {
    release = .0001 (/ms/glu_presyn)
}

BREAKPOINT {
    if (v > 10) {
        gluo = release
    }
}
