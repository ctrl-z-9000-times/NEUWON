TITLE A very simple glutamate presynapse.

NEURON {
    POINT_PROCESS glu_presyn
    USEION glu WRITE gluo
}

PARAMETER {
    release = .001 (/ms/glu_presyn)
}

BREAKPOINT {
    if (v > -37) {
        gluo = release
    }
}
