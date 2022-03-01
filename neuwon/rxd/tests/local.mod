NEURON {
    SUFFIX local
    POINTER hh
}

INITIAL {}

BREAKPOINT {
    SOLVE decay METHOD cnexp
}

DERIVATIVE decay {
    hh' = -hh / 100
}
