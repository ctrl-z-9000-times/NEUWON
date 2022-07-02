TITLE detailed model of glutamate AMPA receptors

COMMENT
-----------------------------------------------------------------------------

    Kinetic model of AMPA receptors
    ===============================

    13-state gating model:
      
         O1    O2    O3    O4
         |     |     |     |
   C0 -- C1 -- C2 -- C3 -- C4
         |     |     |     |
         D1    D2    D3    D4

-----------------------------------------------------------------------------

  Based on voltage-clamp recordings of AMPA receptor-mediated currents in mouse unipolar brush cells, this
  model was fit directly to experimental recordings (by LT in Axograph) in order to obtain the
  optimal values for the parameters.  
  
  Lu, H.W., Balmer, T.S., Romero, G.E., and Trussell, L.O. (2017). 
  Slow AMPAR Synaptic Transmission Is Determined by Stargazin and Glutamate Transporters. Neuron 96, 73-80 e74.
-----------------------------------------------------------------------------

  Based on model described in:

  Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic models of 
  synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition; 
  edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1998, pp. 1-25.


-----------------------------------------------------------------------------
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    POINT_PROCESS AMPA13
    POINTER C
    RANGE C0, C1, C2, C3, C4, D1, D2, D3, D4, O1, O2, O3, O4
    RANGE g, gmax, rb1, rb2, rb3, rb4, Q10_binding, Q10_desensitization, Q10_opening, Q10_unbinding
    GLOBAL Erev
    GLOBAL Rb1, Rb2, Rb3, Rb4, Ru1, Ru2, Ru3, Ru4, Rd1, Rd2, Rd3, Rd4, Rr1, Rr2, Rr3, Rr4, Ro1, Ro2, Ro3, Ro4, Rc1, Rc2, Rc3, Rc4
    GLOBAL vmin, vmax
    NONSPECIFIC_CURRENT i
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
    Q10_binding = 2.4
    Q10_unbinding = 2.4
    Q10_desensitization = 2.4
    Q10_opening = 2.4
    
    celsius       (degC)
    Erev    = 0    (mV) : reversal potential
    gmax    = 20  (pS)  : maximal conductance of a single channel, the subconductance states are fractions of this, see BREAKPOINT
    vmin = -120 (mV)
    vmax = 100  (mV)
    
: Rates

    Rb1 = 800   (/mM /ms) : binding first site
    Rb2 = 600   (/mM /ms) : binding second site
    Rb3 = 400   (/mM /ms) : binding third site
    Rb4 = 200   (/mM /ms) : binding fourth site

    Ru1 = 30    (/ms)   : unbinding (1st site)
    Ru2 = 40  (/ms): unbinding (2nd site)
    Ru3 = 60    (/ms)   : unbinding (3rd site)  
    Ru4 = 80    (/ms)   : unbinding (4th site)  
    
    Rd1 = .25       (/ms)   : desensitization with 1 bound
    Rd2 = .25       (/ms)   : desensitization with 2 bound
    Rd3 = 1     (/ms)   : desensitization with 3 bound
    Rd4 = 1     (/ms)   : desensitization with 4 bound

    Rr1 = 0.05 (/ms)    : resensitization with 1 bound
    Rr2 = 0.05 (/ms)    : resensitization with 2 bound
    Rr3 = 0.022 (/ms)   : resensitization with 3 bound
    Rr4 = 0.022 (/ms)   : resensitization with 4 bound

    Ro1 = 3 (/ms)   : opening with 1 bound
    Ro2 = 4 (/ms)   : opening with 2 bound   
    Ro3 = 4 (/ms)   : opening with 3 bound    
    Ro4 = 4 (/ms)   : opening with 4 bound    
    
    Rc1 = 1.5   (/ms)   : closing with 1 bound
    Rc2 = 1 (/ms)   : closing with 2 bound
    Rc3 = 1 (/ms)   : closing with 3 bound
    Rc4 = 1.5   (/ms)   : closing with 4 bound
}

ASSIGNED {
    v       (mV)        : postsynaptic voltage
    i       (nA)        : current = g*(v - Erev)
    g       (pS)        : conductance
    C       (mM)        : pointer to glutamate concentration

    rb1     (/ms)    : binding first site
    rb2     (/ms)    : binding first site
    rb3     (/ms)    : binding first site
    rb4     (/ms)    : binding first site
    
    Q10b (1)
    Q10u (1)
    Q10dr (1)
    Q10oc (1)
}

STATE {
    : Channel states (all fractions)
    C0      : unbound
    C1      : single glu bound
    C2      : double glu bound
    C3      : 3 glu bound
    C4      : 4 glu bound
    D1      : single glu bound, desensitized
    D2      : double glu bound, desensitized
    D3      : 3 glu bound, desensitized
    D4      : 4 glu bound, desensitized
    O1      : open state 1
    O2      : open state 2
    O3      : open state 3
    O4      : open state 4
}

INITIAL {
    C0=1
    C1=0
    C2=0
    C3=0
    C4=0
    D1=0
    D2=0
    D3=0
    D4=0
    O1=0
    O2=0
    O3=0
    O4=0
    
    Q10b = Q10_binding^((celsius-22)/10)
    Q10u = Q10_unbinding^((celsius-22)/10)
    Q10dr = Q10_desensitization^((celsius-22)/10)
    Q10oc = Q10_opening^((celsius-22)/10)
}

BREAKPOINT {
    SOLVE kstates METHOD sparse

    g = gmax * (O4 + 0.75*O3 + 0.5*O2 + 0.25*O1)
    i = (1e-6) * g * (v - Erev)
}

KINETIC kstates {
    
    rb1 = Rb1 * C 
    rb2 = Rb2 * C
    rb3 = Rb3 * C
    rb4 = Rb4 * C
    ~ C0 <-> C1 (rb1*Q10b,Ru1*Q10u)
    ~ C1 <-> C2 (rb2*Q10b,Ru2*Q10u)
    ~ C2 <-> C3 (rb3*Q10b,Ru3*Q10u)
    ~ C3 <-> C4 (rb4*Q10b,Ru4*Q10u)
    ~ C1 <-> D1 (Rd1*Q10dr,Rr1*Q10dr)
    ~ C2 <-> D2 (Rd2*Q10dr,Rr2*Q10dr)
    ~ C3 <-> D3 (Rd3*Q10dr,Rr3*Q10dr)
    ~ C4 <-> D4 (Rd4*Q10dr,Rr4*Q10dr)
    ~ C1 <-> O1 (Ro1*Q10oc,Rc1*Q10oc)
    ~ C2 <-> O2 (Ro2*Q10oc,Rc2*Q10oc)
    ~ C3 <-> O3 (Ro3*Q10oc,Rc3*Q10oc)
    ~ C4 <-> O4 (Ro4*Q10oc,Rc4*Q10oc)

    CONSERVE C0+C1+C2+C3+C4+D1+D2+D3+D4+O1+O2+O3+O4 = 1
}







