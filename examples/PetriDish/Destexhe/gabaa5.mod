TITLE detailed model of GABA-A receptors

COMMENT
-----------------------------------------------------------------------------

	Kinetic model of GABA-A receptors
	=================================

	5-state gating model from Busch and Sakmann (Cold Spring Harbor
	Symp. Quant. Biol. 55: 69-80, 1990)
  
	C -- C1 -- C2
	     |     |
      	     O1    O2

-----------------------------------------------------------------------------

  Based on voltage-clamp recordings of GABAA receptor-mediated currents in rat
  hippocampal slices (Otis and Mody, Neuroscience 49: 13-32, 1992), this model
  was fit directly to experimental recordings in order to obtain the optimal
  values for the parameters (see Destexhe, Mainen and Sejnowski, 1996).

-----------------------------------------------------------------------------

  This mod file does not include mechanisms for the release and time course
  of transmitter; it is to be used in conjunction with a sepearate mechanism
  to describe the release of transmitter and that provides the concentration
  of transmitter in the synaptic cleft (to be connected to pointer C here).

-----------------------------------------------------------------------------

  See details in:

  Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic models of 
  synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition; 
  edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1998, pp. 1-25.

  (electronic copy available at http://cns.iaf.cnrs-gif.fr)



  Written by Alain Destexhe, Laval University, 1995

-----------------------------------------------------------------------------
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	POINT_PROCESS GABAa5
	POINTER C
	RANGE C0, C1, C2, O1, O2
	RANGE g, gmax, f1, f2
	GLOBAL Erev, kf1, kf2, kb1, kb2, a1, b1, a2, b2
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

	Erev	= -80    (mV)	: reversal potential
	gmax	= 500  (pS)	: maximal conductance
	
: Rates

	: from Destexhe, Mainen and Sejnowski, 1996

	kf1	= 0.02   (/uM /ms)	: binding 		
	kf2	= 0.01   (/uM /ms)	: binding 		
	kb1	= 4.6	(/ms)	: unbinding		
	kb2	= 9.2	(/ms)	: unbinding		
	a1	= 3.3	(/ms)	: opening
	b1	= 9.8	(/ms)	: closing
	a2	= 10.6	(/ms)	: opening
	b2	= 0.41  (/ms)	: closing
}

COMMENT
	: from Busch and Sakmann

	kf1	= 0.2   (/uM /ms)	: binding 		
	kf2	= 0.1   (/uM /ms)	: binding 		
	kb1	= 3	(/ms)	: unbinding		
	kb2	= 6	(/ms)	: unbinding		
	a1	= 0.7	(/ms)	: opening
	b1	= 4	(/ms)	: closing
	a2	= 10	(/ms)	: opening
	b2	= 0.055 (/ms)	: closing
ENDCOMMENT

ASSIGNED {
	v		(mV)		: postsynaptic voltage
	i 		(nA)		: current = g*(v - Erev)
	g 		(pS)		: conductance
	C 		(mM)		: pointer to glutamate concentration

	f1		(/ms)    : binding
	f2		(/ms)    : binding
}

STATE {
	: Channel states (all fractions)
	C0		: unbound
	C1		: single bound
	C2		: double bound
	O1		: open
	O2		: open
}

INITIAL {
	C0 = 1
	C1 = 0
	C2 = 0
	O1 = 0
	O2 = 0
}

BREAKPOINT {
	SOLVE kstates METHOD sparse

	g = gmax * (O1+O2)
	i = (1e-6) * g * (v - Erev)
}

KINETIC kstates {
	
	f1 = kf1 * (1e3) * C 
	f2 = kf2 * (1e3) * C 

	~ C0 <-> C1	(f1,kb1)
	~ C1 <-> C2	(f2,kb2)
	~ C1 <-> O1	(a1,b1)
	~ C2 <-> O2	(a2,b2)

	CONSERVE C0+C1+C2+O1+O2 = 1
}

