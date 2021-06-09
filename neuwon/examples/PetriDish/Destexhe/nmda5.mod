TITLE detailed model of glutamate NMDA receptors

COMMENT
-----------------------------------------------------------------------------

	Kinetic model of NMDA receptors
	===============================

	5-state gating model:
	Clements & Westbrook 1991. Neuron 7: 605.
	Lester & Jahr 1992. J Neurosci 12: 635.
	Edmonds & Colquhoun 1992. Proc. R. Soc. Lond. B 250: 279.
	Hessler, Shirke & Malinow. 1993. Nature 366: 569.
	Clements et al. 1992. Science 258: 1498.
  
	C -- C1 -- C2 -- O
	           |
      	           D

	Voltage dependence of Mg2+ block:
	Jahr & Stevens 1990. J Neurosci 10: 1830.
	Jahr & Stevens 1990. J Neurosci 10: 3178.

-----------------------------------------------------------------------------

  Based on voltage-clamp recordings of NMDA receptor-mediated currents in rat
  hippocampal slices (Hessler et al., Nature 366: 569-572, 1993), this model 
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
  edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1998, pp 1-25.

  (electronic copy available at http://cns.iaf.cnrs-gif.fr)


  Written by Alain Destexhe and Zach Mainen, 1995

-----------------------------------------------------------------------------
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	POINT_PROCESS NMDA5
	POINTER C
	RANGE C0, C1, C2, D, O, B
	RANGE g, gmax, rb
	GLOBAL Erev, mg, Rb, Ru, Rd, Rr, Ro, Rc
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

	Erev	= 0    (mV)	: reversal potential
	gmax	= 500  (pS)	: maximal conductance
	mg	= 0    (mM)	: external magnesium concentration
	vmin = -120	(mV)
	vmax = 100	(mV)
	
: Rates

	: Destexhe, Mainen & Sejnowski, 1996
	Rb	= 5e-3    (/uM /ms)	: binding 		
	Ru	= 12.9e-3  (/ms)	: unbinding		
	Rd	= 8.4e-3   (/ms)	: desensitization
	Rr	= 6.8e-3   (/ms)	: resensitization 
	Ro	= 46.5e-3   (/ms)	: opening
	Rc	= 73.8e-3   (/ms)	: closing
}

COMMENT
	: Clements et al. 1992
	Rb	= 5e-3    (/uM /ms)	: binding 		
	Ru	= 9.5e-3  (/ms)	: unbinding		
	Rd	= 16e-3   (/ms)	: desensitization
	Rr	= 13e-3   (/ms)	: resensitization 
	Ro	= 25e-3   (/ms)	: opening
	Rc	= 59e-3   (/ms)	: closing

	: Hessler Shirke & Malinow 1993
	Rb	= 5e-3    (/uM /ms)	: binding 		
	Ru	= 9.5e-3  (/ms)	: unbinding		
	Rd	= 16e-3   (/ms)	: desensitization
	Rr	= 13e-3   (/ms)	: resensitization 
	Ro	= 25e-3   (/ms)	: opening
	Rc	= 59e-3   (/ms)	: closing

	: Clements & Westbrook 1991
	Rb	=  5    (uM /s)	: binding 		
	Ru	=  5	(/s)	: unbinding -> gives Kd = Rb/Ru = 1 uM
	Rd	=  4.0  (/s)	: desensitization
	Rr	=  0.3  (/s)	: resensitization 
	Ro	= 10  (/s)	: opening
	Rc	= 322   (/s)	: closing

	: Edmonds & Colquhoun 1992
	Rb	=  5    (uM /s)	: binding 		
	Ru	=  4.7  (/s)	: unbinding		
	Rd	=  8.4  (/s)	: desensitization
	Rr	=  1.8  (/s)	: resensitization 
	Ro	= 46.5  (/s)	: opening
	Rc	= 91.6  (/s)	: closing

	: Lester & Jahr 1992
	Rb	= 5    (uM /s)	: binding 		
	Ru	= 6.7   (/s)	: unbinding		
	Rd	= 15.2  (/s)	: desensitization
	Rr	= 9.4   (/s)	: resensitization 
	Ro	= 83.8  (/s)	: opening
	Rc	= 83.8  (/s)	: closing

ENDCOMMENT


ASSIGNED {
	v		(mV)		: postsynaptic voltage
	i 		(nA)		: current = g*(v - Erev)
	g 		(pS)		: conductance
	C 		(mM)		: pointer to glutamate concentration

	rb		(/ms)    : binding
}

STATE {
	: Channel states (all fractions)
	C0		: unbound
	C1		: single bound
	C2		: double bound
	D		: desensitized
	O		: open

	B		: fraction free of Mg2+ block
}

INITIAL {
	rates(v)
	C0 = 1
}

BREAKPOINT {
	rates(v)
	SOLVE kstates METHOD sparse

	g = gmax * O * B
	i = (1e-6) * g * (v - Erev)
}

KINETIC kstates {
	
	rb = Rb * (1e3) * C 

	~ C0 <-> C1	(rb,Ru)
	~ C1 <-> C2	(rb,Ru)
	~ C2 <-> D	(Rd,Rr)
	~ C2 <-> O	(Ro,Rc)

	CONSERVE C0+C1+C2+D+O = 1
}

PROCEDURE rates(v(mV)) {
	TABLE B
	DEPEND mg
	FROM vmin TO vmax WITH 200

	: from Jahr & Stevens

	B = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))
}

