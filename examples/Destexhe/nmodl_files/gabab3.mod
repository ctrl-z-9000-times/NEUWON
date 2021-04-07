TITLE detailed model of GABAB receptors

COMMENT
-----------------------------------------------------------------------------

	Kinetic model of GABA-B receptors
	=================================

	Detailed model of GABAB currents including nonlinear stiumulus
	dependency (fundamental to take into account for GABAB receptors)
	and precise fit to experimentally-recorded currents.

	Features:

  	  - peak at 100 ms; time course fit to Tom Otis' PSC
	  - NONLINEAR SUMMATION (psc is much stronger with bursts)
	    due to cooperativity of G-protein binding on K+ channels


	Approximations:

	  - single binding site on receptor	
	  - desensitization of the receptor
	  - model of alpha G-protein activation (direct) of K+ channel
	  - G-protein dynamics is second-order; simplified as follows:
		- saturating receptor
		- Michaelis-Menten of receptor for G-protein production
		- "resting" G-protein is in excess
		- Quasi-stat of intermediate enzymatic forms
	  - binding on K+ channel is fast


	Kinetic Equations:

	  dR/dt = K1 * T * (1-R-D) - K2 * R + d2 * D

	  dD/dt = d1 * R - d2 * D

	  dG/dt = K3 * R - K4 * G

	  R : activated receptor
	  T : transmitter
	  G : activated G-protein
	  K1,K2,K3,K4,d1,d2 = kinetic rate cst

  n activated G-protein bind to a K+ channel:

	n G + C <-> O		(Alpha,Beta)

  If the binding is fast, the fraction of open channels is given by:

	O = G^n / ( G^n + KD )

  where KD = Beta / Alpha is the dissociation constant

-----------------------------------------------------------------------------

  Based on voltage-clamp recordings of GABAB receptor-mediated currents in rat
  hippocampal slices (Otis et al, J. Physiol. 463: 391-407, 1993), this model 
  was fit directly to experimental recordings in order to obtain the optimal
  values for the parameters (see Destexhe and Sejnowski, 1995).

-----------------------------------------------------------------------------

  This mod file does not include mechanisms for the release and time course
  of transmitter; it is to be used in conjunction with a sepearate mechanism
  to describe the release of transmitter and that provides the concentration
  of transmitter in the synaptic cleft (to be connected to pointer C here).

-----------------------------------------------------------------------------

  See details in:

  Destexhe, A. and Sejnowski, T.J.  G-protein activation kinetics and
  spill-over of GABA may account for differences between inhibitory responses
  in the hippocampus and thalamus.  Proc. Natl. Acad. Sci. USA  92:
  9515-9519, 1995.

  See also: 

  Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic models of 
  synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition; 
  edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1998, pp. 1-25.

  (electronic copy available at http://cns.iaf.cnrs-gif.fr)



  Written by Alain Destexhe, Laval University, 1995

-----------------------------------------------------------------------------
ENDCOMMENT



INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	POINT_PROCESS GABAb3
	POINTER C
	RANGE R, D, G, g, gmax
	NONSPECIFIC_CURRENT i
	GLOBAL K1, K2, K3, K4, KD, d1, d2, Erev
}
UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
	(mM) = (milli/liter)
}

PARAMETER {

:
:	From simplex fitting to experimental data
:	(Destexhe and Sejnowski, 1995)
:
	K1	= 0.66	(/ms mM)	: forward binding rate to receptor
	K2	= 0.020 (/ms)		: backward (unbinding) rate of receptor
	K3	= 0.083 (/ms)		: rate of G-protein production
	K4	= 0.0079 (/ms)		: rate of G-protein decay
	d1	= 0.017 (/ms)		: rate of desensitization
	d2	= 0.0053 (/ms)		: rate of re-sensitization
	KD	= 100			: dissociation constant of K+ channel
	n	= 4			: nb of binding sites of G-protein on K+
	Erev	= -95	(mV)		: reversal potential (E_K)
	gmax		(umho)		: maximum conductance
}


ASSIGNED {
	v		(mV)		: postsynaptic voltage
	i 		(nA)		: current = g*(v - Erev)
	g 		(umho)		: conductance
	C		(mM)		: pointer to transmitter concentration
	Gn
}


STATE {
	R				: fraction of activated receptor
	D				: fraction of desensitized receptor
	G				: fraction of activated G-protein
}


INITIAL {
	R = 0
	D = 0
	G = 0
}

BREAKPOINT {
	SOLVE bindkin METHOD cnexp
	Gn = G^n
	g = gmax * Gn / (Gn+KD)
	i = g*(v - Erev)
}


DERIVATIVE bindkin {

	R' = K1 * C * (1-R-D) - K2 * R + d2 * D
	D' = d1 * R - d2 * D
	G' = K3 * R - K4 * G

}


