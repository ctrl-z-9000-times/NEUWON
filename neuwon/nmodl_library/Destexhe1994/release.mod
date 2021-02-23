TITLE transmitter release

COMMENT
-----------------------------------------------------------------------------

 Simple (minimal?) model of transmitter release

 - single compartment, need calcium influx and efflux

 - Ca++ binds to a "fusion factor" protein F leading to an activated form FA.
   Assuming a cooperativity factor of 4 (see Augustine & charlton, 
   J Physiol. 381: 619-640, 1986), one obtains:

	F + 4 Cai <-> FA	(kb,ku)

 - FA binds to presynaptic vesicles and activates them according to:

	FA + V <-> VA		(k1,k2)

   VA represents the "activated vesicle" which is able to bind to the
   membrane and release transmitter.  Presynaptic vesicles (V) are 
   considered in excess.

 - VA releases nt transmitter molecules in the synaptic cleft

	VA  ->  nt T		(k3)

   This reaction is the slowest and a constant number of transmitter per 
   vesicule is considered (nt).  

 - Finally, T is hydrolyzed according to a first-order reaction

	T  ->  ...		(kh)


   References:

   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. Synthesis of models for
   excitable membranes, synaptic transmission and neuromodulation using a 
   common kinetic formalism, Journal of Computational Neuroscience 1: 
   195-230, 1994.

   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic models of 
   synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition; 
   edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1998, pp 1-25.

  (electronic copy available at http://cns.iaf.cnrs-gif.fr)

   For a more realistic model, see Yamada, WM & Zucker, RS. Time course
   of transmitter release calculated from simulations of a calcium
   diffusion model. Biophys. J. 61: 671-5682, 1992.


  Written by A. Destexhe, Salk Institute, December 1993; modified 1996

-----------------------------------------------------------------------------
ENDCOMMENT
: the following is the comment from the old cad which became incorporated
: into rel ...
: TITLE decay of internal calcium concentration
:
: Internal calcium concentration due to calcium currents and pump.
: Differential equations.
:
: Simple model of ATPase pump with 3 kinetic constants (Destexhe 92)
:     Cai + P <-> CaP -> Cao + P  (k1,k2,k3)
: A Michaelis-Menten approximation is assumed, which reduces the complexity
: of the system to 2 parameters: 
:       kt = <tot enzyme concentration> * k3  -> TIME CONSTANT OF THE PUMP
:	kd = k2/k1 (dissociation constant)    -> EQUILIBRIUM CALCIUM VALUE
: The values of these parameters are chosen assuming a high affinity of 
: the pump to calcium and a low transport capacity (cfr. Blaustein, 
: TINS, 11: 438, 1988, and references therein).  
:
: Units checked using "modlunit" -> factor 10000 needed in ca entry
:
: VERSION OF PUMP + DECAY (decay can be viewed as simplified buffering)
:
: All variables are range variables
:
:
: This mechanism was published in:  Destexhe, A. Babloyantz, A. and 
: Sejnowski, TJ.  Ionic mechanisms for intrinsic slow oscillations in
: thalamic relay neurons. Biophys. J. 65: 1538-1552, 1993)
:
: (electronic copy available at http://cns.iaf.cnrs-gif.fr)
:
: Written by Alain Destexhe, Salk Institute, Nov 12, 1992
:



INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX rel
	USEION ca READ ica, cai WRITE cai
	RANGE T,FA,CA,Fmax,Ves,b,u,k1,k2,k3,nt,kh
: from cad :
	RANGE depth,kt,kd,cainf,taur
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(mM) = (milli/liter)
: from cad:
	(molar) = (1/liter)			: moles do not appear in units
:	(mM)	= (millimolar)
	(um)	= (micron)
:	(mA)	= (milliamp)
	(msM)	= (ms mM)

}
: from cad:

CONSTANT {
	FARADAY = 96489		(coul)		: moles do not appear in units
:	FARADAY = 96.489	(k-coul)	: moles do not appear in units
}

PARAMETER {

	Ves = 0.1 	(mM)		: conc of vesicles
	Fmax = 0.001	(mM)		: conc of fusion factor F
	b = 1e16 	(/mM4-ms)	: ca binding to F
	u = 0.1  	(/ms)		: ca unbinding 
	k1 = 1000   	(/mM-ms)	: F binding to vesicle
	k2 = 0.1	(/ms)		: F unbinding to vesicle
	k3 = 4   	(/ms)		: exocytosis of T
	nt = 10000			: nb of molec of T per vesicle
	kh = 10  	(/ms)		: cst for hydolysis of T
: from cad:
	depth	= .1	(um)		: depth of shell
	taur	= 700	(ms)		: rate of calcium removal
	cainf	= 1e-8	(mM)
	kt	= 1	(mM/ms)		: estimated from k3=.5, tot=.001
	kd	= 5e-4	(mM)		: estimated from k2=250, k1=5e5
}

ASSIGNED {
	ica		(mA/cm2)
	drive_channel	(mM/ms)
	drive_pump	(mM/ms)
}

STATE {
	FA	(mM)
	VA	(mM)
	T	(mM)
	cai	(mM) 
}

INITIAL {
	FA = 0
	VA = 0
	T = 0
:	cai = 1e-8
	cai = kd
}

BREAKPOINT {
	SOLVE state METHOD derivimplicit
}

LOCAL bfc , kfv

DERIVATIVE state {

	bfc = b * (Fmax-FA-VA) * cai^4
	kfv = k1 * FA * Ves
:	this is the old equation incorporated into the below:
:	cai'	= - bfc + 4 * u * FA 
	FA'	= bfc - u * FA - kfv + k2 * VA
	VA'	= kfv - (k2+k3) * VA
	T'	= nt * k3 * VA - kh * T
: from cad:

	drive_channel =  - (10000) * ica / (2 * FARADAY * depth)

	if (drive_channel <= 0.) { drive_channel = 0. }	: cannot pump inward

:	drive_pump = -tot * k3 * cai / (cai + ((k2+k3)/k1) )	: quasistat
	drive_pump = -kt * cai / (cai + kd )		: Michaelis-Menten

:	this is the eq for cai prime from cad incorporated into below:
:	cai' = drive_channel + drive_pump + (cainf-cai)/taur
	cai'= -bfc+4*u*FA + drive_channel + drive_pump + (cainf-cai)/taur

}	
