
COMMENT

High threshold Ca2+ channel

2-state kinetics with sigmoidal voltage-dependence

  C<->O

Goldman-Hodgkin-Katz equations

     # MODEL
    |   MODEL AUTHOR  : D.A. McCormick & J. Huguenard
    |   MODEL DATE    : 1992
    |   MODEL REF     : A model of the electrophysiological properties of 
thalamocortical relay neurons. J Neurophysiol, 1992 Oct, 68(4):1384-400.
 
    # EXPERIMENT
    |   EXP AUTHOR    : Kay AR; Wong RK
    |   EXP DATE      : 1987
    |   EXP REF       : Journal of Physiology, 1987 Nov, 392:603-16.
    |   ANIMAL        : guinea-pig
    |   BRAIN REGION  : hippocampus
    |   CELL TYPE     : Ca1 pyramidal
    |   TECHNIQUE     : slices, whole-cell
    |   RECORDING METHOD  : voltage-clamp
    |   TEMPERATURE   : 20-22
 
Reference:

   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. Synthesis of models for
   excitable membranes, synaptic transmission and neuromodulation using a 
   common kinetic formalism, Journal of Computational Neuroscience 1: 
   195-230, 1994.

  (electronic copy available at http://cns.iaf.cnrs-gif.fr)


ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX caL
	USEION ca READ cai, cao WRITE ica
	RANGE O, C, I
	RANGE a,b
	GLOBAL Ra, Rb, q, th, p
	GLOBAL q10, temp, tadj
}

UNITS {
	F = (faraday) (coulomb)
	R = (k-mole) (joule/degC)
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
	(mM) = (milli/liter)
} 

PARAMETER {
	p    = 0.2e-3  	(cm/s)		: max permeability
	v 		(mV)

	th   = 5	(mV)		: v 1/2 for on/off
	q   = 13	(mV)		: voltage dependence

	: max rates

	Ra   = 1.6	(/ms)		: open (v)
	Rb   = 0.2	(/ms)		: close (v)

	celsius		(degC)
	temp = 22	(degC)		: original temp
	q10  = 3			: temperature sensitivity
} 


ASSIGNED {
	ica 		(mA/cm2)
	cao		(mM)
	cai		(mM)
	a (/ms)	b (/ms)
	tadj
}
 

STATE { C O }

INITIAL { 
	C = 1 
}


BREAKPOINT {
	rates(v)
	SOLVE kstates METHOD sparse
	ica = O * p * ghk(v,cai,cao)
} 


KINETIC kstates {
	~ C <-> O 	(a,b)	
	CONSERVE C+O = 1
}	
	
PROCEDURE rates(v(mV)) {
	TABLE a, b
	DEPEND Ra, Rb, th, celsius, temp, q10
	FROM -100 TO 100 WITH 200

	tadj = q10 ^ ((celsius - temp)/10 (degC))

	a = Ra / (1 + exp(-(v-th)/q)) * tadj
	b = Rb / (1 + exp((v-th)/q)) * tadj
}

: Special gear for calculating the Ca2+ reversal potential
: via Goldman-Hodgkin-Katz eqn.
: [Ca2+]o "cao" and [Ca2+]i "cai" are assumed to be set elsewhere


FUNCTION ghk(v(mV), ci(mM), co(mM)) (0.001 coul/cm3) {
	LOCAL z

	z = (0.001)*2*F*v/(R*(celsius+273.15))
	ghk = (.001)*2*F*(ci*efun(-z) - co*efun(z))
}

FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	}else{
		efun = z/(exp(z) - 1)
	}
}




