NEURON {
SUFFIX KCNQ5_6States
USEION k READ ek WRITE ik
RANGE g, gbar
}

UNITS { (mV) = (millivolt) }

PARAMETER {
Ri=16.6815 ()
Rn=0.0032356 ()
Vc= 80.0 (mV)
Zc= 1.4961 ()
kc= 0.01 (/ms)
ki= 0.00017795 (/ms)
kn= 9.9987 (/ms)
ric_c=0.99807 ()
ric_n=0.14714 ()
vc_ic=9.9985 ()
vic_c=995.4519 ()
vic_n=0.0010059 ()
vn_ic=9.9683 ()
gbar = 36     (millimho/cm2)
}

ASSIGNED {
v    (mV)
ek   (mV)
g    (millimho/cm2)
ik   (milliamp/cm2)
kco  (/ms)
koc  (/ms)
}

STATE { CS OS IC1 CIC1 IN INC1 }

LOCAL F, A,rc_ic,rn_ic

BREAKPOINT {
SOLVE kin METHOD sparse
g = gbar*OS
ik = g*(v - ek)*(1e-3)
}

INITIAL { 
F= 96.485(joule/mV)
A= 8.134(joule/degC)*(celsius+273.15)
rc_ic=ric_c
rn_ic=ric_n
SOLVE kin STEADYSTATE sparse 
}

KINETIC kin {
rates(v)
~ CS <-> OS     (kco, koc)
~ CIC1 <-> IC1  (vc_ic*kco,rc_ic*vc_ic*koc)
~ CS <-> CIC1   (ric_c*vic_c*ki, vic_c*ki*Ri)
~ OS <-> IC1    (ki, ki*Ri)
~ OS <-> IN     (kn, kn*Rn)
~ IC1 <-> INC1  (vn_ic*rn_ic*kn, vn_ic*kn*Rn)
~ IN <-> INC1  (ric_n*vic_n*ki , vic_n*ki*Ri)
CONSERVE CS+OS+IC1+CIC1+IN+INC1 = 1
}

PROCEDURE rates(v(millivolt)) {
kco=kc*exp(Zc*F/A*(v-Vc))
koc=kc*exp(-Zc*F/A*(v-Vc))
}




