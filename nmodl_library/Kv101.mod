NEURON {
SUFFIX Kv101_003_6States
USEION k READ ek WRITE ik
RANGE g, gbar
}

UNITS { (mV) = (millivolt) }

PARAMETER {
Ri=22.8417 ()
Rn=0.2647 ()
Vc= 48.3636 (mV)
Zc= 0.58528 ()
kc= 0.078262 (/ms)
ki= 0.0010849 (/ms)
kn= 1.4307 (/ms)
ric_c=0.99989 ()
ric_n=0.9639 ()
vc_ic=9.9999 ()
vic_c=999.9527 ()
vic_n=0.0010001 ()
vn_ic=9.9991 ()
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




