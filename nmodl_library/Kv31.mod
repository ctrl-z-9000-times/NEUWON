NEURON {
SUFFIX Kv31_6States_temperature2
USEION k READ ek WRITE ik
RANGE g, gbar
}

UNITS { (mV) = (millivolt) }

PARAMETER {
Hkc=0.061023 (/degC)
Hki=-0.012582 (/degC)
Hkn=-0.46898 (/degC)
Hri=0.20775 (/degC)
Hric_n=0.18503 (/degC)
Hrn=-0.13286 (/degC)
Hvc=0.54002 (/degC)
Hvic_c=-0.13958 (/degC)
Hvic_n=0.31079 (/degC)
Hvn_ic=0.91346 (/degC)
Ri=12.0305 ()
Rn=1.4841 ()
Vc= 12.9133 (mV)
Zc= 1.0615 ()
kc= 0.18044 (/ms)
ki= 0.00017744 (/ms)
kn= 0.11543 (/ms)
ric_c=0.091948 ()
ric_n=15.1958 ()
vc_ic=10.0 ()
vic_c=645.7302 ()
vic_n=0.0010205 ()
vn_ic=0.1 ()
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

LOCAL F, A,rc_ic,rn_ic,Hrn_ic

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
Hrn_ic=Hric_n
SOLVE kin STEADYSTATE sparse 
}

KINETIC kin {
rates(v)
~ CS <-> OS     (kco, koc)
~ CIC1 <-> IC1  (vc_ic*kco,rc_ic*vc_ic*koc)
~ CS <-> CIC1   (ric_c*vic_c*ki*exp(Hki*(celsius-25))*exp(Hvic_c*(celsius-25)), vic_c*ki*Ri*exp(Hki*(celsius-25))*exp(Hri*(celsius-25))*exp(Hvic_c*(celsius-25)))
~ OS <-> IC1    (ki*exp(Hki*(celsius-25)), ki*Ri*exp(Hki*(celsius-25))*exp(Hri*(celsius-25)))
~ OS <-> IN     (kn*exp(Hkn*(celsius-25)), kn*Rn*exp(Hkn*(celsius-25))*exp(Hrn*(celsius-25)))
~ IC1 <-> INC1  (vn_ic*rn_ic*kn*exp(Hkn*(celsius-25))*exp(Hrn_ic*(celsius-25))*exp(Hvn_ic*(celsius-25)), vn_ic*kn*Rn*exp(Hkn*(celsius-25))*exp(Hrn*(celsius-25))*exp(Hvn_ic*(celsius-25)))
~ IN <-> INC1  (ric_n*vic_n*ki*exp(Hki*(celsius-25))*exp(Hric_n*(celsius-25))*exp(Hvic_n*(celsius-25)) , vic_n*ki*Ri*exp(Hki*(celsius-25))*exp(Hri*(celsius-25))*exp(Hvic_n*(celsius-25)))
CONSERVE CS+OS+IC1+CIC1+IN+INC1 = 1
}

PROCEDURE rates(v(millivolt)) {
kco=kc*exp(Zc*F/A*(v-Vc-Hvc*(celsius-25)))*exp(Hkc*(celsius-25))
koc=kc*exp(-Zc*F/A*(v-Vc-Hvc*(celsius-25)))*exp(Hkc*(celsius-25))
}




