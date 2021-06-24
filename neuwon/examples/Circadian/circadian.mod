TITLE circadian.mod
COMMENT
ENDCOMMENT

UNITS {
    (mV) = (millivolt)
    (nM) = (nanomole/liter)
    (hour) = (3600 second)
}

STATE {
    MP
    P0
    P1
    P2
    MT
    T0
    T1
    T2
    C
    CN
}

INITIAL {
    MP = 0.0614368 (nM)
    P0 = 0.0169928 (nM)
    P1 = 0.0141356 (nM)
    P2 = 0.0614368 (nM)
    MT = 0.0860342 (nM)
    T0 = 0.0217261 (nM)
    T1 = 0.0213384 (nM)
    T2 = 0.0145428 (nM)
    C  = 0.207614 (nM)
    CN = 1.34728 (nM)
}

PARAMETER {
    vsP = 1.1 (nM / hour)
    vmP = 1.0 (nM / hour)
    KmP = 0.2 (nM)
    KIP = 1.0 (nM)
    ksP = 0.9 (/hour)
    vdP = 2.2 (nM / hour)
    KdP = 0.2 (nM)
    vsT = 1.0 (nM / hour)
    vmT = 0.7 (nM / hour)
    KmT = 0.2 (nM)
    KIT = 1.0 (nM)
    ksT = 0.9 (/hour)
    vdT = 3.0 (nM / hour)
    KdT = 0.2 (nM)
    kdC = 0.01 (nM / hour)
    kdN = 0.01 (nM / hour)
    k1 = 0.8 (/hour)
    k2 = 0.2 (/hour)
    k3 = 1.2 (/nM-hour)
    k4 = 0.6 (/hour)
    kd = 0.01 (nM / hour)
    V1P = 8.0 (nM / hour)
    V1T = 8.0 (nM / hour)
    V2P = 1.0 (nM / hour)
    V2T = 1.0 (nM / hour)
    V3P = 8.0 (nM / hour)
    V3T = 8.0 (nM / hour)
    V4P = 1.0 (nM / hour)
    V4T = 1.0 (nM / hour)
    K1P = 2.0 (nM)
    K1T = 2.0 (nM)
    K2P = 2.0 (nM)
    K2T = 2.0 (nM)
    K3P = 2.0 (nM)
    K3T = 2.0 (nM)
    K4P = 2.0 (nM)
    K4T = 2.0 (nM)
    n = 4
}

BREAKPOINT {
    SOLVE rates METHOD cnexp
    SOLVE reactions METHOD cnexp
}

DERIVATIVE rates {
    MT' = vsT * KIT ^ n / (KIT ^ n + CN ^ n)
    MP' = vsP * KIP ^ n / (KIP ^ n + CN ^ n)
    MT' = -(vmT * MT / (KmT + MT) + kd * MT)
    MP' = -(vmP * MP / (KmP + MP) + kd * MP)
    T0' = ksT * MT
    T0' = -kd * T0
    T1' = -kd * T1
    T2' = -kd * T2
    T2' = -vdT * T2 / (KdT + T2)
    P0' = ksP * MP
    P0' = -kd * P0
    P1' = -kd * P1
    P2' = -kd * P2 - vdP * P2 / (KdP + P2)
    C'  = -kdC * C
    CN' = -kdN * CN
}

KINETIC reactions {
    ~ T0 <-> T1         (V1T * T0 / (K1T + T0), V2T * T1 / (K2T + T1))
    ~ T1 <-> T2         (V3T * T1 / (K3T + T1), V4T * T2 / (K4T + T2))
    ~ P0 <-> P1         (V1P * P0 / (K1P + P0), V2P * P1 / (K2P + P1))
    ~ P1 <-> P2         (V3P * P1 / (K3P + P1), V4P * P2 / (K4P + P2))
    ~ P2 + T2 <-> C     (k3, k4)
    ~ C  <-> CN         (k1, k2)
}
