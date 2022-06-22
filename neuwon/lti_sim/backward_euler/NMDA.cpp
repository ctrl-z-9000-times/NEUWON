
#include <Eigen/Core>
#include <Eigen/LU>

const double dt = TIME_STEP;
const double celsius = 37.0;

const double mg = 1;
const double valence = -2;
const double memb_fraction = 0.8;
const double Rb = 0.01;
const double Ru = 0.0056;
const double Ro = 0.01;
const double Rc = 0.273;
const double Rd1 = 0.0022;
const double Rr1 = 0.0016;
const double Rd2 = 0.00043;
const double Rr2 = 0.0005;
const double Rmb = 5e-05;
const double Rmu = 12.8;
const double Rmc1b = 5e-08;
const double Rmc1u = 0.00243831;
const double Rmc2b = 5e-08;
const double Rmc2u = 0.00504192;
const double Rmd1b = 5e-08;
const double Rmd1u = 0.00298874;
const double Rmd2b = 5e-08;
const double Rmd2u = 0.00295341;
const double RbMg = 0.01;
const double RuMg = 0.0171;
const double RoMg = 0.01;
const double RcMg = 0.548;
const double Rd1Mg = 0.0021;
const double Rr1Mg = 0.00087;
const double Rd2Mg = 0.00026;
const double Rr2Mg = 0.00042;

extern "C"
void advance_state(
        int num_instances,
        const double* __restrict__ glu,
        const int* __restrict__ glu_index,
        const double* __restrict__ voltage,
        const int* __restrict__ voltage_index,
        double* __restrict__ Cl,
        double* __restrict__ ClMg,
        double* __restrict__ D1,
        double* __restrict__ D1Mg,
        double* __restrict__ D2,
        double* __restrict__ D2Mg,
        double* __restrict__ O,
        double* __restrict__ OMg,
        double* __restrict__ U,
        double* __restrict__ UMg)
{
    for(int id = 0; id < num_instances; ++id) {
        const double v = voltage[voltage_index[id]];
        const double C = glu[glu_index[id]];

        Eigen::Matrix<double, 10, 1> X;
        double* nmodl_eigen_x = X.data();
        nmodl_eigen_x[static_cast<int>(0)] = U[id];
        nmodl_eigen_x[static_cast<int>(1)] = Cl[id];
        nmodl_eigen_x[static_cast<int>(2)] = D1[id];
        nmodl_eigen_x[static_cast<int>(3)] = D2[id];
        nmodl_eigen_x[static_cast<int>(4)] = O[id];
        nmodl_eigen_x[static_cast<int>(5)] = UMg[id];
        nmodl_eigen_x[static_cast<int>(6)] = ClMg[id];
        nmodl_eigen_x[static_cast<int>(7)] = D1Mg[id];
        nmodl_eigen_x[static_cast<int>(8)] = D2Mg[id];
        nmodl_eigen_x[static_cast<int>(9)] = OMg[id];

        const double rb = Rb * (1e3) * C;
        const double rbMg = RbMg * (1e3) * C;
        const double rmb = Rmb * mg * (1e3) * exp((v - 40.0) * valence * memb_fraction / 25.0);
        const double rmu = Rmu * exp(( -1.0) * (v - 40.0) * valence * (1.0 - memb_fraction) / 25.0);
        const double rmc1b = Rmc1b * mg * (1e3) * exp((v - 40.0) * valence * memb_fraction / 25.0);
        const double rmc1u = Rmc1u * exp(( -1.0) * (v - 40.0) * valence * (1.0 - memb_fraction) / 25.0);
        const double rmc2b = Rmc2b * mg * (1e3) * exp((v - 40.0) * valence * memb_fraction / 25.0);
        const double rmc2u = Rmc2u * exp(( -1.0) * (v - 40.0) * valence * (1.0 - memb_fraction) / 25.0);
        const double rmd1b = Rmd1b * mg * (1e3) * exp((v - 40.0) * valence * memb_fraction / 25.0);
        const double rmd1u = Rmd1u * exp(( -1.0) * (v - 40.0) * valence * (1.0 - memb_fraction) / 25.0);
        const double rmd2b = Rmd2b * mg * (1e3) * exp((v - 40.0) * valence * memb_fraction / 25.0);
        const double rmd2u = Rmd2u * exp(( -1.0) * (v - 40.0) * valence * (1.0 - memb_fraction) / 25.0);

        const double old_U = U[id];
        const double old_Cl = Cl[id];
        const double old_D1 = D1[id];
        const double old_D2 = D2[id];
        const double old_O = O[id];
        const double old_UMg = UMg[id];
        const double old_ClMg = ClMg[id];
        const double old_D1Mg = D1Mg[id];
        const double old_D2Mg = D2Mg[id];

        Eigen::Matrix<double, 10, 10> J; // Matrix to store jacobian of F(X)
        Eigen::Matrix<double, 10, 1> F; // Vector to store result of function F(X)

        double* nmodl_eigen_j = J.data();
        double* nmodl_eigen_f = F.data();

        nmodl_eigen_f[static_cast<int>(0)] = Ru * nmodl_eigen_x[static_cast<int>(1)] * dt - nmodl_eigen_x[static_cast<int>(0)] * dt * rb - nmodl_eigen_x[static_cast<int>(0)] * dt * rmc1b - nmodl_eigen_x[static_cast<int>(0)] + nmodl_eigen_x[static_cast<int>(5)] * dt * rmc1u + old_U;
        nmodl_eigen_j[static_cast<int>(0)] =  -dt * rb - dt * rmc1b - 1.0;
        nmodl_eigen_j[static_cast<int>(10)] = Ru * dt;
        nmodl_eigen_j[static_cast<int>(20)] = 0.0;
        nmodl_eigen_j[static_cast<int>(30)] = 0.0;
        nmodl_eigen_j[static_cast<int>(40)] = 0.0;
        nmodl_eigen_j[static_cast<int>(50)] = dt * rmc1u;
        nmodl_eigen_j[static_cast<int>(60)] = 0.0;
        nmodl_eigen_j[static_cast<int>(70)] = 0.0;
        nmodl_eigen_j[static_cast<int>(80)] = 0.0;
        nmodl_eigen_j[static_cast<int>(90)] = 0.0;
        nmodl_eigen_f[static_cast<int>(1)] = Rc * nmodl_eigen_x[static_cast<int>(4)] * dt - Rd1 * nmodl_eigen_x[static_cast<int>(1)] * dt - Ro * nmodl_eigen_x[static_cast<int>(1)] * dt + Rr1 * nmodl_eigen_x[static_cast<int>(2)] * dt - Ru * nmodl_eigen_x[static_cast<int>(1)] * dt + nmodl_eigen_x[static_cast<int>(0)] * dt * rb - nmodl_eigen_x[static_cast<int>(1)] * dt * rmc2b - nmodl_eigen_x[static_cast<int>(1)] + nmodl_eigen_x[static_cast<int>(6)] * dt * rmc2u + old_Cl;
        nmodl_eigen_j[static_cast<int>(1)] = dt * rb;
        nmodl_eigen_j[static_cast<int>(11)] =  -Rd1 * dt - Ro * dt - Ru * dt - dt * rmc2b - 1.0;
        nmodl_eigen_j[static_cast<int>(21)] = Rr1 * dt;
        nmodl_eigen_j[static_cast<int>(31)] = 0.0;
        nmodl_eigen_j[static_cast<int>(41)] = Rc * dt;
        nmodl_eigen_j[static_cast<int>(51)] = 0.0;
        nmodl_eigen_j[static_cast<int>(61)] = dt * rmc2u;
        nmodl_eigen_j[static_cast<int>(71)] = 0.0;
        nmodl_eigen_j[static_cast<int>(81)] = 0.0;
        nmodl_eigen_j[static_cast<int>(91)] = 0.0;
        nmodl_eigen_f[static_cast<int>(2)] = Rd1 * nmodl_eigen_x[static_cast<int>(1)] * dt - Rd2 * nmodl_eigen_x[static_cast<int>(2)] * dt - Rr1 * nmodl_eigen_x[static_cast<int>(2)] * dt + Rr2 * nmodl_eigen_x[static_cast<int>(3)] * dt - nmodl_eigen_x[static_cast<int>(2)] * dt * rmd1b - nmodl_eigen_x[static_cast<int>(2)] + nmodl_eigen_x[static_cast<int>(7)] * dt * rmd1u + old_D1;
        nmodl_eigen_j[static_cast<int>(2)] = 0.0;
        nmodl_eigen_j[static_cast<int>(12)] = Rd1 * dt;
        nmodl_eigen_j[static_cast<int>(22)] =  -Rd2 * dt - Rr1 * dt - dt * rmd1b - 1.0;
        nmodl_eigen_j[static_cast<int>(32)] = Rr2 * dt;
        nmodl_eigen_j[static_cast<int>(42)] = 0.0;
        nmodl_eigen_j[static_cast<int>(52)] = 0.0;
        nmodl_eigen_j[static_cast<int>(62)] = 0.0;
        nmodl_eigen_j[static_cast<int>(72)] = dt * rmd1u;
        nmodl_eigen_j[static_cast<int>(82)] = 0.0;
        nmodl_eigen_j[static_cast<int>(92)] = 0.0;
        nmodl_eigen_f[static_cast<int>(3)] = Rd2 * nmodl_eigen_x[static_cast<int>(2)] * dt - Rr2 * nmodl_eigen_x[static_cast<int>(3)] * dt - nmodl_eigen_x[static_cast<int>(3)] * dt * rmd2b - nmodl_eigen_x[static_cast<int>(3)] + nmodl_eigen_x[static_cast<int>(8)] * dt * rmd2u + old_D2;
        nmodl_eigen_j[static_cast<int>(3)] = 0.0;
        nmodl_eigen_j[static_cast<int>(13)] = 0.0;
        nmodl_eigen_j[static_cast<int>(23)] = Rd2 * dt;
        nmodl_eigen_j[static_cast<int>(33)] =  -Rr2 * dt - dt * rmd2b - 1.0;
        nmodl_eigen_j[static_cast<int>(43)] = 0.0;
        nmodl_eigen_j[static_cast<int>(53)] = 0.0;
        nmodl_eigen_j[static_cast<int>(63)] = 0.0;
        nmodl_eigen_j[static_cast<int>(73)] = 0.0;
        nmodl_eigen_j[static_cast<int>(83)] = dt * rmd2u;
        nmodl_eigen_j[static_cast<int>(93)] = 0.0;
        nmodl_eigen_f[static_cast<int>(4)] =  -Rc * nmodl_eigen_x[static_cast<int>(4)] * dt + Ro * nmodl_eigen_x[static_cast<int>(1)] * dt - nmodl_eigen_x[static_cast<int>(4)] * dt * rmb - nmodl_eigen_x[static_cast<int>(4)] + nmodl_eigen_x[static_cast<int>(9)] * dt * rmu + old_O;
        nmodl_eigen_j[static_cast<int>(4)] = 0.0;
        nmodl_eigen_j[static_cast<int>(14)] = Ro * dt;
        nmodl_eigen_j[static_cast<int>(24)] = 0.0;
        nmodl_eigen_j[static_cast<int>(34)] = 0.0;
        nmodl_eigen_j[static_cast<int>(44)] =  -Rc * dt - dt * rmb - 1.0;
        nmodl_eigen_j[static_cast<int>(54)] = 0.0;
        nmodl_eigen_j[static_cast<int>(64)] = 0.0;
        nmodl_eigen_j[static_cast<int>(74)] = 0.0;
        nmodl_eigen_j[static_cast<int>(84)] = 0.0;
        nmodl_eigen_j[static_cast<int>(94)] = dt * rmu;
        nmodl_eigen_f[static_cast<int>(5)] = RuMg * nmodl_eigen_x[static_cast<int>(6)] * dt + nmodl_eigen_x[static_cast<int>(0)] * dt * rmc1b - nmodl_eigen_x[static_cast<int>(5)] * dt * rbMg - nmodl_eigen_x[static_cast<int>(5)] * dt * rmc1u - nmodl_eigen_x[static_cast<int>(5)] + old_UMg;
        nmodl_eigen_j[static_cast<int>(5)] = dt * rmc1b;
        nmodl_eigen_j[static_cast<int>(15)] = 0.0;
        nmodl_eigen_j[static_cast<int>(25)] = 0.0;
        nmodl_eigen_j[static_cast<int>(35)] = 0.0;
        nmodl_eigen_j[static_cast<int>(45)] = 0.0;
        nmodl_eigen_j[static_cast<int>(55)] =  -dt * rbMg - dt * rmc1u - 1.0;
        nmodl_eigen_j[static_cast<int>(65)] = RuMg * dt;
        nmodl_eigen_j[static_cast<int>(75)] = 0.0;
        nmodl_eigen_j[static_cast<int>(85)] = 0.0;
        nmodl_eigen_j[static_cast<int>(95)] = 0.0;
        nmodl_eigen_f[static_cast<int>(6)] = RcMg * nmodl_eigen_x[static_cast<int>(9)] * dt - Rd1Mg * nmodl_eigen_x[static_cast<int>(6)] * dt - RoMg * nmodl_eigen_x[static_cast<int>(6)] * dt + Rr1Mg * nmodl_eigen_x[static_cast<int>(7)] * dt - RuMg * nmodl_eigen_x[static_cast<int>(6)] * dt + nmodl_eigen_x[static_cast<int>(1)] * dt * rmc2b + nmodl_eigen_x[static_cast<int>(5)] * dt * rbMg - nmodl_eigen_x[static_cast<int>(6)] * dt * rmc2u - nmodl_eigen_x[static_cast<int>(6)] + old_ClMg;
        nmodl_eigen_j[static_cast<int>(6)] = 0.0;
        nmodl_eigen_j[static_cast<int>(16)] = dt * rmc2b;
        nmodl_eigen_j[static_cast<int>(26)] = 0.0;
        nmodl_eigen_j[static_cast<int>(36)] = 0.0;
        nmodl_eigen_j[static_cast<int>(46)] = 0.0;
        nmodl_eigen_j[static_cast<int>(56)] = dt * rbMg;
        nmodl_eigen_j[static_cast<int>(66)] =  -Rd1Mg * dt - RoMg * dt - RuMg * dt - dt * rmc2u - 1.0;
        nmodl_eigen_j[static_cast<int>(76)] = Rr1Mg * dt;
        nmodl_eigen_j[static_cast<int>(86)] = 0.0;
        nmodl_eigen_j[static_cast<int>(96)] = RcMg * dt;
        nmodl_eigen_f[static_cast<int>(7)] = Rd1Mg * nmodl_eigen_x[static_cast<int>(6)] * dt - Rd2Mg * nmodl_eigen_x[static_cast<int>(7)] * dt - Rr1Mg * nmodl_eigen_x[static_cast<int>(7)] * dt + Rr2Mg * nmodl_eigen_x[static_cast<int>(8)] * dt + nmodl_eigen_x[static_cast<int>(2)] * dt * rmd1b - nmodl_eigen_x[static_cast<int>(7)] * dt * rmd1u - nmodl_eigen_x[static_cast<int>(7)] + old_D1Mg;
        nmodl_eigen_j[static_cast<int>(7)] = 0.0;
        nmodl_eigen_j[static_cast<int>(17)] = 0.0;
        nmodl_eigen_j[static_cast<int>(27)] = dt * rmd1b;
        nmodl_eigen_j[static_cast<int>(37)] = 0.0;
        nmodl_eigen_j[static_cast<int>(47)] = 0.0;
        nmodl_eigen_j[static_cast<int>(57)] = 0.0;
        nmodl_eigen_j[static_cast<int>(67)] = Rd1Mg * dt;
        nmodl_eigen_j[static_cast<int>(77)] =  -Rd2Mg * dt - Rr1Mg * dt - dt * rmd1u - 1.0;
        nmodl_eigen_j[static_cast<int>(87)] = Rr2Mg * dt;
        nmodl_eigen_j[static_cast<int>(97)] = 0.0;
        nmodl_eigen_f[static_cast<int>(8)] = Rd2Mg * nmodl_eigen_x[static_cast<int>(7)] * dt - Rr2Mg * nmodl_eigen_x[static_cast<int>(8)] * dt + nmodl_eigen_x[static_cast<int>(3)] * dt * rmd2b - nmodl_eigen_x[static_cast<int>(8)] * dt * rmd2u - nmodl_eigen_x[static_cast<int>(8)] + old_D2Mg;
        nmodl_eigen_j[static_cast<int>(8)] = 0.0;
        nmodl_eigen_j[static_cast<int>(18)] = 0.0;
        nmodl_eigen_j[static_cast<int>(28)] = 0.0;
        nmodl_eigen_j[static_cast<int>(38)] = dt * rmd2b;
        nmodl_eigen_j[static_cast<int>(48)] = 0.0;
        nmodl_eigen_j[static_cast<int>(58)] = 0.0;
        nmodl_eigen_j[static_cast<int>(68)] = 0.0;
        nmodl_eigen_j[static_cast<int>(78)] = Rd2Mg * dt;
        nmodl_eigen_j[static_cast<int>(88)] =  -Rr2Mg * dt - dt * rmd2u - 1.0;
        nmodl_eigen_j[static_cast<int>(98)] = 0.0;
        nmodl_eigen_f[static_cast<int>(9)] =  -nmodl_eigen_x[static_cast<int>(0)] - nmodl_eigen_x[static_cast<int>(1)] - nmodl_eigen_x[static_cast<int>(2)] - nmodl_eigen_x[static_cast<int>(3)] - nmodl_eigen_x[static_cast<int>(4)] - nmodl_eigen_x[static_cast<int>(5)] - nmodl_eigen_x[static_cast<int>(6)] - nmodl_eigen_x[static_cast<int>(7)] - nmodl_eigen_x[static_cast<int>(8)] - nmodl_eigen_x[static_cast<int>(9)] + 1.0;
        nmodl_eigen_j[static_cast<int>(9)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(19)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(29)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(39)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(49)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(59)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(69)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(79)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(89)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(99)] =  -1.0;

        X -= J.partialPivLu().solve(F);

        U[id] = nmodl_eigen_x[static_cast<int>(0)];
        Cl[id] = nmodl_eigen_x[static_cast<int>(1)];
        D1[id] = nmodl_eigen_x[static_cast<int>(2)];
        D2[id] = nmodl_eigen_x[static_cast<int>(3)];
        O[id] = nmodl_eigen_x[static_cast<int>(4)];
        UMg[id] = nmodl_eigen_x[static_cast<int>(5)];
        ClMg[id] = nmodl_eigen_x[static_cast<int>(6)];
        D1Mg[id] = nmodl_eigen_x[static_cast<int>(7)];
        D2Mg[id] = nmodl_eigen_x[static_cast<int>(8)];
        OMg[id] = nmodl_eigen_x[static_cast<int>(9)];
    }
}
