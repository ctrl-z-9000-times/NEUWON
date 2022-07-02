
#include <Eigen/Core>
#include <Eigen/LU>

const double dt = TIME_STEP;
const double celsius = 37.0;

const double C1C2b2 = 18;
const double C1C2v2 = -7;
const double C1C2k2 = -10;
const double C2C1b1 = 3;
const double C2C1v1 = -37;
const double C2C1k1 = 10;
const double C2C1b2 = 18;
const double C2C1v2 = -7;
const double C2C1k2 = -10;
const double C2O1b2 = 18;
const double C2O1v2 = -7;
const double C2O1k2 = -10;
const double O1C2b1 = 3;
const double O1C2v1 = -37;
const double O1C2k1 = 10;
const double O1C2b2 = 18;
const double O1C2v2 = -7;
const double O1C2k2 = -10;
const double C2O2b2 = 0.08;
const double C2O2v2 = -10;
const double C2O2k2 = -15;
const double O2C2b1 = 2;
const double O2C2v1 = -50;
const double O2C2k1 = 7;
const double O2C2b2 = 0.2;
const double O2C2v2 = -20;
const double O2C2k2 = -10;
const double O1I1b1 = 8;
const double O1I1v1 = -37;
const double O1I1k1 = 13;
const double O1I1b2 = 17;
const double O1I1v2 = -7;
const double O1I1k2 = -15;
const double I1O1b1 = 1e-05;
const double I1O1v1 = -37;
const double I1O1k1 = 10;
const double I1C1b1 = 0.21;
const double I1C1v1 = -61;
const double I1C1k1 = 7;
const double C1I1b2 = 0.3;
const double C1I1v2 = -61;
const double C1I1k2 = -5.5;
const double I1I2b2 = 0.0015;
const double I1I2v2 = -90;
const double I1I2k2 = -5;
const double I2I1b1 = 0.0075;
const double I2I1v1 = -90;
const double I2I1k1 = 15;

inline double rates2_na11a(double arg_v, double b, double vv, double k) {
    return (b / (1.0 + exp((arg_v - vv) / k)));
}

extern "C"
void advance_state(
        int num_instances,
        const double* __restrict__ voltage,
        const int* __restrict__ node_index,
        double* __restrict__ C1,
        double* __restrict__ C2,
        double* __restrict__ I1,
        double* __restrict__ I2,
        double* __restrict__ O1,
        double* __restrict__ O2)
{
    for(int id = 0; id < num_instances; ++id) {
        const int node_id = node_index[id];
        const double v = voltage[node_id];

        Eigen::Matrix<double, 6, 1> X;
        double* x_data = X.data();
        x_data[static_cast<int>(0)] = C1[id];
        x_data[static_cast<int>(1)] = C2[id];
        x_data[static_cast<int>(2)] = O1[id];
        x_data[static_cast<int>(3)] = O2[id];
        x_data[static_cast<int>(4)] = I1[id];
        x_data[static_cast<int>(5)] = I2[id];

        const double old_C1 = C1[id];
        const double old_C2 = C2[id];
        const double old_O1 = O1[id];
        const double old_O2 = O2[id];
        const double old_I1 = I1[id];

        const double Q10 = pow(3.0, ((celsius - 20.0) / 10.0));
        const double C1C2_a = Q10 * rates2_na11a(v, C1C2b2, C1C2v2, C1C2k2);
        const double C2C1_a = Q10 * (rates2_na11a(v, C2C1b1, C2C1v1, C2C1k1) + rates2_na11a(v, C2C1b2, C2C1v2, C2C1k2));
        const double C2O1_a = Q10 * rates2_na11a(v, C2O1b2, C2O1v2, C2O1k2);
        const double O1C2_a = Q10 * (rates2_na11a(v, O1C2b1, O1C2v1, O1C2k1) + rates2_na11a(v, O1C2b2, O1C2v2, O1C2k2));
        const double C2O2_a = Q10 * rates2_na11a(v, C2O2b2, C2O2v2, C2O2k2);
        const double O2C2_a = Q10 * (rates2_na11a(v, O2C2b1, O2C2v1, O2C2k1) + rates2_na11a(v, O2C2b2, O2C2v2, O2C2k2));
        const double O1I1_a = Q10 * (rates2_na11a(v, O1I1b1, O1I1v1, O1I1k1) + rates2_na11a(v, O1I1b2, O1I1v2, O1I1k2));
        const double I1O1_a = Q10 * rates2_na11a(v, I1O1b1, I1O1v1, I1O1k1);
        const double I1C1_a = Q10 * rates2_na11a(v, I1C1b1, I1C1v1, I1C1k1);
        const double C1I1_a = Q10 * rates2_na11a(v, C1I1b2, C1I1v2, C1I1k2);
        const double I1I2_a = Q10 * rates2_na11a(v, I1I2b2, I1I2v2, I1I2k2);
        const double I2I1_a = Q10 * rates2_na11a(v, I2I1b1, I2I1v1, I2I1k1);

        Eigen::Matrix<double, 6, 1> F; // Vector to store result of function F(X)
        Eigen::Matrix<double, 6, 6> J; // Matrix to store jacobian of F(X)

        // calculate F, J from X using user-supplied function
        const double* nmodl_eigen_x = X.data();
        double* nmodl_eigen_j = J.data();
        double* nmodl_eigen_f = F.data();
        nmodl_eigen_f[static_cast<int>(0)] =  -C1C2_a * nmodl_eigen_x[static_cast<int>(0)] * dt - C1I1_a * nmodl_eigen_x[static_cast<int>(0)] * dt + C2C1_a * nmodl_eigen_x[static_cast<int>(1)] * dt + I1C1_a * nmodl_eigen_x[static_cast<int>(4)] * dt - nmodl_eigen_x[static_cast<int>(0)] + old_C1;
        nmodl_eigen_j[static_cast<int>(0)] =  -C1C2_a * dt - C1I1_a * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(6)] = C2C1_a * dt;
        nmodl_eigen_j[static_cast<int>(12)] = 0.0;
        nmodl_eigen_j[static_cast<int>(18)] = 0.0;
        nmodl_eigen_j[static_cast<int>(24)] = I1C1_a * dt;
        nmodl_eigen_j[static_cast<int>(30)] = 0.0;
        nmodl_eigen_f[static_cast<int>(1)] = C1C2_a * nmodl_eigen_x[static_cast<int>(0)] * dt - C2C1_a * nmodl_eigen_x[static_cast<int>(1)] * dt - C2O1_a * nmodl_eigen_x[static_cast<int>(1)] * dt - C2O2_a * nmodl_eigen_x[static_cast<int>(1)] * dt + O1C2_a * nmodl_eigen_x[static_cast<int>(2)] * dt + O2C2_a * nmodl_eigen_x[static_cast<int>(3)] * dt - nmodl_eigen_x[static_cast<int>(1)] + old_C2;
        nmodl_eigen_j[static_cast<int>(1)] = C1C2_a * dt;
        nmodl_eigen_j[static_cast<int>(7)] =  -C2C1_a * dt - C2O1_a * dt - C2O2_a * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(13)] = O1C2_a * dt;
        nmodl_eigen_j[static_cast<int>(19)] = O2C2_a * dt;
        nmodl_eigen_j[static_cast<int>(25)] = 0.0;
        nmodl_eigen_j[static_cast<int>(31)] = 0.0;
        nmodl_eigen_f[static_cast<int>(2)] = C2O1_a * nmodl_eigen_x[static_cast<int>(1)] * dt + I1O1_a * nmodl_eigen_x[static_cast<int>(4)] * dt - O1C2_a * nmodl_eigen_x[static_cast<int>(2)] * dt - O1I1_a * nmodl_eigen_x[static_cast<int>(2)] * dt - nmodl_eigen_x[static_cast<int>(2)] + old_O1;
        nmodl_eigen_j[static_cast<int>(2)] = 0.0;
        nmodl_eigen_j[static_cast<int>(8)] = C2O1_a * dt;
        nmodl_eigen_j[static_cast<int>(14)] =  -O1C2_a * dt - O1I1_a * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(20)] = 0.0;
        nmodl_eigen_j[static_cast<int>(26)] = I1O1_a * dt;
        nmodl_eigen_j[static_cast<int>(32)] = 0.0;
        nmodl_eigen_f[static_cast<int>(3)] = C2O2_a * nmodl_eigen_x[static_cast<int>(1)] * dt - O2C2_a * nmodl_eigen_x[static_cast<int>(3)] * dt - nmodl_eigen_x[static_cast<int>(3)] + old_O2;
        nmodl_eigen_j[static_cast<int>(3)] = 0.0;
        nmodl_eigen_j[static_cast<int>(9)] = C2O2_a * dt;
        nmodl_eigen_j[static_cast<int>(15)] = 0.0;
        nmodl_eigen_j[static_cast<int>(21)] =  -O2C2_a * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(27)] = 0.0;
        nmodl_eigen_j[static_cast<int>(33)] = 0.0;
        nmodl_eigen_f[static_cast<int>(4)] = C1I1_a * nmodl_eigen_x[static_cast<int>(0)] * dt - I1C1_a * nmodl_eigen_x[static_cast<int>(4)] * dt - I1I2_a * nmodl_eigen_x[static_cast<int>(4)] * dt - I1O1_a * nmodl_eigen_x[static_cast<int>(4)] * dt + I2I1_a * nmodl_eigen_x[static_cast<int>(5)] * dt + O1I1_a * nmodl_eigen_x[static_cast<int>(2)] * dt - nmodl_eigen_x[static_cast<int>(4)] + old_I1;
        nmodl_eigen_j[static_cast<int>(4)] = C1I1_a * dt;
        nmodl_eigen_j[static_cast<int>(10)] = 0.0;
        nmodl_eigen_j[static_cast<int>(16)] = O1I1_a * dt;
        nmodl_eigen_j[static_cast<int>(22)] = 0.0;
        nmodl_eigen_j[static_cast<int>(28)] =  -I1C1_a * dt - I1I2_a * dt - I1O1_a * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(34)] = I2I1_a * dt;
        nmodl_eigen_f[static_cast<int>(5)] =  -nmodl_eigen_x[static_cast<int>(0)] - nmodl_eigen_x[static_cast<int>(1)] - nmodl_eigen_x[static_cast<int>(2)] - nmodl_eigen_x[static_cast<int>(3)] - nmodl_eigen_x[static_cast<int>(4)] - nmodl_eigen_x[static_cast<int>(5)] + 1.0;
        nmodl_eigen_j[static_cast<int>(5)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(11)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(17)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(23)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(29)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(35)] =  -1.0;

        X -= J.partialPivLu().solve(F);

        C1[id] = x_data[static_cast<int>(0)];
        C2[id] = x_data[static_cast<int>(1)];
        O1[id] = x_data[static_cast<int>(2)];
        O2[id] = x_data[static_cast<int>(3)];
        I1[id] = x_data[static_cast<int>(4)];
        I2[id] = x_data[static_cast<int>(5)];
    }
}
