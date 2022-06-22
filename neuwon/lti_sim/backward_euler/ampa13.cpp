
#include <Eigen/Core>
#include <Eigen/LU>

const double dt = TIME_STEP;
const double celsius = 37.0;

const double Q10_binding = 2.4;
const double Q10_unbinding = 2.4;
const double Q10_desensitization = 2.4;
const double Q10_opening = 2.4;

const double Q10b = pow(Q10_binding, ((celsius - 22.0) / 10.0));
const double Q10u = pow(Q10_unbinding, ((celsius - 22.0) / 10.0));
const double Q10dr = pow(Q10_desensitization, ((celsius - 22.0) / 10.0));
const double Q10oc = pow(Q10_opening, ((celsius - 22.0) / 10.0));

const double Rb1 = 800;
const double Rb2 = 600;
const double Rb3 = 400;
const double Rb4 = 200;
const double Ru1 = 30;
const double Ru2 = 40;
const double Ru3 = 60;
const double Ru4 = 80;
const double Rd1 = 0.25;
const double Rd2 = 0.25;
const double Rd3 = 1;
const double Rd4 = 1;
const double Rr1 = 0.05;
const double Rr2 = 0.05;
const double Rr3 = 0.022;
const double Rr4 = 0.022;
const double Ro1 = 3;
const double Ro2 = 4;
const double Ro3 = 4;
const double Ro4 = 4;
const double Rc1 = 1.5;
const double Rc2 = 1;
const double Rc3 = 1;
const double Rc4 = 1.5;

extern "C"
void advance_state(
        int num_instances,
        const double* __restrict__ C,
        const int* __restrict__ node_index,
        double* __restrict__ C0,
        double* __restrict__ C1,
        double* __restrict__ C2,
        double* __restrict__ C3,
        double* __restrict__ C4,
        double* __restrict__ D1,
        double* __restrict__ D2,
        double* __restrict__ D3,
        double* __restrict__ D4,
        double* __restrict__ O1,
        double* __restrict__ O2,
        double* __restrict__ O3,
        double* __restrict__ O4) {
    for(int id = 0; id < num_instances; ++id) {
        const int node_id = node_index[id];
        const double glu = C[node_id];

        Eigen::Matrix<double, 13, 1> X;
        double* nmodl_eigen_x = X.data();
        nmodl_eigen_x[static_cast<int>(0)] = C0[id];
        nmodl_eigen_x[static_cast<int>(1)] = C1[id];
        nmodl_eigen_x[static_cast<int>(2)] = C2[id];
        nmodl_eigen_x[static_cast<int>(3)] = C3[id];
        nmodl_eigen_x[static_cast<int>(4)] = C4[id];
        nmodl_eigen_x[static_cast<int>(5)] = D1[id];
        nmodl_eigen_x[static_cast<int>(6)] = D2[id];
        nmodl_eigen_x[static_cast<int>(7)] = D3[id];
        nmodl_eigen_x[static_cast<int>(8)] = D4[id];
        nmodl_eigen_x[static_cast<int>(9)] = O1[id];
        nmodl_eigen_x[static_cast<int>(10)] = O2[id];
        nmodl_eigen_x[static_cast<int>(11)] = O3[id];
        nmodl_eigen_x[static_cast<int>(12)] = O4[id];

        const double rb1 = Rb1 * glu;
        const double rb2 = Rb2 * glu;
        const double rb3 = Rb3 * glu;
        const double rb4 = Rb4 * glu;

        const double old_C0 = C0[id];
        const double old_C1 = C1[id];
        const double old_C2 = C2[id];
        const double old_C3 = C3[id];
        const double old_C4 = C4[id];
        const double old_D1 = D1[id];
        const double old_D2 = D2[id];
        const double old_D3 = D3[id];
        const double old_D4 = D4[id];
        const double old_O1 = O1[id];
        const double old_O2 = O2[id];
        const double old_O3 = O3[id];

        Eigen::Matrix<double, 13, 1> F;  // Vector to store result of function F(X)
        Eigen::Matrix<double, 13, 13> J;  // Matrix to store jacobian of F(X)

        // calculate F, J from X using user-supplied function
        double* nmodl_eigen_j = J.data();
        double* nmodl_eigen_f = F.data();
        nmodl_eigen_f[static_cast<int>(0)] =  -Q10b * nmodl_eigen_x[static_cast<int>(0)] * dt * rb1 + Q10u * Ru1 * nmodl_eigen_x[static_cast<int>(1)] * dt - nmodl_eigen_x[static_cast<int>(0)] + old_C0;
        nmodl_eigen_j[static_cast<int>(0)] =  -Q10b * dt * rb1 - 1.0;
        nmodl_eigen_j[static_cast<int>(13)] = Q10u * Ru1 * dt;
        nmodl_eigen_j[static_cast<int>(26)] = 0.0;
        nmodl_eigen_j[static_cast<int>(39)] = 0.0;
        nmodl_eigen_j[static_cast<int>(52)] = 0.0;
        nmodl_eigen_j[static_cast<int>(65)] = 0.0;
        nmodl_eigen_j[static_cast<int>(78)] = 0.0;
        nmodl_eigen_j[static_cast<int>(91)] = 0.0;
        nmodl_eigen_j[static_cast<int>(104)] = 0.0;
        nmodl_eigen_j[static_cast<int>(117)] = 0.0;
        nmodl_eigen_j[static_cast<int>(130)] = 0.0;
        nmodl_eigen_j[static_cast<int>(143)] = 0.0;
        nmodl_eigen_j[static_cast<int>(156)] = 0.0;
        nmodl_eigen_f[static_cast<int>(1)] = Q10b * nmodl_eigen_x[static_cast<int>(0)] * dt * rb1 - Q10b * nmodl_eigen_x[static_cast<int>(1)] * dt * rb2 - Q10dr * Rd1 * nmodl_eigen_x[static_cast<int>(1)] * dt + Q10dr * Rr1 * nmodl_eigen_x[static_cast<int>(5)] * dt + Q10oc * Rc1 * nmodl_eigen_x[static_cast<int>(9)] * dt - Q10oc * Ro1 * nmodl_eigen_x[static_cast<int>(1)] * dt - Q10u * Ru1 * nmodl_eigen_x[static_cast<int>(1)] * dt + Q10u * Ru2 * nmodl_eigen_x[static_cast<int>(2)] * dt - nmodl_eigen_x[static_cast<int>(1)] + old_C1;
        nmodl_eigen_j[static_cast<int>(1)] = Q10b * dt * rb1;
        nmodl_eigen_j[static_cast<int>(14)] =  -Q10b * dt * rb2 - Q10dr * Rd1 * dt - Q10oc * Ro1 * dt - Q10u * Ru1 * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(27)] = Q10u * Ru2 * dt;
        nmodl_eigen_j[static_cast<int>(40)] = 0.0;
        nmodl_eigen_j[static_cast<int>(53)] = 0.0;
        nmodl_eigen_j[static_cast<int>(66)] = Q10dr * Rr1 * dt;
        nmodl_eigen_j[static_cast<int>(79)] = 0.0;
        nmodl_eigen_j[static_cast<int>(92)] = 0.0;
        nmodl_eigen_j[static_cast<int>(105)] = 0.0;
        nmodl_eigen_j[static_cast<int>(118)] = Q10oc * Rc1 * dt;
        nmodl_eigen_j[static_cast<int>(131)] = 0.0;
        nmodl_eigen_j[static_cast<int>(144)] = 0.0;
        nmodl_eigen_j[static_cast<int>(157)] = 0.0;
        nmodl_eigen_f[static_cast<int>(2)] = Q10b * nmodl_eigen_x[static_cast<int>(1)] * dt * rb2 - Q10b * nmodl_eigen_x[static_cast<int>(2)] * dt * rb3 - Q10dr * Rd2 * nmodl_eigen_x[static_cast<int>(2)] * dt + Q10dr * Rr2 * nmodl_eigen_x[static_cast<int>(6)] * dt + Q10oc * Rc2 * nmodl_eigen_x[static_cast<int>(10)] * dt - Q10oc * Ro2 * nmodl_eigen_x[static_cast<int>(2)] * dt - Q10u * Ru2 * nmodl_eigen_x[static_cast<int>(2)] * dt + Q10u * Ru3 * nmodl_eigen_x[static_cast<int>(3)] * dt - nmodl_eigen_x[static_cast<int>(2)] + old_C2;
        nmodl_eigen_j[static_cast<int>(2)] = 0.0;
        nmodl_eigen_j[static_cast<int>(15)] = Q10b * dt * rb2;
        nmodl_eigen_j[static_cast<int>(28)] =  -Q10b * dt * rb3 - Q10dr * Rd2 * dt - Q10oc * Ro2 * dt - Q10u * Ru2 * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(41)] = Q10u * Ru3 * dt;
        nmodl_eigen_j[static_cast<int>(54)] = 0.0;
        nmodl_eigen_j[static_cast<int>(67)] = 0.0;
        nmodl_eigen_j[static_cast<int>(80)] = Q10dr * Rr2 * dt;
        nmodl_eigen_j[static_cast<int>(93)] = 0.0;
        nmodl_eigen_j[static_cast<int>(106)] = 0.0;
        nmodl_eigen_j[static_cast<int>(119)] = 0.0;
        nmodl_eigen_j[static_cast<int>(132)] = Q10oc * Rc2 * dt;
        nmodl_eigen_j[static_cast<int>(145)] = 0.0;
        nmodl_eigen_j[static_cast<int>(158)] = 0.0;
        nmodl_eigen_f[static_cast<int>(3)] = Q10b * nmodl_eigen_x[static_cast<int>(2)] * dt * rb3 - Q10b * nmodl_eigen_x[static_cast<int>(3)] * dt * rb4 - Q10dr * Rd3 * nmodl_eigen_x[static_cast<int>(3)] * dt + Q10dr * Rr3 * nmodl_eigen_x[static_cast<int>(7)] * dt + Q10oc * Rc3 * nmodl_eigen_x[static_cast<int>(11)] * dt - Q10oc * Ro3 * nmodl_eigen_x[static_cast<int>(3)] * dt - Q10u * Ru3 * nmodl_eigen_x[static_cast<int>(3)] * dt + Q10u * Ru4 * nmodl_eigen_x[static_cast<int>(4)] * dt - nmodl_eigen_x[static_cast<int>(3)] + old_C3;
        nmodl_eigen_j[static_cast<int>(3)] = 0.0;
        nmodl_eigen_j[static_cast<int>(16)] = 0.0;
        nmodl_eigen_j[static_cast<int>(29)] = Q10b * dt * rb3;
        nmodl_eigen_j[static_cast<int>(42)] =  -Q10b * dt * rb4 - Q10dr * Rd3 * dt - Q10oc * Ro3 * dt - Q10u * Ru3 * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(55)] = Q10u * Ru4 * dt;
        nmodl_eigen_j[static_cast<int>(68)] = 0.0;
        nmodl_eigen_j[static_cast<int>(81)] = 0.0;
        nmodl_eigen_j[static_cast<int>(94)] = Q10dr * Rr3 * dt;
        nmodl_eigen_j[static_cast<int>(107)] = 0.0;
        nmodl_eigen_j[static_cast<int>(120)] = 0.0;
        nmodl_eigen_j[static_cast<int>(133)] = 0.0;
        nmodl_eigen_j[static_cast<int>(146)] = Q10oc * Rc3 * dt;
        nmodl_eigen_j[static_cast<int>(159)] = 0.0;
        nmodl_eigen_f[static_cast<int>(4)] = Q10b * nmodl_eigen_x[static_cast<int>(3)] * dt * rb4 - Q10dr * Rd4 * nmodl_eigen_x[static_cast<int>(4)] * dt + Q10dr * Rr4 * nmodl_eigen_x[static_cast<int>(8)] * dt + Q10oc * Rc4 * nmodl_eigen_x[static_cast<int>(12)] * dt - Q10oc * Ro4 * nmodl_eigen_x[static_cast<int>(4)] * dt - Q10u * Ru4 * nmodl_eigen_x[static_cast<int>(4)] * dt - nmodl_eigen_x[static_cast<int>(4)] + old_C4;
        nmodl_eigen_j[static_cast<int>(4)] = 0.0;
        nmodl_eigen_j[static_cast<int>(17)] = 0.0;
        nmodl_eigen_j[static_cast<int>(30)] = 0.0;
        nmodl_eigen_j[static_cast<int>(43)] = Q10b * dt * rb4;
        nmodl_eigen_j[static_cast<int>(56)] =  -Q10dr * Rd4 * dt - Q10oc * Ro4 * dt - Q10u * Ru4 * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(69)] = 0.0;
        nmodl_eigen_j[static_cast<int>(82)] = 0.0;
        nmodl_eigen_j[static_cast<int>(95)] = 0.0;
        nmodl_eigen_j[static_cast<int>(108)] = Q10dr * Rr4 * dt;
        nmodl_eigen_j[static_cast<int>(121)] = 0.0;
        nmodl_eigen_j[static_cast<int>(134)] = 0.0;
        nmodl_eigen_j[static_cast<int>(147)] = 0.0;
        nmodl_eigen_j[static_cast<int>(160)] = Q10oc * Rc4 * dt;
        nmodl_eigen_f[static_cast<int>(5)] = Q10dr * Rd1 * nmodl_eigen_x[static_cast<int>(1)] * dt - Q10dr * Rr1 * nmodl_eigen_x[static_cast<int>(5)] * dt - nmodl_eigen_x[static_cast<int>(5)] + old_D1;
        nmodl_eigen_j[static_cast<int>(5)] = 0.0;
        nmodl_eigen_j[static_cast<int>(18)] = Q10dr * Rd1 * dt;
        nmodl_eigen_j[static_cast<int>(31)] = 0.0;
        nmodl_eigen_j[static_cast<int>(44)] = 0.0;
        nmodl_eigen_j[static_cast<int>(57)] = 0.0;
        nmodl_eigen_j[static_cast<int>(70)] =  -Q10dr * Rr1 * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(83)] = 0.0;
        nmodl_eigen_j[static_cast<int>(96)] = 0.0;
        nmodl_eigen_j[static_cast<int>(109)] = 0.0;
        nmodl_eigen_j[static_cast<int>(122)] = 0.0;
        nmodl_eigen_j[static_cast<int>(135)] = 0.0;
        nmodl_eigen_j[static_cast<int>(148)] = 0.0;
        nmodl_eigen_j[static_cast<int>(161)] = 0.0;
        nmodl_eigen_f[static_cast<int>(6)] = Q10dr * Rd2 * nmodl_eigen_x[static_cast<int>(2)] * dt - Q10dr * Rr2 * nmodl_eigen_x[static_cast<int>(6)] * dt - nmodl_eigen_x[static_cast<int>(6)] + old_D2;
        nmodl_eigen_j[static_cast<int>(6)] = 0.0;
        nmodl_eigen_j[static_cast<int>(19)] = 0.0;
        nmodl_eigen_j[static_cast<int>(32)] = Q10dr * Rd2 * dt;
        nmodl_eigen_j[static_cast<int>(45)] = 0.0;
        nmodl_eigen_j[static_cast<int>(58)] = 0.0;
        nmodl_eigen_j[static_cast<int>(71)] = 0.0;
        nmodl_eigen_j[static_cast<int>(84)] =  -Q10dr * Rr2 * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(97)] = 0.0;
        nmodl_eigen_j[static_cast<int>(110)] = 0.0;
        nmodl_eigen_j[static_cast<int>(123)] = 0.0;
        nmodl_eigen_j[static_cast<int>(136)] = 0.0;
        nmodl_eigen_j[static_cast<int>(149)] = 0.0;
        nmodl_eigen_j[static_cast<int>(162)] = 0.0;
        nmodl_eigen_f[static_cast<int>(7)] = Q10dr * Rd3 * nmodl_eigen_x[static_cast<int>(3)] * dt - Q10dr * Rr3 * nmodl_eigen_x[static_cast<int>(7)] * dt - nmodl_eigen_x[static_cast<int>(7)] + old_D3;
        nmodl_eigen_j[static_cast<int>(7)] = 0.0;
        nmodl_eigen_j[static_cast<int>(20)] = 0.0;
        nmodl_eigen_j[static_cast<int>(33)] = 0.0;
        nmodl_eigen_j[static_cast<int>(46)] = Q10dr * Rd3 * dt;
        nmodl_eigen_j[static_cast<int>(59)] = 0.0;
        nmodl_eigen_j[static_cast<int>(72)] = 0.0;
        nmodl_eigen_j[static_cast<int>(85)] = 0.0;
        nmodl_eigen_j[static_cast<int>(98)] =  -Q10dr * Rr3 * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(111)] = 0.0;
        nmodl_eigen_j[static_cast<int>(124)] = 0.0;
        nmodl_eigen_j[static_cast<int>(137)] = 0.0;
        nmodl_eigen_j[static_cast<int>(150)] = 0.0;
        nmodl_eigen_j[static_cast<int>(163)] = 0.0;
        nmodl_eigen_f[static_cast<int>(8)] = Q10dr * Rd4 * nmodl_eigen_x[static_cast<int>(4)] * dt - Q10dr * Rr4 * nmodl_eigen_x[static_cast<int>(8)] * dt - nmodl_eigen_x[static_cast<int>(8)] + old_D4;
        nmodl_eigen_j[static_cast<int>(8)] = 0.0;
        nmodl_eigen_j[static_cast<int>(21)] = 0.0;
        nmodl_eigen_j[static_cast<int>(34)] = 0.0;
        nmodl_eigen_j[static_cast<int>(47)] = 0.0;
        nmodl_eigen_j[static_cast<int>(60)] = Q10dr * Rd4 * dt;
        nmodl_eigen_j[static_cast<int>(73)] = 0.0;
        nmodl_eigen_j[static_cast<int>(86)] = 0.0;
        nmodl_eigen_j[static_cast<int>(99)] = 0.0;
        nmodl_eigen_j[static_cast<int>(112)] =  -Q10dr * Rr4 * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(125)] = 0.0;
        nmodl_eigen_j[static_cast<int>(138)] = 0.0;
        nmodl_eigen_j[static_cast<int>(151)] = 0.0;
        nmodl_eigen_j[static_cast<int>(164)] = 0.0;
        nmodl_eigen_f[static_cast<int>(9)] =  -Q10oc * Rc1 * nmodl_eigen_x[static_cast<int>(9)] * dt + Q10oc * Ro1 * nmodl_eigen_x[static_cast<int>(1)] * dt - nmodl_eigen_x[static_cast<int>(9)] + old_O1;
        nmodl_eigen_j[static_cast<int>(9)] = 0.0;
        nmodl_eigen_j[static_cast<int>(22)] = Q10oc * Ro1 * dt;
        nmodl_eigen_j[static_cast<int>(35)] = 0.0;
        nmodl_eigen_j[static_cast<int>(48)] = 0.0;
        nmodl_eigen_j[static_cast<int>(61)] = 0.0;
        nmodl_eigen_j[static_cast<int>(74)] = 0.0;
        nmodl_eigen_j[static_cast<int>(87)] = 0.0;
        nmodl_eigen_j[static_cast<int>(100)] = 0.0;
        nmodl_eigen_j[static_cast<int>(113)] = 0.0;
        nmodl_eigen_j[static_cast<int>(126)] =  -Q10oc * Rc1 * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(139)] = 0.0;
        nmodl_eigen_j[static_cast<int>(152)] = 0.0;
        nmodl_eigen_j[static_cast<int>(165)] = 0.0;
        nmodl_eigen_f[static_cast<int>(10)] =  -Q10oc * Rc2 * nmodl_eigen_x[static_cast<int>(10)] * dt + Q10oc * Ro2 * nmodl_eigen_x[static_cast<int>(2)] * dt - nmodl_eigen_x[static_cast<int>(10)] + old_O2;
        nmodl_eigen_j[static_cast<int>(10)] = 0.0;
        nmodl_eigen_j[static_cast<int>(23)] = 0.0;
        nmodl_eigen_j[static_cast<int>(36)] = Q10oc * Ro2 * dt;
        nmodl_eigen_j[static_cast<int>(49)] = 0.0;
        nmodl_eigen_j[static_cast<int>(62)] = 0.0;
        nmodl_eigen_j[static_cast<int>(75)] = 0.0;
        nmodl_eigen_j[static_cast<int>(88)] = 0.0;
        nmodl_eigen_j[static_cast<int>(101)] = 0.0;
        nmodl_eigen_j[static_cast<int>(114)] = 0.0;
        nmodl_eigen_j[static_cast<int>(127)] = 0.0;
        nmodl_eigen_j[static_cast<int>(140)] =  -Q10oc * Rc2 * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(153)] = 0.0;
        nmodl_eigen_j[static_cast<int>(166)] = 0.0;
        nmodl_eigen_f[static_cast<int>(11)] =  -Q10oc * Rc3 * nmodl_eigen_x[static_cast<int>(11)] * dt + Q10oc * Ro3 * nmodl_eigen_x[static_cast<int>(3)] * dt - nmodl_eigen_x[static_cast<int>(11)] + old_O3;
        nmodl_eigen_j[static_cast<int>(11)] = 0.0;
        nmodl_eigen_j[static_cast<int>(24)] = 0.0;
        nmodl_eigen_j[static_cast<int>(37)] = 0.0;
        nmodl_eigen_j[static_cast<int>(50)] = Q10oc * Ro3 * dt;
        nmodl_eigen_j[static_cast<int>(63)] = 0.0;
        nmodl_eigen_j[static_cast<int>(76)] = 0.0;
        nmodl_eigen_j[static_cast<int>(89)] = 0.0;
        nmodl_eigen_j[static_cast<int>(102)] = 0.0;
        nmodl_eigen_j[static_cast<int>(115)] = 0.0;
        nmodl_eigen_j[static_cast<int>(128)] = 0.0;
        nmodl_eigen_j[static_cast<int>(141)] = 0.0;
        nmodl_eigen_j[static_cast<int>(154)] =  -Q10oc * Rc3 * dt - 1.0;
        nmodl_eigen_j[static_cast<int>(167)] = 0.0;
        nmodl_eigen_f[static_cast<int>(12)] =  -nmodl_eigen_x[static_cast<int>(0)] - nmodl_eigen_x[static_cast<int>(10)] - nmodl_eigen_x[static_cast<int>(11)] - nmodl_eigen_x[static_cast<int>(12)] - nmodl_eigen_x[static_cast<int>(1)] - nmodl_eigen_x[static_cast<int>(2)] - nmodl_eigen_x[static_cast<int>(3)] - nmodl_eigen_x[static_cast<int>(4)] - nmodl_eigen_x[static_cast<int>(5)] - nmodl_eigen_x[static_cast<int>(6)] - nmodl_eigen_x[static_cast<int>(7)] - nmodl_eigen_x[static_cast<int>(8)] - nmodl_eigen_x[static_cast<int>(9)] + 1.0;
        nmodl_eigen_j[static_cast<int>(12)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(25)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(38)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(51)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(64)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(77)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(90)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(103)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(116)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(129)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(142)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(155)] =  -1.0;
        nmodl_eigen_j[static_cast<int>(168)] =  -1.0;

        X -= J.partialPivLu().solve(F);

        C0[id] = nmodl_eigen_x[static_cast<int>(0)];
        C1[id] = nmodl_eigen_x[static_cast<int>(1)];
        C2[id] = nmodl_eigen_x[static_cast<int>(2)];
        C3[id] = nmodl_eigen_x[static_cast<int>(3)];
        C4[id] = nmodl_eigen_x[static_cast<int>(4)];
        D1[id] = nmodl_eigen_x[static_cast<int>(5)];
        D2[id] = nmodl_eigen_x[static_cast<int>(6)];
        D3[id] = nmodl_eigen_x[static_cast<int>(7)];
        D4[id] = nmodl_eigen_x[static_cast<int>(8)];
        O1[id] = nmodl_eigen_x[static_cast<int>(9)];
        O2[id] = nmodl_eigen_x[static_cast<int>(10)];
        O3[id] = nmodl_eigen_x[static_cast<int>(11)];
        O4[id] = nmodl_eigen_x[static_cast<int>(12)];
    }
}
