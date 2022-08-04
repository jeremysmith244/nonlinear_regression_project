#include "NLSolver.h"
#include <stdexcept>
#include <cmath>

using namespace std;

NLSolver::NLSolver() {}

NLSolver::NLSolver(VectorXd b_init, VectorXd x, VectorXd y, 
                   function<VectorXd(VectorXd, VectorXd)> func,
                   string method, float lambd_init) {
    
    b = b_init;
    y_pred = this->func(x, b_init);
    this->x = x;
    this->y = y;
    cout << "Constructed okay" << endl;
    this->func = func;

    if (method == "Gauss-Newton") {
        this->method = method;
        lambd = 0;
    } else if (method == "Levenberg–Marquardt") {
        this->method = method;
    } else {
        throw std::invalid_argument("Invalid ");
    }
        
}

MatrixXd NLSolver::compute_jacobian(VectorXd b, VectorXd b0) {

    int lenx = x.cols();
    int lenb = b.cols();
    VectorXd b_part(lenb);

    MatrixXd J(lenx, lenb);
    VectorXd y0 = compute_residual(b);
    VectorXd delta_res(y0.cols());

    for (int i = 0; i < lenb; i++) {
        b_part = b0;
        b_part(i) = b(i);
        delta_res = y0 - compute_residual(b_part);
        J(all,i) = delta_res / (b(i) - b0(i));
    }
    return J;        
}

VectorXd NLSolver::predict() {
    y_pred = func(x, b);
    return y_pred; 
}

VectorXd NLSolver::compute_residual(VectorXd b) {
    VectorXd y_pred = func(x, b);
    return y - y_pred; 
}

float NLSolver::compute_rmse(VectorXd b) {
    VectorXd r = compute_residual(b);
    return sqrt(r.array().pow(2).mean());
}

MatrixXd NLSolver::compute_pseudoinverse(MatrixXd X) {
    MatrixXd Xt = X.transpose();
    return ((Xt * X).inverse()) * Xt;
}

MatrixXd NLSolver::compute_lm_inverse(MatrixXd X, float lambd) {
    MatrixXd Xt = X.transpose();
    return ((Xt * X + lambd*(Xt * X).diagonal()).inverse()) * Xt;
}

VectorXd NLSolver::fit(float step_init, int max_iter, float stop_tolerance) {
    
    VectorXd b0 = b;
    float rmse_prev;
    b.array() += step_init;
    float rmse = compute_rmse(b);

    VectorXd r(y.cols());
    MatrixXd J(x.cols(), b.cols());
    MatrixXd Ji(b.cols(), x.cols());

    if (method == "Gauss-Newton") {

        for (int i = 0; i < max_iter; i++) {
            rmse_prev = rmse;
            r = compute_residual(b);
            J = compute_jacobian(b, b0);
            Ji = compute_pseudoinverse(J);
            b0 = b;
            b += Ji * r;
            rmse = compute_rmse(b);
            cout << "Gauss-Newton" << "Iteration: " << i << " RMSE: " << rmse << endl;

            if (abs(rmse_prev - rmse) < stop_tolerance)
                break;
        }
        return b;

    } else if (method == "Levenberg-Marquardt") {

        for (int i = 0; i < max_iter; i++) {
            rmse_prev = rmse;
            r = compute_residual(b);
            J = compute_jacobian(b, b0);
            Ji = compute_lm_inverse(J, lambd);
            b0 = b;
            b += Ji * r;
            rmse = compute_rmse(b);
            cout << "Levenberg-Marquardt" << "Iteration: " << i << " RMSE: " << rmse << endl;

            if (abs(rmse_prev - rmse) < stop_tolerance)
                break;
            else if (rmse < rmse_prev)
                lambd *= 0.33;
            else
                lambd *= 2;
        }
        return b;
    } else {
        for (int i = 0; i < b.cols(); i++) {
            b << -999;
        }
        cout << "Invalid Solver Type " << method << endl;
        return b;
    }

}

VectorXd rate_eq(VectorXd x, VectorXd args) {
    VectorXd r(x.cols());
    r = args(0)*x.array() / (args(1) + x.array());
    return r;
};

int main() {

    // Inputs, from wikipedia example
    VectorXd S(7), r(7), b0(2);
    S << 0.038,0.194,0.425,0.626,1.253,2.5,3.74;
    r << 0.05,0.127,0.094,0.2122,0.2729,0.2665,0.3317;
    b0 << 0.1,0.1;
    cout << "Well now that its all setup..." << endl;

    NLSolver nls = NLSolver(b0, S, r, &rate_eq, "Gauss-Newton", 1e-3);
    cout << "An instance of a thing..." << endl;
    VectorXd bfit = nls.fit(1e-6, 100, 1e-10);
    cout << "Finished Fitting, Final B Values: " << bfit << endl;

}