#ifndef NLSOLVER_H
#define NLSOLVER_H

#include <Eigen/Dense>
#include <iostream>
#include <functional>
#include <string>

using namespace std;
using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::VectorXd;
using Eigen::all;

class NLSolver {

    public:
        NLSolver();
        NLSolver(VectorXd b_init, VectorXd x, VectorXd y, 
                 function<VectorXd(VectorXd, VectorXd)> func,
                 string method, float lambd_init);
        VectorXd fit(float step_init, int max_iter, float stop_tolerance);
        VectorXd predict();
        // VectorXd getx();
        // VectorXd gety();
        // VectorXd getb();
        // VectorXd gety_pred();
        // float getlambd();
        // string getmethod();

    private:
        VectorXd b, x, y, y_pred;
        function<VectorXd(VectorXd, VectorXd)> func;
        float lambd;
        string method;
        MatrixXd compute_jacobian(VectorXd b, VectorXd b0);
        VectorXd compute_residual(VectorXd b);
        float compute_rmse(VectorXd b);
        MatrixXd compute_pseudoinverse(MatrixXd X);
        MatrixXd compute_lm_inverse(MatrixXd X, float lambda);

};

#endif