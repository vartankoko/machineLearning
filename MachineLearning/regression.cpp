//
//  regression.cpp
//  MachineLearning
//
//  Created by Thomas Fouan on 07/03/2018.
//  Copyright © 2018 Thomas Fouan. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "Eigen/Dense"

extern "C" {
    
    double* linear_create_regression(int nbParameters) {
        double* w = (double*) malloc(sizeof(double) * nbParameters);
        int i;
        
        for (i = 0; i < nbParameters; i++) {
            w[i] = 0.0;
        }
        
        return w;
    }
    
    void linear_train_regression(double* w, double* parameters, int nbParameters, int parametersLength, double* results, int resultsLength) {
        int i = 0;
        auto wSize = nbParameters + 1;
        
        Eigen::MatrixXd x(parametersLength / nbParameters, nbParameters + 1);
        
        for (i = 0; i < parametersLength / nbParameters; i++) {
            x(i, 0) = 1;
            for (int j = 0; j < nbParameters; j++) {
                x(i, j + 1) = parameters[i * nbParameters + j];
            }
        }
        
        Eigen::MatrixXd y(resultsLength, 1);
        for (i = 0; i < resultsLength; i++) {
            y(i, 0) = results[i];
        }
        
        Eigen::MatrixXd xT = x.transpose();
        Eigen::MatrixXd step1 = xT * x;
        Eigen::MatrixXd step2 = step1.inverse();
        Eigen::MatrixXd step3 = step2 * xT;
        Eigen::MatrixXd matrixW = step3 * y;
        //Eigen::MatrixXd matrixW = ((xT * x).inverse() * xT) * y;
        
        for (i = 0; i < wSize; i++) {
            w[i] = matrixW(i, 0);
        }
    }
}
