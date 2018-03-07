//
//  test.cpp
//  MachineLearning
//
//  Created by Thomas Fouan on 05/03/2018.
//  Copyright © 2018 Thomas Fouan. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern "C" {
    int add_to_42(int nb) {
        return 42 + nb;
    }
    
    double get_current_weights_value(double* w, double* params, int length) {
        double res = 0;
        int i = 0;
        
        for (i = 0; i < length; i++) {
            res += w[i] * params[i];
        }
        
        return res;
    }
    
    double* linear_create(int nbParameters) {
        //int nbElement = 100;
        //int length = nbParameters * nbElement;
        //double* w = (double*) malloc(sizeof(double) * length);
        double* w = (double*) malloc(sizeof(double) * nbParameters);
        int i;
        
        srand(time(NULL));
        
        for (i = 0; i < nbParameters; i++) {
            w[i] = (double)rand()/RAND_MAX*2.0-1.0;
        }
        
        return w;
    }
    
    void* linear_remove(void* array) {
        
        free(array);
        return NULL;
    }
    
    void linear_train_classification(double* w, int wSize, double* parameters, int nbParameters, int parametersLength, double* results, int resultsLength, int alpha) {
        double** x = (double**) malloc(sizeof(double*) * parametersLength);
        int i, j;
        int stop = 0;
        
        for (i = 0; i < parametersLength; i++) {
            x[i] = (double*) malloc(sizeof(double) * nbParameters + 1);
            
            // paramètre de biais
            x[i][0] = 1;
            for (j = 0; j < nbParameters; j++) {
                x[i][j+1] = parameters[i * nbParameters + j];
            }
        }
        
        do {
            for (i = 0; i < parametersLength; i++) {
                for (j = 0; j < wSize; j++) {
                    double diff = results[i] - get_current_weights_value(w, x[i], wSize);
                    w[i] = w[i] + alpha * (diff) * x[i][j];
                }
            }
            stop++;
        } while(stop < 100);
    }
}
