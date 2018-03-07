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
    
    int get_current_weights_value(double* w, double* params, int length) {
        double res = 0;
        int i = 0;
        
        for (i = 0; i < length; i++) {
            res += w[i] * params[i];
        }
        
        return res < 0 ? -1 : 1;
    }
    
    double* linear_create(int nbParameters) {
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
    
    void linear_train_classification(double* w, double* parameters, int nbParameters, int parametersLength, double* results, int resultsLength, double alpha, int maxIter) {
        double** x = (double**) malloc(sizeof(double*) * parametersLength / nbParameters);
        int i, j;
        int stop = 0;
        double diff = 0;
        auto wSize = nbParameters + 1;
        
        srand(time(NULL));
        
        for (i = 0; i < parametersLength / nbParameters; i++) {
            x[i] = (double*) malloc(sizeof(double) * nbParameters + 1);
            
            // paramètre de biais
            x[i][0] = 1;
            for (j = 0; j < nbParameters; j++) {
                x[i][j+1] = parameters[i * nbParameters + j];
            }
        }
        
        do {
            int index = rand()%(parametersLength / nbParameters);
            for (j = 0; j < wSize; j++) {
                int currentRes = get_current_weights_value(w, x[index], wSize);
                diff = results[index] - currentRes;
                
                w[j] = w[j] + alpha * (diff) * x[index][j];
            }
            stop++;
        } while(stop < maxIter);
        
        for (i = 0; i < nbParameters + 1; i++) {
            free(x[i]);
        }
        free(x);
    }
}
