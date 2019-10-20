import datetime
import time
import numpy as np
import helper as hp
import matplotlib.pyplot as plt
from numpy import linalg as la

class LinearRegression:
    def __init__(self, features, target_value, val_features, val_target_value):
        self.features = features
        self.target_value = target_value
        self.parameters = []
        self.val_features = val_features
        self.val_target_value = val_target_value
        self.file_train = None
        self.file_val = None
        
    def weight_value(self):
        '''Calculate the weight.

        Returns:
            nparray: contains the weight value.
        '''
        x = np.dot(self.features, self.parameters)
        weight = np.dot(self.features.T, (x - self.target_value))
        return weight

    def sse_value(self, is_validate):
        '''Calculate the Lost Function (SSE).
        
        Args:
            is_validate (bool): true if doing validate, false otherwise.

        Returns:
            int: SSE result.
        '''
        if is_validate:
            theorical_value = np.dot(self.val_features, self.parameters)
            result = np.sum(np.power((self.val_target_value - theorical_value), 2.0)) / 2.0
        else:
            theorical_value = np.dot(self.features, self.parameters)
            result = np.sum(np.power((self.target_value - theorical_value), 2.0)) / 2.0
        return result

    def gradient_descent(self, learning_rate, converge_point, iter_limit, reg_para, is_validate):
        '''Calculate optimized sse by using gradient descent.

        Args:
            learning_rate (float)
            converge_point (float)
            iter_limit (int)
            reg_para (float)
            is_validate (bool)

        Returns:
            nparray: optimized weights
        '''
        is_converge = False
        iter_times = 0
        self.parameters = np.zeros((self.features.T.shape[0], 1))
        self.file_train = open("result_train.csv", "w")
        if is_validate:
            self.file_val = open("result_val.csv", "w")

        while not is_converge:
            weight_value = self.weight_value()
            reg_values = reg_para * self.parameters
            # except bias
            reg_values[0][0] = 0
            weight_value = weight_value + reg_values
            # normalized
            normalized_weight = la.norm(weight_value)
            # get sse and print it to a file
            sse = self.sse_value(False)
            self.file_train.write(str(iter_times+1) + ',' + str(sse) + ',' + str(normalized_weight) + '\n')
            if is_validate:
                self.file_val.write(str(iter_times+1) + ',' +str(self.sse_value(is_validate)) + '\n')
            # check converge
            if normalized_weight < converge_point: break
            # regularize
            self.parameters = self.parameters - (learning_rate * weight_value)
            # check iteration
            iter_times += 1
            if iter_times == iter_limit: break
            # check sse explode
            if sse == float('Inf') or sse == float('NaN'): break
        print("====================================")
        print("Iterate Times:   " + str(iter_times))
        print("SSE:             " + str(self.sse_value(False)))
        print("Validate SSE:    " + str(self.sse_value(True)))

        if self.file_train: self.file_train.close()
        if self.file_val: self.file_val.close()
        return self.parameters

learning_rate   = 10 ** (-5)
converge_point  = 0.5
iter_limit      = 1000000
reg_para        = 10.0 ** (2)
is_validate     = True
is_normalize    = True
training_file   = 'PA1_train.csv'
val_file        = 'PA1_dev.csv'

print("====================================")
print("Learning Rate:               " + str(learning_rate))
print("Convergence Condition:       " + str(converge_point))
print("Iteration Limitation:        " + str(iter_limit))
print("Regularization Coefficient:  " + str(reg_para))
print("====================================")

dataset = hp.import_data(training_file, is_normalize)
testset = hp.import_data(val_file, is_normalize)

features, val_features = np.matrix(dataset[0]), np.matrix(testset[0])
target_value, val_target_value = np.matrix(dataset[1]).T, np.matrix(testset[1]).T

l = LinearRegression(features, 
                     target_value, 
                     val_features, 
                     val_target_value)

weights = l.gradient_descent(learning_rate,
                            float(converge_point),
                            iter_limit,
                            float(reg_para),
                            is_validate)
print("\nOptimized Weights: \n\n" + str(weights))