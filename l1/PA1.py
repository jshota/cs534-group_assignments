import datetime
import time
import numpy as np
import helper as hp

class LinearRegression:
    def __init__(self, dataset, target_value):
        self.dataset = dataset
        self.target_value = target_value

training_file = 'PA1_train.csv'

dataset = hp.import_data(training_file)
x = np.matrix(dataset[0])[:,5]

print(np.average(x), np.std(x), np.amax(x)-np.amin(x))