import pandas as pd
import numpy as np
from collections import defaultdict


def binarize(x, max_pwr):
    bin_str = format(x, '0{}b'.format(max_pwr+1))
    bin_vect = np.array([int(k) for k in bin_str[::-1]])
    return bin_vect


class DataGenerator(object):
    """
        Create and store data for project
    """

    def __init__(self):
        # self.length = length
        # self.instances = instances
        self.train_cases = defaultdict()
        self.max_bin_magnitude = None

    def generate_x(self, instances, length, max_val):
        self.train_cases['x'] = np.random.randint(0, max_val, size=(instances, length))

    def binarize_x(self):
        if 'x' not in self.train_cases.keys():
            print('Please create X matrix for {} with the generate_x method')
        else:
            vect_binarize = np.vectorize(binarize, otypes=[np.ndarray])
            self.train_cases['binary_x'] = vect_binarize(self.train_cases['x'], self.max_bin_magnitude)

    def generate_y(self):
        if 'x' not in self.train_cases.keys():
            print('Please create X matrix with the generate_x method')
        else:
            self.train_cases['y'] = np.argsort(self.train_cases['x'])

    def generate_case(self, instances, length, max_val):
        self.max_bin_magnitude = len(format(max_val, 'b'))
        self.generate_x(instances, length, max_val)
        self.generate_y()


data = DataGenerator()
MAX_PWR = 5
LIST_LEN = 5
data.generate_case(10, LIST_LEN, 2**MAX_PWR + 1)
data.binarize_x()
rd = data.train_cases['x']
b_rd = data.train_cases['binary_x']

print(rd)
print(b_rd)
