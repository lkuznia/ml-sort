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

    def generate_x(self, instances, length, max_val, name):
        self.train_cases[name] = defaultdict()
        self.train_cases[name]['x'] = np.random.randint(0, max_val, size=(instances, length))

    def binarize_x(self, name):
        if name not in self.train_cases.keys():
            print('Please create X matrix for {} with the generate_x method'.format(name))
        else:
            vect_binarize = np.vectorize(binarize, otypes=[np.ndarray], )
            self.train_cases[name]['binary_x'] = vect_binarize(self.train_cases[name]['x'])

    def generate_y(self, name):
        if name not in self.train_cases.keys():
            print('Please create X matrix for {} with the generate_x method'.format(name))
        else:
            self.train_cases[name]['y'] = np.argsort(self.train_cases[name]['x'])

    def generate_case(self, instances, length, max_val, name):
        self.max_bin_magnitude = len(format(max_val, 'b'))
        self.generate_x(instances, length, max_val, name)
        self.generate_y(name)


data = DataGenerator()
MAX_PWR = 5
LIST_LEN = 5
data.generate_case(10, LIST_LEN, 2**MAX_PWR + 1, 'test')
data.binarize_x('test')
rd = data.train_cases['test']['x']
b_rd = data.train_cases['test']['binary_x']

tt = vect_binarize(data.train_cases['test']['x'], 5)

print(rd)
print(t_rd)
print(tt)
