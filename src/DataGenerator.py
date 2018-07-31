import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import permutations
from math import factorial


class DataGenerator(object):
    """
        Create and store data for project
    """

    def __init__(self, instances, length, max_val):
        self.length = length
        self.instances = instances
        self.max_val = max_val
        self.train_cases = defaultdict()
        self.max_bin_magnitude = len(format(self.max_val, 'b'))
        self.sort_orders = None

    @staticmethod
    def binarize(x, max_pwr):
        bin_str = format(x, '0{}b'.format(max_pwr+1))
        bin_vect = [int(k) for k in bin_str[::-1]]
        return bin_vect

    @staticmethod
    def e_i(size, index):
        arr = np.zeros(size)
        arr[index] = 1
        return arr

    def generate_x(self, instances, length, max_val):
        self.train_cases['x'] = np.random.randint(0, max_val, size=(instances, length))
        self.binarize_x()

    def binarize_x(self):
        if 'x' not in self.train_cases.keys():
            print('Please create X matrix for {} with the generate_x method')
        else:
            # vect_binarize = np.vectorize(DataGenerator.binarize, signature='(m),()->(m,n)')#  otypes=[np.ndarray])
            # self.train_cases['binary_x'] = np.ndarray(vect_binarize(self.train_cases['x'], self.max_bin_magnitude))
            bin_train_data = []
            for case in self.train_cases['x']:
                case_list = []
                for val in case:
                    case_list.append(DataGenerator.binarize(val, self.max_bin_magnitude))
                bin_train_data.append(case_list)
            self.train_cases['binary_x'] = bin_train_data

    def labelize_y(self):
        if 'y' not in self.train_cases.keys():
            print('Please create Y vector with the generate_y method')
        else:
            labels = []
            for y in self.train_cases['y']:
                labels.append(self.sort_orders[tuple(y)]['label'])
            self.train_cases['label_y'] = labels

    def binarize_y(self):
        if 'y' not in self.train_cases.keys():
            print('Please create Y vector with the generate_y method')
        else:
            sorted_label = []
            for y in self.train_cases['y']:
                sorted_label.append(self.sort_orders[tuple(y)]['binary'])
            self.train_cases['binary_y'] = sorted_label

    def generate_y(self):
        if 'x' not in self.train_cases.keys():
            print('Please create X matrix with the generate_x method')
        else:
            self.train_cases['y'] = np.argsort(self.train_cases['x'])
            self.binarize_y()
            self.labelize_y()

    def generate_case(self):
        self.generate_x(self.instances, self.length, self.max_val)
        self.generate_sorted_options(self.length)
        self.generate_y()

    def generate_sorted_options(self, length):
        nb_sort_orders = factorial(length)
        sort_orders = defaultdict()
        for index, order in enumerate(list(permutations(range(length)))):
            sort_orders[order] = {'binary': DataGenerator.e_i(nb_sort_orders, index), 'label': index}
        self.sort_orders = sort_orders
