import pandas as pd
import numpy as np
from collections import defaultdict


class DataGenerator(object):
    """
        Create and store data for project
    """

    def __init__(self):
        # self.length = length
        # self.instances = instances
        self.train_cases = defaultdict()

    def generate_x(self, instances, length, max_val, name):
        self.train_cases[name] = defaultdict()
        self.train_cases[name]['x'] = np.random.randint(-max_val, max_val, size=(instances, length))

    def generate_y(self, name):
        if name not in self.train_cases.keys():
            print('Please create X matrix for {} with the generate_x method'.format(name))
        else:
            self.train_cases[name]['y'] = np.argsort(self.train_cases[name]['x'])

    def generate_case(self, instances, length, max_val, name):
        self.generate_x(instances, length, max_val, name)
        self.generate_y(name)
