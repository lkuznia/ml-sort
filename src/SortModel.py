
import src.DataGenerator as dg
import tensorflow as tf
from math import factorial

MAX_PWR = 5  # This is the maximum value the elements in the vector can take
LIST_LEN = 3  # This is the length of the vector to be sorted
NB_INSTANCES = 100  # This is the number of training instances

NB_HIDDEN = 1000  # number of hidden units

data = dg.DataGenerator(NB_INSTANCES, LIST_LEN, 2**MAX_PWR + 1)
data.generate_case()

train_x = data.train_cases['x']
train_y = data.train_cases['y']
train_x_bin = data.train_cases['binary_x']
train_y_bin = data.train_cases['binary_y']

# X = tf.placeholder("float", [None, ])
# Y = tf.placeholder("float", [None, factorial(LIST_LEN)])
