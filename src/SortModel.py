
import src.DataGenerator as dg
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from math import factorial

MAX_PWR = 5  # This is the maximum value the elements in the vector can take
LIST_LEN = 3  # This is the length of the vector to be sorted
NB_INSTANCES = 100  # This is the number of training instances

NB_HIDDEN = 1000  # number of hidden units
nb_hidden1 = 100
nb_hidden2 = 100

data = dg.DataGenerator(NB_INSTANCES, LIST_LEN, 2**MAX_PWR + 1)
data.generate_case()

train_x = data.train_cases['x']
train_y = data.train_cases['y']
train_x_bin = data.train_cases['binary_x']
train_y_bin = data.train_cases['binary_y']
train_y_label = data.train_cases['label_y']

X = tf.placeholder("float", [None, LIST_LEN])
Y = tf.placeholder("float", [None, factorial(LIST_LEN)])

with tf.name_scope('first_net'):
    hidden1 = fully_connected(X, nb_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, nb_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, )