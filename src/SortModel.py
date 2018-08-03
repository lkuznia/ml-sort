
import src.DataGenerator as dg
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from math import factorial
import numpy as np

MAX_PWR = 5  # This is the maximum value the elements in the vector can take
LIST_LEN = 3  # This is the length of the vector to be sorted
NB_INSTANCES = 1000  # This is the number of training instances
NB_HIDDEN = 1000  # number of hidden units


# We'll want to randomly initialize weights.
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# Define simple 1-layer model
def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)


data = dg.DataGenerator(NB_INSTANCES, LIST_LEN, 2**MAX_PWR + 1)
data.generate_case()

train_x_raw = data.train_cases['x']
train_y_raw = data.train_cases['y']
train_x_bin = data.train_cases['binary_x']
train_x_flat = data.train_cases['flat_binary_x']
train_y_bin = data.train_cases['binary_y']
train_y_label = data.train_cases['label_y']

train_x = data.train_cases['flat_binary_x']
X = tf.placeholder("float", shape=(None, len(train_x[0])))
Y = tf.placeholder("float", shape=(None, factorial(LIST_LEN)))

# Initialize the weights.
w_h = init_weights([len(train_x[0]), NB_HIDDEN])
w_o = init_weights([NB_HIDDEN, factorial(LIST_LEN)])

# Predict y given x using the model.
py_x = model(X, w_h, w_o)

# And we'll make predictions by choosing the largest output.
predict_op = tf.argmax(py_x, 1)

# We'll train our model by minimizing a cost function.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

BATCH_SIZE = 10

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for epoch in range(100):
        # Shuffle the data before each training iteration.
        p = np.random.permutation(range(len(train_x)))
        train_x, train_y = train_x[p], train_y[p]

        # Train in batches of 128 inputs.
        for start in range(0, len(train_x), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_op, feed_dict={X: train_x[start:end], Y: train_y[start:end]})

        # And print the current accuracy on the training data.
        print(epoch, np.mean(np.argmax(train_y, axis=1) ==
                             sess.run(predict_op, feed_dict={X: train_x, Y: train_y})))

    # And now for some fizz buzz
    # numbers = np.arange(1, 101)
    # teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
    # teY = sess.run(predict_op, feed_dict={X: teX})
    # output = np.vectorize(fizz_buzz)(numbers, teY)
    #
    # print(output)

# with tf.name_scope('first_net'):
#     hidden1 = fully_connected(X, nb_hidden1, scope="hidden1")
#     hidden2 = fully_connected(hidden1, nb_hidden2, scope="hidden2")
#     logits = fully_connected(hidden2, )