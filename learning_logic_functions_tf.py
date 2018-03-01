import numpy as np
import tensorflow as tf

# Input array
# All combinations of 0 and 1
X = np.array([[0,0],
            [0,1],
            [1,0],
            [1,1]])

# Labels array
# Values of (AND, OR, XOR) corresponding to input array
y = np.array([[0,0,0],
             [0,1,1],
             [0,1,1],
             [1,1,0]], dtype=np.float32)

# Hyperparameters
# Parameters that determine the properties of the model
# (as opposed to the parameters inside of the model itself)
learning_rate = 0.03
batch_size = 4
num_iterations = 100000
max_error = 0.0001

# Placeholder for input to network (an entry point into the computational graph for input)
input_placeholder = tf.placeholder(tf.float32, [batch_size,2], name = 'Input_Placeholder')
labels_placeholder = tf.placeholder(tf.float32, [batch_size,3], name = 'Labels_Placeholder')

# First fully connected layer
# input shape: [batch_size, 3]
# output shape: [batch_size, 4]

weights_1 = tf.Variable(tf.truncated_normal([2, 4], stddev=0.1), name="Weights_1")
bias_1 = tf.Variable(tf.truncated_normal([4], stddev=0.1), name="Bias_1")
logits_1 = tf.matmul(input_placeholder, weights_1) + bias_1
out_1 = tf.nn.sigmoid(logits_1)
# Note: Sigmoid activation used here but not the best idea in general (flat
#       on ends so gradient very small). In general, newer activation functions
#       such as ReLU, Leaky ReLU, and ELU are much better for deeper nets.

# Second fully connected layer
# input shape: [batch_size, 4]
# output shape: [batch_size, 3]

weights_2 = tf.Variable(tf.truncated_normal([4, 3], stddev=0.1), name="Weights_2")
bias_2 = tf.Variable(tf.truncated_normal([3], stddev=0.1), name="Bias_2")
logits_2 = tf.matmul(out_1, weights_2) + bias_2
out_2 = tf.nn.sigmoid(logits_2)
# Note: Sigmoid activation works well for binary classification as in this case
#       For multiclass classification, you should likely used

# Optimization
# loss: Mean Squared Error (MSE)
#       Note:   We are considering this a classification problem but using MSE.
#               Using cross entropy would help with training.
# optimizer: Gradient Descent Optimizer
#       Note:   Other optimizers are available that add a variety of optimizations and
#               tricks such as adaptive learning rate, learning rate decay, and momentum.

# loss = tf.reduce_mean(tf.squared_difference(labels_placeholder, out_2), name="Mean_Squared_Error")
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels_placeholder, logits = logits_2)
# train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Define the TensorFlow session and initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1, num_iterations+1):
    if i%1000 == 0:
        curr_loss = sess.run(loss, feed_dict = {input_placeholder: X, labels_placeholder:y})
        print("Train step: {}\n\tLoss: {}".format(i, curr_loss))
    sess.run(train, feed_dict = {input_placeholder: X, labels_placeholder:y})

print(sess.run(out_2, feed_dict = {input_placeholder: X, labels_placeholder:y}))
