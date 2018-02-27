import numpy as np

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

y = np.array([[0,0,0],
             [0,1,1],
             [0,1,1],
             [1,1,0]])

# Hyperparameters
learning_rate = 0.003
batch_size = 4
num_iterations = 10000
max_error = 0.0001

# Placeholder for input to network (an entry point into the computational graph for input)
input_placeholder = tf.placeholder(tf.float32, [batch_size,3], name = 'input_placeholder')

# First fully connected layer
# input shape: [batch_size, 3]
# output shape: [batch_size, 4]
weights_1 = tf.Variable(tf.truncated_normal([3, 4], stddev=0.1), name="Weights_1")
bias_1 = tf.Variable(tf.constant(1, shape=[4]), name="Bias_1")
out_1 = tf.nn.sigmoid(tf.matmul(input_placeholder, weights_1) + bias_1)
# Note: Sigmoid activation used here but not the best idea in general
#       (flat on ends so gradient very small)

# Second fully connected layer
# input shape: [batch_size, 4]
# output shape: [batch_size, 3]
weights_2 = tf.Variable(tf.truncated_normal([4, 3], stddev=0.1), name="Weights_2")
bias_2 = tf.Variable(tf.constant(1, shape=[3]), name="Bias_2")
out_2 = tf.nn.sigmoid(tf.matmul(out_1, weights_2) + bias_2)
# Note: Sigmoid activation works well for binary classification as in this case

# Optimization
# loss: Mean Squared Error (MSE)
#       Note:   We are considering this a classification problem but using MSE
#               MSE will work but should usually use something like cross entropy for classifcation
# optimizer: Gradient Descent Optimizer
#       Note:   Other optimizers are available that add a variety of optimizations
#               Tricks such as adaptive learning rate, learning rate decay, and momentum are often useful
loss = tf.reduce_mean(tf.squared_difference(y, out_2), name="Mean Squared Error")
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Define the TensorFlow session and initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())
