# This entity encapsulates the basics of Neural Network Training in TensorFlow
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems
# Copyright (c) 2019. All Rights Reserved.

# Keras provides numerous utilities to save your models and use trained models later
# You can also add checkpoints to save training progress and then, re-trigger it (continue it) if it was interrupted.

# Tensors are immutable
# Tensors are multi-dimensional arrays with data-type and shape ([row, column])
# Tensors are backed by accelerator memory (GPU)
# Similar to numpy ndarrays - to and fro conversion is typically easy and straightforward (tensor.tonumpy())
# However, if the tensor is hosted on a GPU memory, the tensor is copied to the host memory in order to convert it to...
# ...a numpy ndarray

# You can use gradient-tape to enable automatic differentiation
# Record all operations on a "tape" and then call gradient on the tensorflow.GradientTape() object initialized using...
# ...the "with" context manager reference
# Persistence in the tape object allows multiple calls to the gradient() method and resources are released when the...
# ...object is garbage collected
# Ifs and whiles are naturally handled by the tensorflow API with respect to the GradientTape feature and is...
# ...accessible in a similar way for higher order gradients

# Tensors in TensorFlow are stateless immutable objects. However, machine learning models require the tensors to...
# ...change during training and testing (prediction operation). In order to reconcile this, we leverage the stateful...
# ...capabilities of Python

# Tensors do have stateful operations built in such as tf.Variable.assign(), tf.assign_sub(), tf.scatter_update(), etc.

import tensorflow
import matplotlib.pyplot as plt

tensorflow.enable_eager_execution()


# The Learning Model
class LearningModel(object):

    # The constructor
    def __init__(self):
        self.m = tensorflow.Variable(5.0)
        self.b = tensorflow.Variable(0.0)

    # The linear core logic
    def __call__(self, x):
        return (self.m * x) + self.b


# The Observation Model
class ObservationModel(object):
    # The true value of 'm'
    TRUE_SLOPE = 3.0

    # The true value of 'b'
    TRUE_Y_INTERCEPT = 2.0

    # The number of observations
    NUMBER_OF_SAMPLES = 100

    # The constructor
    def __init__(self):
        self.inputs = tensorflow.random_normal(shape=[self.NUMBER_OF_SAMPLES])

    # Make observations based on a AWGN linear observation model
    def make_observations(self):
        return (self.inputs * self.TRUE_SLOPE) + self.TRUE_Y_INTERCEPT


# In this class, we look into building a neural network from first principles using TensorFlow
# A simple linear model f(x) = mx + b, with two variables - slope (m) and y-intercept(b)
class FirstPrinciples(object):
    # The number of training periods
    NUMBER_OF_EPOCHS = 1000

    # The initialization sequence
    def __init__(self):
        print('[INFO] FirstPrinciples Initialization: Bringing things up...')
        self.learning_model = LearningModel()
        self.observation_model = ObservationModel()

    # A mean square error cost function
    # J = \mathbb{E}[(\theta - \hat{\theta})^2]
    @staticmethod
    def cost(predicted_value, true_value):
        return tensorflow.reduce_sum(tensorflow.square(predicted_value - true_value))

    # Train the model
    # Gradient Descent
    # m[k+1] = m[k] + (\alpha * \frac{\partial J}{\partial m[k]})
    # b[k+1] = b[k] + (\alpha * \frac{\partial J}{\partial b[k]})
    def train(self, inputs, outputs, alpha):
        with tensorflow.GradientTape() as t:
            current_cost = self.cost(self.learning_model(inputs), outputs)
        dm, db = t.gradient(current_cost, [self.learning_model.m, self.learning_model.b])
        tensorflow.assign_sub(self.learning_model.m, (alpha * dm))
        tensorflow.assign_sub(self.learning_model.b, (alpha * db))

    # Visualize the training - convergence of gradient descent using tensorflow
    def visualize(self):
        fig, ax = plt.subplots()
        slopes, y_intercepts = [], []
        epochs = [k for k in range(0, self.NUMBER_OF_EPOCHS)]
        for epoch in epochs:
            slopes.append(self.learning_model.m.numpy())
            y_intercepts.append(self.learning_model.b.numpy())
            self.train(self.observation_model.inputs, self.observation_model.make_observations(), 0.0001)
            print('Epoch: {}, Slope: {}, Y-Intercept: {}'.format(epoch, self.learning_model.m.numpy(),
                                                                 self.learning_model.b.numpy()))
        ax.plot(epochs, slopes, 'r', label='Descent Progress of the slope')
        ax.plot(epochs, y_intercepts, label='Descent Progress of the y-intercept')
        ax.plot(epochs,
                [self.observation_model.TRUE_SLOPE for k in range(0, self.NUMBER_OF_EPOCHS)],
                label='True value of the slope')
        ax.plot(epochs, [self.observation_model.TRUE_Y_INTERCEPT for k in range(0, self.NUMBER_OF_EPOCHS)],
                label='True value of the y-intercept')
        plt.xlabel('Epochs')
        plt.ylabel('Parameters under analysis')
        plt.legend()
        plt.show()

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] FirstPrinciples Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] FirstPrinciples Trigger: Starting system assessment!')
    gradientDescentEngine = FirstPrinciples()
    gradientDescentEngine.visualize()
