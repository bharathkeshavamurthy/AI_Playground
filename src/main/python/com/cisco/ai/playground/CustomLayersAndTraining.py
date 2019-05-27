# This entity encapsulates the procedures to develop custom layers in TensorFlow and the high-level Keras API
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems
# Copyright (c) 2019. All Rights Reserved.

import os
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt

tensorflow.enable_eager_execution()


# The full list of available layers in the tensorflow.keras API is given here:
# https://www.tensorflow.org/api_docs/python/tf/keras/layers
# Some examples are Dense (a fully connected network), LSTM, BatchNormalization, Dropout, and Conv2D

# The full list of available activation functions in the tensorflow.keras API are given here:
# https://www.tensorflow.org/api_docs/python/tf/keras/activations
# Some examples are ReLU, Sigmoid, Softmax, and Softsign

# Building custom layers by extending the keras.layers.Layer parent
# __init__, build, and call are the primary calls


# A Custom Layer
class CustomLayer(keras.layers.Layer):

    # The constructor
    def __init__(self, number_of_outputs):
        super(CustomLayer, self).__init__()
        self.number_of_outputs = number_of_outputs
        self.kernel = None

    # Build the model
    def build(self, input_shape):
        self.kernel = self.add_variable('kernel', shape=[int(input_shape[-1]), self.number_of_outputs])

    # The model call
    def call(self, _input, **kwargs):
        return tensorflow.matmul(_input, self.kernel)


# An Iris flower classification engine with a fully connected neural network
class CustomLayersAndTraining(object):

    # Repackaging method
    @staticmethod
    def repack(features, labels):
        features = tensorflow.stack(list(features.values()), axis=1)
        return features, labels

    # The initialization sequence
    def __init__(self):
        print('[INFO] CustomLayersAndTraining Initialization: Bringing things up...')
        file_path = keras.utils.get_file(
            fname=os.path.basename('https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv'),
            origin='https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv')
        column_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']
        label_name = column_names[-1]
        # Generate a dataset from this file
        dataset = tensorflow.contrib.data.make_csv_dataset(file_path, 32, column_names=column_names,
                                                           label_name=label_name,
                                                           num_epochs=1)
        self.features, self.labels = next(iter(dataset))
        # Repack the OrderedDict into a matrix
        self.training_data = dataset.map(self.repack)
        self.features, self.labels = next(iter(self.training_data))
        # The Model
        self.model = None

    # Build the neural network model for this classification problem
    def build_model(self):
        self.model = keras.Sequential([keras.layers.Dense(10, activation=tensorflow.nn.relu, input_shape=(4,)),
                                       keras.layers.Dense(10, activation=tensorflow.nn.relu),
                                       keras.layers.Dense(3)])

    # Evaluate the cost/loss
    def evaluate_cost(self, features, labels):
        _labels = self.model(features)
        return tensorflow.losses.sparse_softmax_cross_entropy(labels=labels, logits=_labels)

    # Evaluate the gradient using the TensorFlow GradientTape feature
    def evaluate_gradient(self, features, labels):
        with tensorflow.GradientTape() as t:
            cost = self.evaluate_cost(features, labels)
        return cost, t.gradient(cost, self.model.trainable_variables)

    # Train the model
    def train(self):
        training_accuracies, training_costs = [], []
        epochs = [k for k in range(0, 5000)]
        optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.0001)
        global_step = tensorflow.Variable(0)
        for epoch in epochs:
            avg_cost = tensorflow.contrib.eager.metrics.Mean()
            avg_accuracy = tensorflow.contrib.eager.metrics.Accuracy()
            print('Epoch: {}, Cost: {}, Accuracy: {}'.format(epoch, avg_cost, avg_accuracy))
            for features, labels in self.training_data:
                cost, gradient = self.evaluate_gradient(features, labels)
                optimizer.apply_gradients(zip(gradient, self.model.trainable_variables), global_step)
                avg_cost(cost)
                avg_accuracy(tensorflow.argmax(self.model(features), axis=1, output_type=tensorflow.int32), labels)
            training_accuracies.append(avg_accuracy.result())
            training_costs.append(avg_cost.result())
        plt.figure()
        plt.plot(epochs, training_costs, label='Training Cost')
        plt.plot(epochs, training_accuracies, label='Training Accuracies')
        plt.xlabel('Epochs')
        plt.ylabel('Training Metrics')
        plt.legend()
        plt.show()

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] CustomLayersAndTraining Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] CustomLayersAndTraining Trigger: Starting system assessment!')
    customTrainer = CustomLayersAndTraining()
    customTrainer.build_model()
    customTrainer.train()
