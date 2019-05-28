# This entity encapsulates the design of a Convolutional Neural Network using TensorFlow Estimators
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems
# Copyright (c) 2019. All Rights Reserved.

import numpy
import tensorflow

tensorflow.enable_eager_execution()
tensorflow.logging.set_verbosity(tensorflow.logging.INFO)


# This class encapsulates a Handwriting Recognition tool using Convolutional Neural Networks (CNN)
class CNNUsingEstimators(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] CNNUsingEstimators Initialization: Bringing things up...')
        # Download and format the dataset
        (self.training_data, self.training_labels), (self.test_data, self.test_labels) = \
            tensorflow.keras.datasets.mnist.load_data()
        self.training_data = self.training_data / numpy.float32(255)
        self.test_data = self.test_data / numpy.float32(255)
        self.training_labels = self.training_labels.astype(numpy.int32)
        self.test_labels = self.test_labels.astype(numpy.int32)

    # The model function for the Convolutional Neural Network
    # A 32 convolutional filter bank extracting 5x5 pixel sub-regions
    # A Pooling layer which performs dimensionality reduction using max-pooling
    # A 64 convolutional filter bank extracting 5x5 pixel sub-regions
    # A Pooling layer which performs dimensionality reduction using max-pooling
    # A Dense Neural Network layer which performs the classification with softmax outputs for the 10 classes (0-9)
    @staticmethod
    def cnn_model_fn(features, labels, mode):
        # Reshape the input
        input_layer = tensorflow.reshape(features['x'], [-1, 28, 28, 1])
        # The first 5x5 ReLU 32 convolutional filter bank
        convolutional_filter_bank_1 = tensorflow.layers.conv2d(input_layer, filters=32, kernel_size=[5, 5],
                                                               activation=tensorflow.nn.relu, padding='same')
        # The first pooling layer 2x2 with stride=2
        pooling_layer_1 = tensorflow.layers.max_pooling2d(convolutional_filter_bank_1, pool_size=[2, 2], strides=2)
        # The second 5x5 ReLU 64 convolutional filter bank
        convolutional_filter_bank_2 = tensorflow.layers.conv2d(pooling_layer_1, filters=64, kernel_size=[5, 5],
                                                               activation=tensorflow.nn.relu, padding='same')
        # The second pooling layer 2x2 with stride=2
        pooling_layer_2 = tensorflow.layers.max_pooling2d(convolutional_filter_bank_2, pool_size=[2, 2], strides=2)
        # Reshape before passing to the dense neural network layer
        flatten = tensorflow.reshape(pooling_layer_2, [-1, 7 * 7 * 64])
        # The first dense neural network layer
        dense_nn_layer_1 = tensorflow.layers.dense(flatten, activation=tensorflow.nn.relu, units=1024)
        # Hinton dropout with rate=0.4
        dropout_layer = tensorflow.layers.dropout(dense_nn_layer_1, 0.4,
                                                  training=(mode == tensorflow.estimator.ModeKeys.TRAIN))
        # The second dense neural network layer
        dense_nn_layer_2 = tensorflow.layers.dense(dropout_layer, units=10)
        predictions = {'classes': tensorflow.argmax(input=dense_nn_layer_2, axis=1),
                       'probabilities': tensorflow.nn.softmax(dense_nn_layer_2, name='softmax_tensor')}
        cost_function = tensorflow.losses.sparse_softmax_cross_entropy(labels, dense_nn_layer_2)
        if mode == tensorflow.estimator.ModeKeys.PREDICT:
            return tensorflow.estimator.EstimatorSpec(predictions=predictions, mode=mode)
        if mode == tensorflow.estimator.ModeKeys.TRAIN:
            optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.0001)
            training_operation = optimizer.minimize(loss=cost_function, global_step=tensorflow.train.get_global_step())
            return tensorflow.estimator.EstimatorSpec(loss=cost_function, mode=mode, train_op=training_operation)
        evaluation_metrics_operations = {'accuracy': tensorflow.metrics.accuracy(labels=labels,
                                                                                 predictions=predictions['classes'])}
        return tensorflow.estimator.EstimatorSpec(mode=mode, loss=cost_function,
                                                  eval_metric_ops=evaluation_metrics_operations)

    # Build, Train, and Evaluate the model
    def train(self):
        # Build the model
        estimator = tensorflow.estimator.Estimator(model_fn=self.cnn_model_fn, model_dir='/tmp/model/cnn/')
        # Logging hook
        tensors_to_log = {'probabilities': 'softmax_tensor'}
        logging_hook = tensorflow.train.LoggingTensorHook(tensors_to_log, every_n_iter=50)
        # The training input function
        training_input_function = tensorflow.estimator.inputs.numpy_input_fn(x={'x': self.training_data},
                                                                             y=self.training_labels,
                                                                             batch_size=100,
                                                                             num_epochs=None,
                                                                             shuffle=True)
        # Train the model
        estimator.train(input_fn=training_input_function, steps=20000, hooks=[logging_hook])
        # The evaluation input function
        evaluation_input_function = tensorflow.estimator.inputs.numpy_input_fn(x={'x': self.test_data},
                                                                               y=self.test_labels,
                                                                               num_epochs=1,
                                                                               shuffle=False)
        # Evaluate the model
        evaluation_results = estimator.evaluate(evaluation_input_function)
        print('[INFO] CNNUsingEstimators train: Evaluation Results - {}'.format(evaluation_results))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] CNNUsingEstimators Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] CNNUsingEstimators Trigger: Starting system assessment!')
    cnn = CNNUsingEstimators()
    cnn.train()
