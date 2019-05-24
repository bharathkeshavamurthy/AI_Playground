# This class evaluates the models and policies used in general in order to understand underfitting and overfitting...
# from the perspective of TensorFlow and the high-level Keras API
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems
# Copyright (c) 2019. All Rights Reserved.

import numpy
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt


# This class uses the illustration of classifying movie reviews in order to nail down the concepts of overfitting and...
# underfitting using TensorFlow and the high-level Keras API
class ModelEvaluation(object):
    # The dimensionality of the word index dictionary
    NUM_OF_WORDS = 10000

    # Generate Multi-Hop sequences from the word-indices sequence
    @staticmethod
    def generate_multihop_sequences(sequences, dimension):
        results = numpy.zeros((len(sequences), dimension))
        for i, word_indices in enumerate(sequences):
            results[i, word_indices] = 1.0
        return results

    # The initialization sequence
    def __init__(self):
        print('[INFO] ModelEvaluation Initialization: Bringing things up...')
        (self.training_data, self.training_labels), (self.test_data, self.test_labels) = keras.datasets.imdb.load_data(
            num_words=self.NUM_OF_WORDS)
        self.training_data = self.generate_multihop_sequences(self.training_data, self.NUM_OF_WORDS)
        self.test_data = self.generate_multihop_sequences(self.test_data, self.NUM_OF_WORDS)
        # Baseline Model
        self.baseline_model = None
        self.baseline_model_history = None
        # Smaller Model
        self.smaller_model = None
        self.smaller_model_history = None
        # Bigger Model
        self.bigger_model = None
        self.bigger_model_history = None
        # Baseline Model with L1 weight regularization
        self.baseline_l1_model = None
        self.baseline_l1_model_history = None
        # Baseline Model with L2 weight regularization (weight decay)
        self.baseline_l2_model = None
        self.baseline_l2_model_history = None
        # Baseline Model with Geoff Hinton Dropout
        self.baseline_dropout_model = None
        self.baseline_dropout_history = None

    # Create several models of varying capacity using the Keras API and evaluate them in terms of the training error...
    # ...and the validation error
    def illustrate_varying_levels_of_fitting(self):
        # A Baseline model - Build it, Compile it, and Train it
        self.baseline_model = keras.Sequential([keras.layers.Dense(16, activation=tensorflow.nn.relu,
                                                                   input_shape=(self.NUM_OF_WORDS,)),
                                                keras.layers.Dense(16, activation=tensorflow.nn.relu),
                                                keras.layers.Dense(1, activation=tensorflow.nn.sigmoid)])
        self.baseline_model.compile(optimizer='adam', loss='binary_crossentropy',
                                    metrics=['accuracy', 'binary_crossentropy'])
        self.baseline_model_history = self.baseline_model.fit(self.training_data, self.training_labels, epochs=20,
                                                              batch_size=512,
                                                              verbose=1,
                                                              validation_data=(self.test_data, self.test_labels))
        # A Smaller model - Build it, Compile it, and Train it
        self.smaller_model = keras.Sequential([keras.layers.Dense(4, activation=tensorflow.nn.relu,
                                                                  input_shape=(self.NUM_OF_WORDS,)),
                                               keras.layers.Dense(4, activation=tensorflow.nn.relu),
                                               keras.layers.Dense(1, activation=tensorflow.nn.sigmoid)])
        self.smaller_model.compile(optimizer='adam', loss='binary_crossentropy',
                                   metrics=['accuracy', 'binary_crossentropy'])
        self.smaller_model_history = self.smaller_model.fit(self.training_data, self.training_labels, epochs=20,
                                                            batch_size=512,
                                                            verbose=1,
                                                            validation_data=(self.test_data, self.test_labels))
        # A Bigger model - Build it, Compile it, and Train it
        self.bigger_model = keras.Sequential([keras.layers.Dense(512, activation=tensorflow.nn.relu,
                                                                 input_shape=(self.NUM_OF_WORDS,)),
                                              keras.layers.Dense(512, activation=tensorflow.nn.relu),
                                              keras.layers.Dense(1, activation=tensorflow.nn.sigmoid)])
        self.bigger_model.compile(optimizer='adam', loss='binary_crossentropy',
                                  metrics=['accuracy', 'binary_crossentropy'])
        self.bigger_model_history = self.bigger_model.fit(self.training_data, self.training_labels, epochs=20,
                                                          batch_size=512,
                                                          verbose=1, validation_data=(self.test_data, self.test_labels))
        # Technique 1: Weight Regularization
        self.apply_weight_regularization()
        # Technique 2: Geoff Hinton Dropout
        self.geoff_hinton_dropout()
        self.visualize_histories([('Baseline Model', self.baseline_model_history),
                                  ('Smaller Model', self.smaller_model_history),
                                  ('Bigger Model', self.bigger_model_history),
                                  ('Baseline Model with L1 weight regularization', self.baseline_l1_model_history),
                                  ('Baseline Model with L2 weight regularization', self.baseline_l2_model_history),
                                  ('Baseline Model with Hinton Dropout', self.baseline_dropout_history)])

    @staticmethod
    # Evaluate the model fit (overfitting or underfitting) by visualizing the histories
    def visualize_histories(histories):
        print('[INFO] ModelEvaluation visualize_histories: Visualizing the fit for the three given models...')
        plt.figure()
        for model_name, history in histories:
            val_obj = plt.plot(history.epoch, history.history['val_binary_crossentropy'], '--',
                               label=model_name + ' Validation')
            plt.plot(history.epoch, history.history['binary_crossentropy'], color=val_obj[0].get_color(),
                     label=model_name + ' Training')
        plt.xlabel('Epochs')
        plt.ylabel('Cost Function: Binary Cross-Entropy')
        plt.suptitle('Cost Function Convergence Analysis for different models')
        plt.legend()
        plt.show()

    # Occam's Razor: Simpler model (lower number of features) generally fare better

    # Apply L1 and L2 weight regularization (Weight Decay)
    # The cost function is penalized for larger weights proportional to the Lp-norm of the weights
    # Lp norm is defined as ||x||_p\ \equiv\ [\sum_{i=1}^n\ (x_i)^p]^(\frac{1}{p})
    def apply_weight_regularization(self):
        self.baseline_l1_model = keras.Sequential([keras.layers.Dense(16, activation=tensorflow.nn.relu,
                                                                      kernel_regularizer=keras.regularizers.l1(0.001),
                                                                      input_shape=(self.NUM_OF_WORDS,)),
                                                   keras.layers.Dense(16, activation=tensorflow.nn.relu,
                                                                      kernel_regularizer=keras.regularizers.l1(0.001)),
                                                   keras.layers.Dense(1, activation=tensorflow.nn.sigmoid)])
        self.baseline_l1_model.compile(optimizer='adam', loss='binary_crossentropy',
                                       metrics=['accuracy', 'binary_crossentropy'])
        self.baseline_l1_model_history = self.baseline_l1_model.fit(self.training_data, self.training_labels,
                                                                    batch_size=512, epochs=20, verbose=1,
                                                                    validation_data=(self.test_data, self.test_labels))
        self.baseline_l2_model = keras.Sequential([keras.layers.Dense(16, activation=tensorflow.nn.relu,
                                                                      kernel_regularizer=keras.regularizers.l2(0.001)),
                                                   keras.layers.Dense(16, activation=tensorflow.nn.relu,
                                                                      kernel_regularizer=keras.regularizers.l2(0.001)),
                                                   keras.layers.Dense(1, activation=tensorflow.nn.sigmoid)])
        self.baseline_l2_model.compile(optimizer='adam', loss='binary_crossentropy',
                                       metrics=['accuracy', 'binary_crossentropy'])
        self.baseline_l2_model_history = self.baseline_l2_model.fit(self.training_data, self.training_labels,
                                                                    batch_size=512, epochs=20, verbose=1,
                                                                    validation_data=(self.test_data, self.test_labels))

    # Apply Geoff Hinton's Dropout
    # Randomly dropping out output features of a layer during training
    # At test time, no feature dropping is done. Instead, the layer's output is scaled down by the dropout factor.
    def geoff_hinton_dropout(self):
        self.baseline_dropout_model = keras.Sequential([keras.layers.Dense(16, activation=tensorflow.nn.relu,
                                                                           input_shape=(self.NUM_OF_WORDS,)),
                                                        keras.layers.Dropout(0.4),
                                                        keras.layers.Dense(16, activation=tensorflow.nn.relu),
                                                        keras.layers.Dropout(0.4),
                                                        keras.layers.Dense(1, activation=tensorflow.nn.sigmoid)])
        self.baseline_dropout_model.compile(optimizer='adam', loss='binary_crossentropy',
                                            metrics=['accuracy', 'binary_crossentropy'])
        self.baseline_dropout_history = self.baseline_dropout_model.fit(self.training_data, self.training_labels,
                                                                        batch_size=512, epochs=20, verbose=1,
                                                                        validation_data=(self.test_data,
                                                                                         self.test_labels))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] ModelEvaluation Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] ModelEvaluation Trigger: Starting system assessment!')
    modelEvaluator = ModelEvaluation()
    modelEvaluator.illustrate_varying_levels_of_fitting()
