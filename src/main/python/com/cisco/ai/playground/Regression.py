# This entity details a simple regression problem leveraging the capabilities of TensorFlow and the high-level Keras API
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems
# Copyright (c) 2019. All Rights Reserved.

from __future__ import absolute_import, division, print_function

import pandas
import tensorflow
import seaborn
from tensorflow import keras
import matplotlib.pyplot as plt


# This class is used to build and train a regression model and use this trained model to predict the...
# ...fuel-efficiency of automobiles
class Regression(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] Regression Initialization: Bringing things up...')
        column_names = ['Miles per Gallon',
                        'Number of Cylinders',
                        'Displacement',
                        'Brake Horse Power',
                        'Weight',
                        'Acceleration',
                        'Model Year',
                        'Origin']
        # Get the dataset
        dataset_path = keras.utils.get_file(
            'auto-mpg.data',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')
        # Convert to a pandas data frame
        pandas_dataframe_object = pandas.read_csv(dataset_path,
                                                  verbose=0,
                                                  na_values='?',
                                                  sep=' ',
                                                  names=column_names,
                                                  comment='\t',
                                                  skipinitialspace=True)
        data_set = pandas_dataframe_object.copy()
        # Prepare the dataset
        data_set = data_set.dropna()
        origin = data_set.pop('Origin')
        data_set['USA'] = (origin == 1) * 1.0
        data_set['Europe'] = (origin == 2) * 1.0
        data_set['India'] = (origin == 3) * 1.0
        # Split the dataset into training and test data
        self.training_data = data_set.sample(frac=0.8, random_state=0)
        self.test_data = data_set.drop(self.training_data.index)
        # Visualize the joint distribution between a few columns
        seaborn.pairplot(self.training_data[['Miles per Gallon',
                                             'Number of Cylinders',
                                             'Displacement',
                                             'Brake Horse Power']],
                         diag_kind='kde')
        # Get the training data statistics
        # Prediction label = MPG - Pop it out
        self.training_data_stats = self.training_data.describe()
        self.training_data_stats.pop('Miles per Gallon')
        self.training_data_stats = self.training_data_stats.transpose()
        self.training_labels = self.training_data.pop('Miles per Gallon')
        self.test_labels = self.test_data.pop('Miles per Gallon')
        # Normalize the training and the test data so that they are in the same scale for faster convergence
        self.normalized_training_data = self.normalize(self.training_data)
        self.normalized_test_data = self.normalize(self.test_data)
        # The Model
        self.model = None

    # Normalize the data
    def normalize(self, x):
        return (x - self.training_data_stats['mean']) / self.training_data_stats['std']

    # Build, Compile, and Train the model
    def build(self):
        # Build the model (Two Dense ReLU layers and one output layer)
        self.model = keras.Sequential([keras.layers.Dense(64,
                                                          activation=tensorflow.nn.relu,
                                                          input_shape=[len(self.training_data.keys())]),
                                       keras.layers.Dense(64, activation=tensorflow.nn.relu),
                                       keras.layers.Dense(1)])
        # Compile the model with a mean squared error cost function and RMSProp optimizer
        self.model.compile(optimizer=keras.optimizers.RMSprop(0.001),
                           loss='mean_squared_error',
                           metrics=['mean_absolute_error', 'mean_squared_error'])
        self.model.summary()
        # Early Stopping Callback
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=0, patience=10)
        # Train the model with validation split being 0.2
        history = self.model.fit(self.normalized_training_data, self.training_labels,
                                 epochs=10000, validation_split=0.1, verbose=0,
                                 callbacks=[early_stopping_callback])
        history_data_frame = pandas.DataFrame(history.history)
        history_data_frame['epoch'] = history.epoch
        print(history_data_frame.tail())
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error ' + r'$\mathbb{E}[|\theta - \hat{\theta}|]$')
        plt.plot(history_data_frame['epoch'],
                 history_data_frame['mean_absolute_error'],
                 label='Training Data Set')
        plt.plot(history_data_frame['epoch'],
                 history_data_frame['val_mean_absolute_error'],
                 label='Validation Data Set')
        plt.legend()
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error ' + r'$\mathbb{E}[(\theta - \hat{\theta})^2]$')
        plt.plot(history_data_frame['epoch'],
                 history_data_frame['mean_squared_error'],
                 label='Training Data Set')
        plt.plot(history_data_frame['epoch'],
                 history_data_frame['val_mean_squared_error'],
                 label='Validation Data Set')
        plt.legend()
        plt.show()

    # Run a simple sample prediction using the model
    def predict(self):
        example_batch = self.normalized_training_data[:10]
        example_batch_prediction_result = self.model.predict(example_batch)
        print(example_batch_prediction_result)

    # Evaluate the model
    def evaluate(self):
        loss, mae, mse = self.model.evaluate(self.normalized_test_data, self.test_labels, verbose=0)
        print('[INFO] Prediction Loss = {}, Mean Absolute Error = {}, Mean Squared Error = {}'.format(loss, mae,
                                                                                                      mse))

    # Visualize the prediction accuracy
    def visualize(self):
        predictions = self.model.predict(self.normalized_test_data).flatten()
        plt.scatter(self.test_labels, predictions)
        plt.xlabel('True MPG Values for the Test Data Set')
        plt.ylabel('Predicted MPG Values for the Test Data Set')
        plt.axis('equal')
        plt.axis('square')
        _ = plt.plot([-100, 100], [-100, 100])
        plt.show()
        prediction_error = predictions - self.test_labels
        plt.hist(prediction_error, bins=25)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency/Count')
        plt.show()

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Regression Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] Regression Trigger: Starting system assessment...')
    regression_engine = Regression()
    regression_engine.build()
    regression_engine.evaluate()
    regression_engine.visualize()
