# |Bleeding-Edge Productions|
# This entity encapsulates an intelligent model to predict the stock prices of a company using RNNs in TensorFlow
# This model can be extended to predict link states in networks [ Link_Up Link_Up Link_Up Link_Down Link_Down ]
# Use a historical context of 3 months to predict the stock prices 5 days (a week) into the future
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems, Inc.
# Copyright (c) 2019. All Rights Reserved.

# The imports
import os
import time
import numpy
import plotly
# import traceback
import functools
import tensorflow
import pandas as pd
import plotly.graph_objs as go
from collections import namedtuple

# Enable Eager Execution for conversions between Tensors and Numpy arrays
tensorflow.enable_eager_execution()

# Plotly user account credentials for visualization
plotly.tools.set_credentials_file(username='bkeshava', api_key='RHqYrDdThygiJEPiEW5S')


# This class predicts the closing stock price of a company leveraging capabilities of RNNs in the...
# ...high-level Keras API within TensorFlow
# Inspired by Language Modelling
class RNNStockAnalysis(object):
    # The column key for the date attribute
    DATE_COLUMN_KEY = 'Date'

    # The column key for the closing stock price attribute
    CLOSING_STOCK_PRICE_COLUMN_KEY = 'Closing_Stock_Price'

    # The cost visualization metric
    # tensorflow.keras.metrics.sparse_categorical_crossentropy
    COST_METRIC = 'sparse_categorical_crossentropy'

    # Batch size
    # This seems to be the best mini_batch_size for injecting the appropriate amount of noise into the SGD process...
    # ...in order to prevent it from settling down at a saddle point. Furthermore, we can better leverage the...
    # ...parallel CUDA capabilities of the NVIDIA GPU if mini_batch_size > 32.
    BATCH_SIZE = 65

    # (1 - DROPOUT_RATE)
    # The keep probability for Hinton Dropout
    # Dropout Factor = 0.1
    # Two runs - one has a dropout factor of 0.9 and the other has a dropout factor of 0.8
    KEEP_PROBABILITY = 0.8

    # The pragmatic limits of the stock price in USD
    PRAGMATIC_STOCK_PRICE_LIMITS = namedtuple('Limits', ['lower_limit', 'upper_limit', 'precision'])

    # The length of the look-back context
    # A lookback context length of 65 days (3 months of look-back = (4 + 4 + 5) weeks * 5 days per week = 65 days)
    # Two runs - one has a look-back of 65 days and the other has a look-back of 26 days
    LOOK_BACK_CONTEXT_LENGTH = 26

    # The length of the look-ahead predictions = The length of the test data set
    # A reasonable look-ahead size given the quality of the dataset (uni-variate - historical stock prices) is 10 days
    # 2 weeks of look-ahead
    LOOK_AHEAD_SIZE = 10

    # The size of the projected vector space
    # A lower dimensional, dense, continuous vector space
    PROJECTED_VECTOR_SIZE = 2600

    # The checkpoint directory
    CHECKPOINT_DIRECTORY = './checkpoints'

    # The number of training epochs
    NUMBER_OF_TRAINING_EPOCHS = 2000

    # The checkpoint trigger factor
    CHECKPOINT_TRIGGER_FACTOR = 20

    # The number of RNN units
    NUMBER_OF_RNN_UNITS = 3900

    # Training data limit
    TRAINING_DATA_LIMIT = 6955

    # Plotly Scatter mode
    PLOTLY_SCATTER_MODE = 'lines+markers'

    # Parameters of the model vocabulary
    LOWER_LIMIT = 0.0
    UPPER_LIMIT = 99.0
    PRECISION = 0.01

    # Prediction randomness coefficient
    # This is called temperature in language modelling
    CHAOS_COEFFICIENT = 0.10

    # The initialization sequence
    def __init__(self):
        print('[INFO] RNNStockAnalysis Initialization: Bringing things up...')
        # The standard checkpoint naming convention checkpoint_{epoch_number}
        self.checkpoint_prefix = os.path.join(self.CHECKPOINT_DIRECTORY,
                                              'checkpoint_{epoch}')
        # The standard tensorboard logging convention
        self.tensorboard_logging_identifier = 'tensorboard-logs/{}'
        # The pragmatic stock price limits and precision are encapsulated in a namedtuple
        # This parameterizes the available vocabulary
        self.pragmatic_stock_information = self.PRAGMATIC_STOCK_PRICE_LIMITS(lower_limit=self.LOWER_LIMIT,
                                                                             upper_limit=self.UPPER_LIMIT,
                                                                             precision=self.PRECISION)
        precision_cutoff = len(str(self.PRECISION)) - str(self.PRECISION).index('.') - 1
        # The available vocabulary for this problem
        self.available_vocabulary = [float(str(x)[:str(x).index('.') + precision_cutoff + 1]) for x in numpy.arange(
            self.pragmatic_stock_information.lower_limit,
            self.pragmatic_stock_information.upper_limit,
            self.pragmatic_stock_information.precision)]
        # Create a mapping from vocabulary space to the set of all positive integers (Z_{++})
        self.vocabulary_to_integer_mapping = {element: integer for integer, element in enumerate(
            self.available_vocabulary)}
        # Create a mapping from the set of all positive integers (Z_{++}) to the vocabulary space
        self.integer_to_vocabulary_mapping = {integer: element for integer, element in enumerate(
            self.available_vocabulary
        )}
        # Load the data
        dataframe = pd.read_csv('datasets/csco.csv',
                                usecols=[0, 4])
        # Rename the columns for aesthetics
        dataframe.columns = [self.DATE_COLUMN_KEY,
                             self.CLOSING_STOCK_PRICE_COLUMN_KEY]
        # Extract the attributes
        self.dates = dataframe[self.DATE_COLUMN_KEY]
        self.stock_prices = dataframe[self.CLOSING_STOCK_PRICE_COLUMN_KEY].apply(
            lambda x: round(x, precision_cutoff))
        # Visualize the stock market trends for CISCO over time
        initial_visualization_trace = go.Scatter(x=self.dates,
                                                 y=self.stock_prices.values,
                                                 mode=self.PLOTLY_SCATTER_MODE)
        initial_visualization_layout = dict(title='CISCO (CSCO) Variations in Stock Price',
                                            xaxis=dict(title='Time'),
                                            yaxis=dict(title='Closing Stock Price'))
        initial_visualization_fig = dict(data=[initial_visualization_trace],
                                         layout=initial_visualization_layout)
        initial_fig_url = plotly.plotly.plot(initial_visualization_fig,
                                             filename='CISCO_Variations_In_Stock_Price')
        # Print the URL in case you're on an environment where a GUI is not available
        print('[INFO] RNNStockAnalysis Initialization: Data Visualization Figure is available at {}'.format(
            initial_fig_url
        ))
        # The data set for training - [0, 6955)
        self.stock_prices_training = self.stock_prices.values[:self.TRAINING_DATA_LIMIT]
        # Integer mapped training data
        self.training_data = numpy.array([self.vocabulary_to_integer_mapping[x] for x in self.stock_prices_training])
        # The data set for testing - [6955 6964]
        self.dates_testing = self.dates[self.TRAINING_DATA_LIMIT:self.TRAINING_DATA_LIMIT + self.LOOK_AHEAD_SIZE]
        self.stock_prices_testing = self.stock_prices.values[
                                    self.TRAINING_DATA_LIMIT:self.TRAINING_DATA_LIMIT + self.LOOK_AHEAD_SIZE]
        # Create individual data samples and convert the data into sequences of lookback context length
        # Sequences of length 66 will be created
        self.batched_data = tensorflow.data.Dataset.from_tensor_slices(self.training_data).batch(
            self.LOOK_BACK_CONTEXT_LENGTH + 1,
            drop_remainder=True)
        # Split the data into inputs and targets
        # <Input is of length 65> and <Target is right shifted along the time axis and is of length 65>
        self.split_dataset = self.batched_data.map(lambda x: (x[:-1], x[1:])).batch(self.BATCH_SIZE,
                                                                                    drop_remainder=True)
        # The model
        self.model = None
        # GPU Availability
        self.gpu_availability = tensorflow.test.is_gpu_available()
        print('[INFO] RNNStockAnalysis Initialization: GPU Availability - [{}]'.format(self.gpu_availability))

    # Build the model using RNN layers from Keras
    def build_model(self, initial_build=True, batch_size=None):
        try:
            batch_size = (lambda: self.BATCH_SIZE, lambda: batch_size)[initial_build is False]()
            # GPU - CuDNNGRU: The NVIDIA Compute Unified Device Architecture (CUDA) based Deep Neural Network library...
            # ... is a GPU accelerated library of primitives for Deep Neural Networks. CuDNNGRU is a fast GRU impl...
            # within the CuDNN framework.
            # CPU - Develop a modified RNN layer by using functools.partial
            custom_gru = (lambda: functools.partial(tensorflow.keras.layers.GRU,
                                                    recurrent_activation='sigmoid'),
                          lambda: tensorflow.keras.layers.CuDNNGRU)[self.gpu_availability]()
            # Construct the model sequentially
            model = tensorflow.keras.Sequential([
                # The Embedding Layer
                # Project the contextual vector onto a dense, continuous vector space
                tensorflow.keras.layers.Embedding(len(self.available_vocabulary),
                                                  self.PROJECTED_VECTOR_SIZE,
                                                  batch_input_shape=[batch_size,
                                                                     None]),
                # The Recurrent Neural Network - use GRU or LSTM units
                # GRUs are used here because structurally they're simpler and hence take smaller training times
                # Also, they don't have a forget gate in them, so they expose the entire memory during their operation
                custom_gru(self.NUMBER_OF_RNN_UNITS,
                           return_sequences=True,
                           # Xavier Uniform Initialization - RNN Cell/System initialization by drawing samples...
                           # ...uniformly from...
                           # ...[-\sqrt{\frac{6}{fan_{in} + fan_{out}}}, \sqrt{\frac{6}{fan_{in} + fan_{out}}}]
                           recurrent_initializer='glorot_uniform',
                           stateful=True),
                # The Hinton dropout layer
                tensorflow.keras.layers.Dropout(rate=1 - self.KEEP_PROBABILITY),
                # The fully connected neural network
                # A classification-type output onto the vocabulary
                tensorflow.keras.layers.Dense(len(self.available_vocabulary))
            ])
            # Print a summary of the designed model
            print('[INFO] RNNStockAnalysis build: A summary of the designed model is given below...')
            model.summary()
            self.model = (lambda: self.model, lambda: model)[initial_build]()
            return True, model
        except Exception as e:
            print('[ERROR] RNNStockAnalysis build: Exception caught while building the model - {}'.format(e))
            # Detailed stack trace
            # traceback.print_tb(e.__traceback__)
            return False, None

    @staticmethod
    # The cost function for the defined model
    def cost_function(y_true_values, y_predicted_values):
        # Sparse Categorical Cross-Entropy is chosen because we have mutually exclusive classes in a classic...
        # ...classification problem
        return tensorflow.keras.losses.sparse_categorical_crossentropy(y_true=y_true_values,
                                                                       y_pred=y_predicted_values,
                                                                       from_logits=True)

    # Set the model up with the optimizer and the cost function
    def compile(self):
        try:
            # The Adam Optimizer, Sparse Categorical Cross-Entropy cost function, and cost metrics for visualization
            self.model.compile(optimizer=tensorflow.train.AdamOptimizer(),
                               loss=self.cost_function,
                               metrics=[tensorflow.keras.metrics.sparse_categorical_crossentropy])
            return True
        except Exception as e:
            print('[ERROR] RNNStockAnalysis compile: Exception caught while compiling the model - {}'.format(e))
            return False

    # Train the model and Visualize the model's progression during training
    def train(self):
        try:
            # TODO: Add a logging hook as a callback and include it in the 'callbacks' collection within the fit routine
            # Checkpoint feature callback
            checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_prefix,
                                                                             monitor='loss',
                                                                             save_weights_only=True,
                                                                             save_best_only=True,
                                                                             verbose=1,
                                                                             mode='min',
                                                                             period=self.CHECKPOINT_TRIGGER_FACTOR)
            # Tensorboard callback
            tensorboard = tensorflow.keras.callbacks.TensorBoard(log_dir=self.tensorboard_logging_identifier.format(
                time.time()))
            # Visualize the progression of the cost function during training
            training_history = self.model.fit(self.split_dataset.repeat(),
                                              epochs=self.NUMBER_OF_TRAINING_EPOCHS,
                                              steps_per_epoch=self.BATCH_SIZE,
                                              callbacks=[checkpoint_callback, tensorboard])
            training_trace = go.Scatter(x=training_history.epoch,
                                        y=training_history.history[self.COST_METRIC],
                                        mode=self.PLOTLY_SCATTER_MODE)
            training_data_trace = [training_trace]
            training_layout = dict(
                title='Cost Progression Analysis during the Training phase',
                xaxis=dict(title='Epochs'),
                yaxis=dict(title='Sparse Categorical Cross-Entropy'))
            training_figure = dict(data=training_data_trace,
                                   layout=training_layout)
            cost_progression_fig_url = plotly.plotly.plot(training_figure,
                                                          filename='Cost_Progression_Visualization_Training')
            # Print the URL in case you're on an environment where a GUI is not available
            print('[INFO] RNNStockAnalysis train: Cost Progression Visualization Figure available at {}'.format(
                cost_progression_fig_url
            ))
            return True, training_history
        except Exception as e:
            print('[ERROR] RNNStockAnalysis train: Exception caught while training the model - {}'.format(e))
            # Detailed stack trace
            # traceback.print_tb(e.__traceback__)
            return False, None

    # Predict the next ${LOOK_AHEAD_SIZE} stock prices
    def predict(self):
        # The output to be returned
        predicted_prices = []
        # GPU Availability - Check again
        self.gpu_availability = tensorflow.test.is_gpu_available()
        print('[INFO] RNNStockAnalysis Initialization: GPU Availability - [{}]'.format(self.gpu_availability))
        # Modify the model for a batch size of 1
        status, modified_model = self.build_model(initial_build=False,
                                                  batch_size=1)
        if status is False:
            print('[ERROR] RNNStockAnalysis predict: The operation failed due to previous errors!')
            return
        try:
            modified_model.load_weights(tensorflow.train.latest_checkpoint(self.CHECKPOINT_DIRECTORY))
            modified_model.build(tensorflow.TensorShape([1, None]))
            # The tail-end look-back context for the initial look-ahead prediction
            # The cumulative context collection is initialized to the last <self.LOOK_BACK_CONTEXT_LENGTH> characters...
            # ... of the test dataset
            cumulative_context = self.training_data[len(self.training_data) - self.LOOK_BACK_CONTEXT_LENGTH:]
            # Reset the states of the RNN
            modified_model.reset_states()
            # Iterate through multiple predictions in a chain
            for i in range(self.LOOK_AHEAD_SIZE):
                trigger = tensorflow.expand_dims(cumulative_context, 0)
                prediction = modified_model(trigger)
                # Remove the useless dimension
                prediction = tensorflow.squeeze(prediction, 0) / self.CHAOS_COEFFICIENT
                # Use a multinomial distribution to determine the predicted value
                predicted_price = tensorflow.multinomial(prediction, num_samples=1)[-1, 0].numpy()
                # Append the predicted value to the output collection
                predicted_prices.append(self.integer_to_vocabulary_mapping[predicted_price])
                # Context modification logic
                # Add the predicted price to the context which would be used for the next iteration
                cumulative_context = numpy.append(cumulative_context, [predicted_price], axis=0)
                # Move the context window to include the latest prediction and discount the oldest contextual element
                cumulative_context = cumulative_context[1:]
        except Exception as e:
            print('[ERROR] RNNStockAnalysis predict: Exception caught during prediction - {}'.format(e))
            # Detailed stack trace
            # traceback.print_tb(e.__traceback__)
        return predicted_prices

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] RNNStockAnalysis Termination: Tearing things down...')
        # Nothing to do


# Visualize the predictions against the corresponding true values
def visualize_predictions(obj, _true_values, _predicted_values):
    x_axis = obj.dates_testing
    real_final_analysis_trace = go.Scatter(x=x_axis,
                                           y=_true_values,
                                           mode=obj.PLOTLY_SCATTER_MODE,
                                           name='True Stock Prices')
    generated_final_analysis_trace = go.Scatter(x=x_axis,
                                                y=_predicted_values,
                                                mode=obj.PLOTLY_SCATTER_MODE,
                                                name='Stock Prices predicted by the RNN model')
    final_analysis_layout = dict(title='Analysis of the predicted stock prices versus the true stock prices from the '
                                       'test data set',
                                 xaxis=dict(title='Time'),
                                 yaxis=dict(title='Closing Stock Price'))
    final_analysis_figure = dict(data=[real_final_analysis_trace, generated_final_analysis_trace],
                                 layout=final_analysis_layout)
    final_analysis_url = plotly.plotly.plot(final_analysis_figure,
                                            filename='Prediction_Analysis_Test_Dataset')
    # An additional array creation logic is inserted in the print statement to format both collections identically...
    # ...for aesthetics.
    print('[INFO] RNNStockAnalysis visualize_predictions: Look-back Context Length - [{}]'.format(
        obj.LOOK_BACK_CONTEXT_LENGTH))
    print('[INFO] RNNStockAnalysis visualize_predictions: Look-ahead Size - [{}]'.format(
        obj.LOOK_AHEAD_SIZE))
    print('[INFO] RNNStockAnalysis visualize_predictions: The true stock prices are - \n{}'.format(
        [k for k in _true_values]))
    print('[INFO] RNNStockAnalysis visualize_predictions: The predicted prices are  - \n{}'.format(
        [k for k in _predicted_values]))
    # Print the URL in case you're on an environment where a GUI is not available
    print('[INFO] RNNStockAnalysis visualize_predictions: The final prediction analysis visualization figure is '
          'available at {}'.format(final_analysis_url))
    return None


# Run Trigger
if __name__ == '__main__':
    print('[INFO] RNNStockAnalysis Trigger: Starting system assessment!')
    rnnStockAnalysis = RNNStockAnalysis()
    # TODO: Use an ETL-type pipeline for this sequence of operations on the model
    if rnnStockAnalysis.build_model()[0] and rnnStockAnalysis.compile() and rnnStockAnalysis.train()[0]:
        print('[INFO] RNNStockAnalysis Trigger: The model has been built, compiled, and trained! '
              'Evaluating the model...')
        visualize_predictions(rnnStockAnalysis,
                              rnnStockAnalysis.stock_prices_testing,
                              rnnStockAnalysis.predict())
    else:
        print('[INFO] RNNStockAnalysis Trigger: The operation failed due to previous errors!')
