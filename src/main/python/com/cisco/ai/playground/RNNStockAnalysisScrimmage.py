# |Bleeding-Edge Productions|
# This entity encapsulates an intelligent model to predict the stock prices of a company using RNNs in TensorFlow.
# This model can be extended to predict link states in networks [ Link_Up Link_Up Link_Up Link_Down Link_Down ] by...
# ...determining the value drift of design and operation critical settings across time.
# Use a historical context of 3 months to predict the stock prices 10 days (two weeks) into the future.
# ------------------------------------------Scrimmage variant-----------------------------------------------------------
# Author: Bharath Keshavamurthy {bkeshava}
# Organization: DC NX-OS, CISCO Systems Inc.
# Copyright (c) 2019. All Rights Reserved.

# TODO: A logging framework instead of all these formatted print statements

"""
Change Log - 01-August-2019:
@author: Bharath Keshavamurthy <bkeshava at cisco dot com>

1. Adding two additional RNN layers for better correlation tracking -> a. RNN_LAYER_1 => NUMBER_OF_RNN_UNITS_1 = 5100
                                                                       b. RNN_LAYER_2 => NUMBER_OF_RNN_UNITS_2 = 3400
                                                                       c. RNN_LAYER_3 => NUMBER_OF_RNN_UNITS_3 = 1700

2. Changing the BATCH_SIZE to 105 from 65 to include all the <input, target> sequence pairs in one batch

3. Changing the embedding size to 3400 from 2600 for better lower-dimensional representation for the vocab of size 9900

4. Changing the look-back context logic in the predict() routine - a converging and diverging context window for a
heuristic that works for this dataset

5. Added two new parameters - CONTEXT_TRIGGER and VARIATION_TRIGGER inline with the modifications to the context window
logic
"""

# The imports
# import pdb
import numpy
import plotly
import traceback
import functools
import tensorflow
import pandas as pd
import plotly.graph_objs as go
from collections import namedtuple

# Enable Eager Execution for conversions between tensors and numpy arrays
tensorflow.enable_eager_execution()

# Plotly user account credentials for visualization
plotly.tools.set_credentials_file(username='bkeshava_cisco',
                                  api_key='qTXrG3oefYkdFtXVjYcv')


# This class predicts the closing stock price of a company leveraging capabilities of RNNs in the...
# ...high-level Keras API within TensorFlow
# Inspired by Language Modelling
# Scrimmage variant - Load weights from the checkpoints created by the OG's run and make predictions
class RNNStockAnalysisScrimmage(object):
    # The column key for the date attribute
    DATE_COLUMN_KEY = 'Date'

    # The column key for the closing stock price attribute
    CLOSING_STOCK_PRICE_COLUMN_KEY = 'Closing_Stock_Price'

    # Batch size
    # This seems to be the best mini_batch_size for injecting the appropriate amount of noise into the SGD process...
    # ...in order to prevent it from settling down at a saddle point.
    # Furthermore, we can better leverage the CUDA capabilities of the NVIDIA GPU if mini_batch_size > 32.
    # 105 => Everything constitutes one batch leading to one step per epoch.
    BATCH_SIZE = 105

    # (1 - DROPOUT_RATE)
    # The keep probability for Hinton Dropout
    # Dropout Factor: 0.2
    # KEEP_PROBABILITY = 0.8

    # The pragmatic limits of the stock price in USD
    PRAGMATIC_STOCK_PRICE_LIMITS = namedtuple('Limits',
                                              ['lower_limit',
                                               'upper_limit',
                                               'precision'])

    # The length of the look-back context
    # A lookback context length of 65 days (3 months of look-back = (4 + 4 + 5) weeks * 5 days per week = 65 days)
    LOOK_BACK_CONTEXT_LENGTH = 65

    # The length of the look-ahead predictions = The length of the test data set
    # A reasonable look-ahead size given the quality of the dataset (uni-variate - historical stock prices) is 10 days
    # 2 weeks of look-ahead
    LOOK_AHEAD_SIZE = 10

    # The size of the projected vector space
    # A lower dimensional, dense, continuous vector space
    PROJECTED_VECTOR_SIZE = 3400

    # The checkpoint directory
    CHECKPOINT_DIRECTORY = './checkpoints'

    # The number of units in the first RNN layer
    NUMBER_OF_RNN_UNITS_1 = 5100

    # The number of units in the second RNN layer
    NUMBER_OF_RNN_UNITS_2 = 3400

    # The number of units in the third RNN layer
    NUMBER_OF_RNN_UNITS_3 = 1700

    # Training data limit
    TRAINING_DATA_LIMIT = 6954

    # The context trigger for look-ahead prediction
    CONTEXT_TRIGGER = 6945

    # The variation trigger added exclusively for the converging-diverging context window heuristic
    VARIATION_TRIGGER = 1

    # Plotly Scatter mode
    PLOTLY_SCATTER_MODE = 'lines+markers'

    # Parameters of the model vocabulary
    LOWER_LIMIT = 0.0
    UPPER_LIMIT = 99.0
    PRECISION = 0.01

    # Prediction randomness coefficient
    # This is called 'Temperature' in language modelling
    # This parameter is no longer needed because we're using the tensorflow.nn.softmax function for analyzing the...
    # ...logits provided by the model during the predict operation.
    # CHAOS_COEFFICIENT = 1e-9

    # The look-back context length factor for validation and/or testing - 60%
    # VALIDATION_LOOK_BACK_CONTEXT_LENGTH_FACTOR = 0.6

    # The initialization sequence
    def __init__(self):
        print('[INFO] RNNStockAnalysisScrimmage Initialization: Bringing things up...')
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
                                usecols=[0,
                                         4])
        # Rename the columns for aesthetics
        dataframe.columns = [self.DATE_COLUMN_KEY,
                             self.CLOSING_STOCK_PRICE_COLUMN_KEY]
        # Extract the attributes
        self.dates = dataframe[self.DATE_COLUMN_KEY]
        self.stock_prices = dataframe[self.CLOSING_STOCK_PRICE_COLUMN_KEY].apply(
            lambda x: round(x, precision_cutoff))
        # Visualize the stock market trends for the given company (Stock_Exchange ticker) over time
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
        print('[INFO] RNNStockAnalysisScrimmage Initialization: Data Visualization Figure is available at {}'.format(
            initial_fig_url
        ))
        # The data set for training - [0, 6954)
        self.stock_prices_training = self.stock_prices.values[:self.TRAINING_DATA_LIMIT]
        # Integer mapped training data
        self.training_data = numpy.array([self.vocabulary_to_integer_mapping[x] for x in self.stock_prices_training])
        # The data set for testing - [6954 6964]
        self.dates_testing = self.dates[self.TRAINING_DATA_LIMIT:self.TRAINING_DATA_LIMIT + self.LOOK_AHEAD_SIZE]
        self.stock_prices_testing = self.stock_prices.values[
                                    self.TRAINING_DATA_LIMIT:self.TRAINING_DATA_LIMIT + self.LOOK_AHEAD_SIZE]
        # The model
        self.model = None
        # GPU Availability
        self.gpu_availability = tensorflow.test.is_gpu_available()
        print('[INFO] RNNStockAnalysisScrimmage Initialization: GPU Availability - [{}]'.format(self.gpu_availability))

    # Build the model using RNN layers from Keras
    def build_model(self, initial_build=True, batch_size=None):
        try:
            batch_size = (lambda: self.BATCH_SIZE,
                          lambda: batch_size)[initial_build is False]()

            # GPU - CuDNNGRU: The NVIDIA Compute Unified Device Architecture (CUDA) based Deep Neural Network library...
            # ... is a GPU accelerated library of primitives for Deep Neural Networks.
            # CuDNNGRU is a fast GRU impl within the CuDNN framework.

            # CPU - Develop a modified RNN layer by using functools.partial

            custom_gru = (lambda: functools.partial(tensorflow.keras.layers.GRU,
                                                    recurrent_activation='sigmoid'),
                          lambda: tensorflow.keras.layers.CuDNNGRU)[self.gpu_availability]()

            # Construct the model sequentially
            model = tensorflow.keras.Sequential([
                # The Embedding Layer
                # Project the contextual vector onto a dense, lower-dimensional continuous vector space
                tensorflow.keras.layers.Embedding(len(self.available_vocabulary),
                                                  self.PROJECTED_VECTOR_SIZE,
                                                  batch_input_shape=[batch_size,
                                                                     None]),
                # The Recurrent Neural Network layers - use GRU or LSTM units
                # GRUs are used here because structurally they're simpler and hence take smaller training times.
                # Also, they don't have a forget gate in them, so they expose the entire memory during their operation.
                # RNN Layer 1
                custom_gru(self.NUMBER_OF_RNN_UNITS_1,
                           return_sequences=True,
                           # Xavier Uniform Initialization - RNN Cell/System initialization by drawing samples...
                           # ...uniformly from...
                           # ...[-\sqrt{\frac{6}{fan_{in} + fan_{out}}}, \sqrt{\frac{6}{fan_{in} + fan_{out}}}]
                           recurrent_initializer='glorot_uniform',
                           stateful=True),
                # RNN Layer 2
                custom_gru(self.NUMBER_OF_RNN_UNITS_2,
                           return_sequences=True,
                           # Xavier Uniform Initialization - RNN Cell/System initialization by drawing samples...
                           # ...uniformly from...
                           # ...[-\sqrt{\frac{6}{fan_{in} + fan_{out}}}, \sqrt{\frac{6}{fan_{in} + fan_{out}}}]
                           recurrent_initializer='glorot_uniform',
                           stateful=True),
                # RNN Layer 3
                custom_gru(self.NUMBER_OF_RNN_UNITS_3,
                           return_sequences=True,
                           # Xavier Uniform Initialization - RNN Cell/System initialization by drawing samples...
                           # ...uniformly from...
                           # ...[-\sqrt{\frac{6}{fan_{in} + fan_{out}}}, \sqrt{\frac{6}{fan_{in} + fan_{out}}}]
                           recurrent_initializer='glorot_uniform',
                           stateful=True),

                # RNN Layer 4
                # custom_gru(self.NUMBER_OF_RNN_UNITS_4,
                #            return_sequences=True,
                #            # Xavier Uniform Initialization - RNN Cell/System initialization by drawing samples...
                #            # ...uniformly from...
                #            # ...[-\sqrt{\frac{6}{fan_{in} + fan_{out}}}, \sqrt{\frac{6}{fan_{in} + fan_{out}}}]
                #            recurrent_initializer='glorot_uniform',
                #            stateful=True),

                # The Hinton dropout layer
                # tensorflow.keras.layers.Dropout(rate=1 - self.KEEP_PROBABILITY),

                # The fully connected neural network
                # A classification-type output onto the vocabulary
                tensorflow.keras.layers.Dense(len(self.available_vocabulary))
            ])
            # Print a summary of the designed model
            print('[INFO] RNNStockAnalysisScrimmage build: A summary of the designed model is given below...')
            model.summary()
            self.model = (lambda: self.model,
                          lambda: model)[initial_build]()
            return True, model
        except Exception as e:
            print('[ERROR] RNNStockAnalysisScrimmage build: Exception caught while building the model - {}'.format(e))
            # Detailed stack trace
            traceback.print_tb(e.__traceback__)
            return False, None

    # Predict the next ${LOOK_AHEAD_SIZE} stock prices
    def predict(self):
        # The output to be returned
        predicted_prices = []
        # GPU Availability - Check again in case something took up the discrete graphics capabilities of the machine
        self.gpu_availability = tensorflow.test.is_gpu_available()
        print('[INFO] RNNStockAnalysis Initialization: GPU Availability - [{}]'.format(self.gpu_availability))
        # Build a new model with a batch-size of 1 -> Load the weights from the trained model -> Reshape the input layer
        status, modified_model = self.build_model(initial_build=False,
                                                  batch_size=1)
        if status is False:
            print('[ERROR] RNNStockAnalysis predict: The operation failed due to previous errors!')
            return
        try:
            # This line allows for the dynamic update of the weights to be loaded in this Scrimmage model
            modified_model.load_weights('./checkpoints/checkpoint_6000')
            modified_model.build(tensorflow.TensorShape([1,
                                                         None]))

            # --------------------------------- Context Window Logic ---------------------------------------------------
            # The tail-end look-back context for the initial look-ahead prediction
            # The cumulative context collection is initialized to the last <self.LOOK_BACK_CONTEXT_LENGTH> characters...
            # ...of the training dataset
            # cumulative_context = self.training_data[len(self.training_data) - self.LOOK_BACK_CONTEXT_LENGTH:]
            # Reset the states of the RNN
            # modified_model.reset_states()
            # Iterate through multiple predictions in a chain
            # for i in range(self.LOOK_AHEAD_SIZE):
            #     trigger = tensorflow.expand_dims(cumulative_context,
            #                                      0)
            #     prediction = modified_model(trigger)
            #     # Remove the useless dimension
            #     prediction = tensorflow.squeeze(prediction, 0) / self.CHAOS_COEFFICIENT
            #     # Use a multinomial distribution to determine the predicted value
            #     predicted_price = tensorflow.multinomial(prediction,
            #                                              num_samples=1)[-1, 0].numpy()
            #     # Append the predicted value to the output collection
            #     predicted_prices.append(self.integer_to_vocabulary_mapping[predicted_price])
            #     # Context modification logic
            #     # Add the predicted price to the context which would be used for the next iteration
            #     cumulative_context = numpy.append(cumulative_context,
            #                                       [predicted_price],
            #                                       axis=0)
            #     # Move the context window to include the latest prediction and discount the oldest contextual element
            #     cumulative_context = cumulative_context[1:]

            # ------------------------------------- Sequential Feedback Logic ------------------------------------------
            # The training context is initialized to the last <self.LOOK_BACK_CONTEXT_LENGTH> characters of the training
            #  dataset
            # context = self.training_data[len(self.training_data) - int(
            #     self.VALIDATION_LOOK_BACK_CONTEXT_LENGTH_FACTOR * self.LOOK_AHEAD_SIZE):]
            #
            # # print('[INFO] RNNStockAnalysis predict: The initial look-back context in the predict() routine is: '
            # #       '\n[{}]'.format(context))
            #
            # # Reset the states of the RNN
            # modified_model.reset_states()
            # # Iterate through multiple predictions in a chain
            # for i in range(self.LOOK_AHEAD_SIZE):
            #     context = tensorflow.expand_dims(context,
            #                                      0)
            #     prediction = modified_model(context)
            #     # Remove the useless dimension
            #     prediction = tensorflow.squeeze(prediction,
            #                                     0)
            #     # Use the tensorflow provided softmax function to convert the logits into probabilities and extract...
            #     # ...the highest probability class from the multinomial output vector
            #     predicted_price = numpy.argmax(
            #         tensorflow.nn.softmax(
            #             prediction[-1]
            #         )
            #     )
            #     # Append the predicted_price to the output collection
            #     predicted_prices.append(self.integer_to_vocabulary_mapping[predicted_price])
            #     # Context modification Logic - Caching the most recent transaction and right-shifting the window...
            #     context = numpy.append(tensorflow.squeeze(context,
            #                                               0),
            #                            [predicted_price],
            #                            axis=0)
            #     context = context[1:]

            # -----------------------------------Converging-Diverging Context Window Logic------------------------------
            # The trigger carried forward from the training dataset
            context = self.training_data[self.CONTEXT_TRIGGER:]

            # print('[INFO] RNNStockAnalysis predict: The initial look-back context in the predict() routine is: '
            #       '\n[{}]'.format(context))

            # Reset the states of the RNN
            modified_model.reset_states()
            # Iterate through multiple predictions in a chain
            for i in range(self.LOOK_AHEAD_SIZE):
                # The clip-off point
                k = (5 - i) if i <= self.VARIATION_TRIGGER else 0
                context = tensorflow.expand_dims(context,
                                                 0)

                # pdb.set_trace()

                # Make a stateful prediction
                prediction = modified_model(context)
                # Remove the useless dimension
                prediction = tensorflow.squeeze(prediction,
                                                0)
                # Use the tensorflow provided softmax function to convert the logits into probabilities and...
                # ...extract the highest probability class from the multinomial output vector
                predicted_price = numpy.argmax(
                    tensorflow.nn.softmax(
                        prediction[-1]
                    )
                )
                # Append the predicted_price to the output collection
                predicted_prices.append(self.integer_to_vocabulary_mapping[predicted_price])
                # Context modification Logic - Append the new value to the context
                context = numpy.append(numpy.squeeze(context,
                                                     0),
                                       [predicted_price],
                                       axis=0)
                # Move the context window according to the converging/diverging heuristic
                context = context[k:]
        except Exception as e:
            print('[ERROR] RNNStockAnalysis predict: Exception caught during prediction - {}'.format(e))
            # Detailed stack trace
            traceback.print_tb(e.__traceback__)
        return predicted_prices

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] RNNStockAnalysisScrimmage Termination: Tearing things down...')
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
    final_analysis_figure = dict(data=[real_final_analysis_trace,
                                       generated_final_analysis_trace],
                                 layout=final_analysis_layout)
    final_analysis_url = plotly.plotly.plot(final_analysis_figure,
                                            filename='Prediction_Analysis_Test_Dataset')
    # An additional array creation logic is inserted in the print statement to format both collections identically...
    # ...for aesthetics.
    print('[INFO] RNNStockAnalysisScrimmage visualize_predictions: Look-back Context Length - [{}]'.format(
        obj.LOOK_BACK_CONTEXT_LENGTH))
    print('[INFO] RNNStockAnalysisScrimmage visualize_predictions: Look-ahead Size - [{}]'.format(
        obj.LOOK_AHEAD_SIZE))
    print('[INFO] RNNStockAnalysisScrimmage visualize_predictions: The true stock prices are - \n{}'.format(
        [k for k in _true_values]))
    print('[INFO] RNNStockAnalysisScrimmage visualize_predictions: The predicted prices are  - \n{}'.format(
        [k for k in _predicted_values]))
    # Print the URL in case you're on an environment where a GUI is not available
    print('[INFO] RNNStockAnalysisScrimmage visualize_predictions: The final prediction analysis '
          'visualization figure is available at {}'.format(final_analysis_url))
    return None


# Run Trigger
if __name__ == '__main__':
    print('[INFO] RNNStockAnalysisScrimmage Trigger: Starting system assessment!')
    rnnStockAnalysisScrimmage = RNNStockAnalysisScrimmage()
    visualize_predictions(rnnStockAnalysisScrimmage,
                          rnnStockAnalysisScrimmage.stock_prices_testing,
                          rnnStockAnalysisScrimmage.predict())
