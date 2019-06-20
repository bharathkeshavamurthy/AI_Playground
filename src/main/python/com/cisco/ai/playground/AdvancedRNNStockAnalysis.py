# This entity encapsulates an intelligent model to predict the stock prices of a company using RNNs in TensorFlow
# This model can be extended to predict link states in networks [ Link_Up Link_Up Link_Up Link_Down Link_Down ]
# Use a historical context of 3 months to predict the stock prices 5 days (a week) into the future
# This is the advanced version of the RNN-based Stock Analysis script.
# In this advanced version, we incorporate custom training into the system to facilitate increased controllability.
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems, Inc.
# Copyright (c) 2019. All Rights Reserved.

# The imports
import os
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
# Custom, Advanced Training
class AdvancedRNNStockAnalysis(object):
    # The column key for the date attribute
    DATE_COLUMN_KEY = 'Date'

    # The column key for the closing stock price attribute
    CLOSING_STOCK_PRICE_COLUMN_KEY = 'Closing_Stock_Price'

    # The cost visualization metric
    # tensorflow.keras.metrics.sparse_categorical_crossentropy
    COST_METRIC = 'sparse_categorical_crossentropy'

    # Batch size
    BATCH_SIZE = 65

    # (1 - DROPOUT_RATE)
    # The keep probability for Hinton Dropout
    KEEP_PROBABILITY = 0.7

    # The pragmatic limits of the stock price in USD
    PRAGMATIC_STOCK_PRICE_LIMITS = namedtuple('Limits', ['lower_limit', 'upper_limit', 'precision'])

    # The length of the look-back context
    # A look back of 3 months ((5 + 4 + 4) weeks * 5 working days = 65)
    LOOK_BACK_CONTEXT_LENGTH = 65

    # The length of the look-ahead predictions = The length of the test data set
    LOOK_AHEAD_SIZE = 464

    # The size of the projected vector space
    # A lower dimensional, dense, continuous vector space
    PROJECTED_VECTOR_SIZE = 1300

    # The checkpoint directory
    CHECKPOINT_DIRECTORY = './checkpoints-advanced'

    # The number of training epochs
    NUMBER_OF_TRAINING_EPOCHS = 5000

    # The number of RNN units
    NUMBER_OF_RNN_UNITS = 2600

    # Training data limit
    TRAINING_DATA_LIMIT = 6500

    # Plotly Scatter mode
    PLOTLY_SCATTER_MODE = 'lines+markers'

    # Parameters of the model vocabulary
    LOWER_LIMIT = 0.0
    UPPER_LIMIT = 99.0
    PRECISION = 0.01

    # The randomness coefficient
    # This is termed temperature in language prediction models
    CHAOS_COEFFICIENT = 0.01

    # The initialization sequence
    def __init__(self):
        print('[INFO] AdvancedRNNStockAnalysis Initialization: Bringing things up...')
        # The standard checkpoint naming convention checkpoint_{epoch_number}
        self.checkpoint_prefix = os.path.join(self.CHECKPOINT_DIRECTORY,
                                              'checkpoint_{epoch}')
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
            lambda x: float(str(x)[:str(x).index('.') + precision_cutoff + 1]))
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
        print('[INFO] AdvancedRNNStockAnalysis Initialization: Data Visualization Figure is available at {}'.format(
            initial_fig_url
        ))
        # The data set for training - [0, 6500)
        self.stock_prices_training = self.stock_prices.values[:self.TRAINING_DATA_LIMIT]
        # Integer mapped training data
        self.training_data = numpy.array([self.vocabulary_to_integer_mapping[x] for x in self.stock_prices_training])
        # The data set for testing - [6500 6964)
        self.dates_testing = self.dates[self.TRAINING_DATA_LIMIT:]
        self.stock_prices_testing = self.stock_prices.values[self.TRAINING_DATA_LIMIT:]
        # Create individual data samples and convert the data into sequences of lookback context length
        # Sequences of length 65 will be created
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
        print('[INFO] AdvancedRNNStockAnalysis Initialization: GPU Availability - [{}]'.format(self.gpu_availability))

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
                           # ...uniformly from (-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out)))
                           recurrent_initializer='glorot_uniform',
                           stateful=True),
                # The Hinton dropout layer
                # TODO: Add Regularization after analyzing the fit of the designed model
                # tensorflow.keras.layers.Dropout(rate=1 - self.KEEP_PROBABILITY),
                # The fully connected neural network
                # A classification-type output onto the vocabulary
                tensorflow.keras.layers.Dense(len(self.available_vocabulary))
            ])
            # Print a summary of the designed model
            print('[INFO] AdvancedRNNStockAnalysis build: A summary of the designed model is given below...')
            model.summary()
            self.model = (lambda: self.model, lambda: model)[initial_build]()
            return True, model
        except Exception as e:
            print('[ERROR] AdvancedRNNStockAnalysis build: Exception caught while building the model - {}'.format(e))
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

    # Custom, Advanced Training of the model
    def advanced_training(self):
        try:
            # I use an AdamOptimizer in the place of the simpler tensorflow.train.GradientDescentOptimizer()...
            # ...because the AdamOptimizer uses the moving average of parameters and this facilitates...
            # ...faster convergence by settling on a larger effective step-size.
            optimizer = tensorflow.train.AdamOptimizer()
            for epoch in range(0, self.NUMBER_OF_TRAINING_EPOCHS):
                self.model.reset_states()
                # Iterate through the training examples
                for (batch_number, (training_context, target)) in enumerate(self.split_dataset):
                    with tensorflow.GradientTape() as gradient_tape:
                        prediction = self.model(training_context)
                        # Evaluate the loss for this training example
                        loss = tensorflow.losses.sparse_softmax_cross_entropy(target,
                                                                              prediction)
                    # Evaluate the gradients and apply the optimization step (descent) on the trainable variables
                    gradients = gradient_tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    print('[TRACE] AdvancedRNNStockAnalysis advanced_training: Epoch #{}, Batch #{}, Loss {}'.format(
                        epoch + 1, batch_number + 1, loss
                    ))
                print('[DEBUG] AdvancedRNNStockAnalysis advanced_training: Epoch #{}, Loss {}'.format(
                    epoch + 1, loss
                ))
                # Manual checkpoint save
                self.model.save_weights(self.checkpoint_prefix.format(epoch=epoch + 1))
            return True
        except Exception as e:
            print('[ERROR] AdvancedRNNStockAnalysis train: Exception caught while training the model - {}'.format(e))
            # Detailed stack trace
            # traceback.print_tb(e.__traceback__)
            return False

    # Predict the next ${LOOK_AHEAD_SIZE} stock prices
    def predict(self):
        # The output to be returned
        predicted_prices = []
        # GPU Availability - Check again
        self.gpu_availability = tensorflow.test.is_gpu_available()
        # Modify the model for a batch size of 1
        status, modified_model = self.build_model(initial_build=False,
                                                  batch_size=1)
        if status is False:
            print('[ERROR] AdvancedRNNStockAnalysis predict: The operation failed due to previous errors!')
            return
        try:
            modified_model.load_weights(tensorflow.train.latest_checkpoint(self.CHECKPOINT_DIRECTORY))
            modified_model.build_model(tensorflow.TensorShape([1, None]))
            # The tail-end look-back context for the initial look-ahead prediction
            # The cumulative context collection is initialized to the last <self.LOOK_BACK_CONTEXT_LENGTH> characters...
            # ... of the test dataset
            cumulative_context = self.training_data[len(self.training_data) - self.LOOK_BACK_CONTEXT_LENGTH:]
            trigger = tensorflow.expand_dims(cumulative_context, 0)
            # Reset the states of the RNN
            modified_model.reset_states()
            # Iterate through multiple predictions in a chain
            for i in range(self.LOOK_AHEAD_SIZE):
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
                trigger = tensorflow.expand_dims([predicted_price], 0)
        except Exception as e:
            print('[ERROR] AdvancedRNNStockAnalysis predict: Exception caught during prediction - {}'.format(e))
            # Detailed stack trace
            # traceback.print_tb(e.__traceback__)
        return predicted_prices

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] AdvancedRNNStockAnalysis Termination: Tearing things down...')
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
    print('[INFO] AdvancedRNNStockAnalysis visualize_predictions: The final prediction analysis visualization figure '
          'is available at {}'.format(final_analysis_url))
    return None


# Run Trigger
if __name__ == '__main__':
    print('[INFO] AdvancedRNNStockAnalysis Trigger: Starting system assessment!')
    rnnStockAnalysis = AdvancedRNNStockAnalysis()
    # Build the model
    if rnnStockAnalysis.build_model()[0]:
        # Compile and Train the model
        if rnnStockAnalysis.advanced_training():
            print('[INFO] AdvancedRNNStockAnalysis Trigger: The model has been built, compiled, and trained! '
                  'Evaluating the model...')
            # Engage the trained model in multi-step time-series predictions
            visualize_predictions(rnnStockAnalysis,
                                  rnnStockAnalysis.stock_prices_testing,
                                  rnnStockAnalysis.predict())
        else:
            print('[ERROR] AdvancedRNNStockAnalysis Trigger: The operation failed while compiling and '
                  'training the model!')
    else:
        print('[ERROR] AdvancedRNNStockAnalysis Trigger: The operation failed while building the model!')
