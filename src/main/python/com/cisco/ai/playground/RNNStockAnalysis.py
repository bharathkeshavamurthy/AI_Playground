# |Bleeding-Edge Productions|
# This entity encapsulates an intelligent model to predict the stock prices of a company using RNNs in TensorFlow.
# This model can be extended to predict link states in networks [ Link_Up Link_Up Link_Up Link_Down Link_Down ].
# Use a historical context of 3 months to predict the stock prices 10 days (two weeks) into the future.
# Author: Bharath Keshavamurthy {bkeshava}
# Organization: DC NX-OS, CISCO Systems Inc.
# Copyright (c) 2019. All Rights Reserved.

# TODO: A logging framework instead of all these formatted print statements

# The good thing I'm seeing in this variant is the similarity between the cost progression and ...
# ...the epoch_sparse_categorical_cross_entropy metric's progression.

# Another good things I'm seeing in the Quadro P4000 run is the consistency across different trials.

"""
Change Log - 29-July-2019:
@author: Bharath Keshavamurthy <bkeshava at cisco dot com>

1. NUMBER_OF_TRAINING_EPOCHS = 10000 [Extended training for convergence analysis - is it the global optimum?]
{The number of epochs is huge because the steps_per_epoch is small}
{I'm hoping I'm injecting sufficient noise in this extended convergence analysis process to prevent the algorithm
  from settling down in saddle points and local optima}

2. CHECKPOINT_TRIGGER_FACTOR = 1000 [Inline with the modified member - NUMBER_OF_TRAINING_EPOCHS]
{Fostering a compromise - a treaty between the limited storage on board the training machine and checkpoint diversity}

3. Adding two additional RNN layers for better correlation tracking -> a. RNN_LAYER_1 => NUMBER_OF_RNN_UNITS_1 = 3900
                                                                       b. RNN_LAYER_2 => NUMBER_OF_RNN_UNITS_2 = 2600
                                                                       c. RNN_LAYER_3 => NUMBER_OF_RNN_UNITS_3 = 1300

4. Changing the BATCH_SIZE to 105 from 65 to include all the <input, target> sequence pairs in one batch

5. Adding a new hyper-parameter BUFFER_SIZE for shuffling the sequences before model training

6. Introducing data shuffling for more randomized training -> one-step look-ahead in sequenced data allows shuffling

7. Modifying the steps_per_epoch field in the model.fit() routine because it was limiting the first time around

8. Changing the embedding size to 3000 from 2600 for better lower-dimensional representation for the vocab of size 9900

9. Adding a new parameter w.r.t the validation and/or testing phase - VALIDATION_LOOK_BACK_CONTEXT_LENGTH_FACTOR (60%)

10. Changing the look-back context logic in the predict() routine - this offers better accuracy according to my
Scrimmage runs [window-size for the initial trigger = 0.6 * 10 = 6 => [6948 6953]]
"""

# The imports
import os
import time
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
                                  api_key='CkvC8nBeRGGIPsnxKzri')


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
    # ...in order to prevent it from settling down at a saddle point.
    # Furthermore, we can better leverage the CUDA capabilities of the NVIDIA GPU if mini_batch_size > 32.
    # Everything constitutes one batch leading to one step per epoch.
    BATCH_SIZE = 105

    # Size of the buffer used for shuffling the sequences during batching, before model training
    BUFFER_SIZE = 1050

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
    PROJECTED_VECTOR_SIZE = 3000

    # The checkpoint directory
    CHECKPOINT_DIRECTORY = './checkpoints'

    # The number of training epochs
    NUMBER_OF_TRAINING_EPOCHS = 10000

    # The checkpoint trigger factor
    CHECKPOINT_TRIGGER_FACTOR = 1000

    # The number of units in the first RNN layer
    NUMBER_OF_RNN_UNITS_1 = 3900

    # The number of units in the second RNN layer
    NUMBER_OF_RNN_UNITS_2 = 2600

    # The number of units in the third RNN layer
    NUMBER_OF_RNN_UNITS_3 = 1300

    # Training data limit
    TRAINING_DATA_LIMIT = 6954

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
    VALIDATION_LOOK_BACK_CONTEXT_LENGTH_FACTOR = 0.6

    # The initialization sequence
    def __init__(self):
        print('[INFO] RNNStockAnalysis Initialization: Bringing things up...')
        # The standard checkpoint naming convention checkpoint_{epoch_number}
        self.checkpoint_prefix = os.path.join(self.CHECKPOINT_DIRECTORY,
                                              'checkpoint_{epoch}')
        # The standard tensorboard logging convention
        self.tensorboard_logging_identifier = 'tensorboard-logs/{}'
        # The pragmatic stock price limits and the precision parameters are encapsulated in a namedtuple
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
            lambda x: round(x,
                            precision_cutoff))
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
        print('[INFO] RNNStockAnalysis Initialization: Data Visualization Figure is available at {}'.format(
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
        # Create individual data samples and convert the data into sequences of <lookback_context_length>
        # Sequences of length 66 will be created
        self.sequenced_data = tensorflow.data.Dataset.from_tensor_slices(self.training_data).batch(
            self.LOOK_BACK_CONTEXT_LENGTH + 1,
            drop_remainder=True)
        # Split the data into inputs and targets
        # Shuffle the data and generate one batch of size 105 [every available training example in this batch]
        # <Input is of length 65> and <Target is right shifted by one along the time axis and is of length 65>
        # As of this variant, everything constitutes one batch
        self.split_dataset = self.sequenced_data.map(lambda x: (x[:-1],
                                                                x[1:])).shuffle(
            self.BUFFER_SIZE).batch(self.BATCH_SIZE,
                                    drop_remainder=True)
        # The model
        self.model = None
        # GPU Availability
        self.gpu_availability = tensorflow.test.is_gpu_available()
        print('[INFO] RNNStockAnalysis Initialization: GPU Availability - [{}]'.format(self.gpu_availability))

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
                # Project the contextual vector onto a dense, lower-dimensional, continuous vector space
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

                # The Hinton dropout layer
                # tensorflow.keras.layers.Dropout(rate=1 - self.KEEP_PROBABILITY),

                # The fully connected neural network
                # A classification-type output onto the vocabulary
                # The model outputs a <LOOK_BACK_CONTEXT_LENGTH x NUMBER_OF_CLASS_LABELS> tensor consisting of...
                # ...unnormalized log probabilities (logits) w.r.t this multi-class classification problem.
                tensorflow.keras.layers.Dense(len(self.available_vocabulary))
            ])
            # Print a summary of the designed model
            print('[INFO] RNNStockAnalysis build: A summary of the designed model is given below...')
            model.summary()
            self.model = (lambda: self.model,
                          lambda: model)[initial_build]()
            return True, model
        except Exception as e:
            print('[ERROR] RNNStockAnalysis build: Exception caught while building the model - {}'.format(e))
            # Detailed stack trace
            traceback.print_tb(e.__traceback__)
            return False, None

    @staticmethod
    # The cost function for the defined model
    def cost_function(y_true_values, y_predicted_values):
        # Sparse Categorical Cross-Entropy is chosen because we have a large number of mutually exclusive classes, i.e.,
        # ...non-binary output labels in a standard multi-class classification problem
        return tensorflow.keras.losses.sparse_categorical_crossentropy(y_true=y_true_values,
                                                                       y_pred=y_predicted_values,
                                                                       from_logits=True)

    # Set the model up with the optimizer and the cost function
    def compile(self):
        try:
            # The Adam Optimizer, Sparse Categorical Cross-Entropy Cost Function, and Visualization Cost Metrics...
            # I need the sparse_categorical_crossentropy metric for cost function progression visualization
            self.model.compile(optimizer=tensorflow.train.AdamOptimizer(),
                               loss=self.cost_function,
                               # Analysis: Do you see a correlation between the cost function and this metric on...
                               # ...Tensorboard?
                               metrics=[tensorflow.keras.metrics.sparse_categorical_crossentropy])
            return True
        except Exception as e:
            print('[ERROR] RNNStockAnalysis compile: Exception caught while compiling the model - {}'.format(e))
            # Detailed stack trace
            traceback.print_tb(e.__traceback__)
            return False

    # Train the model and Visualize the model's progression during training
    def train(self):
        # Evaluate the steps_per_epoch for populating the eponymous member in the model.fit() routine
        # steps_per_epoch = 6955 // (65 * 105) = 1
        steps_per_epoch = self.TRAINING_DATA_LIMIT // (self.LOOK_BACK_CONTEXT_LENGTH * self.BATCH_SIZE)
        try:
            # TODO: Add a logging hook as a callback and include it in the 'callbacks' collection within the fit routine
            # Checkpoint feature callback
            checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_prefix,
                                                                             monitor='loss',
                                                                             save_weights_only=True,
                                                                             # Save all checkpoints - vScrimmage
                                                                             save_best_only=False,
                                                                             verbose=1,
                                                                             mode='min',
                                                                             period=self.CHECKPOINT_TRIGGER_FACTOR)
            # Tensorboard callback
            tensorboard = tensorflow.keras.callbacks.TensorBoard(log_dir=self.tensorboard_logging_identifier.format(
                time.time()))
            # Visualize the progression of the cost function during training
            training_history = self.model.fit(self.split_dataset.repeat(),
                                              epochs=self.NUMBER_OF_TRAINING_EPOCHS,
                                              steps_per_epoch=steps_per_epoch,
                                              callbacks=[checkpoint_callback,
                                                         tensorboard])
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
            # I might be loading a checkpoint where the model had a high loss, ...
            # ...But I have a Scrimmage variant to offset this...
            modified_model.load_weights(tensorflow.train.latest_checkpoint(self.CHECKPOINT_DIRECTORY))
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
            context = self.training_data[len(self.training_data) - int(
                self.VALIDATION_LOOK_BACK_CONTEXT_LENGTH_FACTOR * self.LOOK_AHEAD_SIZE):]

            # print('[INFO] RNNStockAnalysis predict: The initial look-back context in the predict() routine is: '
            #       '\n[{}]'.format(context))

            # Reset the states of the RNN
            modified_model.reset_states()
            # Iterate through multiple predictions in a chain
            for i in range(self.LOOK_AHEAD_SIZE):
                context = tensorflow.expand_dims(context,
                                                 0)
                prediction = modified_model(context)
                # Remove the useless dimension and inject noise into the provided prediction in order to push it out...
                # ...of saddle points
                prediction = tensorflow.squeeze(prediction,
                                                0)
                # Use the tensorflow provided softmax function to convert the logits into probabilities and extract...
                # ...the highest probability class from the multinomial output vector
                predicted_price = numpy.argmax(
                    tensorflow.nn.softmax(
                        prediction[-1]
                    )
                )
                # Append the predicted_price to the output collection
                predicted_prices.append(self.integer_to_vocabulary_mapping[predicted_price])
                # Context modification Logic - Caching the most recent transaction and right-shifting the window...
                context = numpy.append(tensorflow.squeeze(context,
                                                          0),
                                       [predicted_price],
                                       axis=0)
                context = context[1:]

        except Exception as e:
            print('[ERROR] RNNStockAnalysis predict: Exception caught during prediction - {}'.format(e))
            # Detailed stack trace
            traceback.print_tb(e.__traceback__)
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
    final_analysis_figure = dict(data=[real_final_analysis_trace,
                                       generated_final_analysis_trace],
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
    if rnnStockAnalysis.build_model()[0] and \
            rnnStockAnalysis.compile() and \
            rnnStockAnalysis.train()[0]:
        print('[INFO] RNNStockAnalysis Trigger: The model has been built, compiled, and trained! '
              'Evaluating the model...')
        visualize_predictions(rnnStockAnalysis,
                              rnnStockAnalysis.stock_prices_testing,
                              rnnStockAnalysis.predict())
    else:
        print('[INFO] RNNStockAnalysis Trigger: The operation failed due to previous errors!')
