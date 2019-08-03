# |Bleeding-Edge Productions|
# This entity encapsulates an intelligent model to predict hourly pollution indices using RNNs in TensorFlow.
# This model can be extended to predict link states in networks [ Link_Up Link_Up Link_Up Link_Down Link_Down ].
# Author: Bharath Keshavamurthy {bkeshava}
# Organization: DC NX-OS, CISCO Systems Inc.
# Copyright (c) 2019. All Rights Reserved.

# TODO: A logging framework instead of all these formatted print statements

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
                                  api_key='qTXrG3oefYkdFtXVjYcv')


# This class predicts the hourly pollution indices in a given location leveraging capabilities of RNNs in the...
# ...high-level Keras API within TensorFlow
# Inspired by Language Modelling
class RNNPollutionIndexAnalysis(object):
    # The column key for the date-time attribute
    DATE_TIME_COLUMN_KEY = 'DateTime'

    # The column key for the pollution index attribute
    POLLUTION_INDEX_KEY = 'Pollution_Index'

    # The cost visualization metric
    # tensorflow.keras.metrics.sparse_categorical_crossentropy
    COST_METRIC = 'sparse_categorical_crossentropy'

    # Batch size
    # This seems to be the best mini_batch_size for injecting the appropriate amount of noise into the SGD process...
    # ...in order to prevent it from settling down at a saddle point.
    # Furthermore, we can better leverage the CUDA capabilities of the NVIDIA GPU if mini_batch_size > 32.
    BATCH_SIZE = 85

    # Size of the buffer used for shuffling the sequences during batching, before model training
    BUFFER_SIZE = 1050

    # The pragmatic limits of the pollution indices
    PRAGMATIC_POLLUTION_INDEX_LIMITS = namedtuple('Limits',
                                                  ['lower_limit',
                                                   'upper_limit',
                                                   'precision'])

    # The length of the look-back context
    # A lookback context length of a week (24 hours * 7 days a week = 168 hours => 168 examples)
    LOOK_BACK_CONTEXT_LENGTH = 168

    # The length of the look-ahead predictions = The length of the test data set
    # 24 hours of look-ahead => Use the context of a week to determine what happened a day into the future
    LOOK_AHEAD_SIZE = 24

    # The size of the projected vector space
    # A lower dimensional, dense, continuous vector space
    PROJECTED_VECTOR_SIZE = 2048

    # The checkpoint directory
    CHECKPOINT_DIRECTORY = './checkpoints'

    # The number of training epochs
    NUMBER_OF_TRAINING_EPOCHS = 10000

    # The checkpoint trigger factor
    CHECKPOINT_TRIGGER_FACTOR = 1000

    # The number of units in the first RNN layer
    NUMBER_OF_RNN_UNITS_1 = 4096

    # The number of units in the second RNN layer
    NUMBER_OF_RNN_UNITS_2 = 2048

    # The number of units in the third RNN layer
    NUMBER_OF_RNN_UNITS_3 = 1024

    # Training data limit
    TRAINING_DATA_LIMIT = 43776

    # Plotly Scatter mode
    PLOTLY_SCATTER_MODE = 'lines+markers'

    # Parameters of the model vocabulary
    LOWER_LIMIT = 0.0
    UPPER_LIMIT = 1000.0
    PRECISION = 1.0

    # The initialization sequence
    def __init__(self):
        print('[INFO] RNNPollutionIndexAnalysis Initialization: Bringing things up...')
        # The standard checkpoint naming convention checkpoint_{epoch_number}
        self.checkpoint_prefix = os.path.join(self.CHECKPOINT_DIRECTORY,
                                              'checkpoint_{epoch}')
        # The standard tensorboard logging convention
        self.tensorboard_logging_identifier = 'tensorboard-logs/{}'
        # The pragmatic pollution index limits and the precision parameters are encapsulated in a namedtuple
        # This parameterizes the available vocabulary
        self.pragmatic_pollution_index_information = self.PRAGMATIC_POLLUTION_INDEX_LIMITS(lower_limit=self.LOWER_LIMIT,
                                                                                           upper_limit=self.UPPER_LIMIT,
                                                                                           precision=self.PRECISION)
        precision_cutoff = len(str(self.PRECISION)) - str(self.PRECISION).index('.') - 1
        # The available vocabulary for this problem
        self.available_vocabulary = [float(str(x)[:str(x).index('.') + precision_cutoff + 1]) for x in numpy.arange(
            self.pragmatic_pollution_index_information.lower_limit,
            self.pragmatic_pollution_index_information.upper_limit,
            self.pragmatic_pollution_index_information.precision)]
        # Create a mapping from vocabulary space to the set of all positive integers (Z_{++})
        self.vocabulary_to_integer_mapping = {element: integer for integer, element in enumerate(
            self.available_vocabulary)}
        # Create a mapping from the set of all positive integers (Z_{++}) to the vocabulary space
        self.integer_to_vocabulary_mapping = {integer: element for integer, element in enumerate(
            self.available_vocabulary
        )}
        # Load the data
        dataframe = pd.read_csv('datasets/pollution.csv',
                                usecols=[0,
                                         1])
        # Rename the columns for aesthetics
        dataframe.columns = [self.DATE_TIME_COLUMN_KEY,
                             self.POLLUTION_INDEX_KEY]
        # Extract the attributes
        self.dates = dataframe[self.DATE_TIME_COLUMN_KEY]
        self.pollution_indices = dataframe[self.POLLUTION_INDEX_KEY].apply(
            lambda x: round(x,
                            precision_cutoff))
        # Visualize the progression of the training indices over time
        initial_visualization_trace = go.Scatter(x=self.dates,
                                                 y=self.pollution_indices.values,
                                                 mode=self.PLOTLY_SCATTER_MODE)
        initial_visualization_layout = dict(title='Time-Series Progression of the Pollution Index Attribute',
                                            xaxis=dict(title='Time'),
                                            yaxis=dict(title='Pollution Index'))
        initial_visualization_fig = dict(data=[initial_visualization_trace],
                                         layout=initial_visualization_layout)
        initial_fig_url = plotly.plotly.plot(initial_visualization_fig,
                                             filename='Variations_In_Pollution_Indices')
        # Print the URL in case you're on an environment where a GUI is not available
        print('[INFO] RNNPollutionIndexAnalysis Initialization: Data Visualization Figure is available at {}'.format(
            initial_fig_url
        ))
        # The data set for training - [0, 43775]
        self.pollution_indices_training = self.pollution_indices.values[:self.TRAINING_DATA_LIMIT]
        # Integer mapped training data
        self.training_data = numpy.array(
            [self.vocabulary_to_integer_mapping[x] for x in self.pollution_indices_training]
        )
        # The data set for testing - [43776 43799]
        self.dates_testing = self.dates[self.TRAINING_DATA_LIMIT:self.TRAINING_DATA_LIMIT + self.LOOK_AHEAD_SIZE]
        self.pollution_indices_testing = self.pollution_indices.values[
                                         self.TRAINING_DATA_LIMIT:self.TRAINING_DATA_LIMIT + self.LOOK_AHEAD_SIZE]
        # Create individual data samples and convert the data into sequences of <lookback_context_length>
        # Sequences of length 169 will be created
        self.sequenced_data = tensorflow.data.Dataset.from_tensor_slices(self.training_data).batch(
            self.LOOK_BACK_CONTEXT_LENGTH + 1,
            drop_remainder=True)
        # Split the data into inputs and targets
        # Shuffle the data and generate batches of size 85
        # <Input is of length 168> and <Target is right shifted by one along the time axis and is of length 168>
        self.split_dataset = self.sequenced_data.map(lambda x: (x[:-1],
                                                                x[1:])).shuffle(
            self.BUFFER_SIZE).batch(self.BATCH_SIZE,
                                    drop_remainder=True)
        # The model
        self.model = None
        # GPU Availability
        self.gpu_availability = tensorflow.test.is_gpu_available()
        print('[INFO] RNNPollutionIndexAnalysis Initialization: GPU Availability - [{}]'.format(self.gpu_availability))

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
                # The fully connected neural network
                # A classification-type output onto the vocabulary
                # The model outputs a <LOOK_BACK_CONTEXT_LENGTH x NUMBER_OF_CLASS_LABELS> tensor consisting of...
                # ...unnormalized log probabilities (logits) w.r.t this multi-class classification problem.
                tensorflow.keras.layers.Dense(len(self.available_vocabulary))
            ])
            # Print a summary of the designed model
            print('[INFO] RNNPollutionIndexAnalysis build: A summary of the designed model is given below...')
            model.summary()
            self.model = (lambda: self.model,
                          lambda: model)[initial_build]()
            return True, model
        except Exception as e:
            print('[ERROR] RNNPollutionIndexAnalysis build: Exception caught while building the model - {}'.format(e))
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
            print(
                '[ERROR] RNNPollutionIndexAnalysis compile: Exception caught while compiling the model - {}'.format(e))
            # Detailed stack trace
            traceback.print_tb(e.__traceback__)
            return False

    # Train the model and Visualize the model's progression during training
    def train(self):
        # Evaluate the steps_per_epoch for populating the eponymous member in the model.fit() routine
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
            print(
                '[INFO] RNNPollutionIndexAnalysis train: Cost Progression Visualization Figure available at {}'.format(
                    cost_progression_fig_url
                ))
            return True, training_history
        except Exception as e:
            print('[ERROR] RNNPollutionIndexAnalysis train: Exception caught while training the model - {}'.format(e))
            # Detailed stack trace
            traceback.print_tb(e.__traceback__)
            return False, None

    # Predict the next ${LOOK_AHEAD_SIZE} pollution indices
    def predict(self):
        # The output to be returned
        predicted_indices = []
        # GPU Availability - Check again in case something took up the discrete graphics capabilities of the machine
        self.gpu_availability = tensorflow.test.is_gpu_available()
        print('[INFO] RNNPollutionIndexAnalysis Initialization: GPU Availability - [{}]'.format(self.gpu_availability))
        # Build a new model with a batch-size of 1 -> Load the weights from the trained model -> Reshape the input layer
        status, modified_model = self.build_model(initial_build=False,
                                                  batch_size=1)
        if status is False:
            print('[ERROR] RNNPollutionIndexAnalysis predict: The operation failed due to previous errors!')
            return
        try:
            modified_model.load_weights(tensorflow.train.latest_checkpoint(self.CHECKPOINT_DIRECTORY))
            modified_model.build(tensorflow.TensorShape([1,
                                                         None]))
            # The trigger carried forward from the training dataset
            context = self.training_data[len(self.training_data) - self.LOOK_BACK_CONTEXT_LENGTH:]
            # Reset the states of the RNN
            modified_model.reset_states()
            # Iterate through multiple predictions in a chain
            for i in range(self.LOOK_AHEAD_SIZE):
                context = tensorflow.expand_dims(context,
                                                 0)
                # Make a stateful prediction
                prediction = modified_model(context)
                # Remove the useless dimension
                prediction = tensorflow.squeeze(prediction,
                                                0)
                # Use the tensorflow provided softmax function to convert the logits into probabilities and...
                # ...extract the highest probability class from the multinomial output vector
                predicted_index = numpy.argmax(
                    tensorflow.nn.softmax(
                        prediction[-1]
                    )
                )
                # Append the predicted_index to the output collection
                predicted_indices.append(self.integer_to_vocabulary_mapping[predicted_index])
                # Context modification Logic - Append the new value to the context
                context = numpy.append(numpy.squeeze(context,
                                                     0),
                                       [predicted_index],
                                       axis=0)
                # Move the context window one step to the right
                context = context[1:]
        except Exception as e:
            print('[ERROR] RNNPollutionIndexAnalysis predict: Exception caught during prediction - {}'.format(e))
            # Detailed stack trace
            traceback.print_tb(e.__traceback__)
        return predicted_indices

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] RNNPollutionIndexAnalysis Termination: Tearing things down...')
        # Nothing to do


# Visualize the predictions against the corresponding true values
def visualize_predictions(obj, _true_values, _predicted_values):
    x_axis = obj.dates_testing
    real_final_analysis_trace = go.Scatter(x=x_axis,
                                           y=_true_values,
                                           mode=obj.PLOTLY_SCATTER_MODE,
                                           name='True Pollution Indices')
    generated_final_analysis_trace = go.Scatter(x=x_axis,
                                                y=_predicted_values,
                                                mode=obj.PLOTLY_SCATTER_MODE,
                                                name='Pollution Indices predicted by the RNN model')
    final_analysis_layout = dict(title='Analysis of the predicted pollution indices vs the true pollution indices from '
                                       'the test data set',
                                 xaxis=dict(title='Time'),
                                 yaxis=dict(title='Pollution Indices'))
    final_analysis_figure = dict(data=[real_final_analysis_trace,
                                       generated_final_analysis_trace],
                                 layout=final_analysis_layout)
    final_analysis_url = plotly.plotly.plot(final_analysis_figure,
                                            filename='Prediction_Analysis_Test_Dataset')
    # An additional array creation logic is inserted in the print statement to format both collections identically...
    # ...for aesthetics.
    print('[INFO] RNNPollutionIndexAnalysis visualize_predictions: Look-back Context Length - [{}]'.format(
        obj.LOOK_BACK_CONTEXT_LENGTH))
    print('[INFO] RNNPollutionIndexAnalysis visualize_predictions: Look-ahead Size - [{}]'.format(
        obj.LOOK_AHEAD_SIZE))
    print('[INFO] RNNPollutionIndexAnalysis visualize_predictions: The true pollution indices are - \n{}'.format(
        [k for k in _true_values]))
    print('[INFO] RNNPollutionIndexAnalysis visualize_predictions: The predicted pollution indices are  - \n{}'.format(
        [k for k in _predicted_values]))
    # Print the URL in case you're on an environment where a GUI is not available
    print(
        '[INFO] RNNPollutionIndexAnalysis visualize_predictions: The final prediction analysis visualization figure is '
        'available at {}'.format(final_analysis_url))
    return None


# Run Trigger
if __name__ == '__main__':
    print('[INFO] RNNPollutionIndexAnalysis Trigger: Starting system assessment!')
    rnnPollutionIndexAnalysis = RNNPollutionIndexAnalysis()
    # TODO: Use an ETL-type pipeline for this sequence of operations on the model
    if rnnPollutionIndexAnalysis.build_model()[0] and \
            rnnPollutionIndexAnalysis.compile() and \
            rnnPollutionIndexAnalysis.train()[0]:
        print('[INFO] RNNPollutionIndexAnalysis Trigger: The model has been built, compiled, and trained! '
              'Evaluating the model...')
        visualize_predictions(rnnPollutionIndexAnalysis,
                              rnnPollutionIndexAnalysis.pollution_indices_testing,
                              rnnPollutionIndexAnalysis.predict())
    else:
        print('[INFO] RNNPollutionIndexAnalysis Trigger: The operation failed due to previous errors!')
