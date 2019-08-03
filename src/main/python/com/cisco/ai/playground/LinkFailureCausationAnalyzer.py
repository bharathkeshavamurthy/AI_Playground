# |Bleeding-Edge Productions|
# This entity details a technique to provide an explanation for the predictions made by black-box deep-learning models.
# Author: Bharath Keshavamurthy {bkeshava}
# Organization: DC NX-OS, CISCO Systems Inc.
# Copyright (c) 2019. All Rights Reserved.

# TODO: A logging framework instead of all these formatted print statements

"""
Change Log - 23-July-2019:
@author: Bharath Keshavamurthy <bkeshava at cisco dot com>

1. Increasing the confidence bound from 100 to 200 for greater stability during convergence analysis

2. Removing the additional current_convergence_check in the projection while loop which is offset by the increased
    convergence threshold

3. Fixing the bug in the global deregister() routine

4. Increasing the number of training epochs in the NeuralNetworkClassificationEngine from 10 to 3000: Quadro P4000 run

5. Lowering the kernel_width ($\\sigma^2$) (to 0.2) for providing better weights to the closest neighbors

6. Increasing the regularization_constant ($\\alpha$) (to 150) to be more inline with the change in the kernel_width

7. @ipasha: Adding a confusion matrix to visualize the number of false positives and the number of false negatives

8. Using a predefined hard-coded sample index of 11516 to facilitate localized curve-fitting analysis for negative
classifications only [allows for a reduced set of abnormal/anomalous feature values - narrows our focus]

9. Increasing the batch size from 144 to 256 for reduced noise injection in the SGD process

10. Increasing the number of training epochs for convergences that are more stable

11. Changing the initial weights {to (60.0, 60.0)} in the PGD process during localized curve-fitting in order to get a
better learning range and hence, more accurate weights for the local models under PGD optimization
"""

# The imports
import sys
import math
import numpy
import plotly
import pandas
import random
import itertools
import traceback
from sklearn.metrics import confusion_matrix

# import threading

import tensorflow

# import progressbar

from tabulate import tabulate
from recordclass import recordclass
from collections import namedtuple

# Get rid of the false positive pandas "SettingWithCopyWarning" message: There's no chaining happening in this entity
pandas.options.mode.chained_assignment = None

# Plotly user account credentials for visualization
plotly.tools.set_credentials_file(username='bkeshava',
                                  api_key='RHqYrDdThygiJEPiEW5S')


# The optimizer
# The Projection Gradient Descent (PGD) entity for Constrained Convex Optimization
# This algorithmic implementation is suitable only for polyhedra.
# Something other than vector projection needs to be used for feasible sets which are not...
# ...finite intersections of half-spaces.
# TODO: Projection Gradient Descent optimizer implementations for problems with non-polyhedron feasible sets
class ProjectionGradientDescent(object):

    # The initialization sequence
    def __init__(self, _dimensionality, _intercept_constraint, _line_segments, _rationale_engine):
        print('[INFO] ProjectionGradientDescent Initialization: Bringing things up...')
        # The initialization status flag
        self.status = False
        # The support for dimensionality is currently limited to ${SUPPORTED_DIMENSIONALITY}-dimensional feasible sets.
        # You may have to use a Lagrangian based formulation with Stochastic Projection Gradient Descent in the Dual...
        # ...for dimensionalities greater than ${SUPPORTED_DIMENSIONALITY).
        # TODO: Projection Gradient Descent for N-dimensional feasible sets or domains (N > ${SUPPORTED_DIMENSIONALITY})
        if _dimensionality != SUPPORTED_DIMENSIONALITY:
            print('[ERROR] ProjectionGradientDescent Initialization: Support for problems with dimensionality other '
                  'than [{}] is not currently available. Please check back later!'.format(SUPPORTED_DIMENSIONALITY))
            return
        # The dimensionality of the problem
        self.dimensionality = _dimensionality
        # The initial weights [\theta_0 \theta_1 ...]
        self.initial_weights = (24.0, -62.0)
        # The intercept constraint in the given linear inequality, i.e. the regularization constant
        self.intercept_constraint = _intercept_constraint
        # The default step size during training
        # TODO: Use Exact Line Search or Backtracking Line Search for choosing the step size more scientifically
        self.default_step_size = 0.01
        # Two-dimensional Projection Gradient Descent
        # The line segments of the polyhedron which constitute the feasible set
        self.line_segments = _line_segments
        # The iterations array which models the x-axis of the convergence plot
        self.iterations = []
        # The function values array which models the y-axis of the convergence plot
        self.function_values = []
        # The confidence bound for convergence
        self.confidence_bound = 200
        # The number of digits post the decimal point to be considered for convergence analysis
        self.convergence_significance = 10  # modelled from significant digits of a number
        # The maximum number of iterations allowed during training
        self.max_iterations = 1e6
        # The rationaleEngine for loss function and gradient evaluation
        self.rationale_engine = _rationale_engine
        # The initialization has been completed successfully - captured by this status flag
        self.status = True

    # Vector Projection utility routine
    # Return the distance and the closest point w.r.t to that specific line segment
    def vector_projection(self, oob_point, line_segment):
        x1, y1 = line_segment[0]
        x2, y2 = line_segment[1]
        # The out-of-bounds point which needs to be constrained based on the equality and/or inequality constraints
        x, y = oob_point
        # Is the point outside the feasible set?
        if (abs(x) + abs(y)) > self.intercept_constraint:
            dot_product = ((x - x1) * (x2 - x1)) + ((y - y1) * (y2 - y1))
            norm_square = ((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1))
            # Invalid base vector
            if norm_square == 0:
                print('[ERROR] ProjectionGradientDescent vector_projection: The norm square of the base vector is 0!')
                return False, None, None
            # Find the component of the operand vector along the base vector
            # This is a standard vector projection technique
            component = dot_product / norm_square
            # (x1, y1) is the closest point - obtuse angle
            if component < 0:
                return True, math.sqrt(((x - x1) * (x - x1)) + ((y - y1) * (y - y1))), (x1, y1)
            # (x2, y2) is the closest point - acute angle, larger magnitude
            elif component > 1:
                return True, math.sqrt(((x - x2) * (x - x2)) + ((y - y2) * (y - y2))), (x2, y2)
            # Find the closest point using the projection component
            else:
                # x-component => what portion of the base vector will the operand vector be projected upon w.r.t x
                x_closest = x1 + (component * (x2 - x1))
                # y-component => what portion of the base vector will the operand vector be projected upon w.r.t y
                y_closest = y1 + (component * (y2 - y1))
                return True, math.sqrt(((x - x_closest) * (x - x_closest)) + ((y - y_closest) * (y - y_closest))), (
                    x_closest, y_closest)
        else:
            # No Projection
            return True, 0, oob_point

    # The main projection wrapper routine
    def projection(self, oob_point):
        # A collection to house the output tuples of the vector projection routine
        vector_projection_output_tuples_collection = []
        # A namedtuple for cleaner storage
        vector_projection_output_tuple = namedtuple('output_tuple',
                                                    ['distance',
                                                     'closest_point'])
        for line_segment in self.line_segments:
            # Perform vector projection of the operand vector (oob_point) with respect to the current line_segment
            status_flag, distance, closest_point = self.vector_projection(oob_point,
                                                                          line_segment)
            # No errors during the vector projection operation
            if status_flag is False:
                print('[ERROR] ProjectionGradientDescent projection: Something went wrong during vector projection. '
                      'Please refer to the earlier logs for more information on what went wrong!')
                return False, None
            vector_projection_output_tuples_collection.append(
                vector_projection_output_tuple(distance=distance,
                                               closest_point=closest_point))
        # Return the closest_point of all the projection outputs
        return True, min(vector_projection_output_tuples_collection,
                         key=lambda x: x.distance).closest_point

    # The convergence check for the Projection Gradient Descent algorithm
    def convergence_check(self, previous_point, current_point):
        if previous_point is not None and \
                (numpy.round(previous_point[0],
                             self.convergence_significance),
                 numpy.round(previous_point[1],
                             self.convergence_significance)) == (numpy.round(current_point[0],
                                                                             self.convergence_significance),
                                                                 numpy.round(current_point[1],
                                                                             self.convergence_significance)):
            status, (x_projected_point, y_projected_point) = self.projection(current_point)
            if status:
                # Check if the projection of the point is the point itself
                # Termination condition of PGD: [\vec{\theta}^*]^{+} = \vec{\theta}^*
                # Again, this is a 2-dimensional PGD routine - this is not valid for PGDs of different dimensionalities
                if (numpy.round(x_projected_point,
                                self.convergence_significance),
                    numpy.round(y_projected_point,
                                self.convergence_significance)) == (numpy.round(current_point[0],
                                                                                self.convergence_significance),
                                                                    numpy.round(current_point[1],
                                                                                self.convergence_significance)):
                    return True
            else:
                print('[WARN] ProjectionGradientDescent convergence_check: Something went wrong during the convergence '
                      'check operation. Please refer to the earlier logs for more information on this error.')
        return False

    # The main optimization routine
    def optimize(self, _step_size):
        # Pick the step size
        step_size = (lambda: self.default_step_size,
                     lambda: _step_size)[_step_size is not None and
                                         isinstance(_step_size, float) and
                                         _step_size > 0.0]()
        previous_point = None
        # The convergence is theoretically guaranteed in finite time irrespective of the initialization.
        # For a detailed proof, please refer to "Convex Optimization" by Boyd and Vandenberghe.
        current_point = self.initial_weights
        confidence = 0
        iteration_count = 0
        # Limit the number of iterations allowed
        # Enable confidence check for convergence
        while (iteration_count < self.max_iterations) and (confidence < self.confidence_bound):
            if self.convergence_check(previous_point,
                                      current_point):
                # Increment confidence, if converged
                confidence += 1
            previous_point = current_point
            # Collections for cost function progression visualization
            self.iterations.append(iteration_count)
            self.function_values.append(self.rationale_engine.evaluate_loss(current_point))
            # The core: Projection Gradient Descent
            gradient = self.rationale_engine.evaluate_gradients(current_point,
                                                                self.dimensionality)
            everything_is_okay, current_point = self.projection((current_point[0] - (step_size * gradient[0]),
                                                                 current_point[1] - (step_size * gradient[1])))
            # Propagating failure upward to the calling routine
            # TODO: Maybe send a callback into the mess below and check its status upon its return?
            #  It's much cleaner that way!
            if everything_is_okay is False:
                return False, None, None
            iteration_count += 1
        # After convergence, visualize the progression of the cost function
        # TODO: Do I need the visualization of the optimization process?
        #  The number of plots generated increases exponentially with the number of features in the classifier.
        #  So, for now, comment out the visualization call for every locally interpretable model optimization.
        # self.visualize()
        # Return the converged loss (function value) and the converged parameters of the regression model
        return True, self.function_values[-1], current_point

    # Visualize the progression of the cost function
    def visualize(self):
        try:
            # The data trace
            cost_function_trace = plotly.graph_objs.Scatter(x=self.iterations,
                                                            y=self.function_values,
                                                            mode=PLOTLY_SCATTER_MODE)
            # The layout
            plot_layout = dict(title='Cost Progression Analysis during Projection Gradient Descent',
                               xaxis=dict(title='Iterations'),
                               yaxis=dict(title='Cost Function'))
            # The figure instance
            cost_function_progression_fig = dict(data=[cost_function_trace],
                                                 layout=plot_layout)
            # The url for the figure
            cost_function_progression_fig_url = plotly.plotly.plot(cost_function_progression_fig,
                                                                   filename='PGD_Cost_Progression_Analysis')
            print('[INFO] ProjectionGradientDescent visualize: The cost progression visualization figure '
                  'is available at [{}]'.format(cost_function_progression_fig_url))
        except Exception as e:
            print('[ERROR] ProjectionGradientDescent visualize: Exception caught in the plotly cost function '
                  'progression visualization routine - {}'.format(e))
            traceback.print_tb(e.__traceback__)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] ProjectionGradientDescent Termination: Tearing things down...')


# A utilities class
class Utility(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] Utility Initialization: Bringing things up...')

    # A utility method
    # Calculate the L1 norm of the given vector
    @staticmethod
    def l1_norm(vector):
        return sum(abs(x) for x in vector)

    # A utility method
    # Calculate the L2 norm of the given vector
    @staticmethod
    def l2_norm(vector):
        return math.sqrt(sum(x ** 2 for x in vector))

    # A utility method
    # Calculate the exponential kernel coefficient
    def exponential_kernel_coefficient(self, _vector_x, _vector_z, _width):
        # Dot Product using iterable comprehension
        vector_x = _vector_x.values[0]
        vector_z = _vector_z.values[0]
        dot_product = sum(i * j for i, j in zip(vector_x, vector_z))
        # Cosine similarity evaluation and the subsequent exponential kernel value calculation
        cosine_similarity = dot_product / (self.l2_norm(vector_x) * self.l2_norm(vector_z))
        return math.exp((cosine_similarity ** 2) / _width)

    # A utility method
    # Visualize the progress using the sys.stdout.write() and sys.stdout.flush() routines
    # The current_value OR end_value args should be 'float'
    @staticmethod
    def sys_progress(current_value, end_value, bar_length=100, log_level='TRACE',
                     entity='', routine='', key='Progress'):
        arrow_position = '-' * int(((current_value / end_value) * bar_length) - 1) + '>'
        empty_spaces = ' ' * int((bar_length - len(arrow_position)))
        sys.stdout.write('\r[{}] {} {}: {} - [{}] {}%'.format(log_level,
                                                              entity,
                                                              routine,
                                                              key,
                                                              arrow_position + empty_spaces,
                                                              numpy.round((current_value / end_value) * 100, 2)))
        sys.stdout.flush()

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Utility Termination: Tearing things down...')


# The global view members

# The global utilities instance
utility = Utility()

# The global repository of tasks whose respective engines need to be analyzed for prediction rationale.
task_repository = {}

# The supported dimensionality
# For varying dimensionalities, the software architecture would be to register dimensionality-specific optimizers...
# ...and trigger them accordingly.
# Or, use Stochastic Projection Gradient Descent in the Dual for a global consolidated solution.
SUPPORTED_DIMENSIONALITY = 2

# Plotly Scatter mode
PLOTLY_SCATTER_MODE = 'lines+markers'


# The global view register routine
def register(_id, _task):
    if isinstance(_task, ClassificationTask):
        task_repository[_id] = _task
        print('[INFO] GlobalView register: Successfully added a classification task with ID: [{}] '
              'to the task repository'.format(_id))
        return True
    print('[ERROR] GlobalView register: Invalid classification task entity has been received. All classification tasks'
          'whose predictions are to be analyzed for rationale '
          'have to be instances of {}'.format(ClassificationTask.__name__))
    return False


# The global view de-register routine
def deregister(_id):
    try:
        task = task_repository.pop(_id)
        print('[INFO] GlobalView de-register: Successfully removed task [{}: {}] '
              'from the task repository'.format(_id,
                                                task.__class__.__name__))
    except KeyError:
        print('[ERROR] GlobalView de-register: The task ID [{}] does not exist in the task repository! Nothing to be '
              'done!'.format(_id))


# The parent classification task wrapper
class ClassificationTask(object):

    # The initialization sequence
    def __init__(self, _task_id, _features):
        print('[INFO] ClassificationTask Initialization: Bringing things up...')
        self.registration_status = register(_task_id,
                                            self)
        self.features = _features

    # Abstract contract for model building
    def build_model(self):
        pass

    # Abstract contract for model training
    def train_model(self):
        pass

    # Abstract contract for model evaluation
    def evaluate_model(self):
        pass

    # Abstract contract for making a prediction for a specific sample from the test dataset
    def make_a_prediction(self):
        pass

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb, _id):
        print('[INFO] ClassificationTask Termination: Tearing things down...')
        deregister(_id)


# Pre-Processing semaphore
# proc_semaphore = threading.Lock()


# The concurrency process controller
# def proc_control(classification_engine, feature_name, internal_proc_counter=0.0):
#     # A temporary processed data storage
#     proc_control_array = []
#     # A progress bar instance for progress visualization
#     # progress_bar = progressbar.ProgressBar(maxval=len(self.features[feature_name]),
#     #                                        widgets=[progressbar.Bar('-',
#     #                                                                 '[',
#     #                                                                 ']'),
#     #                                                 progressbar.Percentage()])
#     # progress_bar.start()
#     # Internal feature data pre-processing loop
#     for i in range(len(classification_engine.features[feature_name])):
#         # Progress visualization [progressbar]
#         # progress_bar.update(internal_proc_counter)
#         # Progress visualization [sys]
#         utility.sys_progress(internal_proc_counter,
#                              len(classification_engine.features[feature_name]),
#                              bar_length=20,
#                              log_level='DEBUG',
#                              entity=classification_engine.__class__.__name__,
#                              routine='Initialization',
#                              key='Pre-Processing Progress for {}'.format(feature_name))
#         internal_proc_counter += 1.0
#         proc_control_array.append(classification_engine.process_data(classification_engine.features[feature_name][i],
#                                                                      feature_name))
#     # progress_bar.finish()
#     proc_control_series = pandas.Series(proc_control_array)
#     # Acquire the proc_semaphore before accessing the global dataframe
#     proc_semaphore.acquire()
#     classification_engine.features[feature_name] = proc_control_series
#     # Release the proc_semaphore after accessing the global dataframe
#     proc_semaphore.release()


# This class employs a fully-connected neural network to perform the classification task.
class NeuralNetworkClassificationEngine(ClassificationTask):
    # The task id
    TASK_ID = 'NN_CLASSIFIER_1'

    # A data structure for easy, clean storage and access of data
    DATA = namedtuple('Data',
                      ['features',
                       'label'])

    # Use ${TRAINING_SPLIT} * 100% of the data for training and the remaining for testing and/or validation
    TRAINING_SPLIT = 0.8

    # The number of neurons in the input layer of the NN model
    NUMBER_OF_HIDDEN_UNITS_1 = 4096

    # The number of neurons in the hidden layer of the NN model
    NUMBER_OF_HIDDEN_UNITS_2 = 2048

    # The batch size for training (inject noise into the SGD process - leverage CUDA cores, if available)
    BATCH_SIZE = 256

    # The number of epochs to train the model
    NUMBER_OF_TRAINING_EPOCHS = 3000

    # Process the data before feeding it into the Classifier
    def process_data(self, data, family):
        # Strip the trailing whitespaces from examples of all string features
        data = (lambda: data.strip(),
                lambda: data)[isinstance(data, str) is False]()
        if family not in self.data_processor_memory.keys():
            # Vocabulary Creation
            vocabulary = sorted(set(self.dataframe[family]))
            self.feature_vocabulary_mapping[family] = vocabulary
            # A word to integer mapping for categorical columns
            feature_index_map = (lambda: {u.strip(): i for i, u in enumerate(vocabulary)},
                                 lambda: {k: j for j, k in enumerate(vocabulary)})[isinstance(data, str) is False]()

            # Evaluate the mean and standard deviation of the integer mappings for normalization
            # Add the feature index map, the mean, and the standard deviation of the feature family to the...
            # ...feature vocabulary dict

            # Redundancy check
            std_dev = numpy.std(list(feature_index_map.values()))
            if std_dev == 0.0:
                print('\n[WARN] NeuralNetworkClassificationEngine process_data: The given dataset contains redundant '
                      'information: [{}]'.format(family))
                self.redundant_features.append(family)
            self.data_processor_memory[family] = (feature_index_map,
                                                  numpy.mean(list(feature_index_map.values())),
                                                  std_dev)

        feature_index_map, mean, standard_deviation = self.data_processor_memory[family]
        # Normalize the value according to the stats for that particular family and return the normalized value
        # A corner case check - DivideByZero Error
        if standard_deviation != 0.0:
            return float(feature_index_map[data] - mean) / standard_deviation
        return feature_index_map[data]

    # The initialization sequence
    def __init__(self):
        print('[INFO] NeuralNetworkClassificationEngine Initialization: Bringing things up...')
        # The path to the dataset file
        data_file = 'datasets/link.xlsx'
        # The memory of the data processor
        self.data_processor_memory = {}
        # The useless/redundant columns in the given dataset are housed in this collection
        self.redundant_features = []
        # The feature vocabulary mapping
        self.feature_vocabulary_mapping = {}
        try:
            # Read the dataset
            self.dataframe = pandas.read_excel(data_file)
            # The call to the parent
            ClassificationTask.__init__(self,
                                        self.TASK_ID,
                                        self.dataframe.columns)
            # The complete dataset
            features, labels = self.dataframe[self.dataframe.columns[:-1]], self.dataframe[
                self.dataframe.columns[-1]]

            # Concurrency Collection: Pre-Processing all the feature families concurrently
            # proc_threads = []
            # Processing the input features to make them compatible with the Classification Engine
            # for feature_column in self.features.columns:
            #     proc_thread = threading.Thread(target=proc_control, args=(self,
            #                                                               feature_column,
            #                                                               0.0))
            #     proc_threads.append(proc_thread)
            #     proc_thread.start()
            # # Wait for completion and join the proc threads back to the main thread
            # for thread in proc_threads:
            #     thread.join()

            # Processing the input features to make them compatible with the Classification Engine
            for feature_column in features.columns:

                # # Change the data-type of this column [dtype = numpy.int64] for normalization
                # if isinstance(features[feature_column][0], numpy.int64):
                #     features[feature_column] = features[feature_column].astype(numpy.float)

                internal_proc_counter = 0.0
                for i in range(len(features[feature_column])):
                    internal_proc_counter += 1.0
                    features[feature_column][i] = self.process_data(features[feature_column][i],
                                                                    feature_column)
                    utility.sys_progress(internal_proc_counter,
                                         len(features[feature_column]),
                                         bar_length=20,
                                         log_level='DEBUG',
                                         entity=self.__class__.__name__,
                                         routine='Initialization',
                                         key='Pre-Processing Progress for {}'.format(feature_column))
            # Processing the output labels to make them compatible with the Classification Engine
            # internal_proc_counter = 0.0
            for j in range(len(labels)):
                # It is a contract between the calling entity and this engine that the columns in the dataset be...
                # structured in the standard way, i.e. having all the features to the left of the label in the dataframe
                # TODO: Unnecessary processing - Do I need this?
                # internal_proc_counter += 1.0
                # labels[j] = self.process_data(labels[j], self.dataframe.columns[-1])
                # utility.sys_progress(internal_proc_counter,
                #                      len(labels),
                #                      bar_length=20,
                #                      log_level='DEBUG',
                #                      entity=self.__class__.__name__,
                #                      routine='Initialization',
                #                      key='Pre-Processing Progress for {}'.format(self.dataframe.columns[-1]))

                # A simple conditional within an outer example iterator
                labels[j] = (lambda: 0,
                             lambda: 1)[labels[j].strip() == 'up']()
            # Remove the redundant features from the pre-processed dataframe
            for k in self.redundant_features:
                del features[k]
            print('\n[INFO] NeuralNetworkClassificationEngine Initialization: The redundant features are: '
                  '{}'.format(self.redundant_features))
            print('[INFO] NeuralNetworkClassificationEngine Initialization: The important relevant features are: '
                  '{}'.format(list(features.columns.values)))
            split = math.floor(len(features) * self.TRAINING_SPLIT)
            # The training data
            self.training_features, self.training_labels = features[:split], labels[:split]
            self.training_labels = [k for k in self.training_labels]
            self.training_data = []
            for k in range(len(self.training_features)):
                self.training_data.append(self.DATA(features=self.training_features.loc[[k]],
                                                    label=self.training_labels[k]))
            # The test data
            self.test_features, self.test_labels = features[split:], labels[split:]
            self.test_labels = [k for k in self.test_labels]
            self.test_data = []
            for k in range(len(self.training_features),
                           len(self.test_features)):
                self.test_data.append(self.DATA(features=self.test_features.loc[[k]],
                                                label=self.test_labels[k]))
            # The model
            self.model = None
            # The initialization and the data processing is complete
            # Set the status flag - check for registration success first
            self.status = (lambda: False, lambda: True)[self.registration_status]()
            # Print the parameters of the dataset
            self.number_of_features = len(self.dataframe.columns) - 1
            self.number_of_training_examples = len(self.training_labels)
            self.number_of_test_examples = len(self.test_labels)
            print('\n[INFO] NeuralNetworkClassificationEngine Initialization: The dataset is - \n{}'.format(
                tabulate(self.dataframe,
                         headers='keys',
                         tablefmt='psql')
            ))
            # Print the overall (redundant + non-redundant) number of features along with any other relevant information
            print('[INFO] NeuralNetworkClassificationEngine Initialization: The parameters of the dataset are - '
                  'Number of Features = [{}], '
                  'Number of Training Examples = [{}], and '
                  'Number of Test Examples = [{}]'.format(self.number_of_features,
                                                          self.number_of_training_examples,
                                                          self.number_of_test_examples))
        except Exception as e:
            print('[ERROR] NeuralNetworkClassificationEngine Initialization: Exception caught while initializing '
                  'the NN Classifier - {}'.format(e))
            traceback.print_tb(e.__traceback__)
            self.status = False

    # Build the Neural Network model
    def build_model(self):
        try:
            # Since I'm running on a TensorFlow backend, the model will automatically run on a GPU if one is detected.
            # Construct a standard NN model with one hidden layer and ReLU & sigmoid non-linearities
            self.model = tensorflow.keras.Sequential([
                # The input layer (input_shape = (len(self.training_features.columns),)) and the first hidden layer
                tensorflow.keras.layers.Dense(units=self.NUMBER_OF_HIDDEN_UNITS_1,
                                              input_shape=(len(self.training_features.columns),),
                                              activation=tensorflow.nn.relu),
                # The hidden layer
                tensorflow.keras.layers.Dense(units=self.NUMBER_OF_HIDDEN_UNITS_2,
                                              activation=tensorflow.nn.relu),
                # The output layer
                tensorflow.keras.layers.Dense(units=1,
                                              activation=tensorflow.nn.sigmoid)
            ])
            # Model building is complete!
            print('[INFO] NeuralNetworkClassificationEngine build_model: The model summary is - ')
            self.model.summary()
            return True
        except Exception as e:
            print('[ERROR] NeuralNetworkClassificationEngine build_model: Exception caught while building '
                  'the model - {}'.format(e))
            traceback.print_tb(e.__traceback__)
            return False

    # Train the model on the training dataset
    def train_model(self):
        try:

            # I use an AdamOptimizer in the place of the simpler tensorflow.train.GradientDescentOptimizer()...
            # ...because the AdamOptimizer uses the moving average of parameters and this facilitates...
            # ...faster convergence by settling on a larger effective step-size.

            # Advanced Customized Training
            # optimizer = tensorflow.train.AdamOptimizer()
            # for epoch in range(self.NUMBER_OF_TRAINING_EPOCHS):
            #     for training_example in self.training_data:
            #         with tensorflow.GradientTape() as gradient_tape:
            #             predicted_label = self.model.predict(training_example.features)[0]
            #             cost = tensorflow.keras.losses.sparse_categorical_crossentropy(training_example.label,
            #                                                                            predicted_la   bel)
            #         gradients = gradient_tape.gradient(cost, self.model.trainable_variables)
            #         optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            #     print('[DEBUG] NeuralNetworkClassificationEngine train_model: Epoch {} Cost {}'.format(epoch + 1,
            #                                                                                            cost))
            # # Model training is complete!
            # return True

            # Standard Compilation - loss function definition and optimizer inclusion
            self.model.compile(loss=tensorflow.keras.losses.binary_crossentropy,
                               optimizer=tensorflow.train.AdamOptimizer(),
                               metrics=['accuracy'])
            # Standard Training
            self.model.fit(self.training_features,
                           self.training_labels,
                           batch_size=self.BATCH_SIZE,
                           epochs=self.NUMBER_OF_TRAINING_EPOCHS,
                           verbose=0)
            return True
        except Exception as e:
            print('[ERROR] NeuralNetworkClassificationEngine train_model: Exception caught while training '
                  'the classifier - {}'.format(e))
            traceback.print_tb(e.__traceback__)
            return False

    # After training the model, evaluate the model against the test data
    def evaluate_model(self):
        try:
            test_predictions = [int(k[0]) for k in self.model.predict(self.test_features)]
            # External sklearn confusion_matrix
            print('[INFO] NeuralNetworkClassificationEngine evaluate: '
                  'The confusion matrix with respect to the test data is: '
                  '\n{}'.format(confusion_matrix(self.test_labels,
                                                 test_predictions)))

            # Internal TensorFlow/Keras evaluation
            # prediction_loss, prediction_accuracy = self.model.evaluate(self.test_features,
            #                                                            self.test_labels)

            # print('[INFO] NeuralNetworkClassificationEngine evaluate: Test Data Prediction Loss = {}, '
            #       'Test Data Prediction Accuracy = {}'.format(prediction_loss,
            #                                                   prediction_accuracy))

            # Model evaluation is complete!
            return True
        except Exception as e:
            print('[ERROR] NeuralNetworkClassificationEngine evaluate: Exception caught while evaluating the '
                  'prediction accuracy of the model - {}'.format(e))
            traceback.print_tb(e.__traceback__)
            return False

    # Make a prediction using the trained model for the given feature vector
    def make_a_prediction(self):
        # Choose a random sample from the test features collection
        # sample_index = random.sample(range(len(self.training_features),
        #                                    len(self.training_features) + len(self.test_features)),
        #                              1)[0]

        # Hard-coding the sample index to 11516 which has a classification of "down" because positive classifications...
        # ...are influenced by a lot of "normal" feature values while negative classifications are influenced by...
        # ...a very small set of "abnormal" feature values. This narrows down our focus.
        sample_index = 11516
        feature_vector = self.test_features.loc[[sample_index]]
        # Return the instance for analysis
        return sample_index, feature_vector, self.model.predict(feature_vector)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb, task_id):
        print('[INFO] NeuralNetworkClassificationEngine Termination: Tearing things down...')
        ClassificationTask.__exit__(self,
                                    exc_type,
                                    exc_val,
                                    exc_tb,
                                    self.TASK_ID)


# This class describes the procedure to provide intuitive, interpretable explanations for the predictions made by...
# ...Black Box Deep Learning Models for binary classification tasks.
# This rationale engine is model agnostic.
class LinkFailureCausationAnalyzer(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] LinkFailureCausationAnalyzer Initialization: Bringing things up...')
        # The status of this engine
        self.status = False
        # The dimensionality of the locally interpretable model used in the rationale engine
        self.dimensionality = SUPPORTED_DIMENSIONALITY
        # The width parameter of the exponential kernel function
        self.kernel_width = 0.2
        # Instance definition
        # $\vec{x} \in \mathcal{X}\ where\ \mathcal{X} \equiv \mathbb{R}^{K}$ is the feature vector
        # $y \in \mathcal{Y}\ where\ \mathcal{Y} \equiv \{0,\ 1\} is the output classifier label
        # RecordClasses are mutable alternatives to namedtuples
        self.instance = recordclass('instance',
                                    ['features',
                                     'label',
                                     'weight'])
        # A named-tuple for individual models
        self.model_results = namedtuple('model_results',
                                        ['features',
                                         'loss',
                                         'parameters'])
        # The number of features to be included in the interpretable model = \kappa
        self.interpretable_features_count = self.dimensionality
        # The number of perturbed instances sampled from the instance under analysis = N
        self.perturbed_samples_count = 10000
        # The regularization constraint = \alpha
        self.regularization_constraint = 150
        # The <classifier_id, classifier> pairs under prediction-rationale analysis
        self.classifiers_under_analysis = task_repository.items()
        # A collection for successfully built, compiled, and trained classifiers
        self.competent_classifiers = []
        # The prediction instance under rationale analysis - definition
        self.instance_under_analysis = self.instance(features=None,
                                                     label=None,
                                                     weight=0.0,
                                                     )
        # The stripped down perturbed samples collection
        self.baremetal_perturbed_samples_collection = []
        # The polyhedra constituting the feasible set of the optimization problem
        self.line_segments = [[(0, -self.regularization_constraint), (self.regularization_constraint, 0)],
                              [(0, self.regularization_constraint), (self.regularization_constraint, 0)],
                              [(0, self.regularization_constraint), (-self.regularization_constraint, 0)],
                              [(0, -self.regularization_constraint), (-self.regularization_constraint, 0)]]
        # The optimizer - ProjectionGradientDescentOptimizer
        self.optimizer = ProjectionGradientDescent(self.dimensionality,
                                                   self.regularization_constraint,
                                                   self.line_segments,
                                                   self)
        # The initial steps have been completed. The core logic follows.
        self.status = True
        # The optimizer is setup correctly
        if self.optimizer.status:
            # Build, Train, and Evaluate the global prediction accuracy of the classifiers in the repository
            for classifier_id, classifier in self.classifiers_under_analysis:
                print('[DEBUG] LinkFailureCausationAnalyzer Initialization: Analyzing the predictions of {} '
                      'which is a {}'.format(classifier_id,
                                             classifier.__class__.__name__))
                classifier_status = \
                    classifier.build_model() and classifier.train_model() and classifier.evaluate_model()
                # The classifier is setup correctly
                if classifier_status:
                    self.competent_classifiers.append(classifier)
                    # Start the explanation sequence
                    explanation = self.get_interpretable_explanation(classifier)
                    if explanation is None:
                        print(
                            '[ERROR] LinkFailureCausationAnalyzer Explanation: Something went wrong while developing '
                            'the interpretation. Please refer to the earlier logs for more information '
                            'on what went wrong!')
                        # Additional enforcement
                        classifier_status = False
                        self.status = False
                    else:
                        print('[INFO] LinkFailureCausationAnalyzer Explanation: The locally interpretable explanation '
                              'for the classifier [{}] with ID [{}] is described by [{}] whose evaluated loss during '
                              'local curve fitting is [{}] and whose respective associated weights '
                              'are [{}].'.format(classifier.__class__.__name__,
                                                 classifier_id,
                                                 explanation.features,
                                                 explanation.loss,
                                                 explanation.parameters))
                else:
                    self.status = False
                print('[INFO] LinkFailureCausationAnalyzer Initialization: '
                      'Classifier tagging status - [{}]'.format(classifier_status))
            print('[INFO] LinkFailureCausationAnalyzer Initialization: '
                  'All registered classifiers have been tagged and their prediction rationales have been explained. '
                  'Final status: [{}]'.format(self.status))
        else:
            print('[ERROR] LinkFailureCausationAnalyzer Initialization: Failure during initialization. '
                  'Please refer to the earlier logs for more information on what went wrong!')

    # Get weights using the exponential family of kernels based on a cosine similarity distance metric
    def get_weights(self, sample_instance, perturbed_instance):
        return utility.exponential_kernel_coefficient(sample_instance.features,
                                                      perturbed_instance.features,
                                                      self.kernel_width)

    # Get a locally interpretable explanation for a prediction
    def get_interpretable_explanation(self, classifier):
        features_under_analysis = list(classifier.features.columns.values)
        print('[INFO] LinkFailureCausationAnalyzer get_interpretable_explanation: The features under sensitivity '
              'analysis are: {}'.format(features_under_analysis))
        # Explain a prediction
        # Make a prediction using the built, compiled, and trained classifier
        sample_index, features, label = classifier.make_a_prediction()
        self.instance_under_analysis.features = features
        self.instance_under_analysis.label = label
        # The dot product will be 1 -> cos 0 = 1 -> perfect cosine similarity (1) -> Obviously!
        self.instance_under_analysis.weight = self.get_weights(self.instance_under_analysis,
                                                               self.instance_under_analysis)
        # It is a contract between the data processing entity and this engine that the columns in the dataset be...
        # structured in the standard way, i.e. having all the features to the left of the label in the dataframe
        print('[INFO] LinkFailureCausationAnalyzer get_interpretable_explanation: Sample '
              'instance under rationale analysis - '
              '\nFeatures = \n{} and '
              '\nTrue Label = {}'.format(tabulate(classifier.dataframe.loc[[sample_index]][
                                                      [k for k in classifier.dataframe.columns[:-1]]],
                                                  headers='keys',
                                                  tablefmt='psql'
                                                  ),
                                         classifier.dataframe.loc[[sample_index]][
                                             [classifier.dataframe.columns[-1]]].values[0,
                                                                                        0]
                                         )
              )
        print(
            '[INFO] LinkFailureCausationAnalyzer get_interpretable_explanation: Sample normalized prediction instance '
            'under rationale analysis after redundancy removal - '
            '\nFeatures = \n{} and '
            '\nPredicted Label = {}'.format(tabulate(features,
                                                     headers='keys',
                                                     tablefmt='psql'),
                                            (lambda: 0,
                                             # Conversion to a list is done to avoid the numpy.bool DeprecationWarning
                                             lambda: 1)[
                                                label.tolist()[0][0] > 0.5]()
                                            )
        )
        # All possible combinations of the features under analysis, n=#global_features, r=#local_interpretable_features
        all_possible_feature_family_combinations = [k for k in itertools.combinations(features_under_analysis,
                                                                                      self.interpretable_features_count
                                                                                      )]
        # A collection to house the loss and parameters of each individual local fit
        model_results_collection = []
        model_analysis_counter = 0.0
        for feature_family_tuple in all_possible_feature_family_combinations:
            utility.sys_progress(model_analysis_counter,
                                 len(all_possible_feature_family_combinations),
                                 bar_length=20,
                                 log_level='DEBUG',
                                 entity=self.__class__.__name__,
                                 routine='get_interpretable_explanation',
                                 key='Analyzing locally interpretable linear models: '
                                     'Starting model analysis for {}'.format(feature_family_tuple))
            # Purge the collection after analysis
            self.baremetal_perturbed_samples_collection.clear()
            # Creating <#perturbed_instances> perturbed samples for vicinity-based model fitting
            for i in range(self.perturbed_samples_count):
                # The initial instance before perturbation analysis and locally interpretable model fitting
                perturbed_sample = self.instance(features=features,
                                                 label=None,
                                                 weight=0.0)
                # The perturbed sample stripped of unchanged components from the original sample
                baremetal_perturbed_sample = self.instance(features=[],
                                                           label=None,
                                                           weight=0.0)
                for feature_family in feature_family_tuple:
                    # Get the values array and the statistics for the sampled family
                    family_values, family_mean, family_std = classifier.data_processor_memory[feature_family]
                    # Standard Normalization technique 1 - Sample from the vocabulary
                    # Standard Normalization technique 2 - Subtract the family mean from the value
                    # Standard Normalization technique 3 - Divide by the standard deviation of the family
                    if family_std != 0.0:
                        perturbed_value = (((random.sample(list(family_values.values()),
                                                           1)[0]) - family_mean) / family_std)
                    else:
                        perturbed_value = random.sample(list(family_values.values()), 1)[0]
                    perturbed_sample.features[feature_family] = perturbed_value
                    baremetal_perturbed_sample.features.append(perturbed_value)
                # The target
                perturbed_sample.label = (lambda: 0,
                                          lambda: 1)[
                    # Conversion to a list is done to avoid the numpy.bool DeprecationWarning
                    classifier.model.predict(perturbed_sample.features).tolist()[0][0] > 0.5]()
                # The weight
                perturbed_sample.weight = self.get_weights(self.instance_under_analysis,
                                                           perturbed_sample)
                # Populate the baremetal sample with the label and the weight
                baremetal_perturbed_sample.label = perturbed_sample.label
                baremetal_perturbed_sample.weight = perturbed_sample.weight
                self.baremetal_perturbed_samples_collection.append(baremetal_perturbed_sample)
            model_output = self.optimizer.optimize(None)
            if model_output[0] is False:
                print(
                    '[ERROR] LinkFailureCausationAnalyzer get_interpretable_explanation: Something went wrong during '
                    'optimization. Please refer to the earlier logs for more information on what went wrong!')
                return None
            model_results_collection.append(self.model_results(features=feature_family_tuple,
                                                               loss=model_output[1],
                                                               parameters=model_output[2]))
            model_analysis_counter += 1.0
            utility.sys_progress(model_analysis_counter,
                                 len(all_possible_feature_family_combinations),
                                 bar_length=20,
                                 log_level='DEBUG',
                                 entity=self.__class__.__name__,
                                 routine='get_interpretable_explanation',
                                 key='Analyzing locally interpretable linear models: '
                                     'Completed model analysis for {}'.format(feature_family_tuple))
        print('[INFO] LinkFailureCausationAnalyzer get_interpretable_explanation: '
              'Completed locally interpretable linear model analyses for all possible '
              'feature family tuples of length {}. '
              'Finding the best explanation among these analyzed models...'.format(self.interpretable_features_count))
        best_model = min(model_results_collection,
                         key=lambda x: x.loss)
        return best_model

    # Evaluate the cost function / loss at the current point in $\mathbb{R}^{\kappa}$
    def evaluate_loss(self, parameter_vector):
        loss = 0.0
        for baremetal_perturbed_sample in self.baremetal_perturbed_samples_collection:
            loss = loss + (baremetal_perturbed_sample.weight * (
                    (baremetal_perturbed_sample.label - sum(
                        i * j for i, j in zip(parameter_vector, baremetal_perturbed_sample.features))) ** 2))
        return loss / self.perturbed_samples_count

    # Evaluate the gradient of the cost function or loss at the current point $\mathbb{R}^{\kappa}$
    def evaluate_gradients(self, parameter_vector, dimensionality):
        if dimensionality is not self.dimensionality:
            raise NotImplementedError('[ERROR] LinkFailureCausationAnalyzer evaluate_gradients: '
                                      'Gradient Evaluation for models with dimension not equal to [{}] is not '
                                      'currently supported. Please check back later!'.format(self.dimensionality))
        gradient = [k - k for k in range(len(parameter_vector))]
        for baremetal_perturbed_sample in self.baremetal_perturbed_samples_collection:
            inner_term = numpy.multiply(baremetal_perturbed_sample.features,
                                        (sum(i * j for i, j in zip(parameter_vector,
                                                                   baremetal_perturbed_sample.features)) -
                                         baremetal_perturbed_sample.label))
            gradient = gradient + inner_term
        return (2 / self.perturbed_samples_count) * gradient

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] LinkFailureCausationAnalyzer Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] LinkFailureCausationAnalyzer Trigger: Starting system assessment!')
    # This is one example of a classification engine suitable for use in this environment
    nnClassifier = NeuralNetworkClassificationEngine()
    # NeuralNetworkClassificationEngine SUCCESS
    if nnClassifier.status:
        # Analyze the rationale behind the predictions made by this engine
        causationAnalyzer = LinkFailureCausationAnalyzer()
        # LinkFailureCausationAnalyzer SUCCESS
        if causationAnalyzer.status:
            print('[INFO] LinkFailureCausationAnalyzer Trigger: Prediction Rationale Analysis is Successful!')
        # LinkFailureCausationAnalyzer FAILURE
        else:
            print('ERROR] LinkFailureCausationAnalyzer Trigger: Prediction Rationale Analysis Failed! '
                  'Something went wrong during the initialization of {}'.format(LinkFailureCausationAnalyzer.__name__))
        print('[INFO] LinkFailureCausationAnalyzer Trigger: System assessment has been completed!')
    # NeuralNetworkClassificationEngine FAILURE
    else:
        print('[ERROR] LinkFailureCausationAnalyzer Trigger: Something went wrong during the initialization '
              'of {}'.format(NeuralNetworkClassificationEngine.__name__))
