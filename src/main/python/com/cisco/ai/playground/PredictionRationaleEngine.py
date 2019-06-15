# |Bleeding-Edge Productions|
# This entity details a technique to provide an explanation for the prediction made by a NN-based classification model.
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems, Inc.
# Copyright (c) 2019. All Rights Reserved.

# The imports
import math
import numpy
import pandas
import random
import traceback
import tensorflow
from tabulate import tabulate
from collections import namedtuple


# The Projection Gradient Descent entity for Constrained Convex Optimization
class ProjectionGradientDescent(object):

    # The initialization sequence
    def __init__(self, _dimensionality, _intercept_constraint):
        # The initialization status flag
        self.status = False
        print('[INFO] ProjectionGradientDescent Initialization: Bringing things up...')
        # Support for dimensionality is currently limited to two-dimensional feasible sets, i.e. \mathbb{R}^2
        # TODO: Projection Gradient Descent for N-dimensional feasible sets / domains (N >= 2)
        if _dimensionality != 2:
            print('[ERROR] ProjectionGradientDescent Initialization: Support for problems with dimensionality other '
                  'than 2 is not currently available. Please check back later!')
            return
        # The dimensionality of the problem
        self.dimensionality = _dimensionality
        # The intercept constraint in the given linear inequality constraint
        self.intercept_constraint = _intercept_constraint
        # The default step size during training
        self.default_step_size = 0.001
        # The line segments of the polyhedron which comprises the feasible set
        self.line_segments = [[(0, 0), (self.intercept_constraint, 0)],
                              [(0, 0), (0, self.intercept_constraint)],
                              [(self.intercept_constraint, 0), (0, self.intercept_constraint)]]
        # The iterations array which models the x-axis of the convergence plot
        self.iterations = []
        # The function values array which models the y-axis of the convergence plot
        self.function_values = []
        # The confidence bound for convergence
        self.confidence_bound = 10
        # The maximum number of iterations allowed during training
        self.max_iterations = 1e6

    # Vector Projection utility routine
    # Return the distance and the closest point w.r.t to that specific line segment
    def vector_projection(self, oob_point, line_segment):
        x1, y1 = line_segment[0]
        x2, y2 = line_segment[1]
        x, y = oob_point
        # Is the point outside the feasible set?
        if (x < 0) or (y < 0) or ((x + y) > self.intercept_constraint):
            dot_product = ((x - x1) * (x2 - x1)) + ((y - y1) * (y2 - y1))
            norm_square = ((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1))
            # Invalid base vector
            if norm_square == 0:
                print('[ERROR] ProjectionGradientDescent vector_projection: The norm square of the base vector is 0!')
                return False, None, None
            # Find the component of the operand vector along the base vector
            # This is a standard vector projection technique
            component = dot_product / norm_square
            # (x1, y1) is the closest point
            if component < 0:
                return True, math.sqrt(((x - x1) * (x - x1)) + ((y - y1) * (y - y1))), (x1, y1)
            # (x2, y2) is the closest point
            elif component > 1:
                return True, math.sqrt(((x - x2) * (x - x2)) + ((y - y2) * (y - y2))), (x2, y2)
            # Find the closest point using the projection component
            else:
                x_closest = x1 + (component * (x2 - x1))
                y_closest = y1 + (component * (y2 - y1))
                return True, math.sqrt(((x - x_closest) * (x - x_closest)) + ((y - y_closest) * (y - y_closest))), (
                    x_closest, y_closest)
        else:
            # No Projection
            return True, 0, oob_point

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
        return sum([abs(x) for x in vector])

    # A utility method
    # Calculate the L2 norm of the given vector
    @staticmethod
    def l2_norm(vector):
        return math.sqrt(sum([x ** 2 for x in vector]))

    # A utility method
    # Calculate the exponential kernel coefficient
    def exponential_kernel_coefficient(self, _vector_x, _vector_z, _width):
        dot_product = sum(i * j for i, j in zip(_vector_x, _vector_z))
        cosine_similarity = dot_product / (self.l2_norm(_vector_x) * self.l2_norm(_vector_z))
        return math.exp((cosine_similarity ** 2) / _width)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Utility Termination: Tearing things down...')


# The global utilities instance
utility = Utility()

# The global repository of tasks whose respective engines need to be analyzed for prediction rationale.
task_repository = {}


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
        task_repository.pop(_id)
        print('[INFO] GlobalView de-register: Successfully removed task with ID [{}] '
              'from the task repository'.format(_id))
    except KeyError:
        print('[ERROR] GlobalView de-register: The task ID [{}] does not exist in the task repository! Nothing to be '
              'done!'.format(_id))


# The parent classification task wrapper
class ClassificationTask(object):

    # The initialization sequence
    def __init__(self, _task_id, _features):
        print('[INFO] ClassificationTask Initialization: Bringing things up...')
        self.registration_status = register(_task_id, self)
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

    # Abstract contract for making a prediction for a random sample from the test dataset
    def make_a_prediction(self):
        pass

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb, _id):
        print('[INFO] ClassificationTask Termination: Tearing things down...')
        deregister(_id)


# This class employs fully-connected neural networks to perform the classification task.
class NeuralNetworkClassificationEngine(ClassificationTask):
    # The task id
    TASK_ID = 'NN_CLASSIFIER_1'

    # A data structure for easy, clean storage and access of data
    DATA = namedtuple('Data', ['features', 'label'])

    # Use ${TRAINING_SPLIT} * 100% of the data for training and the remaining for testing and/or validation
    TRAINING_SPLIT = 0.8

    # The number of neurons in the hidden layer of the NN model
    NUMBER_OF_HIDDEN_UNITS = 1024

    # The number of epochs to train the model
    NUMBER_OF_TRAINING_EPOCHS = 5000

    # Process the data before feeding it into the Classifier
    def process_data(self, data, family):
        # Vocabulary Creation
        vocabulary = sorted(set(self.dataframe[family]))
        self.feature_vocabulary_mapping[family] = vocabulary
        if family not in self.data_processor_memory.keys():
            # A word to integer mapping for categorical columns
            feature_index_map = (lambda: {u: i + 1 for i, u in enumerate(vocabulary)},
                                 lambda: {k: k for k in vocabulary})[isinstance(data, str) is False]()
            # Evaluate the mean and standard deviation of the integer mappings for normalization
            mean = numpy.mean(feature_index_map.values())
            standard_deviation = numpy.std(feature_index_map.values())
            self.data_processor_memory[family] = feature_index_map, mean, standard_deviation
        feature_index_map, mean, standard_deviation = self.data_processor_memory[family]
        return (feature_index_map[data] - mean) / standard_deviation

    # The initialization sequence
    def __init__(self):
        print('[INFO] NeuralNetworkClassificationEngine Initialization: Bringing things up...')
        # The path to the dataset file
        data_file = 'datasets/housing.csv'
        # The memory of the data processor
        self.data_processor_memory = {}
        # The feature vocabulary mapping
        self.feature_vocabulary_mapping = {}
        try:
            # Read the dataset
            self.dataframe = pandas.read_csv(data_file)
            # Rename the columns for aesthetics
            self.dataframe.columns = ['Age', 'Job', 'Marital-Status',
                                      'Education', 'Default', 'Balance',
                                      'Housing', 'Loan', 'Contact', 'Day', 'Month',
                                      'Duration', 'Campaign', 'PayDays', 'Previous', 'Paid-Outcome', 'Class']
            # The call to the parent
            ClassificationTask.__init__(self, self.TASK_ID, self.dataframe.columns)
            # The complete dataset
            features, labels = self.dataframe[self.dataframe.columns[:-1]], self.dataframe[self.dataframe.columns[-1]]
            # Processing the input features to make them compatible with the Classification Engine
            for feature_column in features.columns:
                for i in range(0, len(features[feature_column])):
                    features[feature_column][i] = self.process_data(features[feature_column][i],
                                                                    feature_column)
            # Processing the output labels to make them compatible with the Classification Engine
            for j in range(0, len(labels)):
                # It is a contract between the calling entity and this engine that the columns in the dataset be...
                # structured in the standard way, i.e. having all the features to the left of the label in the dataframe
                labels[j] = self.process_data(labels[j], self.dataframe.columns[-1])
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
            for k in range(len(self.test_features)):
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
            print('[INFO] NeuralNetworkClassificationEngine Initialization: The dataset is - \n{}'.format(
                tabulate(self.dataframe,
                         headers='keys',
                         tablefmt='psql')
            ))
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
            # Construct a standard NN model with one hidden layer and ReLU & sigmoid non-linearities
            self.model = tensorflow.keras.Sequential([
                # The input layer
                tensorflow.keras.layers.Dense(units=len(self.dataframe.columns) - 1,
                                              activation=tensorflow.nn.relu),
                # The hidden layer
                tensorflow.keras.layers.Dense(units=self.NUMBER_OF_HIDDEN_UNITS,
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
            optimizer = tensorflow.train.AdamOptimizer()
            for epoch in range(self.NUMBER_OF_TRAINING_EPOCHS):
                for training_example in self.training_data:
                    with tensorflow.GradientTape() as gradient_tape:
                        predicted_label = self.model(training_example.features)
                        cost = tensorflow.keras.losses.sparse_categorical_crossentropy(training_example.label,
                                                                                       predicted_label)
                    gradients = gradient_tape.gradient(cost, self.model.trainable_variables)
                    optimizer.apply_gradients(gradients, self.model.trainable_variables)
                print('[DEBUG] NeuralNetworkClassificationEngine train_model: Epoch {} Cost {}'.format(epoch + 1, cost))
            # Model training is complete!
            return True
        except Exception as e:
            print('[ERROR] NeuralNetworkClassificationEngine train_model: Exception caught while training '
                  'the classifier - {}'.format(e))
            traceback.print_tb(e.__traceback__)
            return False

    # After training the model, evaluate the model against the test data
    def evaluate_model(self):
        try:
            prediction_loss, prediction_accuracy = self.model.evaluate(self.test_features, self.test_labels)
            print('[INFO] NeuralNetworkClassificationEngine evaluate: Prediction Loss = {}, '
                  'Prediction Accuracy = {}'.format(prediction_loss, prediction_accuracy))
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
        feature_vector = self.test_features.loc[[random.sample(range(len(self.test_labels)))]]
        # Return the instance for analysis
        return feature_vector, self.model(feature_vector)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb, task_id):
        print('[INFO] NeuralNetworkClassificationEngine Termination: Tearing things down...')
        ClassificationTask.__exit__(self, exc_type, exc_val, exc_tb, self.TASK_ID)


# This class describes the procedure to provide intuitive, interpretable explanations for the predictions made by a...
# a Neural Network based binary classification model using standard ML tools and techniques.
# This rationale engine is model agnostic.
class PredictionRationaleEngine(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] PredictionRationaleEngine Initialization: Bringing things up...')
        # The width parameter of the exponential kernel function
        self.kernel_width = 1
        # Instance definition
        # $\vec{x} \in \mathcal{X}\ where\ \mathcal{X} \equiv \mathbb{R}^{n}$ is the feature vector
        # $y \in \mathcal{Y}\ where\ \mathcal{Y} \equiv \{0,\ 1\} is the output classifier label
        self.instance = namedtuple('instance', ['features', 'label'])
        # The number of features to be included in the interpretable model
        self.interpretable_features_count = 2
        # The number of perturbed instances sampled from the instance under analysis
        self.perturbed_samples_count = 1000
        # The regularization constraint
        self.regularization_constraint = 10
        # The classifiers under prediction-rationale analysis
        self.classifiers_under_analysis = task_repository.items()
        # Successfully built, compiled, and trained classifiers
        self.competent_classifiers = []
        # Build, Train, and Evaluate the global prediction accuracy of the classifiers in the repository
        for classifier_id, classifier in self.classifiers_under_analysis:
            classifier_status = classifier.build_model() and classifier.train_model() and classifier.evaluate_model()
            if classifier_status:
                self.competent_classifiers.append(classifier)
            print('[INFO] PredictionRationaleEngine Initialization: '
                  'Classifier tagging status - [{}]'.format(classifier_status))

    # Get weights using the exponential family of kernels based on a cosine similarity distance metric
    def get_weight(self, sample_instance, perturbed_instance):
        return utility.exponential_kernel_coefficient(sample_instance, perturbed_instance, self.kernel_width)

    # Get a locally interpretable explanation for a prediction
    def get_interpretable_explanation(self, classifier):
        features_under_analysis = classifier.dataframe.columns[:-1]
        # An array of perturbed samples
        perturbed_samples = []
        # Make a prediction using the built, compiled, and trained classifier
        features, label = classifier.make_a_prediction()
        print('[INFO] PredictionRationaleEngine get_interpretable_explanation: Sample prediction instance under '
              'rationale analysis - Features = {} and Predicted Label = {}'.format(features, label))
        # Creating <#perturbed_instances> perturbed samples for vicinity-based model fitting
        for i in range(self.perturbed_samples_count):
            # An empty instance
            perturbed_sample = self.instance(features=pandas.DataFrame(columns=classifier.dataframe.columns),
                                             label=None)
            sampled_features = []
            # Uniform sampling of <#interpretable_features>
            for k in range(self.interpretable_features_count):
                sampled_feature_family = random.sample(features_under_analysis)
                # Already sampled this family. Move on...
                if sampled_feature_family in sampled_features:
                    continue
                sampled_features.append(sampled_feature_family)
                # Get the values array and the statistics for the sampled family
                family_values, family_mean, family_std = classifier.data_processor_memory[sampled_feature_family]
                # Standard Normalization technique
                perturbed_sample.features[sampled_feature_family] = (random.sample(family_values) -
                                                                     family_mean) / family_std
            perturbed_samples.append(perturbed_sample)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] PredictionRationaleEngine Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] PredictionRationaleEngine Trigger: Starting system assessment!')
    # This is one example of a classification engine suitable for use in this environment
    nnClassifier = NeuralNetworkClassificationEngine()
    if nnClassifier.status:
        rationaleEngine = PredictionRationaleEngine()
    else:
        print('[ERROR] PredictionRationaleEngine Trigger: Something went wrong during the initialization '
              'of {}'.format(NeuralNetworkClassificationEngine.__name__))
