# |Bleeding-Edge Productions|
# This entity explores the capabilities of Model Class Reliance (MCR) in determining the importance of specific...
# ...features across several variants of classification engines, i.e. models.
# This entity encapsulates a PoC for understanding the capabilities of the MCR regime when applied to a simple binary...
# ...classification task.
# The classification task involves predicting whether individuals modelled as a vector of features are financially...
# ...capable of owning a house in the Bay Area.
# The PoC can be extended to a prototype for Link Failure Prediction and Correlation-Causation Analysis.
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems, Inc.
# Copyright (c) 2019. All Rights Reserved.

# The imports
import math
import pandas
import warnings
import traceback
import tensorflow
from collections import namedtuple
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# This is a bad hack.
warnings.filterwarnings('ignore')

# The global engine repository
repository = {}


# The register routine
def register(_id, _model):
    repository[_id] = _model
    print('[INFO] GlobalView register: Added {} to the global engine repository!'.format(_id))


# The unregister routine
def unregister(_id):
    try:
        repository.pop(_id)
        print('[INFO] GlobalView unregister: Removed {} from the global engine repository!'.format(_id))
    except KeyError as e:
        print('[ERROR] GlobalView unregister: The key {} does not exist in the global engine repository!'.format(_id))
        traceback.print_tb(e.__traceback__)


# RF_ENGINE: RandomForests Based Classification Engine
# This class encapsulates the design of Random Forests for a classification tasks and then, leverage the capabilities...
# ...of these Random Forests to determine feature importances
class RandomForestsClassificationEngine(object):
    # The engine identifier
    ENGINE_IDENTIFIER = 'RF_ENGINE'

    # Use ${TRAINING_SPLIT} * 100% of the data for training and the remaining for testing and/or validation
    TRAINING_SPLIT = 0.8

    # Process the data before feeding it into the Classifier
    def process_data(self, data, family):
        if type(data) is str:
            if family not in self.data_processor_memory.keys():
                vocabulary = sorted(set(self.dataframe[family]))
                word_to_index_map = {u: i + 1 for i, u in enumerate(vocabulary)}
                self.data_processor_memory[family] = word_to_index_map
            word_to_index_map = self.data_processor_memory[family]
            return word_to_index_map[data]
        else:
            return data

    # The initialization sequence
    def __init__(self, data_file):
        print('[INFO] RandomForestsClassificationEngine Initialization: Bringing things up...')
        # Read the dataset
        self.dataframe = pandas.read_csv(data_file)
        # Rename the columns for aesthetics
        self.dataframe.columns = ['Age', 'Job', 'Marital-Status', 'Education', 'Default', 'Balance', 'Housing',
                                  'Loan', 'Contact', 'Day', 'Month', 'Duration', 'Campaign', 'PayDays', 'Previous',
                                  'Paid-Outcome', 'Class']
        # The memory of the data processor
        self.data_processor_memory = {}
        # The complete dataset
        features, labels = self.dataframe[self.dataframe.columns[:-1]], self.dataframe[self.dataframe.columns[-1]]
        # The feature iteration counter
        for feature_column in features.columns:
            for i in range(0, len(features[feature_column])):
                features[feature_column][i] = self.process_data(features[feature_column][i],
                                                                feature_column)
        # The label iteration counter
        for j in range(0, len(labels)):
            labels[j] = self.process_data(labels[j], 'Class')
        split = math.floor(len(features) * self.TRAINING_SPLIT)
        # The training data
        self.training_features, self.training_labels = features[:split], labels[:split]
        # Modification because the model was throwing unknown y_type error
        self.training_labels = [k for k in self.training_labels]
        # The test data
        self.test_features, self.test_labels = features[split:], labels[split:]
        # Modification because the model was throwing unknown y_type error
        self.test_labels = [k for k in self.test_labels]
        # The initialization and the data processing is complete
        # Register with the Global Engine Repository
        register(self.ENGINE_IDENTIFIER, self)

    @staticmethod
    # Build the RandomForestClassifier
    def build_model():
        return RandomForestClassifier()

    # Train the model - fit the model to the training data
    def train_model(self, model):
        model.fit(self.training_features, self.training_labels)
        return model

    # Evaluate the model against the test data
    def evaluate_model(self, model):
        training_predictions = model.predict(self.training_features)
        test_predictions = model.predict(self.test_features)
        print('[INFO] RandomForestsClassificationEngine evaluate_feature_importances: The trained classifier '
              'is - {}'.format(model))
        print('[INFO] RandomForestsClassificationEngine evaluate_feature_importances: The Training Accuracy '
              'Score - {}'.format(accuracy_score(self.training_labels, training_predictions)))
        print('[INFO] RandomForestsClassificationEngine evaluate_feature_importances: The Test Accuracy '
              'Score - {}'.format(accuracy_score(self.test_labels, test_predictions)))
        print('[INFO] RandomForestsClassificationEngine evaluate_feature_importances: The Confusion Matrix for the '
              'predictions on the test data is - \n{}'.format(confusion_matrix(self.test_labels, test_predictions)))

    # After training the model and evaluating it using the test data, evaluate the feature importances
    def evaluate_feature_importances(self, model):
        return pandas.DataFrame(model.feature_importances_,
                                index=self.training_features.columns,
                                columns=['importance']).sort_values('importance', ascending=False)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] RandomForestsClassificationEngine Termination: Tearing things down...')
        # Unregister from the global engine repository
        unregister(self.ENGINE_IDENTIFIER)


# ENGINE_2: Neural Network based Classification Engine
# This class employs fully-connected neural networks to perform the classification task.
# This classification engine serves as a model under MCR analysis.
class NeuralNetworkClassificationEngine(object):
    # The engine identifier
    ENGINE_IDENTIFIER = 'ENGINE_2'

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
        # Categorical Data
        if type(data) is str:
            if family not in self.data_processor_memory.keys():
                vocabulary = sorted(set(self.dataframe[family]))
                # A word to integer mapping for categorical columns
                word_to_index_map = {u: i + 1 for i, u in enumerate(vocabulary)}
                self.data_processor_memory[family] = word_to_index_map
            word_to_index_map = self.data_processor_memory[family]
            return word_to_index_map[data]
        # Numerical data
        else:
            return data

    # The initialization sequence
    def __init__(self, data_file):
        print('[INFO] NeuralNetworkClassificationEngine Initialization: Bringing things up...')
        # The memory of the data processor
        self.data_processor_memory = {}
        # Read the dataset
        self.dataframe = pandas.read_csv(data_file)
        # Rename the columns for aesthetics
        self.dataframe.columns = ['Age', 'Job', 'Marital-Status', 'Education', 'Default', 'Balance', 'Housing', 'Loan',
                                  'Contact', 'Day', 'Month', 'Duration', 'Campaign', 'PayDays', 'Previous',
                                  'Paid-Outcome', 'Class']
        # The complete dataset
        features, labels = self.dataframe[self.dataframe.columns[:-1]], self.dataframe[self.dataframe.columns[-1]]
        # Processing the input features to make them compatible with the Classification Engine
        for feature_column in features.columns:
            for i in range(0, len(features[feature_column])):
                features[feature_column][i] = self.process_data(features[feature_column][i], feature_column)
        # Processing the output labels to make them compatible with the Classification Engine
        for j in range(0, len(labels)):
            labels[j] = self.process_data(labels[j], 'Class')
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
        # The initialization and the data processing is complete
        register(self.ENGINE_IDENTIFIER, self)

    # Build the Neural Network model
    def build_model(self):
        # Construct a standard NN model with one hidden layer and ReLU & sigmoid non-linearities
        model = tensorflow.keras.Sequential([
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
        return model

    # Train the model on the training dataset
    def train_model(self, model):
        # I use an AdamOptimizer in the place of the simpler tensorflow.train.GradientDescentOptimizer()...
        # ...because the AdamOptimizer uses the moving average of parameters and this facilitates...
        # ...faster convergence by settling on a larger effective step-size.
        optimizer = tensorflow.train.AdamOptimizer()
        for epoch in range(self.NUMBER_OF_TRAINING_EPOCHS):
            for training_example in self.training_data:
                with tensorflow.GradientTape() as gradient_tape:
                    predicted_label = model(training_example.features)
                    cost = tensorflow.keras.losses.sparse_categorical_crossentropy(training_example.label,
                                                                                   predicted_label)
                gradients = gradient_tape.gradient(cost, model.trainable_variables)
                optimizer.apply_gradients(gradients, model.trainable_variables)
            print('[DEBUG] NeuralNetworkClassificationEngine train_model: Epoch {} Cost {}'.format(epoch + 1, cost))
        return model

    # After training the model, evaluate the model against the test data
    def evaluate_model(self, model):
        prediction_loss, prediction_accuracy = model.evaluate(self.test_features, self.test_labels)
        print('[INFO] NeuralNetworkClassificationEngine evaluate: Prediction Loss = {}, '
              'Prediction Accuracy = {}'.format(prediction_loss, prediction_accuracy))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] NeuralNetworkClassificationEngine Termination: Tearing things down...')
        # Unregister from the global engine repository
        unregister(self.ENGINE_IDENTIFIER)


# This class describes a potential application of Model Class Reliance (MCR) for classification tasks.
class ModelClassReliance(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] ModelClassReliance Initialization: Bringing things up...')

    # # Estimate the Model Class Reliance for the given feature
    # def estimate_model_class_reliance(self, feature):
    #     return NotImplementedError('This routine is yet to be implemented. Check back later!')
    #
    # # Estimate the correlation between two given features
    # def estimate_correlation_coefficient(self, x1, x2):
    #     return NotImplementedError('This routine is yet to be implemented. Check back later!')
    #
    # # Assuming the model has fully exhausted the content in x2, I wish to estimate the model reliance on x1
    # # Project the x2 vector onto x1 to find the
    # def project(self, x1, x2):
    #     return NotImplementedError('This routine is yet to be implemented. Check back later!')
    #
    # # Obtain the orthogonal component of the
    # def get_orthogonal_component(self, base_vector, operand_vector):
    #     return NotImplementedError('This routine is yet to be implemented. Check back later!')

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] ModelClassReliance Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    dataset_file_path = 'datasets/housing.csv'
    nnModel = NeuralNetworkClassificationEngine(dataset_file_path)
    rfModel = RandomForestsClassificationEngine(dataset_file_path)
    modelClassRelianceAnalyzer = ModelClassReliance()
