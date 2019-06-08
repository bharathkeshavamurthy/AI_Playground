# This entity describes the design of RandomForests and how they can be used to determine feature importance
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems, Inc
# Copyright (c) 2019. All Rights Reserved.

# The imports
import math
import pandas
import warnings
from tabulate import tabulate
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')


# This class encapsulates the design of Random Forests for a classification tasks and then, leverage the capabilities...
# ...of these Random Forests to determine feature importances
class RandomForests(object):
    # Use ${TRAINING_SPLIT} * 100% of the data for training and the remaining for testing and/or validation
    TRAINING_SPLIT = 0.8

    # The initialization sequence
    def __init__(self):
        print('[INFO] RandomForests Initialization: Bringing things up...')
        # Read the dataset
        self.dataframe = pandas.read_csv('datasets/bank.csv')
        # Rename the columns for aesthetics
        self.dataframe.columns = ['Age', 'Job', 'Marital-Status', 'Education', 'Default', 'Balance', 'Housing',
                                  'Loan', 'Contact', 'Day', 'Month', 'Duration', 'Campaign', 'PayDays', 'Previous',
                                  'Paid-Outcome', 'Class']
        # The memory of the data processor
        self.data_processor_memory = {}
        # Pretty Print the Data for debugging and/or analysis
        print('[INFO] RandomForests Initialization: Sample Training Examples - \n{}'.format(
            tabulate(self.dataframe,
                     headers='keys',
                     tablefmt='psql')))
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
        # Initialization and Data Processing Done

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

    @staticmethod
    # Build the RandomForestClassifier
    def build_model():
        return RandomForestClassifier()

    # Fit the training data to the model, evaluate the model against the test data, and...
    # ...determine the feature importances
    def evaluate_feature_importances(self, model):
        model.fit(self.training_features,
                  self.training_labels)
        training_predictions = model.predict(self.training_features)
        test_predictions = model.predict(self.test_features)
        print('[INFO] RandomForests evaluate_feature_importances: The trained classifier is - {}'.format(model))
        print('[INFO] RandomForests evaluate_feature_importances: The Training Accuracy Score - {}'.format(
            accuracy_score(self.training_labels,
                           training_predictions)
        ))
        print('[INFO] RandomForests evaluate_feature_importances: The Test Accuracy Score - {}'.format(
            accuracy_score(self.test_labels,
                           test_predictions)
        ))
        print('[INFO] RandomForests evaluate_feature_importances: The Confusion Matrix for the predictions on the test '
              'data is - \n{}'.format(confusion_matrix(self.test_labels,
                                                       test_predictions)))
        model.score(self.test_features, self.test_labels)
        return pandas.DataFrame(model.feature_importances_,
                                index=self.training_features.columns,
                                columns=['importance']).sort_values('importance', ascending=False)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] RandomForests Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] RandomForests main: Starting system assessment!')
    randomForests = RandomForests()
    classifier = randomForests.build_model()
    print('[INFO] RandomForests main: Feature Importances for this Classification problem are \n{}'.format(
        tabulate(randomForests.evaluate_feature_importances(classifier),
                 headers='keys',
                 tablefmt='psql'
                 )
    ))
