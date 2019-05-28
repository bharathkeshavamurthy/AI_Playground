# This entity describes training a Gradient Boosting model using decision trees in TensorFlow leveraging the...
# ...capabilities of the tensorflow.estimators API
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems
# Copyright (c) 2019. All Rights Reserved.

import pandas
import tensorflow
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

tensorflow.enable_eager_execution()
tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
tensorflow.set_random_seed(123)


# This class uses BoostedTrees to predict the survival chances of passengers aboard the HMS Titanic
class BoostedTrees(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] BoostedTrees Initialization: Bringing things up...')
        # The training dataset
        self.training_dataframe = pandas.read_csv('https://storage.googleapis.com/tfbt/titanic_train.csv')[:200]
        self.y_training = self.training_dataframe.pop('survived')
        # The evaluation dataset
        self.evaluation_dataframe = pandas.read_csv('https://storage.googleapis.com/tfbt/titanic_eval.csv')[:10]
        self.y_evaluation = self.evaluation_dataframe.pop('survived')

    # Categorical column generation
    @staticmethod
    def one_hot_encode(feature_name, vocabulary):
        return tensorflow.feature_column.indicator_column(
            tensorflow.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    # Extract and categorize features into columns for the model
    def extract_and_categorize_features(self):
        categorical_columns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
        numerical_columns = ['age', 'fare']
        feature_names = []
        for feature_name in categorical_columns:
            vocabulary = self.training_dataframe[feature_name].unique()
            feature_names.append(self.one_hot_encode(feature_name, vocabulary))
        for feature_name in numerical_columns:
            feature_names.append(tensorflow.feature_column.numeric_column(feature_name, dtype=tensorflow.float32))
        return feature_names

    # Input functions are required to train the Estimators in TensorFlow
    def create_input_function(self, x, y, num_epochs=None, shuffle=True):
        def input_fn():
            dataset = tensorflow.data.Dataset.from_tensor_slices((dict(x), y))
            if shuffle:
                dataset = dataset.shuffle(len(self.y_training))
            dataset = dataset.repeat(num_epochs)
            dataset = dataset.batch(len(self.y_training))
            return dataset

        return input_fn

    # Build, Train, and Evaluate the model
    def train(self):
        training_input_function = self.create_input_function(self.training_dataframe, self.y_training)
        evaluation_input_function = self.create_input_function(self.evaluation_dataframe, self.y_evaluation)
        boosted_trees_model = tensorflow.estimator.BoostedTreesClassifier(self.extract_and_categorize_features(),
                                                                          n_batches_per_layer=1)
        boosted_trees_model.train(training_input_function, max_steps=10)
        boosted_trees_results = boosted_trees_model.evaluate(evaluation_input_function)
        print('[INFO] BoostedTrees train(): TensorFlow BoostedTrees API | Accuracy = {}'.format(
            boosted_trees_results['accuracy']))
        predicted_dicts = list(boosted_trees_model.predict(evaluation_input_function))
        probabilities = [prediction['probabilities'][1] for prediction in predicted_dicts]
        false_positive_rate, true_positive_rate = roc_curve(self.y_evaluation, probabilities)
        plt.figure()
        plt.plot(false_positive_rate, true_positive_rate)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for the Boosted Trees Model')

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] BoostedTrees Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] BoostedTrees Trigger: Starting system assessment!')
    boostedTrees = BoostedTrees()
    boostedTrees.train()

# Local Interpretability:
# Additionally, we can analyze the contributions of individual features to a certain model prediction using...
# ...Directional Feature Contributions (DFCs) by leveraging the experimental_predict_with_explanations(input_fn) method.

# Global Interpretability:
# We can also visualize feature contributions on a global scale.
# Gain based Feature Importances - Measure loss change when splitting on a particular feature
# Use experimental_feature_importances for GFI
# Permutations based Feature Importances -
# Evaluate model performance on the evaluation dataset by shuffling each feature one-by-one and...
# ...attributing the model change to the shuffled feature

# Note: Gradient Boosted Decision Trees fit the underlying data much better - the higher the number of trees, ...
# ...the better the fit. Use tf.estimator.BoostedTreesRegressor(fc, n_batches_per_layer=1, n_trees=n_trees).
