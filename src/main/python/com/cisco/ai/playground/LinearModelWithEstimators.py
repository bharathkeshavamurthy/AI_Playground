# This entity solves a simple binary classification problem using estimators available in TensorFlow
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems
# Copyright (c) 2019. All Rights Reserved.

# Estimators are TensorFlow's most scalable and production-oriented model-type.

import os
import sys
import tempfile
import functools
import tensorflow

models_path = os.path.join(os.getcwd(), 'models')
sys.path.append(models_path)

from models.official.wide_deep import census_dataset

tensorflow.enable_eager_execution()


# This class encapsulates a simple Logistic Regression problem
# 0 - a person has income less than $50,000
# 1 - a person has income more than $50,000
# A Binary Classification problem
# Features - Age, Occupation, Education, and Marital Status
class LinearModelWithEstimators(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] LinearModelWithEstimators Initialization: Bringing things up...')
        census_dataset.download('/tmp/census_data')
        self.training_file = '/tmp/census_data/adult.data'
        self.test_file = '/tmp/census_data/adult.test'

    # Extract and classify features - Continuous, Categorical, Bucketized, Crossed
    @staticmethod
    def extract_and_correlate_features():
        # A categorical column with a predefined vocabulary list
        relationship = tensorflow.feature_column.categorical_column_with_vocabulary_list('relationship',
                                                                                         ['Husband',
                                                                                          'Not-in-family',
                                                                                          'Wife',
                                                                                          'Own-child',
                                                                                          'Unmarried',
                                                                                          'Other-relative'])
        # A categorical column with a predefined vocabulary list
        education = tensorflow.feature_column.categorical_column_with_vocabulary_list('education',
                                                                                      ['Bachelors',
                                                                                       'HS-grad',
                                                                                       '11th',
                                                                                       'Masters',
                                                                                       '9th',
                                                                                       'Some college',
                                                                                       'Assoc-acdm', 'Assoc-voc',
                                                                                       '7th-8th', 'Doctorate',
                                                                                       'Prof-school',
                                                                                       '5th-6th', '10th', '1st-4th',
                                                                                       'Preschool', '12th'])
        # A categorical column with a predefined vocabulary list
        marital_status = tensorflow.feature_column.categorical_column_with_vocabulary_list('marital_status',
                                                                                           ['Married-civ-spouse',
                                                                                            'Divorced',
                                                                                            'Married-spouse-absent',
                                                                                            'Never-married',
                                                                                            'Separated',
                                                                                            'Married-AF-spouse',
                                                                                            'Widowed'])
        # A categorical column with a predefined vocabulary list
        work_class = tensorflow.feature_column.categorical_column_with_vocabulary_list('workclass',
                                                                                       ['Self-emp-not-inc',
                                                                                        'Private',
                                                                                        'State-gov',
                                                                                        'Federal-gov',
                                                                                        'Local-gov',
                                                                                        '?',
                                                                                        'Self-emp-inc',
                                                                                        'Without-pay',
                                                                                        'Never-worked'])
        # A categorical column with no predefined vocabulary list - use a hashbucket definition
        occupation = tensorflow.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)
        # A numeric column - continuous values
        age = tensorflow.feature_column.numeric_column('age')
        # Bucketized column - place values into bins for better data fits
        age_buckets = tensorflow.feature_column.bucketized_column(age,
                                                                  boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
        # Learning complex relationships using cross-correlation - crossed column
        education_x_occupation = tensorflow.feature_column.crossed_column(['education', 'occupation'],
                                                                          hash_bucket_size=1000)
        age_buckets_x_education_x_occupation = tensorflow.feature_column.crossed_column([age_buckets,
                                                                                         'education',
                                                                                         'occupation'],
                                                                                        hash_bucket_size=1000)
        return [education, marital_status, relationship, work_class, occupation, age_buckets] + \
               [education_x_occupation, age_buckets_x_education_x_occupation]

    # Build, Train, and Evaluate the model
    def train(self):
        # The training data input function - wraps the data into tensors with features in batches and labels in batches
        training_input_function = functools.partial(census_dataset.input_fn, self.training_file,
                                                    num_epochs=40, shuffle=True, batch_size=64)
        # The test data input function - wraps the data into tensors with features in batches and labels in batches
        test_input_function = functools.partial(census_dataset.input_fn, self.test_file,
                                                num_epochs=40, shuffle=True, batch_size=64)
        feature_columns = self.extract_and_correlate_features()
        # Baseline Model
        baseline_model = tensorflow.estimator.LinearClassifier(model_dir=tempfile.mkdtemp(),
                                                               feature_columns=feature_columns,
                                                               optimizer=tensorflow.train.FtrlOptimizer(
                                                                   learning_rate=0.1))
        baseline_model.train(training_input_function)
        baseline_results = baseline_model.evaluate(test_input_function)
        print('Evaluation Results for the Baseline Model')
        for key, value in baseline_results.items():
            print('%s: %0.2f' % (key, value))
        # L1 regularization
        l1_regularized_model = tensorflow.estimator.LinearClassifier(feature_columns=feature_columns,
                                                                     optimizer=tensorflow.train.FtrlOptimizer(
                                                                         learning_rate=0.1,
                                                                         l1_regularization_strength=10.0,
                                                                         l2_regularization_strength=0.0
                                                                     ))
        l1_regularized_model.train(training_input_function)
        l1_regularized_results = l1_regularized_model.evaluate(test_input_function)
        print('Evaluation Results for the L1 Regularized Model')
        for key, value in l1_regularized_results.items():
            print('%s: %0.2f' % (key, value))
        # L2 regularization
        l2_regularized_model = tensorflow.estimator.LinearClassifier(feature_columns=feature_columns,
                                                                     optimizer=tensorflow.train.FtrlOptimizer(
                                                                         learning_rate=0.1,
                                                                         l1_regularization_strength=0.0,
                                                                         l2_regularization_strength=10.0
                                                                     ))
        l2_regularized_model.train(training_input_function)
        l2_regularized_results = l1_regularized_model.evaluate(test_input_function)
        print('Evaluation Results for the L2 Regularized Model')
        for key, value in l2_regularized_results.items():
            print('%s: %0.2f' % (key, value))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] LinearModelWithEstimators Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] LinearModelWithEstimators Trigger: Starting system assessment!')
    linearModelWithEstimators = LinearModelWithEstimators()
    linearModelWithEstimators.train()
