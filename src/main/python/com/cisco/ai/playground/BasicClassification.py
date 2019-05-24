# This is a simple classification algorithm using TensorFlow and the high-level Keras API
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems
# Copyright (c) 2019. All Rights Reserved.

import numpy
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt


# This class encapsulates a basic classification algorithm using TensorFlow and the high-level Keras API
class BasicClassification(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] BasicClassification Initialization: Bringing things up...')
        # The data sets - training and test data-sets
        (self.training_images, self.training_labels), (self.test_images, self.test_labels) = keras.datasets. \
            fashion_mnist.load_data()
        # The classifiers - Output of the final layer of the neural net
        self.class_names = ['Tees', 'Pants', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                            'Ankle boot']
        # Preprocessing - Scaling the pixel values from 0-255 to 0-1
        self.training_images = self.training_images / 255.0
        self.test_images = self.test_images / 255.0
        # The model
        self.model = None

    # Build the model, compile it, and train it
    def build(self):
        print('[INFO] BasicClassification build: Build, Compile, and Train the model...')
        # Building the network
        self.model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                                       keras.layers.Dense(128, activation=tensorflow.nn.relu),
                                       keras.layers.Dense(10, activation=tensorflow.nn.softmax)])
        # Compile the model with 'accuracy' as the metric of interest, cost function, and the update procedure
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Train the model using the training data
        self.model.fit(self.training_images, self.training_labels, epochs=10)

    # Make predictions for the test data-set and evaluate accuracy
    def evaluate(self):
        # Evaluate how the model performs on the test data-set
        prediction_loss, prediction_accuracy = self.model.evaluate(self.test_images, self.test_labels)
        print('[INFO] BasicClassification evaluate: Prediction Loss = {}, Prediction Accuracy = {}'.format(
            prediction_loss, prediction_accuracy))

    # Plot the image (blue x-label if the prediction is right and a red x-label if the prediction is wrong)
    # Also include the predicted label and the true label in the x-label
    def plot_image(self, i, predicted_labels, true_labels, image):
        predicted_label_probability_array, true_label, img = predicted_labels[i], true_labels[i], image[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        predicted_label = numpy.argmax(predicted_label_probability_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel('{} {:2.0f}% ({}) '.format(self.class_names[predicted_label],
                                              numpy.max(predicted_label_probability_array) * 100,
                                              self.class_names[true_label], color=color))

    # Visualize a bar graph for each image based on the output probabilities corresponding to each class
    @staticmethod
    def plot_value_array(i, predicted_labels, true_labels):
        predicted_label_probabilities_array, true_label = predicted_labels[i], true_labels[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        # Bar graph
        bar_graph = plt.bar(range(10), predicted_label_probabilities_array, color='#777777')
        plt.ylim([0, 1])
        predicted_label = numpy.argmax(predicted_label_probabilities_array)
        bar_graph[predicted_label].set_color('red')
        bar_graph[true_label].set_color('blue')

    # With the model evaluated, use this method to visualize the prediction accuracy for the classes
    def visualize(self):
        predictions = self.model.predict(self.test_images)
        number_of_rows = 5
        number_of_columns = 3
        number_of_images = number_of_rows * number_of_columns
        plt.figure(figsize=(2 * 2 * number_of_columns, 2 * number_of_rows))
        for i in range(number_of_images):
            plt.subplot(number_of_rows, 2 * number_of_columns, 2 * i + 1)
            self.plot_image(i, predictions, self.test_labels, self.test_images)
            plt.subplot(number_of_rows, 2 * number_of_columns, 2 * i + 2)
            self.plot_value_array(i, predictions, self.test_labels)
        plt.show()

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] BasicClassification Termination: Tearing things down...')


# The run trigger
if __name__ == '__main__':
    print('[INFO] BasicClassification Trigger: Starting system assessment...')
    basic_classifier = BasicClassification()
    basic_classifier.build()
    basic_classifier.evaluate()
    basic_classifier.visualize()
