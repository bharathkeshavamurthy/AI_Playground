# This entity encapsulates a binary text classifier using TensorFlow and the high-level Keras API
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems
# Copyright (c) 2019. All Rights Reserved.

import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt


# This class is responsible for classifying movie reviews as positive or negative
class BinaryTextClassification(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] BinaryTextClassification Initialization: Bringing things up...')
        self.data_set = keras.datasets.imdb
        (self.training_data, self.training_labels), (self.test_data, self.test_labels) = self.data_set.load_data(
            num_words=10000)
        self.word_index = self.data_set.get_word_index()
        # The model
        self.model = None
        # The training and validation history
        self.fit_history = {}
        # The first few indices are reserved
        self.word_index = {k: (v + 3) for k, v in self.word_index.items()}
        self.word_index['<PAD>'] = 0
        self.word_index['<START>'] = 1
        self.word_index['<UNKNOWN>'] = 2
        self.word_index['<UNUSED>'] = 3
        self.reversed_word_index = dict([(value, key) for (key, value) in self.word_index.items()])

    # Map Words to Integers
    def word_to_integer_mapping(self, text):
        return ' '.join([self.reversed_word_index.get(i, '?') for i in text])

    # Pre-process training data
    def preprocess(self):
        self.training_data = keras.preprocessing.sequence.pad_sequences(self.training_data,
                                                                        value=self.word_index['<PAD>'],
                                                                        padding='post',
                                                                        maxlen=256)
        self.test_data = keras.preprocessing.sequence.pad_sequences(self.test_data,
                                                                    value=self.word_index['<PAD>'],
                                                                    padding='post',
                                                                    maxlen=256)

    # Build, Compile, and Train the model
    def build(self):
        vocabulary_size = 10000
        self.model = keras.Sequential([keras.layers.Embedding(vocabulary_size, 16),
                                       keras.layers.GlobalAveragePooling1D(),
                                       keras.layers.Dense(16, activation=tensorflow.nn.relu),
                                       keras.layers.Dense(1, activation=tensorflow.nn.sigmoid)])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Extract a validation data subset from the original data set
        validation_data = self.training_data[:10000]
        validation_labels = self.training_labels[:10000]
        partial_training_data = self.training_data[10000:]
        partial_training_labels = self.training_labels[10000:]
        self.fit_history = self.model.fit(partial_training_data,
                                          partial_training_labels,
                                          verbose=1,
                                          validation_data=(validation_data, validation_labels),
                                          epochs=40,
                                          batch_size=512).history

    # Evaluate the model against the test data set
    def evaluate(self):
        prediction_loss, prediction_accuracy = self.model.evaluate(self.test_data, self.test_labels)
        print('Prediction Loss: {}, Prediction Accuracy: {}'.format(prediction_loss, prediction_accuracy))

    # Visualize the training accuracy and the validation accuracy of the model across epochs
    def visualize(self):
        training_accuracy = self.fit_history['acc']
        validation_accuracy = self.fit_history['val_acc']
        epochs = range(1, len(training_accuracy) + 1)
        plt.plot(epochs, training_accuracy, linewidth=1.0, color='b', marker='o', label='Training Accuracy')
        plt.plot(epochs, validation_accuracy, linewidth=1.0, color='r', marker='x', label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Model Accuracy')
        plt.legend()
        plt.show()

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] BinaryTextClassification Termination: Tearing things down...')


# The run trigger
if __name__ == '__main__':
    print('[INFO] BinaryTextClassification Trigger: Starting system assessment ...')
    binary_text_classifier = BinaryTextClassification()
    binary_text_classifier.preprocess()
    binary_text_classifier.build()
    binary_text_classifier.evaluate()
    binary_text_classifier.visualize()
