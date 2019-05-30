# This class describes the design of a Recurrent Neural Network (RNN) using TensorFlow and the high-level Keras API
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems
# Copyright (c) 2019. All Rights Reserved.

import numpy
import functools
import tensorflow

tensorflow.enable_eager_execution()


# This class encapsulates a Text Generation Scheme using RNNs by leveraging the capabilities of RNNs
class RecurrentNeuralNetworks(object):

    # Split the text in order to generate input and target pairs (right-shifted by one place)
    @staticmethod
    def split_based_on_sequence_length(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    # The initialization sequence
    def __init__(self):
        print('[INFO] RecurrentNeuralNetworks Initialization: Bringing things up...')
        file_path = tensorflow.keras.utils.get_file(
            'Sample_Text.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        self.text = open(file_path, 'rb').read().decode(encoding='utf-8')
        self.vocabulary = sorted(set(self.text))
        self.character_to_int_mapping = {u: i for i, u in enumerate(self.vocabulary)}
        self.int_to_character_mapping = numpy.array(self.vocabulary)
        text_as_int = [self.character_to_int_mapping[character] for character in self.text]
        character_dataset = tensorflow.data.Dataset.from_tensor_slices(text_as_int).batch(101, drop_remainder=True)
        self.batched_split_data_set = character_dataset.map(self.split_based_on_sequence_length).shuffle(10000).batch(
            64, drop_remainder=True)
        self.model = None

    # Build the model using keras layers
    def build(self):
        # Develop a modified RNN using the keras GRU module
        modified_rnn = functools.partial(tensorflow.keras.layers.GRU, recurrent_activation='sigmoid')
        self.model = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Embedding(len(self.vocabulary), 256, batch_input_shape=[64, None]),
            modified_rnn(1024, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True),
            # A logits output vector with a probability emphasis on the likelihood of it being the next character
            tensorflow.keras.layers.Dense(len(self.vocabulary))
        ])
        print('[INFO] RecurrentNeuralNetworks build: Model Summary - {}'.format(self.model.summary()))
        # Return this in case something needs this
        return self.model

    # Define the cost function
    @staticmethod
    def cost(labels, logits):
        return tensorflow.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    # Compile the model using the AdamOptimizer and the Sparse Categorical Cross-Entropy cost function
    def compile(self):
        self.model.compile(optimizer=tensorflow.train.AdamOptimizer(),
                           loss=self.cost)

    # Train the model based on the batched split data set
    def train(self):
        print('[INFO] RecurrentNeuralNetworks train: Training History - {}'.format(
            self.model.fit(self.batched_split_data_set.repeat(), epochs=30, steps_per_epoch=len(self.text))))

    # Generate new text sequences after training the model
    def generate_text(self):
        start_string = u'First: '
        # A measure of randomness
        temperature = 1.0
        num_of_characters_to_generate = 100
        # Convert the start string into the corresponding integer/index mappings
        evaluation_input = [self.character_to_int_mapping[character] for character in start_string]
        evaluation_input = tensorflow.expand_dims(evaluation_input, 0)
        text_generated = []
        # Reset states of the RNN - Initialization
        self.model.reset_states()
        for i in range(0, num_of_characters_to_generate):
            # Predict the next character using the trained RNN model
            predictions = self.model(evaluation_input)
            predictions = tensorflow.squeeze(predictions, 0)
            predictions = predictions / temperature
            prediction_index = tensorflow.multinomial(predictions, num_samples=1)[-1, 0].numpy()
            evaluation_input = tensorflow.expand_dims([prediction_index], 0)
            text_generated.append(self.int_to_character_mapping[prediction_index])
        return start_string + ' '.join(text_generated)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] RecurrentNeuralNetworks Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] RecurrentNeuralNetworks Trigger: Starting system assessment!')
    RNN = RecurrentNeuralNetworks()
    RNN.build()
    RNN.compile()
    RNN.train()
    print('[INFO] RecurrentNeuralNetworks Trigger: Generated Text - {}'.format(RNN.generate_text()))
