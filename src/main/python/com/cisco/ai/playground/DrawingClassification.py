# This entity encapsulates the design of a RNN based Drawing Classifier using TensorFlow and the high-level Keras API
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems
# Copyright (c) 2019. All Rights Reserved.


# Classify drawings into categories using a Recurrent Neural Network
class DrawingClassification(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] DrawingClassification Initialization: Bringing things up...')

    # Read and parse the data
    # def read_parse_data(self):
    # The data is available as TFRecords
    # Read it and parse it

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] DrawingClassification Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] DrawingClassification Trigger: Starting system assessment!')
