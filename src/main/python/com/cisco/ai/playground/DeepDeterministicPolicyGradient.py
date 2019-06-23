# This entity describes the design of a Deep Deterministic Policy Gradient (DDPG) algorithm built upon a traditional...
# Actor-Critic based Double Deep Q-Learning model with Experience Replay and Soft-Target Training.
# The Actor is the policy network: updated using deterministic policy gradient.
# The Critic is the value function network : updated using the loss function defined by the Bellman equation.
# Batch Normalization is employed to remove the co-variance shift and whiten the input into the layers of the NN.
# Exploration through Gaussian noise addition to the actor policy.
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems, Inc.
# Copyright (c) 2019. All Rights Reserved.


# The imports
import tflearn
import tensorflow


# The Actor network - the policy
class Actor(object):

    # The initialization sequence
    def __init__(self, _tensorflow_session, _state_dimension, _action_dimension, _action_interval, _learning_rate,
                 _target_tracker_coefficient, _batch_size):
        print('[INFO] Actor Initialization: Bringing things up...')
        # The tensorflow training session passed down from the calling method
        self.tensorflow_session = _tensorflow_session
        # The dimensionality of an environmental state
        self.state_dimension = _state_dimension
        # The dimensionality of an action
        self.action_dimension = _action_dimension
        # The action space is continuous yet unconstrained, i.e. uncountable set in the interval [-A, A]
        self.action_interval = _action_interval
        # The learning rate in the Deterministic Policy Gradient Algorithm
        self.learning_rate = _learning_rate
        # The tracking coefficient used in the soft target updates
        self.target_tracking_coefficient = _target_tracker_coefficient
        # The batch size used while normalizing the actor gradients
        self.batch_size = _batch_size
        # The actor network - trainer
        self.inputs, self.outputs, self.scaled_outputs = self.build_model()
        # TensorFlow internally houses the trainable variables for the model created
        self.network_parameters = tensorflow.trainable_variables()
        # The target network for the actor - target
        self.target_inputs, self.target_outputs, self.scaled_target_outputs = self.build_model()
        # Extract the newly available trainable variables
        self.target_network_parameters = tensorflow.trainable_variables()[len(self.network_parameters):]
        # The target update - soft update using the target_update_coefficient (/tau)
        self.updated_target_network_parameters = [self.target_network_parameters[i].assign(
            tensorflow.multiply(self.network_parameters[i], self.target_tracking_coefficient) + tensorflow.multiply(
                self.target_network_parameters[i], (1.0 - self.target_tracking_coefficient))) for i in
            self.target_network_parameters]
        # The actor update sequence
        # The action_gradient - given by the critic
        self.action_gradients = tensorflow.placeholder(tensorflow.float32,
                                                       [None, self.action_dimension])
        self.unnormalized_actor_gradients = tensorflow.gradients(self.scaled_outputs,
                                                                 self.network_parameters,
                                                                 -self.action_gradients)
        self.action_gradients = list(map(lambda x: tensorflow.div(x, self.batch_size),
                                         self.unnormalized_actor_gradients))
        self.optimized_output = tensorflow.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.action_gradients, self.network_parameters)
        )

    # Build the Neural Network which would be instrumental in determining the action policy using the DDPG algorithm
    def build_model(self):
        # The input to the actor network is the state
        input_data = tflearn.input_data(shape=[None, self.state_dimension])
        # The input layer
        model = tflearn.fully_connected(input_data,
                                        400)
        # A layer of Batch Normalization
        # Batch Normalization normalizes each dimension across the samples to have unit mean and unit variance.
        # Batch Normalization mitigates covariance shift during training by whitening the inputs to the layers of the NN
        model = tflearn.layers.normalization.batch_normalization(model)
        model = tflearn.activations.relu(model)
        model = tflearn.fully_connected(model,
                                        300)
        # Another layer of Batch Normalization
        model = tflearn.layers.normalization.batch_normalization(model)
        model = tflearn.activations.relu(model)
        # Initialize the weights and use these weights in the final layer
        initial_weights = tflearn.initializations.uniform(minval=-0.003,
                                                          maxval=0.003)
        # tanh range [-1, 1] fits in well for the final layer
        output_data = tflearn.fully_connected(model,
                                              self.action_dimension,
                                              activation='tanh',
                                              weights_init=initial_weights)
        # Reign in the action to within the allowed interval
        scaled_output_data = tensorflow.multiply(output_data, self.action_interval)
        return input_data, output_data, scaled_output_data

    # Populate the placeholder and use the tensorflow training session passed earlier to train the model
    def train_model(self, _inputs, _action_gradients):
        self.tensorflow_session.run(self.optimized_output,
                                    feed_dict={
                                        self.inputs: _inputs,
                                        self.action_gradients: _action_gradients
                                    })

    # Run all preceding operations to produce the action from the policy actor network
    def predict(self, _inputs):
        return self.tensorflow_session.run(self.scaled_outputs,
                                           feed_dict={self.inputs: _inputs})

    # Run all preceding operations to produce the action from the target actor network
    def predict_targets(self, _target_inputs):
        return self.tensorflow_session.run(self.scaled_target_outputs,
                                           feed_dict={self.target_inputs: _target_inputs})

    # Run all preceding operations to update the target actor network parameters
    def update_target_network(self):
        self.tensorflow_session.run(self.updated_target_network_parameters)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Actor Termination: Tearing things down...')


# The Critic Network - the value function
class Critic(object):

    # The initialization sequence
    def __init__(self, _tensorflow_session, _state_dimension, _action_dimension, _learning_rate,
                 _target_tracking_coefficient, _number_of_actor_variables):
        print('[INFO] Critic Initialization: Bringing things up...')
        # The tensorflow training session passed down from the calling method
        self.tensorflow_session = _tensorflow_session
        # The dimensionality of an environment state
        self.state_dimension = _state_dimension
        # The dimensionality of an action executed by the agent, i.e. selected by the policy network
        self.action_dimension = _action_dimension
        # The learning rate used in the Deterministic Policy Gradient algorithm
        self.learning_rate = _learning_rate
        # The tracking coefficient used in the target updates - soft target updates
        self.target_tracking_coefficient = _target_tracking_coefficient
        # The parameters of the critic
        self.input, self.action, self.output = self.build_model()
        self.network_parameters = tensorflow.trainable_variables()[_number_of_actor_variables:]
        # The parameters of the critic target network
        self.target_input, self.target_action, self.target_output = self.build_model()
        self.target_network_parameters = tensorflow.trainable_variables()[
                                         len(self.network_parameters) + _number_of_actor_variables:]
        # Perform the target network update
        self.updated_target_network_parameters = [self.target_network_parameters[i].assign(
            tensorflow.multiply(self.network_parameters[i], self.learning_rate) +
            tensorflow.multiply(self.target_network_parameters[i], (1.0 - self.learning_rate))) for i in
            range(self.target_network_parameters)]
        self.predicted_q_value = tensorflow.placeholder(tensorflow.float32,
                                                        shape=[None, 1])
        self.loss = tflearn.mean_square(self.predicted_q_value,
                                        self.output)
        self.optimized_result = tensorflow.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.action_gradients = tensorflow.gradients(self.output,
                                                     self.action)

    # Build the critic network - value function network
    def build_model(self):
        input_data = tflearn.input_data(shape=[None, self.state_dimension])
        action_data = tflearn.input_data(shape=[None, self.action_dimension])
        model = tflearn.fully_connected(input_data, 400)
        model = tflearn.layers.normalization.batch_normalization(model)
        model = tflearn.activations.relu(model)
        temporary_hidden_layer_1 = tflearn.fully_connected(model, 300)
        temporary_hidden_layer_2 = tflearn.fully_connected(action_data, 300)
        model = tflearn.activation(
            tensorflow.matmul(model, temporary_hidden_layer_1.W) +
            tensorflow.matmul(action_data, temporary_hidden_layer_2) +
            temporary_hidden_layer_2.b,
            activation='relu')
        initial_weights = tflearn.initializations.uniform(minval=-0.003,
                                                          maxval=0.003)
        output_data = tflearn.fully_connected(model,
                                              1,
                                              weights_init=initial_weights)
        return input_data, action_data, output_data

    # Populate the predicted Q-value (y_i) and train the built model
    def train_model(self, _input, _action, _predicted_q_value):
        self.tensorflow_session.run([self.output, self.optimized_result],
                                    feed_dict={
                                        self.input: _input,
                                        self.action: _action,
                                        self.predicted_q_value: _predicted_q_value
                                    })

    # Run all the preceding operations to update the target network's parameters
    def update_target_parameters(self):
        self.tensorflow_session.run(self.updated_target_network_parameters)

    # Run all the preceding operations to predict the Q-value of a given state-action pair - Critic
    def predict(self, _input, _action):
        return self.tensorflow_session.run(self.output,
                                           feed_dict={
                                               self.input: _input,
                                               self.action: _action
                                           })

    # Run all the preceding operations to predict the Q-value of a given state-action pair - Target Critic
    def predict_target(self, _input, _action):
        return self.tensorflow_session.run(self.target_output,
                                           feed_dict={
                                               self.input: _input,
                                               self.action: _action
                                           })

    # Run all the preceding operations to obtain the action gradients which will be used in the actor for the DDPG alg
    def get_action_gradients(self, _input, _action):
        return self.tensorflow_session.run(self.action_gradients,
                                           feed_dict={
                                               self.input: _input,
                                               self.action: _action
                                           })

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Critic Termination: Tearing things down...')


# This class encapsulates the design of a DDPG-based Actor-Critic DDQN model for solving problems with very large...
# ...state spaces and continuous action spaces - The Pendulum Problem (env provided by OpenAI Gym).
class DeepDeterministicPolicyGradient(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] DeepDeterministicPolicyGradient Initialization: Bringing things up...')

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] DeepDeterministicPolicyGradient Termination: Tearing things down...')
