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
import gym
import numpy
import random
import tflearn
import tensorflow
from gym import wrappers
from collections import deque


# The Actor network - the policy
class Actor(object):

    # The initialization sequence
    def __init__(self, _tensorflow_session, _state_dimension, _action_dimension, _action_bound, _learning_rate,
                 _target_tracker_coefficient, _batch_size):
        print('[INFO] Actor Initialization: Bringing things up...')
        # The tensorflow training session passed down from the calling method
        self.tensorflow_session = _tensorflow_session
        # The dimensionality of an environmental state
        self.state_dimension = _state_dimension
        # The dimensionality of an action
        self.action_dimension = _action_dimension
        # The action space is continuous yet unconstrained, i.e. uncountable set in the interval [-A, A]
        self.action_bound = _action_bound
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
                self.target_network_parameters[i],
                (1.0 - self.target_tracking_coefficient))) for i in range(len(self.target_network_parameters))]
        # The actor update sequence
        # The action_gradient - given by the critic
        self.action_gradients = tensorflow.placeholder(tensorflow.float32,
                                                       [None, self.action_dimension])
        self.unnormalized_actor_gradients = tensorflow.gradients(self.scaled_outputs,
                                                                 self.network_parameters,
                                                                 -self.action_gradients)
        self.actor_gradients = list(map(lambda x: tensorflow.div(x, self.batch_size),
                                        self.unnormalized_actor_gradients))
        self.optimized_output = tensorflow.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.actor_gradients, self.network_parameters)
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
        scaled_output_data = tensorflow.multiply(output_data,
                                                 self.action_bound)
        return input_data, output_data, scaled_output_data

    # Populate the placeholder and use the tensorflow training session passed earlier to train the model
    def train_model(self, _inputs, _action_gradients):
        return self.tensorflow_session.run(self.optimized_output,
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
    def update_target_network_parameters(self):
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
            range(len(self.target_network_parameters))]
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
            tensorflow.matmul(action_data, temporary_hidden_layer_2.W) +
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
        return self.tensorflow_session.run([self.output, self.optimized_result],
                                           feed_dict={
                                               self.input: _input,
                                               self.action: _action,
                                               self.predicted_q_value: _predicted_q_value
                                           })

    # Run all the preceding operations to update the target network's parameters
    def update_target_network_parameters(self):
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
                                               self.target_input: _input,
                                               self.target_action: _action
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


# Ornstein-Uhlenbeck Noise Process
# Off-Policy learning algorithms decouple the exploration process from the learning process
# The exploration policy involves adding this generated Ornstein-Uhlenbeck noise
# Vasicek Model - \[dX_t = \theta (\mu - X_t) dt + \sigma dW_t\] - W_t is a Wiener Process
# Simplified Model: \[X_{t+\alpha} = X_t + (\theta (\mu - X_t) dt) + (\sigma \sqrt(dt) \mathcal{N}(0, 1))\]
class Noise(object):

    # The initialization sequence
    def __init__(self, _x0, _mu, _theta=0.15, _sigma=0.3, _dt=1e-2):
        # Refer to the Vasicek model of Ornstein-Uhlenbeck process for more information on what the following...
        # ...parameters refer to
        # The centre to which the process gravitates to as it moves further and further away from it
        self.mu = _mu
        # The parameter $\theta > 0$ in the Stochastic Differential Equation
        self.theta = _theta
        # The parameter $\sigma > 0$ in the Stochastic Differential Equation
        self.sigma = _sigma
        # The time delta $\triangleup t$
        self.dt = _dt
        # The initialization of X
        self.x_prev = (lambda: _x0,
                       lambda: numpy.zeros_like(self.mu))[_x0 is None]()

    # The core routine
    def __call__(self):
        x = self.x_prev + (self.theta * (self.mu - self.x_prev) * self.dt) + (
                self.sigma * numpy.sqrt(self.dt) * numpy.random.normal(self.mu.shape))
        return x

    # String representation of the object
    def __repr__(self):
        return 'Ornstein-Uhlenbeck Noise Process[mu={}, sigma={}]'.format(self.mu, self.sigma)


# The experiential replay memory
class Memory(object):

    # The initialization sequence
    def __init__(self, _size, _random_seed):
        print('[INFO] Memory Initialization: Bringing things up...')
        # The memory capacity - size of the replay buffer
        self.size = _size
        # The random seed for consistency across multiple runs
        self.random_seed = _random_seed
        random.seed(self.random_seed)
        # Creating the replay memory
        self.memory = deque(maxlen=self.size)

    # Add the experience to memory
    def remember(self, _current_state, _action, _reward, _next_state, _termination):
        self.memory.append((_current_state,
                            _action, _reward,
                            _next_state,
                            _termination))

    # Sample an experience uniformly at random from memory
    def replay(self, _batch_size):
        if len(self.memory) < _batch_size:
            sample = random.sample(self.memory, len(self.memory))
        else:
            sample = random.sample(self.memory, _batch_size)
        # state, action, reward, next_state, termination
        return numpy.array([k[0] for k in sample]), numpy.array([k[1] for k in sample]), numpy.array(
            [k[2] for k in sample]), numpy.array([k[3] for k in sample]), numpy.array([k[4] for k in sample])

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Memory Termination: Tearing things down...')


# This class encapsulates the design of a DDPG-based Actor-Critic DDQN model for solving problems with very large...
# ...state spaces and continuous action spaces - The Pendulum Problem (env provided by OpenAI Gym).
class Pendulum(object):
    # The OpenAI Gym Pendulum Environment ID
    ENVIRONMENT_ID = 'Pendulum-v0'

    # The Random Seed
    RANDOM_SEED = 7

    # The Learning Rate used in the Actor Network
    ACTOR_LEARNING_RATE = 1e-4

    # The Learning Rate used in the Critic Network
    CRITIC_LEARNING_RATE = 1e-3

    # The Target Tracker Coefficient
    TARGET_TRACKER_COEFFICIENT = 1e-2

    # The Discount Factor employed in the target Q-value estimation in the Critic Network
    DISCOUNT_FACTOR = 0.9

    # The Mini-Batch Size for Stochastic Gradient Descent
    MINI_BATCH_SIZE = 64

    # The Capacity of the Replay Memory
    REPLAY_MEMORY_CAPACITY = 1e9

    # Maximum number of iterations per episode
    ITERATIONS_PER_EPISODE = 1000

    # Maximum number of episodes
    MAXIMUM_NUMBER_OF_EPISODES = 100000

    # The initialization sequence
    def __init__(self):
        print('[INFO] DeepDeterministicPolicyGradient Initialization: Bringing things up...')
        # Open AI Gym's monitor directory
        self.monitor_directory = './pendulum-monitor'
        # The TensorFlow training and interaction summary directory
        self.summary_directory = './pendulum-summary'
        # Start the tensorflow session
        with tensorflow.Session() as session:
            # Create the Pendulum-v0 environment
            self.environment = gym.make(self.ENVIRONMENT_ID)
            # Set the random seed for consistency across multiple runs
            numpy.random.seed(self.RANDOM_SEED)
            self.environment.seed(self.RANDOM_SEED)
            tensorflow.set_random_seed(self.RANDOM_SEED)
            # The dimensions of the environment state
            self.state_dimension = self.environment.observation_space.shape[0]
            # The dimensions of the actions allowed
            self.action_dimension = self.environment.action_space.shape[0]
            # The upper bound in the allowed action space
            self.action_bound = self.environment.action_space.high
            # Create the actor instance
            self.actor = Actor(session,
                               self.state_dimension,
                               self.action_dimension,
                               self.action_bound,
                               self.ACTOR_LEARNING_RATE,
                               self.TARGET_TRACKER_COEFFICIENT,
                               self.MINI_BATCH_SIZE)
            # Create the critic instance
            self.critic = Critic(session,
                                 self.state_dimension,
                                 self.action_dimension,
                                 self.CRITIC_LEARNING_RATE,
                                 self.TARGET_TRACKER_COEFFICIENT,
                                 len(self.actor.network_parameters + self.actor.target_network_parameters))
            # Create the Ornstein-Uhlenbeck noise process instance
            self.exploration_noise = Noise(_x0=None,
                                           _mu=numpy.zeros(self.action_dimension))
            # Enable monitoring and rendering of the environment reactions and the feedback process
            self.environment = wrappers.Monitor(self.environment,
                                                self.monitor_directory,
                                                force=True)
            # Start the Actor-Critic DDPG with Experiential Replay process
            self.train(session)
            # Stop the monitoring and rendering services
            self.environment.monitor.close()

    # Visualize the process summary in the console
    @staticmethod
    def build_summaries():
        episodic_reward = tensorflow.Variable(0.0)
        tensorflow.summary.scalar("Episodic Reward", episodic_reward)
        episodic_average_max_q_value = tensorflow.Variable(0.0)
        tensorflow.summary.scalar("Episodic Average Maximum Q-value", episodic_average_max_q_value)
        summary_variables = [episodic_reward, episodic_average_max_q_value]
        summary_ops = tensorflow.summary.merge_all()
        return summary_ops, summary_variables

    # Start the interaction and training process
    def train(self, session):
        summary_ops, summary_variables = self.build_summaries()
        session.run(tensorflow.global_variables_initializer())
        writer = tensorflow.summary.FileWriter(self.summary_directory, session.graph)
        self.actor.update_target_network_parameters()
        self.critic.update_target_network_parameters()
        replay_memory = Memory(int(self.REPLAY_MEMORY_CAPACITY), self.RANDOM_SEED)
        for episode in range(self.MAXIMUM_NUMBER_OF_EPISODES):
            state = self.environment.reset()
            episodic_reward = 0.0
            episodic_average_max_q_value = 0.0
            for iteration in range(self.ITERATIONS_PER_EPISODE):
                self.environment.render()
                action = self.actor.predict(numpy.reshape(state, (1, self.state_dimension))) + self.exploration_noise()
                next_state, reward, termination, metadata = self.environment.step(action[0])
                replay_memory.remember(numpy.reshape(state, (self.state_dimension,)),
                                       numpy.reshape(action, (self.action_dimension,)),
                                       reward,
                                       numpy.reshape(next_state, (self.state_dimension,)),
                                       termination)
                if len(replay_memory.memory) > self.MINI_BATCH_SIZE:
                    s_batch, a_batch, r_batch, s2_batch, t_batch = replay_memory.replay(self.MINI_BATCH_SIZE)
                    target_q = self.critic.predict_target(s2_batch,
                                                          self.actor.predict_targets(s2_batch))
                    target_q_values = []
                    for k in range(self.MINI_BATCH_SIZE):
                        if t_batch[k]:
                            target_q_values.append(r_batch[k])
                        else:
                            target_q_values.append(r_batch[k] + (self.DISCOUNT_FACTOR * target_q[k]))
                    critic_training_result = self.critic.train_model(s_batch,
                                                                     a_batch,
                                                                     numpy.reshape(target_q_values,
                                                                                   (self.MINI_BATCH_SIZE, 1)))
                    episodic_average_max_q_value += numpy.amax(critic_training_result[0])
                    action_output = self.actor.predict(s_batch)
                    action_gradients = self.critic.get_action_gradients(s_batch, action_output)
                    self.actor.train_model(s_batch, action_gradients[0])
                    self.actor.update_target_network_parameters()
                    self.critic.update_target_network_parameters()
                state = next_state
                episodic_reward += reward
                if termination:
                    summary = session.run(summary_ops,
                                          feed_dict={
                                              summary_variables[0]: episodic_reward,
                                              summary_variables[1]: episodic_average_max_q_value / float(iteration)
                                          })
                    writer.add_summary(summary, episode)
                    writer.flush()
                    print('[INFO] DeepDeterministicPolicyGradient train: Episode = {} | Reward = {} | '
                          'Q_max = {}'.format(episode,
                                              episodic_reward,
                                              episodic_average_max_q_value / float(iteration)))
                    break

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] DeepDeterministicPolicyGradient Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] DeepDeterministicPolicyGradient Trigger: Starting system assessment!')
    ddpg_engine = Pendulum()
