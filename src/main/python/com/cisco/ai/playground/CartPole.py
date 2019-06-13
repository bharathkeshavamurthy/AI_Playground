# This entity encapsulates the design of a Deep Q Network using features from TensorFlow and the high-level Keras API
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems, Inc.
# Copyright (c) 2019. All Rights Reserved.

import gym
import numpy
import random
import tensorflow
from collections import deque

# The CartPole game id in OpenAI Gym
CARTPOLE_GAME_ID = 'CartPole-v0'

# The number of episodes
NUMBER_OF_EPISODES = 10000

# The maximum optimal score threshold
MAXIMUM_SCORE_THRESHOLD = 500

# Batch size
BATCH_SIZE = 32


# This class describes an agent to play the CartPole game efficiently
class CartPole(object):
    # The memory capacity of the agent
    MEMORY_SIZE = 20000

    # The initialization sequence
    def __init__(self, state_size, action_size):
        print('[INFO] CartPole Initialization: Bringing things up...')
        # The size of the state for this problem
        self.state_size = state_size
        # The size of the action for this problem
        self.action_size = action_size
        # The memory for experiential replay
        # The experiential replay technique solves the problem of non-linearities encountered while using the NN
        # The experiential replay technique stabilizes the Q-value approximation operation performed by the NN and...
        # ...helps it converge
        self.memory = deque(maxlen=int(self.MEMORY_SIZE))
        # The hyper-parameters of this problem
        # The exploration factor - start off big, then decay it down to epsilon_min
        self.epsilon = 1.0
        # The epsilon decay factor for transitioning from exploration to exploitation as the model improves
        self.epsilon_decay = 0.6
        # The minimum amount of exploration needed
        self.epsilon_min = 0.001
        # The learning rate
        self.alpha = 0.01
        # The discount factor
        self.gamma = 0.9

    # Build the Neural Network in order to determine the Q-values for the state-action pairs
    # Using a Neural Network will solve the high-dimensionality problem that's generally encountered while...
    # ...determining the Q-values for all the state-action pairs
    def build_nn(self):
        # Design the model
        _model = tensorflow.keras.Sequential([
            # Input Layer
            # Non-Linearity - ReLU
            tensorflow.keras.layers.Dense(units=1024,
                                          input_dim=self.state_size,
                                          activation=tensorflow.nn.relu),
            # Hidden Layer
            # Non-Linearity - ReLU
            tensorflow.keras.layers.Dense(units=1024,
                                          activation=tensorflow.nn.relu),
            tensorflow.keras.layers.Dense(units=self.action_size,
                                          activation='linear')
        ])
        # Specify the Cost Function and the Optimizer
        # loss = (reward + \gamma max_{a'}Q(s', a') - Q(s, a))^2
        _model.compile(loss=tensorflow.keras.losses.mean_squared_error,
                       optimizer=tensorflow.train.AdamOptimizer(learning_rate=self.alpha),
                       metrics=[tensorflow.keras.metrics.mean_absolute_error,
                                tensorflow.keras.metrics.mean_squared_error])
        return _model

    # The experiential replay technique's remember routine
    def remember(self, _state, _action, _reward, _next_state, _done):
        self.memory.append((_state, _action, _reward, _next_state, _done))

    # Choose an action based on the Exploration/Exploitation levels
    def act(self, _state, _model):
        if numpy.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return numpy.argmax(_model.predict(_state)[0])

    # Train the Neural Network with the experiences stored in memory
    def replay(self, _batch_size, _model):
        experiences = random.sample(self.memory, _batch_size)
        for _state, _action, _reward, _next_state, _done in experiences:
            target = _reward
            if not _done:
                target = _reward + (self.gamma * numpy.amax(_model.predict(_next_state)[0]))
            target_train = _model.predict(_state)
            target_train[0][_action] = target
            _model.fit(_state, target_train, epochs=1, verbose=0)
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return _model

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] CartPole Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] CartPole Trigger: Starting system assessment!')
    # The environment
    environment = gym.make(CARTPOLE_GAME_ID)
    # The agent
    agent = CartPole(environment.observation_space.shape[0], environment.action_space.n)
    # Build and Compile the model
    model = agent.build_nn()
    for episode in range(NUMBER_OF_EPISODES):
        state = environment.reset()
        state = numpy.reshape(state, [1, environment.observation_space.shape[0]])
        # The time for which the pole is balanced is the agent's score
        for score in range(MAXIMUM_SCORE_THRESHOLD):
            environment.render()
            action = agent.act(state, model)
            next_state, reward, done, _ = environment.step(action)
            next_state = numpy.reshape(next_state, [1, environment.observation_space.shape[0]])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print('Episode {}/{}: Score = {}'.format(episode, NUMBER_OF_EPISODES, score))
                break
        if len(agent.memory) > BATCH_SIZE:
            model = agent.replay(BATCH_SIZE, model)
