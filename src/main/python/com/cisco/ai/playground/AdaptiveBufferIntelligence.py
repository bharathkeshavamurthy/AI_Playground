# |Bleeding-Edge Productions|
# This entity describes the design of an intelligent, adaptive buffer allocation engine using Deep Deterministic...
# ...Policy Gradients (DDPG) in an Asynchronous Advantage Actor Critic (A3C) architecture based on the Double Deep...
# ...Q-Networks Prioritized Experiential Learning (DDQN-PER) framework.
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems, Inc.
# Copyright (c) 2019. All Rights Reserved.

# The imports
import math
import enum
import numpy
import random
import tensorflow
from collections import namedtuple, deque

# A global instance
# Environmental Feedback API entity
FEEDBACK = namedtuple('feedback',
                      ['reward',
                       'next_state'])


# Enumeration entities critical to an extensible design begin here...

# An extensible enumeration entity listing the possible statuses of modules in this design
class Status(enum):
    # The module is in a correct working state
    SUCCESS = 0

    # The module is not up - some modular operation failed
    FAILURE = 1


# An extensible enumeration entity listing the possible priority types in the design
class Priority(enum):
    # Systems with higher service rates
    HIGH_PRIORITY = 0

    # Systems with comparatively lower service rates
    LOW_PRIORITY = 1


# An extensible enumeration entity listing the possible prioritization techniques employed in Mnemosyne
class Prioritization(enum):
    # Prioritization using TD-error
    TD_ERROR_PRIORITIZATION = 0

    # Stochastic sampling using proportional TD-error
    STOCHASTIC_PRIORITIZATION_PROPORTIONAL = 1

    # Stochastic sampling using rank of a transition within the replay memory
    STOCHASTIC_PRIORITIZATION_RANK = 2

    # Purely random sampling strategy
    RANDOM = 3


# An extensible enumeration entity listing the possible exploration strategies employed in the RL agent
class ExplorationStrategy(enum):
    # Additive Ornstein-Uhlenbeck Noise
    ORNSTEIN_UHLENBECK_NOISE = 0

    # \epsilon-greedy selection with decaying exploration factor (\epsilon)
    EXPLORATION_DECAY = 1


# Enumeration entities critical to an extensible design end here...

# The switch environment [Nexus] begins here...

# The CISCO Nexus DC switch environment
# Definitions of states, actions, rewards, transitions, emissions, and steady-state initializations are encapsulated...
# ...within this class.
class Nexus(object):
    # The default number of ports in the switch
    NUMBER_OF_PORTS = 3

    # The default number of queues in the switch
    NUMBER_OF_QUEUES_PER_PORT = 3

    # The default global pool size
    GLOBAL_POOL_SIZE = 120

    # The default local pool size (per port)
    DEDICATED_POOL_SIZE_PER_PORT = 40

    # The penalty for invalid transitions, i.e. invalid actions
    INCOMPETENCE_PENALTY = 5.0

    # A queue entity
    QUEUE = namedtuple('Queue',
                       ['queue_identifier',
                        'priority',
                        'required_minimum_capacity',
                        'allowed_maximum_buffer_capacity',
                        'allocated_buffer_units',
                        'packet_drop_count'
                        ])

    # A port entity
    PORT = namedtuple('Port',
                      ['port_identifier',
                       'leftover_buffer_units_in_the_dedicated_pool',
                       'queues'
                       ])

    # A system state
    STATE = namedtuple('State',
                       ['ports',
                        'leftover_buffer_units_in_the_global_pool'
                        ])

    # The initialization sequence
    def __init__(self, _number_of_ports, _number_of_queues_per_port, _global_pool_size, _dedicated_pool_size_per_port):
        print('[INFO] Nexus Initialization: Bringing things up...')
        # A flag checking the status of the switch
        self.status = Status.FAILURE
        # Setting the design parameters - default to hard-coded values upon invalidation
        self.number_of_ports = (lambda: self.NUMBER_OF_PORTS,
                                lambda: _number_of_ports)[
            _number_of_ports is not None and
            isinstance(_number_of_ports, int) and
            _number_of_ports > 0]()
        self.number_of_queues_per_port = (lambda: self.NUMBER_OF_QUEUES_PER_PORT,
                                          lambda: _number_of_queues_per_port)[
            _number_of_queues_per_port is not None and
            isinstance(_number_of_queues_per_port, int) and
            _number_of_queues_per_port > 0]()
        self.global_pool_size = (lambda: self.GLOBAL_POOL_SIZE,
                                 lambda: _global_pool_size)[
            _global_pool_size is not None and
            isinstance(_global_pool_size, int) and
            _global_pool_size > 0]()
        self.dedicated_pool_size_per_port = (lambda: self.DEDICATED_POOL_SIZE_PER_PORT,
                                             lambda: _dedicated_pool_size_per_port)[
            _dedicated_pool_size_per_port is not None and
            isinstance(_dedicated_pool_size_per_port, int) and
            _dedicated_pool_size_per_port > 0
            ]()
        # Initialize the environment
        self.state = self.start()
        # The allowed action skeleton - the switch will have an API to validate and execute compliant actions
        # The initial compliant action is [ [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 0]
        self.action_skeleton = [[((port * 0) + (queue * 0)) for queue in self.number_of_queues_per_port] for port in
                                self.number_of_ports]
        # Successful initialization
        self.status = Status.SUCCESS
        # A flag which indicated gross incompetence exhibited by the RL agent
        # If this flag is set to true, impose sky-high tariffs...
        self.incompetence = False
        # The initialization sequence has been completed...

    # Initialize the system state
    def start(self):
        print('[DEBUG] Nexus start: Initializing the switch environment state...')
        # The ports
        ports = []
        for port in range(self.number_of_ports):
            # The queues
            queues = []
            # The distribution factors
            low_priority_distribution_factor = 2.0 / (
                    self.number_of_queues_per_port + (self.number_of_queues_per_port - 1))
            high_priority_distribution_factor = 0.5 * low_priority_distribution_factor
            for queue in range(self.number_of_queues_per_port):
                priority = (lambda: Priority.LOW_PRIORITY,
                            lambda: Priority.HIGH_PRIORITY)[queue == 0]()
                queues.append(self.QUEUE(queue_identifier='P' + str(port) + 'Q' + str(queue),
                                         # The first queue in every port is designated a High-Priority port
                                         # High-Priority ports have higher service rates and hence have smaller buffers
                                         priority=priority,
                                         required_minimum_capacity=0,
                                         allowed_maximum_buffer_capacity=(
                                             lambda: math.floor(
                                                 low_priority_distribution_factor * self.dedicated_pool_size_per_port),
                                             lambda: math.floor(
                                                 high_priority_distribution_factor * self.dedicated_pool_size_per_port)
                                         )[priority.value == Priority.HIGH_PRIORITY](),
                                         allocated_buffer_units=0,
                                         packet_drop_count=0))
            ports.append(self.PORT(port_identifier='P' + str(port),
                                   leftover_buffer_units_in_the_dedicated_pool=self.dedicated_pool_size_per_port,
                                   queues=queues))
        # Return the system state
        return self.STATE(ports=ports,
                          leftover_buffer_units_in_the_global_pool=self.global_pool_size)
        # The switch environment state initialization has been completed...

    # Get the current state of the switch environment
    def state(self):
        # A simple getter method for external callers
        return self.state

    # Validate the new state - check if it complies with the switch design, given the current state
    def validate(self, new_state):
        print('[DEBUG] Nexus validate: Validating the state transition of the underlying MDP...')
        base_availability = self.global_pool_size + (self.dedicated_pool_size_per_port * self.number_of_ports)
        aspirational_availability = new_state.leftover_buffer_units_in_the_global_pool
        # Ports Loop - i
        for i in range(self.number_of_ports):
            aspirational_availability += new_state.ports[i].leftover_buffer_units_in_the_dedicated_pool
            # Queues Loop - j
            for j in range(self.number_of_queues_per_port):
                aspirational_availability += new_state.ports[i].queues[j].allocated_buffer_units
        # Aspirational != Base Reality
        if aspirational_availability != base_availability:
            print('[ERROR] Nexus validate: The number of buffer units in the new state does not meet the design '
                  'requirements for this switch.')
            return False
        # A deeper check - global pool compliance, dedicated pools compliance, individual queue-specific compliance
        for port in new_state.ports:
            port_specific_availability = port.leftover_buffer_units_in_the_dedicated_pool
            for queue in port.queues:
                port_specific_availability += queue.allocated_buffer_units
            if port_specific_availability < self.dedicated_pool_size_per_port:
                self.incompetence = True
                print('[WARN] Nexus validate: State transition denied! Incompetence is set to True.')
                return False
        # Everything's perfectly compliant with the design
        return True
        # The state transition validation has been completed...

    # Evaluate the utility of the recommendation made by the RL agent
    def reward(self):
        print('[DEBUG] Nexus reward: Evaluating the utility of the recommendation...')
        reward = -sum([q.packet_drop_count for p in self.state.ports for q in p.queues])
        if self.incompetence:
            return self.INCOMPETENCE_PENALTY * reward
        return reward

    # Transition from the current state to the next state, Validate the transition
    # Return <reward, next_state>
    def execute(self, action):
        print('[DEBUG] Nexus transition: Transitioning the underlying MDP...')
        # Initial structural validation of the action
        internal_length_operator = numpy.vectorize(len)
        if len(action) != self.number_of_ports or \
                sum(internal_length_operator(action)) != (self.number_of_ports * self.number_of_queues_per_port):
            print('[ERROR] Nexus validate_action: Non-Compliant Action received from the recommendation system - '
                  '{}'.format(str(action)))
            return False
        # C_{global} global pool update
        leftover_buffer_units_in_the_global_pool = self.state.leftover_buffer_units_in_the_global_pool + action[
            self.number_of_ports]
        ports = []
        # Ports Loop - i
        for i in range(self.number_of_ports):
            queues = []
            # C_{local}^{P_i} dedicated pool update
            leftover_buffer_units_in_the_dedicated_pool = self.state.ports[
                                                              i].leftover_buffer_units_in_the_dedicated_pool + action[
                                                              i][self.number_of_queues_per_port]
            # Queues Loop - j
            for j in range(self.number_of_queues_per_port):
                queues.append(self.QUEUE(queue_identifier=self.state.ports[i].queues[j].queue_identifier,
                                         priority=self.state.ports[i].queues[j].priority,
                                         required_minimum_capacity=self.state.ports[i].queues[
                                             j].required_minimum_capacity,
                                         allowed_maximum_buffer_capacity=self.state.ports[i].queues[
                                             j].allowed_maximum_buffer_capacity,
                                         allocated_buffer_units=self.state.ports[i].queues[
                                                                    j].allocated_buffer_units + action[i][j],
                                         packet_drop_count=self.state.ports[i].queues[j].packet_drop_count))
            ports.append(self.PORT(
                port_identifier=self.state.ports[i].port_identifier,
                leftover_buffer_units_in_the_dedicated_pool=leftover_buffer_units_in_the_dedicated_pool,
                queues=queues))
        next_state = self.STATE(ports=ports,
                                leftover_buffer_units_in_the_global_pool=leftover_buffer_units_in_the_global_pool)
        # Validate the new state and either authorize or deny the transition
        # Denial Philosophy: Persistence during Incompetence
        self.state, self.incompetence = (lambda: self.state, True,
                                         lambda: next_state, False)[self.validate(next_state)]()
        return FEEDBACK(reward=self.reward(),
                        next_state=self.state)
        # The state transition of the underlying MDP has been completed...

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Nexus Termination: Tearing things down...')


# The switch environment [Nexus] ends here...

# The Actor Network begins here...

# The Actor Network
class Actor(object):
    # The number of neurons in the input layer of the actor network
    NUMBER_OF_INPUT_NEURONS = 5200

    # The number of neurons in the hidden layer of the actor network
    NUMBER_OF_HIDDEN_NEURONS = 3900

    # The initialization sequence
    def __init__(self, _state_dimension, _action_dimension, _action_bounds, _learning_rate,
                 _target_tracker_coefficient, _batch_size):
        print('[INFO] Actor Initialization: Bringing things up...')
        # Initializing the essential input parameters with the given arguments
        self.state_dimension = _state_dimension
        self.action_dimension = _action_dimension
        self.action_bounds = _action_bounds
        self.learning_rate = _learning_rate
        self.target_tracker_coefficient = _target_tracker_coefficient
        self.batch_size = _batch_size
        # Declaring the internal entities and outputs
        self.model = None
        self.target_model = None
        # The initialization of the actor instance has been completed...

    # Build the Actor Network
    def build(self):
        self.model = tensorflow.keras.Sequential(
            layers=[tensorflow.keras.layers.Dense(units=self.NUMBER_OF_INPUT_NEURONS,
                                                  input_shape=self.state_dimension,
                                                  activation=tensorflow.keras.activations.relu),
                    tensorflow.keras.layers.BatchNormalization(),
                    tensorflow.keras.layers.Dense(units=self.NUMBER_OF_HIDDEN_NEURONS,
                                                  activation=tensorflow.keras.activations.relu),
                    tensorflow.keras.layers.BatchNormalization(),
                    tensorflow.keras.layers.Dense(units=self.action_dimension,
                                                  activation=tensorflow.keras.activations.tanh,
                                                  kernel_initializer='glorot_uniform')
                    ],
            name='Apollo-Actor-v1')
        # Return the model in case an external caller needs this
        return self.model
        # Model-Building has been completed...

    # Train the built model - define the loss, evaluate the gradients, and use the right optimizer...
    def train(self, scaled_outputs, action_gradients):
        # Deep Deterministic Policy Gradient (DDPG)
        unnormalized_actor_gradients = tensorflow.gradients(scaled_outputs,
                                                            self.model.trainable_variables,
                                                            -action_gradients)
        # Normalization w.r.t the batch size - this refers to the expectation operator in the Policy Gradient step
        normalized_actor_gradients = list(map(lambda x: tensorflow.div(x, self.batch_size),
                                              unnormalized_actor_gradients)
                                          )
        # Adam Optimizer
        return tensorflow.train.AdamOptimizer(learning_rate=self.learning_rate).apply_gradients(
            zip(normalized_actor_gradients, self.model.trainable_variables)
        )
        # DDPG-based Actor Network optimization has been completed...

    # Predict actions for either the batch of states sampled from the replay memory OR...
    # ...for the individual state observed from the switch environment
    def predict(self, state_batch):
        return tensorflow.multiply(self.model(state_batch),
                                   self.action_bounds)

    # Soft target update procedure
    def update_targets(self):
        # (\theta \tau) + (\theta' (1 - \tau))
        for i in range(len(self.target_model.trainable_variables)):
            self.target_model.trainable_variables[i].assign(
                tensorflow.multiply(self.model.trainable_variables, self.target_tracker_coefficient) + (
                    tensorflow.multiply(self.target_model.trainable_variables, (1 - self.target_tracker_coefficient))))

    # Predict actions from the target network for target Q-value evaluation during training, i.e. (r_t + \gamma Q(s, a))
    def predict_targets(self, target_state_batch):
        return tensorflow.multiply(self.target_model(target_state_batch),
                                   self.action_bounds)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Actor Termination: Tearing things down...')


# The Actor Network ends here...

# The Critic Network begins here...

# The Critic Network
class Critic(object):
    # The number of neurons in the input layer of the critic network
    NUMBER_OF_INPUT_NEURONS = 5200

    # The number of neurons in the hidden layer of the critic network
    NUMBER_OF_HIDDEN_NEURONS = 3900

    # The initialization sequence
    def __init__(self, _state_dimension, _action_dimension, _learning_rate, _target_tracker_coefficient):
        print('[INFO] Critic Initialization: Bringing things up...')
        # Initializing the input parameters with the given arguments
        self.state_dimension = _state_dimension
        self.action_dimension = _action_dimension
        self.learning_rate = _learning_rate
        self.target_tracker_coefficient = _target_tracker_coefficient
        # Declaring the internal entities and outputs
        self.model = None
        self.target_model = None
        # The initialization sequence has been completed...

    # Build the Critic Network
    # While fitting, it takes in both state and action as inputs...
    def build(self):
        action_hidden_layer = tensorflow.keras.layers.Dense(units=self.NUMBER_OF_HIDDEN_NEURONS,
                                                            input_shape=self.action_dimension)
        self.model = tensorflow.keras.Sequential(
            layers=[tensorflow.keras.layers.Dense(units=self.NUMBER_OF_INPUT_NEURONS,
                                                  activation=tensorflow.keras.activations.relu,
                                                  input_shape=self.state_dimension,
                                                  kernel_initializer='glorot_uniform'),
                    tensorflow.keras.layers.BatchNormalization(),
                    tensorflow.keras.layers.Dense(units=self.NUMBER_OF_HIDDEN_NEURONS)
                    ],
            name='Apollo-Critic-v1')
        action_modification_layer = tensorflow.keras.layers.Multiply(action_hidden_layer.weights)
        state_modification_layer = tensorflow.keras.layers.Multiply(self.model.weights)
        combination_layer = tensorflow.keras.layers.Add(action_modification_layer,
                                                        state_modification_layer,
                                                        action_hidden_layer.bias)
        self.model.add(combination_layer)
        self.model.add(tensorflow.keras.layers.Activation(activation=tensorflow.keras.activations.relu))
        self.model.add(tensorflow.keras.layers.Dense(units=1,
                                                     kernel_initializer='glorot_uniform',
                                                     ))
        # Return the model in case some external caller needs it...
        return self.model
        # Model-Building has been completed...

    # Train the built model, define the loss function, and use the right optimizer...
    def train(self, predicted_q_value, target_q_value):
        loss = tensorflow.keras.losses.mean_squared_error(target_q_value, predicted_q_value)
        # Adam Optimizer
        optimized_result = tensorflow.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimized_result
        # MSE-optimization of the Critic Network has been completed...

    @staticmethod
    # \triangledown_\{a}\ Q(\vec{S}, a)
    def get_action_gradients(predicted_q_value, action):
        return tensorflow.gradients(predicted_q_value, action)

    # Estimate the Q-value for the given state-action pair
    def predict(self, state, action):
        return self.model(state, action)

    # Estimate the Q-value for the state-action pair using the Target Critic Network
    def predict_targets(self, state, action):
        return self.target_model(state, action)

    # Soft target update procedure
    def update_targets(self):
        # (\theta \tau) + (\theta' (1 - \tau))
        for i in range(len(self.target_model.trainable_variables)):
            self.target_model.trainable_variables[i].assign(
                tensorflow.multiply(self.model.trainable_variables, self.target_tracker_coefficient) + (
                    tensorflow.multiply(self.target_model.trainable_variables,
                                        (1 - self.target_tracker_coefficient))))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Critic Termination: Tearing things down...')


# The Critic Network ends here...

# The Prioritized Experiential Replay Memory [Mnemosyne] begins here...

# The Replay Memory with Stochastic Prioritization
# The reason the design incorporates Prioritized Experiential Replay is because I want to leverage te
class Mnemosyne(object):
    # The default capacity of the PER memory
    MEMORY_CAPACITY = 1e6

    # The default prioritization strategy within the PER memory
    PRIORITIZATION_STRATEGY = Prioritization.RANDOM

    # The default random seed (if the prioritization strategy is Prioritization.RANDOM)
    RANDOM_SEED_VALUE = 12345

    # A positive constant that prevents edge-case transitions from being sampled/revisited
    # The default revisitation constraint constant
    REVISITATION_CONSTRAINT_CONSTANT = 1.0

    # The default level of prioritization essential in this PER design [\alpha]
    PRIORITIZATION_LEVEL = 1.0

    # The initialization sequence
    def __init__(self, _memory_capacity, _prioritization_strategy, _revisitation_constraint_constant=None,
                 _prioritization_level=None, _random_seed=None):
        print('[INFO] Mnemosyne Initialization: Bringing things up...')
        # Initialize the input parameters of the PER memory with the given arguments
        # Default to hard-coded values upon invalidation
        self.memory_capacity = (lambda: self.MEMORY_CAPACITY,
                                lambda: _memory_capacity)[_memory_capacity is not None and
                                                          isinstance(_memory_capacity, int) and
                                                          _memory_capacity > 0]()
        self.prioritization_strategy = (lambda: self.PRIORITIZATION_STRATEGY,
                                        lambda: _prioritization_strategy)[_prioritization_strategy is not None and
                                                                          isinstance(_prioritization_strategy,
                                                                                     Prioritization)]()
        # Only for proportional stochastic TD-error prioritization - (|\delta| + \xi,\ \xi > 0)
        if self.prioritization_strategy == Prioritization.STOCHASTIC_PRIORITIZATION_PROPORTIONAL:
            self.revisitation_constraint_constant = (lambda: self.REVISITATION_CONSTRAINT_CONSTANT,
                                                     lambda: _revisitation_constraint_constant)[
                _revisitation_constraint_constant is not None and
                isinstance(_revisitation_constraint_constant, float) and
                _revisitation_constraint_constant > 0]()
        # Only for proportional stochastic TD-error prioritization and rank-based stochastic TD-error prioritization
        if self.prioritization_strategy == Prioritization.STOCHASTIC_PRIORITIZATION_PROPORTIONAL or \
                self.prioritization_strategy == Prioritization.STOCHASTIC_PRIORITIZATION_RANK:
            self.prioritization_level = (lambda: self.PRIORITIZATION_LEVEL,
                                         lambda: _prioritization_level)[_prioritization_level is not None]()
        # A lotta random sampling done here...it's best to initialize the system with a predefined random sample...
        # ...for consistency across the board
        self.random_seed = (lambda: self.RANDOM_SEED_VALUE,
                            lambda: _random_seed)[_random_seed is not None and
                                                  isinstance(_random_seed, int) and
                                                  _random_seed > 0]()
        random.seed(self.random_seed)
        numpy.random.seed(self.random_seed)
        # The memory
        self.memory = deque(maxlen=self.memory_capacity)
        # The replay memory initialization sequence has been completed...

    # Remember the experience (state, action, reward, next_state, td_error)
    def remember(self, state, action, reward, next_state, td_error):
        experience = (state, action, reward, next_state, td_error)
        # Append the experience to the replay memory
        self.memory.append(experience)

    # Sample experiences from the prioritized replay memory according to a stochastic TD-error prioritization strategy
    # Proportional or Rank
    def generate_sample(self, transition_priorities, batch_size):
        # \sum_{i}\ (p_i)^{\alpha}
        denominator = sum([p ** self.prioritization_level for p in transition_priorities])
        # P(i)
        transition_probabilities = [(p ** self.prioritization_level) / denominator for p in transition_priorities]
        sampling_probability = numpy.random.random_sample()
        # Find the #batch_size experiences whose transition probabilities are closest to the sampling_probability
        available_indices = [k for k in range(len(transition_probabilities))]
        sample = []
        for m in range(batch_size):
            pilot_transition_index = min(available_indices,
                                         key=(
                                             lambda idx: abs(transition_probabilities[idx] - sampling_probability)))
            sample.append(self.memory[pilot_transition_index])
            del available_indices[pilot_transition_index]
        return sample

    # Replay the experiences from memory based on the prioritization strategy
    def replay(self, _batch_size):
        batch_size = (lambda: len(self.memory),
                      lambda: _batch_size)[_batch_size <= len(self.memory)]()
        # TD_ERROR_PRIORITIZATION
        if self.prioritization_strategy == Prioritization.TD_ERROR_PRIORITIZATION:
            sample = sorted(self.memory,
                            key=(lambda x: x[5]),
                            reverse=True)[0:batch_size]
        # STOCHASTIC_PRIORITIZATION_PROPORTIONAL
        elif self.prioritization_strategy == Prioritization.STOCHASTIC_PRIORITIZATION_PROPORTIONAL:
            # p_i
            transition_priorities = [numpy.abs(x[5]) + self.revisitation_constraint_constant for x in self.memory]
            sample = self.generate_sample(transition_priorities=transition_priorities,
                                          batch_size=batch_size)
        # STOCHASTIC_PRIORITIZATION_RANK
        elif self.prioritization_strategy == Prioritization.STOCHASTIC_PRIORITIZATION_RANK:
            sorted_memory = sorted(self.memory,
                                   key=(lambda x: numpy.abs(x[5])),
                                   reverse=True)
            # p_i
            transition_priorities = [(1 / (k + 1)) for k in range(len(sorted_memory))]
            sample = self.generate_sample(transition_priorities=transition_priorities,
                                          batch_size=batch_size)
        # Either the prioritization strategy is specified to be RANDOM or...
        # ...the design encountered an invalid/unsupported prioritization strategy
        else:
            sample = random.sample(self.memory, batch_size)
        return numpy.array([k[0] for k in sample]), numpy.array([k[1] for k in sample]), numpy.array(
            [k[2] for k in sample]), numpy.array(k[3] for k in sample)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Mnemosyne Termination: Tearing things down...')


# The Prioritized Experiential Replay Memory [Mnemosyne] ends here...

# The Exploration Noise definition begins here...

class ExplorationNoise(object):
    # The default exploration strategy - exploration factor with temporal decay
    EXPLORATION_STRATEGY = ExplorationStrategy.EXPLORATION_DECAY

    # The default exploration factor initialization - exploration-decay strategy only
    EXPLORATION_FACTOR = 1.0

    # The default exploration decay - exploration-decay strategy only
    EXPLORATION_DECAY = 0.1

    # The default minimum allowed exploration factor - exploration-decay strategy only
    MINIMUM_EXPLORATION_FACTOR = 1e-6

    # The initialization sequence
    def __init__(self, _exploration_strategy, _action_dimension, _exploration_factor=None, _exploration_decay=None,
                 _exploration_factor_min=None, _x0=None, _mu=None, _theta=0.15, _sigma=0.3, _dt=1e-2):
        print('[INFO] ExplorationNoise Initialization: Bringing things up...')
        # Initializing the input parameters with the given arguments...
        self.action_dimension = _action_dimension
        self.exploration_strategy = (lambda: self.EXPLORATION_STRATEGY,
                                     lambda: _exploration_strategy)[_exploration_strategy is not None and
                                                                    isinstance(_exploration_strategy,
                                                                               ExplorationStrategy)]()
        # EXPLORATION_DECAY
        if self.exploration_strategy == ExplorationStrategy.EXPLORATION_DECAY:
            self.exploration_factor = (lambda: self.EXPLORATION_FACTOR,
                                       lambda: _exploration_factor)[_exploration_factor is not None and
                                                                    _exploration_factor > 0]()
            self.exploration_decay = (lambda: self.EXPLORATION_DECAY,
                                      lambda: _exploration_decay)[_exploration_decay is not None and
                                                                  _exploration_decay > 0]()
            self.exploration_factor_min = (lambda: self.MINIMUM_EXPLORATION_FACTOR,
                                           lambda: _exploration_factor_min)[_exploration_factor_min is not None and
                                                                            _exploration_factor_min > 0]()
        # ORNSTEIN_UHLENBECK_NOISE
        if self.exploration_strategy == ExplorationStrategy.ORNSTEIN_UHLENBECK_NOISE:
            self.mu = (lambda: numpy.zeros(self.action_dimension),
                       lambda: _mu)[_mu is not None]()
            self.x_prev = (lambda: numpy.zeros_like(self.mu),
                           lambda: _x0)[_x0 is not None]()
            self.theta = _theta
            self.sigma = _sigma
            self.dt = _dt
        # The initialization sequence for the ExplorationNoise entity has been completed...

    # Return the action appended/modified with the exploration noise
    def execute(self, action):
        # EXPLORATION_DECAY
        if self.exploration_strategy == ExplorationStrategy.EXPLORATION_DECAY:
            if numpy.random.rand() <= self.exploration_factor:
                return numpy.random.random_sample(self.action_dimension)
            return action
        # ORNSTEIN_UHLENBECK_NOISE
        if self.exploration_strategy == ExplorationStrategy.ORNSTEIN_UHLENBECK_NOISE:
            noise = self.generate_noise()
            return action + noise

    # A simple utility method to decay the exploration factor
    def decay(self):
        self.exploration_factor *= self.exploration_decay

    # Generate Ornstein-Uhlenbeck Noise
    # The exploration policy involves adding this generated Ornstein-Uhlenbeck noise
    # Vasicek Model - \[dX_t = \theta (\mu - X_t) dt + \sigma dW_t\] - W_t is a Wiener Process
    # Simplified Model: \[X_{t+\alpha} = X_t + (\theta (\mu - X_t) dt) + (\sigma \sqrt(dt) \mathcal{N}(0, 1))\]
    def generate_noise(self):
        # Wiener process W_t has both Independent Increments and Gaussian Increments
        x = self.x_prev + (self.theta * (self.mu - self.x_prev) * self.dt) + (
                self.sigma * numpy.sqrt(self.dt) * numpy.random.normal(self.mu.shape))
        self.x_prev = x
        return x

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] ExplorationNoise Termination: Tearing things down...')


# The Exploration Noise definition ends here...

# The integrated RL-agent [Apollo] begins here...

# The RL-agent defined in this class controls the
class Apollo(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] Apollo Initialization: Bringing things up...')

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Apollo Termination: Tearing things down...')

# The integrated RL-agent [Apollo] ends here...
