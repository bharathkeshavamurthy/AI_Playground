# |Bleeding-Edge Productions|
# This entity describes the design of an intelligent, adaptive buffer allocation engine using Deep Deterministic...
# ...Policy Gradients (DDPG) in an Asynchronous Advantage Actor Critic (A3C) architecture based on the Double Deep...
# ...Q-Networks Prioritized Experiential Learning (DDQN-PER) framework.
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems, Inc.
# Copyright (c) 2019. All Rights Reserved.

# TODO: A logging framework instead of all these formatted print statements...

# TODO: A3C:This implementation does not currently make use of multiple workers - Can multiple workers be implemented?

# The imports
import math
import enum
import numpy
import random
import traceback
import threading
import tensorflow
from collections import namedtuple, deque

# Enable eager execution for converting tensors into numpy arrays and vice versa
tensorflow.enable_eager_execution()

# Mutexes for various threads to access shared objects

# The shared environment mutex
caerus = threading.Lock()

# Named-Tuples are cleaner and more readable...

# Environmental Feedback API entity - global instance
FEEDBACK = namedtuple('feedback',
                      ['reward',
                       'next_state'])

# The environment details are encapsulated in this namedtuple - global instance
ENVIRONMENT_DETAILS = namedtuple('environment_details',
                                 ['number_of_ports',
                                  'number_of_queues_per_port',
                                  'global_pool_size',
                                  'dedicated_pool_size_per_port'])

# The actor network design details are encapsulated in this namedtuple - global instance
ACTOR_DESIGN_DETAILS = namedtuple('actor_design_details',
                                  ['learning_rate',
                                   'target_tracker_coefficient',
                                   'batch_size',
                                   ])

# The critic network design details are encapsulated in this namedtuple - global instance
CRITIC_DESIGN_DETAILS = namedtuple('critic_design_details',
                                   ['learning_rate',
                                    'target_tracker_coefficient'])

# The Prioritized Experiential Replay (PER) memory details are encapsulated in this namedtuple - global instance
REPLAY_MEMORY_DETAILS = namedtuple('replay_memory_details',
                                   ['memory_capacity',
                                    'prioritization_strategy',
                                    'revisitation_constraint_constant',
                                    'prioritization_level',
                                    'random_seed'])

# The exploration strategy details are encapsulated in this namedtuple - global instance
EXPLORATION_STRATEGY_DETAILS = namedtuple('exploration_strategy_details',
                                          ['exploration_strategy',
                                           'action_dimension',
                                           'exploration_factor',
                                           'exploration_decay',
                                           'exploration_factor_min',
                                           'x0',
                                           'mu',
                                           'theta',
                                           'sigma',
                                           'dt'])


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

# Utilities crucial to the design start here...

# A Utilities entity consisting of design-critical validation and/or parsing routines
class Utilities(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] Utilities Initialization: Bringing things up...')
        # No specific bring-up here...

    # A namedtuple (ref) instance validation
    @staticmethod
    def custom_instance_validation(_obj, _ref):
        _type = type(_obj)
        _base = _type.__bases__
        if len(_base) == 1 and _base[0] == tuple:
            _fields = getattr(_type, '_fields', None)
            _ref_fields = getattr(_ref, '_fields', None)
            if isinstance(_fields, tuple) and _fields == _ref_fields:
                return True
        return False

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Utilities Termination: Tearing things down...')
        # No specific tear-down here...


# Utilities crucial to the design end here...

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

    # The penalty multiplier for invalid transitions, i.e. invalid actions
    INCOMPETENCE_PENALTY_MULTIPLIER = 5.0

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

    # The service rate of high-priority queues - Nexus design parameter
    HIGH_PRIORITY_QUEUE_SERVICE_RATE = 40

    # The service rate of low-priority queues - Nexus design parameter
    LOW_PRIORITY_QUEUE_SERVICE_RATE = 8

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
            _dedicated_pool_size_per_port > 0]()
        # Initialize the environment
        self.state = self.start()
        # The allowed action skeleton - the switch will have an API to validate and execute compliant actions
        # The initial compliant action is [ [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0] ]
        self.action_skeleton = [[((port * 0) + (queue * 0)) for queue in range(self.number_of_queues_per_port + 1)]
                                for port in self.number_of_ports]
        self.action_skeleton.append([0])
        # Successful initialization
        # This seems unnecessary - maybe we can find a use for this later...
        self.status = Status.SUCCESS
        # A flag which indicated gross incompetence exhibited by the RL agent
        # If this flag is set to true, impose sky-high tariffs...
        self.incompetence = False
        # A termination flag to indicate that the switch has been shut down
        self.shutdown = False
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
                                         # The first queue in every port is designated a High-Priority queue
                                         # High-Priority queues have higher service rates and hence have smaller buffers
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
        # This is a high-level Nexus state representation
        # This is not the state tensor - conversion to state tensor done in the get_state_tensor() routine...
        return self.STATE(ports=ports,
                          leftover_buffer_units_in_the_global_pool=self.global_pool_size)
        # The switch environment state initialization has been completed...

    # Get the current state [high-level] of the switch environment
    def get_state(self):
        # A simple getter method for external callers
        return self.state

    # Construct a tensor from the high-level Nexus state and pass it down to the calling routine for...
    # ...analysis/input into the NNs
    # As far as switch design is concerned, we don't need caerus here because external callers should be responsible...
    # ...for mutex acquisitions and releases...
    def get_state_tensor(self):
        ports = []
        try:
            # The ports loop - p
            for port in range(self.number_of_ports):
                queues = []
                # The queues loop - q
                for queue in range(self.number_of_queues_per_port):
                    queue_state = self.state.ports[port].queues[queue]
                    queues.append([queue_state.required_minimum_capacity,
                                   queue_state.allowed_maximum_buffer_capacity,
                                   queue_state.allocated_buffer_units,
                                   queue_state.packet_drop_count])
                # Create an iterable instance of queue tensors specific to port 'p' for stacking later in the routine
                ports.append(tensorflow.convert_to_tensor(
                    numpy.append(tensorflow.constant(queues,
                                                     dtype=tensorflow.int32).numpy(),
                                 self.state.ports[port].leftover_buffer_units_in_the_dedicated_pool),
                    dtype=tensorflow.int32))
            # Stack the port-specific tensors, Append the global pool size (leftover: state-specific), ...
            # ...and return the resultant state tensor
            return tensorflow.convert_to_tensor(
                numpy.append(tensorflow.stack(ports, axis=0).numpy(),
                             self.state.leftover_buffer_units_in_the_global_pool),
                dtype=tensorflow.int32)
        except Exception as exception:
            print('[ERROR] Nexus get_state_tensor: Exception caught while formatting the state of the switch - '
                  '[{}]'.format(exception))
            traceback.print_tb(exception.__traceback__)
        return None

    # In order to get the action dimension, assume an initial policy of NOP across all queues, ports, and pools.
    # Return the dimensions of the action - output of Apollo
    def get_action_dimension(self):
        ports = []
        try:
            # The ports loop - p
            for port in range(self.number_of_ports):
                # NOPs w.r.t the queues of port 'p' and the dedicated pool of port 'p'
                ports.append([k - k for k in range(self.number_of_queues_per_port + 1)])
            # Append an NOP w.r.t the global pool of Nexus and return the dimensions of the resultant tensor
            return tensorflow.convert_to_tensor(
                numpy.append(tensorflow.constant(ports, axis=0).numpy(),
                             0),
                dtype=tensorflow.int32).shape
        except Exception as exception:
            print('[ERROR] Nexus get_action_dimension: Exception caught while determining the '
                  'compliant action dimension - [{}]'.format(exception))
            traceback.print_tb(exception.__traceback__)
        return None

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
            return self.INCOMPETENCE_PENALTY_MULTIPLIER * reward
        return reward

    # Transition from the current state to the next state and validate the transition
    # Return <reward, next_state>
    def execute(self, action):
        print('[DEBUG] Nexus transition: Transitioning the underlying MDP...')
        # TODO: Do we need a structural validation even after the actor getting a compliant action from ...
        #  ...Nexus before execution?
        # # Initial structural validation of the action
        # internal_length_operator = numpy.vectorize(len)
        # if len(action) != self.number_of_ports or \
        #         sum(internal_length_operator(action)) != (self.number_of_ports * self.number_of_queues_per_port):
        #     print('[ERROR] Nexus validate_action: Non-Compliant Action received from the recommendation system - '
        #           '{}'.format(str(action)))
        #     return False
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

    # Shutdown the system
    def initiate_shutdown(self):
        self.shutdown = True
        # Normal exit sequence
        self.__exit__(None, None, None)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Nexus Termination: Tearing things down...')


# The switch environment [Nexus] ends here...

# The Actor Network begins here...

# TODO: Do I need to use action-bounds or will a critical negative reward strategy encapsulated in the...
#  ...Nexus.validate() method be sufficient to incentivise the Actor to generate compliant actions...

# The Actor Network
class Actor(object):
    # The number of neurons in the input layer of the actor network
    NUMBER_OF_INPUT_NEURONS = 5200

    # The number of neurons in the hidden layer of the actor network
    NUMBER_OF_HIDDEN_NEURONS = 3900

    # The initialization sequence
    def __init__(self, _state_dimension, _action_dimension, _learning_rate, _target_tracker_coefficient, _batch_size):
        print('[INFO] Actor Initialization: Bringing things up...')
        # Initializing the essential input parameters with the given arguments
        self.state_dimension = _state_dimension
        self.action_dimension = _action_dimension
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
        normalized_actor_gradients = list(map(lambda x: tensorflow.div(x,
                                                                       self.batch_size),
                                              unnormalized_actor_gradients))
        # Adam Optimizer
        return tensorflow.train.AdamOptimizer(learning_rate=self.learning_rate).apply_gradients(
            zip(normalized_actor_gradients,
                self.model.trainable_variables))
        # DDPG-based Actor Network optimization has been completed...

    # Predict actions for either the batch of states sampled from the replay memory OR...
    # ...for the individual state observed from the switch environment
    def predict(self, state_batch):
        # TODO: Scaling here, if necessary
        return self.model(state_batch)

    # Soft target update procedure
    def update_targets(self):
        # (\theta \tau) + (\theta' (1 - \tau))
        for i in range(len(self.target_model.trainable_variables)):
            self.target_model.trainable_variables[i].assign(
                tensorflow.multiply(self.model.trainable_variables, self.target_tracker_coefficient) + (
                    tensorflow.multiply(self.target_model.trainable_variables, (1 - self.target_tracker_coefficient))))

    # Predict actions from the target network for target Q-value evaluation during training, i.e. (r_t + \gamma Q(s, a))
    def predict_targets(self, target_state_batch):
        # TODO: Scaling here, if necessary
        return self.target_model(target_state_batch)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Actor Termination: Tearing things down...')


# The Actor Network ends here...

# The Critic Network begins here...

# TODO: Is there a better way to handle the dual-input (state and action) model building strategy...

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

# The Replay Memory with Stochastic Prioritization / Uniform Random Sampling
class Mnemosyne(object):
    # The default capacity of the PER memory
    MEMORY_CAPACITY = 1e6

    # The default prioritization strategy within the PER memory
    PRIORITIZATION_STRATEGY = Prioritization.RANDOM

    # The default random seed (if the prioritization strategy is Prioritization.RANDOM)
    RANDOM_SEED_VALUE = 666

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
        # TODO: Is a deque the correct data structure to use here?
        self.memory = deque(maxlen=self.memory_capacity)
        # The replay memory initialization sequence has been completed...

    # Remember the experience (state, action, reward, next_state, td_error)
    def remember(self, state, action, reward, next_state, td_error):
        experience = (state, action, reward, next_state, td_error)
        # Append the experience to the replay memory
        self.memory.append(experience)

    # Sample experiences from the prioritized replay memory according to a Stochastic TD-error Prioritization strategy
    # ...(Proportional or Rank) or a Random Sampling strategy
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
            [k[2] for k in sample]), numpy.array(k[3] for k in sample), numpy.array(k[4] for k in sample)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Mnemosyne Termination: Tearing things down...')


# The Prioritized Experiential Replay Memory [Mnemosyne] ends here...

# The Exploration Noise [Artemis] definition begins here...

class Artemis(object):
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
        print('[INFO] Artemis Initialization: Bringing things up...')
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
        # The initialization sequence for the Artemis entity has been completed...

    # Return the action appended/modified with the exploration noise
    def execute(self, action):
        # EXPLORATION_DECAY
        if self.exploration_strategy == ExplorationStrategy.EXPLORATION_DECAY:
            # Decay the exploration factor and employ the well-known $\epsilon-greedy$ logic
            self.decay()
            if numpy.random.rand() <= self.exploration_factor:
                return numpy.random.random_sample(self.action_dimension)
            return action
        # ORNSTEIN_UHLENBECK_NOISE
        if self.exploration_strategy == ExplorationStrategy.ORNSTEIN_UHLENBECK_NOISE:
            noise = self.generate_noise()
            # Assuming the action is a row vector, we add the generated Ornstein-Uhlenbeck noise to each action entry
            # TODO: If the action is a tensor of a different shape, change this exploration noise addition logic...
            return [(k + noise) for k in action]

    # A simple utility method to decay the exploration factor
    def decay(self):
        self.exploration_factor *= self.exploration_decay

    # Generate Ornstein-Uhlenbeck Noise
    # The exploration policy involves adding this generated Ornstein-Uhlenbeck noise
    # Vasicek Model - \[dX_t = \theta (\mu - X_t) dt + \sigma dW_t\]: W_t is a Wiener Process [Independent Increments...
    # ...and Gaussian Increments]
    # Simplified Model: \[X_{t+\alpha} = X_t + (\theta (\mu - X_t) dt) + (\sigma \sqrt(dt) \mathcal{N}(0, 1))\]
    def generate_noise(self):
        # Wiener process W_t has both Independent Increments and Gaussian Increments
        x = self.x_prev + (self.theta * (self.mu - self.x_prev) * self.dt) + (
                self.sigma * numpy.sqrt(self.dt) * numpy.random.normal(self.mu.shape))
        self.x_prev = x
        return x

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Artemis Termination: Tearing things down...')


# The Exploration Noise [Artemis] definition ends here...

# The integrated RL-agent [Apollo] begins here...

# The RL-agent defined in this class controls the Actor-Critic DDQN-PER framework
class Apollo(object):
    # The batch size for sampling from the prioritized experiential replay memory
    BATCH_SIZE = 64

    # The default discount factor employed in the target Q-value estimation within the Critic
    DISCOUNT_FACTOR = 0.9

    # The default maximum number of iterations per episode
    ITERATIONS_PER_EPISODE = 1e3

    # The default maximum number of episodes
    MAXIMUM_NUMBER_OF_EPISODES = 1e5

    # The initialization sequence
    def __init__(self, _environment_details, _actor_design_details, _critic_design_details,
                 _replay_memory_details, _exploration_strategy_details, _batch_size, _discount_factor,
                 _iterations_per_episode, _maximum_number_of_episodes):
        print('[INFO] Apollo Initialization: Bringing things up...')
        # The status indicator flag
        self.status = Status.SUCCESS
        # Initializing the relevant members - default to hard-coded values upon invalidation
        self.utilities = Utilities()
        self.environment_details = (lambda: ENVIRONMENT_DETAILS(number_of_ports=None,
                                                                number_of_queues_per_port=None,
                                                                global_pool_size=None,
                                                                dedicated_pool_size_per_port=None),
                                    lambda: _environment_details)[_environment_details is not None and
                                                                  self.utilities.custom_instance_validation(
                                                                      _environment_details,
                                                                      ENVIRONMENT_DETAILS)]()
        self.actor_design_details = (lambda: ACTOR_DESIGN_DETAILS(learning_rate=None,
                                                                  target_tracker_coefficient=None,
                                                                  batch_size=None),
                                     lambda: _actor_design_details)[_actor_design_details is not None and
                                                                    self.utilities.custom_instance_validation(
                                                                        _actor_design_details,
                                                                        ACTOR_DESIGN_DETAILS)]()
        self.critic_design_details = (lambda: CRITIC_DESIGN_DETAILS(learning_rate=None,
                                                                    target_tracker_coefficient=None),
                                      lambda: _critic_design_details)[_critic_design_details is not None and
                                                                      self.utilities.custom_instance_validation(
                                                                          _critic_design_details,
                                                                          CRITIC_DESIGN_DETAILS)]()
        self.replay_memory_details = (lambda: REPLAY_MEMORY_DETAILS(memory_capacity=None,
                                                                    prioritization_strategy=None,
                                                                    revisitation_constraint_constant=None,
                                                                    prioritization_level=None,
                                                                    random_seed=None),
                                      lambda: _replay_memory_details)[_replay_memory_details is not None and
                                                                      self.utilities.custom_instance_validation(
                                                                          _replay_memory_details,
                                                                          REPLAY_MEMORY_DETAILS)]()
        self.exploration_strategy_details = (lambda: EXPLORATION_STRATEGY_DETAILS(exploration_strategy=None,
                                                                                  action_dimension=None,
                                                                                  exploration_factor=None,
                                                                                  exploration_decay=None,
                                                                                  exploration_factor_min=None,
                                                                                  x0=None,
                                                                                  mu=None,
                                                                                  theta=None,
                                                                                  sigma=None,
                                                                                  dt=None),
                                             lambda: _exploration_strategy_details)[
            _exploration_strategy_details is not None and
            self.utilities.custom_instance_validation(_exploration_strategy_details,
                                                      EXPLORATION_STRATEGY_DETAILS)]()
        self.batch_size = (lambda: self.BATCH_SIZE,
                           lambda: _batch_size)[_batch_size is not None and
                                                isinstance(_batch_size, int) and
                                                _batch_size > 0]()
        self.discount_factor = (lambda: self.DISCOUNT_FACTOR,
                                lambda: _discount_factor)[_discount_factor is not None and
                                                          0.5 < _discount_factor < 1]()
        self.iterations_per_episode = (lambda: self.ITERATIONS_PER_EPISODE,
                                       lambda: _iterations_per_episode)[_iterations_per_episode is not None and
                                                                        isinstance(_iterations_per_episode, int) and
                                                                        _iterations_per_episode > 0]()
        self.maximum_number_of_episodes = (lambda: self.MAXIMUM_NUMBER_OF_EPISODES,
                                           lambda: _maximum_number_of_episodes)[
            _maximum_number_of_episodes is not None and
            isinstance(_maximum_number_of_episodes, int) and
            _maximum_number_of_episodes > 0]()
        # Create the Nexus switch environment
        self.nexus = Nexus(self.environment_details.number_of_ports,
                           self.environment_details.number_of_queues_per_port,
                           self.environment_details.global_pool_size,
                           self.environment_details.dedicated_pool_size_per_port)
        # Get the switch environment details from the created Nexus instance
        # NOTE: I don't need a mutex acquisition and release strategy in the initialization routine of Apollo
        self.state_dimension = self.nexus.get_state_tensor().shape
        self.action_dimension = self.nexus.get_action_dimension()
        # State and Action validation check
        if self.state_dimension is None or self.action_dimension is None:
            print('[ERROR] Apollo Initialization: Something went wrong while obtaining the state and compliant action '
                  'information from Nexus. Please refer to the earlier logs for more details on this error.')
            self.status = Status.FAILURE
        # Create the Actor and Critic Networks
        self.actor = Actor(self.state_dimension,
                           self.action_dimension,
                           self.actor_design_details.learning_rate,
                           self.actor_design_details.target_tracker_coefficient,
                           self.actor_design_details.batch_size)
        self.critic = Critic(self.state_dimension,
                             self.action_dimension,
                             self.critic_design_details.learning_rate,
                             self.critic_design_details.target_tracker_coefficient)
        # Initialize the Prioritized Experiential Replay Memory
        self.mnemosyne = Mnemosyne(self.replay_memory_details.memory_capacity,
                                   self.replay_memory_details.prioritization_strategy,
                                   self.replay_memory_details.revisitation_constraint_constant,
                                   self.replay_memory_details.prioritization_level,
                                   self.replay_memory_details.random_seed)
        # Initialize the Exploration Noise Generator
        self.artemis = Artemis(self.exploration_strategy_details.exploration_strategy,
                               self.exploration_strategy_details.action_dimension,
                               self.exploration_strategy_details.exploration_factor,
                               self.exploration_strategy_details.exploration_decay,
                               self.exploration_strategy_details.exploration_factor_min,
                               self.exploration_strategy_details.x0,
                               self.exploration_strategy_details.mu,
                               self.exploration_strategy_details.theta,
                               self.exploration_strategy_details.sigma,
                               self.exploration_strategy_details.dt)
        # The initialization sequence has been completed

    # Start the interaction with the environment and the training process
    def start(self):
        print('[INFO] Apollo train: Interacting with the switch environment and initiating the training process')
        try:
            # Build the Actor and Critic Networks
            self.actor.build()
            self.critic.build()
            # Start the interaction with Nexus
            for episode in range(self.maximum_number_of_episodes):
                for iteration in range(self.iterations_per_episode):
                    # Initialize/Re-Train/Update the target networks in this off-policy DDQN-architecture
                    self.actor.update_targets()
                    self.critic.update_targets()
                    # Observe the state, execute an action, and get the feedback from the switch environment
                    # Automatic next_state transition fed in by using the Nexus instance
                    # Transition and validation is encapsulated within Nexus
                    # Mutex acquisition
                    caerus.acquire()
                    state = self.nexus.get_state_tensor()
                    action = self.artemis.execute(self.actor.predict(state))
                    feedback = self.nexus.execute(action)
                    # Mutex release
                    caerus.release()
                    # Validation - exit if invalid
                    if feedback is None or self.utilities.custom_instance_validation(feedback,
                                                                                     FEEDBACK) is False:
                        print('[ERROR] Apollo train: Invalid feedback received from the environment. '
                              'Please check the compatibility between Apollo and the Nexus variant')
                        return False
                    # Find the target Q-value, the predicted Q-value, and subsequently the TD-error
                    target_q = feedback.reward + (self.discount_factor * self.critic.predict_targets(
                        feedback.next_state,
                        self.actor.predict_targets(feedback.next_state)))
                    predicted_q = self.critic.predict(state,
                                                      action)
                    td_error = predicted_q - target_q
                    # Remember this experience
                    self.mnemosyne.remember(state,
                                            action,
                                            feedback.reward,
                                            feedback.next_state,
                                            td_error)
                    # Start the replay sequence for training
                    if len(self.mnemosyne.memory) >= self.batch_size:
                        # Prioritization strategy specific replay
                        s_batch, a_batch, r_batch, s2_batch, td_error_batch = self.mnemosyne.replay(self.batch_size)
                        target_q = self.critic.predict_targets(s2_batch,
                                                               self.actor.predict_targets(s2_batch))
                        target_q_values = []
                        for k in range(self.batch_size):
                            target_q_values.append(r_batch[k] + (self.discount_factor * target_q[k]))
                        # Train the Critic - standard MSE optimization
                        self.critic.train(self.critic.predict(s_batch,
                                                              a_batch),
                                          numpy.reshape(target_q_values,
                                                        newshape=(1,
                                                                  self.batch_size)))
                        # Get the action gradients for DDPG
                        action_gradients = self.critic.get_action_gradients(s_batch,
                                                                            self.actor.predict(s_batch))
                        # Train the Actor - DDPG
                        self.actor.train(self.actor.predict(s_batch),
                                         action_gradients[0])
        except Exception as exception:
            print('[ERROR] Apollo train: Exception caught while interacting with the Nexus switch environment and '
                  'training the Actor-Critic DDQN-PER framework - [{}]'.format(exception))
            traceback.print_tb(exception.__traceback__)
        finally:
            # Success or Failure: Episodic analysis has been completed - Send a command to the switch to shut down
            # Mutex acquisition
            caerus.acquire()
            self.nexus.initiate_shutdown()
            # Mutex release
            caerus.release()
        # Environment Interaction and Training has been completed...

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Apollo Termination: Tearing things down...')


# The integrated RL-agent [Apollo] ends here...

# The test entity [Ares] begins here...

# This class models the event arrival and service process at each queue in the switch
# This entity also serves as a monitoring API - monitoring the arrival times and departure times of packets,...
# ...the number of packets pending in the queueing system, and the number of packets dropped at each queue.
class Ares(object):

    # The initialization sequence
    def __init__(self, _nexus):
        print('[INFO] Ares Initialization: Bringing things up...')
        # Setting up the switch environment and re-assigning input parameters
        self.nexus = _nexus
        # Assign arrival rates to each individual queue
        self.arrival_rates = []
        for p in range(self.nexus.number_of_ports):
            port_specific_arrival_rates = []
            for q in range(self.nexus.number_of_queues_per_port):
                if self.nexus.get_state().ports[p].queues[q].priority == Priority.HIGH_PRIORITY:
                    # A slightly higher arrival rate for high-priority queues
                    port_specific_arrival_rates.append(random.randrange(1, 16, 0.1))
                else:
                    # A lower arrival rate for low-priority queues
                    port_specific_arrival_rates.append(random.randrange(1, 8, 0.1))
            self.arrival_rates.append(port_specific_arrival_rates)
        # Assign service rates to each individual queue
        self.service_rates = []
        for p in range(self.nexus.number_of_ports):
            port_specific_service_rates = []
            for q in range(self.nexus.number_of_queues_per_port):
                if self.nexus.get_state().ports[p].queues[q].priority == Priority.HIGH_PRIORITY:
                    port_specific_service_rates.append(self.nexus.HIGH_PRIORITY_QUEUE_SERVICE_RATE)
                else:
                    port_specific_service_rates.append(self.nexus.LOW_PRIORITY_QUEUE_SERVICE_RATE)
            self.service_rates.append(port_specific_service_rates)
        # Initializing the pending number of packets in each queueing system
        self.pending_packets = [[((0 * p) + (0 * q)) for q in range(self.nexus.number_of_queues_per_port)]
                                for p in range(self.nexus.number_of_ports)]
        # The initialization sequence has been completed...

    # Simulate a Poisson arrival process, an Exponential service process
    # Extract the number of available buffer spaces in each queue
    # Evaluate the number of simulated packets that would be dropped
    # Populate the "packet_drop_count" field per queue per port in the environment's state variable
    def start(self):
        print('[INFO] Ares start: Acquiring the mutex and analyzing the state of the switch environment...')
        caerus.acquire()
        try:
            # The switch should be up and running
            if self.nexus.shutdown is False:
                for p in range(self.nexus.number_of_ports):
                    for q in range(self.nexus.number_of_queues_per_port):
                        # Model the arrival process
                        # Don't set a seed here - let's analyze the operation across multiple inconsistent runs...
                        # TODO: Do we need a see here...do we need consistency?
                        arrival_count = 0
                        arrival_times = []
                        # Inverse Transform: Generation of exponential inter-arrival times from...
                        # ...a uniform random variable
                        arrival_time = (-numpy.log(numpy.random.random_sample())) / self.arrival_rates[p][q]
                        # Per second arrival rate - this is fixed (\lambda_{pq} is the arrival rate [per second])
                        while arrival_time <= 1.0:
                            arrival_count += 1
                            arrival_times.append(arrival_time)
                            arrival_time -= (-numpy.log(numpy.random.random_sample())) / self.arrival_rates[p][q]
                        pending = self.pending_packets[p][q] + len(arrival_times)
                        # Model the service process
                        # Don't set a seed here - let's analyze the operation across multiple inconsistent runs...
                        # TODO: Do we need a see here...do we need consistency?
                        service_times = (-numpy.log(numpy.random.random(1, pending))) / self.service_rates[p][q]
                        localized_run_time = 0
                        for packet in range(pending):
                            localized_run_time += service_times[packet]
                            if localized_run_time <= 1.0:
                                pending -= 1
                            else:
                                break
                        dropped = 0
                        allocated = self.nexus.get_state().ports[p].queues[q].allocated_buffer_units
                        # Update the pending count based on the current allocation
                        if allocated < pending:
                            dropped = pending - allocated
                            self.pending_packets[p][q] = allocated
                        else:
                            self.pending_packets[p][q] = pending
                        # Update the drop count
                        self.nexus.get_state().ports[p].queues[q].packet_drop_count = dropped
        except Exception as exception:
            print('[ERROR] Ares start: Exception caught while simulating packet arrival and '
                  'analyzing switch performance - [{}]'.format(exception))
            traceback.print_tb(exception.__traceback__)
        finally:
            # Release the mutex
            caerus.release()
        # Ares analysis has been completed...

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Ares Termination: Tearing things down...')


# The test entity [Ares] ends here...

# The AdaptiveBufferIntelligence main routine begins here...
if __name__ == '__main__':
    print('[INFO] AdaptiveBufferIntelligence Trigger: Starting system assessment...')
    # Nexus Design
    number_of_ports = 3
    number_of_queues_per_port = 3
    dedicated_pool_size_per_port = 40
    global_pool_size = 120
    nexus = Nexus(number_of_ports,
                  number_of_queues_per_port,
                  global_pool_size,
                  dedicated_pool_size_per_port)
    action_dimension = nexus.get_action_dimension()
    if action_dimension is None:
        print('[ERROR] AdaptiveBufferIntelligence Trigger: Something went wrong while obtaining the action dimension'
              'from Nexus. Please refer to the earlier logs for more details on this error. Exiting!')
        raise SystemExit
    environment_details = ENVIRONMENT_DETAILS(number_of_ports=number_of_ports,
                                              number_of_queues_per_port=number_of_queues_per_port,
                                              global_pool_size=global_pool_size,
                                              dedicated_pool_size_per_port=dedicated_pool_size_per_port)
    # Actor Design
    actor_design_details = ACTOR_DESIGN_DETAILS(learning_rate=1e-4,
                                                target_tracker_coefficient=0.01,
                                                batch_size=64)
    # Critic Design
    critic_design_details = CRITIC_DESIGN_DETAILS(learning_rate=1e-5,
                                                  target_tracker_coefficient=0.01)
    # Replay Memory Design
    replay_memory_design_details = REPLAY_MEMORY_DETAILS(memory_capacity=1e9,
                                                         # Random Sampling based replay strategy
                                                         prioritization_strategy=Prioritization.RANDOM,
                                                         revisitation_constraint_constant=None,
                                                         prioritization_level=None,
                                                         random_seed=666)
    # Exploration Strategy Design
    exploration_strategy_design_details = EXPLORATION_STRATEGY_DETAILS(
        # We're actually using an Exploration Decay based exploration strategy here...
        exploration_strategy=ExplorationStrategy.EXPLORATION_DECAY,
        action_dimension=action_dimension,
        exploration_factor=None,
        exploration_decay=None,
        exploration_factor_min=None,
        # Ornstein-Uhlenbeck Exploration Noise: Populated the default parameters even though they're not essential...
        # ...in this use case
        x0=None,
        mu=numpy.zeros(action_dimension),
        theta=0.15,
        sigma=0.3,
        dt=1e-2)
    # Batch Size / Batch Area
    batch_area = 64
    # Discount Factor
    discount_factor = 0.9
    # Iterations per Episode
    iterations_per_episode = 1e3
    # Maximum Number of Episodes
    maximum_number_of_episodes = 1e5
    # Create a timer thread for Ares initialized with Nexus and start the evaluation thread
    # Additionally, create an instance of Apollo and start that simultaneously
    apollo = Apollo(environment_details, actor_design_details, critic_design_details, replay_memory_design_details,
                    exploration_strategy_design_details, batch_area, discount_factor, iterations_per_episode,
                    maximum_number_of_episodes)
    if apollo.status == Status.FAILURE:
        print('[ERROR] AdaptiveBufferIntelligence Trigger: Something went wrong during the initialization of Apollo. '
              'Please refer to the earlier logs for more information on this error.')
        raise SystemExit
    ares = Ares(nexus)
    # The timer interval for Cronus controlling Ares - fix this to 1.0
    timer_interval = 1.0
    # Create individual threads for Ares (cronus_thread) and Apollo (apollo_thread)
    cronus_thread = threading.Timer(timer_interval, ares.start)
    apollo_thread = threading.Thread(target=apollo.start)
    print('[INFO] AdaptiveBufferIntelligence Trigger: Starting Ares... [cronus_thread]')
    # Start the Ares and Apollo threads
    cronus_thread.start()
    print('[INFO] AdaptiveBufferIntelligence Trigger: Starting Apollo... [apollo_thread]')
    apollo_thread.start()
    print('[INFO] AdaptiveBufferIntelligence Trigger: Joining all spawned threads...')
    # Join the threads upon completion - Integrate with the [main] thread
    cronus_thread.join()
    apollo_thread.join()
    print('[INFO] AdaptiveBufferIntelligence Trigger: Completed system assessment...')
    # AdaptiveBufferIntelligence system assessment has been completed...
