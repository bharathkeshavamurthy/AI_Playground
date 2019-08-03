# |Bleeding-Edge Productions|
# This entity describes the design of an intelligent, adaptive buffer allocation engine using Deep Deterministic...
# ...Policy Gradients (DDPG) in an Asynchronous Advantage Actor Critic (A3C) architecture based on the Double Deep...
# ...Q-Networks Prioritized Experiential Learning (DDQN-PER) framework.
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems, Inc.
# Copyright (c) 2019. All Rights Reserved.

# TODO: A logging framework instead of all these formatted print statements...

# TODO: A3C: This implementation does not currently make use of multiple workers - Can multiple workers be implemented?

# The imports
import math
import time
import numpy
import random
import traceback
import threading
import tensorflow
from enum import Enum
from recordclass import recordclass
from collections import deque, namedtuple

# from tensorflow.python.keras.backend import set_session

# Enable eager execution for converting tensors into numpy arrays and vice versa
# tensorflow.enable_eager_execution()

# TODO: Why isn't TensorFlow session able to work with keras layers?
#  Is it a compatibility issue with different implementations of Keras or
#  Is it because the sessions are not thread-safe?

# A global thread-safe tensorflow session
# Change Log: A thread-safe global reference is no longer needed because Apollo is now part of the main thread
# global_tensorflow_session = tensorflow.Session()
# set_session(global_tensorflow_session)

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
class Status(Enum):
    # The module is in a correct working state
    SUCCESS = 0

    # The module is not up - some modular operation failed
    FAILURE = 1


# An extensible enumeration entity listing the possible priority types in the design
class Priority(Enum):
    # Systems with higher service rates
    HIGH_PRIORITY = 0

    # Systems with comparatively lower service rates
    LOW_PRIORITY = 1


# An extensible enumeration entity listing the possible prioritization techniques employed in Mnemosyne
class Prioritization(Enum):
    # Prioritization using TD-error
    TD_ERROR_PRIORITIZATION = 0

    # Stochastic sampling using proportional TD-error
    STOCHASTIC_PRIORITIZATION_PROPORTIONAL = 1

    # Stochastic sampling using rank of a transition within the replay memory
    STOCHASTIC_PRIORITIZATION_RANK = 2

    # Purely random sampling strategy
    RANDOM = 3


# An extensible enumeration entity listing the possible exploration strategies employed in the RL agent
class ExplorationStrategy(Enum):
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
    INCOMPETENCE_PENALTY_MULTIPLIER = 100

    # The penalty additive for invalid transitions, i.e. invalid actions
    INCOMPETENCE_PENALTY_ADDITIVE = -100

    # The initial heavy penalty for recommending non-compliant actions
    # INITIAL_INCOMPETENCE_PENALTY = -1000

    # The members of the queue instance for multiplier evaluation
    QUEUE_INSTANCE_MEMBERS = ['queue_identifier',
                              'priority',
                              'required_minimum_capacity',
                              'allowed_maximum_buffer_capacity',
                              'allocated_buffer_units',
                              'packet_drop_count'
                              ]
    # A queue entity
    QUEUE = recordclass('Queue',
                        QUEUE_INSTANCE_MEMBERS)

    # The number of fields in the QUEUE namedtuple that are irrelevant to the state
    # This is used for reshaping the tensor and include only the relevant fields in the state given to the Actor
    STATE_IRRELEVANT_FIELDS = 2

    # A port entity
    PORT = recordclass('Port',
                       ['port_identifier',
                        'leftover_buffer_units_in_the_dedicated_pool',
                        'queues'
                        ])

    # A system state
    STATE = recordclass('State',
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
        # self.action_skeleton = [[((port * 0) + (queue * 0)) for queue in range(self.number_of_queues_per_port + 1)]
        #                         for port in range(self.number_of_ports)]
        # self.action_skeleton.append([0])

        # Successful initialization
        # This seems unnecessary - maybe we can find a use for this later...
        self.status = Status.SUCCESS
        # The multiplier parameter is exposed for use by Artemis...
        # The multiplier used in action creation from the output of the constrained_randomization() routine
        self.multiplier = len(self.QUEUE_INSTANCE_MEMBERS) - self.STATE_IRRELEVANT_FIELDS
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

    # Change Log: This method is no longer needed to a change in the Actor architecture
    # Get the reshape factor in order to reshape the state tensor at the Actor
    # def get_reshape_factor(self):
    #     return self.number_of_ports * (
    #         (self.number_of_queues_per_port * (
    #                 len(self.QUEUE._fields) - self.STATE_IRRELEVANT_FIELDS) + 1)) + 1

    # Get the current state [high-level] of the switch environment
    def get_state(self):
        # A simple getter method for external callers
        return self.state

    # Change Log: Eager Execution is not supported when using placeholders (in 1.13.1)
    # def get_state_tensor(self):
    #     ports = []
    #     try:
    #         # The ports loop - p
    #         for port in range(self.number_of_ports):
    #             queues = []
    #             # The queues loop - q
    #             for queue in range(self.number_of_queues_per_port):
    #                 queue_state = self.state.ports[port].queues[queue]
    #                 queues.append([queue_state.required_minimum_capacity,
    #                                queue_state.allowed_maximum_buffer_capacity,
    #                                queue_state.allocated_buffer_units,
    #                                queue_state.packet_drop_count])
    #             # Create an iterable instance of queue tensors specific to port 'p' for stacking later in the routine
    #             ports.append(tensorflow.convert_to_tensor(
    #                 numpy.append(tensorflow.constant(queues,
    #                                                  dtype=tensorflow.int32).numpy(),
    #                              self.state.ports[port].leftover_buffer_units_in_the_dedicated_pool),
    #                 dtype=tensorflow.int32))
    #         # Stack the port-specific tensors, Append the global pool size (leftover: state-specific), ...
    #         # ...and return the resultant state tensor
    #         return tensorflow.convert_to_tensor(
    #             numpy.append(tensorflow.stack(ports,
    #                                           axis=0).numpy(),
    #                          self.state.leftover_buffer_units_in_the_global_pool),
    #             dtype=tensorflow.int32)
    #     except Exception as exception:
    #         print('[ERROR] Nexus get_state_tensor: Exception caught while formatting the state of the switch - '
    #               '[{}]'.format(exception))
    #         traceback.print_tb(exception.__traceback__)
    #     return None

    # Construct a tensor from the high-level Nexus state and pass it down to the calling routine for...
    # ...analysis/input into the NNs
    # As far as switch design is concerned, we don't need caerus here because external callers should be responsible...
    # ...for mutex acquisitions and releases...
    def get_state_tensor(self):
        state = []
        try:
            # The ports loop - p
            for port in range(self.number_of_ports):
                # The queues loop - q
                for queue in range(self.number_of_queues_per_port):
                    queue_state = self.state.ports[port].queues[queue]
                    state.append([queue_state.required_minimum_capacity,
                                  queue_state.allowed_maximum_buffer_capacity,
                                  queue_state.allocated_buffer_units,
                                  queue_state.packet_drop_count])
                # Append the port-specific dedicated pool state
                state.append([self.state.ports[port].leftover_buffer_units_in_the_dedicated_pool])
            # Append the global pool state
            state.append([self.state.leftover_buffer_units_in_the_global_pool])
            # The final state representation - restructure the state vector and convert it into a tensor
            final_state_representation = [k for entry in state for k in entry]
            return tensorflow.constant(final_state_representation,
                                       dtype=tensorflow.int32)
        except Exception as exception:
            print('[ERROR] Nexus get_state_tensor: Exception caught while formatting the state of the switch - '
                  '[{}]'.format(exception))
            traceback.print_tb(exception.__traceback__)
        return None

    # Return the state in the form of an iterable
    def get_state_iterable(self):
        state = []
        try:
            # The ports loop - p
            for port in range(self.number_of_ports):
                # The queues loop - q
                for queue in range(self.number_of_queues_per_port):
                    queue_state = self.state.ports[port].queues[queue]
                    state.append([queue_state.required_minimum_capacity,
                                  queue_state.allowed_maximum_buffer_capacity,
                                  queue_state.allocated_buffer_units,
                                  queue_state.packet_drop_count])
                # Append the port-specific dedicated pool state
                state.append([self.state.ports[port].leftover_buffer_units_in_the_dedicated_pool])
            # Append the global pool state
            state.append([self.state.leftover_buffer_units_in_the_global_pool])
            # The final state representation - restructure the state vector and convert it into a tensor
            return [k for entry in state for k in entry]
        except Exception as exception:
            print('[ERROR] Nexus get_state_iterable: Exception caught while formatting the state of the switch - '
                  '[{}]'.format(exception))
            traceback.print_tb(exception.__traceback__)
        return None

    # Return the given custom state in the form of an iterable
    def get_custom_state_iterable(self, custom_state):
        state = []
        try:
            # The ports loop - p
            for port in range(self.number_of_ports):
                # The queues loop - q
                for queue in range(self.number_of_queues_per_port):
                    queue_state = custom_state.ports[port].queues[queue]
                    state.append([queue_state.required_minimum_capacity,
                                  queue_state.allowed_maximum_buffer_capacity,
                                  queue_state.allocated_buffer_units,
                                  queue_state.packet_drop_count])
                # Append the port-specific dedicated pool state
                state.append([custom_state.ports[port].leftover_buffer_units_in_the_dedicated_pool])
            # Append the global pool state
            state.append([custom_state.leftover_buffer_units_in_the_global_pool])
            # The final state representation - restructure the state vector and convert it into a tensor
            return [k for entry in state for k in entry]
        except Exception as exception:
            print('[ERROR] Nexus get_custom_state_iterable: Exception caught while formatting the state of the switch '
                  '- [{}]'.format(exception))
            traceback.print_tb(exception.__traceback__)
        return None

    # Change Log: Eager Execution is not supported when using placeholders (1.13.1)
    # def get_action_dimension(self):
    #     ports = []
    #     try:
    #         # The ports loop - p
    #         for port in range(self.number_of_ports):
    #             # NOPs w.r.t the queues of port 'p' and the dedicated pool of port 'p'
    #             ports.append([k - k for k in range(self.number_of_queues_per_port + 1)])
    #         # Append an NOP w.r.t the global pool of Nexus and return the dimensions of the resultant tensor
    #         return tensorflow.convert_to_tensor(
    #             numpy.append(tensorflow.constant(ports, dtype=tensorflow.int32).numpy(),
    #                          0),
    #             dtype=tensorflow.int32).shape
    #     except Exception as exception:
    #         print('[ERROR] Nexus get_action_dimension: Exception caught while determining the '
    #               'compliant action dimension - [{}]'.format(exception))
    #         traceback.print_tb(exception.__traceback__)
    #     return None

    # In order to get the action tensor and its associated dimension, assume an initial policy of NOP across...
    # ...all queues, ports, and pools.
    # Return the action tensor [shape will be extracted later]
    def get_action_tensor(self):
        action = []
        try:
            # The ports loop - p
            for port in range(self.number_of_ports):
                # NOPs w.r.t the queues of port 'p' and the dedicated pool of port 'p'
                action.append([k - k for k in range(self.number_of_queues_per_port + 1)])
            # Append an NOP w.r.t the global pool of Nexus and return the dimensions of the resultant tensor
            action.append([0])
            # The final restructured representation of the compliant action
            final_action_representation = [k for entry in action for k in entry]
            return tensorflow.constant(final_action_representation,
                                       dtype=tensorflow.int32)
        except Exception as exception:
            print('[ERROR] Nexus get_action_dimension: Exception caught while determining the '
                  'compliant action dimension - [{}]'.format(exception))
            traceback.print_tb(exception.__traceback__)
        return None

    # In order to get the action iterable and its associated dimension, assume an initial policy of NOP across...
    # ...all queues, ports, and pools.
    # Return the action iterable [shape will be extracted later]
    def get_action_iterable(self):
        action = []
        try:
            # The ports loop - p
            for port in range(self.number_of_ports):
                # NOPs w.r.t the queues of port 'p' and the dedicated pool of port 'p'
                action.append([k - k for k in range(self.number_of_queues_per_port + 1)])
            # Append an NOP w.r.t the global pool of Nexus and return the dimensions of the resultant tensor
            action.append([0])
            # The final restructured representation of the compliant action
            return [k for entry in action for k in entry]
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
        reward = -sum(q.packet_drop_count for p in self.state.ports for q in p.queues)
        if self.incompetence:
            return (self.INCOMPETENCE_PENALTY_MULTIPLIER * reward) + self.INCOMPETENCE_PENALTY_ADDITIVE
        return reward

    # Transition from the current state to the next state and validate the transition
    # Return <reward, next_state>
    def execute(self, action):
        try:
            # Before the transition...
            print('[DEBUG] Nexus transition: Transitioning the underlying MDP - \nState = \n{} and '
                  '\nAction = \n{}'.format(list(self.get_state_iterable()),
                                           list(action)))

            # This is no longer necessary...
            # I've changed the switch API to accept the state transition as a recommendation instead of a cruder +/-
            # TODO: Do we need a structural validation even after the actor getting a compliant action from ...
            #  ...Nexus before execution?
            # # Initial structural validation of the action
            # internal_length_operator = numpy.vectorize(len)
            # if len(action) != self.number_of_ports or \
            #         sum(internal_length_operator(action)) != (self.number_of_ports * self.number_of_queues_per_port):
            #     print('[ERROR] Nexus validate_action: Non-Compliant Action received from the recommendation system - '
            #           '{}'.format(str(action)))
            #     return False

            # Change Log: The action structure was changed to reflect changes in the Actor Network
            # C_{global} global pool update
            # leftover_buffer_units_in_the_global_pool = self.state.leftover_buffer_units_in_the_global_pool + action[
            #     self.number_of_ports]
            # ports = []
            # # Ports Loop - i
            # for i in range(self.number_of_ports):
            #     queues = []
            #     # C_{local}^{P_i} dedicated pool update
            #     leftover_buffer_units_in_the_dedicated_pool =
            #     self.state.ports[i].leftover_buffer_units_in_the_dedicated_pool +
            #     action[i][self.number_of_queues_per_port]
            #     # Queues Loop - j
            #     for j in range(self.number_of_queues_per_port):
            #         queues.append(self.QUEUE(queue_identifier=self.state.ports[i].queues[j].queue_identifier,
            #                                  priority=self.state.ports[i].queues[j].priority,
            #                                  required_minimum_capacity=self.state.ports[i].queues[
            #                                      j].required_minimum_capacity,
            #                                  allowed_maximum_buffer_capacity=self.state.ports[i].queues[
            #                                      j].allowed_maximum_buffer_capacity,
            #                                  allocated_buffer_units=self.state.ports[i].queues[
            #                                                             j].allocated_buffer_units + action[i][j],
            #                                  packet_drop_count=self.state.ports[i].queues[j].packet_drop_count))
            #     ports.append(self.PORT(
            #         port_identifier=self.state.ports[i].port_identifier,
            #         leftover_buffer_units_in_the_dedicated_pool=leftover_buffer_units_in_the_dedicated_pool,
            #         queues=queues))
            # next_state = self.STATE(ports=ports,
            #                         leftover_buffer_units_in_the_global_pool=leftover_buffer_units_in_the_global_pool)
            # # Validate the new state and either authorize or deny the transition
            # # Denial Philosophy: Persistence during Incompetence
            # self.state, self.incompetence = (lambda: self.state, True,
            #                                  lambda: next_state, False)[self.validate(next_state)]()
            # return FEEDBACK(reward=self.reward(),
            #                 next_state=self.state)

            # leftover_buffer_units_in_the_global_pool = self.state.leftover_buffer_units_in_the_global_pool + action[
            #     (self.number_of_queues_per_port + 1) * self.number_of_ports]
            # ports = []
            # # Ports Loop - i
            # for i in range(self.number_of_ports):
            #     queues = []
            #     # C_{local}^{P_i} dedicated pool update
            #     leftover_buffer_units_in_the_dedicated_pool = \
            #         self.state.ports[i].leftover_buffer_units_in_the_dedicated_pool + action[(i * (
            #                 self.number_of_queues_per_port + 1)) + self.number_of_queues_per_port]
            #     # Queues Loop - j
            #     for j in range(self.number_of_queues_per_port):
            #         queues.append(self.QUEUE(queue_identifier=self.state.ports[i].queues[j].queue_identifier,
            #                                  priority=self.state.ports[i].queues[j].priority,
            #                                  required_minimum_capacity=self.state.ports[i].queues[
            #                                      j].required_minimum_capacity,
            #                                  allowed_maximum_buffer_capacity=self.state.ports[i].queues[
            #                                      j].allowed_maximum_buffer_capacity,
            #                                  allocated_buffer_units=self.state.ports[i].queues[
            #                                                             j].allocated_buffer_units + action[
            #                                                             (i * (self.number_of_queues_per_port + 1))
            #                                                             + j],
            #                                  packet_drop_count=self.state.ports[i].queues[j].packet_drop_count))
            #     ports.append(self.PORT(
            #         port_identifier=self.state.ports[i].port_identifier,
            #         leftover_buffer_units_in_the_dedicated_pool=leftover_buffer_units_in_the_dedicated_pool,
            #         queues=queues))
            # next_state = self.STATE(ports=ports,
            #                         leftover_buffer_units_in_the_global_pool=leftover_buffer_units_in_the_global_pool)

            # I've changed the switch API to accept the state transition as a recommendation instead of a cruder +/-
            # Mutex has been acquired: min, max, and packet_drop_count extraction
            ports = []
            for p in range(self.number_of_ports):
                queues = []
                for q in range(self.number_of_queues_per_port):
                    queues.append(self.QUEUE(queue_identifier=self.state.ports[p].queues[q].queue_identifier,
                                             priority=self.state.ports[p].queues[q].priority,
                                             required_minimum_capacity=self.state.ports[p].queues[
                                                 q].required_minimum_capacity,
                                             allowed_maximum_buffer_capacity=self.state.ports[p].queues[
                                                 q].allowed_maximum_buffer_capacity,
                                             allocated_buffer_units=action[
                                                 (p * (self.number_of_queues_per_port + 1) + q)
                                             ],
                                             packet_drop_count=self.state.ports[p].queues[q].packet_drop_count
                                             ))
                ports.append(self.PORT(port_identifier=self.state.ports[p].port_identifier,
                                       leftover_buffer_units_in_the_dedicated_pool=action[
                                           ((p * (self.number_of_queues_per_port + 1)) + self.number_of_queues_per_port)
                                       ],
                                       queues=queues))
            next_state = self.STATE(ports=ports,
                                    leftover_buffer_units_in_the_global_pool=action[-1])
            print('[DEBUG] Nexus execute: The recommended state transition is \n{}\n'.format(
                self.get_custom_state_iterable(next_state)
            ))
            # Validate the new state and either authorize or deny the transition
            # Denial Philosophy: Persistence during Incompetence
            if self.validate(next_state) is False:
                self.incompetence = True
            else:
                self.incompetence = False
                self.state = next_state
            # Temporary mutex release
            if caerus.locked():
                caerus.release()
            # Sleep for some time for Ares to catch up...
            time.sleep(10.0)
            # TODO: All mutex operations should be outside Nexus...this is a bad hack
            # Get back the mutex for reward analysis
            caerus.acquire()
            # Analyze the reward after Ares had some time to thoroughly comprehend the changes...
            reward = self.reward()
            # After the transition...
            print('[DEBUG] MDP Transitioned - \nState = \n{} '
                  '\nReward = {}'.format(list(self.get_state_iterable()),
                                         reward))
            return FEEDBACK(reward=reward,
                            next_state=self.get_state_iterable())
        except Exception as e:
            print('[ERROR] Nexus execute: Exception caught while executing the recommended MDP transition - '
                  '{}'.format(e))
            traceback.print_tb(e.__traceback__)
            return None
        finally:
            # Safe mutex release
            if caerus.locked():
                caerus.release()
        # The state transition of the underlying MDP has been completed...

    # Get the limits of the action vector to be recommended by the Actor Network
    def get_action_limits(self):
        action_limits = []
        for p in range(self.number_of_ports):
            for q in range(self.number_of_queues_per_port):
                action_limits.append(self.state.ports[p].queues[q].allowed_maximum_buffer_capacity)
            action_limits.append(self.DEDICATED_POOL_SIZE_PER_PORT)
        action_limits.append(self.GLOBAL_POOL_SIZE)
        return action_limits

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
    # The number of neurons in the first hidden layer of the actor network
    NUMBER_OF_HIDDEN_UNITS_1 = 5200

    # The number of neurons in the second hidden layer of the actor network
    NUMBER_OF_HIDDEN_UNITS_2 = 3900

    # Change Log: The reshape_factor arg is no longer needed due to a change in the way I build the network and ...
    # ...predict the output

    # def __init__(self, _state_dimension, _action_dimension, _learning_rate,
    #              _target_tracker_coefficient, _batch_size, _reshape_factor):

    # The initialization sequence
    def __init__(self, _tensorflow_session, _state_dimension, _action_dimension, _action_bounds,
                 _learning_rate, _target_tracker_coefficient, _batch_size):
        print('[INFO] Actor Initialization: Bringing things up...')
        # Initializing the essential input parameters with the given arguments
        self.tensorflow_session = _tensorflow_session
        self.state_dimension = _state_dimension
        self.action_dimension = _action_dimension
        self.action_bounds = _action_bounds
        self.learning_rate = _learning_rate
        self.target_tracker_coefficient = _target_tracker_coefficient
        self.batch_size = _batch_size

        # self.reshape_factor = _reshape_factor

        # Declaring the internal entities and outputs
        self.input, self.output = self.build()
        self.network_parameters = tensorflow.trainable_variables()
        self.target_input, self.target_output = self.build()
        self.target_network_parameters = tensorflow.trainable_variables()[len(self.network_parameters):]

        # Turn eager mode off
        # tensorflow.disable_eager_execution()

        self.action_gradients = tensorflow.placeholder(tensorflow.float32,
                                                       [None, None, self.action_dimension.dims[0].value])
        # The internal gradient operation in the Policy Gradient step
        self.unnormalized_actor_gradients = tensorflow.gradients(self.output,
                                                                 self.network_parameters,
                                                                 -self.action_gradients)
        # Normalization w.r.t the batch size - this refers to the expectation operator in the Policy Gradient step
        self.normalized_actor_gradients = list(map(lambda x: tensorflow.div(x,
                                                                            self.batch_size),
                                                   self.unnormalized_actor_gradients))
        # The Adam Optimizer
        self.optimized_result = tensorflow.train.AdamOptimizer(learning_rate=self.learning_rate).apply_gradients(
            zip(self.normalized_actor_gradients,
                self.network_parameters))
        # The target update - soft update using the target_update_coefficient (/tau)
        self.updated_target_network_parameters = [self.target_network_parameters[i].assign(
            tensorflow.multiply(self.network_parameters[i],
                                self.target_tracker_coefficient) + tensorflow.multiply(
                self.target_network_parameters[i],
                (1.0 - self.target_tracker_coefficient))) for i in range(len(self.target_network_parameters))]
        # The initialization of the actor instance has been completed...

    # Build the Actor Network
    def build(self):
        # A sequential model with keras layers
        input_data = tensorflow.keras.layers.Input(shape=[None, None, self.state_dimension.dims[0].value])
        dense_layer_1 = tensorflow.keras.layers.Dense(
            units=self.NUMBER_OF_HIDDEN_UNITS_1,
            activation=tensorflow.keras.activations.relu)(input_data)
        batch_normalization_layer_1 = tensorflow.keras.layers.BatchNormalization()(dense_layer_1)
        dense_layer_2 = tensorflow.keras.layers.Dense(
            units=self.NUMBER_OF_HIDDEN_UNITS_2,
            activation=tensorflow.keras.activations.relu)(batch_normalization_layer_1)
        batch_normalization_layer_2 = tensorflow.keras.layers.BatchNormalization()(dense_layer_2)
        output_data = tensorflow.keras.layers.Dense(
            units=self.action_dimension.dims[0].value,
            activation=tensorflow.keras.activations.sigmoid,
            kernel_initializer='glorot_uniform')(batch_normalization_layer_2)
        return input_data, tensorflow.multiply(output_data[0],
                                               self.action_bounds)

        # Print the summary of the model for aesthetic purposes
        # print('[INFO] Actor build: The summary of the model built is printed below...')
        # model.summary()

        # Don't return the model per se, return the output (a different approach)
        # return tensorflow.keras.models.Model(inputs=[input_data],
        #                                      outputs=[output_data])

        # Model-Building has been completed...

    # Train the built model - define the loss, evaluate the gradients, and use the right optimizer...
    def train(self, _input, _action_gradients):
        # Deep Deterministic Policy Gradient (DDPG)
        return self.tensorflow_session.run(self.optimized_result,
                                           feed_dict={
                                               self.input: _input,
                                               self.action_gradients: _action_gradients
                                           })
        # DDPG-based Actor Network optimization has been completed...

    # Predict actions for either the batch of states sampled from the replay memory OR...
    # ...for the individual state observed from the switch environment
    def predict(self, state_batch):
        # Reshaping is not necessary now
        # state_batch = tensorflow.reshape(state_batch,
        #                                  [self.reshape_factor, ])

        # TODO: Scaling here, if necessary
        return self.tensorflow_session.run(self.output,
                                           feed_dict={
                                               self.input: state_batch
                                           })

    # Soft target update procedure
    def update_targets(self):
        return self.tensorflow_session.run(self.updated_target_network_parameters)

    # Predict actions from the target network for target Q-value evaluation during training, i.e. (r_t + \gamma Q(s, a))
    def predict_targets(self, target_state_batch):
        # Reshaping is not necessary now
        # target_state_batch = tensorflow.reshape(target_state_batch,
        #                                         (self.reshape_factor,
        #                                          1))

        # TODO: Scaling here, if necessary
        return self.tensorflow_session.run(self.target_output,
                                           feed_dict={
                                               self.target_input: target_state_batch
                                           })

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Actor Termination: Tearing things down...')


# The Actor Network ends here...

# The Critic Network begins here...

# TODO: Is there a better way to handle the dual-input (state and action) model building strategy...

# The Critic Network
class Critic(object):
    # The number of neurons in the first hidden layer of the critic network w.r.t the state_input
    NUMBER_OF_HIDDEN_UNITS_STATE_1 = 5200

    # The number of neurons in the second hidden layer of the critic network w.r.t the state_input
    NUMBER_OF_HIDDEN_UNITS_STATE_2 = 3900

    # The number of neurons in the first hidden layer of the critic network w.r.t the action_input
    NUMBER_OF_HIDDEN_UNITS_ACTION_1 = 5200

    # The number of neurons in the second hidden layer of the critic network w.r.t the action_input
    NUMBER_OF_HIDDEN_UNITS_ACTION_2 = 3900

    # The initialization sequence
    def __init__(self, _tensorflow_session, _state_dimension, _action_dimension,
                 _learning_rate, _target_tracker_coefficient, _number_of_actor_network_weights):
        print('[INFO] Critic Initialization: Bringing things up...')
        # Initializing the input parameters with the given arguments
        self.tensorflow_session = _tensorflow_session
        self.state_dimension = _state_dimension
        self.action_dimension = _action_dimension
        self.learning_rate = _learning_rate
        self.target_tracker_coefficient = _target_tracker_coefficient
        self.number_of_actor_weights = _number_of_actor_network_weights
        # Declaring the internal entities and outputs
        self.state, self.action, self.output = self.build()
        self.network_parameters = tensorflow.trainable_variables()[self.number_of_actor_weights:]
        self.target_state, self.target_action, self.target_output = self.build()
        self.target_network_parameters = tensorflow.trainable_variables()[
                                         self.number_of_actor_weights + len(self.network_parameters):]
        # The target Q-value from the Bellman equation is set as a placeholder here...
        self.target_q_value = tensorflow.placeholder(dtype=tensorflow.float32,
                                                     shape=[None, None, None, 1])
        # A standard mean-squared error loss function
        self.loss = tensorflow.losses.mean_squared_error(self.output,
                                                         self.target_q_value)
        # The Adam Optimizer
        self.optimized_result = tensorflow.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # The gradient of the predicted Q-value w.r.t the action
        self.action_gradients = tensorflow.gradients(self.output,
                                                     self.action)
        # The target update - soft update using the target_update_coefficient (/tau)
        self.updated_target_network_parameters = [self.target_network_parameters[i].assign(
            tensorflow.multiply(self.network_parameters[i],
                                self.target_tracker_coefficient) + tensorflow.multiply(
                self.target_network_parameters[i],
                (1.0 - self.target_tracker_coefficient))) for i in range(len(self.target_network_parameters))]
        # The initialization sequence has been completed...

    # Build the Critic Network
    # While fitting, it takes in both state and action as inputs...
    def build(self):
        # The state input layers
        state_input = tensorflow.keras.layers.Input(shape=[None, None, self.state_dimension.dims[0].value],
                                                    name='state_input')
        state_input_modified = tensorflow.keras.layers.Dense(units=self.NUMBER_OF_HIDDEN_UNITS_STATE_1,
                                                             activation=tensorflow.keras.activations.relu)(state_input)
        state_input_whitened = tensorflow.keras.layers.BatchNormalization()(state_input_modified)
        state_input_whitened_modified = tensorflow.keras.layers.Dense(
            units=self.NUMBER_OF_HIDDEN_UNITS_STATE_2,
            activation=tensorflow.keras.activations.relu)(state_input_whitened)
        # The action input layers
        action_input = tensorflow.keras.layers.Input(shape=[None, None, self.action_dimension.dims[0].value],
                                                     name='action_input')
        action_input_modified = tensorflow.keras.layers.Dense(
            units=self.NUMBER_OF_HIDDEN_UNITS_ACTION_1,
            activation=tensorflow.keras.activations.relu)(action_input)
        action_input_whitened = tensorflow.keras.layers.BatchNormalization()(action_input_modified)
        action_input_whitened_modified = tensorflow.keras.layers.Dense(
            units=self.NUMBER_OF_HIDDEN_UNITS_ACTION_2,
            activation=tensorflow.keras.activations.relu)(action_input_whitened)
        # Merge the two branches corresponding to the state input and the action input
        merged_result = tensorflow.keras.layers.concatenate([state_input_whitened_modified,
                                                             action_input_whitened_modified])
        output = tensorflow.keras.layers.Dense(units=1,
                                               kernel_initializer='glorot_uniform')(merged_result)
        return state_input, action_input, output

        # Change Log: Don't return the model anymore, directly return the output...
        # model = tensorflow.keras.models.Model(inputs=[state_input,
        #                                               action_input],
        #                                       outputs=[output])

        # Print the summary of the model for aesthetic purposes
        # print('[INFO] Critic build: The summary of the model is printed below.')
        # model.summary()

        # return model

        # Critic model building has been completed...

    # Train the built model, define the loss function, and use the right optimizer...
    def train(self, _state, _action, _target_q_value):
        return self.tensorflow_session.run([self.output, self.optimized_result],
                                           feed_dict={
                                               self.state: _state,
                                               self.action: _action,
                                               self.target_q_value: _target_q_value
                                           })
        # MSE-optimization of the Critic Network has been completed...

    # \triangledown_\{a}\ Q(\vec{S}, a)
    def get_action_gradients(self, _state, _action):
        return self.tensorflow_session.run(self.action_gradients,
                                           feed_dict={
                                               self.state: _state,
                                               self.action: _action
                                           })

    # Estimate the Q-value for the given state-action pair
    def predict(self, _state, _action):
        return self.tensorflow_session.run(self.output,
                                           feed_dict={
                                               self.state: _state,
                                               self.action: _action
                                           })

    # Estimate the Q-value for the state-action pair using the Target Critic Network
    def predict_targets(self, _target_state, _target_action):
        return self.tensorflow_session.run(self.target_output,
                                           feed_dict={
                                               self.target_state: _target_state,
                                               self.target_action: _target_action
                                           })

    # Soft target update procedure
    def update_targets(self):
        return self.tensorflow_session.run(self.updated_target_network_parameters)

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
            [k[2] for k in sample]), numpy.array([k[3] for k in sample]), numpy.array([k[4] for k in sample])

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
    EXPLORATION_DECAY = 0.99

    # The default minimum allowed exploration factor - exploration-decay strategy only
    MINIMUM_EXPLORATION_FACTOR = 0.1

    # The initialization sequence
    def __init__(self, _environment, _exploration_strategy, _action_dimension, _exploration_factor=None,
                 _exploration_decay=None, _exploration_factor_min=None, _x0=None, _mu=None,
                 _theta=0.15, _sigma=0.3, _dt=1e-2):
        print('[INFO] Artemis Initialization: Bringing things up...')
        # Initializing the input parameters with the given arguments...
        self.environment = _environment
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

    # The constrained randomization procedure for the EXPLORATION_DECAY strategy
    # A static utility method to generate random allocation strategies within the specified bounds
    # This routine is O(n) as opposed to O(n^m) that would be needed for conventional methods
    @staticmethod
    def constrained_randomization(slots, ceiling):
        allocated = []
        divisor = numpy.random.uniform(0.0,
                                       float(ceiling))
        remaining = ceiling
        for i in range(slots - 1):
            x = int(math.floor(remaining / divisor))
            allocated.append(x)
            remaining = remaining - x
        allocated.append(remaining)
        return allocated

    # Return the action appended/modified with the exploration noise
    def execute(self, action):

        # EXPLORATION_DECAY - Unconstrained Randomization
        # if self.exploration_strategy == ExplorationStrategy.EXPLORATION_DECAY:
        #     # Decay the exploration factor and employ the well-known $\epsilon-greedy$ logic
        #     self.decay()
        #     if numpy.random.rand() <= self.exploration_factor:
        #         return numpy.reshape(numpy.random.random_sample(self.action_dimension),
        #                              newshape=(1,
        #                                        1,
        #                                        self.action_dimension))
        #     return action

        # EXPLORATION_DECAY - Constrained Randomization
        if self.exploration_strategy == ExplorationStrategy.EXPLORATION_DECAY:
            # The constrained random action here is a recommended state transition...(the switch API handles the +/-)
            constrained_random_action = []
            # Decay the exploration factor and apply the $\epsilon-greedy$ logic
            self.decay()
            if numpy.random.rand() <= self.exploration_factor:

                # env_state = self.environment.get_state_iterable()

                # Global allocation - the ports and the global pool
                global_allocation = self.constrained_randomization(self.environment.number_of_ports + 1,
                                                                   self.environment.global_pool_size)
                for port in range(self.environment.number_of_ports):
                    # Local allocation - the queues and the local pool
                    for k in self.constrained_randomization(
                            self.environment.number_of_queues_per_port + 1,
                            global_allocation[port] + self.environment.dedicated_pool_size_per_port):
                        constrained_random_action.append(k)
                constrained_random_action.append(
                    global_allocation[self.environment.number_of_ports]
                )
                # Reshape the action to make it compliant with the Actor-Critic-Nexus framework
                return numpy.reshape(constrained_random_action,
                                     newshape=(1,
                                               1,
                                               self.action_dimension)
                                     )
            # The non-random action here is a recommended state transition...(the switch API handles the +/-)
            # Restructuring the Actor-recommended transition...
            return numpy.reshape([int(k) for k in numpy.squeeze(action)],
                                 newshape=(1,
                                           1,
                                           self.action_dimension)
                                 )
        # TODO: Does this seem reasonable (random noise addition) when I decided to directly deal in transitions...
        #  and hide the +/- operations within the switch's exposed API?
        # ORNSTEIN_UHLENBECK_NOISE - Unconstrained Randomization
        if self.exploration_strategy == ExplorationStrategy.ORNSTEIN_UHLENBECK_NOISE:
            noise = self.generate_noise()
            # Assuming the action is a row vector, we add the generated Ornstein-Uhlenbeck noise to each action entry
            # TODO: If the action is a tensor of a different shape, change this exploration noise addition logic...
            return numpy.reshape([(k + noise) for k in action],
                                 newshape=(1,
                                           1,
                                           self.action_dimension)
                                 )

    # A simple utility method to decay the exploration factor
    def decay(self):
        if self.exploration_factor >= self.exploration_factor_min:
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
    # Change Log: The environment_details member is no longer needed as I'm directly getting a reference to Nexus.
    def __init__(self, _nexus, _actor_design_details, _critic_design_details, _replay_memory_details,
                 _exploration_strategy_details, _batch_size, _discount_factor,
                 _iterations_per_episode, _maximum_number_of_episodes):
        print('[INFO] Apollo Initialization: Bringing things up...')
        # The status indicator flag
        self.status = Status.SUCCESS
        # Initializing the relevant members - default to hard-coded values upon invalidation
        self.utilities = Utilities()

        # This is unnecessary - I'm directly getting a reference to Nexus.
        # self.environment_details = (lambda: ENVIRONMENT_DETAILS(number_of_ports=None,
        #                                                         number_of_queues_per_port=None,
        #                                                         global_pool_size=None,
        #                                                         dedicated_pool_size_per_port=None),
        #                             lambda: _environment_details)[_environment_details is not None and
        #                                                           self.utilities.custom_instance_validation(
        #                                                               _environment_details,
        #                                                               ENVIRONMENT_DETAILS)]()

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

        # This is unnecessary - I'm directly getting a reference to Nexus.
        # Create the Nexus switch environment
        # self.nexus = Nexus(self.environment_details.number_of_ports,
        #                    self.environment_details.number_of_queues_per_port,
        #                    self.environment_details.global_pool_size,
        #                    self.environment_details.dedicated_pool_size_per_port)

        self.nexus = _nexus

        # Get the switch environment details from the created Nexus instance
        # NOTE: I don't need a mutex acquisition and release strategy in the initialization routine of Apollo
        self.state_dimension = self.nexus.get_state_tensor().shape
        self.action_dimension = self.nexus.get_action_tensor().shape

        # Change Log: This is no longer needed as I changed the architecture
        # self.reshape_factor = self.nexus.get_reshape_factor()

        # State and Action validation check
        if self.state_dimension is None or self.action_dimension is None:
            print('[ERROR] Apollo Initialization: Something went wrong while obtaining the state and compliant action '
                  'information from Nexus. Please refer to the earlier logs for more details on this error.')
            self.status = Status.FAILURE

        # Change Log: A change in the Actor constructor because I no longer need the reshape_factor argument
        # self.actor = Actor(self.state_dimension,
        #                    self.action_dimension,
        #                    self.actor_design_details.learning_rate,
        #                    self.actor_design_details.target_tracker_coefficient,
        #                    self.actor_design_details.batch_size,
        #                    self.reshape_factor)

        # Moving these initializations to the start() routine to persist the tensorflow session...
        # with tensorflow.Session() as self.session:
        #     # Create the Actor and Critic Networks
        #     self.actor = Actor(self.session,
        #                        self.state_dimension,
        #                        self.action_dimension,
        #                        self.actor_design_details.learning_rate,
        #                        self.actor_design_details.target_tracker_coefficient,
        #                        self.actor_design_details.batch_size)
        #     self.critic = Critic(self.session,
        #                          self.state_dimension,
        #                          self.action_dimension,
        #                          self.critic_design_details.learning_rate,
        #                          self.critic_design_details.target_tracker_coefficient,
        #                          len(self.actor.network_parameters) + len(self.actor.target_network_parameters))
        #     # Initialize the Prioritized Experiential Replay Memory
        #     self.mnemosyne = Mnemosyne(self.replay_memory_details.memory_capacity,
        #                                self.replay_memory_details.prioritization_strategy,
        #                                self.replay_memory_details.revisitation_constraint_constant,
        #                                self.replay_memory_details.prioritization_level,
        #                                self.replay_memory_details.random_seed)
        #     # Initialize the Exploration Noise Generator
        #     self.artemis = Artemis(self.exploration_strategy_details.exploration_strategy,
        #                            self.exploration_strategy_details.action_dimension,
        #                            self.exploration_strategy_details.exploration_factor,
        #                            self.exploration_strategy_details.exploration_decay,
        #                            self.exploration_strategy_details.exploration_factor_min,
        #                            self.exploration_strategy_details.x0,
        #                            self.exploration_strategy_details.mu,
        #                            self.exploration_strategy_details.theta,
        #                            self.exploration_strategy_details.sigma,
        #                            self.exploration_strategy_details.dt)

        # The initialization sequence has been completed

    # Start the interaction with the environment and the training process
    def start(self):
        print('[INFO] Apollo train: Interacting with the switch environment and initiating the training process')
        # The tensorflow session declaration for clean handling
        session = None
        try:

            # Change log: Explicit call is no longer needed - I perform internal model building during initialization
            # # Build the Actor and Critic Networks
            # self.actor.build()
            # self.critic.build()

            # Thread-safe global reference
            # Change Log: A thread-safe global reference is no longer needed because Apollo is now part of ...
            # ...the main thread
            # global global_tensorflow_session

            # The tensorflow session initialization
            session = tensorflow.Session()
            with session:
                # Create the Actor and Critic Networks
                actor = Actor(session,
                              self.state_dimension,
                              self.action_dimension,
                              self.nexus.get_action_limits(),
                              self.actor_design_details.learning_rate,
                              self.actor_design_details.target_tracker_coefficient,
                              self.actor_design_details.batch_size)
                critic = Critic(session,
                                self.state_dimension,
                                self.action_dimension,
                                self.critic_design_details.learning_rate,
                                self.critic_design_details.target_tracker_coefficient,
                                len(actor.network_parameters) + len(actor.target_network_parameters))
                # Initialize the Prioritized Experiential Replay Memory
                mnemosyne = Mnemosyne(self.replay_memory_details.memory_capacity,
                                      self.replay_memory_details.prioritization_strategy,
                                      self.replay_memory_details.revisitation_constraint_constant,
                                      self.replay_memory_details.prioritization_level,
                                      self.replay_memory_details.random_seed)
                # Initialize the Exploration Noise Generator
                artemis = Artemis(self.nexus,
                                  self.exploration_strategy_details.exploration_strategy,
                                  self.exploration_strategy_details.action_dimension,
                                  self.exploration_strategy_details.exploration_factor,
                                  self.exploration_strategy_details.exploration_decay,
                                  self.exploration_strategy_details.exploration_factor_min,
                                  self.exploration_strategy_details.x0,
                                  self.exploration_strategy_details.mu,
                                  self.exploration_strategy_details.theta,
                                  self.exploration_strategy_details.sigma,
                                  self.exploration_strategy_details.dt)
                # Initialize the tensorflow session
                session.run(tensorflow.global_variables_initializer())
                # Start the interaction with Nexus
                for episode in range(self.maximum_number_of_episodes):
                    for iteration in range(self.iterations_per_episode):
                        # Initialize/Re-Train/Update the target networks in this off-policy DDQN-architecture
                        actor.update_targets()
                        critic.update_targets()
                        # Observe the state, execute an action, and get the feedback from the switch environment
                        # Automatic next_state transition fed in by using the Nexus instance
                        # Transition and validation is encapsulated within Nexus
                        # Mutex acquisition
                        caerus.acquire()
                        state = numpy.expand_dims(numpy.expand_dims(numpy.reshape(numpy.asarray(
                            self.nexus.get_state_iterable()),
                            (1,
                             self.state_dimension.dims[0].value)),
                            axis=0),
                            axis=0)
                        action = artemis.execute(actor.predict(state))
                        feedback = self.nexus.execute(numpy.squeeze(action))
                        # Mutex released from within Nexus...
                        # Validation - exit if invalid
                        if feedback is None or self.utilities.custom_instance_validation(feedback,
                                                                                         FEEDBACK) is False:
                            print('[ERROR] Apollo train: Invalid feedback received from the environment. '
                                  'Please check the compatibility between Apollo and the Nexus variant')
                            return False
                        # Find the target Q-value, the predicted Q-value, and subsequently the TD-error
                        target_q = feedback.reward + (self.discount_factor * critic.predict_targets(
                            numpy.expand_dims(numpy.expand_dims(numpy.reshape(numpy.asarray(feedback.next_state),
                                                                              (1,
                                                                               self.state_dimension.dims[0].value)),
                                                                axis=0),
                                              axis=0),
                            numpy.expand_dims(actor.predict_targets(numpy.expand_dims(numpy.expand_dims(numpy.reshape(
                                numpy.asarray(feedback.next_state),
                                (1,
                                 self.state_dimension.dims[0].value)),
                                axis=0),
                                axis=0)),
                                axis=0)))
                        predicted_q = critic.predict(state,
                                                     numpy.expand_dims(action,
                                                                       axis=0))
                        td_error = predicted_q - target_q
                        # Remember this experience
                        mnemosyne.remember(numpy.reshape(numpy.squeeze(state),
                                                         (1,
                                                          self.state_dimension.dims[0].value)),
                                           numpy.reshape(numpy.squeeze(action),
                                                         (1,
                                                          self.action_dimension.dims[0].value)),
                                           feedback.reward,
                                           numpy.reshape(numpy.asarray(feedback.next_state),
                                                         (1,
                                                          self.state_dimension.dims[0].value)),
                                           td_error)
                        # Start the replay sequence for training
                        if len(mnemosyne.memory) >= self.batch_size:
                            # Prioritization strategy specific replay
                            s_batch, a_batch, r_batch, s2_batch, td_error_batch = mnemosyne.replay(self.batch_size)
                            target_q = numpy.squeeze(critic.predict_targets(numpy.expand_dims(s2_batch,
                                                                                              axis=0),
                                                                            numpy.expand_dims(actor.predict_targets(
                                                                                numpy.expand_dims(s2_batch, axis=0)),
                                                                                axis=0)))
                            target_q_values = []
                            for k in range(self.batch_size):
                                target_q_values.append(r_batch[k] + (self.discount_factor * target_q[k]))
                            # Train the Critic - standard MSE optimization
                            critic.train(numpy.expand_dims(s_batch, axis=0),
                                         numpy.expand_dims(a_batch, axis=0),
                                         numpy.expand_dims(numpy.expand_dims(numpy.reshape(target_q_values,
                                                                                           newshape=(self.batch_size,
                                                                                                     1)),
                                                                             axis=0),
                                                           axis=0))
                            # Get the action gradients for DDPG
                            action_output = actor.predict(numpy.expand_dims(s_batch, axis=0))
                            action_gradients = critic.get_action_gradients(numpy.expand_dims(s_batch, axis=0),
                                                                           numpy.expand_dims(action_output,
                                                                                             axis=0))
                            # Train the Actor - DDPG
                            actor.train(numpy.expand_dims(s_batch, axis=0),
                                        action_gradients[0][0])
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
            if caerus.locked():
                caerus.release()
            # If the tensorflow session is up and running, close it...
            if session is not None:
                session.close()
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
                    port_specific_arrival_rates.append(random.randrange(1, 32, 1))
                else:
                    # A lower arrival rate for low-priority queues
                    port_specific_arrival_rates.append(random.randrange(1, 16, 1))
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
        try:
            # TODO: The code looks and seems childish: Clean it up...
            # The switch should be up and running
            # Change Log: Loop control for the timer logic included in-house
            while True:
                # Acquire the mutex for accessing the switch infrastructure
                caerus.acquire()
                for p in range(self.nexus.number_of_ports):
                    for q in range(self.nexus.number_of_queues_per_port):
                        # Check if the switch is up...
                        if self.nexus.shutdown:
                            print('[ERROR] Ares start: The Nexus environment is no longer available to receive packets')
                            break
                        # New Arrivals
                        arrival_times = []
                        # Inverse Transform: Generation of exponential inter-arrival times from...
                        # ...a uniform random variable
                        arrival_time = (-numpy.log(numpy.random.random_sample())) / self.arrival_rates[p][q]
                        # Per second arrival rate - this is fixed (\lambda_{pq} is the arrival rate [per second])
                        while arrival_time <= 1.0:
                            arrival_times.append(arrival_time)
                            arrival_time += (-numpy.log(numpy.random.random_sample())) / self.arrival_rates[p][q]
                        allocated = self.nexus.get_state().ports[p].queues[q].allocated_buffer_units
                        # Pending packets
                        pending = self.pending_packets[p][q]
                        # No allocated units for this queue...
                        if allocated == 0:
                            self.nexus.get_state().ports[p].queues[q].packet_drop_count = pending + len(arrival_times)
                            self.pending_packets[p][q] = 0
                            continue
                        # Less allocated space than what was previously pending...
                        elif allocated < pending:
                            free = 0
                            dropped = pending - allocated
                        # The allocated space is greater than or equal to what was previously pending...
                        else:
                            free = allocated - pending
                            dropped = 0
                        # Start simulating the queueing process with arrivals and departures...
                        pending_run_time = 0.0
                        # Previous checkpoint for arrival analysis
                        previous_time_checkpoint = 0.0
                        while pending != 0:
                            # Inverse-Transform method: Generation of exponential service times
                            pending_run_time += (-numpy.log(
                                numpy.random.random_sample())) / self.service_rates[p][q]
                            # Are you still in this window?
                            if pending_run_time <= 1.0:
                                free += 1
                                # Current checkpoint for arrival analysis
                                current_time_checkpoint = pending_run_time
                                for arrival in range(len(arrival_times)):
                                    if previous_time_checkpoint < arrival_times[arrival] < current_time_checkpoint:
                                        if free > 0:
                                            free -= 1
                                        else:
                                            dropped += 1
                                previous_time_checkpoint = current_time_checkpoint
                                pending = allocated - free
                            else:
                                break
                        self.nexus.get_state().ports[p].queues[q].packet_drop_count = dropped
                        self.pending_packets[p][q] = allocated - free
                # Change Log: Sleep for 1 second before pumping packets and simulating the queueing system
                # Release the mutex
                if caerus.locked():
                    caerus.release()
                time.sleep(1.0)
        except Exception as exception:
            print('[ERROR] Ares start: Exception caught while simulating packet arrival and '
                  'analyzing switch performance - [{}]'.format(exception))
            traceback.print_tb(exception.__traceback__)
        finally:
            # Safe release of the mutex
            if caerus.locked():
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
    action_dimension = len(nexus.get_action_iterable())
    if action_dimension is None:
        print('[ERROR] AdaptiveBufferIntelligence Trigger: Something went wrong while obtaining the action dimension '
              'from Nexus. Please refer to the earlier logs for more details on this error. Exiting!')
        raise SystemExit

    # This is unnecessary right now. I'm directly passing a reference to Nexus.
    # environment_details = ENVIRONMENT_DETAILS(number_of_ports=number_of_ports,
    #                                           number_of_queues_per_port=number_of_queues_per_port,
    #                                           global_pool_size=global_pool_size,
    #                                           dedicated_pool_size_per_port=dedicated_pool_size_per_port)

    # Actor Design
    actor_design_details = ACTOR_DESIGN_DETAILS(learning_rate=1e-4,
                                                target_tracker_coefficient=0.01,
                                                batch_size=64)
    # Critic Design
    critic_design_details = CRITIC_DESIGN_DETAILS(learning_rate=1e-5,
                                                  target_tracker_coefficient=0.01)
    # Replay Memory Design
    replay_memory_design_details = REPLAY_MEMORY_DETAILS(memory_capacity=int(1e9),
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
        exploration_factor=1.0,
        exploration_decay=0.99,
        exploration_factor_min=0.1,
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
    iterations_per_episode = 1e2
    # Maximum Number of Episodes
    maximum_number_of_episodes = 1e4

    # Old call: The environment_details arg is no longer needed. Pass Nexus directly to Apollo.
    # apollo = Apollo(environment_details, actor_design_details, critic_design_details, replay_memory_design_details,
    #                 exploration_strategy_design_details, batch_area, discount_factor, iterations_per_episode,
    #                 maximum_number_of_episodes)

    # Create an instance of Apollo and start that simultaneously
    apollo = Apollo(nexus, actor_design_details, critic_design_details, replay_memory_design_details,
                    exploration_strategy_design_details, batch_area, discount_factor, int(iterations_per_episode),
                    int(maximum_number_of_episodes))
    # Status check
    if apollo.status == Status.FAILURE:
        print('[ERROR] AdaptiveBufferIntelligence Trigger: Something went wrong during the initialization of Apollo. '
              'Please refer to the earlier logs for more information on this error.')
        raise SystemExit
    apollo_thread = threading.Thread(target=apollo.start)
    print('[INFO] AdaptiveBufferIntelligence Trigger: Starting Apollo...[apollo_thread]')
    apollo_thread.start()

    # The timer interval for Cronus controlling Ares - fix this to 1.0
    timer_interval = 1.0

    # Create a timer thread for Ares initialized with Nexus and start the evaluation thread
    # Change Log: A thread only for Ares - Apollo will be the main thread
    # Change Log: No timer thread for Ares - handle loop within start() of Ares

    ares = Ares(nexus)
    cronus_thread = threading.Thread(target=ares.start)

    # cronus_thread = threading.Timer(timer_interval, ares.start)

    # Create individual threads for Ares (cronus_thread) and Apollo (apollo_thread)
    # cronus_thread = threading.Timer(timer_interval, ares.start)
    # apollo_thread = threading.Thread(target=apollo.start)

    print('[INFO] AdaptiveBufferIntelligence Trigger: Starting Ares...[cronus_thread]')
    # Start the Ares and Apollo threads
    cronus_thread.start()
    print('[INFO] AdaptiveBufferIntelligence Trigger: Joining all spawned threads...')
    # Join the apollo thread upon completion - Integrate with the [main] thread and cancel the timer
    apollo_thread.join()
    cronus_thread.join()
    print('[INFO] AdaptiveBufferIntelligence Trigger: Completed system assessment...')
    # AdaptiveBufferIntelligence system assessment has been completed...
