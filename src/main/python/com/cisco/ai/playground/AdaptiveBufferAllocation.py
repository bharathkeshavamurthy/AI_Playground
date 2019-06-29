# |Bleeding-Edge Productions|
# This entity describes an evaluation of Reinforcement Learning techniques for adaptive, intelligent buffer...
# ...management in CISCO's Nexus Data Center switches.
# Authors: Imran Pasha, Bharath Keshavamurthy
# Email: <ipasha, bkeshava>@cisco.com
# Organization: CISCO Systems, Inc.
# Copyright (c) 2019. All Rights Reserved.

# ----LEGACY APPROACH----

# Salient Features:
# State - Global Allocation Metrics
# Multi-Step Actions - Increase P1Q1, Decrease P2Q1, No change to P3Q3, and so on...
# The increased state and action spaces lead to increased computational complexity - warrants the use of DDQNs with PER
# Object-Oriented approach to the design of the Switch Environment: It's cleaner this way
# Continuous Reward Metrics: Throughput / Packet Drop Rate per episode - Discrete Rewards seem haphazard in this domain
# Double Deep-Q-Network: Prevent over-estimation of Q-values - One NN for Estimation and another for Selection
# Prioritized Experiential Replay: Priority based experience sampling from memory - Accelerated, Controlled Convergence

# The imports
import enum


# An enumeration listing possible buffer <priority> types
class Priority(enum):
    # High Priority Buffer
    HIGH_PRIORITY = 1

    # Low Priority Buffer
    LOW_PRIORITY = 2


# The buffer metrics of individual queues in the Nexus Data Center switches
class Queue(object):

    # The initialization sequence
    def __init__(self, _queue_id, _required_minimum_buffer_capacity, _allowed_maximum_buffer_capacity,
                 _allocated_buffer_units, _queue_availability, _packet_drop_count, _priority):
        # The queue identifier for logistical planning
        self.queue_id = _queue_id
        print('[INFO] Queue Initialization: Bringing up Queue [{}]...'.format(self.queue_id))
        # Old terminology: QMin - The required minimum space in the buffer - Design requirement
        self.required_minimum_capacity = _required_minimum_buffer_capacity
        # Old terminology: QMax - The maximum allowed buffer capacity - Design constraint
        self.allowed_maximum_buffer_capacity = _allowed_maximum_buffer_capacity
        # Old terminology: QAlloc - The number of buffer slots allocated by the agent - Operational
        self.allocated_buffer_units = _allocated_buffer_units
        # The queue availability indicator flag - Operational
        self.queue_availability = _queue_availability
        # Old terminology: QDrop
        # The number of packets dropped at this queue due to unavailable buffer slots - Objective
        self.packet_drop_count = _packet_drop_count
        # The priority flag for this queue instance - <Priority enumeration object>
        self.priority = _priority

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Queue Termination: Tearing Queue [{}] down...'.format(self.queue_id))


# The metrics of individual ports in the Nexus Data Center switches
class Port(object):

    # The initialization sequence
    def __init__(self, _port_id, _queue_data, _pool_size, _port_availability, _port_drop_count):
        # The port identifier for logistical planning
        self.port_id = _port_id
        print('[INFO] Port Initialization: Bringing up Port [{}]'.format(self.port_id))
        # The Queue object corresponding to the port
        self.queue_data = _queue_data
        # Old terminology: PAvail - The buffer slots pool size for this port instance - Operational
        self.pool_size = _pool_size
        # The port availability indicator flag - Operational
        self.port_availability = _port_availability
        # Old terminology: PDrop - The number of packets dropped at this port - Objective
        self.port_drop_count = _port_drop_count

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Port Termination: Tearing Port [{}] down...'.format(self.port_id))


# The state of the environment is encapsulated in this Python Class
class State(object):

    # The initialization sequence
    def __init__(self, _port_data):
        print('[INFO] State Initialization: Creating a new state object...')
        # The collection of ports (and the queues within each port) constitute a state in this environment
        self.port_data = _port_data

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] State Initialization: Tearing down the state object...')


# The multi-ported finite M/M/1 queueing system in CISCO's Nexus Data Center switches
class Environment(object):
    # The number of ports in the switch
    NUMBER_OF_PORTS = 3

    # The number of queues per port
    NUMBER_OF_QUEUES = 3

    # The allowed buffer capacity per port
    ALLOWED_BUFFER_CAPACITY = 28

    # The maximum allowed capacity in a high-priority queue's buffer
    ALLOWED_BUFFER_CAPACITY_HIGH_PRIORITY = 10

    # The maximum allowed capacity in a low-priority queue's buffer
    ALLOWED_BUFFER_CAPACITY_LOW_PRIORITY = 18

    # The initialization sequence
    def __init__(self):
        print('[INFO] Environment Initialization: Bringing things up...')
        # Initialize the environment by creating the very first state
        ports = []
        for p in range(self.NUMBER_OF_PORTS):
            queues = []
            # PxQ0 is a high priority queue - faster service rate
            # PxQ1 and PxQ2 are low priority queues - comparatively slower service rate
            for q in range(self.NUMBER_OF_QUEUES):
                queue_identifier = 'P' + str(p) + 'Q' + str(q)
                queue = Queue(_queue_id=queue_identifier,
                              _required_minimum_buffer_capacity=0,
                              _allowed_maximum_buffer_capacity=(
                                  lambda: self.ALLOWED_BUFFER_CAPACITY_LOW_PRIORITY,
                                  lambda: self.ALLOWED_BUFFER_CAPACITY_HIGH_PRIORITY)[q == 0](),
                              _allocated_buffer_units=0,
                              _queue_availability=True,
                              _packet_drop_count=0,
                              _priority=(lambda: Priority.LOW_PRIORITY,
                                         lambda: Priority.HIGH_PRIORITY)[q == 0]())
                queues.append(queue)
            port_identifier = 'P' + str(p)
            port = Port(_port_id=port_identifier,
                        _queue_data=queues,
                        _pool_size=self.ALLOWED_BUFFER_CAPACITY,
                        _port_availability=True,
                        _port_drop_count=0)
            ports.append(port)
        self.state = State(ports)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Environment Termination: Tearing things down...')


# The agent that executes allocation recommendations on the environment, optimally
class Agent(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] Agent Initialization: Bringing things up...')

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Agent Termination: Tearing things down...')


# The memory of the agent is categorized into a separate Python Class
# This entity facilitates prioritized experiential replay
class Memory(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] Agent-Memory Initialization: Bringing things up...')

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Agent-Memory Termination: Tearing things down...')


# This class encapsulates the design of an intelligent buffer allocation agent by leveraging tools from Q-Learning.
class AdaptiveBufferAllocation(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] AdaptiveBufferAllocation Initialization: Bringing things up...')

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] AdaptiveBufferAllocation Termination: Tearing things down...')
