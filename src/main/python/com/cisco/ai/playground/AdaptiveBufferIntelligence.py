# |Bleeding-Edge Productions|
# This entity describes the design of an intelligent, adaptive buffer allocation engine using Deep Deterministic...
# ...Policy Gradients (DDPG) in an Asynchronous Advantage Actor Critic (A3C) architecture based on the Double Deep...
# ...Q-Networks Prioritized Experiential Learning (DDQN-PER) framework.
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems, Inc.
# Copyright (c) 2019. All Rights Reserved.


# The CISCO Nexus DC switch environment
# Definitions of states, actions, rewards, transitions, emissions, and steady-state initializations are encapsulated...
# ...within this class.
class Nexus(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] Nexus Initialization: Bringing things up...')

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Nexus Termination: Tearing things down...')