import sys
sys.path.insert(0,'./utils')

import numpy as np
import math
from functools import reduce

from tensorforce.environments import Environment

from DevsModel import TrafficSystem
from pypdevs.infinity import INFINITY
from pypdevs.simulator import Simulator

from trace2timeseries import *
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from Distance import *

class DevsEnvironment(Environment):
    def __init__(self):
        print('DevsEnvironment 3')
        self.ta = np.array([0, 0, 0])
        
        f = open('traces/reftrace.devs', 'r')
        self.refTs = trace2timeseries(f)
        f.close()

        super().__init__()
    
    def states(self):
        return dict(type='float', shape=(3,), min_value=0, max_value=120)

    def actions(self):
        actions = dict(type='int', shape=(3,), num_values=5)
        return actions
        
    def actionMask(self):
        return np.array([
                [True, self.ta[0] < 115, self.ta[0] < 120, self.ta[0] > 5, self.ta[0] > 0],
                [True, self.ta[1] < 115, self.ta[1] < 120, self.ta[1] > 5, self.ta[1] > 0],
                [True, self.ta[2] < 115, self.ta[2] < 120, self.ta[2] > 5, self.ta[2] > 0]
                ])

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def close(self):
        super().close()

    def reset(self):
        self.timestep = 0
        self.ta = np.random.randint(size=(3,), low = 5, high = 115)
        states = dict(state=self.ta, action_mask=self.actionMask())
        return states

    def response(self, action):
        #print(action)
        rAction = action[0]
        gAction = action[1]
        yAction = action[2]
        
        update = np.array([0, 0, 0])
        
        if rAction == 0:
            update += np.array([0, 0, 0])
        elif rAction == 1:
            update += np.array([5, 0, 0])
        elif rAction == 2:
            update += np.array([1, 0, 0])
        elif rAction == 3:
            update += np.array([-5, 0, 0])
        elif rAction == 4:
            update += np.array([-1, 0, 0])
        else:
            raise Exception('Unknown action')    
            
        if gAction == 0:
            update += np.array([0, 0, 0])
        elif gAction == 1:
            update += np.array([0, 5, 0])
        elif gAction == 2:
            update += np.array([0, 1, 0])
        elif gAction == 3:
            update += np.array([0, -5, 0])
        elif gAction == 4:
            update += np.array([0, -1, 0])
        else:
            raise Exception('Unknown action')
        
        if yAction == 0:
            update += np.array([0, 0, 0])
        elif yAction == 1:
            update += np.array([0, 0, 5])
        elif yAction == 2:
            update += np.array([0, 0, 1])
        elif yAction == 3:
            update += np.array([0, 0, -5])
        elif yAction == 4:
            update += np.array([0, 0, -1])
        else:
            raise Exception('Unknown action')
        
        return self.ta+update
    
    def getMeantime(self, period):
        return reduce(lambda a, b: a+b, [item[0] for item in period]) / len(period)

    def getMeantimesVector(self, trace):
        meanTimes = []
        
        for period in trace:
            meanTime = self.getMeantime(period)
            meanTimes += [meanTime]

        return meanTimes
    
    def termFunc(self, clock, model):
        #print(model.trafficLight.generated)
        if model.trafficLight.generated > 50:
            return True
        else:
            return False
        
    def reward_compute(self):
        reward = 0

        trafficSystem = TrafficSystem(name="trafficSystem")
        sim = Simulator(trafficSystem)
        #sim.setAllowLocalReinit(True)
        #sim.setTerminationTime(400.0)
        sim.setTerminationCondition(self.termFunc)
        sim.setCustomTracer("SimpleTracer", "SimpleTracer", ["traces/trace.devs"])
        sim.setClassicDEVS()
        trafficSystem.trafficLight.updateAllTa(self.ta)
        sim.simulate()
        
        f = open("traces/trace.devs", "r")
        ts2 = trace2timeseries(f)
        f.close()
        
        """
        currentTrace = []
        f = open('traces/trace.devs', 'r')
        currentTrace = getTrace(f)
        f.close()
        """
        
        distance = calculateDistance(self.refTs, ts2)
        
        #reward = -pow(distance, 3)
        
        #distance = mse(currentTrace, self.refTrace)
        #reward = -pow(distance, 1)
        
        #distance = (currentTrace, self.refOccurences)
        #distance = dummyDistance(currentTrace, self.refTrace)
        #distance = dummyDistance(self.ta, [60, 50, 10])
        reward = -pow(distance, 2)

        return reward

    def execute(self, actions):
        self.timestep += 1
        self.ta = self.response(actions)
        #print(self.ta)
        #print(self.ta[0])
        reward = self.reward_compute()
        terminal = False
        
        states = dict(state=self.ta, action_mask=self.actionMask())
        return states, terminal, reward