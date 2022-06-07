import os
import sys
sys.path.insert(0,'../utils')

import pickle
from statistics import mean
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import csv
from itertools import chain
import random

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.environments import Environment

from DevsEnvironment3 import DevsEnvironment


from ProgressBar import printProgressBar

class Runner:

    def __init__(self):
        self.environment = Environment.create(
            environment=DevsEnvironment,
            max_episode_timesteps=100)
        
        """
        self.agent = Agent.create(
            agent='tensorforce',
            environment=self.environment,
            update=64,
            optimizer=dict(optimizer='adam', learning_rate=1e-4),
            objective='policy_gradient',
            reward_estimation=dict(horizon=1)
        )
        """
        
        self.agent = Agent.create(
            agent='ppo',
            environment=self.environment,
            batch_size=10,
            exploration=0.01,
            #exploration=0.05,
            likelihood_ratio_clipping=0.2,
            discount=1.0,
            entropy_regularization=0.0,
            l2_regularization=0.0,
            subsampling_fraction=0.33,
            multi_step=10,
            network=[
                dict(type='dense', size=64),
                dict(type='register', tensor='action1-embedding'),
                dict(type='dense', size=64),
                dict(type='register', tensor='action2-embedding'),
                dict(type='dense', size=64)
            ]
        )
        
        """
        self.agent = Agent.create(
            agent='random',
            environment=self.environment
        )
        """
        
    def prepare(self, trainOption, saveOption, loadOption):
        if trainOption:
            self.train(trainOption)
        if saveOption:
            self.saveAgent(saveOption)
        if loadOption:
            self.loadAgent(loadOption)

    def train(self, trainOption):
        print('_____________________________ TRAINING _____________________________')
        episodes = trainOption
        printProgressBar(0, episodes, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for i in range(episodes):
            states = self.environment.reset()
            terminal = False
            while not terminal:
                actions = self.agent.act(states=states)
                states, terminal, reward = self.environment.execute(actions=actions)
                self.agent.observe(terminal=terminal, reward=reward)
            printProgressBar(i + 1, episodes, prefix = 'Progress:', suffix = 'Complete', length = 50)
        print('...AGENT TRAINED')

    def saveAgent(self, saveOption):
        print('_____________________________ SAVING AGENT _____________________________')
        checkpointsDirectory = 'checkpoints'
        agentDirectory = '{}/{}'.format(checkpointsDirectory, saveOption)
        self.agent.save(directory=agentDirectory)
        
        f = open('{}/{}'.format(checkpointsDirectory, 'latest.agents'), 'w')
        f.write(saveOption)
        f.close()
        
        print('...AGENT {} SAVED'.format(saveOption))
        
    def loadAgent(self, loadOption):
        print('_____________________________ LOADING AGENT _____________________________')
        checkpointsDirectory = 'checkpoints'
        
        agentName = ''
        
        if loadOption:
            agentName = loadOption
            agentDirectory = '{}/{}'.format(checkpointsDirectory, agentName)
        else:
            f = open('{}/{}'.format(checkpointsDirectory, 'latest.agents'), "r")
            agentName = f.read()
            agentDirectory = '{}/{}'.format(checkpointsDirectory, agentName)
            f.close()
        
        print('AGENT DIRECTORY: {}'.format(agentDirectory))
        
        if not os.path.exists(agentDirectory):
            sys.exit('Agent does not exist')
            
        self.agent = Agent.load(directory=agentDirectory)
        print('...AGENT {} LOADED'.format(agentName))

    def run(self, runOption):
        print(runOption)
        self.environment.reset()
        print('_____________________________ RUN _____________________________')
        
        self.environment.ta = np.array(runOption)
        states = self.environment.ta

        internals = self.agent.initial_internals()
        terminal = False

        ### Run an episode
        tas = [self.environment.ta]
        rewards = []
        while not terminal:
            actions, internals = self.agent.act(states=states, internals=internals, independent=True)
            #print(actions)
            states, terminal, reward = self.environment.execute(actions=actions)
            tas += [states['state']]
            rewards += [reward]
        
        print(rewards)
        
        ### Plot the run
        self.plot(tas, rewards)
        
    
    def plot(self, tas, rewards):
        #plt.figure(figsize=(12, 4))
        fig, axs = plt.subplots(2)
        axs[0].set_ylim([-5, 125])
        rTa = [ta[0] for ta in tas]
        gTa = [ta[1] for ta in tas]
        yTa = [ta[2] for ta in tas]
        xmaxDashed = max(len(rTa), len(gTa), len(yTa))-1
        axs[0].plot(range(len(rTa)), rTa, color = '#ff4a59')
        axs[0].plot(range(len(gTa)), gTa, color = '#22ff00')
        axs[0].plot(range(len(yTa)), yTa, color = '#ffff00')
        axs[0].hlines(y=60, xmin=0, xmax=xmaxDashed, color='#FF0000', linestyle='dashed')
        axs[0].hlines(y=50, xmin=0, xmax=xmaxDashed, color='#00FF00', linestyle='dashed')
        axs[0].hlines(y=10, xmin=0, xmax=xmaxDashed, color='#CCC900', linestyle='dashed')
        axs[0].title.set_text('ta')
        
        axs[1].plot(range(len(rewards)), rewards, color = '#e87000')
        axs[1].title.set_text('Reward')
        
        #plt.title('Time advance vs. redTa')
        plt.show()
        
    def evaluate(self, evalStrategy):
        print('_____________________________ EVALUATION _____________________________')
        print(runner.agent)
        
        conditions = []
        
        print(evalStrategy)
        
        if evalStrategy == "interval-20":
            #+/-20% evaluation -- 1920 combinations
            for i in range(48, 72):
                for j in range(40,60):
                    for k in range(8,12):
                        conditions += [(i,j, k)]
        elif evalStrategy == "far-initials":
            #Evaluation on far initial conditions -- 1000 combinations
            for i in chain(range(10, 15), range(90, 95)):
                for j in chain(range(10,15), range(85, 90)):
                    for k in chain(range(1,5), range(75, 81)):
                        conditions += [(i,j, k)]
        elif evalStrategy == "short":
            #Short evaluation -- 400 combinations
            for i in range(55, 65):
                for j in range(45,55):
                    for k in range(8,12):
                        conditions += [(i,j, k)]
        elif evalStrategy == "alternative-30-20-15":
            #Evaluation tailored to the alternative-30-20-15 system -- 1000 combinations
            for i in range(25, 35):
                for j in range(15, 25):
                    for k in range(10, 20):
                        conditions += [(i,j, k)]
        elif evalStrategy == "alternative-50-40-15":
            #Evaluation tailored to the alternative-50-40-15 system -- 1000 combinations
            for i in range(45, 55):
                for j in range(35, 45):
                    for k in range(10, 20):
                        conditions += [(i,j, k)]
        elif evalStrategy == "alternative-80-30-10":
            #Evaluation tailored to the alternative-80-30-10 system -- 1000 combinations
            for i in range(75, 85):
                for j in range(25, 35):
                    for k in range(5, 15):
                        conditions += [(i,j, k)]
        elif evalStrategy == "randomsample":
            #Random sample  -- 1000 combinations
            for i in random.sample(range(5, 115), 10):
                for j in random.sample(range(5, 115), 10):
                    for k in random.sample(range(5, 115), 10):
                        conditions += [(i,j, k)]
        elif evalStrategy == "randomsample2":
            file = open('sample-001.p','rb')
            conditions = pickle.load(file)
            file.close()
        else:
            raise Exception('Unknown evaluation strategy')
        
        evalData = []
        evalDirectory = 'eval'
        
        #conditions += [(40, 20, 10)]
        #conditions += [(30, 20, 10)]
        
        printProgressBar(0, len(conditions), prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        for i in range(len(conditions)):
            condition = conditions[i]
            #print(condition)
            self.environment.reset()

            self.environment.ta = np.array(condition)
            states = self.environment.ta

            internals = self.agent.initial_internals()
            terminal = False

            ### Run an episode
            tas = [self.environment.ta]
            rewards = []
            while not terminal:
                actions, internals = self.agent.act(states=states, internals=internals, independent=True)
                states, terminal, reward = self.environment.execute(actions=actions)
                tas += [states['state']]
                rewards += [reward]

            #print(rewards)
            evalData += [[condition[0], condition[1], condition[2]] + rewards]
            #print(mean(rewards))
            #print("-------")
            #print(evalData)
            #print("-------")
            printProgressBar(i + 1, len(conditions), prefix = 'Progress:', suffix = 'Complete', length = 50)
            
        
        with open('{}/evalData.csv'.format(evalDirectory),'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for data in evalData:
                writer.writerow(data)
        print('Data saved')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--train', help='Preparing agent with training', nargs='?', const=500, type=int)
    parser.add_argument('-s','--saveas', help='Save agent by name', nargs='?', const='agent-{}'.format(int(time.time())), type=str)
    parser.add_argument('-l','--load', help='Load agent by name', type=str)
    parser.add_argument('-r','--run', help='Running with values', type=int, nargs=3)
    parser.add_argument('-e','--evaluate', help='Evaluate', type=str, nargs=1)
    args = parser.parse_args()
    
    if args.saveas and not args.train:
        parser.error('--saveas requires --train')
    #if args.evaluate and not args.load:
    #    parser.error('--evaluate requires --load')
    if not args.run:
        print("WARNING: Agent won't run because --run was not set.")
    
    runner = Runner()
    runner.prepare(args.train, args.saveas, args.load)
    
    if args.run:
        runner.run(args.run)
        
    if args.evaluate:
        runner.evaluate(args.evaluate[0])
        
    print('Closing agent and environment')
    runner.agent.close()
    print('Agent closed')
    runner.environment.close()
    print('Environment closed')
    sys.exit()