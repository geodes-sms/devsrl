import numpy as np

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from numpy.linalg import norm
import math

import random

def calculateDistance(timeseries1, timeseries2):
    return euclideanDistance(timeseries1, timeseries2)
    #return dtwDistance(timeseries1, timeseries2)
    
def euclideanDistance(timeseries1, timeseries2):
    #print("calculate euclidean")
    distance = abs(norm(np.array(timeseries1)) - norm(np.array(timeseries2)))
    return distance

"""
Contrary to its name, the fastdtw library is inexplicably slow.
"""
def dtwDistance(timeseries1, timeseries2):
    print("calculate dtw")
    distance, path = fastdtw(timeseries1, timeseries2, dist=euclidean)
    return distance

def findSplitIndex(trace, timestamp):
    for i in range(0, len(trace)):
        if trace[i][0]>=timestamp:
            return i
            
def findClosest(testState, trace):
    timestamp = testState[0]
    stateLabel = testState[1]
    
    splitIndex = findSplitIndex(trace, timestamp)
    
    #print('Split index is: {}'.format(splitIndex))
    
    #if we find the state immediately at the index
    if trace[splitIndex][1]==stateLabel:
        return trace[splitIndex]
    
    #otherwise find next right and left, and return the closest
    nextRight = None
    for state in trace[splitIndex:]:
        if state[1]==stateLabel:
            nextRight = state
            break
            
    nextLeft = None
    for state in reversed(trace[0:splitIndex]):
        if state[1]==stateLabel:
            nextLeft = state
            break
            
    if not nextRight and nextLeft:
        return nextLeft
    elif not nextLeft and nextRight:
        return nextRight
    else:
        if abs(nextRight[0]-timestamp) < abs(nextLeft[0]-timestamp):
            return nextRight
        else:
            return nextLeft

    raise Exception('Unexpected stuff happened')

def getMax(trace):
    return trace[-1][0]
    
def closestMatchDistance(currentTrace, refTrace, sampleNum=10):
    automatedStates = ['red', 'green', 'yellow']
    candidates = [item for item in currentTrace if item[1] in automatedStates and item[0] < getMax(refTrace)]
    
    diffs = []
    for i in range(0, sampleNum):
        testState = random.choice(candidates)
        closestRefState = findClosest(testState, refTrace)
        diff = abs(testState[0]-closestRefState[0])
        diffs += [diff]
        
    return sum(diffs)/len(diffs)
    
def mse(currentTrace, refTrace):
    automatedStates = ['red', 'green', 'yellow']
    candidates = [item for item in currentTrace if item[1] in automatedStates and item[0] < getMax(refTrace)]
    
    diffs = []
    
    for state in candidates:
        closestRefState = findClosest(state, refTrace)
        diff = state[0]-closestRefState[0]
        diffs += [diff]

    return sum([pow(diff, 2) for diff in diffs])/len(diffs)
    
def countOccurrences(currentTrace, refOccurences):
    distanceVector = np.array([sum(item[1]=='red' for item in currentTrace), sum(item[1]=='green' for item in currentTrace), sum(item[1]=='yellow' for item in currentTrace)]) - refOccurences
    
    distance = 0
    for item in distanceVector:
        distance += abs(item)
        
    return distance
    
def pairwiseDistance(currentTrace, refTrace):
    refRed = [item for item in refTrace if item[1]=='red']
    refGreen = [item for item in refTrace if item[1]=='green']
    refYellow = [item for item in refTrace if item[1]=='yellow']
    currentRed = [item for item in currentTrace if item[1]=='red']
    currentGreen = [item for item in currentTrace if item[1]=='green']
    currentYellow = [item for item in currentTrace if item[1]=='yellow']
    
    length = 0
    redDistance = 0
    for i in range(0, max(len(currentRed), len(refRed))):
        if i >= len(currentRed) or i >= len(refRed):
            redDistance += 1000000
        else:
            redDistance += pow(currentRed[i][0]-refRed[i][0], 2)
        length += i
    #redDistance += abs(len(currentRed)-len(refRed))
    #redDistance = redDistance/length if length>0 else 0
    
    #length = 0
    greenDistance = 0
    for i in range(0, max(len(currentGreen), len(refGreen))):
        if i >= len(currentGreen) or i >= len(refGreen):
            greenDistance += 1000000
        else:
            greenDistance += pow(currentGreen[i][0]-refGreen[i][0], 2)
        length += i
    #greenDistance += abs(len(currentGreen)-len(refGreen))*9
    #greenDistance = greenDistance/length if length>0 else 0
    
    #length = 0
    yellowDistance = 0
    for i in range(0, max(len(currentYellow), len(refYellow))):
        if i >= len(currentYellow) or i >= len(refYellow):
            yellowDistance += 1000000
        else:
            yellowDistance += pow(currentYellow[i][0]-refYellow[i][0], 2)
        length += i
    #yellowDistance += abs(len(currentYellow)-len(refYellow))*9
    #yellowDistance = yellowDistance/length if length>0 else 0
    
    d = redDistance+greenDistance+yellowDistance
    return math.sqrt(d) #pow(math.exp(-1 / (redDistance+greenDistance+yellowDistance)), length)*10 if d!=0 else 0
    
def dummyDistance(currentTas, refTas):
    d = []
    for i in range(0, len(currentTas)):
        d += [abs(pow(refTas[i]-currentTas[i], 1))]
    return sum(d)