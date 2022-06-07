import re
import numpy as np

from enum import Enum

class State(Enum):
    idle = 0
    red = 10
    green = 20
    yellow = 30
    manual = 40

def trace2timeseries(trace):
    ts = []
    
    lastTimestamp = 0
    
    for row in trace.readlines():
        currentTimestamp = getTimestamp(row)
        rawState = getState(row)
        
        duration = currentTimestamp - lastTimestamp

        for _ in range(duration):
            ts += getEncodedState(rawState)

        lastTimestamp = currentTimestamp
        
    return np.array(ts)
    
def trace2timeseries2(trace):
    ts = []
    
    lastTimestamp = 0
    
    for row in trace.readlines():
        currentTimestamp = getTimestamp(row)
        rawState = getState(row)
        
        duration = currentTimestamp - lastTimestamp

        for i in range(duration):
            ts += getEncodedState2(lastTimestamp+i, rawState)

        lastTimestamp = currentTimestamp
        
    return np.array(ts)
    
def trace2timeseries3(trace):
    ts = []
    
    lastTimestamp = 0
    lastState = None
    
    for row in trace.readlines():
        currentTimestamp = getTimestamp(row)
        rawState = getState(row)
        
        duration = currentTimestamp - lastTimestamp

        #print('{}+{}: {}'.format(currentTimestamp, duration, lastState))
                
        for i in range(duration):
            ts += getEncodedState3(lastTimestamp+i, lastState)

        lastTimestamp = currentTimestamp
        lastState = rawState
    
    ts += getEncodedState3(lastTimestamp, lastState)
    
    #print(ts)
    
    return np.array(ts)

def getTrace(traceFile):
    trace = []
    
    for row in traceFile.readlines():
        timestamp = getTimestamp(row)
        state = getState(row)
        
        trace += [(timestamp, state)]
        
    return trace
    
def getGroupedTrace(traceFileLocation, automatedStates):
    f = open(traceFileLocation, 'r')
    trace = getTrace(f)
    f.close()
    
    groupedTrace = []
    currentPeriod = []

    for item in trace:
        state = item[1]
        
        if(state in automatedStates):
            currentPeriod += [item]
        elif(state == 'manual'):
            groupedTrace += [currentPeriod]
            currentPeriod = []
            
    return groupedTrace
    
def getTimestamp(row):
    return int(row.split(',')[0])
    
def getState(row):
    state = row.split(',')[1]
    state = re.sub('[^A-Za-z0-9]+', '', state)
    return state
    
def getEncodedState(rawState):
    return [State[rawState].value]
    
def getEncodedState2(timestamp, rawState):
    return [[timestamp, State[rawState].value]]
    
def getEncodedState3(timestamp, rawState):
    return [[timestamp, rawState]]