## Requirements
* Install PythonPDEVS as desrcibed on its [official page](https://msdl.uantwerpen.be/documentation/PythonPDEVS/installation.html).
* Install required Python libraries: ```pip install -r requirements.txt```.

## Structure

* ```/checkpoints``` - Agent checkpoints.
* ```/debugtraces``` - Debug traces generated during runs.
* ```/devs/devs_runner``` - Generates trace to be learned.
* ```/eval``` - Folder storing previous evaluations.
* ```/traces``` - Prepared traces for learning.
  * ```/traces/reftrace.devs``` - Baseline reference trace.
* ```/utils``` - Utility functions.
* ```DevsEnvironment3.py``` - RL environment that instantiates a PyPDEVS simulator to calculate the reward based on the simulation traces.
* ```DevsModel.py``` - Model to be learned.
* ```Distance.py``` - Calculation of distance metrics.
* ```Runner.py``` - Main runner. 
* ```SimpleTracer.py``` - Tracer for PyPDEVS to support the custom format.
  
## Usage
* Run with ```python Runner.py```.
* Parameters:
  * ```--train [number of episodes]``` or ```--train``` with the default number of episodes = 500. If the parameter is not used, the runner loads the last saved agent.
  * ```--saveas [name]``` or ```--saveas``` with the default name being Agent-[epochtime]. Requires ```--train```.
  * ```--load [name]``` or ```--load``` with the default name being the latest trained agent. (Latest name saved in the ```checkpoints/latest.agents``` file.)
  * ```--run [initial condition]```, e.g. ```--run 45 30 20```
  * ```--evaluate [name]```  – evaluates the agent's performance after run. Implemented evaluations: ```interval-20```, ```far-initials```, ```short```, ```alternative-30-20-15```, ```alternative-50-40-15```, ```alternative-80-30-10```, ```randomsample```, ```randomsample2```