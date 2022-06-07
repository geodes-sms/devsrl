from pypdevs.util import runTraceAtController
import sys

class SimpleTracer():
    
    def __init__(self, uid, server, filename):
        """
        Constructor

        :param uid: the UID of this tracer
        :param server: the server to make remote calls on
        :param filename: file to save the trace to, can be None for output to stdout
        """
        if server.getName() == 0:
            self.filename = filename
        else:
            self.filename = None
        self.server = server
        self.prevtime = (-1, -1)
        self.uid = uid


    def startTracer(self, recover):
        """
        Starts up the tracer

        :param recover: whether or not this is a recovery call (so whether or not the file should be appended to)
        """
        if self.filename is None:
            self.verb_file = sys.stdout
        elif recover:
            self.verb_file = open(self.filename, 'a+')
        else:
            self.verb_file = open(self.filename, 'w')


    def stopTracer(self):
        """
        Stops the tracer
        """
        self.verb_file.flush()

    def trace(self, time, text):
        """
        Actual tracing function

        :param time: time at which this trace happened
        :param text: the text that was traced
        """
        string = "%s\n" % text
        
        try:
            self.verb_file.write(string)
        except TypeError:
            self.verb_file.write(string.encode())
        


    def traceInternal(self, aDEVS):
        """
        Tracing done for the internal transition function

        :param aDEVS: the model that transitioned
        """
        text = "{}, ".format(int(aDEVS.time_last[0]))
        text += "%s" % str(aDEVS.state)
        runTraceAtController(self.server, 
                             self.uid, 
                             aDEVS, 
                             [aDEVS.time_last, '"' + text + '"'])


    def traceConfluent(self, aDEVS):
        """
        Tracing done for the confluent transition function

        :param aDEVS: the model that transitioned
        """
        text = "{}, ".format(int(aDEVS.time_last[0]))
        text += "%s" % str(aDEVS.state)
        runTraceAtController(self.server, 
                             self.uid, 
                             aDEVS, 
                             [aDEVS.time_last, '"' + text + '"'])


    def traceExternal(self, aDEVS):
        """
        Tracing done for the external transition function

        :param aDEVS: the model that transitioned
        """
        text = "{}, ".format(int(aDEVS.time_last[0]))
        text += "%s" % str(aDEVS.state)
        runTraceAtController(self.server, 
                             self.uid, 
                             aDEVS, 
                             [aDEVS.time_last, '"' + text + '"'])


    def traceInit(self, aDEVS, t):
        """
        Tracing done for the initialisation

        :param aDEVS: the model that was initialised
        :param t: time at which it should be traced
        """
        text = "{}, ".format(int(aDEVS.time_last[0]))
        text += "%s" % str(aDEVS.state)
        runTraceAtController(self.server, 
                             self.uid, 
                             aDEVS, 
                             [t, '"' + text + '"'])


    def traceUser(self, time, aDEVS, variable, value):
        text = "\n"
        text += "\tUSER CHANGE in model <%s>\n" % aDEVS.getModelFullName()
        text += "\t\tAltered attribute <%s> to value <%s>\n" % (variable, value)
        # Is only called at the controller, outside of the GVT loop, so commit directly
        self.trace(time, text)