
import numpy as np
from numpy import *


class DemandMatrix():
    #a origin-destination traffic/demand matrix is a matrix NxN where N is the number of hosts in the network. An element n(i,j) of the matrix is the traffic demand between the origin i and the destination j
    
    #initialize demand matrix
    def __init__(self, nHost):
        self.nHost = nHost
        self.DM = np.zeros((self.nHost,self.nHost))


    def getDemandMatrix(self):
        return self.DM[:][:]
    
    def updateDemandMatrix(self, sourceID, destinationID, size, update):
        self.DM[sourceID][destinationID] += update * size
