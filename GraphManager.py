
from topoGeneration import *
import time
from packet import *

class GraphManager():

    def __init__(self, nNodes):#bandwidth_list, delay_list):
        self.sizeQueue = [0] *  nNodes # list with all 0
#        self.points_list = points_list
#        self.bandwidth_list = bandwidth_list
#        genG = topoGeneration(nNodes, points_list, bandwidth_list, delay_list)
#        self.G = genG.Graph()
#        genG.showGraph()
        self.alivepk = []
        self.lstPackets = [[] for _ in xrange(nNodes)] #list of packet object
#        self.delay_list = delay_list
        self.aliveflow = []
        self.alivePaths = []


    def updateSizeQueue(self, nodeID, newPk):
#        print("before queue size")
#        print(self.sizeQueue[nodeID])
        self.sizeQueue[nodeID] = self.sizeQueue[nodeID] + newPk
#        print("then queue size")
#        print(self.sizeQueue[nodeID])

    def showSizeQueue(self, nodeID):
        return self.sizeQueue[nodeID]

    def updatelstPackets(self, nodeID, packet):
        self.lstPackets[nodeID].append(packet)

    def dellstPackets(self, nodeID, packet):
        self.lstPackets[nodeID].remove(packet)

    def showIndexPacket(self, nodeID, packet):
        return self.lstPackets[nodeID].index(packet)

    def getfirstpacket(self, nodeID):
        #first = [item[0] for item in self.lstPackets[nodeID]]
        return self.lstPackets[nodeID][0]
    
    def getPoints_list(self):
        #first = [item[0] for item in self.lstPackets[nodeID]]
        return self.points_list
    
    def getBandwidth_list(self):
        #first = [item[0] for item in self.lstPackets[nodeID]]
        return self.bandwidth_list

    def getGraph(self):
        return self.G

    def getalivepk(self):
        return self.alivepk

    def updatealivepk(self, flowID, pkID):
        self.alivepk.append([flowID, pkID])

    def delalivepk(self, flowID, pkID):
        self.alivepk.remove([flowID, pkID])
    
    def updatealiveflow(self, flowID):
        self.aliveflow.append(flowID)
    
    def delaliveflow(self, flowID):
        self.aliveflow.remove(flowID)
    
    def getaliveflow(self):
        return self.aliveflow

#    def getDelayEdges(self, nodeA, nodeB):
##        print(self.delay_list)
#        delay = self.delay_list[nodeA, nodeB]
#        return delay

    def updatealivePaths(self, path):
        self.alivePaths.append(path)
    
    def getalivePaths(self):
        return self.alivePaths
    
    def delalivePaths(self, path_):
        self.alivePaths.remove(path_)

    def delalivePaths(self, path_):
        self.alivePaths.remove(path_)

