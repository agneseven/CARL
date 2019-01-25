



class packet:
    #construct topology
    def __init__(self, flowID, packetID, sourceID, currentNodeID, destinationID, path):
        self.sourceID = sourceID
        self.flowID = flowID

        self.packetID = packetID
        self.currentNodeID = currentNodeID
        self.destinationID = destinationID
        self.path = path
        self.nextnodeID = self.path[1]

    def updatepathID(self, flowID, packetID):
        self.path = self.path + [flowID, packetID]
    
    def updateCurrentNode(self, currentNodeID):
        self.currentNodeID = currentNodeID

    def getCurrentNode(self):
        return self.currentNodeID

#    def removePacket(self, packetID,currentNodeID):
#        np.delete(arr, 1, 0)

    def setDepartureTime(self, packetID, departureTime):
        departureTime = departureTime

    def getDepartureTime(self):
        return departureTime

    def getpacketID(self):
        return self.packetID
    
    def getflowID(self):
        return self.flowID

    def getsourceID(self):
        return self.sourceID

    def getdestinationID(self):
        return self.destinationID

    def getpath(self):
        return self.path

    def getnextnodeID(self):
        return self.nextnodeID

    def updateNextNode(self, packetID, newNextNode):
        self.nextnodeID = newNextNode
