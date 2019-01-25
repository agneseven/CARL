



class source:
    #construct topology
    def __init__(self, sourceID, nPackets):
        self.sourceID = sourceID
        self.nPackets = nPackets

    def updateNpackets(self, numbNewPackets):
        self.nPackets = self.nPackets + numbNewPackets

