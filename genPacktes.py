#import packet as pk
##import source as sc
#from source import *
#from packet import *
#from EventManager import *
#import networkx as nx
#import numpy as np
#from ecmp import *
#from simpy import *
#
#def genPackets(flowID, tempoOra2, timeout2, EM, GM, allpaths, numpackets):
#    istante2 = tempoOra2 + 1
#    Npks = -1
#    while(Npks < numpackets):
#        #                        print("time {}".format(time.ctime(istante2)))
#        newpk = packet( flowID, pkID, sourceID, sourceID, destinationID, path)
#        savept = path + [flowID, pkID]
#        allpaths.append(path + [flowID, pkID])
#        newpk.updatepathID(flowID, pkID)
#        GM.updatealivepk(flowID, pkID)
#        pkID = pkID + 1
#        EM.GENERATE(GM, newpk)
#            
#            
#        Npks += 1
#    return allpaths
#
#
#Timer(transmissionTime, self.DEPARTURE, (GM, newpk, newindex)).start()
