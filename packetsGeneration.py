import packet as pk
#import source as sc
from source import *
from packet import *
from EventManager import *
import networkx as nx
import numpy as np
from ecmp import *
from simpy import *
from genPacktes import *
from semi_oblivious import *
from DemandMatrix import *
from collections import Counter




def frange(start, stop, step):
     x = start
     while x < stop:
         yield x
         x += step

class packetsGeneration:
    #construct topology
    def __init__(self, nFlow):
        self.nFlow = nFlow
    

#
    def generation(self, GM, nNode, nHost, nSwitch, graph, model, parameters, RL1, dotFormat):
        """
            function to generate flows from a random source to a random destination. The traffic generation model is a Poission process (we fix lambda = arrival rate)
            Args:
            GM
            nNode
            graph
            model
            parameters
            RL1
            
            """

        now1 = time.time()

        EM = EventManager()
        i = 0
        
        #generate random source and destination of a flow---------------
#        sourceID = 2#random.randint(0, nNode-1)
#        destinationID = 3#random.randint(0, nNode-1)
#        #generate again the destination if it overlaps with the source------------------
#        while(destinationID == sourceID):
#            destinationID = random.randint(0, nNode-1)

#        #flowID of the first flow generated: flowID = 0-------------------
#        flowID = flowID + 1
        flowID = -1 #inizialization of the ID of flows that will be generated
        pkID = 0 #inizialization of the ID of first packet of the flow
        


        packetsize = 100
        #array that stores all the paths of the other flows present in the system. We need it for the RL alghoritm
#        allpathsRL = []

        allpaths = []

        DemM = DemandMatrix(nHost)
        DemM.getDemandMatrix()
#        print('DM', DemM.getDemandMatrix())

        #NOTE: all the packets belonging to the same flow follow the same path
        PathMODE = str(parameters[5]) #algorithm used to evaluate the path between source and destinatio
        #available options:
        #RL = reinforcement learning
        #OSPF = shortest path = path with the minimum weight = if all the weights are 1, it's the path with the minimum number of hops
        #ECMP = equal cost multi-path = all the shortest paths are evaluated, then, every time a new flow is generated, the shortest path is chosen according to the round-robin tecnique
        
        
        pathnotfoud = 0 #index to count how many times the RL algorithm didn't work

    #--------------------------------
        allecmpPaths = all_ecmp_flow(nNode, graph, nHost)
#        print("allecmpPaths {}".format(allecmpPaths))

        now = time.time()
        startTime = now

        istante = time.time()
        #end of the simulation: input parameter
        min = int(parameters[6])
        timeout = time.time() + 60*min #tot minutes from now on
#        print("fine", time.ctime(timeout))
        tempoOra = time.time()
#----------------lambda!!!!!!!!!------------------------------------------------------------------------
        interArrivalTime = np.random.poisson(50)
#        print("Poisson ", interArrivalTime)
        istante = tempoOra + interArrivalTime
        print("startTime {}, endtime {}, nextarrival {}".format( time.ctime(startTime), time.ctime(timeout), time.ctime(istante)))

        while (tempoOra < timeout):

            tempoOra = time.time()
#            print('now {}', tempoOra)
#            print('next istante {}', istante)

            if(tempoOra >= istante ):
#                print('qui')
#generation of random source and destination (only hosts can be source or destination)
                sourceID = 0#random.randint(0, nHost-1)
                destinationID = 2#random.randint(0, nHost-1)#random.randint(0, nNode-1)
                while(destinationID == sourceID):
                    destinationID = random.randint(0, nHost-1)
            
                if parameters[5] == 'so':
                #path semi-oblivious
                    DemM.updateDemandMatrix(sourceID, destinationID, packetsize, 1)
                    DM = []
                    DM = DemM.getDemandMatrix()
#                    print('DM 2', DM)

                flowID = flowID + 1
                GM.updatealiveflow(flowID)
                listSizeQueues = []
                for queueNode in range(nNode):
                    listSizeQueues.append(GM.showSizeQueue(queueNode))
                print('size nodes', listSizeQueues)
                
#evaluation of the flows/paths alive every time that a new flow is generated-----
                allpathsRL2 = []
                allpathsRL2 = GM.getalivePaths()
                new_allpathsRL2 = list(allpathsRL2)
                



                
#generation of the path according to the chosen MODE------------------------------
                if( PathMODE == 'rl'):

#dummy paths if the current number of active paths is lower than the maximum number of flows that can coexist in the system.

                    if(len(allpathsRL2) < self.nFlow): #numero flussi totali previsti???????????????
                        for l in range(self.nFlow-len(new_allpathsRL2)-1):#numero flussi totali previsti???????????????

                            DummyPath = [0, 0]
                            new_allpathsRL2.append(DummyPath)

#                    print('....allpathsRL2', allpathsRL2)
                    path, win, loop, fail = RL1.testAlgo(8, new_allpathsRL2, graph, model, sourceID, destinationID)
                    if(loop == 1):
                        path, win, loop, fail = RL1.testAlgo(8, new_allpathsRL2, graph, model, destinationID, sourceID)
                        path = list(reversed(path))
                        print("path not found with RL 1")

                        if(loop == 1):
                            print("path not found with RL 2")
                            path = nx.dijkstra_path(graph, sourceID, destinationID)
                            pathnotfoud += 1
            
                elif(PathMODE == 'ospf'):
                    path = nx.dijkstra_path(graph, sourceID, destinationID)
                elif( PathMODE == 'ecmp'):
                    allecmpPaths, pathIndex, indice_path2, path  = ecmp_path(sourceID, destinationID, allecmpPaths)
                elif( PathMODE == 'so'):
                    allSOpath(dotFormat, DemM.getDemandMatrix(), nHost)
                    path  = semi_oblivious(dotFormat, sourceID, destinationID, nHost)

#                print('alive paths 2.5', GM.getalivePaths())
#                if(len(allpathsRL2) == 0):
                GM.updatealivePaths(path)

#                print('last found path', path)
#                print('alive paths 3', GM.getalivePaths())

#                GM.updatealivepk(flowID, pkID)
                pkID = 0

#                allpathsRL.append(path)
#                print('tutti i path vivi: ', allpathsRL)

    #evaluation of the collisions : overlapping links and nodes among the alive paths
#                allpaths.append(path)

                
#                min = int(parameters[6])
#                istante += 10
                interArrivalTime2 = np.random.poisson(50)
#                print("Poisson2 ", interArrivalTime2)
                istante += 3#interArrivalTime2
                print('istante', istante)
                timeout2 = istante #time.time() + 60*2 #4 minutes from now
                tempoOra2 = time.time()
                istante2 = tempoOra2 + 1
                print("timeStart {} timeEnd {} istante {}".format(time.ctime(tempoOra2), time.ctime(timeout2), time.ctime(istante2)))

#                allpaths = genPackets(flowID, tempoOra2, timeout2, EM, GM, allpaths)
                Npks = -1
                numpackets = 10
                while(Npks < numpackets):
        #                        print("time {}".format(time.ctime(istante2)))
                    newpk = packet( flowID, pkID, sourceID, sourceID, destinationID, path)
                    savept = path + [flowID, pkID]
                    allpaths.append(path + [flowID, pkID])
                    newpk.updatepathID(flowID, pkID)
                    GM.updatealivepk(flowID, pkID)
                    pkID = pkID + 1
                    EM.GENERATE(GM, newpk, DemM, packetsize)
        
        
                    Npks += 1
                    
                    if parameters[5] == 'so':
                    #path semi-oblivious
                        DemM.updateDemandMatrix(sourceID, destinationID, packetsize, 1)
                        DemM.getDemandMatrix()
#                        print('DM', DemM.getDemandMatrix())


                allpaths3 = GM.getalivePaths()#alivePaths(allpaths, GM)
    
                collisionLinks, numLinks, perc_collissionLink = collisions(allpaths3)
#        performanceStatistics()



        return pathnotfoud



def collisions(allpathsRL):
    """
        function to
        Args:
        
        output:
            overlap_link = total number of overlapping links
            len(d2) = total number of occupied links
            perc_ovlink = ratio overlap_link/len(d2)
        """
#    print('allpathsRL')
#    print(allpathsRL)
    linkList = []
    overlap_link = 0
    #evaluate overlapping link among all active paths
    for path in allpathsRL:
        for elem in range(1,len(path)):
            linkList.append([path[elem-1], path[elem]])
#    print('linkList',linkList)

    d1 = Counter(str(e) for e in linkList)
    d2 = d1.items()
#    print('count link', d1)
#    print(d2)
    for d_ in d2:
        if(d_[1] > 1):
#        overlap_link = sum(d.values())
            overlap_link += d_[1]

#    print('tot ov links', overlap_link)
#    print('tot link', len(d2))
    perc_ovlink = overlap_link/len(d2)


    return overlap_link, len(d2), perc_ovlink

#
#def performanceStatistics():
#    """
#        function to
#        Args:
#
#
#        """

