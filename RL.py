#!/usr/bin/env python
import sys

#try:
from topoGeneration import topoGeneration
from packetsGeneration import *
from GraphManager import *
from EventManager import *
from ReinforcementLearning import *
#from sigopt import Connection
#from briteGen import *
import os.path
import pygraphviz
from networkx.drawing import nx_agraph
from semi_oblivious import *
import argparse
from argparse import RawTextHelpFormatter

#except ImportError:
#    print "Error: missing one of the libraries (numpy, pyfits, scipy, matplotlib)"
#    sys.exit()




def main():

    parser = argparse.ArgumentParser(description='Simulate run of routing strategies\n\n python RL.py <EPISODES> <GAMMA> <EPOCH> <BATCHSIZE> <BUFFER> <TOPOLOGY> <TSIM> <ROUTING> \n \n === topologies options === \n [3cycle] \n [4cycle] \n [4diag] \n [6s4hMultiSwitch] \n [Aarnet] \n [Abilene] \n \n === routing options ===\n[-rl] \t\t run reinforcement learning \n[-so]  \t\t run semi-oblivious \n[-ecmp]  \t run ecmp \n[-ospf]  \t run ospf', formatter_class=RawTextHelpFormatter)
    
    parser.add_argument('episodes', type=int, help='Number of episodes to train the Reinforcement Learning algorith')
    parser.add_argument('gamma', type=float, help='Discount factor')
    parser.add_argument('epochs', type=int, help='Number of epochs')
    parser.add_argument('batchSize', type=int, help='batch size of the Experience Replay buffer')
    parser.add_argument('buffer', type=int, help='size of the replay buffer (for the Experience Replay)')
    parser.add_argument('topology', type=str, help='Simulation topology')
    parser.add_argument('simTime', type=int, help='Simulation Time')
    parser.add_argument('routing', type=str, help='Routing protocol')


    args = parser.parse_args()
    
    if args.gamma > 1 or args.gamma < 0:
        parser.error("gamma cannot be larger than 1 and smaller than 0")
    
    parameters = [args.episodes, args.gamma, args.epochs,  args.batchSize, args.buffer, args.routing, args.simTime]
    print('parameters', parameters)


    initialnFlow = 3#10 #maximum number of flows that can coexist in the system (for the training)
    #    RL1 = ReinforcementLearning(initialnFlow, 0)
    #    model, nNode, points_list, bandwidth_list, delay_list = RL1.training(parameters)
    dotFormat = './topologies/'+args.topology+'.dot'
    G = nx_agraph.from_agraph(pygraphviz.AGraph(dotFormat))
    #    pos = nx.spring_layout(G)
    #    nx.draw_networkx_nodes(G,pos)
    #    nx.draw_networkx_edges(G,pos)
    #    nx.draw_networkx_labels(G,pos)
    labels = nx.get_edge_attributes(G,'weight')


#topology parameters
    node_name = nx.get_node_attributes(G, 'type')
    print('lables', node_name)
    nNode = nx.number_of_nodes(G)
    print('nnode', nNode)

    nHost = 0
    print('nodi: ', nNode)
    for attr in node_name:
#        print('attr', attr)
        if(attr[0] == 'h'):
            nHost += 1
    nSwitch = nNode - nHost
#    print('nhost, nswitch', nHost, nSwitch)

#newgraph with nodes' name from 0 to nNode-1
    myG = G

    mapping = {}
    cuonter_h = 1
    cuonter_s = 1

    num_h = 0
    num_s = num_h + nHost
    index = 0
    while index < nNode:
        for n in G.nodes():
            if(n[0] == 's' and n[1:] == str(cuonter_s)):
                mapping[n] = num_s
                cuonter_s += 1
                num_s += 1
                index += 1
            elif(n[0] == 'h' and n[1:] == str(cuonter_h)):
                mapping[n] = num_h
                cuonter_h += 1
                num_h += 1
                index += 1
    myG=nx.relabel_nodes(G,mapping)

    model = None
    RL1 = None
    if args.routing == 'rl':
#trainig RL!!
        activeFlows = 0
        RL1 = ReinforcementLearning(initialnFlow, myG, 0)
        model = RL1.training(parameters, nNode, myG, nHost, activeFlows)




#    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
#    plt.show()

    #save name
#save_path = 'C:/example/'
#name_of_file = raw_input("What is the name of the file: ")
#completeName = os.path.join(save_path, name_of_file+".txt")
#file1 = open(completeName, "w")
#    toFile = raw_input(model)
#    file1.write(toFile)


    GM = GraphManager(nNode)

    #start generation packet and normal simulation

    packetsGen = packetsGeneration(initialnFlow)
    failedRL = packetsGen.generation(GM, nNode, nHost, nSwitch, myG, model, parameters, RL1, dotFormat)



if __name__ == '__main__':
    main()
