import networkx as nx
import subprocess
import os
import sys
import numpy as np

import re

def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line

def allSOpath(dotFormat, DemandMatrix, nHost):
    
    #save demand matrix in the format accepted by yates
#    print('DM', DemandMatrix)
    text = dotFormat

    DemMatrix = []
    for i in range(nHost):
        for j in range(nHost):
            DemMatrix.append(int(DemandMatrix[i][j]))

    m = re.search('topologies/(.+?).dot', text)
    if m:
        topo = m.group(1)
    print(topo)

#    print('yates format DM', DemMatrix)
    f  = open("demand_" + topo + ".txt", "w")
    for item in DemMatrix:
        f.write(str(item))
        f.write(" ")
    f.close()


    os.system("sh ./yatescode.sh {}".format(topo, ))


def semi_oblivious(dotFormat, sourceID, destinationID, nHost):
    
    path = []
    path.append(sourceID)
    #run yates
    text = dotFormat
    
    nPaths = 0
    next_line = []
    prob = []
    
    hostS = 'h' + str(sourceID+1)
    hostD = 'h' + str(destinationID+1)

    m = re.search('topologies/(.+?).dot', text)
    if m:
        topo = m.group(1)
    print(topo)

#    os.system("sh ./yatescode.sh {}".format(topo))

#yates generates the output file
#from this output file, I extract the path from sourceID to destinationID
    with open('./data/results/'+ topo +'/paths/semimcfraeke_0', 'r') as f:
        for line in nonblank_lines(f):
#        for line in f:
            line_data = line.split()
#            print(line_data)


            if (line_data[1] == '->' and line_data[0] == hostS and line_data[2] == hostD):
                next_line_ = f.next()
                next_line_ = next_line_.split()
#                print('next_line_', next_line_)
                prob.append(float(next_line_[-1]))
                next_line.append(next_line_)
                nPaths += 1
#                print('sum', np.sum(prob))
                while(np.sum(prob) != 1 ):
                    next_line_ = f.next()
                    next_line_ = next_line_.split()
                    next_line.append(next_line_)
                    nPaths += 1
#                    print('next_line_', next_line_)
#                    print('next_line_[-1]', next_line_[-1])
                    prob.append(float(next_line_[-1]))

#                    print('sum', np.sum(prob))


                ra = range(nPaths)
#                print('lista', ra,  prob)
                ind = np.random.choice(ra,None,p=prob)
                
                splitnext_line = next_line[ind]

                                
#            if (line_data[1] == '->' and line_data[0] == hostS and line_data[2] == hostD):
#                next_line = f.next()
#                next_line = next_line.split()
##                print('next line split', next_line[-1])
#                while(float(next_line[-1]) < 0.5):
#                    next_line = f.next()
#                    next_line = next_line.split()
##                print('found', next_line[-1])
##                print('len(next_line)', len(next_line))
                for index in range(1,len(splitnext_line)-2):
#                    print('next_line[index][1]', next_line[index][1])
                    if(splitnext_line[index][1] == 'h'):
                        IDhost = int(splitnext_line[index][2])-1
                        path.append(IDhost)
#                        print('id', IDhost)

                    if(splitnext_line[index][1] == 's'):
                        IDswitch = nHost + int(splitnext_line[index][2])-1
                        path.append(IDswitch)
#                        print('id', IDswitch)

                path.append(destinationID)


    print(path)
    return path




#    return

