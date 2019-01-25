import networkx as nx
#import matplotlib.pyplot as plt



class topoGeneration:
    #construct topology
    def __init__(self, nNodes, points_list, bandwidth_list, delay_list):
        self.nNodes = nNodes
        self.points_list = points_list
        self.bandwidth_list = bandwidth_list
        self.delay_list = delay_list
    
    #self.nSources = nSources


    def Graph(self):
        # upload the graph
        G=nx.Graph()
        G.add_edges_from(self.points_list)
        # add weights to edges
        for e in G.edges():
            G[e[0]][e[1]]['weight'] = self.bandwidth_list[e]
        return G

#    def showGraph(self):
#    # show graph in a Figure
#        G = self.Graph()
#        pos = nx.spring_layout(G)
#        nx.draw_networkx_nodes(G,pos)
#        nx.draw_networkx_edges(G,pos)
#        nx.draw_networkx_labels(G,pos)
#        labels = nx.get_edge_attributes(G,'weight')
#        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
#        plt.show()

    def adjMatrix(self):
        adjMatrix = nx.adjacency_matrix(G)
        print(adjMatrix.todense())

    def getDelayEdges(self, nodeA, nodeB):
        delay = self.delay_list(nodeA,nodeB)
        return delay


