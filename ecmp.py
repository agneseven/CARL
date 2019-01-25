import networkx as nx

#class ecmp:
#    
#    def __init__(self):
#        """
#            Args:
#            """
#        self.paths = []
#
#    def updateECMP(self, paths, start, end):
#        """
#            it keeps track of when I use one of the path. so next time I will use the other path
#            Args:
#            """
#        len1 = len(self.paths)
#        if (len1 > 0):
#            for p in len1:
#                if(start == self.paths[p][0] and end == self.paths[p][1]):                    self.paths[(start,end)] = self.paths[(start,end)] + 1
#                else:
#                    self.paths[(start,end)] = 0
#        print('paths: {}'.format(self.paths))
#
#
def apply_ecmp_flow( graph, source, destination):
    """
                it evaluates all the shortest paths for a source-destination pair + counter
                Args:
                graph, source, destination
                
                Output:
                    paths = the shortest paths from source to destination
                
         """
    try:
        paths = list(nx.all_shortest_paths(graph, source, destination))
#        print('paths', paths)
        for p in range(len(paths)):
            paths[p].append(0)
        return paths
    except (KeyError, nx.NetworkXNoPath):
        print ("Error, no path for %s to %s in apply_ecmp_flow()" % (source, destination))
        raise


def all_ecmp_flow(nNode, graph, nHost):
    """
        it evaluates all the shortest paths for ALL the source-destination pair
        Args:
        nNode = number of nodes in the graph
        graph
        
        Output:
        paths = all the shortest paths for ALL the source-destination pair
        
        """
    #array with all the paths
    paths = []
    
    #evaluate all the shortest path for each node pair + counter
    for s in range(nHost):
        path = []
        for e in range(nHost):
            if(s != e):
                path.append(apply_ecmp_flow(graph,s,e))
            else:
                path.append(0)
        paths.append(path)
    return paths

def ecmp_path(start, end, paths):
    """
        it evaluates the shortest path to use among the available shortest paths previously calculated. It picks the shortest path with the lowest counter (= last position in the array)
        Args:
        start = source node
        end = destination node
        paths = all the available shortest paths between all source-destination pairs
        
        Output:
        paths = paths array with the updated counter
        paths[start][end][indice_path] = selected shortest path
        indice_path = index of the selected shortest path in the paths array
        
        """
    minimum = paths[start][end][0][-1]
    indice_path = 0
    for u in range(len(paths[start][end])):
#        print(paths[start][end][u])
#        print(paths[start][end][u][-1])
        if minimum > paths[start][end][u][-1]:
            minimum = paths[start][end][u][-1]
            indice_path = u
    #update counter
    paths[start][end][indice_path][-1] = paths[start][end][indice_path][-1] + 1
    #path without the counter
    realPath = paths[start][end][indice_path][:-1]
#    print("paths[start][end][indice_path] {} indice_path {} realPath {}".format(paths[start][end][indice_path], indice_path, realPath))

#    print paths[1][3][indice_path]
    return paths, paths[start][end][indice_path], indice_path, realPath


