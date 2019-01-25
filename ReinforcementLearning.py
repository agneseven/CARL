import networkx as nx
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
#import matplotlib.pyplot as plt
#import pylab as plt
#from IPython.display import clear_output
import random
from keras.callbacks import History
#from sklearn import linear_model, datasets
#from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import History
import subprocess
import os
import sys
from topoGeneration import topoGeneration
#from briteGen import *





import tensorflow as tf



class ReinforcementLearning:

    def __init__(self, nFlow, G, double_dqn):
        """
            Q-Learning algorithm initializer
            Args:
                nNode = number of nodes in the topology
                nFlow = number of flows that we want to train

            """
        self.nNode = None
        self.nFlow = nFlow
        self.double_dqn = double_dqn
        self.nHost = None
    
    def initVisit(self):
        """
            Function that inizialize the variable visit that I need to count the number of visis to each node during an epoch.
            We need it to modify alpha (= the learning rate).
            In this way we learn slowly from the pairs that we visited more
            Args:
            visit = numpy.ndarray with all the elements set to 1
            
            """
        ACTIONS = range(self.nNode)
        visit = np.ones((self.nNode,self.nNode,self.nFlow))

        return visit


    def randPair(self, s,e):
        """
            function for generation random starting node and destionation node between s and e
            Args:
                s = first number of the range (included)
                e = last number of the range (not included)
            Returns:
                np.random.randint(s,e), 0 = random location of the flow in the system (random starting node and destionation node)
            """
        return np.random.randint(s,e), 0
    

    def findLoc(self, state, posZ):
        """
            function for finding the location of a flow or the goal of the new flow in the topology
            Args:
                state = current state of the sysyem
                posZ = element that we want to find (0 = new flow, self.nFlow + 1 = goal of the new flow)
            Returns:
                i = current node where the element is located
                j = 1
            """
        z = posZ
        for i in range(self.nNode):
            for j in range(1):
                if(state[i,j,z] == 1).all():
                    return i,j


    
    def initGridNnodesFflow(self, activeFlows):
        """
            function that initializes the topology randomly to start the training
            Args:
                -
            Returns:
                state = random generated state
            """
        state = np.zeros((self.nNode,1,self.nFlow + 1))
        #place new flow randomly: 1 only if the node is a host
        a,b = self.randPair(0, self.nHost)
        state[a,b,0] = 1
        #place other flow randomly (on two or one node)
        for n in range (1,self.nFlow-activeFlows):
            a1,b1 = self.randPair(0,self.nNode)
            state[a1,b1,n] = 1
#            a2,b2 = self.randPair(0,self.nNode)
#            state[a2,b2,n] = 1

        #place goal of the new flow: 1 only if the node is a host
        a_g,b_g = self.randPair(0,self.nHost)
        state[a_g,b_g,self.nFlow] = 1

        print("starting location")
        print(state)

        flow1 = self.findLoc(state, 0)
        goal1 = self.findLoc(state, self.nFlow)

        if(flow1 == goal1):
#            print('Invalid grid. Rebuilding..')
            return self.initGridNnodesFflow(activeFlows)
        return state
    
    
    def makeMove(self, state, actions):
        """
            function that update the state making the flows moving.
            it doesn't delete the previous locations:
                new flow = old locations are set to 2, new location is set to 1
                other flows = old locations and new location are set to 1
                
            Args:
                state = current state of the system
                actions = set of actions that all the flows want to take
            Returns:
                state = updated state after all flows moved
            """
        #current location of flow1
        flow1 = self.findLoc(state, 0)
        #goal of flow1
        goal1 = self.findLoc(state, self.nFlow)
        #future location of flow1
        new_loc_flow1 = (actions[0], 0)
        #future location of the flow1 and of the other flows
        new_loc = []
        for n in range(len(actions)):
                new_loc.append(actions[n])

        #set to 1 the new loc of the new flow in the matrix
        state[new_loc_flow1][0] = 1
        #set to 2 the old location of the new flow if the new location changed
#        if(new_loc_flow1 != flow1):
#            state[flow1][0] = 2
        if(new_loc_flow1 != flow1):
            state[flow1][0] = 0

        #update the other flows
#        for f in range(1,self.nFlow):
        for f in range(len(new_loc)):
            state[new_loc[f],0][f] = 1

        return state
    

    def getReward(self, state, AdjNodes, start, path):
        """
            function that handles the rewards.
            1) if the flow reaches the goal it receives a positive reward = 100
            2) until the flow doen't reach the goal it receive:
                a) -1 if it moves to an empty node
                b) -1*sumRow if it moves to a node where there are already other flows. sumRow = number of other flows in the node
            
            Args:
            state = current state of the system
            Returns:
            reward = obtained reward
            """
        #current location of flow1
        flow1 = self.findLoc(state, 0)
        #goal of flow1
        goal1 = self.findLoc(state, self.nFlow)
#        print(state)
#        print('flow1, goal1 {} {}'.format(flow1, goal1))
#        print('stato verifica {}', state)
#        print('stato', state)

        sumRow = 0
        for f in range(self.nFlow):
            if(state[flow1][f] != 0):
                sumRow += 1
#        print('sumRow {}'.format(sumRow))
        maxSumRow = self.nFlow
        sumRowVicini = np.zeros((self.nNode))

        for f in range(self.nFlow):
            for n in range(self.nNode):
                if(state[n,0][f] != 0).all():
                    sumRowVicini[n] += 1 #conto quanti flussi ci sono in ogni nodo
#        print(sumRowVicini)
#        print('AdjNodes {} '.format(AdjNodes))
#        print('AdjNodes[0] {}'.format(AdjNodes[0]))
        vectorvicini = 0
        otherAdjNodes = np.setdiff1d(AdjNodes[0],flow1[0])
#        print('otherAdjNodes {}'.format(otherAdjNodes))

##I want the reward included in [-1, 1]
        penalty = 1/self.nFlow
#1) goal not yet achieved, the node has already been visited
# a higher penalty to avoid loops
#        visited = 0
#        for hop in path:
#            if(flow1[0] == hop):
#                visited = 1
#
#        if(flow1 != goal1 and visited == 1): #sono gia passata per quel nodo
#            return -0.1

#2) goal not yet achieved, only flow in that node
# a small penalty only to avoid too many movements
        if (flow1 != goal1 and sumRow == 1): #
            return -0.04
#            return penalty

#3) goal not yet achieved, more flows in that node,  not the only possible destination, other possible destinations are emptier
# a penalty proportional to the number of flows in the node
        for v in otherAdjNodes:
#            print(v)
            if(flow1 != goal1 and sumRow != 1 and len(AdjNodes[0]) != 1 and (sumRowVicini[v]+1) < sumRow): #if other neighbors have a lower number of flow exit
#                print('qui {} {}'.format(sumRowVicini[v], sumRow))
                vectorvicini = 1

        if(vectorvicini == 1):
#            print('sum row', sumRow)
            return -0.25 * (sumRow)
#            return penalty * sumRow

#4) goal not yet achieved, more flows in that node, that node is the only possible destinatioN
# a small penalty only to avoid too many movements, like (2)

        if(flow1 != goal1 and sumRow != 1 and len(AdjNodes[0]) == 1):
#            print('len 1')
            return -0.04
#            return penalty

#        elif(flow1 != goal1 and flow1 == start):
#            return -0.25
#            return -1
#        elif(flow1 != goal1 and sumRow == 1 and ):
#            return -0.1 * (sumRow)
#                    return -1
############################goals reached############################
#5) goal reached, reward 1!
        elif (flow1 == goal1):
            return 1
#6) any other cases not considered before, a small penalty
#for example: goal not yet achieved, more flows in that node,  not the only possible destination, other possible destinations are fuller
        else:
#            print('else 0.1')
#            return -0.04
            return penalty




#TRAINING---------------------------------------------------------------------------
    def training(self, parameters, nNode, myG, nHost, activeFlows):
        """
            function that handles the training of the neural network.
            
            Args:
            G = graph of the topology that represents the system
            parameters = set of parameters that are passed as input
                epochs =
                gamma =
                batchSize =
                buffer =
            Returns:
            model = trained neural network
            """
    #generation of the topology
        
        
#        self.nNode, points_list, bandwidth_list, delay_list = briteGen()
        self.nNode = nNode
        self.nHost = nHost
        G = myG
#        topoGen = topoGeneration(nNode, points_list, bandwidth_list, delay_list)
#        G = topoGen.Graph()

        state = self.initGridNnodesFflow(activeFlows)
        NodePairs = np.zeros((nNode, nNode))
#        print('stato iniziale {}', state)




        Sbatch_size = 32

        print('nnode dense', (self.nNode*(self.nFlow+1)+self.nNode)/2)


        model = Sequential()
#        model.add(Dense((2*self.nNode*(self.nFlow+1)/3), input_dim=self.nNode*(self.nFlow+1), kernel_initializer ='he_normal', activation='relu'))#164
        model.add(Dense(((self.nNode*(self.nFlow+1)+self.nNode)/2), input_dim=self.nNode*(self.nFlow+1), kernel_initializer ='he_normal', activation='relu'))#164

#        model.add(Dense(24, kernel_initializer ='he_normal', activation='relu'))#150
#        model.add(Dense(150, kernel_initializer ='he_normal', activation='relu'))
#        model.add(Dense(150, kernel_initializer ='he_normal', activation='relu'))
        model.add(Dense(self.nNode, kernel_initializer ='he_normal', activation='linear'))

#        model.add(Dense(164, init='he_normal', input_shape=(self.nNode*(self.nFlow+1),)))#(9x1x4) #Initializations define the way to set the initial random weights of Keras layers ##lecun_uniform
#        model.add(Activation('relu'))
##        model.add(Activation(LeakyReLU(alpha=0.1)))
#        #model.add(Dropout(0.2)) # I don't know 100% for sure that it's bad to use dropout in Deep Reinforcement Learning, but 1) it's certainly not common, and 2) intuitively it doesn't seem necessary. Dropout is used to combat overfitting in supervised learning, but overfitting is not really much of a risk in Reinforcement Learning (at least, not if you're just trying to train for a single game at a time like you are here).
#
#        model.add(Dense(150, init='he_normal'))
#        model.add(Activation('relu'))#        model.add(Activation(LeakyReLU(alpha=0.1)))
#        #model.add(Dropout(0.2))
#        model.add(Dense(150, init='he_normal'))
#        model.add(Activation('relu'))#        model.add(Activation(LeakyReLU(alpha=0.1)))
#        #model.add(Dropout(0.2))
#        model.add(Dense(150, init='he_normal'))
#        model.add(Activation('relu'))#        model.add(Activation(LeakyReLU(alpha=0.1)))
#        #model.add(Dropout(0.2))
#
#
#        model.add(Dense(self.nNode, init='self.nNode'))
#        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms, metrics=['accuracy']) #reset weights of neural network

        
#        model.fit(state.reshape(1,self.nNode*(self.nFlow+1)), reward_value, epochs=1, verbose=0)

        prediction = model.predict(state.reshape(1,self.nNode*(self.nFlow+1)), batch_size=Sbatch_size)
#        print(prediction)

        epochs = int(parameters[0])
        gamma = float(parameters[1])#since it may take several moves to goal, making gamma high
        NB_epoch = int(parameters[2])
        epsilon_start = 1 #epsilon-greedy = exploration (random), exploitation (greedy)
        epsilon_decay_length = 1e5        # number of steps over which to linearly decay epsilon
        epsilon_decay_exp = 0.97    # exponential decay rate after reaching epsilon_end (per episode)y
        #epsilon_end = 0.1
        epsilon = epsilon_start
        min_epsilon = 0.1
        alpha =  0.1#learning rate
        batchSize = int(parameters[3])
        buffer = int(parameters[4])
        replay = []
        h = 0
        
        print("Parameters currently in use")
        print('epochs {} gamma {} epsilon {} alpha {} batchSize {} buffer {}'.format(epochs, gamma, epsilon, alpha, batchSize,  buffer))
        
#        topoGen = topoGeneration(nNode, points_list, bandwidth_list, delay_list)
#        graph = topoGen.Graph()
#        topoGen.showGraph()
        #for counting how many times each node is conidered as starting and ending location during the trainig
        posIniziale = np.zeros(self.nNode)
        posFinale = np.zeros(self.nNode)

        updates = []
        updates2 = []
        arrayepsilon = []
        success = np.zeros((self.nNode, self.nNode))
        visit = self.initVisit()
        arrayepsilon.append(epsilon)
        score = []
        r_avg_list = []
        history = History()
        
        best_score = None
        reward_list = []
        averageReward = 0

        epsilon_linear_step = (epsilon_start-min_epsilon)/epochs
        
        activeFlows = 1
        increment = 0

        for i in range(epochs):
            if (i == (epochs/self.nFlow + increment)):
                activeFlows = activeFlows + 1
                increment = increment + epochs/self.nFlow
                print('incremento {}'.format(i))
            
            
            state = self.initGridNnodesFflow(activeFlows)
            st = self.findLoc(state, 0)
            en = self.findLoc(state, self.nFlow)
            NodePairs[st[0]][en[0]] += 1
#            print('st, en {} {}'.format(st, en))

            pos_1 = self.findLoc(state, 0)
            goal1 = self.findLoc(state, self.nFlow)

            posIniziale[pos_1[0]] += 1
            posFinale[goal1[0]] += 1

            players_loc = []
            action = []
          
            for n in range(self.nFlow-activeFlows):
                players_loc.append(self.findLoc(state, n))#format : (current_node, 0)
                action.append(players_loc[n][0])
            
            #epsilon = max(min_epsilon, epsilon*decay)
#            epsilon = min_epsilon + (1.0 - min_epsilon) * np.exp(-decay * i)
#epsilon * decay#(1-(epsilon_start + epsilon_end)/epochs)# epsilon * epsilon_decay
# linearly decay epsilon from epsilon_start to epsilon_end over epsilon_decay_length steps
            if epsilon > min_epsilon:
                epsilon -= epsilon_linear_step
#            print("epsilon")
#            print(epsilon)
            arrayepsilon.append(epsilon)
            
            bufferaction = []
            for b in range(self.nNode):
                bufferaction.append(0)
            finalpath = []
            finalpath.append(pos_1[0])
            cumrew = []

            adjMatrix = nx.adjacency_matrix(G)


            status = 1
            r_sum = 0
            index2 = [0] * self.nFlow


            #while game still in progress, when game finishes, the status is set to 0
            #it can finish because flow1 reached the goal or because there were too many moves
            while(status == 1):
#                print('state',state)
                goal1 = self.findLoc(state, self.nFlow)

                #find current location of flow 1
                player_loc = self.findLoc(state, 0)#np.array([1,0,0,0]))
#                print('current flow 1 location: {}'.format(player_loc))
                # current location of all the flows
                players_loc[0] = player_loc
                for f in range(1,self.nFlow-activeFlows):
                    players_loc[f] = (action[f],0)
#                print('players_loc {}'.format(players_loc))

                #we need adjMatrix only to know the valide actions. The valide actions are the adjacent nodes of the current node

                nonAdjNodes = []
                AdjNodes = []
#                    print('range 1 {}'.fw-activeFlows)))
#                    print('range 2 {}'.format(range(self.nNodormat(range(self.nFloe)))
                for f in range(self.nFlow-activeFlows):
                    nonAdjNodesList = []
                    for k in range(self.nNode):
#                        print('adjMatrix.todense()[players_loc[f][0],k] {}{}'.format(k, adjMatrix.todense()[players_loc[f][0],k]))
                        if (adjMatrix.todense()[players_loc[f][0],k] == 0 and players_loc[f][0] != k):
                            #if (players_loc[f][0] != k):
                            nonAdjNodesList.append(k)
#                    nonAdjNodesList = sorted(nonAdjNodesList)
                    nonAdjNodes.append(nonAdjNodesList)
                nonAdjNodes[0].append(players_loc[0][0])

#                nonAdjNodes = sorted(nonAdjNodes)

#                print('nonAdjNodes %s ' %nonAdjNodes)
#                nonAdjNodes[0].append(players_loc[0][0])
#                nonAdjNodes[0][:] = sorted(nonAdjNodes[0][:])

#                print('nonAdjNodes[0]: {}'.format(nonAdjNodes[0][:]) )

                NodeList = range(self.nNode)

                
                for f in range(self.nFlow-activeFlows):
                    AdjNodes.append(list(set(NodeList).difference(set(nonAdjNodes[f][:]))))
#                print('AdjNodes %s ' %AdjNodes)


                #Let's run our Q function on S to get Q values for all possible actions
                qval = model.predict(state.reshape(1,self.nNode*(self.nFlow+1)), batch_size=Sbatch_size)

                #epsilon-greedy alghoritm for exploration and exploitation of the flow1:
                #at the beginning more exploration (epsilon is bigger)
                #then the epsilon decreases (linearly), therefore there will be more exploitation of the Q function
#                print('goal {}'.format(goal1[0]))
                if (random.random() < epsilon): #choose random action
#                    print('random action')
                    if(player_loc != goal1):#if the goal has not been reached yet
                        index = np.random.randint(0,len(AdjNodes[0][:]))
                        action[0] = AdjNodes[0][index]
            
                    else:#if the goal has already been reached
#                        print('else')
                        AdjNodes[0].append(goal1[0])
#                        AdjNodes[0][:] = sorted(AdjNodes[0][:])
                        nonAdjNodes[0] = np.setdiff1d(nonAdjNodes[0],goal1[0])
#                        print('AdjNodes, nonAdjNodes {} {}'.format(AdjNodes[0], nonAdjNodes[0] ))
                        action[0] = goal1[0]

                else: #choose best action from Q(s,a) values
#                    print('qval action')
                    if(player_loc != goal1):#if the goal has not been reached yet
                        qvalValid = np.copy(qval)

                        for non in nonAdjNodes[0][:]:
                            qvalValid[0, non] = -100000
                            action[0] = (np.argmax(qvalValid))
                    else:#if the goal has already been reached
#                        print('else')
                        AdjNodes[0].append(goal1[0])
#                        AdjNodes[0][:] = sorted(AdjNodes[0][:])
                        nonAdjNodes[0] = np.setdiff1d(nonAdjNodes[0],goal1[0])
#                        print('AdjNodes, nonAdjNodes {} {}'.format(AdjNodes[0], nonAdjNodes[0] ))
                        action[0] = goal1[0]
#                print('AdjNodes  {}'.format(AdjNodes[0] ))
#                print('action[0] {}'.format(action[0]))



                bufferaction[action[0]] += 1
#                print('index2 {}'.format(index2))
                #actions of the other flows are simply random actions among the valid actions
                if(len(finalpath) < 3):
                    for f in range(1,self.nFlow-activeFlows):
                        index2[f] = np.random.randint(0, len(AdjNodes[f][:]))
#                        print('index2 {}'.format(index2))

                        action[f] = AdjNodes[f][index2[f]]
                action_ = np.copy(action)
                if(len(finalpath) >= 3):
                    for f in range(1,self.nFlow-activeFlows):
                        action[f] = action_[f]

#                print('all actions {}'.format(action))



                visit[player_loc[0], action[0]][0] += 1
                    #print('visit[player_loc, action[0]][0] {}'.format(visit[player_loc, action[0]][0]))
                alpha =  1./(visit[player_loc[0], action[0]][0])
#                print('alpha {}'.format(alpha))
#                print('action: {}'.format(action[0]))
                finalpath.append(action[0])
                
                #Take action, observe new state S'
                new_state = self.makeMove(state, action)
                
        
        
                #Observed reward
                reward = self.getReward(new_state, AdjNodes, st, finalpath)
#                print('rew: {}'.format(reward))
                updates.append(reward)
                #evaluation of the total reward: sum of the reward obtained by each action
                r_sum += reward

                if(reward == 1):
                    success[st[0]][en[0]] += 1


                for b in range(self.nNode):
                    if(bufferaction[b] >= 100):
                        reward = -1000

#                print('reward1: %s' %reward)

                #Experience replay storage
                #Experience Replay stores experiences (state transitions, rewards and actions) which are necessary data to perform Q learning, and makes mini-batches to update neural networks. This technique expects the following merits.
                #reduces correlation between experiences in updating DNN
                #increases learning speed with mini-batches
                #reuses past transitions to avoid catastrophic forgetting
#                During gameplay all the experiences <s,a,r,s'> are stored in a replay memory. When training the network, random samples from the replay memory are used instead of the most recent transition. This breaks the similarity of subsequent training samples, which otherwise might drive the network into a local minimum. Also experience replay makes the training task more similar to usual supervised learning, which simplifies debugging and testing the algorithm. One could actually collect all those experiences from human gameplay and the train network on these.


                if (len(replay) < buffer): #if buffer not filled, add to it
                    replay.append((state, action[0], reward, new_state))
                else: #if buffer full, overwrite old values
                    if (h < (buffer-1)):
                        h += 1
                    else:
                        h = 0
                    replay[h] = (state, action[0], reward, new_state)
                    #randomly sample our experience replay memory
                    minibatch = random.sample(replay, batchSize)
                    X_train = []
                    y_train = []
                    o = 0
                    for memory in minibatch:
                        o = o + 1
                        #Get max_Q(S',a)
                        old_state_m, action_m, reward_m, new_state_m = memory
#                        print('action_m {}'.format(action_m))
                        old_qval = model.predict(old_state_m.reshape(1,self.nNode*(self.nFlow+1)), batch_size=Sbatch_size)
#                        newQ = model.predict(new_state_m.reshape(1,self.nNode*(self.nFlow+1)), batch_size=1)
#                        maxQ = np.max(newQ)
                        if self.double_dqn:
#                            The "double DQN approach"":
#                           1) Select an item from Memory Bank
#                           2) Using Online Network, from St+1 determine the index of the best action At+1.
#                           3) Using Target Network, from St+1 get the Q-value of that action.
#                           4) Do corrections as usual using that Q-value
                            targetQ = model.predict(new_state_m.reshape(1,self.nNode*(self.nFlow+1)), batch_size=Sbatch_size)
                            maxindexQ = np.argmax(targetQ)
#                            print('targetQ: {}, index maxQ: {}'.format(targetQ, maxindexQ))

                            doubleQ = model.predict(old_state_m.reshape(1,self.nNode*(self.nFlow+1)), batch_size=Sbatch_size)
                            maxQ = doubleQ[0, maxindexQ]
#                            print('doubleQ: {}, index maxQ: {}'.format(doubleQ, maxQ))


                        else:
                            #classic approach
#                            1) Select an item from Memory Bank
#                            2) Using Target Network, from St+1 compute the best action At+1 and its Q-value Q
#                            3) Do corrections as usual
                            doubleQ = model.predict(new_state_m.reshape(1,self.nNode*(self.nFlow+1)), batch_size=Sbatch_size)
                            maxQ = np.max(doubleQ)
#                            print('doubleQ: {}, maxQ: {}'.format(doubleQ, maxQ))

#                        if (maxQ > 0):
#                            score.append(np.sum(doubleQ)/1000)
#                        else:
#                            score.append(0)


                        y = np.zeros((1,self.nNode))
                        y[:] = old_qval[:]
#                        print(y)
#                        print('reward_m, alpha, gamma, maxQ {} {} {} {}'.format(reward_m, alpha, gamma, maxQ))

#                        if reward_m == -0.1: #non-terminal state
                        if reward_m != 1 and reward_m != -1000: #non-terminal state
                            update = alpha*(reward_m + (gamma * maxQ))
#                            target = reward(s,a) + gamma * max(Q(s'))
                            updates2.append(update)
        
                        else: #terminal state
                            update = reward_m
                            updates2.append(update)



#                        print('update {}'.format(update))
#                        y[0][action_m[0]] = update
                        y[0][action_m] = update
                        X_train.append(old_state_m.reshape(self.nNode*(self.nFlow+1),))
                        y_train.append(y.reshape(self.nNode,))

                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    print("Game #: %s" % (i,))
                            #model.fit(state.reshape(1,self.nNode*4), y, batch_size=1, nb_epoch=3, verbose=1)
                    train_history = model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=NB_epoch, validation_data=(X_train, y_train),verbose=1, callbacks=[history])




                    state = new_state
            #batch_size: integer. Number of samples per gradient update.
            #nb_epoch: integer, the number of epochs to train the model.
            #verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.   https://stackoverflow.com/questions/47902295/what-is-the-use-of-verbose-in-keras-while-validating-the-model
            #state = new_state
#                    print('new reward: %s' %reward)
#                    print('new action: %s' action)

        #        scores.append(update)
        
                if reward == 1 or reward == -1000:#!= -0.1:# terminal state when I repeated too many times the same move or I arrived to the goal
#                    print('finalpath2 {}'.format(finalpath))
                    cumrew.append(reward)
                    status = 0
#                clear_output(wait=True)
#                print('r sum: { }', r_sum)

            if epsilon > 0.1:
                epsilon -= (1/epochs)
#            print(i)
            if(i>0): #other epochs
#                print(r_avg_list)
#cumulative reward evaluated at the end of each episode
                r_avg_list.append(r_sum+r_avg_list[i-1])
                reward_list.append(r_sum)
                averageReward = np.mean(reward_list[-100:])
                # print(type(reward_list), type(100), type(best_score), type(averageReward))
                # if(len(reward_list) >= 100 and best_score is 0 or best_score < averageReward):
                #     best_score = averageReward
            elif(i==0): #first epoch
                r_avg_list.append(r_sum)
                reward_list.append(r_sum)



#        loss = history.history['loss']
#        val_loss = history.history['val_loss']
#        acc = history.history['acc']
#        epoch_count = range(1, len(loss) + 1)
## Visualize loss history
#        plt.plot(epoch_count, loss, 'r--')
#        plt.plot(epoch_count, val_loss, 'b-')
#        plt.legend(['Training Loss', 'Test Loss'])
#        plt.xlabel('Epoch')
#        plt.ylabel('Loss')
#        plt.show();
##        print(r_avg_list)
#        plt.plot(r_avg_list)
#        plt.show()

#        print(NodePairs)
#        print(visit)
#        print('posIniziale {}'.format(posIniziale))
#        print('posFinale {}'.format(posFinale))
#plot rewards
#plt.plot(scores)
#plt.show()
#        plt.plot(loss)
#        plt.show()
################cumulative reward
#        plt.plot(cumrew)
#        plt.show()
##        print('success: {}'.format(success))
####################epsilon##################
##        plt.plot(arrayepsilon)
##        plt.show()
#        plt.plot(score)
#        plt.show()
#        plt.plot(loss)
#        plt.plot(val_loss)
#        plt.plot(acc)
#        plt.legend(['loss', 'val_loss', 'acc'])
#        plt.show()
        #plotter.block()
#        return best_score
        return model
            #, points_list, bandwidth_list, delay_list


    def testAlgo( self, init, path, G, model, start, end):
        i = 0
        if init==0:
            state, indexPATH = self.testGrid1(path)
        elif init==1:
            state, indexPATH = self.testGrid2(path)
        elif init==2:
            state, indexPATH = self.testGrid3(path, start, end)
        elif init==3:
            state, indexPATH = self.testGrid4(path)
        elif init==4:
            state, indexPATH = self.testGrid5(path)
        elif init==5:
            state, indexPATH = self.testGrid6(path)
        elif init==6:
            state, indexPATH = self.testGrid7(path)
        elif init==7:
            state, indexPATH = self.testGrid8(path)
        elif init==8:
            state, indexPATH = self.testGrid9(path, start, end)
#            print (state)

        Sbatch_size = 32

        loop = 0
        win = 0
        fail = 0
        listPath = []

#        print("Initial State:")
#        print(state)
        contFlow = 0
        for f in range(1,self.nFlow-1):
            if(len(path[f]) == 2 and path[f][0] == 0 and path[f][1] == 0):
                contFlow += 1
        #perche' alcuni flussi potrebbero non essere presenti
        nFlow = self.nFlow - contFlow -1
        print('active flow number', nFlow)


        adjMatrix = nx.adjacency_matrix(G)
        st = self.findLoc(state, 0)


        locflow1 = self.findLoc(state, 0)

        actions = [] #next action of all the flows
        actions.append(0) #fake action of the 1st flow
        # actions of the others flows: last node of the previous known path
#        print('paths {}'.format(path))

        if(nFlow > 1):
            for f in range(nFlow-1):
                print('f', f)
                actions.append(path[f][indexPATH[f]])
#        print('actions {}'.format(actions))

        listPath.append(locflow1[0])
    
        j = 0
        status = 1
        indexPath = indexPATH
        #while game still in progress
        while(status == 1):
#            print(state)

            players_loc = []
        
            player1_loc = self.findLoc(state, 0)
            
            players_loc.append(player1_loc[0])
            
            if(nFlow > 1):
                for f in range(1,nFlow):
                    players_loc.append(actions[f])
            
#            print('players_loc {}'.format(players_loc))

            NodeList = range(self.nNode)
            AdjNodes = []
            nonAdjNodes = []
#            if(nFlow > 2):

            for f in range(nFlow):
                nonAdjNodesList = []
                for k in range(self.nNode):
#                    if (adjMatrix.todense()[players_loc[f],k] == 0).all():
                    if (adjMatrix.todense()[players_loc[f],k] == 0 and players_loc[f] != k):
#                        if (players_loc[f] != k):
                        nonAdjNodesList.append(k)
#                nonAdjNodesList = sorted(nonAdjNodesList)
                nonAdjNodes.append(nonAdjNodesList)
                nonAdjNodes[0].append(players_loc[0])
#            print('nonAdjNodes %s ' %nonAdjNodes)

#            if(nFlow > 2):

            for f in range(nFlow):
                AdjNodes.append(list(set(NodeList).difference(set(nonAdjNodes[f][:]))))
#            print('AdjNodes %s ' %AdjNodes)

            qval = model.predict(state.reshape(1,self.nNode*(self.nFlow+1)), batch_size=Sbatch_size)

            qvalValid = np.copy(qval)

            for non in nonAdjNodes[0][:]:
                qvalValid[0, non] = -100000
#            print(qvalValid)
            actions[0] = (np.argmax(qvalValid))

            if(nFlow > 2):
                for f in range(nFlow-1):
                    actions[f+1] = (path[f][indexPath[f]])

#            print('actions : {}'.format(actions))

            state = self.makeMove(state, actions)

            reward = self.getReward(state, AdjNodes, st, listPath)

            nextNode = actions[0]

            listPath.append(nextNode)
        
            if reward == 1:
                status = 0
#                print("Reward: %s" % (reward,))

            i += 1 #If we're taking more than 30 actions, just stop, we probably can't win this game
            if (i > 30):
#                print("Game lost; too many moves.")
                loop = 1
                break
        if reward == 1:
            win = 1
        if reward == -1000:
            fail = 1
    
        return listPath, win, loop, fail



    def testGrid1(self,path):
        
        state = np.zeros((self.nNode,1,self.nFlow+1))
        #place packet1
        a,b = 0, 0 #self.randPair(0,8)
        state[a,b,0] = 1
        #place packet2
        a1,b1 = path[0], 0
        state[a1,b1,1] = 1

        a2,b2 = path[1], 0 #self.randPair(0,8)
        state[a2,b2,1] = 1

        a2,b2 = path[2], 0 #self.randPair(0,8)
        state[a2,b2,1] = 1
        
        #place goal1
        a2,b2 = 3, 0 #self.randPair(0,8)
        state[a2,b2,2] = 1
#        #place goal2
#        a3,b3 = path[path.size -1], 0
#        state[a3,b3,3] = 1

        indexPATH = 0

        return state, indexPATH


    def testGrid2(self, path):
    
        state = np.zeros((self.nNode,1,self.nFlow + 1))
        #place packet1
        a,b = 0, 0 #self.randPair(0,8)
        state[a,b,0] = 1
        #place packet2
        a1,b1 = path[0], 0
        state[a1,b1,1] = 1
        
        a2,b2 = path[1], 0 #self.randPair(0,8)
        state[a2,b2,1] = 1
        
        a2,b2 = path[2], 0 #self.randPair(0,8)
        state[a2,b2,1] = 1
        
        #place goal1
        a2,b2 = 3, 0 #self.randPair(0,8)
        state[a2,b2,2] = 1
#        #place goal2
#        a3,b3 = path[path.size -1], 0
#        state[a3,b3,3] = 1

        indexPATH = 1
    
        return state, indexPATH


    def testGrid3(self, path, start, end):
#        print('path {}'.format(path))
        state = np.zeros((self.nNode,1,self.nFlow + 1))
        #place packet1
        a,b = start, 0
        state[a,b,0] = 1
        #place packet2
        for f in range(self.nFlow-1):
            for p in range(len(path[f])):
                a1,b1 = path[f][p], 0
                state[a1,b1,f+1] = 1

        
        #place goal1
        a2,b2 = end, 0
        state[a2,b2,self.nFlow] = 1
#        #place goal2
#        a3,b3 = path[path.size -1], 0
#        state[a3,b3,3] = 1

        print('start {}, {}'.format(a,b))
        print('end {}, {}'.format(a2,b2))
        indexPATH = []
        for f in range(self.nFlow-1):
            print(len(path[f]))
            indexPATH.append(len(path[f])-1) #current position

        return state, indexPATH

    def testGrid4(self, path):
    
        state = np.zeros((self.nNode,1,self.nFlow + 1))
        #place packet1
        a,b = 7, 0
        state[a,b,0] = 1
        #place packet2
        for f in range(self.nFlow-1):
            for p in range(len(path[f])):
                a1,b1 = path[f][p], 0
                state[a1,b1,f+1] = 1


        #place goal1
        a2,b2 = 4, 0
        state[a2,b2,self.nFlow] = 1
#        #place goal2
#        a3,b3 = path[path.size -1], 0
#        state[a3,b3,3] = 1

        print('start {}, {}'.format(a,b))
        print('end {}, {}'.format(a2,b2))
        indexPATH = []
        for f in range(self.nFlow-1):
            print(len(path[f]))
            indexPATH.append(len(path[f])-1) #current position

        return state, indexPATH


    def testGrid5(self, path):
    
        state = np.zeros((self.nNode,1,self.nFlow + 1))
        #place packet1
        a,b = 7, 0
        state[a,b,0] = 1
        #place packet2
        for f in range(self.nFlow-1):
            for p in range(len(path[f])):
                a1,b1 = path[f][p], 0
                state[a1,b1,f+1] = 1


        #place goal1
        a2,b2 = 5, 0
        state[a2,b2,self.nFlow] = 1
#        #place goal2
#        a3,b3 = path[path.size -1], 0
#        state[a3,b3,3] = 1

        print('start {}, {}'.format(a,b))
        print('end {}, {}'.format(a2,b2))
        indexPATH = []
        for f in range(self.nFlow-1):
            print(len(path[f]))
            indexPATH.append(len(path[f])-1) #current position

        return state, indexPATH


    def testGrid6(self, path):
    
        state = np.zeros((self.nNode,1,self.nFlow + 1))
        #place packet1
        a,b = 4, 0
        state[a,b,0] = 1
        #place packet2
        for f in range(self.nFlow-1):
            for p in range(len(path[f])):
                a1,b1 = path[f][p], 0
                state[a1,b1,f+1] = 1


        #place goal1
        a2,b2 = 3, 0
        state[a2,b2,self.nFlow] = 1
#        #place goal2
#        a3,b3 = path[path.size -1], 0
#        state[a3,b3,3] = 1

        print('start {}, {}'.format(a,b))
        print('end {}, {}'.format(a2,b2))
        indexPATH = []
        for f in range(self.nFlow-1):
            print(len(path[f]))
            indexPATH.append(len(path[f])-1) #current position

        return state, indexPATH


    def testGrid7(self, path):
    
        state = np.zeros((self.nNode,1,self.nFlow + 1))
        #place packet1
        a,b = 6, 0
        state[a,b,0] = 1
        #place packet2
        for f in range(self.nFlow-1):
            for p in range(len(path[f])):
                a1,b1 = path[f][p], 0
                state[a1,b1,f+1] = 1


        #place goal1
        a2,b2 = 7, 0
        state[a2,b2,self.nFlow] = 1
#        #place goal2
#        a3,b3 = path[path.size -1], 0
#        state[a3,b3,3] = 1

#        print('start {}, {}'.format(a,b))
#        print('end {}, {}'.format(a2,b2))
        indexPATH = []
        for f in range(self.nFlow-1):
#            print(len(path[f]))
            indexPATH.append(len(path[f])-1) #current position

        return state, indexPATH


    def testGrid8(self, path):
    
        state = np.zeros((self.nNode,1,self.nFlow + 1))
        #place packet1
        a,b = 6, 0
        state[a,b,0] = 1
        #place packet2
        for f in range(self.nFlow-1):
            for p in range(len(path[f])):
                if( p == 0 and path(len(path[f])) == 0):
                    a1,b1 = 0, 0
                    state[a1,b1,f+1] = 0
                else:
                    a1,b1 = path[f][p], 0
                    state[a1,b1,f+1] = 1


        #place goal1
        a2,b2 = 2, 0
        state[a2,b2,self.nFlow] = 1
#        #place goal2
#        a3,b3 = path[path.size -1], 0
#        state[a3,b3,3] = 1

#        print('start {}, {}'.format(a,b))
#        print('end {}, {}'.format(a2,b2))
        indexPATH = []
        for f in range(self.nFlow-1):
#            print(len(path[f]))
            indexPATH.append(len(path[f])-1) #current position

        return state, indexPATH




    def testGrid9(self, path, start, end):
#        print("path", path)
        state = np.zeros((self.nNode,1,self.nFlow + 1))
        #place flow1
        a,b = start, 0
        state[a,b,0] = 1
        
        fine = []
        #place other flows
        for f in range(self.nFlow-1):
            fine.append(len(path[f])-1)
        
#        for f in range(self.nFlow-1):
#            print('len path', len(path[f]))
#            for p in range(len(path[f])):
##                print(p)
##                print(path[fine[f]])
#                if( len(path[f]) == 2 and path[0] == 0 and path[1] == 0):
#                    print('qui')
#                    a1,b1 = 0, 0
#                    state[a1,b1,f+1] = 0
#                else:
#                    a1,b1 = path[f][p], 0
#                    state[a1,b1,f+1] = 1
        for f in range(self.nFlow-1):
            if(len(path[f]) == 2 and path[f][0] == 0 and path[f][1] == 0):
                print('qui')
                a1,b1 = 0, 0
                state[a1,b1,f+1] = 0
            else:
                for p in range(len(path[f])):
                    a1,b1 = path[f][p], 0
                    state[a1,b1,f+1] = 1
        #place goal1
        a2,b2 = end, 0
        state[a2,b2,self.nFlow] = 1
        
        print(state)
#        #place goal2
#        a3,b3 = path[path.size -1], 0
#        state[a3,b3,3] = 1


#        print('start {}, {}'.format(a,b))
#        print('end {}, {}'.format(a2,b2))
#        print(state)
        indexPATH = []
        for f in range(self.nFlow-1):
#            print(len(path[f]))
            indexPATH.append(len(path[f])-1) #current position

        return state, indexPATH




