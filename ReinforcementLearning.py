import networkx as nx
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from random import randint
# import pylab as plt
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
                double_dqn = flag -> 1: active double; 0: not using double
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
            model = trained neural network (NN)
            """
    #generation of the topology
        
        
        self.nNode = nNode
        self.nHost = nHost
        G = myG

        # The batch size limits the number of samples to be shown to the network before a weight update can be performed. This same limitation is then imposed when making predictions with the fit model. Specifically, the batch size used when fitting your model controls how many predictions you must make at a time.
        Sbatch_size = 32
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
        #initialize buffer for replay memory
        replay = []
        # h = 0

        print("Parameters currently in use")
        print('epochs {} gamma {} epsilon {} alpha {} batchSize {} buffer {}'.format(epochs, gamma, epsilon, alpha, batchSize,  buffer))
        

        state = self.initGridNnodesFflow(activeFlows)
        NodePairs = np.zeros((nNode, nNode))
#        print('stato iniziale {}', state)

        # The Sequential class is used to define a linear stack of network layers which then, collectively, constitute a model.
        model = Sequential() #Sequential constructor to create a model, which will then have layers added to it using the add() method.
        # first (input) layer of the model
        #layer of type Dense ("Just your regular densely-connected NN layer").
        #The Dense layer has output size (self.nNode*(self.nFlow+1)+self.nNode)/2 -> ?
        #input size = self.nNode*(self.nFlow+1) -> environment state size (row = nodes; column= flows)
        #activation function = To check the Y values produced by a neuron and decide wheter outside connections should conside this neuron as fired
        #or not. or ratherlet us say activated or nor (best funzion for hidden layers is ReLu)
        # kernel_initializer = the way to set the initial random weights of Keras layers
        model.add(Dense(((self.nNode*(self.nFlow+1)+self.nNode)/2), input_dim=self.nNode*(self.nFlow+1), kernel_initializer ='he_normal', activation='relu'))
         # only the first layer of the model requires the input dimension to be explicitly stated; the following layers are able to infer from the previous linear stacked layer
         #next Dense layer
         #output size = total number of nodes, because we want one of the nodes as next hop
        model.add(Dense(self.nNode, kernel_initializer ='he_normal', activation='linear'))
         # RMSprop = optimization algorithm, adaptation of rprop algorithm for mini-batch learning. Simple learning rate schedule is not dynamic enough to handle changes in input during the training. Many RL training uses RMSProp or Adam optimizer.
        rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

        # loss = tf.losses.huber_loss(delta=2.0)
        #Before training a model, you need to configure the learning process, which is done via the compile method. 
        #loss = A loss function. This is the objective that the model will try to minimize. 
        #optimizer = An optimizer.
        #metrics = A list of metrics. For any classification problem you will want to set this to metrics=['accuracy']. A metric could be the string identifier of an existing metric or a custom metric function.
        model.compile(loss='logcosh', optimizer=rms, metrics=['accuracy']) #Compile defines the loss function, the optimizer and the metrics. That's all. If you compile a model again, you will lose the optimizer states.

        
        # prediction = model.predict(state.reshape(1,self.nNode*(self.nFlow+1)), batch_size=Sbatch_size)


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
        
        # activeFlows = 1
        increment = epochs/self.nFlow

        avg_rew_per_ep = [] #save the avg reward for each episode
        list_rew_per_ep = [] #save the reward for each action in one episode


        #start of episodes
        for i in range(epochs):
            #change the number of flows in the state incrementally
            if (i == (epochs/self.nFlow + increment)):
                # activeFlows will not change at the first episode
                activeFlows = activeFlows - 1
                increment = increment + epochs/self.nFlow
                print('incremento {}'.format(i))


            #save the reward for each action in one episode
            #initialized at the beginning of each episode
            list_rew_per_ep[:] = []


            #INITIALIZE THE STARTING STATE
            state = self.initGridNnodesFflow(activeFlows)
            
            #retrieve starting and ending node of the agent
            st = self.findLoc(state, 0)
            en = self.findLoc(state, self.nFlow)
            # NodePairs[st[0]][en[0]] += 1
            pos_1 = self.findLoc(state, 0)
            goal1 = self.findLoc(state, self.nFlow)

            #initialize list of other agents' locations and actions
            players_loc = []
            action = []
          
            #retrieve location and action of other agents
            for n in range(self.nFlow-activeFlows):
                players_loc.append(self.findLoc(state, n))#format : (current_node, 0)
                action.append(players_loc[n][0])
            #for the epsilon-greedy algorithm (exploration-exploitation)
            # linearly decay epsilon from epsilon_start to epsilon_end over epsilon_decay_length steps
            if epsilon > min_epsilon:
                epsilon -= epsilon_linear_step
            arrayepsilon.append(epsilon)
            

            bufferaction = []
            for b in range(self.nNode):
                bufferaction.append(0)

            finalpath = []
            finalpath.append(pos_1[0])
            cumrew = []


            #we need adjMatrix only to know the valide actions. The valide actions are the adjacent nodes of the current node
            adjMatrix = nx.adjacency_matrix(G)


            status = 1
            r_sum = 0
            index2 = [0] * self.nFlow


            #while game still in progress status = 1, when game ends, the status is set to 0
            #it can finish because flow1 reached the goal or because there were too many moves
            #for each time step in one episode
            while(status == 1):

                #find current location of flow 1
                player_loc = self.findLoc(state, 0)
#                print('current flow 1 location: {}'.format(player_loc))
                
                # current location of all the flows
                players_loc[0] = player_loc
                for f in range(1,self.nFlow-activeFlows):
                    players_loc[f] = (action[f],0)
#                print('players_loc {}'.format(players_loc))

                #we need adjMatrix only to know the valide actions. The valide actions are the adjacent nodes of the current node
                nonAdjNodes = []
                AdjNodes = []
                for f in range(self.nFlow-activeFlows):
                    nonAdjNodesList = []
                    for k in range(self.nNode):
                        if (adjMatrix.todense()[players_loc[f][0],k] == 0 and players_loc[f][0] != k):
                            nonAdjNodesList.append(k)
                    nonAdjNodes.append(nonAdjNodesList)
                nonAdjNodes[0].append(players_loc[0][0])

                NodeList = range(self.nNode)
   
                for f in range(self.nFlow-activeFlows):
                    AdjNodes.append(list(set(NodeList).difference(set(nonAdjNodes[f][:]))))


                #Let's run our Q function on S to get Q values for all possible actions
                #I evaluate qval = vestor with a Q value for each node of the topology (each action)
                #I do it now because then qval (explotation) or a random node (exploration) will be 
                #use as action
                qval = model.predict(state.reshape(1,self.nNode*(self.nFlow+1)), batch_size=Sbatch_size)

# Target networks are used in Deep Learning to improve the stability of training. 
# target network is updated every tau = 1000 steps
                Qtarget = qval.copy()
                #epsilon-greedy alghoritm for exploration and exploitation of the flow1:
                #at the beginning more exploration (epsilon is bigger)
                #then the epsilon decreases (linearly), therefore there will be more exploitation (using qval)
                #EXPLORATION -> random neighbor
                if (random.random() < epsilon): #choose random action
                #random.random() Return the next random floating point number in the range [0.0, 1.0).
                    print('random action')
                    if(player_loc != goal1):#if the goal has not been reached yet
                        index = np.random.randint(0,len(AdjNodes[0][:]))#pink a random neighbor
                        action[0] = AdjNodes[0][index]
            
                    else:#if the goal has already been reached
                        AdjNodes[0].append(goal1[0])
                        nonAdjNodes[0] = np.setdiff1d(nonAdjNodes[0],goal1[0])
                        action[0] = goal1[0]
                #EXPLOITATION -> max qval
                else: #choose best action from Q(s,a) values
                    if(player_loc != goal1):#if the goal has not been reached yet
                        qvalValid = np.copy(qval)
                        #so the max q value cannot be one associated to a node that is not a neighobor
                        for non in nonAdjNodes[0][:]:
                            qvalValid[0, non] = -100000
                        action[0] = (np.argmax(qvalValid))
                    else:#if the goal has already been reached
                        AdjNodes[0].append(goal1[0])
                        nonAdjNodes[0] = np.setdiff1d(nonAdjNodes[0],goal1[0])
                        action[0] = goal1[0]


                bufferaction[action[0]] += 1
#!!!!!!!!!!!!!!!!!!!!!?????????????
                #actions of the other flows are simply random actions among the valid actions (neighbora)
                if(len(finalpath) < self.nNode-3):
                    for f in range(1,self.nFlow-activeFlows):
                        index2[f] = np.random.randint(0, len(AdjNodes[f][:]))
                        action[f] = AdjNodes[f][index2[f]]
                action_ = np.copy(action)
                if(len(finalpath) >= self.nNode-3):
                    for f in range(1,self.nFlow-activeFlows):
                        action[f] = action_[f]



                visit[player_loc[0], action[0]][0] += 1

                alpha =  1./(visit[player_loc[0], action[0]][0])

                finalpath.append(action[0])
                
                #Take action, observe new state S'
                new_state = self.makeMove(state, action)

                print('new state {}'.format(new_state))
        
                #Observed reward
                reward = self.getReward(new_state, AdjNodes, pos_1, finalpath)
                print('reward: {}'.format(reward))
                updates.append(reward)

                #evaluation of the total reward: sum of the reward obtained by each action
                r_sum += reward
                #calculate the average reward for each episode
                list_rew_per_ep.append(reward)

                if(reward == 1):
                    success[st[0]][en[0]] += 1

                for b in range(self.nNode):
                    if(bufferaction[b] >= 100):
                        reward = -1000

                if reward == 1 or reward == -1000:#!= -0.1:# terminal state when I repeated too many times the same move or I arrived to the goal
                    cumrew.append(reward)
                    status = 0 #to exit from the episode and start next episode

#REPLAY MEMORY
#we save in a buffer a certain number of tuple <s,a,r,s'>, state, action, reward, next state (following time step)
#then we sample a random batch from the buffer
                #reduces correlation between experiences in updating DNN
                #increases learning speed with mini-batches
                #reuses past transitions to avoid catastrophic forgetting
#                During gameplay all the experiences <s,a,r,s'> are stored in a replay memory. When training the network, random samples from the replay memory are used instead of the most recent transition. 
#This breaks the similarity of subsequent training samples, which otherwise might drive the network into a local minimum. Also experience replay makes the training task more similar to usual supervised learning, 
#which simplifies debugging and testing the algorithm. One could actually collect all those experiences from human gameplay and the train network on these.

                        #classic approach
#                       1) Select an item from Memory buffer
#                       2) optimize loss between Q-network and Q-learning target
                if (len(replay) < buffer): #if buffer not filled, add a new tuple
                    replay.append((state, action[0], reward, new_state))
                else: #if buffer full, perform mini-batch sampling for trainig and overwrite old values
                    print('replay')#it is full soon because in one episode i can have a lot of states

                    h = randint(0, buffer-1)
                    #randomly replace a tuple with a new one
                    replay[h] = (state, action[0], reward, new_state)
                    #randomly sample from the experience replay memory
                    # Return a batchSize length list of unique elements chosen from the population sequence (tuple in replay). 
                    minibatch = random.sample(replay, batchSize)

                    X_batch = []
                    Y_batch = []

                    for item_b in minibatch:
                        old_state_m, action_m, reward_m, new_state_m = item_b 
                     # First, predict the Q values of the next states. 
                        y_target = model.predict(old_state_m.reshape(1,self.nNode*(self.nFlow+1)), batch_size=Sbatch_size)
                        if reward_m != 1 and reward_m != -1000: #non-terminal state
                            update = reward_m + (gamma * np.max(model.predict(new_state_m.reshape(1,self.nNode*(self.nFlow+1)))))
                            print('update: {}'.format(update))
                            updates2.append(update)
                        else: #terminal state
                            update = reward_m
                            updates2.append(update)
                        y_target[0][action_m] = update
                        X_batch.append(old_state_m.reshape(self.nNode*(self.nFlow+1),))
                        Y_batch.append(y_target.reshape(self.nNode,))

# optimize loss between Q-network and Q-learning targe
                    # Fit the keras model.  

                        X_train = np.array(X_batch)
                        Y_train = np.array(Y_batch)
            #batch_size: integer. Number of samples per gradient update.
            #nb_epoch: integer, the number of epochs to train the model.
            #verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.   https://stackoverflow.com/questions/47902295/what-is-the-use-of-verbose-in-keras-while-validating-the-model
                        train_history = model.fit(X_train, Y_train, batch_size=batchSize, nb_epoch=NB_epoch, validation_data=None,verbose=1, callbacks=[history])


                    print("Game #: %s" % (i,))
                    

        # to reduce epsilon for exploitation-exploration
            if epsilon > 0.1:
                epsilon -= (1/epochs)

            if(i>0): #other epochs
#                print(r_avg_list)
#cumulative reward evaluated at the end of each episode
                r_avg_list.append(r_sum+r_avg_list[i-1])
                reward_list.append(r_sum)
                averageReward = np.mean(reward_list[-100:])
                if(len(list_rew_per_ep) != 0):
                    avg_rew_per_ep.append(sum(list_rew_per_ep)/len(list_rew_per_ep))
                # print(type(reward_list), type(100), type(best_score), type(averageReward))
                if(len(reward_list) >= 100 and best_score is None or best_score < averageReward):
                    best_score = averageReward
            elif(i==0): #first epoch
                r_avg_list.append(r_sum)
                reward_list.append(r_sum)



#        print(r_avg_list)
        plt.plot(avg_rew_per_ep)
        plt.show()

        plt.plot(r_avg_list)
        plt.draw()

        plt.show()

        plt.plot(reward_list)
        plt.show()

        plt.plot(cumrew)
        plt.show()

        return model

















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
        # st = self.findLoc(state, 0)


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




