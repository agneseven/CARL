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
from keras.optimizers import Adam

#from sklearn import linear_model, datasets
#from sklearn.model_selection import RandomizedSearchCV
import subprocess
import os
import sys
from topoGeneration import topoGeneration
#from briteGen import *
import tensorflow as tf
from keras.models import model_from_json



class ReinforcementLearning:

    def __init__(self, nFlow, myG, nNode, nHost, parameters, activeFlows, double_dqn):

        self.nNode = nNode
        self.nFlow = nFlow
        self.double_dqn = double_dqn
        self.nHost = nHost
        self.myG = myG
        self.parameters = parameters
        self.activeFlows = activeFlows
        self.episodes = None
        self.gamma = None
        self.alpha = None
        self.Sbatch_size = None
        self.epochs = None
        self.epsilon = None
        self.bufferSize = None
        self.bufferBatch = None
        self.adjMatrix = nx.adjacency_matrix(self.myG)




#initalize environment
    def resetEnv(self):

        state = np.zeros((self.nNode,1,self.nFlow + 1))
        #randomly place new flow: 1 only if the node is a host
        a,b = self.randPair(0, self.nHost)
        # state[a,b,0] = 1
        state[0,0,0] = 1

        #randomly place other flows (strating from one node till the maximum number of flow)
        for n in range (1,self.nFlow-self.activeFlows):
            a1,b1 = self.randPair(0,self.nNode)
            state[a1,b1,n] = 1

        #randomly place goal of the new flow: 1 only if the node is a host
        a_g,b_g = self.randPair(0,self.nHost)
        # state[a_g,b_g,self.nFlow] = 1
        state[2,0,self.nFlow] = 1

        print("starting location")
        print(state)

        flow1 = self.findLoc(state, 0)
        goal1 = self.findLoc(state, self.nFlow)

        if(flow1 == goal1):
#            print('Invalid grid. Rebuilding..')
            return self.resetEnv()
        return state


#define possible actions
    def possibleActions(self, players_loc):
        #we need adjMatrix only to know the valide actions. The valide actions are the adjacent nodes of the current node
        nonAdjNodes = [] 
        AdjNodes = []
        print('vv', (self.nFlow-self.activeFlows))
        print('vv', (players_loc[0][0],1))
        if(self.nFlow-self.activeFlows > 1):
            print('vv', (players_loc[0][1],1))
        for f in range(self.nFlow-self.activeFlows):
            nonAdjNodesList = []
            for k in range(self.nNode):
                if (self.adjMatrix.todense()[players_loc[0][f],k][0] == 0 and players_loc[0][f] != k):
                    nonAdjNodesList.append(k)
            nonAdjNodes.append(nonAdjNodesList)
        nonAdjNodes[0].append(players_loc[0][0])

        NodeList = range(self.nNode)

        for f in range(self.nFlow-self.activeFlows):
            AdjNodes.append(list(set(NodeList).difference(set(nonAdjNodes[f][:]))))

        return AdjNodes, nonAdjNodes


#update state when making a move
    def makeMove(self, state, actions):
        #current location of flow1
        flow1 = self.findLoc(state, 0)
        oldloc = []
        for n in range(len(actions)):
            oldloc.append(self.findLoc(state, n+1)) 
        print("old loc", oldloc)
        #goal of flow1
        goal1 = self.findLoc(state, self.nFlow)
        #future location of flow1
        new_loc_flow1 = (actions[0],0)
        #future location of the flow1 and of the other flows
        new_loc = []
        for n in range(len(actions)):
                new_loc.append(actions[n])

        #set to 1 the new loc of the new flow in the matrix
        state[new_loc_flow1][0] = 1

#comment to keep the old position of Flow1
        if(new_loc_flow1 != flow1):
            state[flow1][0] = 0

        # for n in range(len(new_loc)-1):
        #     if(new_loc[n] != oldloc[n][0]):
        #         state[flow1][0] = 0


        #update the other flows
        if(len(new_loc) > 1):
            for f in range(len(new_loc)):
                state[new_loc[f],0][f] = 1

        return state



#calculate reward    
    def getReward(self, state, start, path, players_loc):
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
        print("state", state)
        flow1 = self.findLoc(state, 0)
        print("flow1", flow1)
        AdjNodes, nonAdjNodes = self.possibleActions(players_loc)
        #goal of flow1
        goal1 = self.findLoc(state, self.nFlow)
        sumRow = 0

        for f in range(self.nFlow):
            if(state[flow1][f] != 0):
                sumRow += 1
        maxSumRow = self.nFlow
        sumRowVicini = np.zeros((self.nNode))

        for f in range(self.nFlow):
            for n in range(self.nNode):
                if(state[n,0][f] != 0).all():
                    sumRowVicini[n] += 1 #conto quanti flussi ci sono in ogni nodo

        vectorvicini = 0
        otherAdjNodes = np.setdiff1d(AdjNodes[0],flow1[0])

##I want the reward included in [-1, 1]
        # penalty = 1/self.nFlow
#1) goal not yet achieved, the node has already been visited
# a higher penalty to avoid loops
#        visited = 0
#        for hop in path:
#            if(flow1[0] == hop):
#                visited = 1
#
#        if(flow1 != goal1 and visited == 1): #sono gia passata per quel nodo
#            return -0.1
        print("self.nFlow", self.nFlow)
        print("self.activeFlows", self.activeFlows)
        print("self.nFlow-self.activeFlows", self.nFlow-self.activeFlows)
        penalty = 1/float((self.nFlow-self.activeFlows)*self.nFlow)

#2) goal not yet achieved, only flow in that node
# a small penalty only to avoid too many movements
        if (flow1 != goal1 and sumRow == 1): #
            return -penalty
#            return penalty


#3) goal not yet achieved, more flows in that node,  not the only possible destination, other possible destinations are emptier
# a penalty proportional to the number of flows in the node
        for v in otherAdjNodes:
#            print(v)
            if(flow1 != goal1 and sumRow != 1 and len(AdjNodes[0]) != 1): #if other neighbors have a lower number of flow exit
            # if(flow1 != goal1 and sumRow != 1 and len(AdjNodes[0]) != 1 and (sumRowVicini[v]+1) < sumRow): #if other neighbors have a lower number of flow exit
#                print('qui {} {}'.format(sumRowVicini[v], sumRow))
                vectorvicini = 1

        if(vectorvicini == 1):
            print('penalty * sumRow', -penalty * sumRow)
            return -(penalty * sumRow)
            # return -0.25 * (sumRow)

#            return penalty * sumRow

#4) goal not yet achieved, more flows in that node, that node is the only possible destinatioN
# a small penalty only to avoid too many movements, like (2)

        if(flow1 != goal1 and sumRow != 1 and len(AdjNodes[0]) == 1):
            print('lpenalty', -penalty)
            return -penalty
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



#hyperparameter  
    def setHyperpar(self):

        #number of total games we want to play for training the NN, one episode = one game
        self.episodes = int(self.parameters[0])
        #discount factor [0,1] = 0-> we do not care about the future reward that we can obtain from that state, making that action
        #discount factor [0,1] = 1-> we do care about the future reward that we can obtain from that state, making that action
        self.gamma =  0.975 #float(self.parameters[1])  

        #GRADIENT DESCENT parameters
        #learning rate = di quanto mi sposto sullacurva per trovare il minimo
        #small -> small steps, it can converge slowly
        #big -> big steps, it can diverge
        #0.0001<alpha<0.01
        self.alpha =  0.00025
        #number of example that I use to calculate the gradient descent (mini-batch optimization)
        self.Sbatch_size = 32
        #One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
        #One epoch leads to underfitting of the curve in the graph (below).
        #????????????????
        self.epochs = 1

        #EXLOITATION-EXPLORATION
        #epsilon-greedy = exploration (random), exploitation (greedy)
        self.epsilon = 1
        self.epsilon = float(self.epsilon)

        #REPLAY MEMORY
        #total size of the buffer
        self.bufferSize = int(self.parameters[4])
        #batch of the replay buffer
        self.bufferBatch = int(self.parameters[3])
        self.stepUpdateTargetNetwork = 50


#create DQN
    def createDQNNet(self):
        self.input_size = self.nNode*(self.nFlow+1)
        self.output_size = self.nNode
        # The Sequential class is used to define a linear stack of network layers which then, collectively, constitute a model.
        # create model
        model = Sequential() 
        #Sequential constructor to create a model, which will then have layers added to it using the add() method.
        # first (input) layer of the model
        #layer of type Dense ("Just your regular densely-connected NN layer").
        #The Dense layer has output size (self.nNode*(self.nFlow+1)+self.nNode)/2 -> ?
        #input size = self.nNode*(self.nFlow+1) -> environment state size (row = nodes; column= flows)
        #activation function = To check the Y values produced by a neuron and decide wheter outside connections should conside this neuron as fired
        #or not. or ratherlet us say activated or nor (best funzion for hidden layers is ReLu)
        # kernel_initializer = the way to set the initial random weights of Keras layers
        model.add(Dense(((self.nNode*(self.nFlow+1)+self.nNode)/2), input_dim=self.input_size, kernel_initializer ='he_normal', activation='relu'))
         # only the first layer of the model requires the input dimension to be explicitly stated; the following layers are able to infer from the previous linear stacked layer
         #next Dense layer
         #output size = total number of nodes, because we want one of the nodes as next hop
        model.add(Dense(self.output_size, kernel_initializer ='he_normal', activation='linear'))
         # RMSprop = optimization algorithm, adaptation of rprop algorithm for mini-batch learning. Simple learning rate schedule is not dynamic enough to handle changes in input during the training. Many RL training uses RMSProp or Adam optimizer.
        #lr is the learning rate
        rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)#loss='logcosh
        # loss = tf.losses.huber_loss(delta=2.0)
        #Before training a model, you need to configure the learning process, which is done via the compile method. 
        #loss = A loss function. This is the objective that the model will try to minimize. 
        #optimizer = An optimizer.
        #metrics = A list of metrics. For any classification problem you will want to set this to metrics=['accuracy']. A metric could be the string identifier of an existing metric or a custom metric function.
        #model.compile(loss='logcosh', optimizer=rms, metrics=['accuracy']) #Compile defines the loss function, the optimizer and the metrics. That's all. If you compile a model again, you will lose the optimizer states.
        #adam has the learning rate 0.0001 as input
        model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['accuracy', 'mse', 'mae', 'mape', 'cosine']) #Compile defines the loss function, the optimizer and the metrics. That's all. If you compile a model again, you will lose the optimizer states.
        #Metric values are recorded at the end of each epoch on the training dataset.
        #All metrics are reported in verbose output and in the history object returned from calling the fit() function. 
        # prediction = model.predict(state.reshape(1,self.nNode*(self.nFlow+1)), batch_size=Sbatch_size)
        return model


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



#retrieve location of all flows
    def getallLocations(self, state):
        players_loc = []
          
        #retrieve location and action of other agents
        for n in range(self.nFlow-self.activeFlows):
            players_loc.append(self.findLoc(state, n))#format : (current_node, 0)
        return players_loc


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

    def getRandomAction(self, players_loc):
        AdjNodes, nonAdjNodes = self.possibleActions(players_loc)
        print("possible actions", AdjNodes)
        index = np.random.randint(0,len(AdjNodes[0]))
        return AdjNodes[0][index]


    def getQvalAction(self, state, model, players_loc):
        qval = model.predict(state.reshape(1,self.input_size), batch_size=self.Sbatch_size)
        qvalValid = np.copy(qval)
        #so the max q value cannot be one associated to a node that is not a neighobor
        adj, nonAdjNodes = self.possibleActions(players_loc)
        for non in nonAdjNodes[0][:]:
            qvalValid[0, non] = -100000
        
        return np.argmax(qvalValid)

    def getActionOtherAgents(self, finalpath, players_loc):
        action = [0] * (self.nFlow-self.activeFlows-1) 
        index2 = [0] * (self.nFlow+1)
        AdjNodes, nonAdjNodes = self.possibleActions(players_loc)
        print("otherag", AdjNodes)
        print("otherag", AdjNodes[0])
        print("otherag", AdjNodes[1])
        print("otherag", range(1,self.nFlow-self.activeFlows))
        #actions of the other flows are simply random actions among the valid actions (neighbora)
        if(len(finalpath) < self.nNode):
            for f in range(1,self.nFlow-self.activeFlows):
                index2[f] = np.random.randint(0, len(AdjNodes[f]))
                print("index2[f]", index2[f])
                action[f-1] = AdjNodes[f][index2[f]]
        action_ = np.copy(action)
        if(len(finalpath) >= self.nNode):
            for f in range(1,self.nFlow-self.activeFlows):
                action[f-1] = action_[f-1]
        return action  

#update target network
# This function helps us to copy one set of variables to another
# In our case we use it when we want to copy the parameters of DQN to Target_network

    def updateTargetNetwork(self, model, targetModel):
        weights = model.get_weights()
        targetModel.set_weights(weights)


    def saveModel(self, model, name):
        # serialize model to JSON
        model_json = model.to_json()
        filename = "%s.json" % name
        with open(filename, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        fileN2 = "%s.h5" % name
        model.save_weights(fileN2)
        print("Saved model to disk")

    def loadModel(self, name):
                # load json and create model
        filename = "%s.json" % name
        json_file = open(filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        fileN2 = "%s.h5" % name
        loaded_model.load_weights(fileN2)
        print("Loaded model from disk")
        return loaded_model



    def training(self):

        #set Hyperparameters
        self.setHyperpar()
        history = History()


        #varibales
        increment = 0
        replay = [] #initialize buffer for replay memory
        arrayepsilon = []
        rewardsarray = []
        reward_sum = 0
        totalStepCounter = 0 #for updating target NN
        avg_rew_per_ep = []
        rewardsumarray = []
        arrayAvgRewperEp = []

        adjMatrixX = nx.adjacency_matrix(self.myG)
        print("adj", adjMatrixX.todense())
        #create target NN
        target_model = self.createDQNNet()
        print(target_model.summary())
        print("pesi target", target_model.get_weights())

        #create online NN
        online_model = self.createDQNNet()
        print("pesi online", online_model.get_weights())

        #start training over episodes: one episode is one entire game
        for ep in range(self.episodes):

            #variables
            finalpath = []
            actions = []
            actions.append(0)
            status = 1    
            players_loc = []   
            numberSteps = 0
            AvgRewperEp = []
     

            #change the number of flows in the state incrementally
            if (ep == (self.episodes/self.nFlow + increment)):
                # activeFlows will not change at the first episode
                self.saveModel(target_model, self.activeFlows)
                target_model = self.createDQNNet()
                print(target_model.summary())
                print("pesi target", target_model.get_weights())

                #create online NN
                online_model = self.createDQNNet()
                print("pesi online", online_model.get_weights())

                replay = [] #initialize buffer for replay memory

                self.activeFlows = self.activeFlows - 1
                increment = increment + self.episodes/self.nFlow
                # print('incremento {}'.format(i))
                self.epsilon = 1


            print("Game n: %s" % (ep,))
            #INITIALIZE THE STARTING STATE
            state = self.resetEnv()

            #save path of flow1
            pos_1 = self.findLoc(state, 0)
            finalpath.append(pos_1[0])

            #while game still in progress status = 1, when game ends, the status is set to 0
            #it can finish because flow1 reached the goal or because there were too many moves
            #for each time step in one episode
            while(status == 1):

                #retrieve locations of all flows but the first one
                players_loc = []
                myagentloc = self.findLoc(state, 0)
                goal1 = self.findLoc(state, self.nFlow)
                # players_loc.append(myagentloc) 
                players_loc.append(self.getallLocations(state))
                print("players_loc", players_loc)


    #EXPLORATION
                if (random.random() < self.epsilon): #choose random action
                    #pick a random neighbor
                    print("random")
                    actions[0] = self.getRandomAction(players_loc)
    #EXPLOITATION
                else:
                    print("qval action")
                    actions[0] = self.getQvalAction(state, online_model, players_loc) 

                if((self.nFlow-self.activeFlows)>1):
                    print("self.nFlow-self.activeFlows", self.nFlow-self.activeFlows)
                    actions[1:] = self.getActionOtherAgents(finalpath, players_loc)

                finalpath.append(actions[0])
                print("array actions", actions)
    #EXECTUTE ACTION AND GET
    #next state       
                new_state = self.makeMove(state, actions)
    #Reward from that action
                reward = self.getReward(new_state, myagentloc, finalpath, players_loc)
                print("reward", reward)

                rewardsarray.append(reward)
                reward_sum += reward
                rewardsumarray.append(reward_sum)
                if(len(rewardsarray) != 0):
                    avg_rew_per_ep.append(sum(rewardsarray)/len(rewardsarray))
                AvgRewperEp.append(reward)
               
                if (reward == 1):
                    status = 0
                    print("finalpath", finalpath)
                numberSteps += 1 #If we're taking more than 30 actions, just stop, we probably can't win this game
                if (numberSteps > 10000):
                    reward = -1000
                    status = 0
                    print("more than 10000")




#STORE IN REPLAY MEMORY.
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
                if (len(replay) < self.bufferSize): #if buffer not filled, add a new tuple
                    replay.append((state, actions[0], reward, new_state))

                else: #if buffer full, perform mini-batch sampling for trainig and overwrite old values
                    X_batch = []
                    Y_batch = []
                    print("X_batch", X_batch)
                    print("Y_batch", Y_batch)

                    #randomly replace a tuple with a new one
                    h = randint(0, self.bufferSize-1)
                    replay[h] = (state, actions[0], reward, new_state)
#SAMPLE MINI BATCH FRO MREPLAY MEMORY
                    #randomly sample from the experience replay memory
                    # Return a batchSize length list of unique elements chosen from the population sequence (tuple in replay). 
                    minibatch = random.sample(replay, self.bufferBatch)

                    for item_b in minibatch:
                        old_state_m, action_m, reward_m, new_state_m = item_b 
#Calculate loss between 
#output from the network for the action in the experience tuple: q-value doing action_m -> update in 3 lines. 
#corresponding optimal Q-value, or target Q-value, for the same action.
                        qValues = online_model.predict(old_state_m.reshape(1,self.input_size))
                        qValuesNewState = target_model.predict(new_state_m.reshape(1,self.input_size))
                        if reward_m != 1 and reward_m != -1000: #non-terminal state
                            targetValue = reward_m + (self.gamma * np.max(qValuesNewState))
                            #updates2.append(update)
                        else: #terminal state
                            targetValue = reward_m
                            #updates2.append(update)  
  
                        X_batch.append(old_state_m.reshape(self.input_size,)) 
                        Y_sample = qValues 
                        Y_sample[0][action_m] = targetValue
                        Y_batch.append(Y_sample.reshape(self.output_size,))

#Gradient descent updates weights in the policy network to minimize loss.
                    #gradient descent 
                    X_train = np.array(X_batch)
                    Y_train = np.array(Y_batch)
                    history = online_model.fit(X_train, Y_train, batch_size=self.bufferBatch, nb_epoch=1, validation_data=None,verbose=1, callbacks=[history])
#every C steps reset Q
            totalStepCounter += 1
            print("total stepCounter", totalStepCounter)
            if totalStepCounter % self.stepUpdateTargetNetwork == 0:
                self.updateTargetNetwork(online_model, target_model)
                print "updating target network"


        # to reduce epsilon for exploitation-exploration
            if self.epsilon > 0.1:
                print("1/self.episodes", (1/float(self.episodes/self.nFlow)))
                self.epsilon -= 1/float(self.episodes/self.nFlow)
            arrayepsilon.append(self.epsilon)

            print(target_model.summary())
            print(online_model.summary())
            print("pesi online", online_model.get_weights())
            print("pesi target", target_model.get_weights())
            arrayAvgRewperEp.append(sum(AvgRewperEp)/len(AvgRewperEp))


        # print("reward_sum", avg_rew_per_ep)
        # print("reward_sum", rewardsumarray)
        # print("reward_sum", reward_sum)
        # print("epsilon", self.epsilon)
        # print("rewardsarray", rewardsarray)

        self.saveModel(target_model, self.activeFlows)

        # plot metrics
        plt.plot(arrayAvgRewperEp)
        plt.show()
        plt.plot(avg_rew_per_ep)
        plt.show()
        plt.plot(rewardsumarray)
        plt.show()

        plt.plot(rewardsarray)
        plt.show()
        plt.plot(arrayepsilon)
        plt.show()

        return target_model





    def testGrid(self, path, start, end):
        state = np.zeros((self.nNode,1,self.nFlow + 1))
        #place flow1
        a,b = start, 0
        state[a,b,0] = 1
        
        #place other flows

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
        indexPATH = []
        for f in range(self.nFlow-1):
            indexPATH.append(len(path[f])-1) #current position

        return state, indexPATH






    def testAlgo( self, init, path, G, start, end):
        if init==8:
            state, indexPATH = self.testGrid(path, start, end)

        #variables
        status = 1
        numberSteps = 0
        loop = 0
        win = 0
        fail = 0
        finalpath = []
        actions = [] #next action of all the flows
        actions.append(0)

        self.activeFlows = 0
        players_locFlow = []
        players_locFlow.append(self.getallLocations(state))
        print("players_locFlow", players_locFlow)

        self.activeFlows = players_locFlow[0].count(None)
        print(" self.activeFlows",  self.activeFlows)

        
        model = self.loadModel(self.activeFlows)

        while(status == 1):
            players_loc = []
            myagentloc = self.findLoc(state, 0)
            players_loc.append(self.getallLocations(state))
            print("players_loc", players_loc)


            actions[0] = self.getQvalAction(state, model, players_loc) 
            finalpath.append(actions[0])
            new_state = self.makeMove(state, actions)
            reward = self.getReward(new_state, myagentloc, finalpath, players_loc)

            if reward == 1:
                status = 0
                win = 1
                print("finalpath", finalpath)

            numberSteps += 1 #If we're taking more than 30 actions, just stop, we probably can't win this game
            if (numberSteps > 30):
                loop = 1
                print("loop out")
                break


        if reward == -1000:
            print("fail")
            fail = 1

        return finalpath, win, loop, fail









              


            







