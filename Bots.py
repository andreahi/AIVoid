import math
import random
import numpy as np

from Client import send_data
from FFModel import FFModel
from Game import *
from RunningMax import nextRunningMax
from Server import saveDataAndLabels


class Bot:
    # A bot that just plays the game using the values Theta1 and Theta2 for neural network parameters
    def __init__(self, Theta1, Theta2, game):
        self.Theta1 = Theta1
        self.Theta2 = Theta2
        self.game = game
        self.model = FFModel(11)

        
    def Sigmoid(self, x):
        return (1 + math.exp((-1)*x))**(-1)

    
    
    def PreProcess(self):
        # Use the relative coordinates of the falling objects to generate the input numpy vector the neural network (exploit game symmetry to use only one net)
        state_new = []
        for aster in self.game.asteroids:          # Scaling input values
            state_new.append(aster[0]/(self.game.Halfwidth+0.0))
            state_new.append(aster[1]/(self.game.Height+0.0))
        #state_new.append(1)     # Add the bias term
        #if action == 'L':
        #    for i in range(self.game.N):
        #        state_new[2*i] *= -1
        #layer1 = np.empty([2*self.game.N+1, 1])
        #for i in range(2*self.game.N):
        #    layer1[i, 0] = state_new[i]
        return np.asarray([state_new])

    

    def ForwardPropagate(self):
        # Evalue the neural network for the current game state with the given L/R action; Returns triple of values/vectors (one for each layer)
        state = self.PreProcess()
        # layer2_temp = np.dot(np.transpose(self.Theta1), layer1)
        # for i in range(layer2_temp.shape[0]):
        #     layer2_temp[i,0] = self.Sigmoid(layer2_temp[i,0])
        # layer2 = np.append(layer2_temp, [[1]], axis=0)
        # layer3 = np.dot(np.transpose(self.Theta2), layer2)
        # result = self.Sigmoid(layer3[0,0])
        predict = self.model.predict(state)
        self.state = state
        #return min(max(predict[0], 0), 1), min(max(predict[1], 0), 1)
        return predict[0], predict[1]
        #return (layer1, layer2, result)



    
    def TestStep(self):
        # Determines the optimal direction in the next move by using the given Theta1, Theta2 parameters
        outputL, outputR = self.ForwardPropagate()
        if outputL < outputR:
            self.game.ChangeDirection('L')
        else:
            self.game.ChangeDirection('R')
        result = self.game.GameOver()
        return result



class BotTrain(Bot):
    # A bot that performs reinforcement learning to opitmize the Theta1, Theta2 parameters in the neural network
    def __init__(self, GameParameters, HiddenSize=12, gamma=0.9995, GameOverCost=1, NSim=500, NTest=100, TestTreshold=200, NumberOfSessions=None, Inertia=0.8, p=0.0, a=1.0, epsilon=0.2, epsilon_decay_rate=1, discount = 0.999, p_decay_rate=0.5):
        Theta1 = np.random.uniform(-1.0, 1.0, (2*GameParameters["N"]+1, HiddenSize))
        Theta2 = np.random.uniform(-1.0, 1.0, (HiddenSize+1, 1))        
        game = Game(**GameParameters)
        Bot.__init__(self, Theta1, Theta2, game)

        self.GameParameters = GameParameters        
        self.HiddenSize = HiddenSize     # Size of the neural network hidden layer 
        self.gamma = gamma     # gamma parameter in the game cost function E[gamma^N]
        self.GameOverCost = GameOverCost     # Game Over Cost (set to 1.0 for standard game cost function E[gamma^N])
        self.NSim = NSim     # Number of consecutive learning games
        self.NTest = NTest     # Number of consecutive test games
        self.TestTreshold = TestTreshold    # Stop learning when median test score goes over TestTreshold (set to None for fixed number of sessions)
        self.NumberOfSessions = NumberOfSessions     # Number of learn train/test session (active only if TestTreshold = None)
        self.Inertia = Inertia     # (1 - Inertia) is the probability of resampling the game direction while learning
        self.p = p     # Probability of chosing learned move in reinforcement learning        
        self.a = a     # Reinforcement learning rate (set to 1.0 since it can be absorbed into gradient descent step factor)
        self.epsilon = epsilon     # Initial gradient descent step factor
        self.epsilon_decay_rate = epsilon_decay_rate     # Exponent in power decay for the gradient descent step factor
        self.discount = discount    # Discount exponent in reinforcement learning
        self.p_decay_rate = p_decay_rate    # Exponent in power decay for the policy greedines parameter

        self.counter = []    # Container for average and median test scores
        self.best_score = 0    # Best score among all training sessions
        self.batch_size = 100
        self.max_data_size = 2000000
        self.distributed_mode = False

        self.runningMax = 0
        
    def BackPropagate(self, output, expected, layer1, layer2):
        # Backpropagation algorithm for neural network; computes the partial derivatives with respect to parameters and performs the stochastic gradient descent
        delta3 = output - expected
        delta2 = delta3*self.Theta2
        for i in range(self.HiddenSize):
            delta2[i,0] *= layer2[i,0]*(1-layer2[i,0])
        for i in range(2*self.game.N+1):
            for j in range(self.HiddenSize):
                self.Theta1[i,j] -= self.epsilon*layer1[i,0]*delta2[j,0]
        for i in range(self.HiddenSize+1):
            self.Theta2[i,0] -= self.epsilon*delta3*layer2[i,0]
  

            
    def ReinforcedLearningStep(self, data, labels, explore_change):
        # Performs one step of reinforcement learning
        outputL, outputR = self.ForwardPropagate()

        if random.random() < explore_change:
            #            if random.random() > self.Inertia:
            new_direction = random.choice(['L', 'R'])
            if random.random() > 0.1:
                if self.game.Direction == 'R':
                    new_direction = 'R'
                elif self.game.Direction == 'L':
                    new_direction = 'L'
                else:
                    print("error wrong direction")

            self.game.ChangeDirection(new_direction)
            if new_direction == 'L':
                output = outputL
                out_index = 0
            elif new_direction == 'R':
                output = outputR
                out_index = 1
            else:
                print "error: invalid direction"
        else:
            output = min(outputL, outputR)
            out_index = [outputL, outputR].index(output)

            if outputL < outputR:
                self.game.ChangeDirection('L')
            else:
                self.game.ChangeDirection('R')
        state = self.state
        if random.random()<0.00002:
            # Occasionally prints out the current value of the network (useful for adjusting various learning parameters, especially gamma)
            print output
            
        result = self.game.UpdateStep()

        estimateL, estimateR = self.ForwardPropagate()
        #print "estimateL ", estimateL
        #print "estimateR ", estimateR

        if out_index == 0:
            #estimateL = min(estimateL, estimateR) * 0.99
            estimateL = outputL + (min(estimateL, estimateR) * 0.99 - outputL)
            estimateR = outputR
        elif out_index == 1:
            estimateL = outputL
            #estimateR = min(estimateL, estimateR) * 0.99
            estimateR = outputR + (min(estimateL, estimateR) * 0.99 - outputR)
        else:
            print "error out_index"
        if result[-1]:
            if out_index == 0:
                estimateL = self.GameOverCost
            else:
                estimateR = self.GameOverCost
            #estimate = self.GameOverCost
            #estimate = 1 - (float(min(count, prev_count)) / prev_count)

            #if result[1]:
            #    estimate *= self.gamma
        #expected = (1-self.a)*output + self.a*estimate


        #label = np.asarray([estimateL, estimateR])
        estimateL = max(estimateL, 1)
        estimateR = max(estimateR, 1)

        if random.random() < 0.0002 :
            print "label : ", [estimateL, estimateR]

        data.append(state[0].tolist())
        labels.append([estimateL, estimateR, out_index])
        #labels.append([random.randint(0,1), random.randint(0,1)])
        #self.model.train(self.state_new, label)

        return result

    

    def Training(self, data = [], labels = []):
        # Run NSim consecutive training games
        train_scores = []

        print("traning started")
        if self.distributed_mode:
            self.model = FFModel(11)
        for i in range(self.NSim):
            explore_change = random.random()/5
            #print "explore chance: ", explore_change
            for j in range(self.batch_size):
                stop = False
                count = 0
                step_labels = []
                while not stop:
                    count += 1
                    (update, kill, stop) = self.ReinforcedLearningStep(data, step_labels, explore_change)


                train_scores.append(self.game.counter)
                self.game = Game(**self.GameParameters)
                self.runningMax = nextRunningMax(count, self.runningMax)
                self.printR("count: " + str(count))
                self.printR("runningMax: " + str(self.runningMax))

                new_labels = []
                for idx in range(len(step_labels)):
                    index = step_labels[idx][2]
                    if count > self.runningMax:
                        if index == 0:
                            new_labels.append([step_labels[idx][0]*0.8, step_labels[idx][1]])
                        else:
                            new_labels.append([step_labels[idx][0], step_labels[idx][1]*0.8])

                    else:
                        if index == 0:
                            new_labels.append([step_labels[idx][0] * 1.2, step_labels[idx][1]])
                        else:
                            new_labels.append([step_labels[idx][0], step_labels[idx][1] * 1.2])

                labels += new_labels
            if self.distributed_mode:
                #send_data([data, labels])
                saveDataAndLabels(data, labels)
            else :
                self.model.train(data, labels)
                self.model.save()
            data = []
            labels = []

            self.printR("data size: " + str(len(labels)))
            if len(labels) > self.max_data_size:
                del data[: len(data) - self.max_data_size]
                del labels[: len(labels) - self.max_data_size]
        print("traning done ")
        return train_scores

    def printR(self, print_str, c=0.02):
        if random.random() < c:
            print print_str
            # prev_count = max(prev_count, count)

    def Testing(self):
        # Run NTest consecutive test games to evaluate learned performance; prints out all the test values and records average and median values
        s = 0
        alist = []
        print("testing started")

        for i in range(self.NTest):
            stop = False
            while not stop:
                stop = self.TestStep()
                self.game.UpdateStep()
            alist.append(self.game.counter)
            self.game = Game(**self.GameParameters)
        m1 = sum(alist)/(len(alist)+0.0)
        m2 = np.median(alist)
        self.counter.append((m1,m2))
        print "Test Results:", self.counter

        print("testing done")
        if m1 > self.best_score:
            self.best_score = m1
            np.savez("parameters_best", GameParameters = self.GameParameters, Theta1 = self.Theta1, Theta2 = self.Theta2)


        

    def TrainSession(self):
        # Performs a learning session until median scores achieves TestTreshold or for fixed number of learn/test sessions
        self.Testing()
        keep_going = True
        i = 0
        data = []
        labels = []
        while keep_going:
            i += 1
            print
            print "N:", self.game.N
            print "Session:", i
            train_scores = self.Training(data, labels)
            print "Train average and median score:", sum(train_scores)/(len(train_scores)+0.0), np.median(train_scores)
            #self.Testing()
            #print "Test Results:", self.counter
            #new, old = self.counter[-1][-1], self.counter[-2][-1]
            #self.epsilon *= (old/new)**self.epsilon_decay_rate
            print "Gradient Learning Rate:", self.epsilon
            #self.p = 1 - (1-self.p)*((old/new)**self.p_decay_rate)
            #if self.p < 0:
            #    self.p = 0.0
            print "p", self.p
            print
            if self.TestTreshold == None and not self.NumberOfSessions == None:
                if i >= self.NumberOfSessions:
                    keep_going = False
            elif not self.TestTreshold == None:                 
                if self.counter[-1][-1] >= self.TestTreshold:
                    keep_going = False

            
