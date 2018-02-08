import numpy as np
from itertools import product
from collections import deque, defaultdict
from .generalFunctions import *
from math import pi

class QLearner():
    def __init__(self, federate, numericactions, states, seed = 0):
        self.states = states
        self.stateDict = {e: i for i,e in enumerate(self.states)}
        self.actions = numericactions
        # print(numericactions, self.states)
        self.q = np.zeros((len(self.states), len(numericactions)))
        self.gamma = 0.9
        self.alpha = 0.8
        # self.n_states = len(self.stateDict)
        # self.n_actions = len(numericactions)
        self.epsilon = 0.05
        self.random_state = federate.context.masterStream
        self.time = 0
        self.federate = federate


    def splitresolution(self):
        columns = [self.q[:, i] for i in range(self.q.shape[1])]
        meancolumns = [(columns[i]+columns[i-1])/2. for i in range(1, len(columns))]
        newcolums = (len(columns) + len(meancolumns))*[None]
        newcolums[::2] = columns
        newcolums[1::2] = meancolumns

        actionslist = list(self.actions)
        meanactionslist = [(actionslist[i]+actionslist[i-1])/2. for i in range(1, len(actionslist))]
        newactions = (len(actionslist) + len(meanactionslist))*[None]
        newactions[::2] = actionslist
        newactions[1::2] = meanactionslist

        # print(newcolums)
        self.q = np.column_stack(newcolums)
        self.actions = newactions

class QlearnerStorage(QLearner):
    def __init__(self, federate, numericactions, states, seed=0):
        QLearner.__init__(self, federate, numericactions, states, seed)
        self.recentstateactions = defaultdict(deque)
        self.timeStateDict = {}
        self.actionlist = deque([])
        self.inertia = 6
        self.elementInertia = defaultdict(lambda: self.inertia)
        self.elementAction = defaultdict(list)
        self.delta = 3
        self.splitflag = True


    def update_q(self, element, rewards):
        self.time = element.federateOwner.time
        name = element.name
        while self.recentstateactions[name] and self.recentstateactions[name][0][2] < self.time - self.delta:
            self.recentstateactions[name].popleft()
        # current_state = self.stateDict[(int(element.capacity - element.content), element.section)]
        # self.timeStateDict[self.time] = current_state
        # print("recent state actions:", self.recentstateactions[element.name])
        # print("rewards:", rewards)
        N = float(len(self.recentstateactions))
        S = len(self.states)
        A = len(self.actions)

        for state, action, t in self.recentstateactions[name]:
            kernel = calGaussianKernel(state, action, S, A)
            alphamesh = self.alpha * kernel

            for s in range(S):
                for a in range(A):
                    qsa = self.q[s, a]
                    new_q = qsa + alphamesh[s, a] * (rewards/N + self.gamma * max(self.q[s, :]) - qsa)
                    self.q[s, a] = new_q

                    # qsa = self.q[state, action]
                    # # print(time, element.timeStateDict[time + 1], self.stateDict[element.timeStateDict[time + 1]])
                    # # new_q = qsa + self.alpha * (rewards/N + self.gamma * max(self.q[self.stateDict[element.timeStateDict[t + 1]], :]) - qsa)
                    # new_q = qsa + self.alpha * (rewards/N + self.gamma * max(self.q[self.stateDict[element.timeStateDict[t + 1][0]], :]) - qsa)
                    # # print("qsa, nextstate, max:", name, qsa, element.timeStateDict[t], element.timeStateDict[t + 1], max(self.q[self.stateDict[element.timeStateDict[t + 1]], :]), new_q)
                    # self.q[state, action] = new_q
                    # # print("new q:", state, new_q, [int(e) for e in list(self.q[state])])
                    # # renormalize row to be between 0 and 1
                    # # rn = self.q[state] / np.sum(self.q[state])
                    # # self.q[state] = [round(r, 2) for r in rn]

                    # print("SUM OF Q:", self.q.sum())
            # print(self.q[state])
        # print("Q matrix:", self.q)

    def getAction(self, element):
        self.time = element.federateOwner.time
        # self.elementInertia[element.name] -= 1
        if self.elementAction[element.name] and self.time -  self.elementAction[element.name][1] < self.inertia:
            return 6*[self.elementAction[element.name][0]]
            # return self.recentstateactions[element.name][-1][1]

        # if self.time > 300 and self.splitflag:
        #     self.splitresolution()
        #     self.splitflag = False
        # self.elementInertia[element.name] = 10

        # current_state = self.stateDict[(int(element.capacity - element.content), element.section)]
        current_state = self.stateDict[int(element.capacity - element.content)]


        # valid_moves = self.r[current_state] >= 0
        if self.elementAction[element.name]:
            lastaction = self.elementAction[element.name][0]
            lastindex = self.actions.index(lastaction)
        else:
            # lastindex = (1+len(self.actions))//2
            cost = list(self.federate.costDic.values())
            avgcost = 1.*sum(list(self.federate.costDic.values()))/len(list(self.federate.costDic.values()))
            lastindex = findClosestIndex(avgcost,self.actions)
            # print("Action: cost vs action:", cost, self.actions[lastindex])

        # newepsilon = self.epsilon*max(0.1, (1 - self.time/3000))
        if self.random_state.random() < self.epsilon or np.sum(self.q[current_state]) <= 0:
            action = self.random_state.choice(self.actions[max(0, lastindex-1):min(lastindex+2, len(self.actions))])
            # action = random.choice(self.actions)
        else:
            # if np.sum(self.q[current_state]) > 0:
            # print("q row:", [int(e) for e in self.q[current_state]])
            maxq = max(self.q[current_state])
            indices = [i for i, e in enumerate(self.q[current_state]) if e == maxq]
            action = self.actions[self.random_state.choice(indices)]
            # print("maximual action:", action)

        # print([element.name, current_state, self.actions.index(action), self.time])
        self.recentstateactions[element.name].append(tuple([current_state, self.actions.index(action), self.time]))
        # print("Action:", element.name, action)
        self.actionlist.append(action)
        self.elementAction[element.name] = (action, self.time)
        return 6*[action]



class QlearnerCost(QLearner):
    def __init__(self, federate, numericactions, states=list(range(6)), seed=0):
        QLearner.__init__(self, federate, numericactions, states, seed)
        self.stateActionDict = defaultdict(list)
        self.priceEvolution = []
        self.federateAction = defaultdict(tuple)
        self.inertia = 10

    def update_q(self, action, reward):
        # print("update action reward:", action, reward)
        actionindex = self.actions.index(action)
        self.time = self.federate.time
        action_sector = self.time%6

        M = len(self.states)
        N = len(self.actions)

        kernel = calGaussianKernel(action_sector, actionindex, M, N)
        alphamesh = self.alpha*kernel

        for s in range(M):
            for a in range(N):
                qsa = self.q[s, a]
                new_q = qsa + alphamesh[s,a] * (reward + self.gamma * max(self.q[s, :]) - qsa)
                self.q[s, a] = new_q

        # print(self.q)

    def getAction(self):
        self.time = self.federate.time
        current_state = self.time%6

        if self.federateAction[self.federate.name] and self.time - self.federateAction[self.federate.name][1] < self.inertia:
            return self.federateAction[self.federate.name][0]

        if self.stateActionDict[current_state]:
            lastaction = self.stateActionDict[current_state][0]
            lastindex = self.actions.index(lastaction)

        else:
            lastindex = len(self.actions)//2 - 1

        # newepsilon = self.epsilon*max(0.1, (1 - self.time/3000))
        if self.random_state.random() < self.epsilon or np.sum(self.q[current_state]) <= 0:
            action = self.random_state.choice(self.actions[max(0, lastindex-1):min(lastindex+2, len(self.actions))])
            # action = random.choice(self.actions)
        else:
            # if np.sum(self.q[current_state]) > 0:
            # print("q row:", [int(e) for e in self.q[current_state]])
            maxq = max(self.q[current_state])
            indices = [i for i, e in enumerate(self.q[current_state]) if e == maxq]
            action = self.actions[self.random_state.choice(indices)]
            # print("maximual action:", action)

        self.stateActionDict[current_state] = (action, self.time)

        # print("cost action:", current_state, action)
        # self.priceEvolution.append((self.federate.time, action))
        self.federateAction[self.federate.name] = (action, self.time)
        return action
