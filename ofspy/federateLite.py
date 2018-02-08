import re
# import random
from .elementLite import Satellite, GroundStation
from .qlearner import QlearnerStorage, QlearnerCost
from .generalFunctions import matchVariance
from collections import defaultdict


class FederateLite():
    def __init__(self, name, context, costSGL, costISL, storagePenalty = 100, strategy = 1):
        """
        @param name: the name of this federate
        @type name: L{str}
        @param initialCash: the initial cash for this federate
        @type initialCash: L{float}
        @param elementlist: the elementlist controlled by this federate
        @type elementlist: L{list}
        @param contracts: the contracts owned by this federate
        @type contracts: L{list}
        @param operations: the operations model of this federate
        @type operations: L{Operations}
        """
        self.context = context
        self.name = name
        # self.initialCash = initialCash
        self.cash = 0 #self.initialCash
        self.cashlist = []
        self.elements = []
        self.satellites = []
        self.stations = []
        # self.operation = operation
        self.costDic = {'oSGL': costSGL, 'oISL': costISL}
        self.storagePenalty = storagePenalty
        self.tasks = {}
        self.transcounter = 0
        self.transrevenue = 0.
        self.time = context.time

        self.taskduration  = {i: 2. for i in range(1,7)}
        self.taskvalue = {i: 500. for i in range(1,7)}
        self.taskcounter = {i: 0 for i in range(1,7)}
        self.pickupOpportunities = 0

        self.activeTasks = set([])
        self.supperGraph = None
        self.pickupProbability = context.pickupProbability
        self.uniqueBundles = []
        self.nodeElementDict = {}

        self.storagelearning = False
        self.costlearning = False
        self.stochastic = (costSGL == -3)
        # self.strategyDict = {1: 'QLearning'}
        self.strategy = strategy
        self.sharedlinks = defaultdict(int)


    def getElements(self):
        """
        Gets the elementlist controlled by this controller.
        @return L{list}
        """
        return self.elements[:]


    def getTasks(self):
        """
        Gets the contracts controlled by this controller.
        @return L{list}
        """
        return self.tasks


    def ticktock(self):
        """
        Ticks this federate in a simulation.
        @param sim: the simulator
        """
        # print "federete tick tock"
        self.time = self.context.time
        for element in self.elements:
            element.ticktock()

    def setCost(self, protocol, cost):
        self.costDic[protocol] = cost

    def getCost(self, protocol = None, task = None, federate = None):
        # print(self.name, federate.name, self.name == federate.name, self.costDic[protocol])
        if federate and self.name == federate.name:
            return 0.


        if self.stochastic:
            # a = max(min(self.costDic[protocol], 1200-20), 20)
            # b = 1200 - a
            # a, b = matchVariance(a, b, 0.015)
            # ret = 1200*self.context.masterStream.betavariate(a, b)
            # # print(self.costDic[protocol], ret)
            # ret = 100*round(11*self.context.masterStream.random())
            numbercount = {0:1, 100: 2, 200: 3, 300:4, 400:5, 500: 6, 600: 7, 700:6, 800: 5, 900: 4, 1000: 3, 1100:2, 1200: 1}
            numlist = []
            for n, c in numbercount.items():
                while c> 0:
                    numlist.append(n)
                    c -= 1
            return self.context.masterStream.choice(numlist)


        return self.costDic[protocol]
        # key = '{}-{}'.format(federate, protocol)
        # return self.costDic[protocol] if key not in self.costDic else self.costDic[key]

    def addTransRevenue(self, protocol, amount):
        if protocol in self.transCounter:
            self.transrevenue[protocol] += amount
            self.transcounter[protocol] += 1
        else:
            self.transrevenue[protocol] = amount
            self.transcounter[protocol] = 1

    def getTransRevenue(self):
        return self.transrevenue

    def getTransCounter(self):
        return self.transcounter

    def getStorageCostList(self, element, taskvaluelist = None):
        # print("federate storage penalty:", self.storagePenalty)
        # if self.storagePenalty == -2:
        #     return self.qlearnerstorage.run(element)

        if self.storagePenalty>=0:
            return 6*[self.storagePenalty]

        elif self.storagePenalty == -3:
            return self.context.masterStream.sample(range(1200), 6)

        section = element.section
        # assert section in range(1, 7)
        storagecostlist = []
        temptime = self.time
        # print("storage cost pickup probability:", self.pickupProbability)
        for i in range(1, 7):
            # print(i, section, len(self.taskduration), len(self.taskvalue),len(taskvaluelist))
            storagecostlist.append(self.pickupProbability*(self.taskvalue[section]/self.taskduration[section]) + taskvaluelist[i-1] - taskvaluelist[min(i, 5)])
            temptime += 1
            section = section%6+1

            # print("storage cost:", [int(e) for e in storagecostlist])
        # print("storage penalty -1:", storagecostlist)
        # return [e/max(1,element.capacity-element.content) for e in storagecostlist]
        # print("storage coef:", self.storagePenalty,)
        return 6*[abs(self.storagePenalty*storagecostlist[0]/max(1,element.capacity-element.content))]

    def discardTask(self):
        for e in self.elements:
            for stask in e.savedtasks:
                if stask.getValue(self.time)<=0:
                    self.defaultTask(self, stask)

    def reportPickup(self, task):
        self.context.addPickup(task.taskid)
        self.activeTasks.add(task)

    def finishTask(self, task):
        path = task.path
        taskvalue = task.getValue(self.time)
        # print(self.time, self.context.ofs.maxTime)
        # if self.time >= self.context.ofs.maxTime//2:
        self.cash += taskvalue
        # assert path.pathBid == sum(path.linkbidlist)
        tempcost = 0

        for cost, federate in zip(path.linkbidlist, path.linkfederatelist):
            if federate is not self:# and self.time >= self.context.ofs.maxTime//2:
                federate.sharedlinks[self.time] += 1
                tempcost += cost
                federate.cash += cost
                self.cash -= cost

        # assert path.pathBid == tempcost
        # print("finished:", taskvalue)
        self.context.totalcash += taskvalue


        assert task in self.activeTasks
        section = task.getSection()
        assert self.time >= task.initTime
        duration = max(1, self.time - task.initTime)
        assert section in range(1, 7)

        # print "Finished tasks (section, taskvalue, taskduration):", section, taskvalue, duration
        self.taskduration[section] = (self.taskduration[section]*self.taskcounter[section] + duration)/(self.taskcounter[section] + 1.)
        self.taskvalue[section]  = (self.taskvalue[section]*self.taskcounter[section] + taskvalue)/(self.taskcounter[section] + 1.)
        self.taskcounter[section] += 1
        # print("finishtask: pickup opp and task counter:", sum(list(self.taskcounter.values())), self.pickupOpportunities)
        # if self.learning:
        #     print("self.rewards:", self.rewards)
        #     self.rewards += taskvalue

        self.activeTasks.remove(task)

    def defaultTask(self, task):
        # print "defaulted task:", task.taskid
        element = task.elementOwner
        element.removeSavedTask(task)
        task.pathcost = 0.
        self.finishTask(task)

    def addElement(self, element, location, capacity):
        orbit, section = (re.search(r'(\w)\w+(\d)', location).group(1), int(re.search(r'(\w)\w+(\d)', location).group(2)))
        if 'Ground' in element:
            gs = GroundStation(self, 'GS.%s.%d'%(self.name, len(self.stations)+1), location, 600)
            self.elements.append(gs)
            self.stations.append(gs)

        elif 'Sat' in element:
            ss = Satellite(federate=self, name = 'S%s.%s.%d'%(orbit, self.name, len(self.satellites)+1),
                           location=location, cost = 800 ,capacity=capacity)
            self.elements.append(ss)
            self.satellites.append(ss)

    def deliverTasks(self, context):
        for element in self.elements:
            # print "deliver task in Federate:", element
            if element.isSpace():
                # element.updateGraph(context)
                savedtasks = element.savedTasks[:]
                for task in savedtasks:
                    # print  "time and task activation time:", self.time, task.activationTime
                    assert task.activationTime >= self.time
                    if self.time >= task.activationTime:
                        element.deliverTask(task)
                        # print "len of saved tasks:", len(element.savedTasks),
                        element.removeSavedTask(task)
                        # print len(element.savedTasks)


    def getBundleBid(self, bundlelist):
        # print("Federates: bundellist:", edgebundlelist)
        alledges = [edge for bundle in bundlelist for edge in bundle.edgelist]
        assert all([re.search(r'.+\.(F\d)\..+', tup[1]).group(1) == self.name for tup in alledges])
        # edgeAskerDict = {edge: bundle.federateAsker for bundle in edgebundlelist for edge in bundle.edgelist}
        bundlecostdict = {}
        for bundle in bundlelist:
            edgeAskerDict = {}
            asker = bundle.federateAsker

            tuplecostdict = {edge: self.getCost('oISL', asker) if self.nodeElementDict[edge[1]].isSpace() else self.getCost('oSGL', asker) for edge in alledges}

            bundlecostdict[bundle] = sum([tuplecostdict[b] for b in bundle.edgelist])

        # print("federates: bundlecst:")
        #
        # for b in bundlecostdict:
        #     print(b.federateAsker.name, self.name, bundlecostdict[b])
        return bundlecostdict

    def grantBundlePrice(self, bundle):
        keybundle = UniqueBundle(bundle)
        if keybundle in self.uniqueBundles:
            ubundle = self.uniqueBundles[self.uniqueBundles.index(keybundle)]
            opportunitycost = bundle.price
            ubundle.setGenOppCost(opportunitycost)
        else:
            self.uniqueBundles.append(keybundle)




class FederateLearning(FederateLite):
    def __init__(self, name, context, costSGL, costISL, storagePenalty = -2):
        FederateLite.__init__(self, name, context, costSGL = costSGL, costISL = costISL, storagePenalty = storagePenalty)
        # print("storage penalty:", self.storagePenalty)
        self.qlearnerstorage = QlearnerStorage(self, numericactions=list(range(0, 1101, 100)), states =list(range(int(self.context.ofs.capacity)+1)))
        self.rewards = 0.
        self.storagelearning = True


    def getStorageCostList(self, element, taskvaluelist = None):
        # print("federate storage penalty:", self.storagePenalty)
        storagepanalty = self.qlearnerstorage.getAction(element)
        # print("storage penalty -2:", storagepanalty)
        return storagepanalty

    def finishTask(self, task):
        # path = task.path
        # taskvalue = task.getValue(self.time)
        # self.cash += taskvalue
        # for cost, federate in zip(path.linkbidlist, path.linkfederatelist):
        #     if federate is not self:
        #         federate.cash += cost
        #         self.cash -= cost
        #
        # # print("active tasks:", [t.taskid for t in self.activeTasks])
        # # print("finish task:", task.taskid, task.elementOwner.name, task.federateOwner.name, self.name)
        #
        # assert task in self.activeTasks
        # section = task.getSection()
        # assert self.time >= task.initTime
        # duration = max(1, self.time - task.initTime)
        # assert section in range(1, 7)
        #
        # # print "Finished tasks (section, taskvalue, taskduration):", section, taskvalue, duration
        # self.taskduration[section] = (self.taskduration[section]*self.taskcounter[section] + duration)/(self.taskcounter[section] + 1.)
        # self.taskvalue[section]  = (self.taskvalue[section]*self.taskcounter[section] + taskvalue)/(self.taskcounter[section] + 1.)
        # self.taskcounter[section] += 1
        # self.activeTasks.remove(task)
        FederateLite.finishTask(task)
        taskvalue = task.getValue(self.time)
        self.rewards += taskvalue

    def ticktock(self):
        """
        Ticks this federate in a simulation.
        @param sim: the simulator
        """
        # print "federete tick tock"
        # for element in self.elements:
        #     if element.isSpace():
        #         self.qlearnerstorage.update_q(element, self.rewards)

        self.time = self.context.time
        for element in self.elements:
            element.ticktock()
            if element.isSpace():
                # print("self. rewrds:",self.rewards)
                self.qlearnerstorage.update_q(element, self.rewards)

        self.rewards = 0.


class FederateLearning2(FederateLite):
    def __init__(self, name, context, storagePenalty = -1):
        FederateLite.__init__(self, name, context, costSGL=0, costISL=0, storagePenalty = storagePenalty)
        self.qlearnercost = QlearnerCost(self, numericactions=list(range(0, 1101, 100)))
        self.costlearning = True
        self.currentcost = None
        self.costlist = []
        self.timepricelist = []


    def getCost(self, protocol = None, federate = None):
        # print(self.name, federate.name, self.name == federate.name, self.costDic[protocol])
        if federate and self.name == federate.name:
            return 0.

        self.currentcost = 100*round(self.qlearnercost.getAction()/100.)
        self.costlist.append(self.currentcost)
        # print("Get cost:", self.currentcost)
        return self.currentcost

    def updateBestBundle(self, bestbundle):
        tasklist = bestbundle.tasklist
        reward = 0.
        for task in tasklist:
            path = task.path
            if task.federateOwner is self:
                reward += task.getValue(self.time+path.deltatime) - path.pathBid
            else:
                for bid, federate in zip(path.linkbidlist, path.linkfederatelist):
                    if federate is self:
                        reward += bid

        self.qlearnercost.update_q(self.currentcost, reward)
























