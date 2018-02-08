import random
import numpy as np
from .task import Task
# import queue
try:
    import Queue as queue
except ImportError:
    import queue
import time

from .federateLite import FederateLite, FederateLearning, FederateLearning2
import re

from .graph import SuperG
from .auctioneer import Auctioneer

class ContextLite():
    def __init__(self):
        """
        @param locations: the locations in this context
        @type locations: L{list}
        @param events: the events in this context
        @type events: L{list}
        @param federations: the federations in this context
        @type federations: L{list}
        @param seed: the seed for stochastic events
        @type seed: L{int}
        """
        self.initTime = 0
        self.maxTime = 0
        self.time = 0
        self.federates = []
        self.elements = []
        self.masterfederate = []
        self.currentTasks = {i: queue.Queue(maxsize = 3) for i in range(1,7)}
        self.nodeLocations = []
        self.shortestPathes = []
        self.G = None
        self.taskid = 0
        self.pickupProbability = 1.
        self.auctioneer = None
        self.nodeElementDict = {}
        self.nodeFederateDict = {}
        self.pickeduptasks = set([])
        self.taskperturn = 1000
        self.totalcash = 0.

    def init(self, ofs):
        self.ofs = ofs
        self.time = ofs.initTime
        self.initTime = ofs.initTime
        self.maxTime = ofs.maxTime
        self.links = ofs.links

        self.masterStream = random.Random(ofs.seed)
        self.shuffleStream = random.Random(self.masterStream.random())
        self.orderStream = random.Random(self.masterStream.random())


        self.generateFederates(ofs)
        # self.generateTasks()

        self.G = SuperG(self)
        self.G.createGraph(self)

        # for e in [element for element in self.elementlist if element.isSpace()]:
        #     elementGraph = e.elementGraph
            # for key, pathlist in elementGraph.orderPathDict.items():
            #     print(key, [p.nodelist for p in pathlist])
            # print(elementGraph.orderPathDict)

        self.auctioneer = Auctioneer(self) #nodefederatedict=nodefederatedict, nodeelementdict=nodeelementdict)

    def getTaskid(self):
        self.taskid += 1
        return self.taskid - 1
    
    def getElementOwner(self, element):
        return next((federate for federate in self.federates
                     if element in federate.elements), None)

    def getTaskOwner(self, task):
        return task.federateOwner

    def findTask(self, task):

        return next((element for federate in self.federates
                     for element in federate.elements
                     if task in element.savedTasks), None)

    def propagate(self, scheme ='federated'):
        """
        Executes operational models.
        """
        self.time += 1
        federates = self.federates
        random.shuffle(federates, random=self.orderStream.random)
        for federate in federates:
            # print "Pre federate operation cash:", federate.cash
            federate.ticktock()
            # print "Post federate operation cash:", federate.cash



    # def updatePickupProbablity(self):
    #     opportunitycounter = sum([f.pickupOpportunities for f in self.federates])
    #
    #     taskcounter = sum([sum(list(f.taskcounter.values())) for f in self.federates])
    #     self.pickupProbability = taskcounter/float(opportunitycounter)
    #     for f in self.federates:
    #         f.pickupProbability = self.pickupProbability

    def ticktock(self, ofs):
        """
        Tocks this context in a simulation.
        """
        # self.time = ofs.time
        self.auctioneer.initiateAuction()
            # self.pickupTasks()
            # self.Graph.drawGraph(self)
            # print [e.queuedTasks.qsize() for e in self.elementlist if e.isSpace()]
            # print [len(e.savedTasks) for e in self.elementlist if e.isSpace()]
            # print "Graphorder:", [e.Graph.graphOrder for e in self.elementlist if e.isSpace()], self.Graph.graphOrder
        self.deliverTasks()
        # self.updatePickupProbablity()
        self.propagate()
        # print("context finished task counter:", len(self.pickeduptasks))

        # print "Context - Assigned Tasks:", self.taskid
        # print self.time, [a.getLocation() for a in self.elementlist]

    # def generateTasks(self, N=6):
    #     # tasklocations = np.random.choice(range(1,7), N)
    #     for l in self.currentTasks:
    #         if self.currentTasks[l].full():
    #             self.currentTasks[l].get()
    #
    #         while not self.currentTasks[l].full():
    #             self.currentTasks[l].put(Task(self.time, id = self.taskid))
    #             self.taskid += 1
    #
    #     # print "current tasks size:", [c.qsize() for c in self.currentTasks.values()]

    def generateFederates(self, ofs):
        # elist = elementlist.split(' ')
        elements = ofs.elements
        costSGL = ofs.costSGL
        costISL = ofs.costISL
        storagePenalty = ofs.storagePenalty
        elementgroups = []
        for e in elements:
            elementgroups.append(re.search(r'\b(\d+)\.(\w+)@(\w+\d).*\b', e).groups())
        fedset = sorted(list(set([e[0] for e in elementgroups])))
        # print elementgroups
        # print fedset

        # print("generate federates: storage penalty and federates:", storagePenalty, fedset)
        for i, f in enumerate(fedset):
            if costSGL[i] == -2:
                self.federates.append(FederateLearning2(name='F' + str(i + 1), context=self))
            elif storagePenalty[i] == -2:
                self.federates.append(FederateLearning(name='F'+str(i+1), context=self, costSGL=costSGL[i], costISL=costISL[i]))
            else:
                self.federates.append(FederateLite(name = 'F'+str(i+1), context = self, costSGL = costSGL[i], costISL = costISL[i], storagePenalty = storagePenalty[i]))
                if costSGL[i] == -3:
                    self.federates[-1].stochastic = True

        for element in elementgroups:
            # print(element)
            index = fedset.index(element[0])
            self.federates[index].addElement(element=element[1], location=element[2], capacity = ofs.capacity)

        for f in self.federates:
            self.elements += f.getElements()[:]

        self.nodeElementDict = {e.name: e for e in self.elements}
        self.nodeFederateDict = {e.name: e.federateOwner for e in self.elements}

    def deliverTasks(self):
        # print "delivering tasks"
        # # Graph = self.Graph.getGraph()
        # graphorder = self.Graph.graphOrder
        for federate in self.federates:
            federate.deliverTasks(self)

    def addPickup(self, tasksid):
        assert tasksid not in self.pickeduptasks
        self.pickeduptasks.add(tasksid)

    def canpickup(self):
        if len(self.pickeduptasks)>self.taskperturn*self.time:
            return False
        return True














