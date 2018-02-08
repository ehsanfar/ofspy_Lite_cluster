
"""
Elements classes.
"""
import re
try:
    import Queue as queue
except ImportError:
    import queue

from .task import Task
from .generalFunctions import *
# import random

class Element():
    def __init__(self, federate, name, location, cost=0):
        self.name = name
        self.location = re.search('(\w+)(\d)', location).group(1)
        self.section = int(re.search('(\w+)(\d)', location).group(2))
        self.designCost = cost
        self.federateOwner = federate
        self.savedTasks = []
        self.pickupprobablity = self.federateOwner.pickupProbability
        self.elementG = None

    def getOwner(self):
        return self.federateOwner

    def getLocation(self):
        return str(self.location)+str(self.section)

    def getSection(self):
        return self.section

    def getSectionAt(self, time):
        if 'LEO' in self.location:
            return (time*2+self.section-1)%6+1

        elif 'MEO' in self.location:
            return (time+self.section-1)%6+1

        else:
            return self.section

    def ticktock(self):
        if self.isSpace():
            if 'LEO' in self.location:
                self.section = (self.section + 2 - 1)%6 + 1
            elif 'MEO' in self.location:
                self.section = (self.section + 1 - 1)%6 + 1


            self.timeStateDict[self.federateOwner.time] = (self.capacity - self.content, self.section)
            # print("element time state dict:", self.name, self.federateOwner.time, self.timeStateDict[self.federateOwner.time])




    def getDesignCost(self):
        return self.designCost


    def canSave(self, task):
        if self.isGround():
            return True

        # print "capacity and content:", self.capacity, self.content
        if task.datasize <= (self.capacity - self.content):
            return True

        return False

    def canTransmit(self, rxElement, task):
        return rxElement.couldReceive(self, task)

    def transmitTask(self, task, pathiter):
        # print self.name, task.taskid
        if self.isGround():
            self.saveTask(task)
            return True

        # assert len(pathlist)>=1
        # if len(pathlist)<2:
        #     federate = task.federateOwner
        #     federate.discardTask(self, task)
        task.nextstop = nextstop = next(pathiter)

        received = nextstop.transmitTask(task, pathiter)
        return received

    def isGEO(self):
        if self.isSpace() and 'GEO' in self.location:
            return True

        return False




class GroundStation(Element):
    def __init__(self, federate, name, location, cost):
        Element.__init__(self, federate, name, location, cost)

    def isGround(self):
        return True

    def isSpace(self):
        return False

    def saveTask(self, task, nextstop = None):
        self.savedTasks.append(task)
        task.federateOwner.finishTask(task)
        return True




class Satellite(Element):
    def __init__(self, federate, name, location, cost, capacity=2.):
        Element.__init__(self, federate, name, location, cost)
        self.capacity = capacity
        self.content = 0.
        self.queuedTasks = queue.Queue()
        self.elementGraph = None
        self.timeStateDict = {}

    def getCapacity(self):
        return self.capacity

    def getContentsSize(self):
        return self.content

    def isGround(self):
        return False

    def isSpace(self):
        return True

    def deliverTask(self, task):
        self.transmitTask(task, iter(task.path.elementlist[1:]))

    def saveTask(self, task, deltatime):
        if self.canSave(task):
            task.updateActivationTime(task.initTime + deltatime)
            self.savedTasks.append(task)
            self.content += task.datasize
            return True

        # task.federateOwner.discardTask(self, task)
        return False

    def removeSavedTask(self, task):
        assert task in self.savedTasks
        self.content -= task.datasize
        self.savedTasks.remove(task)

    def updateGraph(self, context):
        self.elementGraph.graphList = context.Graph.getGraphList()
        self.elementGraph.graphOrder = context.Graph.getGraphOrder()
        self.elementGraph.elementOwners = context.Graph.getElementOwners()
        self.elementGraph.createGraph()

    def collectTasks(self, context):
        if not self.isGEO():
            # print("Element collect new tasks")
            # print("new task:", self.federateOwner.name, self.name)
            if not context.canpickup():
                # print("Full")
                return False

            nextTask = Task(time=self.federateOwner.time, id=context.getTaskid(), federate=self.federateOwner, element=self)
            # print("new task id:", nextTask.taskid)

            taskvaluelist = [nextTask.getValue(self.federateOwner.time + i, inittime=self.federateOwner.time) for i in
                             range(6)]
            # print("Element tasks value list:", taskvaluelist)
            self.elementG.updateGraph(context, taskvaluelist=taskvaluelist)
            #
            # edges = self.elementG.Graph.edges()


            self.federateOwner.pickupOpportunities += 1
            if self.canSave(nextTask):

                # print("Element Return task")
                return nextTask
        return False

    def pickupTask(self, task):
        # print("pickcup tasks:", self.federateOwner.name, self.name)
        staticpath, deltatime2 = convertPath2StaticPath(task.path)
        # print(task.path.nodelist)
        self.saveTask(task, deltatime2)
        # print("pickup task id:", task.taskid)
        self.federateOwner.reportPickup(task)



        # # print "elementLite - taskid:", self.name, taskid, self.section
        # if not self.isGEO():# or random.random()<self.pickupprobablity:
        #     # print "it is satellite"
        #     # print "current tasks:", currentTasks
        #     # print self.section
        #     # print currentTasks[self.section].qsize()
        #     # assert not currentTasks[self.section].empty()
        #     self.updateGraph(context)
        #
        #     # currentTasks = context.currentTasks
        #     nextTask = Task(time = self.federateOwner.time, id = context.taskid, federate = self.federateOwner, element = self)
        #     # tempqueue = currentTasks[self.section].queue
        #     # temptask = tempqueue[0]
        #     taskvaluelist = [nextTask.getValue(self.federateOwner.time + i, inittime=self.federateOwner.time) for i in range(6)]
        #     # print("pickup : task value list:", taskvaluelist)
        #     self.elementGraph.updateSuperGraph(taskvaluelist= taskvaluelist)
        #     if not self.canSave(nextTask):
        #         # print "cannot save"
        #         return False
        #
        #
        #     #
        #     # deltatime1 = nextTask.duration
        #     # # print("pickup: graph order and time:", context.time, self.federateOwner.time)
        #     # pathcost, pathname = self.elementGraph.findcheapestpath(deltatime= deltatime1)
        #     # staticpath, deltatime2 = convertPath2StaticPath(pathname)
        #     # self.federateOwner.pickupOpportunities += 1
        #     #
        #     # prospectiveValue = nextTask.getValue(self.federateOwner.time + deltatime2, inittime= self.federateOwner.time)
        #     # if pathcost >= prospectiveValue:
        #     #     # print("Pickup: pathcost vs task value:", pathcost, prospectiveValue)
        #     #     return False
        #     #
        #     # elementpath = [next((e for e in context.elements if e.name == p)) for p in staticpath]
        #     # context.taskid += 1
        #     # nextTask.setSection(self.section)
        #     # nextTask.updatePath(elementpath, pathcost)
        #     # # print "element next task inittime:", self.name, taskid, nextTask.initTime
        #     # self.saveTask(nextTask, deltatime2)
        #     # self.federateOwner.reportPickup(nextTask)
        #     return True
        #
        # return False






