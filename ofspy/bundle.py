from .generalFunctions import checkEqual2

#
# class UniqueBundle():
#     def __init__(self, bundleobj):
#         self.edgelist = bundleobj.edgelist
#         self.federateOwner = bundleobj.federateOwner
#         self.opportunityCost = bundleobj.opportunityCost
#         self.counter = 1
#
#     def updateBundleOppCost(self, cost):
#         self.opportunityCost = (cost + self.opportunityCost*(self.counter))/(self.counter + 1.)
#         self.counter += 1
#
#     def __eq__(self, other):
#         return (self.edgelist == other.edgelist)
#
#     def __hash__(self):
#         return hash(self.edgelist)
#
# class EdgeBundle():
#     def __init__(self, edgeilst, path, federate):
#         self.edgelist = tuple(sorted(edgeilst))
#         self.pathAsker = path
#         self.taskAsker = path.task
#         self.federateAsker = self.taskAsker.federateOwner
#         self.federateOwner = federate
#         self.generalOpportunityCost = 0
#         self.localOpportunityCost = 0
#         self.fedOppCostDict = {}
#         self.bid = None
#         self.price = None
#
#     def updateBid(self, cost):
#         self.price = self.bid = cost
#
#     def updatePrice(self, price):
#         self.price = price
#
#     def getBundlePrice(self):
#         return self.price
#
#     def setGenOppCost(self, cost):
#         self.generalOpportunityCost = cost
#
#     def getGeneralOppCost(self):
#         return self.generalOpportunityCost
#
#     def setLocalOppCost(self, cost):
#         self.localOpportunityCost = cost
#
#     def getLocalOppCost(self):
#         return self.localOpportunityCost
#
#     def getBid(self):
#         return self.bid
#
#     def __eq__(self, other):
#         return (self.edgelist == other.edgelist) and (self.taskAsker.taskid is other.taskAsker.taskid)
#
#     def __lt__(self, other):
#         if len(self.edgelist) != len(other.edgelist):
#             return len(self.edgelist) > len(other.edgelist)
#         else:
#             return self.edgelist < other.edgelist
#
#
#     def __hash__(self):
#         temp = list(self.edgelist)+ [self.taskAsker.taskid, self.federateAsker.name]
#         # print("hash:", temp)
#         return hash(tuple(temp))
#

class PathBundle():
    def __init__(self, tasklist, pathlist):
        self.tasklist = tuple(tasklist)
        self.pathlist = tuple(pathlist)
        self.length = len(pathlist)
        if tasklist:
            self.updateTime()
            self.bundleBid = sum([path.pathBid for path in self.pathlist])
            self.taskvalues = [task.getValue(self.time + path.deltatime) for task, path in zip(self.tasklist, self.pathlist)]
            self.bundleRevenue = sum([task.getValue(self.time + path.deltatime)
                                      for task, path in zip(self.tasklist, self.pathlist)])
            self.bundleCost = sum([path.pathCost for path in self.pathlist])
        # self.updateTime()
        # print("Path bundle time:", self.time)
        # self.tasklist = [path.task for path in pathlist]
        # self.taskProfit = {path.task.taskid: path.task.getValue(self.time) - path.pathBid for path in pathlist}

        # self.updateTime()
        # self.updateValues()
        # self.updateRevenue()
    #
    def updateTime(self):
        tlist = [t.initTime for t in self.tasklist]
        assert checkEqual2(tlist)
        self.time = tlist[0]

    def updateValues(self):
        # print([p.nodelist for p in self.pathlist])
        # print([path.pathBid for path in self.pathlist])
        self.bundleBid = sum([path.pathBid for path in self.pathlist])
        self.bundleCost = sum([path.pathCost for path in self.pathlist])
        self.tasklist = [path.task for path in self.pathlist]
        self.taskProfit = {path.task.taskid: path.task.getValue(self.time) - path.pathCost for path in self.pathlist}

    def updateRevenue(self):
        self.bundleRevenue = sum([path.task.getValue(self.time) for path in self.pathlist])

    def getBundleRevenue(self):
        return self.bundleRevenue# - self.bundleCost

    # def getTaskProfit(self, task):
    #     # print("Path list and task dict:", [p.nodelist for p in self.pathlist], self.taskProfit)
    #     return self.taskProfit[task.taskid] if task.taskid in self.taskProfit else 0.
    '''
    def getBundleAlternativeCost(self, federate, price):
        alternativecost = 0.
        # print("bundle bid:", self.bundleBid, self.bundleCost)
        for path in self.pathlist:
            # print("federates:", path.elementOwner.federateOwner.name, federate.name)
            if path.elementOwner.federateOwner is federate:
                # print("continue")
                alternativecost += path.pathCost
                continue

            linkfederatelist = path.linkfederatelist[:]
            linkcostlist = path.linkcostlist[:]
            costlist = [bid for bid, fed in zip(linkcostlist, linkfederatelist) if fed is not federate]
            newpricelinks = [price for fed in linkfederatelist if fed is federate]
            # print([e.name for e in linkfederatelist])
            # print(len(newpricelinks), len(bidlist), len(linkbidlist))
            # print(price, sum(bidlist), sum(newpricelinks))
            alternativecost += sum(costlist) + sum(newpricelinks)
        # if alternativecost != self.bundleCost:
        #     print("Bid vs new Bid:", len(self.pathlist), price, self.bundleCost, alternativecost)
        return alternativecost
    '''


    def getTaskList(self):
        return self.tasklist

    def __eq__(self, other):
        return self.pathlist == other.pathlist

    def __lt__(self, other):
         if len(self.pathlist) != len(other.pathlist):
             return len(self.pathlist) < len(other.pathlist)
         else:
             return self.pathlist < other.pathlist

    def __hash__(self):
        return hash(tuple(len(self.pathlist), self.pathlist))



class PathBundleLite():
    def __init__(self, tasklist, pathlist):
        # self.tasklist = tuple(tasklist)
        self.bundleBid = sum([path.pathBid for path in pathlist])
        self.bundleRevenue = sum([task.getValue(task.initTime + path.deltatime) for task, path in zip(tasklist, pathlist)])
        self.bundleCost = sum([path.pathCost for path in pathlist])


