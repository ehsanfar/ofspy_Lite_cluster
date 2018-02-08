from .generalFunctions import *

class Path():
    def __init__(self, element, nodelist):
        self.nodelist = nodelist
        self.linklist = convertPath2Edge(nodelist)
        self.elementOwner = element
        self.elementlist = [element.federateOwner.context.nodeElementDict[n[:-2]] for n in nodelist]
        self.nodeElementDict = {n:e for n,e in zip(self.nodelist, self.elementlist)}
        self.linkfederatelist = [self.nodeElementDict[tup[1]].federateOwner for tup in self.linklist]
        self.deltatimelist = self.updateDeltaTime()
        self.deltatime = max(self.deltatimelist)
        # self.timelist = []
        # self.federates = []
        # self.federateCost = {}
        # self.federatePrice = {}
        # self.federateBundleDict = {}
        # self.edgebundles = []
        self.task = None
        self.linkcostlist = []
        self.linkbidlist = []
        # self.linkpricelist = []
        self.pathCost = None
        self.pathBid = None
        # self.pathPrice = None

    def __call__(self, task):
        self.task = task

    def updateDeltaTime(self):
        orderlist = [int(n[-1]) for n in self.nodelist]
        time = 0 #self.elementOwner.federateOwner.context.time
        # print(orderlist, time)
        # assert orderlist[0] == time%6
        timelist = [time]
        for i, o in enumerate(orderlist):
            if i > 0:
                timelist.append(timelist[-1] + 1 if o != orderlist[i - 1] else timelist[-1])

        # print(timelist)
        # print(orderlist)
        # assert all([o == b%6 for o,b in zip(orderlist, timelist)])
        return timelist

    # def updateValues(self):
    #     bid_list = [b.bid for b in self.edgebundles]
    #     price_list = [b.price for b in self.edgebundles]
    #     # print("Path: nodelist and costlist:", self.nodelist, bid_list)
    #     if all(isinstance(c, float) or isinstance(c, int) for c in bid_list):
    #         self.pathBid = sum(bid_list)
    #         self.pathPrice = sum(price_list)
    #     else:
    #         self.pathBid = None
    #         self.pathPrice = None
    #     # print("update pathBid:", self.nodelist, self.pathBid)
    def updateBid(self, linkbids):
        self.linkbidlist = linkbids
        self.pathBid = sum(linkbids)
        # self.pathBid2 = sum(linkbids2)

    def updateCost(self, linkcosts):
        self.linkcostlist = linkcosts
        self.pathCost = sum(linkcosts)
        # self.pathPrice = price

    def updateWithFederateBid(self, fedbiddict):
        newlinkbidlist = []
        # print(fedbiddict)
        for linkcost, fed in zip(self.linkcostlist, self.linkfederatelist):
            if fed is self.elementOwner.federateOwner:
                newlinkbidlist.append(0.)
                # newlinkcostlist.append(linkcost)
            else:
                newlinkbidlist.append(fedbiddict[fed.name])

        self.linkbidlist = newlinkbidlist[:]
        self.pathBid = sum(newlinkbidlist)

    def updateBundles(self, federatebundledict):
        self.federateBundleDict = federatebundledict
        self.edgebundles = list(federatebundledict.values())

    # def getPathBid(self):
    #     if self.pathBid is None:
    #         self.updateValues()
    #     return self.pathBid

    # def getPathPrice(self):
    #     return self.pathPrice

    def getNodeList(self):
        return self.nodelist

    def getFederateBundle(self):
        return self.federateEdge

    def __eq__(self, other):
        return tuple(self.nodelist) == tuple(other.nodelist)

    def __lt__(self, other):
        if len(self.nodelist) != len(other.nodelist):
            return len(self.nodelist) < len(other.nodelist)
        else:
            return self.nodelist < other.nodelist

    def __hash__(self):
        return hash(tuple(self.nodelist))


