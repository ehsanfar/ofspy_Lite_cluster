import re
from collections import defaultdict, Counter
from .bundle import PathBundle, PathBundleLite
import itertools
import time
from .generalFunctions import returnCompatiblePaths

class Auction():
    def __init__(self, auctioneer, time, taskslist, links):
        self.auctioneer = auctioneer
        self.tasklist = taskslist
        self.time = time
        self.pathlist = []
        self.taskPathDict = defaultdict(list) #{t.taskid: [] for t in taskslist}
        self.taskDict = {t.taskid: t for t in taskslist}
        self.compatibleBundles = []
        self.bestPathBundle = None
        self.maxlink = links
        self.auctionFederates = set([])
        # self.federateset = []


    def inquirePrice(self, costDict):
        self.taskPathDict = defaultdict(list)
        federateBundleDict = {}
        seen = set([])
        seen_add = seen.add
        elementlist = [t.elementOwner for t in self.tasklist]
        elementset = [e for e in elementlist if e.name not in seen or seen_add(e.name)]
        federatelist = [e.federateOwner for e in elementset]
        seen = set([])
        seen_add = seen.add
        federateset = [f for f in federatelist if f.name not in seen or seen_add(f.name)]


        self.pathlist = [e.elementG.orderPathDict[self.time%6] for e in elementset]

        # auctionlinklist = [e for l in [path.nodelist for path in self.pathlist] for e in l]
        # for fed in federateset:
        #     fed.updateCost(auctionlinklist)


        self.pathlist = [e for l in self.pathlist for e in l]
        # print("pathlist:", self.pathlist)
        for path in self.pathlist:
            elementOwner = path.elementOwner
            ownername = path.elementOwner.federateOwner.name
            linkbids = []
            linkcosts = []

            for link in path.linklist:
                fname = re.search(r'.+\.(F\d)\..+', link[1]).group(1)
                if fname == ownername:
                    cost = elementOwner.elementG.Graph[link[0]][link[1]]['weight']

                    # print(elementOwner.name, link, cost)
                    linkbids.append(0)
                else:
                    self.auctionFederates.add(fname)
                    cost = costDict[fname]# if 'GS' in link[1] else costISLDict[fname]
                    linkbids.append(cost)

                    # print('link cost:', cost)

                linkcosts.append(cost)


            # if path.elementOwner.federateOwner.storagePenalty == -1:
            #     print(path.elementOwner.federateOwner.storagePenalty, linkbids, linkcosts)

            path.updateBid(linkbids)
            path.updateCost(linkcosts)
            # print("path bids vs path cost:", path.linklist, path.linkbidlist, path.linkcostlist)
            # print("all tasks:", [t.taskid for t in self.tasklist])
            for task in self.tasklist:
                if task.elementOwner.name == path.elementOwner.name:
                    # print("path cost:", path.pathCost, task.getValue(task.initTime + path.deltatime), path.pathCost < task.getValue(task.initTime + path.deltatime))
                    if path.pathCost < task.getValue(task.initTime + path.deltatime):

                        self.taskPathDict[task.taskid].append(path)

        # print("tasksPathDict length:", [len(set(v)) for v in self.taskPathDict.values()])
        # for taskid in self.taskPathDict:
        #     for path in self.taskPathDict[taskid]:
        #         print(path.linklist)



    # def findBestBundle(self, tasklist = None):
    #     # if compatiblebundles:
    #     #     possible_bundles = compatiblebundles
    #     # else:
    #     #     if self.compatibleBundles:
    #     #         possible_bundles = self.compatibleBundles
    #     #     else:
    #     #         self.findCompatiblePaths()
    #     #         possible_bundles = self.compatibleBundles
    #     #
    #     # if not possible_bundles:
    #     #     # self.bestPathBundle = None
    #     #     return False
    #     if tasklist:
    #         self.findCompatiblePaths(tasklist)
    #     else:
    #         self.findCompatiblePaths()
    #
    #     possible_bundles = self.compatibleBundles
    #     if not self.compatibleBundles:
    #         return False
    #     # print("length of compatible bundles:", len(self.compatibleBundles))
    #
    #     path_bundle_cost = [b.bundleCost for b in possible_bundles]
    #     path_bundle_revenue = [b.bundleRevenue for b in possible_bundles]
    #     path_bundle_profit = [x-y for (x,y) in zip(path_bundle_revenue, path_bundle_cost)]
    #     path_bundle_length = [b.length for b in possible_bundles]
    #     # print("pathbundle cost:", path_bundle_cost)
    #     # sortedcost = sorted(list(zip(path_bundle_cost, possible_bundles)), reverse = True)
    #     # print("sorted cost:", sortedcost)
    #     # print(sorted(path_bundle_length, reverse = True))
    #     sorted_revenue = sorted(list(zip(path_bundle_profit, possible_bundles)), reverse = True)
    #     # print("sorted revenue:", [(x, [p.nodelist for p in y.pathlist]) for x,y in sorted_revenue])
    #     self.bestPathBundle = sorted_revenue[0][1]
    #     # print("best path bundle:", self.bestPathBundle)
    #     for path in self.bestPathBundle.pathlist:
    #         # print("path noelsit:", path.nodelist)
    #         path(next((task for task in self.tasklist if task.elementOwner.name == path.elementOwner.name)))
    #         path.task.updatePath(path)
    #         # path.updateTime()
    #         timelink = list(zip([path.task.initTime + dt for dt in path.deltatimelist], path.linklist))
    #         for time, link in timelink:
    #             # print(time, link)
    #             # print(defaultdict)
    #             self.auctioneer.updateTimeLinks(time, link)
    #
    #     # federateDict = {f.name: f for f in self.auctioneer.context.federates}
    #     # for fed in self.auctionFederates:
    #     #     federate = federateDict[fed]
    #     #     if federate.costlearning:
    #     #        federate.updateBestBundle(self.bestPathBundle)
    #
    #     return True

    def findBestBundleinAuction(self, tasklist = None):
        # if compatiblebundles:
        #     possible_bundles = compatiblebundles
        # else:
        #     if self.compatibleBundles:
        #         possible_bundles = self.compatibleBundles
        #     else:
        #         self.findCompatiblePaths()
        #         possible_bundles = self.compatibleBundles
        #
        # if not possible_bundles:
        #     # self.bestPathBundle = None
        #     return False
        if tasklist:
            self.findCompatiblePaths(tasklist)
        else:
            self.findCompatiblePaths()

        possible_bundles = self.compatibleBundles
        if not self.compatibleBundles:
            return False
        # print("length of compatible bundles:", len(self.compatibleBundles))

        path_bundle_cost = [b.bundleCost for b in possible_bundles]
        path_bundle_revenue = [b.bundleRevenue for b in possible_bundles]
        path_bundle_profit = [x-y for (x,y) in zip(path_bundle_revenue, path_bundle_cost)]
        path_bundle_length = [b.length for b in possible_bundles]
        # print("pathbundle cost:", path_bundle_cost)
        # sortedcost = sorted(list(zip(path_bundle_cost, possible_bundles)), reverse = True)
        # print("sorted cost:", sortedcost)
        # print(sorted(path_bundle_length, reverse = True))
        sorted_revenue = sorted(list(zip(path_bundle_profit, possible_bundles)), reverse = True)
        maxprofit_org = sorted_revenue[0][0]
        possible_bundles = [tup for tup in sorted_revenue if tup[0] == maxprofit_org]
        self.bestPathBundle = self.auctioneer.context.masterStream.choice(possible_bundles)[1]
        '''
        for federate in self.auctioneer.context.federates:
            cost_0 = [b.getBundleAlternativeCost(federate, price = 0) for b in possible_bundles]
            cost_inf = [b.getBundleAlternativeCost(federate, price= 1100) for b in possible_bundles]
            # if path_bundle_cost[0] == 540:
            #     print("bid cost:", path_bundle_cost)
            #     print("zero cos:", cost_0)
            #     print("max cost:", cost_inf)
            profit_0 = [x-y for (x,y) in zip(path_bundle_revenue, cost_0)]
            profit_inf = [x-y for (x,y) in zip(path_bundle_revenue, cost_inf)]
            sorted_0 = sorted(list(zip(profit_0, possible_bundles)), reverse=True)
            sorted_inf = sorted(list(zip(profit_inf, possible_bundles)), reverse=True)
            best_0 = sorted_0[0][1]
            best_inf = sorted_inf[0][1]
            maxprofit_0 = sorted_0[0][0]
            maxprofit_inf = sorted_inf[0][0]
            if len(set([maxprofit_0, maxprofit_org, maxprofit_inf]))>1:
                print("Bundle profit inf - profit 0: ", federate.name, maxprofit_0, int(maxprofit_org), maxprofit_inf)
                # print('')
        # print("sorted revenue:", [(x, [p.nodelist for p in y.pathlist]) for x,y in sorted_revenue])
        '''

        # print("best path bundle:", self.bestPathBundle)
        for path in self.bestPathBundle.pathlist:
            # print("path noelsit:", path.nodelist)
            path(next((task for task in self.tasklist if task.elementOwner.name == path.elementOwner.name)))
            path.task.updatePath(path)
            # path.updateTime()
            timelink = list(zip([path.task.initTime + dt for dt in path.deltatimelist], path.linklist))
            for time, link in timelink:
                # print(time, link)
                # print(defaultdict)
                self.auctioneer.updateTimeLinks(time, link)

        # federateDict = {f.name: f for f in self.auctioneer.context.federates}
        # for fed in self.auctionFederates:
        #     federate = federateDict[fed]
        #     if federate.costlearning:
        #        federate.updateBestBundle(self.bestPathBundle)

        return True

    def findCompatiblePaths(self, tasklist = None):
        # # print("Compatible paths: tasks:", [(t, len(p)) for t, p in self.taskPathDict.items()])
        taskPathDict = {t: self.taskPathDict[t] for t in tasklist} if tasklist else self.taskPathDict
        taskspaths = list(taskPathDict.items())
        # print("tasks paths:", taskPathDict)
        # # print("Update compatible bundles: all paths:", all_paths)
        # # print("All paths:", self.pathdict)
        # print("length of all paths:", [len(e[1]) for e in all_paths])
        # taskspaths = [(t, p) for t, p in all_paths]
        # taskpathgenerator = self.uniquePermutations(pathindex)
        # print("length of all products",len(list(taskpathgenerator)))
        # # print("find compatible, length of products:", len(taskpathgenerator))
        # possible_bundles = []
        # j = 0
        # for taskids, paths in taskpathgenerator:
        #     j += 1
        #     # print("tasks id:", taskids, len(paths))
        #     checktime1 = time.time()
        #     check = self.checkPathCombinations(paths)
        #     checktime2 = time.time()
        #     if check:
        #         possible_bundles.append(PathBundle(tuple([self.taskDict[id] for id in taskids]), paths))
        #     checktime3 = time.time()
        #     print("time differences: 1 , 2 :", checktime2 - checktime1, checktime3 - checktime2, j)
        # # possible_bundles = [PathBundle([self.taskDict[id] for id in taskids], paths) for taskids, paths in taskpathgenerator if self.checkPathCombinations(paths)]
        # # print
        # # print("Auctioneer: possible path combinations:", [p.pathlist for p in possible_bundles])
        # print("Length of possible bundles:", len(possible_bundles))
        # # print([t.length for t in possible_bundles])
        self.compatibleBundles = list(self.returnFeasibleBundles(taskspaths))
        # print(self.compatibleBundles)
        # return possible_bundles

    # def checkPathCombinations(self, plist):
    #     self.auctioneer.timeOccupiedLinkDict = {t: v for t, v in self.auctioneer.timeOccupiedLinkDict.items() if t>=self.time}
    #     # print("time:", self.time)
    #     # print("Check compatible: ", self.auctioneer.timeOccupiedLinkDict.keys())
    #     alledges = set([a for alllinks in self.auctioneer.timeOccupiedLinkDict.values() for a in alllinks])
    #     # print("all edges:", alledges)
    #     # print("check path combination:", [p.nodelist for p in pathlist])
    #     for path in plist:
    #         newlinks = set(path.linklist)
    #         # print("new edges:", newedges)
    #         intersection = alledges.intersection(newlinks)
    #         # print("intersection:", intersection)
    #         if intersection:
    #             # print(False)
    #             return False
    #
    #         alledges = alledges.union(newlinks)
    #     # print(True)
    #     return True
    #
    # def uniquePermutations(self, taskpathindices):
    #     # print("task paths size:", [len(tp[1]) for tp in taskpathindices])
    #     # print("taskpathlist:", taskpathlist)
    #     # print("uniquePermulations:", [[p.nodelist for p in pathlist] for pathlist in taskpathlist])
    #     taskpathlist = [tp for tp in taskpathindices if tp[1]]
    #     ntasks = len(taskpathlist)
    #     permutations = []
    #     combinations =  []
    #     for n in range(1,ntasks+1):
    #         tempcombinations = itertools.combinations(range(ntasks), n)
    #         combinations += list(tempcombinations)
    #
    #     for c in combinations:
    #         tasks = [taskpathindices[i][0] for i in c]
    #         pathindeces = [taskpathindices[i][1] for i in c]
    #         # print("newlist:", newlist)
    #         indexproducts = itertools.product(*pathindeces)
    #         # print("Permutations:", [p.nodelist for p in list(tempproducts)])
    #         for indexes in indexproducts:
    #             yield tasks, indexes
    #         # permutations.append((tasks, indexproducts))

    def returnFeasibleBundles(self, taskspaths):
        tlist = [tp[0] for tp in taskspaths]
        plist = [tp[1] for tp in taskspaths]
        ntasks = len(taskspaths)
        combinations =  []

        time = self.auctioneer.context.time
        timelink = {t: v for t, v in self.auctioneer.timeOccupiedLinkDict.items() if t >= time}
        links = list([e[0] for e in timelink.values()])
        # print
        linkcounter = defaultdict(int, Counter(links))
        # print("link counter:", time, linkcounter, set(list(linkcounter.values())))

        for n in range(1,ntasks+1):
            tempcombinations = itertools.combinations(range(ntasks), n)
            combinations += list(tempcombinations)

        for c in combinations:
            # print(c)
            taskcomblist = [tlist[i] for i in c]
            # print("new bundle tasks:", taskcomblist)
            pathcomblist = [plist[i] for i in c]
            # print("path combination list:", pathcomblist)
            # linkset = set([])
            for comb in returnCompatiblePaths(pathcomblist, linkcounter, maxlink=self.maxlink):
                # print(comb)
                bundle = PathBundle(tuple([self.taskDict[id] for id in taskcomblist]), comb)
                # print("bundle length:", c, len(comb), bundle.length)
                yield bundle

        # return permutations
        # print("Unique products:", )

        #     tempdict = self.updatePathFederateBundleDict(path)
        #     # print("Auctioneer: path edge dict:", [(a, bundle.edgelist) for (a,bundle) in tempdict.items()])
        #     federateBundleDict = self.setDict2Dict(federateBundleDict, tempdict)
        #
        # # print("auctioneer: federateBundleDict:", federateBundleDict)
        # # print("Auctioneer: federate and bundles:", [(f, [b.edgelist for b in bundles]) for (f,bundles) in federateBundleDict.items()])
        # self.bundleBidDict = {}
        # for fed, bundleset in federateBundleDict.items():
        #     bundlelist = list(bundleset)
        #     # print("Federate:", fed)
        #     # print("bundle list:", edgebundlelist)
        #     # print("Inquireprice: bundle list federates:", [[(self.nodeFederateDict[x].name, self.nodeFederateDict[y].name) for (x,y) in bundle.edgelist] for bundle in edgebundlelist])
        #     # print("Auctioneer: fed and bundleset", fed, [b.edgelist for b in edgebundlelist])
        #     tempdict = self.namefederatedict[fed].getBundleBid(bundlelist)
        #     # print("Auctioneer: asker federate protocol cost:", [(b.federateAsker.name, fed, c) for (b,c) in tempdict.items()])
        #
        #     for b in tempdict:
        #         assert b not in self.bundleBidDict
        #         self.bundleBidDict[b] = tempdict[b]
        #
        #     # bundleBidDict = {x: y for x,y in zip(edgebundlelist, costlist)}
        # self.updateBundleBid()
        # self.updateBundles()
        # self.updateCompatibleBundles()