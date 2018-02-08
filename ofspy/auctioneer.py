from .generalFunctions import convertPath2Edge, returnAvgPathCost, combineBundles, optimizeCost
from .path import Path
from .bundle import PathBundle, PathBundleLite
import itertools
from .auction import Auction
from collections import defaultdict, Counter


class Auctioneer():
    def __init__(self, context):
        self.context = context
        self.usededges = set([])
        self.auctions = []
        self.currenttasks = []
        self.timeOccupiedLinkDict = defaultdict(list)
        self.links = context.links
        # self.federates = context.nodeFederateDict.values()
        # self.namefederatedict = {f.name: f for f in self.federates}
        # # print("Auctioneer nodes:",nodes)
        # # print("Auctioneer federates:", self.namefederatedict)
        # self.nodeFederateDict = context.nodeElementDict
        # self.nodeElementDict = context.nodeFederateDict
        # self.nodes = [e.name ]
        #
        # # print(self.federateDict)
        # self.pathdict = {}
        # self.pathlist = []
        # self.edgebundlelist = []
        # self.federateEdgeBundles = {}
        # self.tasks = []
        # self.compatibleBundles = None
        # self.bundleBidDict = {}

    def updateTimeLinks(self, time, link):
        if time not in self.timeOccupiedLinkDict:
            self.timeOccupiedLinkDict[time] = []
        self.timeOccupiedLinkDict[time].append(link)
        for t in list(self.timeOccupiedLinkDict.keys()):
            if t<self.context.time:
                self.timeOccupiedLinkDict.pop(t)


    def reset(self):
        self.pathdict = {}

    # def addPath(self, task, nodelist):
    #     taskid = task.taskid
    #     self.tasks.append(task)
    #     # obj = None
    #     # print("Task nodelist:", taskid.taskid, nodelist)
    #     obj = Path(task, nodelist)
    #     if taskid in self.pathdict:
    #         self.pathdict[taskid].append(obj)
    #     else:
    #         self.pathdict[taskid] = [obj]
    #
    #     self.pathlist.append(obj)

    # def runAlternative(self, auction, fedcostdict):
    #     pathlist = [path for pathlist in auction.taskPathDict.values() for path in pathlist]
    #     for path in pathlist:
    #         path.updateWithFederateCost(fedcostdict)
    #
    #     taskcostlist = returnAvgPathCost(auction.taskPathDict)
    #     bundles = []
    #     for _, taskid in taskcostlist:
    #         if auction.findBestBundleinAuction([taskid]):
    #             newbundle = auction.bestPathBundle
    #             bundles.append(newbundle)
    #
    #     alltasks, allpaths = combineBundles(bundles)
    #     auction.bestPathBundle = PathBundle(alltasks, allpaths)
    #     return auction.bestPathBundle
    def shuffleTasks(self, sortedcosttasks):
        costlist = [s[0] for s in sortedcosttasks]
        i = 0
        tasklist = [s[1] for s in sortedcosttasks]
        listoflist = []
        while i < len(costlist):
            templist = []
            while True:
                templist.append(i)
                i += 1
                if i >= len(costlist) or costlist[i] != costlist[i-1]:
                    break
            self.context.shuffleStream.shuffle(templist)
            # random.Random(0).shuffle(templist)
            listoflist.append([tasklist[j] for j in templist])

        newtasklist = [t for l in listoflist for t in l]
        return newtasklist

    def runAuction(self, tasklist):
        # print(tasklist)
        auction =  Auction(self, self.context.time, tasklist, self.links)
        initCostDict = {f.name: f.getCost('oSGL') for f in self.context.federates}

        costDictList = []
        costTupleList = []
        casenameList = []
        # for fed in initCostDict:
        #     for cost in [0, 1100]:
        #         tempdict = dict(initCostDict)
        #         tempdict[fed] = cost
        #         costDictList.append(tempdict)
        #         # print(tempdict)
        #         costTupleList.append(tuple([e for _, e in sorted(tempdict.items())]))
        #         casenameList.append(fed+'_zero' if cost == 0 else fed + '_inf')
        #
        costDictList.append({f: 0 for f in initCostDict})
        costTupleList.append(tuple(len(initCostDict)*[0]))
        casenameList.append('T_zero')
        # costDictList.append({f: 1100 for f in initCostDict})
        # costTupleList.append(tuple(len(initCostDict)*[1100]))
        # casenameList.append('T_inf')

        costDictList.append(initCostDict)
        costTupleList.append(tuple([e for _, e in sorted(initCostDict.items())]))
        casenameList.append('Adaptive')

        # print(costTupleList)
        bestBundleDict = {}
        federateCashDict = {}

        # print("task list length:", len(tasklist))
        for costDict, costTuple, casename in zip(costDictList, costTupleList, casenameList):
            # print("Finding best bundle for cost:", costTuple)
            auction.inquirePrice(costDict= costDict)
            # print(costDict, casename, "task path dict:", len(auction.taskPathDict))
            taskcostlist = returnAvgPathCost(auction.taskPathDict)
            tasklist = self.shuffleTasks(taskcostlist)
            bundles = []
            for taskid in tasklist:
                if auction.findBestBundleinAuction([taskid]):
                    newbundle = auction.bestPathBundle
                    bundles.append(newbundle)

            alltasks, allpaths = combineBundles(bundles)
            bundle = PathBundle(alltasks, allpaths)
            # print(casename, len(bundle.pathlist))
            bestBundleDict[casename] = bundle #auction.bestPathBundle#PathBundleLite(alltasks, allpaths)
            # federateCashDict[casename] = bestBundleDict[costTuple].bundleRevenue

        # priceDict = self.suggestPriceDict(initCostDict, federateCashDict, auction.bestPathBundle)
        # if bestBundleDict['Adaptive'].bundleRevenue != bestBundleDict['T_zero'].bundleRevenue:
        #     print(bestBundleDict['Adaptive'].bundleRevenue, bestBundleDict['T_zero'].bundleRevenue)
            
        if self.context.ofs.auctioneer:
            newprice = optimizeCost(initCostDict, bestBundleDict['Adaptive'], bestBundleDict['T_zero'])

            # print("Auctioneer true and new price:", newprice)
            # print(initCostDict, newprice)

            # if bestBundleDict['Adaptive'].pathlist != bestBundleDict['T_zero'].pathlist:
            #     print(bestBundleDict['Adaptive'].bundleRevenue, bestBundleDict['T_zero'].bundleRevenue)
        else:
            newprice = False

        # print("initial price vs new price:", [e[1] for e in sorted(initCostDict.items())], [int(e) for e in newprice])
        # for costtuple, pathbundlelite in bestBundleDict.items():
        #     print("cost tuple:", costtuple, int(pathbundlelite.bundleBid), int(pathbundlelite.bundleCost), int(pathbundlelite.bundleRevenue))

        if newprice:
            for f in self.context.federates:
                f.timepricelist.append((self.context.time, newprice[f.name]))

            bestBundle = bestBundleDict['T_zero']
            for path in bestBundle.pathlist:
                path.updateWithFederateBid(newprice)

            bestBundle.updateValues()
            auction.bestPathBundle = bestBundle
            # print("Auctioneer revenue vs adaptive revenue:", bestBundleDict['Adaptive'].bundleRevenue, bestBundle.bundleRevenue)
        else:
            bestBundle = bestBundleDict['Adaptive']
            auction.bestPathBundle = bestBundle

        federateDict = {f.name: f for f in self.context.federates}
        for fed in auction.auctionFederates:
            federate = federateDict[fed]
            if federate.costlearning:
               federate.updateBestBundle(auction.bestPathBundle)

        return auction.bestPathBundle

    def initiateAuction(self):
        self.currenttasks = []
        # print(self.costSGLDict)
        # print("elements:",[e.name for e in self.context.elements])
        self.context.shuffleStream.shuffle(self.context.elements)
        for element in self.context.elements:
            if element.isSpace():
                newtask = element.collectTasks(self.context)
                if newtask:
                    self.currenttasks.append(newtask)

        if self.currenttasks:
            # print('new tasks:', [t.taskid for t in self.currenttasks])
            # print("current tasks federates:", [t.federateOwner.name for t in self.currenttasks])
            # print("curresnt tasks and owners:", [t.taskid for t in self.currenttasks], [t.elementOwner.name for t in self.currenttasks])
            pathbundle = self.runAuction(self.currenttasks)
            # print("best bundle tasks and federates:", [p.elementOwner.federateOwner.name for p in pathbundle.pathlist])
            # print("path bundle tasks:", [p.task.taskid for p in pathbundle.pathlist])
            tasklist = [p.task for p in pathbundle.pathlist]
            for task in tasklist:
                element = task.elementOwner
                element.pickupTask(task)

    # def suggestPriceDict(self, initCostDict, federateCashDict, bestBundle):
    #     totalDeltaCash = federateCashDict['T_zero'] - federateCashDict['T_inf']
    #     sumdeltaCash = 0
    #     priceDict = dict(initCostDict)
    #     linkCount = defaultdict(int)
    #     pathlist = bestBundle.pathlist
    #     for path in pathlist:
    #         linkfederates = [e.name for e in path.linkfederatelist]
    #         federateCount = Counter(linkfederates)
    #         for f, c in federateCount.items():
    #             linkCount[f] += c
    #
    #     print(linkCount)
    #     for fed in initCostDict:
    #         if fed not in linkCount:
    #             continue
    #
    #         deltaCash = federateCashDict[fed + '_zero'] - federateCashDict[fed + '_inf']
    #         sumdeltaCash += deltaCash
    #         priceDict[fed] = max(priceDict[fed], deltaCash/float(linkCount[fed]))
    #
    #     print('sum of delta vs total cash:', sumdeltaCash, totalDeltaCash)
    #     return priceDict

    #
    #
    # def updatePathFederateBundleDict(self, path):
    #     nodelist = path.getNodeList()
    #     # print("auctioneer: nodelist:", nodelist)
    #     edgelist = convertPath2Edge(nodelist)
    #     federatebundledict = {}
    #     for edge in edgelist:
    #         # print("auctioneer: edge:", edge)
    #         # print(self.nodes)
    #         federate = self.nodeFederateDict[edge[1]]
    #         if federate.name not in federatebundledict:
    #             federatebundledict[federate.name] = []
    #
    #         federatebundledict[federate.name].append((edge))
    #         # print("Auctioneer: Find path fed dict:", federate.name, edge)
    #
    #     # print("path:", nodelist, edgelist)
    #     # print("Auctioneer: federate and bundles:", [(a, len(b)) for (a,b) in federatebundledict.items()])
    #     federatebundledict = {k: EdgeBundle(v, path, self.namefederatedict[k]) for (k, v) in federatebundledict.items()}
    #
    #     path.updateBundles(federatebundledict)
    #     # print("Path and federate bundle list:", path.nodelist, [b.edgelist for b in federatebundledict.values()])
    #     self.edgebundlelist.extend(federatebundledict.values())
    #     # print("Auctioneer: federatebundledict:", federatebundledict)
    #     return federatebundledict
    #
    # def setDict2Dict(self, dict1, dict2):
    #     for key in dict2:
    #         # print("dict key and dict1 and dict2:", key, dict1[key] if key in dict1 else None, dict2[key])
    #         if key in dict1:
    #             # print("2 bundles:", dict1[key], dict2[key])
    #             dict1[key] = dict1[key].union(set([dict2[key]]))
    #         else:
    #             # print("bundle edgelist:", dict2[key].edgelist)
    #             dict1[key] = set([dict2[key]])
    #     return dict1
    #
    # def uniquePermutations(self, indexlist):
    #     # print("indexlist:", indexlist)
    #     # print("uniquePermulations:", [[p.nodelist for p in pathlist] for pathlist in indexlist])
    #     ntasks = len(indexlist)
    #     permutations = []
    #     combinations =  []
    #     for n in range(1,ntasks+1):
    #         tempcombinations = itertools.combinations(range(ntasks), n)
    #         combinations += list(tempcombinations)
    #
    #     for c in combinations:
    #         newlist = [indexlist[i] for i in c]
    #         # print("newlist:", newlist)
    #         tempproducts = itertools.product(*newlist)
    #         # print("Permutations:", [p.nodelist for p in list(tempproducts)])
    #         permutations.extend(list(tempproducts))
    #
    #     # print("Unique products:", )
    #     return permutations
    #
    #
    # def checkPathCombinations(self, pathlist):
    #     alledges = set([])
    #     # print("check path combination:", [p.nodelist for p in pathlist])
    #     for path in pathlist:
    #         newedges = set(convertPath2Edge(path.nodelist))
    #         # print("new edges:", newedges)
    #         intersection = alledges.intersection(newedges)
    #         # print("intersection:", intersection)
    #
    #         if intersection:
    #             # print(False)
    #             return False
    #
    #         alledges = alledges.union(newedges)
    #     # print(True)
    #     return True
    #
    # def updateCompatibleBundles(self, forced = False):
    #     if not self.compatibleBundles or forced:
    #         all_paths = list(self.pathdict.values())
    #         # print("Update compatible bundles: all paths:", all_paths)
    #         # print("All paths:", self.pathdict)
    #         probable_products = self.uniquePermutations(all_paths)
    #         possible_bundles = [PathBundle(plist) for plist in probable_products if self.checkPathCombinations(plist)]
    #
    #         # print("Auctioneer: possible path combinations:", [p.pathlist for p in possible_bundles])
    #         # print([t.length for t in possible_bundles])
    #         self.compatibleBundles = possible_bundles
    #         # return possible_bundles
    #
    #     for pathbundle in self.compatibleBundles:
    #         pathbundle.updateValues()
    #
    # def updateBundleBid(self):
    #     for edgebundle in self.edgebundlelist:
    #         edgebundle.updateBid(self.bundleBidDict[edgebundle])
    #
    # def updateBundles(self):
    #     for path in self.pathlist:
    #         path.updateValues()
    #
    #     self.updateCompatibleBundles()
    #     return self.findBestBundle()
    #
    # def removeBundles(self, bundlelist):
    #     # print([b.edgelist for b in bundlelist], " are removed")
    #     self.edgebundlelist = sorted([b for b in self.edgebundlelist if b not in bundlelist])
    #     bundleset = set(bundlelist)
    #     # for p in self.pathlist:
    #     #     print("path and intersection with bundles:", p.nodelist, [b.edgelist for b in set(p.edgebundles).intersection(bundleset)])
    #     self.pathlist = [p for p in self.pathlist if not set(p.edgebundles).intersection(bundleset)]
    #     for taskid, paths in self.pathdict.items():
    #         newpaths = [p for p in paths if p in self.pathlist]
    #         self.pathdict[taskid] = newpaths
    #
    #     emptykeys = [k for k in self.pathdict if not self.pathdict[k]]
    #     for k in emptykeys:
    #         self.pathdict.pop(taskid, None)
    #
    #     self.updateCompatibleBundles(forced = True)
    #
    #
    # def inquirePrice(self):
    #     federateBundleDict = {}
    #     for path in self.pathlist:
    #         tempdict = self.updatePathFederateBundleDict(path)
    #         # print("Auctioneer: path edge dict:", [(a, bundle.edgelist) for (a,bundle) in tempdict.items()])
    #         federateBundleDict = self.setDict2Dict(federateBundleDict, tempdict)
    #
    #     # print("auctioneer: federateBundleDict:", federateBundleDict)
    #     # print("Auctioneer: federate and bundles:", [(f, [b.edgelist for b in bundles]) for (f,bundles) in federateBundleDict.items()])
    #     self.bundleBidDict = {}
    #     for fed, bundleset in federateBundleDict.items():
    #         bundlelist = list(bundleset)
    #         # print("Federate:", fed)
    #         # print("bundle list:", edgebundlelist)
    #         # print("Inquireprice: bundle list federates:", [[(self.nodeFederateDict[x].name, self.nodeFederateDict[y].name) for (x,y) in bundle.edgelist] for bundle in edgebundlelist])
    #         # print("Auctioneer: fed and bundleset", fed, [b.edgelist for b in edgebundlelist])
    #         tempdict = self.namefederatedict[fed].getBundleBid(bundlelist)
    #         # print("Auctioneer: asker federate protocol cost:", [(b.federateAsker.name, fed, c) for (b,c) in tempdict.items()])
    #
    #         for b in tempdict:
    #             assert b not in self.bundleBidDict
    #             self.bundleBidDict[b] = tempdict[b]
    #
    #         # bundleBidDict = {x: y for x,y in zip(edgebundlelist, costlist)}
    #     self.updateBundleBid()
    #     self.updateBundles()
    #     self.updateCompatibleBundles()
    #
    # def findBestBundle(self, compatiblebundels = []):
    #     # print("length of compatible bundles:", len(self.compatibleBundles))
    #     if compatiblebundels:
    #         possible_bundles = compatiblebundels
    #     else:
    #         possible_bundles = self.compatibleBundles if self.compatibleBundles else self.updateCompatibleBundles()
    #
    #     if not possible_bundles:
    #         # self.bestPathBundle = None
    #         return False
    #
    #     path_bundle_cost = [b.bundleCost for b in possible_bundles]
    #     path_bundle_revenue = [b.bundleRevenue for b in possible_bundles]
    #     path_bundle_profit = [x-y for (x,y) in zip(path_bundle_revenue, path_bundle_cost)]
    #     # path_bundle_length = [b.length for b in possible_bundles]
    #     # print("pathbundle cost:", path_bundle_cost)
    #     # sortedcost = sorted(list(zip(path_bundle_cost, path_bundle_length)))
    #     # print("sorted cost:", sortedcost)
    #     sorted_revenue = sorted(list(zip(path_bundle_profit, possible_bundles)), reverse = True)
    #     # print("sorted revenue:", [(x, [p.nodelist for p in y.pathlist]) for x,y in sorted_revenue[:1]])
    #     self.bestPathBundle = sorted_revenue[0][1]
    #     return True
    #
    # def checkBundleinBundle(self, pathbundle, edgebundle):
    #     all_bundles = []
    #     for path in pathbundle.pathlist:
    #         all_bundles.extend(path.edgebundles)
    #
    #     # print("chekc bundle in bundle:", all_bundles)
    #     if edgebundle in all_bundles:
    #         return True
    #     return False
    #
    # def updateOpportunityCost(self):
    #     previousprofit = self.bestPathBundle.getBundleRevenue()
    #     # print("Default profit is: ", previousprofit)
    #     # print(len(self.edgebundlelist))
    #     for b in self.edgebundlelist:
    #         profit_0 = profit_1 = taskProfit_0 = taskProfit_1 = 0
    #         # print("Update opp cost: length of compatible bbundles:", len(self.compatibleBundles))
    #         # print("bundle:", b.edgelist)
    #         tempprice = b.price
    #         b.updateCost(0)
    #         if self.updateBundles():
    #             b.updateCost(tempprice)
    #             profit_0 = self.bestPathBundle.getBundleRevenue()
    #             taskProfit_0 = self.bestPathBundle.getTaskProfit(b.taskAsker)
    #
    #         # self.findBestBundle(compatiblebundles)
    #         compatiblebundles = [pathbundle for pathbundle in self.compatibleBundles if not self.checkBundleinBundle(pathbundle, b)]
    #         if self.findBestBundle(compatiblebundles):
    #             profit_1 = self.bestPathBundle.getBundleRevenue()
    #             taskProfit_1 = self.bestPathBundle.getTaskProfit(b.taskAsker)
    #
    #         # b.updateCost(10000)
    #         # self.updateBundles()
    #         # if not (profit_1<= previousprofit and profit_0>= previousprofit):
    #             # print("bundle max, min, OC, task OC:", list(b.edgelist), profit_0, profit_1, profit_0 - profit_1, taskProfit_0, taskProfit_1, taskProfit_0 - taskProfit_1)
    #         # assert profit_1<= previousprofit, profit_0>= previousprofit
    #         b.setGenOppCost(profit_0 - profit_1)
    #         # print("updated opportunity cost:", b.generalOpportunityCost)
    #
    #     self.updateBundles()
    #
    # def evolveBundles(self):
    #     self.updateOpportunityCost()
    #     for b in self.edgebundlelist:
    #         b.updateCost(max(b.getGeneralOppCost(), b.getBid()))
    #
    #     # print("edge bundle list:", self.edgebundlelist, len(self.edgebundlelist))
    #     while True:
    #         # print("Evolve bundles")
    #         removelist = sorted([(b.getGeneralOppCost() - b.getBid(), b) for b in self.edgebundlelist if b.getGeneralOppCost() < b.getBid()])
    #         # print("remove list:", [(c, b.edgelist) for c,b in removelist[:1]])
    #         if not removelist:
    #             break
    #         self.removeBundles([r[1] for r in removelist[:1]])
    #         # print("length of self.edgebundlelist after remove:", len(self.edgebundlelist))
    #         self.updateOpportunityCost()
    #
    #         # for b in self.edgebundlelist:
    #         #     # print("update price")
    #         #     b.updateCost(b.generalOpportunityCost)
    #         #
    #         # self.updateBundles()
    #     self.findBestBundle()
    #
    # def deliverTasks(self):
    #     taskpath = [(p.task, p) for p in self.bestPathBundle.pathlist]
    #     for task, path in taskpath:
    #         task.updatePath(path)
    #         element = task.elementOwner
    #         print("Task final value and pathprice:", task.getValue(task.federateOwner.time) , path.pathPrice)
    #         if task.getValue(task.federateOwner.time) - path.pathPrice >0:
    #             element.deliverTask(task)
    #
    # # def offerPrice2Federates(self):
    # #
    # #     for b in self.edgebundlelist:
    # #         federate = b.federateOwner
    # #
    # #         federate.grantBundlePrice(b)






















