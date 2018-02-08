import networkx as nx
import re
# import matplotlib.pyplot as plt
import networkx.algorithms.isomorphism as iso
import math
import numpy as np
import time
from itertools import cycle
from collections import deque, defaultdict
from .generalFunctions import *
from .path import Path


class SuperG():
    def __init__(self, context):
        self.Graph = self.createGraph(context)
        self.createElementGraphs(context)
        # self.drawGraph()

    def createElementGraphs(self, context):
        for e in [element for element in context.elements if element.isSpace()]:
            e.elementG = ElementG(self, e)

    def canTransmit(self, txElement, rxElement):
        txsection = int(re.search(r'.+(\d)', txElement.getLocation()).group(1))
        rxsection = int(re.search(r'.+(\d)', rxElement.getLocation()).group(1))
        canT = False

        if txElement.isSpace() and rxElement.isSpace():
            if abs(txsection - rxsection) <= 1 or abs(txsection - rxsection) == 5:
                canT = True

        elif txElement.isSpace() and rxElement.isGround():
            if txsection == rxsection:
                canT = True

        return canT

    def createGraph(self, context):
        G = nx.DiGraph()
        torder = context.time%6
        d = deque(range(6))
        d.rotate(-torder)
        for graphorder in d:
            enames = [e.name+'.%d'%graphorder for e in context.elements]
            G.add_nodes_from(enames)
            for tx in enames:
                for rx in [e for e in enames if e != tx]:
                    telement = context.nodeElementDict[tx[:-2]]
                    relement = context.nodeElementDict[rx[:-2]]
                    if self.canTransmit(telement, relement):
                        # txowner = telement.federateOwner
                        # rxowner = relement.federateOwner
                        cost = 0.
                        # if txowner != rxowner:
                        #     if relement.isSpace():
                        #         cost = rxowner.getCost('oISL', txowner)
                        #     elif relement.isGround():
                        #         cost = rxowner.getCost('oSGL', txowner)

                        # print("new edge:", telement.name, relement.name)
                        G.add_edge(tx, rx, weight=cost)

            context.propagate()
        return G

    def drawGraph(self):
        plt.figure()
        # nx.draw_networkx_nodes(self.Graph, pos)
        # nx.draw_networkx_edges(self.Graph, pos)
        nx.draw(self.Graph)
        plt.show()

class ElementG():
    def __init__(self, SuperG, element):
        self.storagePenalty = deque(6*[0])
        self.Graph = None
        self.elementOwner = element
        self.orderPathDict = defaultdict(list)

        self.createGraph(SuperG.Graph)
        self.addPaths()
            # self.rawSuperGraph = None
        # self.SuperGaph = None
        # self.elementOwner = element
        # self.superShorestPaths = None
        # self.superPathsCost = None
        # self.graphList = []
    def getPaths(self, time):
        torder = time%6
        return self.orderPathDict[torder]

    def createGraph(self, G):
        self.Graph = G.copy()
        for i, s in enumerate(self.storagePenalty):
            name1 = '%s.%d'%(self.elementOwner.name, i%6)
            name2 = '%s.%d'%(self.elementOwner.name, (i+1)%6)
            # print(name1, name2)
            self.Graph.add_edge(name1, name2, weight= s)

    def updateGraph(self, context, taskvaluelist):
        self.storagePenalty = deque(self.elementOwner.federateOwner.getStorageCostList(self.elementOwner, taskvaluelist=taskvaluelist))

        torder = self.elementOwner.federateOwner.time%6
        self.storagePenalty.rotate(-torder)
        # print(self.storagePenalty)
        # edges = self.Graph.edges()
        # fedname = self.elementOwner.federateOwner.name
        # for e1, e2 in edges:
        #     fname = re.search(r'.+\.(F\d)\..+', e2).group(1)
        #     if fedname == fname:
        #         self.Graph[e1][e2]['weight'] = 0.
        #     elif 'GS' in e2:
        #         self.Graph[e1][e2]['weight'] = context.auctioneer.costSGLDict[fname]
        #     else:
        #         self.Graph[e1][e2]['weight'] = context.auctioneer.costISLDict[fname]
        #
        #     # print("updateGraph:", e1, e2, self.Graph[e1][e2]['weight'])

        for i, s in enumerate(self.storagePenalty):
            name1 = '%s.%d'%(self.elementOwner.name, i%6)
            name2 = '%s.%d'%(self.elementOwner.name, (i+1)%6)
            # print name1, name2, s
            self.Graph[name1][name2]['weight'] = s

        # edges = self.Graph.edges()
        # print(context.auctioneer.costSGLDict, context.auctioneer.costISLDict)
        # for e in edges:
        #     print("edge data:", self.elementOwner.federateOwner.name, e, self.Graph.get_edge_data(*e))


    def bfs_paths(self, source, destination):
        q = [(source, [source])]
        while q:
            v, path = q.pop(0)
            for next in set(self.Graph.neighbors(v)) - set(path):
                if next == destination:
                    yield path + [next]
                else:
                    q.append((next, path + [next]))

    def findAllPaths(self, source, destinations):
        allpathes = []
        for d in destinations:
            allpathes.extend(self.bfs_paths(source, d))

        return allpathes

    def addPaths(self):
        nodes = self.Graph.nodes()
        sources = [n for n in nodes if self.elementOwner.name in n]
        destinations = [n for n in nodes if 'GS' in n]
        for s in sources:
            # print("source:", s)
            nodelist = self.findAllPaths(s, destinations)
            # print("nodelist:", nodelist)
            # print("order:", int(s[-1]))
            self.orderPathDict[int(s[-1])] = [Path(self.elementOwner, nl) for nl in nodelist] if nodelist else []

    def findcheapestpath(self, deltatime):
        pass
    # def findShortestPathes(self, Graph):
    #     nodes = Graph.nodes()
    #     # print "nodes:", nodes
    #     sourcename = '%s.%d'%(self.elementOwner.name, self.graphOrder)
    #
    #     groundstations = [n for n in nodes if 'GS' in n]
    #     # print "ground stations:", groundstations
    #     temppathlist = []
    #     pathcostlist = []
    #     for i in range(len(self.storagePenalty)):
    #         for g in groundstations:
    #             # print sourcename, g
    #             if nx.has_path(Graph, source=sourcename,target=g):
    #                 sh = nx.shortest_path(Graph, sourcename, g)
    #                 temppathlist.append(sh)
    #                 tuplist = convertPath2Edge(sh)
    #                 # print tuplist
    #                 costlist = []
    #                 for (source, target) in tuplist:
    #                     cost = (0 if (self.elementOwners[sourcename[:-2]] == self.elementOwners[target[:-2]] and sourcename[:-2] != target[:-2])
    #                             else Graph[source][target]['weight'])
    #                     costlist.append(cost)
    #
    #                 pathcostlist.append(costlist)
    #
    #     # print pathcostlist
    #     # print "find shortest paths:", temppathlist, pathcostlist
    #     return temppathlist, pathcostlist

    # def findcheapestpath(self, deltatime):
    #     future = (self.graphOrder + deltatime)%6
    #     futurename = '%s.%d'%(self.elementOwner.name, future)
    #
    #     pathlist = self.superShorestPaths
    #     costlist = self.superPathsCost
    #     pathcost = [tup for tup in zip(costlist, pathlist) if futurename in tup[1]]
    #
    #     sortedpath = sorted([(sum(x), y) for x,y in pathcost])
    #     # print "cost vs path:", sorted(zip([sum(c) for c in costlist], pathlist))
    #
    #     # return convertPath2Edge(sortedpath[0])
    #     return sortedpath[0]

        # def setGraphList(self, context):
        #     self.graphList = context.Graph.graphList
        #     self.graphOrder = context.Graph.graphOrder









