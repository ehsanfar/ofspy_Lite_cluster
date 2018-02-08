# import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import re
import math
from collections import Counter, defaultdict
from matplotlib.font_manager import FontProperties
# from matplotlib import gridspec
from scipy.optimize import minimize


# from .bundle import PathBundle

def checkEqual2(iterator):
   return len(set(iterator)) <= 1

def findbestxy(N):
    if N % 2 != 0:
        N += 1
    temp = int(N ** 0.5)
    while N % temp != 0:
        temp -= 1

    return (temp, N // temp)

def convertPath2Edge(pathlist):
    tuplist = []
    for i in range(len(pathlist) - 1):
        tuplist.append((pathlist[i], pathlist[i + 1]))

    return tuplist

def convertLocation2xy(location):
    if 'SUR' in location:
        r = 0.5
    elif 'LEO' in location:
        r = 1.
    elif 'MEO' in location:
        r = 1.5
    elif "GEO" in location:
        r = 2
    else:
        r = 2.35

    sect = int(re.search(r'.+(\d)', location).group(1))
    tetha = +math.pi / 3 - (sect - 1) * math.pi / 3

    x, y = (r * math.cos(tetha), r * math.sin(tetha))
    # print location, x, y
    return (x, y)


def convertPath2StaticPath(path):
    temppath = [e[:-2] for e in path.nodelist]
    ends = [e[-1] for e in path.nodelist]
    seen = set([])
    seen_add = seen.add
    staticpath = [e for e in temppath if not (e in seen or seen_add(e))]
    # print "convert path 2 static path:", path, staticpath
    deltatime = path.deltatime
    assert len(set(ends[deltatime:])) == 1
    return (staticpath, deltatime)

def fillBetween3Points(a, b, c):
    sortedpoints = sorted([a,b,c])
    # print sortedpoints
    z = zip(a,b,c)
    x1 = np.linspace(sortedpoints[0][0], sortedpoints[1][0], num=2)
    x2 = np.linspace(sortedpoints[1][0], sortedpoints[2][0], num=2)
    # print x1, x2
    y1 = (sortedpoints[0][1]-sortedpoints[1][1])*(x1-sortedpoints[0][0])/float(sortedpoints[0][0]-sortedpoints[1][0]) + sortedpoints[0][1] if sortedpoints[0][0]-sortedpoints[1][0] != 0 else None
    y2 = (sortedpoints[0][1]-sortedpoints[2][1])*(x1-sortedpoints[0][0])/float(sortedpoints[0][0]-sortedpoints[2][0]) + sortedpoints[0][1] if sortedpoints[0][0]-sortedpoints[2][0] != 0 else None
    y3 = (sortedpoints[0][1]-sortedpoints[2][1])*(x2-sortedpoints[2][0])/float(sortedpoints[0][0]-sortedpoints[2][0]) + sortedpoints[2][1] if sortedpoints[0][0]-sortedpoints[2][0] != 0 else None
    y4 = (sortedpoints[1][1]-sortedpoints[2][1])*(x2-sortedpoints[2][0])/float(sortedpoints[1][0]-sortedpoints[2][0]) + sortedpoints[2][1] if sortedpoints[1][0]-sortedpoints[2][0] != 0 else None
    # plt.fill_betweenx(X, Y, interpolate=True)
    col = 'yellow'
    if y1 is None:
        # plt.plot(x2, y3, 'g')
        # plt.plot(x2, y4, 'g')
        plt.fill_between(x2, y3, y4, color = col)
    elif y4 is None:
        # plt.plot(x1, y1, 'g')
        # plt.plot(x1, y2, 'g')
        plt.fill_between(x1, y1, y2, color= col)

    elif y2 is None or y3 is None:
        print("there is error with points")

    else:
        # plt.plot(x1, y1, 'g')
        # plt.plot(x1, y2, 'g')
        # plt.plot(x2, y3, 'g')
        # plt.plot(x2, y4, 'g')
        plt.fill_between(x1, y1, y2, color= col)
        plt.fill_between(x2, y3, y4, color= col)



def drawGraph(graph, context):
    G = graph.graphList[graph.graphOrder]

    if not plt.fignum_exists(1):
        plt.figure(1)
        plt.ion()
        plt.show()

    plt.clf()
    nodes = [e.name for e in graph.elements]
    nameselementdict = {x: y for (x, y) in zip(nodes, graph.elements)}
    # print "nodes:", nodes
    satellites = [n for n in nodes if 'GS' not in n]
    # alltuples = set([])
    # for s in satellites:
    #     path = graph.findcheapestpath(s)
    #     pathedges = convertPath2Edge(path)
    #     # print "graphorder & source & path:", s, pathedges
    #     alltuples = alltuples.union(pathedges)

    alltuples = set([])
    print("Number of saved tasks:", [len(element.savedTasks) for element in graph.elements])
    currenttasks = [e for l in [element.savedTasks for element in graph.elements] for e in l]
    assert len(set(currenttasks)) == len(currenttasks)
    activetasks = [t for t in currenttasks if t.activationTime == context.time]
    for actives in activetasks:
        path = [a.name for a in actives.pathlist]
        pathedges = convertPath2Edge(path)
        alltuples = alltuples.union(pathedges)

    # for s in satellites:
    #     path = graph.findcheapestpath(s)
    #     pathedges = convertPath2Edge(path)
    #     # print "graphorder & source & path:", s, pathedges
    #     alltuples = alltuples.union(pathedges)
    recenttasks = [t.taskid for t in currenttasks if t.initTime == context.time-1]
    # print "recent tasks:", recenttasks
    elementswithrecenttasks = [e for e in graph.elements if set([t.taskid for t in e.savedTasks]).intersection(recenttasks)]
    # print "elementlist with recent tasks:", elementswithrecenttasks

    section2pointsdict = {1: [(0, 1), (0.866, 0.5)], 2: [(0.866, 0.5), (0.866, -0.5)], 3: [(0.866, -0.5), (0, -1)], 4: [(0, -1), (-0.866, -0.5)], 5: [(-0.866, -0.5), (-0.866, 0.5)], 6: [(-0.866, 0.5), (0, 1)]}

    nodeLocations = [e.getLocation() for e in graph.elements]
    pos = {e.name: convertLocation2xy(nodeLocations[i]) for i, e in enumerate(graph.elements)}

    positionsection = [[pos[e.name]]+section2pointsdict[e.section] for e in elementswithrecenttasks]
    # print "position and section :", positionsection

    sec = {e.name: nodeLocations[i] for i, e in enumerate(graph.elements)}
    labels = {n: n[0] + n[-3:] for n in nodes}
    labelpos = {n: [v[0], v[1] + 0.3] for n, v in pos.items()}
    x = np.linspace(-1.0, 1.0, 50)
    y = np.linspace(-1.0, 1.0, 50)
    X, Y = np.meshgrid(x, y)
    F = X ** 2 + Y ** 2 - 0.75
    plt.contour(X, Y, F, [0])
    # print nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in nodes if 'GS' not in n and nameselementdict[n].savedTasks],
                           node_color='r', node_size=100)

    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in nodes if 'GS' not in n and not nameselementdict[n].savedTasks],
                           node_color='g', node_size=100)

    # nx.draw_networkx_nodes(Graph, pos, nodelist=[n for n in nodes if 'GS' not in n and 'LE' in sec[n]], node_color='g', node_size=100)

    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in nodes if 'GS' in n], node_color='b', node_size=100)

    # print "Graph all tuples: ", alltuples
    for ps in positionsection:
        fillBetween3Points(*ps)

    nx.draw_networkx_edges(G, pos, edgelist=list(alltuples))
    # nx.draw_networkx_edges(Graph, pos)
    nx.draw_networkx_labels(G, labelpos, labels, font_size=8)

    plt.xticks([])
    plt.yticks([])
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    # plt.draw()
    plt.draw()
    plt.waitforbuttonpress()
    # plt.pause(0.5)


# Figure is closed

def drawGraphbyDesign(number, design):
    elements = design.split(' ')
    federates = set([int(e[0]) for e in elements])
    federates_location_dict = defaultdict(list)
    federates_type_dict = defaultdict(list)
    federate_coordinates_dict = defaultdict(list)
    my_dpi = 150
    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    for r in [4, 2.25, 1.]:
        x = np.linspace(-1.0*r, 1.0*r, 50)
        y = np.linspace(-1.0*r, 1.0*r, 50)
        X, Y = np.meshgrid(x, y)
        F = X ** 2 + Y ** 2 - r
        plt.contour(X, Y, F, [0], colors='k', linewidths = 0.3, origin = 'lower', zorder = -1)

    font = FontProperties()
    font.set_style('italic')
    font.set_weight('bold')
    font.set_size('x-small')
    for x,y,lab in [(0,0,'SUR'), (0, 1, "LEO"),(0, 1.5, 'MEO'),(0, 2, 'GEO')]:
        # plt.annotate(lab, xy = (x,y), xytext = (x-0.2, y-0.1))
        plt.text(x,y, ha="center", va="center", s = lab, bbox = dict(fc="w", ec="w", lw=2),fontproperties=font)

    for i, (x, y) in enumerate([convertLocation2xy(e) for e in ['OOO'+str(i) for i in range(1,7)]]):
        plt.text(x, y, ha="center", va="center", s=str(i+1), bbox=dict(fc="none", ec="none", lw=2), fontproperties=font)

    font.set_size('medium')
    plt.text(0, 2.3 , ha="left", va="center", s=r'$|\rightarrow \theta$', bbox=dict(fc="w", ec="w", lw=2), fontproperties=font)

    types_dict = {'GroundSta': "G", 'Sat': 'S'}
    colordict = {'F1': 'yellow', 'F2': 'lightcyan', 'F3': 'lightgrey'}
    allpossiblelocations = []
    for location in ['SUR', 'LEO', 'MEO', 'GEO']:
        for i in range(1,7):
            allpossiblelocations.append(location + str(i))

    allpossiblecoordinates = [convertLocation2xy(e) for e in allpossiblelocations]
    plt.scatter(*zip(*allpossiblecoordinates), marker = "H", s = 800, color = 'k', facecolors = 'w')
    for f in federates:
        types = [re.search(r'\d\.(.+)@(\w+\d)', e).group(1) for e in elements if '%d.' % f in e]
        federates_type_dict['F%d'%f] = [types_dict[t] for t in types]
        federates_location_dict['F%d'%f] = [re.search(r'(.+)@(\w+\d)', e).group(2) for e in elements if '%d.'%f in e]
        federate_coordinates_dict['F%d'%f] = [convertLocation2xy(loc) for loc in federates_location_dict['F%d'%f]]
        plt.scatter(*zip(*federate_coordinates_dict['F%d'%f]), marker = "H", s = 800, edgecolors = 'k', facecolors = colordict['F%d'%f], linewidth='3')
        for x, y in federate_coordinates_dict['F%d'%f]:
            plt.annotate('F%d'%f, xy = (x, y), xytext = (x-0.1, y-0.075))


    plt.xticks([])
    plt.yticks([])
    rlim = 2.5
    plt.xlim(-rlim, rlim)
    plt.ylim(-rlim+0.2, rlim)
    plt.axis('off')
    des_roman_dict = {1: 'I', 2: 'II', 3:'III', 4:'IV', 5:'V'}
    plt.savefig("Design_%s.pdf"%des_roman_dict[number], bbox_inches='tight')

    # plt.show()


def drawGraphs(graph):
    # pos = None

    plt.figure()
    n1, n2 = findbestxy(len(graph.graphList))
    # print n1,n2
    earth = plt.Circle((0, 0), 1.1, color='k', fill=True)

    for j, g in enumerate(graph.graphList):
        nodes = [e.name for e in graph.elements]
        pos = {e.name: convertLocation2xy(graph.nodeLocations[j][i]) for i, e in enumerate(graph.elements)}
        sec = {e.name: graph.nodeLocations[j][i] for i, e in enumerate(graph.elements)}
        labels = {n: n[0] + n[-3:] for n in nodes}
        labelpos = {n: [v[0], v[1] + 0.3] for n, v in pos.items()}
        ax = plt.subplot('%d%d%d' % (n1, n2, j + 1))
        x = np.linspace(-1.0, 1.0, 50)
        y = np.linspace(-1.0, 1.0, 50)
        X, Y = np.meshgrid(x, y)
        F = X ** 2 + Y ** 2 - 0.75
        plt.contour(X, Y, F, [0])
        # print nodes
        nx.draw_networkx_nodes(g, pos, nodelist=[n for n in nodes if 'Ground' not in n and 'LE' not in sec[n]],
                               node_color='r', node_size=100)
        nx.draw_networkx_nodes(g, pos, nodelist=[n for n in nodes if 'Ground' not in n and 'LE' in sec[n]],
                               node_color='g', node_size=100)
        nx.draw_networkx_nodes(g, pos, nodelist=[n for n in nodes if 'Ground' in n], node_color='b', node_size=100)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, labelpos, labels, font_size=8)
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        # ax.set_title('Graph:'+str(j))
        # print j, graph.shortestPathes[j]

    # plt.savefig("Networks_elements%d_.png"%len(graph.elementlist), bbox_inches='tight')
    plt.show()


def bfs_paths(G, source, destination):
    queue = [(source, [source])]
    while queue:
        v, path = queue.pop(0)
        for next in set(G.neighbors(v)) - set(path):
            if next == destination:
                yield path + [next]
            else:
                queue.append((next, path + [next]))

def findAllPaths(G, sources, destinations):
    allpathes = []
    for s in sources:
        for d in destinations:
            allpathes.extend(bfs_paths(G, s, d))

    return allpathes

#
# class Path():
#     def __init__(self, l):
#         self.linklist = l
def findClosestIndex(value, valulist):
    abslist = [abs(v-value) for v in valulist]
    return abslist.index(min(abslist))


def addDict2Dict(dict1, dict2):
    dict3 = dict1.copy()
    for d, c in dict2.items():
        dict3[d] += c
    return dict3

def returnCompatiblePaths(pathlist, linkcounter, maxlink = 1):
    # for path in pathlist[0]

    # print("length of pathlist:", len(pathlist))
    # print([p.linklist for p in pathlist[0]], linkcounter)
    if pathlist:
        queue = [(0, [], linkcounter)]
        while queue:
            n, histpath, s = queue.pop(0)
            # print("length of pathlist and n:", len(pathlist), n)
            # if n == len(pathlist) - 1:
            #     yield histpath
            # else:
            nextpaths = []
            for path in pathlist[n]:
                newcounter = Counter(path.linklist)
                combinedcounter = addDict2Dict(s, newcounter)
                valueset = list(combinedcounter.values())
                # print("counter value set:", valueset)
                # print(combinedcounter)
                if max(valueset) <= maxlink:
                    nextpaths.append(path)
                # else:
                #     print(max(valueset))

            # print(len(pathlist[n]), len(nextpaths))
            # print("current path, next pathf:\n", [e.linklist for e in histpath],'\n', [p.linklist for p in nextpaths])
            # print("set:", s)
            n += 1
            for np in nextpaths:
                # print("new path:", np.linklist)
                if n == len(pathlist):
                    # print([p.linklist for p in histpath + [np]])
                    yield histpath + [np]
                else:
                    # scopy = s.union(set(np.linklist))
                    combinedcounter = addDict2Dict(s, newcounter)
                    queue.append((n, histpath + [np], combinedcounter))


def returnAvgPathCost(taskPathDict):
    tasksumcostnum = [(min([p.pathCost for p in paths]), len(paths), taskid) for taskid, paths in taskPathDict.items()]
    # tasksumcostnum = [(min([len(p.nodelist) for p in paths]), len(paths), taskid) for taskid, paths in taskPathDict.items()]
    avgcosttask = sorted([(x, z) for x,y,z in tasksumcostnum])
    # print("avg cost task:", avgcosttask)
    return avgcosttask

def combineBundles(bundles):
    alltasks = []
    allpaths = []
    for b in bundles:
        alltasks.extend(list(b.tasklist))
        allpaths.extend(list(b.pathlist))

    return alltasks, allpaths


def generateFops(costrange, storange):
    fops = []
    for cost in costrange:
        costsgl = cost
        costisl = cost

        for sto in storange:
            stopen = sto
            for sto2 in storange:
                stopen2 = sto2
                yield ["x%d,%d,%d" % (costsgl, costisl, stopen2), "x%d,%d,%d" % (costsgl, costisl, stopen), "x%d,%d,%d" % (costsgl, costisl, sotpen)]


def calGaussianKernel(x, y, M, N, scale = 0.8):

    sigma1 = scale*M/6.
    sigma2 = scale*N/10.

    kernelmesh = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            delta1 = min(abs(x-i), M-abs(x-i))
            delta2 = abs(y-j)
            kernelmesh[i, j] = math.exp(-delta1**2/(2*sigma1**2)-delta2**2/(2*sigma2**2))

    kernelmesh = kernelmesh/sum(sum(kernelmesh))
    return kernelmesh


def matchVariance(a, b, var0):
    # print(var0 ** 0.5)
    # currentvar = a * b / ((a + b + 1.) * (a + b) ** 2)
    # print(currentvar ** 0.5)
    k = (a * b - var0 * (a + b) ** 2) / (var0 * (a + b) ** 3)
    # print("a, b, coef:", a, b, k)
    return (k * a, k * b)

initcostlist = linkCountList = federatelist = pathLinkCount = taskValueDict = linkCountList_A = None
R_LiA = R_TiA = pathTaskValueList = pathlist = []

def calTaskLinkRevenue(costlist, fi, federatelist, pathlist, pathLinkCount, linkCountDict, taskValueDict):
    federatelinkcost = 0.
    for path, federateCount in zip(pathlist, pathLinkCount):
        if path.elementOwner.federateOwner.name == federatelist[fi]:
            for fed, cost in zip(federatelist, costlist):
                federatelinkcost += cost * federateCount[fed]

    fed = federatelist[fi]
    linkreveune = costlist[fi] * linkCountDict[fed]
    # print(taskValueDict[fed] , pathCostDict[fed])
    taskrevenue = taskValueDict[fed] - federatelinkcost

    return (taskrevenue, linkreveune)

class Constraint1():
    def __init__(self, fi, linkCountDict, pathlist, federatelist, pathLinkCount, taskValueDict, R_LiA, R_TiA):
        self.fi = fi
        self.linkCountDict = linkCountDict
        self.pathlist = pathlist
        # self.inicostlist = initcostlist
        self.federatelist = federatelist
        self.pathLinkCount = pathLinkCount
        self.taskValueDict = taskValueDict
        self.R_LiA = R_LiA
        self.R_TiA = R_TiA

    def calTaskLinkRevenue(self, costlist, fi, federatelist, pathlist, pathLinkCount, linkCountDict, taskValueDict):
        federatelinkcost = 0.
        for path, federateCount in zip(pathlist, pathLinkCount):
            if path.elementOwner.federateOwner.name == federatelist[fi]:
                for fed, cost in zip(federatelist, costlist):
                    federatelinkcost += cost * federateCount[fed]

        fed = federatelist[fi]
        linkreveune = costlist[fi] * linkCountDict[fed]
        # print(taskValueDict[fed] , pathCostDict[fed])
        taskrevenue = taskValueDict[fed] - federatelinkcost

        return (taskrevenue, linkreveune)

    def __call__(self, costlist):
        R_Ti, R_Li = self.calTaskLinkRevenue(costlist, self.fi, self.federatelist, self.pathlist, self.pathLinkCount, self.linkCountDict, self.taskValueDict)
        # print(self.fi, R_Li + R_Ti, self.R_LiA[self.fi] + self.R_TiA[self.fi])
        return R_Li + R_Ti - self.R_LiA[self.fi] - self.R_TiA[self.fi]



class Constraint2():
    def __init__(self, pi, path, pathTaskValueList, pathLinkCount, federatelist):
        self.pi = pi
        self.path = path
        self.pathTaskValueList = pathTaskValueList
        self.pathLinkCount = pathLinkCount
        self.federatelist = federatelist
    def __call__(self, costlist):
        self.federateCount = self.pathLinkCount[self.pi]
        # pathcost = sum([c for c, f in zip(self.path.linkcostlist, self.path.linkfederatelist) if f is self.path.elementOwner.federateOwner])
        pathcost = 0.
        # print("Already path cost:", pathcost)
        value = self.pathTaskValueList[self.pi]
        for fed, cost in zip(self.federatelist, costlist):
            pathcost += cost * self.federateCount[fed]

        return value - pathcost

class Objective():
    def __init__(self, linkcostlist):
        self.linkcostlist = linkcostlist

    def __call__(self, costlist):
        return -1*sum([a*b for a,b in zip(costlist, self.linkcostlist)])

def optimizeCost(initCostDict, adaptiveBestBundle, bestBundle):
    global linkCountList
    # global initcostlist, linkCountList, federatelist, pathLinkCount, taskValueDict, pathCostDict0, R_LiA, R_TiA
    initCostItems = sorted(list(initCostDict.items()))
    federatelist = [e[0] for e in initCostItems]
    initcostlist = [e[1] for e in initCostItems]


    # pathCostDict = defaultdict(int)
    pathLinkCount = []
    pathTaskValueList = []
    linkCountDict = defaultdict(int)
    taskValueDict = defaultdict(int)
    # pathCostDict0 = defaultdict(int)
    pathlist = bestBundle.pathlist
    taskvalues = bestBundle.taskvalues

    for taskvalue, path in zip(taskvalues, pathlist):
        federateOwner = path.elementOwner.federateOwner.name
        taskValueDict[federateOwner] += taskvalue
        linkfederates = [e.name for e in path.linkfederatelist if federateOwner != e.name]
        pathTaskValueList.append(taskvalue)
        # print(federateOwner, [e.name for e in path.linkfederatelist], linkfederates)
        federateCount = defaultdict(int, Counter(linkfederates))
        pathLinkCount.append(federateCount)
        for f, c in federateCount.items():
            linkCountDict[f] += c

    linkCountList = [e[1] for e in sorted(list(linkCountDict.items()))]
    # print(linkCountDict, linkCountList)

    pathLinkCount_A = []
    linkCountDict_A = defaultdict(int)
    taskValueDict_A = defaultdict(int)
    # pathCostDict0 = defaultdict(int)
    pathlist_A = adaptiveBestBundle.pathlist
    taskvalues_A = adaptiveBestBundle.taskvalues

    for taskvalue, path in zip(taskvalues_A, pathlist_A):
        federateOwner = path.elementOwner.federateOwner.name
        taskValueDict_A[federateOwner] += taskvalue
        linkfederates = [e.name for e in path.linkfederatelist if federateOwner != e.name]
        federateCount = defaultdict(int, Counter(linkfederates))
        pathLinkCount_A.append(federateCount)
        for f, c in federateCount.items():
            linkCountDict_A[f] += c

    # linkCountList_A = [e[1] for e in sorted(list(linkCountDict_A.items()))]
    # print("Federate link Count list:", linkCountList_A)

    # pathCostDict_A = defaultdict(int)
    # pathCostDict_A[federateOwner] = sum([federateCount[fed] * cost
    #                                        for federateCount, fed, cost in zip(pathLinkCount_A, federatelist, initcostlist)])
    # if len(pathlist) != len(pathlist_A):
    #     print(len(pathlist), len(pathlist_A))
    R_LiA = []
    R_TiA = []
    for i in range(len(federatelist)):
        Rt, Rl = calTaskLinkRevenue(initcostlist, i, federatelist, pathlist_A, pathLinkCount_A, linkCountDict_A, taskValueDict_A)
        R_LiA.append(Rl)
        R_TiA.append(Rt)

    # print("Revenue Adaptive:", sum(R_TiA) + sum(R_LiA))

    # print("zero and adaptive links:", linkCountDict, linkCountDict_A)

    # print("Adaptive task and link revenue:", R_TiA, R_LiA)
    # def objective(costlist):
    #     global linkCountList
    #     # print("objective funciton :", )
    #     # print(linkCountList)
    #     return -1*sum([a*b for a,b in zip(costlist, linkCountList)])
    objective = Objective(linkCountList)

    conslist1 = [{'type': 'ineq', 'fun': Constraint1(i, linkCountDict, pathlist, federatelist, pathLinkCount, taskValueDict, R_LiA, R_TiA)} for i in range(len(initcostlist))]
    conslist2 = [{'type': 'ineq', 'fun': Constraint2(i, path, pathTaskValueList, pathLinkCount, federatelist)} for i, path in enumerate(pathlist)]


    # con1 = {'type': 'ineq', 'fun': constraint1}
    # con2 = {'type': 'ineq', 'fun': constraint2}
    # con3 = {'type': 'ineq', 'fun': constraint3}

    cons = conslist1 + conslist2 # [con1, con2, con3][:len(initCostDict)]

    bnds = [(min(0, 1100), 1101) for c in initcostlist]
    # print("boundaries:", bnds)

    # print("length of constraints:", len(initCostDict), len(cons))
    templist = initcostlist[:]
    initcostlist = [0 for i in range(len(initcostlist))]

    sol = minimize(objective, initcostlist, method = 'SLSQP', bounds = bnds, constraints = cons)

    # print("solution:", sol.x)
    # print("constraints:")
    # for con in cons:
    try:
        cons_changes = []
        for con in cons:
            if not con['fun'](sol.x):
                return False
            cons_changes.append(int(round(con['fun'](sol.x))))
            # print(cons_changes)
            # consresults = all([e >= 0 for e in [int(round(con['fun'](sol.x))) for con in cons]])
        if all([e >= 0 for e in cons_changes]) and sum(cons_changes[:2])>0:
            # if True:
            # print(templist, [int(e) for e in sol.x])
            # print("Revenue 2, 1:", [int(round(con['fun'](sol.x))) for con in cons])
            # print('')
            return {'F%d' % (i+1): c for i, c in enumerate(list(sol.x))}

        return False
    except:
        return False

        # print(calGaussianKernel(0,7,6,10, 0.6))


# nactions = 12
# nstates = 6
# N = nactions * 10
# M = nstates * 10
# n = int(N/3)
# m = int(2*M/3)
# kernelmesh = np.zeros((M, N))
#
# kernelmesh = calGaussianKernel(m,n, M, N)
#
# print(sum(sum(kernelmesh)))
# print(kernelmesh.shape)
#
# # f, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=True)
# gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
# ax1 = plt.subplot(gs[0])
# ax2 = plt.subplot(gs[1], sharey=ax1)
# plt.setp(ax2.get_yticklabels(), visible=False)
#
# ax1.plot(100*kernelmesh[m, :], 'k--', zorder = -1)
# ax1.text(1,0.21, ha="left", va="center", s = 'sector = 4')
# ax1.axvline(m, zorder = -2)
# x1 = [i for i in range(0, N-1, 10)]
# y = list(100*kernelmesh[m,:])[::10]
# ax1.scatter(x1, y, marker = 'o', s = 100, facecolors = 'w', edgecolors= 'k')
# ax1.set_xlabel('action (k$)')
# ax1.set_ylabel(r'Q learning factor: $\alpha$')
# # ax1.set_title('sector = %d'%m)
# ax2.plot(list(100*kernelmesh[:, n]), 'k--', zorder = -1)
# ax2.axvline(n, zorder = -2)
# ax2.text(1,0.21, ha="left", va="center", s = 'action = 0.4')
#
# x2 = [i for i in range(0, M-1, 10)]
# y = list(100*kernelmesh[:, n])[::10]
# ax2.scatter(x2, y, marker = 'o', s = 100, facecolors = 'w', edgecolors= 'k')
# ax2.set_xlabel('states (sectors)')
# # ax2.set_title('action = %1.1f'%(n/100.))
# plt.sca(ax1)
# plt.xticks(x1, [100*a/1000 for a in range(nactions)], rotation = 0)
# plt.xlim(-5, N-5)
#
# plt.sca(ax2)
# plt.xticks(x2, [(i+1) for i in range(nstates)], rotation = 0)
# plt.xlim(-5, M-5)
# plt.tight_layout()
#
# plt.savefig("Q_Gaussian_qupdate.pdf", bbox_inches='tight')
# plt.suptitle('state-action-reward: (4, 0.4, 1)')
# plt.subplots_adjust(top=0.93)
# plt.show()
#     #


    # for path in pathlist[0]:
    #     print(path)
    #     tempset = set(path.linklist)
    #     print("Linkset and temp set:", linkset, tempset)
    #     inter = linkset.intersection(tempset)
    #     print(inter)
    #     if inter:
    #         continue
    #     else:
    #         nextset = linkset.union(tempset)
    #         print("nextset:", nextset)
    #         print("length:", len(pathlist))
    #         if len(pathlist)>1:
    #
    #             yield returnCompatiblePaths(pathlist[1:], nextset, histpath + [path])
    #         else:
    #             yield histpath + [path]

# l1 = [(1,2), (2,3), (3,4)]
# l2 = [(2,4), (4,9)]
# l3 = [(1,3), (4,5)]
#
#
# p1 = Path(l1)
# p2 = Path(l2)
# p3 = Path(l3)
# p4 = Path([(1,4),(5,6)])
#
# gen = returnCompatiblePaths([[p1, p2, p3, p4], [p1, p2, p3, p4],[p1, p2, p3, p4]])
# print(len(list(gen)))
# for g in gen:
#     print([e.linklist for e in g])

# nodes = range(1,12)
# edges = [(1,7), (4,7), (4,2), (6,2), (4,7), (7,3), (7,5), (2,5), (2,8), (3,11), (3,9), (5,11), (5,9), (8,9), (8,10)]
# sources = [1, 4, 6]
# destinations = [9, 10, 11]
#
# Graph = nx.DiGraph()
# Graph.add_nodes_from(nodes)
# Graph.add_edges_from(edges)
#
# # for s in sources:
# #     print s
# #     gen = findAllPaths(Graph, [s], destinations)
# #     print gen
#
#
# print findAllPathes(Graph, sources, destinations)


# hardcoded_designs = (
#         # "1.GroundSta@SUR1 2.GroundSta@SUR4 1.Sat@MEO1 2.Sat@MEO3 1.Sat@LEO1 2.Sat@LEO2",
#         # "1.GroundSta@SUR1 2.GroundSta@SUR4 1.Sat@GEO1 1.Sat@MEO1 2.Sat@MEO3 1.Sat@LEO1 2.Sat@LEO2",
#         "1.GroundSta@SUR1 2.GroundSta@SUR4 1.Sat@MEO1 1.Sat@MEO4 2.Sat@MEO5 1.Sat@LEO1 2.Sat@LEO2",
#         # "1.GroundSta@SUR1 2.GroundSta@SUR4 1.Sat@MEO1 1.Sat@MEO3 1.Sat@MEO4 2.Sat@MEO5 2.Sat@MEO6",
#         "1.GroundSta@SUR1 2.GroundSta@SUR4 2.Sat@GEO4 1.Sat@MEO1 1.Sat@MEO4 2.Sat@MEO5 1.Sat@LEO1 2.Sat@LEO2",
#         # "1.GroundSta@SUR1 2.GroundSta@SUR3 3.GroundSta@SUR5 2.Sat@GEO3 1.Sat@MEO1 2.Sat@MEO3 3.Sat@MEO6 1.Sat@LEO2",
#         "1.GroundSta@SUR1 2.GroundSta@SUR3 3.GroundSta@SUR5 1.Sat@MEO1 1.Sat@MEO2 2.Sat@MEO3 2.Sat@MEO5 3.Sat@MEO6",
#         "1.GroundSta@SUR1 2.GroundSta@SUR3 3.GroundSta@SUR5 3.Sat@GEO5 1.Sat@MEO1 1.Sat@MEO2 2.Sat@MEO3 2.Sat@MEO5 3.Sat@MEO6",
#         "1.GroundSta@SUR1 2.GroundSta@SUR3 3.GroundSta@SUR5 1.Sat@MEO1 2.Sat@MEO2 3.Sat@MEO5 1.Sat@LEO2 2.Sat@LEO4 3.Sat@LEO6",
#         # "1.GroundSta@SUR1 2.GroundSta@SUR3 3.GroundSta@SUR5 1.Sat@GEO1 1.Sat@MEO1 2.Sat@MEO4 3.Sat@MEO5 1.Sat@LEO2 2.Sat@LEO4 3.Sat@LEO6",
#     )
#

