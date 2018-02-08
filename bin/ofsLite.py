import socket
# print(socket.gethostbyname("localhost"))
import argparse
import itertools
import logging
# import pymongo
# from scoop import futures
import sys, os
import hashlib
import re
import pickle
from numpy import linspace
# import random

# add ofspy to system path
sys.path.append(os.path.abspath('..'))

from ofspy.ofsLite import OFSL
from ofspy.result import QResult
from ofspy.generalFunctions import *
import json

from multiprocessing import Process





def execute(dbHost, dbPort, experiment, start, stop, design, numPlayers, numTurns, fops, capacity, links):
    """
    Executes a general experiment.
    @param dbHost: the database host
    @type dbHost: L{str}
    @param dbPort: the database port
    @type dbPort: L{int}
    @param dbName: the database collection name
    @type dbName: L{str}
    @param start: the starting seed
    @type start: L{int}
    @param stop: the stopping seed
    @type stop: L{int}
    @param design: the list of designs to execute
    @type design: L{list}
    @param numPlayers: the number of players
    @type numPlayers: L{int}
    @param initialCash: the initial cash
    @type initialCash: L{int}
    @param numTurns: the number of turns
    @param numTurns: the number of turns
    @type numTurns: L{int}
    @param ops: the operations definition
    @type ops: L{str}
    @param fops: the federation operations definition
    @type fops: L{str}
    """
    # print "design:", design
    # print start, stop
    executions = [(dbHost, dbPort, experiment,
                   [e for e in elements.split(' ') if e != ''],
                   numPlayers, numTurns, seed, fops, capacity, links)
                  for (seed, elements) in itertools.product(range(start, stop), design)]
    numComplete = 0.0
    logging.info('Executing {} design with seeds from {} to {} for {} total executions.'
                 .format(len(design), start, stop, len(executions)))
    # for results in futures.map(queryCase, executions):
    # results = futures.map(queryCase, executions)
    # print(len(list(executions)))
    # print([list(e) for e in executions])
    # map(queryCase, executions)
    for execution in executions:
        argslist = list(execution)
        # print(argslist)
        queryCase(*argslist)
    # print "results :", results
    # N = len(results[0])
    # This line calculates the average of each element of each tuple for all the lists in the results, in other words assuming that each tuple of each results shows one seed of the same identity
    # print [[sum(x)/float(N) for x in zip(*l)] for l in [[l[j] for l in results] for j in range(N)]]

def queryCase(dbHost, dbPort, experiment, elements, numPlayers, numTurns, seed, fops, capacity, links):
    """
    Queries and retrieves existing results or executes an OFS simulation.
    @param dbHost: the database host
    @type dbHost: L{str}
    @param dbPort: the database port
    @type dbPort: L{int}
    @param dbName: the database collection name
    @type dbName: L{str}
    @param elements: the design specifications
    @type elements: L{list}
    @param numPlayers: the number of players
    @type numPlayers: L{int}
    @param initialCash: the initial cash
    @type initialCash: L{int}
    @param numTurns: the number of turns
    @type numTurns: L{int}
    @param seed: the random number seed
    @type seed: L{int}
    @param ops: the operations definition
    @type ops: L{str}
    @param fops: the federation operations definition
    @type fops: L{str}
    @return: L{list}
    """
    # print "elementlist:", elementlist
    # executeCase(elementlist, numPlayers, initialCash,
    #              numTurns, seed, ops, fops)
    # experiment = experiment
    m = hashlib.md5()

    # print(json.loads(fops), , ' '.join(fops))
    fops = json.loads(fops)
    des = ' '.join(elements)
    # print(experiment, des, ' '.join(fops), numTurns, seed, capacity, links)
    resultsstr = "%s %s %s %d %d %d %d"%(experiment, des, ' '.join(fops), numTurns, seed, capacity, links)
    print(resultsstr)

    ustr = resultsstr.encode('utf-16')
    m.update(ustr)

    filename = '../data/%s.p'%str(m.hexdigest())

    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            Q = pickle.load(f, encoding='latin1')

        if not isinstance(Q, str):
            print(type(Q))
            return None
        else:
            print(Q)
        # print "Found:\n", resultsstr
        # return None
    else:
        # pickle.dump("reserved", open("../data/%s.p" % str(m.hexdigest()), "wb"))
        pass

    return None
    query = {u'experiment': experiment,
             u'elementlist': des,
             u'fops': fops,
             u'numTurns': numTurns,
             u'seed': seed,
             u'capacity': capacity,
             u'links': links,
             }

    results, biddinglist, sharedlinks, pricelist, cashlist = executeCase(experiment, elements, numPlayers, numTurns, seed,
                                                              fops, capacity, links)

    federaratedcash = [e[1] for e in results]


    # resultObj = QResult(m.hexdigest(), query, federaratedcash, biddinglist, pricelist, sharedlinks, cashlist)
    resultObj = QResult(m.hexdigest(), query, federaratedcash, [], [], sharedlinks, cashlist)

    # print("hashid:", m.hexdigest())
    # print("prices:", len(resultObj.prices), [len(e) for e in resultObj.prices])#, resultObj.prices)
    # print("links:", len(resultObj.links), [len(e) for e in resultObj.links])#, resultObj.links)
    # print("biddings:", len(resultObj.biddings), [len(e) for e in resultObj.biddings])#, resultObj.biddings)
    # print("cashlist:", len(resultObj.cashlist), [len(e) for e in resultObj.biddings])#, resultObj.cashlist)
    # print("totalcash:", resultObj.totalcash)

    pickle.dump(resultObj, open("../data/%s.p"%str(m.hexdigest()), "wb"))

    return results


def executeCase(experiment, elements, numPlayers, numTurns, seed, fops, capacity, links):
    """
    Executes an OFS simulation.
    @param elements: the design specifications
    @type elements: L{list}
    @param numPlayers: the number of players
    @type numPlayers: L{int}
    @param initialCash: the initial cash
    @type initialCash: L{int}
    @param numTurns: the number of turns
    @type numTurns: L{int}
    @param seed: the random number seed
    @type seed: L{int}
    @param ops: the operations definition
    @type ops: L{str}
    @param fops: the federation operations definition
    @type fops: L{str}
    """
    # print "ofs-exp-vs elementlist: ", elementlist
    #
    # return OFSL(elementlist=elementlist, numPlayers=numPlayers, initialCash=initialCash, numTurns=numTurns, seed=seed, ops=ops, fops=fops).execute()
    ofsl = OFSL(experiment = experiment, elements=elements, numPlayers=numPlayers, numTurns=numTurns, seed=seed, fops=fops, capacity = capacity, links = links)
    return ofsl.execute()


def fopsGenStorage(costrange, storange, numplayers):
    # costSGLList = list(range(0, 1001, 200))
    # # costISLList = [c/2. for c in costSGLList]
    # storagePenalty = list(range(0, 1001, 200))+[-1]
    # yield numplayers * ["x%d,%d" % (-2, -2)]
    for sgl in costrange:
        for s in storange:
            # yield ["x%d,%d,%d"%(sgl, sgl, -2)] + (numplayers-1)*["x%d,%d,%d"%(sgl, sgl, s)]
            yield numplayers* ["x%d,%1.2f,%d"%(sgl, s, -1)]
            # yield numplayers* ["x%d,%d,%d"%(-3, s, -1)]

        # yield numplayers * ["x%d,%d,%d" % (sgl, -1, -1)]


def fopsGenAdaptive(costrange, numplayers):
    # yield numplayers * ["x%d,%d,%d" % (-2, -1, 1)]
    # yield numplayers * ["x%d,%d,%d" % (-2, -1, -1)]
    # yield numplayers * ["x%d,%d,%d" % (10, -1, -1)]
    for sgl in costrange:
        if sgl == -2:
            # yield numplayers * ["x%d,%d,%d" % (sgl, -1, -1)]
            yield numplayers * ["x%d,%d,%d" % (-2, -1, -1)]

        else:
            yield numplayers * ["x%d,%d,%d" % (sgl, -1, -1)]
            # if sgl == -3:
            #     print(["x%d,%d,%d"%(-2, -1, -1)] + (numplayers-1)*["x%d,%d,%d"%(sgl, -1, -1)])

            # for n in range(numplayers):
            #     fops = []
            #     fops.extend(n*["x%d,%d,%d"%(-2, -1, -1)])
            #     fops.extend(["x%d,%d,%d" % (sgl, -1, -1)])
            #     fops.extend((numplayers-n-1)*["x%d,%d,%d"%(-2, -1, -1)])
            #     # yield n*[] + ["x%d,%d,%d"%(-2, -1, -1)] + (numplayers-1)*["x%d,%d,%d"%(sgl, -1, -1)]
            #     yield fops

            for n in range(numplayers):
                fops = []
                fops.extend(n*["x%d,%d,%d"%(sgl, -1, -1)])
                fops.extend(["x%d,%d,%d"%(-2, -1, -1)])
                fops.extend((numplayers-n-1)*["x%d,%d,%d"%(sgl, -1, -1)])
                # yield n*[] + ["x%d,%d,%d"%(-2, -1, -1)] + (numplayers-1)*["x%d,%d,%d"%(sgl, -1, -1)]
                yield fops

            if numplayers>2:
                for n in range(numplayers):

                    fops = []
                    fops.extend(n*["x%d,%d,%d"%(-2, -1, -1)])
                    fops.extend(["x%d,%d,%d"%(sgl, -1, -1)])
                    fops.extend((numplayers-n-1)*["x%d,%d,%d"%(-2, -1, -1)])
                    # yield n*[] + ["x%d,%d,%d"%(-2, -1, -1)] + (numplayers-1)*["x%d,%d,%d"%(sgl, -1, -1)]
                    # print("new fops")
                    yield fops

            # yield 2 * ["x%d,%d,%d"%(-2, -1, -1)] + (numplayers - 2) * ["x%d,%d,%d"%(sgl, -1, -1)]
            # yield numplayers * ["x%d,%d,%d" % (sgl, -1, -1)]
            # yield numplayers * ["x%d,%d,%d"%(-2, -1, -1)]

        # yield 2*["x%d,%d" % (-2, -1)] + (numplayers-2)*["x%d,%d"%(sgl, -1)]

# def generateFops(costrange, storange):
#     fops = []
#     for cost in costrange:
#         costsgl = cost
#         costisl = cost
#
#         for sto in storange:
#             stopen = sto
#             for sto2 in storange:
#                 stopen2 = sto2
#                 yield ["x%d,%d,%d" % (costsgl, costisl, stopen2), "x%d,%d,%d" % (costsgl, costisl, stopen), "x%d,%d,%d" % (costsgl, costisl, stopen)]
# def fopsGenStorage(numPlayers):
#     yield numPlayers * ["x%d,%1.2f,%d" % (600, 400, -1)]
#     yield numPlayers * ["x%d,%1.2f,%d" % (600, 800, -1)]
#     yield numPlayers * ["x%d,%1.2f,%d" % (-3, 400, -1)]
#     yield numPlayers * ["x%d,%1.2f,%d" % (-3, 800, -1)]
#     for k in linspace(0., 1.99, 19):
#         yield numPlayers * ["x%d,%1.2f,%d" % (-3, -1*k, -1)]
#         yield numPlayers * ["x%d,%1.2f,%d" % (600, -1*k, -1)]

def runseed(seedlist, argsdict):
    global hardcoded_designs
    for seed in seedlist:
        argsdict['start'] = seed
        argsdict['stop'] = seed + 1
        for design in reversed(hardcoded_designs):
            # print design
            if '4.' in design:
                numPlayers = 4
            elif '3.' in design:
                numPlayers = 3
            else:
                numPlayers = 2

            argsdict['design'] = [design]
            costrange = [10, 600]
            storange = list([400, 800, -1])
            for fops in list(fopsGenAdaptive(costrange, numPlayers)):

                argsdict['experiment'] = 'Adaptive Cost V4'

                argsdict['fops'] = json.dumps(fops)
                argsdict['capacity'] = 2
                argsdict['links'] = 2
                execute(**argsdict)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="This program runs an OFS experiment.")
    # parser.add_argument('-e', help = 'experiment to run', type=str, nargs='+', default= 'Adaptive'
    #                     help='the experiment to run: adaptive, auctioneer')
    parser.add_argument('-d', '--numTurns', type=int, default=10000,
                        help='simulation duration (number of turns)')
    parser.add_argument('-p', '--numPlayers', type=int, default=None,
                        help='number of players')
    parser.add_argument('--cores', type=int, default=4,
                        help='cores on server')
    # parser.add_argument('-c', '--capacity', type=int, default=2.,
    #                     help='satellite capacity')
    # parser.add_argument('-l', '--links', type=int, default=2.,
    #                     help='links per edge')
    # parser.add_argument('-o', '--ops', type=str, default='d6',
    #                     help='federate operations model specification')
    parser.add_argument('-f', '--fops', type=str, default='',
                        help='federation operations model specification')
    # parser.add_argument('-l', '--logging', type=str, default='error',
    #                     choices=['debug', 'info', 'warning', 'error'],
    #                     help='logging level')
    parser.add_argument('-s', '--start', type=int, default=0,
                        help='starting random number seed')
    parser.add_argument('-t', '--stop', type=int, default=300,
                        help='stopping random number seed')
    parser.add_argument('--dbHost', type=str, default=None,
                        help='database host')
    parser.add_argument('--dbPort', type=int, default=27017,
                        help='database port')


    args = parser.parse_args()

        # count number of players

    numPlayers = args.numPlayers if 'numPlayers' in args else 2

    hardcoded_designs = (
        "1.GroundSta@SUR1 2.GroundSta@SUR4 1.Sat@MEO1 1.Sat@MEO4 2.Sat@MEO5 1.Sat@LEO1 2.Sat@LEO2",
        "1.GroundSta@SUR1 2.GroundSta@SUR4 2.Sat@GEO4 1.Sat@MEO1 1.Sat@MEO4 2.Sat@MEO5 1.Sat@LEO1 2.Sat@LEO2",
        "1.GroundSta@SUR1 2.GroundSta@SUR3 3.GroundSta@SUR5 1.Sat@MEO1 1.Sat@MEO2 2.Sat@MEO3 2.Sat@MEO5 3.Sat@MEO6",
        "1.GroundSta@SUR1 2.GroundSta@SUR3 3.GroundSta@SUR5 3.Sat@GEO5 1.Sat@MEO1 1.Sat@MEO2 2.Sat@MEO3 2.Sat@MEO5 3.Sat@MEO6",
        "1.GroundSta@SUR1 2.GroundSta@SUR3 3.GroundSta@SUR5 1.Sat@MEO1 2.Sat@MEO2 3.Sat@MEO3 1.Sat@LEO1 2.Sat@LEO2 3.Sat@LEO3",
    )

    argsdict = vars(args)
    np = argsdict['cores']
    argsdict.pop('cores')
    # print(np)
    for n in range(np):
        seedlist =  list(range(argsdict['start']+n, argsdict['stop'], np))
        Process(target=runseed, args=(seedlist, argsdict.copy())).start()

