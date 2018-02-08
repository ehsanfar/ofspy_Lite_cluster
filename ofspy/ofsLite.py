
from .contextLite import ContextLite
import re


class OFSL(object):
    def __init__(self, experiment, elements, numPlayers,
                 numTurns, seed, fops, capacity, links):

        self.context = ContextLite()
        self.experiment = experiment
        self.time = 0
        self.initTime = 0
        self.maxTime = numTurns
        self.elements = elements
        # print("fops:", fops)
        # args = re.search('x(\d+),(\d+),([-v\d]+)', fops)
        fops = fops#json.loads(fops)
        self.costSGL = [int(re.search('x([-\d]+),([-\.\d]+),([-\d]+)', f).group(1)) for f in fops]
        self.costISL = self.costSGL
        self.storagePenalty = [float(re.search('x([-\d]+),([-\.\d]+),([-\d]+)', f).group(2)) for f in fops]
        self.auctioneer = int(re.search('x([-\d]+),([-\.\d]+),([-\d]+)', fops[0]).group(3)) == 1
        # print("auctioneer:", int(re.search('x([-\d]+),([-\.\d]+),([-\d]+)', fops[0]).group(3)), self.auctioneer)
        self.capacity = int(capacity)
        self.links = links
        self.seed = seed

        # print "OFSL elementlist:", elementlist

        self.context.init(self)
        # results = self.execute()


    def execute(self):
        """
        Executes an OFS.
        @return: L{list}
        """
        self.time = self.initTime
        for t in range(self.initTime, self.maxTime):
            # print self.time
            self.time = t
            self.context.ticktock(self)
            for f in self.context.federates:
                f.cashlist.append(f.cash)

        figs = []
        # print("figures")
        biddinglist = []
        sharedlinks = []
        pricelist = []
        cashlist = []

        federatelist = sorted(self.context.federates, key = lambda x: x.name)
        for i, f in enumerate(federatelist):
            tempsharedlinks = sorted(f.sharedlinks.items())

            sharedlinks.append(tempsharedlinks)
            cashlist.append(f.cashlist)
            if f.costlearning:
                biddinglist.append(f.costlist)
                # print("if auctioneer:", self.context.ofs.auctioneer)
                pricelist.append(f.timepricelist)
                # if not self.context.auctioneer:
                    # print(f.timepricelist, f.timepricelist)
            else:
                biddinglist.append([f.costDic['oSGL']])
    #         if not f.costlearning:
    #             continue
    #         if not plt.fignum_exists(i):
    #             plt.figure(i)
    #             plt.ion()
    #             plt.show()
    #
    #         plt.clf()
    #         # print(f.storagePenalty)
    #         costlist = f.costlist
    # #
    #         plt.plot(costlist)
    #         plt.draw()
    #         plt.show()
    #         plt.savefig("cost_volution_%s_storagepenalty_%d.png" % (f.name, self.storagePenalty[0]), bbox_inches='tight')

        #
        # for f in figs:
        #     plt.close(f)
        # print("number of finished tasks:", len(self.context.pickeduptasks))
        # print("total cash of task values:", self.context.totalcash)
        # print("sum of cash of federates:", sum([f.cash for f in self.context.federates]))
        #
        # if not plt.fignum_exists(1):
        #     plt.figure(1)
        #     plt.ion()
        #     plt.show()
        #
        # plt.clf()
        # figs = []
        # print("figures")
        # for i, f in enumerate(self.context.federates):
        #     if not plt.fignum_exists(i):
        #         figs.append(plt.figure(i))
        #         plt.ion()
        #         plt.show()
        #
        #     plt.clf()
        #     print(f.storagePenalty)
        #     if f.storagePenalty == -2:
        #         actionlist = f.qlearnerstorage.actionlist
        # #
        #         plt.plot(actionlist)
        #         plt.draw()
        #         plt.savefig("Evolution_%d_%s.png" % (sum(self.costSGL)/float(len(self.costSGL)), f.name), bbox_inches='tight')
        #
        # for f in figs:
        #     plt.close(f)

                # plt.waitforbuttonpress()

        # plt.show()
        # time.sleep(1)
        # plt.close()


        # self.context.Graph.drawGraphs()
        # for e in self.context.elementlist:
        #     print(e.name, len(e.savedTasks))

        results = []
        for f in self.context.federates:
            results.append((f.name, f.cash))
            # print("task duration and value dictionary and counter:")
            # print(f.taskduration)
            # print(f.taskvalue)
            # print(f.taskcounter)

        return sorted(results), biddinglist, sharedlinks, pricelist, cashlist







