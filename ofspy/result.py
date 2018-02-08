class QResult():
    def __init__(self, hashid, query, federatecash, biddings, prices, links, cashlist):
        self.hashid = hashid
        self.query = query
        self.federatecash = federatecash
        self.totalcash = sum(federatecash)
        self.biddings = biddings
        self.prices = prices
        self.links = links
        self.cashlist = cashlist
