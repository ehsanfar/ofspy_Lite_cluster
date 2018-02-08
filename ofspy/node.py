class Node():
    def __init__(self, element, graphorder):
        self.graphorder = graphorder
        self.element = element
        self.name = element.name + '.%d'%graphorder
        self.sector = None