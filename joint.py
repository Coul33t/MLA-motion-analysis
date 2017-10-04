class Joint():
    def __init__(self, name=None, positions=None, orientations=None, parent=None, childs=None):
        self.name = name
        self.channels = []
        self.positions = numpy.array(positions)
        self.orientations = numpy.array(orientations)
        self.parents = set()
        self.childs = set()

    def add_child(child):
        self.childs.add(child)

    def add_parent(parent):
        if(len(self.parents) == 0):
            self.parents.add(parent)
        else:
            print("Error: joint ({}) already has a parent ({}).".format(self.name, self.parent.name))

    def del_parent():
        self.parents.clear()

    def del_childs():
        self.childs.clear()