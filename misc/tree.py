import random as rn

class Tree:
    def __init__(self):
        self.nodes = []
        self.root = []

    def add_node(self, n):

        if n not in self.nodes:
            self.nodes.append(n)

            if not n.parent:
                self.root = n

    def print_tree(self):
        print('Number of nodes: {}'.format(len(self.nodes)))

        for n in self.nodes:
            n.print_node()


class Node:
    def __init__(self, value=0):
        self.value = value
        self.parent = None
        self.childs = []

    def set_parent(self, p):
        self.parent = p
        p.add_child(self)

    def add_child(self, c):
        self.childs.append(c)

    def print_node(self):
        print('Value: {}'.format(self.value))

        if self.parent:
            print('Parent: {}'.format(self.parent.value))

        print('Childs: {}'.format(len(self.childs)))

        print('\n\n\n')


def DFS(n, to_display=None):
    if n.childs:
        print(n.value[to_display])
        for c in n.childs:
            DFS(c, to_display)

    else:
        print(n.value[to_display])


def BFS(n, to_display=None):
    if n.childs:
        if not n.parent:
            print(n.value[to_display])
        for c in n.childs:
            print(c.value[to_display])
        for c in n.childs:
            BFS(c, to_display)



def generate_tree():
    t = Tree()

    n1 = Node(value=1)
    t.add_node(n1)

    n2 = Node(value=2)
    n2.set_parent(n1)
    t.add_node(n2)


    n3 = Node(value=3)
    n3.set_parent(n1)
    t.add_node(n3)

    n4 = Node(value=4)
    n4.set_parent(n2)
    t.add_node(n4)

    n5 = Node(value=5)
    n5.set_parent(n2)
    t.add_node(n5)

    return t
if __name__ == '__main__':

    t = generate_tree()

    print("DFS:")
    DFS(t.root)

    print("BFS:")
    BFS(t.root)
