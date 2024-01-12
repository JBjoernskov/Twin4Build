import sys
import os
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 3)
    sys.path.append(file_path)

import twin4build.saref4syst.system as system
from itertools import count
class Node(system.System):
    instance_count = count()
    def __init__(self, cls=None, **kwargs):
        if "id" not in kwargs:
            kwargs["id"] = str(next(Node.instance_count))
        else:
            assert isinstance(id, str), "\"cls\" must be instance of class String"

        if cls is None:
            cls = ()
        else:
            assert isinstance(cls, tuple), "\"cls\" must be instance of class Tuple"
        super().__init__(**kwargs)
        self.cls = cls

class ContextSignature():
    def __init__(self):
        self.nodes = []
        self._edges = []
        self.input = {}
        self._inputs = []

    def add_edge(self, a, b):
        assert isinstance(a, Node) and isinstance(b, Node), "\"a\" and \"b\" must be instances of class Node"
        self.nodes.append(a)
        self.nodes.append(b)
        self._edges.append(f"{a.id} --> {b.id}")
        a.connectedBefore.append(b)
        b.connectedAfter.append(a)

    def add_input(self, key, node):
        self.input[key] = node
        self._inputs.append(f"{node.id} | {key}")

    def print_edges(self):
        print("")
        print("===== EDGES =====")
        for e in self._edges:
            print(f"     {e}")
        print("=================")

    def print_inputs(self):
        print("")
        print("===== INPUTS =====")
        print("  Node  |  Input")
        # print("_________________")
        for i in self._inputs:
            print(f"      {i}")
        print("==================")


def test():
    
    node1 = Node(cls=())
    node2 = Node(cls=())
    node3 = Node(cls=())
    cs = ContextSignature()
    cs.add_edge(node1, node2)
    cs.add_edge(node2, node3)
    cs.add_input("x1", node1)
    cs.add_input("x2", node2)
    cs.add_input("x3", node2)

    cs.print_edges()
    cs.print_inputs()
    

if __name__=="__main__":
    test()