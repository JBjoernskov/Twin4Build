
from twin4build.saref4syst.system import System
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.istype import istype
from twin4build.utils.get_object_attributes import get_object_attributes
from itertools import count
from inspect import getmembers

class NodeBase:
    NODE_INSTANCE_COUNT = count()
    def __init__(self, cls=None, **kwargs):
        if cls is None:
            cls = ()
        else:
            assert isinstance(cls, tuple), "\"cls\" must be instance of class Tuple"
        self.cls = cls



def Node(cls):
    cls = cls + (NodeBase, )
    class Node_(*cls):
        def __init__(self, cls=None, **kwargs):
            if any([issubclass(c, (System, )) for c in cls]):
                if "id" not in kwargs:
                    kwargs["id"] = str(next(NodeBase.NODE_INSTANCE_COUNT))
            else:
                self.id = str(next(NodeBase.NODE_INSTANCE_COUNT))
            super().__init__(**kwargs)
    node = Node_(cls)
    return node

class ContextSignature():
    def __init__(self):
        self.nodes = []
        self._edges = []
        self.input = {}
        self._inputs = []


    def add_edge(self, a, b, relation):
        assert isinstance(a, NodeBase) and isinstance(b, NodeBase), "\"a\" and \"b\" must be instances of class Node"
        if a not in self.nodes:
            self.nodes.append(a)
        if b not in self.nodes:
            self.nodes.append(b)
        attributes_a = get_object_attributes(a)
        assert relation in attributes_a, f"The \"relation\" argument must be one of the following: {', '.join(attributes_a)} - \"{relation}\" was provided."
        rsetattr(a, relation, b)
        self._edges.append(f"{a.id} ----{relation}---> {b.id}")


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


