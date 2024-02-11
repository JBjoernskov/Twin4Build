
from twin4build.saref4syst.system import System
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.istype import istype
from twin4build.utils.get_object_attributes import get_object_attributes
from itertools import count
from inspect import getmembers

class NodeBase:
    NODE_INSTANCE_COUNT = count()
    def __init__(self):
        pass

def Node(cls):
    cls = cls + (NodeBase, )
    class Node_(*cls):
        def __init__(self, cls=None, **kwargs):
            if any([issubclass(c, (System, )) for c in cls]):
                if "id" not in kwargs:
                    kwargs["id"] = str(next(NodeBase.NODE_INSTANCE_COUNT))
            else:
                self.id = str(next(NodeBase.NODE_INSTANCE_COUNT))
            self.cls = cls
            self.attributes = set()
            super().__init__(**kwargs)
    node = Node_(cls)
    return node

class ContextSignature():
    def __init__(self):
        self._nodes = []
        self._edges = []
        self.input = {}
        self._inputs = []
        self._modeled_nodes = set()

    @property
    def nodes(self):
        assert len(self._nodes)>0, "No nodes in the SignaturePattern. It must contain at least 1 node."
        return self._nodes

    @property
    def modeled_nodes(self):
        assert len(self._modeled_nodes)>0, "No nodes in the SignaturePattern has been marked as modeled. At least 1 node must be marked."
        return self._modeled_nodes

    def add_edge(self, a, b, relation):
        assert isinstance(a, NodeBase) and isinstance(b, NodeBase), "\"a\" and \"b\" must be instances of class Node"
        if a not in self._nodes:
            self._nodes.append(a)
        if b not in self._nodes:
            self._nodes.append(b)
        attributes_a = get_object_attributes(a)
        assert relation in attributes_a, f"The \"relation\" argument must be one of the following: {', '.join(attributes_a)} - \"{relation}\" was provided."
        attr = rgetattr(a, relation)
        if isinstance(attr, list):
            attr.append(b)
        else:
            rsetattr(a, relation, b)
        a.attributes.add(relation)
        self._edges.append(f"{a.id} ----{relation}---> {b.id}")

    def add_input(self, key, node):
        self.input[key] = node
        self._inputs.append(f"{node.id} | {key}")

    def add_modeled_node(self, node):
        self._modeled_nodes.add(node)

    def remove_modeled_node(self, node):
        self._modeled_nodes.remove(node)

    

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


