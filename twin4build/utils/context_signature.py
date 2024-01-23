import sys
import os
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 3)
    sys.path.append(file_path)
import twin4build as tb
from twin4build.saref4syst.system import System
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.istype import istype
from itertools import count
from inspect import getmembers
import datetime
from dateutil import tz
class NodeBase:
    NODE_INSTANCE_COUNT = count()
    def __init__(self, cls=None, **kwargs):
        if cls is None:
            cls = ()
        else:
            assert isinstance(cls, tuple), "\"cls\" must be instance of class Tuple"
        self.cls = cls



def get_attributes(obj):
    if isinstance(obj, tuple):
        attributes = []
        for obj_ in obj:
            members = []
            for member in getmembers(obj_):
                if '__init__' in member:
                    members.append(member[1].__code__.co_names)
            # attributes_ = dir(obj_)
            # attributes_ = [attr for attr in attributes_ if attr[:2]!="__"]#Remove callables
            attributes.extend(members)
            
    else: 
        attributes = dir(obj)
        attributes = [attr for attr in attributes if attr[:2]!="__"]#Remove callables
    return attributes

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
        self.nodes.append(a)
        self.nodes.append(b)
        attributes_a = get_attributes(a)
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


def test():
    
    node1 = Node(cls=(tb.Fan, tb.Coil, tb.AirToAirHeatRecovery))
    node2 = Node(cls=(tb.Coil,))
    node3 = Node(cls=(tb.Pump,))
    node4 = Node(cls=(tb.Valve,))
    node5 = Node(cls=(tb.Valve,))
    # node6 = Node(cls=(tb.Valve))
    node7 = Node(cls=(tb.Controller,))
    node8 = Node(cls=(tb.OpeningPosition,))
    cs = ContextSignature()
    cs.add_edge(node1, node2, "connectedBefore")
    cs.add_edge(node3, node2, "connectedBefore")
    cs.add_edge(node2, node4, "connectedBefore")
    cs.add_edge(node4, node3, "connectedBefore")
    cs.add_edge(node2, node5, "connectedBefore")
    cs.add_edge(node5, node8, "hasProperty")
    cs.add_edge(node7, node8, "actuatesProperty")
    cs.add_input("airFlow", node1)
    cs.add_input("inletAirTemperature", node1)
    cs.add_input("supplyWaterTemperature", node2)
    cs.add_input("valvePosition", node5)

    cs.print_edges()
    cs.print_inputs()


    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "Twin4build-Case-Studies", "LBNL", "configuration_template_LBNL.xlsm")
    stepSize = 60
    startTime = datetime.datetime(year=2022, month=2, day=1, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen")) 
    endTime = datetime.datetime(year=2022, month=2, day=2, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))

    model = tb.Model(id="model", saveSimulationResult=True)
    model.load_model(infer_connections=False, semantic_model_filename=filename)

if __name__=="__main__":
    test()