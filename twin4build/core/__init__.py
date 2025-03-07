import rdflib

from twin4build.systems.saref4syst.system import System
from twin4build.systems.saref4syst.connection import Connection
from twin4build.systems.saref4syst.connection_point import ConnectionPoint

from twin4build.model.model import Model
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.model.semantic_model.semantic_model import SemanticObject
from twin4build.model.semantic_model.semantic_model import SemanticType
from twin4build.model.semantic_model.semantic_model import SemanticProperty
from twin4build.model.simulation_model import SimulationModel
from twin4build.simulator.simulator import Simulator
from twin4build.monitor.monitor import Monitor
from twin4build.estimator.estimator import Estimator
from twin4build.evaluator.evaluator import Evaluator
from twin4build.translator.translator import Translator

# from types import NoneType
NoneType = type(None)


FSO = rdflib.Namespace("https://w3id.org/fso#")
SAREF = rdflib.Namespace("https://saref.etsi.org/core/")
S4BLDG = rdflib.Namespace("https://saref.etsi.org/saref4bldg/")
S4SYST = rdflib.Namespace("https://saref.etsi.org/saref4syst/")
XSD = rdflib.Namespace("http://www.w3.org/2001/XMLSchema#")

def get_ontologies():
    sm = SemanticModel()
    FSO = "https://alikucukavci.github.io/FSO/fso.ttl"
    SAREF = "https://saref.etsi.org/core/"
    S4BLDG = "https://saref.etsi.org/saref4bldg/"
    S4SYST = "https://saref.etsi.org/saref4syst/"
    namespaces = [FSO, SAREF, S4BLDG, S4SYST]
    sm.parse_namespaces(sm.graph, namespaces)
    return sm

ontologies = get_ontologies()
