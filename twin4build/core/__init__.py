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
NoneType = type(None)
import rdflib

FSO = rdflib.Namespace("https://w3id.org/fso#")
SAREF = rdflib.Namespace("https://saref.etsi.org/core/")
S4BLDG = rdflib.Namespace("https://saref.etsi.org/saref4bldg/")
S4SYST = rdflib.Namespace("https://saref.etsi.org/saref4syst/")
XSD = rdflib.Namespace("http://www.w3.org/2001/XMLSchema#")
SIM = rdflib.Namespace("http://simulation.org/")

def get_ontologies():
    FSO = "https://alikucukavci.github.io/FSO/fso.ttl" # This is different from the FSO namespace defined above. The namespace defined above cannot be parsed (gives a 404 error)0.
    SAREF = "https://saref.etsi.org/core/"
    S4BLDG = "https://saref.etsi.org/saref4bldg/"
    S4SYST = "https://saref.etsi.org/saref4syst/"
    # namespaces = [FSO, SAREF, S4BLDG, S4SYST]
    namespaces = {"FSO": FSO, "SAREF": SAREF, "S4BLDG": S4BLDG, "S4SYST": S4SYST}
    sm = SemanticModel(namespaces=namespaces)
    return sm

ontologies = get_ontologies()

