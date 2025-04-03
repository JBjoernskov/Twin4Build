import rdflib


from twin4build.utils.lazy_loader import LazyLoader
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



# Create a single lazy loader instance
# _lazy = LazyLoader()

# # Register lazy classes
# _lazy.add_lazy_class("System", "twin4build.systems.saref4syst.system", "System")
# _lazy.add_lazy_class("Connection", "twin4build.systems.saref4syst.connection", "Connection")
# _lazy.add_lazy_class("ConnectionPoint", "twin4build.systems.saref4syst.connection_point", "ConnectionPoint")
# _lazy.add_lazy_class("Model", "twin4build.model.model", "Model")
# _lazy.add_lazy_class("SemanticModel", "twin4build.model.semantic_model.semantic_model", "SemanticModel")
# _lazy.add_lazy_class("SemanticObject", "twin4build.model.semantic_model.semantic_model", "SemanticObject")
# _lazy.add_lazy_class("SemanticType", "twin4build.model.semantic_model.semantic_model", "SemanticType")
# _lazy.add_lazy_class("SemanticProperty", "twin4build.model.semantic_model.semantic_model", "SemanticProperty")
# _lazy.add_lazy_class("SimulationModel", "twin4build.model.simulation_model", "SimulationModel")
# _lazy.add_lazy_class("Simulator", "twin4build.simulator.simulator", "Simulator")
# _lazy.add_lazy_class("Monitor", "twin4build.monitor.monitor", "Monitor")
# _lazy.add_lazy_class("Estimator", "twin4build.estimator.estimator", "Estimator")
# _lazy.add_lazy_class("Evaluator", "twin4build.evaluator.evaluator", "Evaluator")
# _lazy.add_lazy_class("Translator", "twin4build.translator.translator", "Translator")

# # Export classes through the module
# System = _lazy.System
# Connection = _lazy.Connection
# ConnectionPoint = _lazy.ConnectionPoint
# Model = _lazy.Model
# SemanticModel = _lazy.SemanticModel
# SemanticObject = _lazy.SemanticObject
# SemanticType = _lazy.SemanticType
# SemanticProperty = _lazy.SemanticProperty
# SimulationModel = _lazy.SimulationModel
# Simulator = _lazy.Simulator
# Monitor = _lazy.Monitor
# Estimator = _lazy.Estimator
# Evaluator = _lazy.Evaluator
# Translator = _lazy.Translator


# from types import NoneType
NoneType = type(None)


FSO = rdflib.Namespace("https://w3id.org/fso#")
SAREF = rdflib.Namespace("https://saref.etsi.org/core/")
S4BLDG = rdflib.Namespace("https://saref.etsi.org/saref4bldg/")
S4SYST = rdflib.Namespace("https://saref.etsi.org/saref4syst/")
XSD = rdflib.Namespace("http://www.w3.org/2001/XMLSchema#")
        

def get_ontologies():
    FSO = "https://alikucukavci.github.io/FSO/fso.ttl" # This is different from the FSO namespace defined above. Using the namespace defined above gives a 404 error.
    SAREF = "https://saref.etsi.org/core/"
    S4BLDG = "https://saref.etsi.org/saref4bldg/"
    S4SYST = "https://saref.etsi.org/saref4syst/"
    namespaces = [FSO, SAREF, S4BLDG, S4SYST]
    sm = SemanticModel(namespaces=namespaces)
    return sm

# Create the lazy property at the module level
ontologies = get_ontologies()

