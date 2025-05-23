"""Core module for Twin4Build package.

This module provides the fundamental components and ontologies used throughout the Twin4Build package.
It defines essential namespaces and provides access to the semantic model that underpins the digital twin functionality.

Key Components:
    - Namespaces (FSO, SAREF, S4BLDG, S4SYST, XSD, SIM): Core ontologies for building systems
    - SemanticModel: The central model for managing building system semantics
    - Ontology Management: Functions for accessing and managing building system ontologies

The module integrates various building system ontologies including:
    - FSO (Facility Smart Objects)
    - SAREF (Smart Applications REFerence)
    - S4BLDG (SAREF for Building)
    - S4SYST (SAREF for System)
"""

from twin4build.systems.saref4syst.system import System
from twin4build.systems.saref4syst.connection import Connection
from twin4build.systems.saref4syst.connection_point import ConnectionPoint
from twin4build.model.model import Model
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.model.semantic_model.semantic_model import SemanticObject
from twin4build.model.semantic_model.semantic_model import SemanticType
from twin4build.model.semantic_model.semantic_model import SemanticProperty
from twin4build.model.simulation_model.simulation_model import SimulationModel
from twin4build.simulator.simulator import Simulator
from twin4build.estimator.estimator import Estimator
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
    """Retrieve and initialize the semantic model with required ontologies.
    
    This function initializes the semantic model with the following ontologies:
        - FSO (Facility Smart Objects)
        - SAREF (Smart Applications REFerence)
        - S4BLDG (SAREF for Building)
        - S4SYST (SAREF for System)
    
    Returns:
        SemanticModel: An initialized semantic model containing all required ontologies.
    
    Note:
        The FSO ontology URL is different from the namespace definition due to parsing limitations
        with the namespace URL.
    """
    FSO = "https://alikucukavci.github.io/FSO/fso.ttl"
    SAREF = "https://saref.etsi.org/core/"
    S4BLDG = "https://saref.etsi.org/saref4bldg/"
    S4SYST = "https://saref.etsi.org/saref4syst/"
    namespaces = {"FSO": FSO, "SAREF": SAREF, "S4BLDG": S4BLDG, "S4SYST": S4SYST}
    sm = SemanticModel(namespaces=namespaces)
    return sm

ontologies = get_ontologies()

