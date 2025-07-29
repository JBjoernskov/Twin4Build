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

# Third party imports
import rdflib

# Local application imports
from twin4build.estimator.estimator import Estimator
from twin4build.model.model import Model
from twin4build.model.semantic_model.semantic_model import (
    SemanticModel,
    SemanticObject,
    SemanticProperty,
    SemanticType,
)
from twin4build.model.simulation_model.simulation_model import SimulationModel
from twin4build.simulator.simulator import Simulator
from twin4build.systems.saref4syst.connection import Connection
from twin4build.systems.saref4syst.connection_point import ConnectionPoint
from twin4build.systems.saref4syst.system import System
from twin4build.translator.translator import Translator

NoneType = type(None)


class namespace:
    FSO = rdflib.Namespace("https://w3id.org/fso#")
    SAREF = rdflib.Namespace("https://saref.etsi.org/core/")
    S4BLDG = rdflib.Namespace("https://saref.etsi.org/saref4bldg/")
    S4SYST = rdflib.Namespace("https://saref.etsi.org/saref4syst/")
    BRICK = rdflib.Namespace("https://brickschema.org/schema/Brick#")
    XSD = rdflib.Namespace("http://www.w3.org/2001/XMLSchema#")
    SIM = rdflib.Namespace("http://simulation.org/")
    RDF = rdflib.Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    RDFS = rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#")


class ontology:
    FSO = "https://alikucukavci.github.io/FSO/fso.ttl"
    SAREF = "https://saref.etsi.org/core/"
    S4BLDG = "https://saref.etsi.org/saref4bldg/"
    S4SYST = "https://saref.etsi.org/saref4syst/"
    BRICK = "https://brickschema.org/schema/1.4.1/Brick.ttl"
    XSD = "http://www.w3.org/2001/XMLSchema#"
    SIM = "http://simulation.org/"


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
    namespaces = {
        "FSO": namespace.FSO,
        "SAREF": namespace.SAREF,
        "S4BLDG": namespace.S4BLDG,
        "S4SYST": namespace.S4SYST,
        "BRICK": namespace.BRICK,
    }
    sm = SemanticModel(namespaces=namespaces)
    return sm


ontologies = get_ontologies()
