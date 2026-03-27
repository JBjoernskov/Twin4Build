"""
Core module for Twin4Build package.

This module provides the fundamental components and ontologies used throughout the Twin4Build package.
It defines essential namespaces and provides access to the semantic model that underpins the digital twin functionality.

Key Components:
    - Namespaces (FSO, SAREF, S4BLDG, S4SYST, XSD, T4B): Core ontologies for building systems
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
    SemanticEntity,
    SemanticObject,
    SemanticInstance,
    SemanticLiteral,
    SemanticProperty,
    SemanticType,
    SemanticPredicate,
)
from twin4build.model.simulation_model.simulation_model import SimulationModel
from twin4build.simulator.simulator import Simulator
from twin4build.systems.saref4syst.connection import Connection
from twin4build.systems.saref4syst.connection_point import ConnectionPoint
from twin4build.systems.saref4syst.system import System
from twin4build.translator.translator import (
    Translator,
    SignaturePattern,
    Diff,
    Exact,
    Optional_,
    SinglePath,
    MultiPath,
)

NoneType = type(None)


class namespace:
    FSO = rdflib.Namespace("https://w3id.org/fso#")
    SAREF = rdflib.Namespace("https://saref.etsi.org/core/")
    S4BLDG = rdflib.Namespace("https://saref.etsi.org/saref4bldg/")
    S4SYST = rdflib.Namespace("https://saref.etsi.org/saref4syst/")
    BRICK = rdflib.Namespace("https://brickschema.org/schema/Brick#")
    XSD = rdflib.Namespace("http://www.w3.org/2001/XMLSchema#")
    T4B = rdflib.Namespace("http://twin4build.org/")
    RDF = rdflib.Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    RDFS = rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#")
    REC = rdflib.Namespace("https://w3id.org/rec#")
    OWL = rdflib.Namespace("http://www.w3.org/2002/07/owl#")
    FPO = rdflib.Namespace("https://w3id.org/fpo#")
    SENAPS = rdflib.Namespace("http://senaps.io/schema/1.0/senaps#")
    BRICKREF = rdflib.Namespace("https://brickschema.org/schema/Brick/ref#")


class ontology:
    FSO = "https://alikucukavci.github.io/FSO/fso.ttl"
    SAREF = "https://saref.etsi.org/core/v3.1.1/"
    S4BLDG = "https://saref.etsi.org/saref4bldg/v1.1.2/"
    S4SYST = "https://saref.etsi.org/saref4syst/"
    BRICK = "https://brickschema.org/schema/1.4.1/Brick.ttl"
    T4B = "http://twin4build.org/"


