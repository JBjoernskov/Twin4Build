"""
Core module for the Twin4Build package.

This module re-exports the central classes of the package so they can be accessed
via ``twin4build.core`` without importing the individual submodules, and defines
namespace/ontology constants and small shared helpers.

Re-exported classes:
    - Model, SimulationModel, SemanticModel: Model interfaces (unified, simulation, semantic)
    - Semantic wrappers: SemanticEntity, SemanticObject, SemanticInstance,
      SemanticLiteral, SemanticProperty, SemanticType, SemanticPredicate
    - Simulator, Estimator: Simulation and parameter estimation
    - Translator and signature-pattern classes: SignaturePattern, Diff, StepRule,
      NoStepRule, SetStepRule, OptionalRule, PathRule, AnyPathRule
    - SAREF4SYST base classes: System, Connection, ConnectionPoint

Defined here:
    - BlankNode: Sentinel for matching untyped RDF blank nodes in signature patterns
    - sanitize_id / LEGAL_ID_CHARS: Component-id sanitization helper
    - namespace: rdflib Namespace constants (FSO, SAREF, S4BLDG, S4SYST, BRICK, XSD,
      T4B, RDF, RDFS, REC, OWL, FPO, BOT, SENAPS, BRICKREF)
    - ontology: URLs of the ontology files used for on-demand parsing
"""

# Third party imports
import rdflib


class BlankNode:
    """Sentinel used as a ``cls`` entry in :class:`Node` to match RDF resources
    that have **no** ``rdf:type`` assertion (untyped blank nodes).

    Usage::

        externalref = Node(cls=(BRICKREF.ExternalReference, BlankNode))

    Matches an instance typed as ``BRICKREF.ExternalReference`` *or* one with
    no type at all.
    """

    pass

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
    StepRule,
    NoStepRule,
    SetStepRule,
    OptionalRule,
    PathRule,
    AnyPathRule,
)

NoneType = type(None)

# Non-alphanumeric characters that are still legal in component IDs.
# Any character outside this set (and outside alphanumerics) will be replaced
# with ``_`` by :func:`sanitize_id`.
LEGAL_ID_CHARS = {"_", "-", " ", "(", ")", "[", "]"}


def sanitize_id(id_str: str) -> str:
    """Replace every character not allowed in a component ID with an underscore."""
    return "".join(c if c.isalnum() or c in LEGAL_ID_CHARS else "_" for c in id_str)


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
    BOT = rdflib.Namespace("https://w3id.org/bot#")
    SENAPS = rdflib.Namespace("http://senaps.io/schema/1.0/senaps#")
    BRICKREF = rdflib.Namespace("https://brickschema.org/schema/Brick/ref#")


class ontology:
    FSO = "https://alikucukavci.github.io/FSO/fso.ttl"
    SAREF = "https://saref.etsi.org/core/v3.1.1/"
    S4BLDG = "https://saref.etsi.org/saref4bldg/v1.1.2/"
    S4SYST = "https://saref.etsi.org/saref4syst/"
    BRICK = "https://brickschema.org/schema/1.4.1/Brick.ttl"
    T4B = "http://twin4build.org/"
    BOT = "http://www.w3id.org/bot/bot.ttl"
