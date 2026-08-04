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
    - ontology: vendored local ontology files used for on-demand parsing
    - ontology_remote: original remote URLs (fallback + snapshot refreshing)
"""

# Standard library imports
import os

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


_ONTOLOGY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ontologies")


class ontology:
    """Local (vendored) ontology files used for on-demand parsing.

    Translation-time reasoning must not depend on the network: a failed
    ontology download silently degrades signature-pattern matching and drops
    components from the translated model (issue #114).  The vendored files in
    ``twin4build/core/ontologies/`` are therefore the primary source;
    :class:`ontology_remote` holds the original URLs used as fallback and for
    refreshing the snapshots.
    """

    FSO = os.path.join(_ONTOLOGY_DIR, "fso.ttl")
    SAREF = os.path.join(_ONTOLOGY_DIR, "saref.ttl")
    S4BLDG = os.path.join(_ONTOLOGY_DIR, "saref4bldg.ttl")
    S4SYST = os.path.join(_ONTOLOGY_DIR, "saref4syst.ttl")
    BRICK = os.path.join(_ONTOLOGY_DIR, "brick.ttl")
    BOT = os.path.join(_ONTOLOGY_DIR, "bot.ttl")
    RDF = os.path.join(_ONTOLOGY_DIR, "rdf.ttl")
    RDFS = os.path.join(_ONTOLOGY_DIR, "rdfs.ttl")
    OWL = os.path.join(_ONTOLOGY_DIR, "owl.ttl")
    REC = os.path.join(_ONTOLOGY_DIR, "rec.ttl")

    @classmethod
    def local_source(cls, namespace_uri) -> "str | None":
        """Vendored ontology file for a namespace URI (None if not vendored)."""
        return _LOCAL_BY_NAMESPACE.get(str(namespace_uri))


class ontology_remote:
    """Remote ontology URLs: fallback when a vendored file is missing, and the
    pinned sources for refreshing ``twin4build/core/ontologies/``."""

    FSO = "https://alikucukavci.github.io/FSO/fso.ttl"
    SAREF = "https://saref.etsi.org/core/v3.1.1/"
    S4BLDG = "https://saref.etsi.org/saref4bldg/v1.1.2/"
    S4SYST = "https://saref.etsi.org/saref4syst/"
    BRICK = "https://brickschema.org/schema/1.4.1/Brick.ttl"
    T4B = "http://twin4build.org/"
    BOT = "http://www.w3id.org/bot/bot.ttl"

    @classmethod
    def remote_source(cls, namespace_uri) -> "str | None":
        """Remote ontology URL for a namespace URI (None if unknown)."""
        return _REMOTE_BY_NAMESPACE.get(str(namespace_uri))


_LOCAL_BY_NAMESPACE = {
    str(namespace.FSO): ontology.FSO,
    str(namespace.SAREF): ontology.SAREF,
    str(namespace.S4BLDG): ontology.S4BLDG,
    str(namespace.S4SYST): ontology.S4SYST,
    str(namespace.BRICK): ontology.BRICK,
    str(namespace.BOT): ontology.BOT,
    str(namespace.RDF): ontology.RDF,
    str(namespace.RDFS): ontology.RDFS,
    str(namespace.OWL): ontology.OWL,
    str(namespace.REC): ontology.REC,
}

_REMOTE_BY_NAMESPACE = {
    str(namespace.FSO): ontology_remote.FSO,
    str(namespace.SAREF): ontology_remote.SAREF,
    str(namespace.S4BLDG): ontology_remote.S4BLDG,
    str(namespace.S4SYST): ontology_remote.S4SYST,
    str(namespace.BRICK): ontology_remote.BRICK,
    str(namespace.T4B): ontology_remote.T4B,
    str(namespace.BOT): ontology_remote.BOT,
}
