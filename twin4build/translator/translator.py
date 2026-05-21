from __future__ import annotations

# ---------------------------------------------------------------------------
# Rule taxonomy cheatsheet
# ---------------------------------------------------------------------------
# Register rules on a SignaturePattern with ``sp.add_rule(...)``
# (``sp.add_triple(...)`` is a deprecated alias).
#
#   Stem encodes topology:
#     Step  = one hop (a single (s, p, o) triple — the atom)
#     Path  = traversal of many hops (a sequence of steps)
#
#   Prefix encodes variation on the default (required, scalar) semantics:
#     No    = negation          (NoStepRule)
#     Set   = set-valued binding (SetStepRule)
#     Any   = any of several alternative paths accepted (AnyPathRule)
#     (none) = default: required, scalar branching
#
#   Modality decorator:
#     OptionalRule(inner=...) = conditional presence; wraps any rule
#
# | Class         | Topology    | Binding          | Presence |
# |---------------|-------------|------------------|----------|
# | StepRule      | one hop     | scalar, branches | required |
# | NoStepRule    | one hop     | none (veto)      | forbidden|
# | SetStepRule   | one hop     | tuple (set)      | required |
# | PathRule      | single path | scalar, branches | required |
# | AnyPathRule   | any of many | scalar, branches | required |
# | OptionalRule  | decorator   | inherited        | optional |
#
# See each class's docstring for asserts, matcher behavior, composition,
# and example usage. Deprecated names (``ExactRule``/``NoExactRule``/
# ``UniPathRule``/``MultiPathRule`` and older ``Exact``/``SinglePath``/
# ``MultiPath``/``Optional_``) remain available as aliases that emit a
# single :class:`DeprecationWarning` per level.
# ---------------------------------------------------------------------------

# Standard library imports
import collections
import hashlib
import inspect
import os
import time  # ##
import warnings
from dataclasses import dataclass
from itertools import count
from typing import Any, Dict, List, Optional, Tuple, Union

# Third party imports
import numpy as np
import torch

# import twin4build.saref4syst.system as system
# import twin4build.model.simulation_model as simulation_model
# import twin4build.model.semantic_model.semantic_model as semantic_model
import torch.nn as nn
from rdflib import Literal, URIRef
from scipy.optimize import Bounds, LinearConstraint, milp
from sympy import I

# Local application imports
import twin4build.core as core
import twin4build.systems as systems
import twin4build.utils.types as tps
from twin4build.utils.print_progress import LOGGER, autoreset_print
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rsetattr import rsetattr


# ---------------------------------------------------------------------------
# Matcher diagnostic dump (env-var gated, zero-cost when disabled).
#
# Enable with::
#
#     set TWIN4BUILD_MATCH_DIAG_FILE=matcher_diag.log
#     # optional: filter to one pattern id (substring match)
#     set TWIN4BUILD_MATCH_DIAG_PATTERN=controller_identification_vav
#
# Writes one line per decision in three scopes:
#   [PHASE1] per (sp_node, sm_node) enumeration start, and per complete/incomplete mapping
#   [BRCAST] per-element prune decisions inside __broadcast_recurse
#   [MERGE]  per merge attempt in _match (accept / reject + reason)
#
# The file is opened on first write and flushed per line so partial runs
# still produce useful output if the process is interrupted.
# ---------------------------------------------------------------------------
_MATCH_DIAG_PATH = os.environ.get("TWIN4BUILD_MATCH_DIAG_FILE")
_MATCH_DIAG_PATTERN_FILTER = os.environ.get("TWIN4BUILD_MATCH_DIAG_PATTERN")
_MATCH_DIAG_FH = None


def _match_diag_enabled(signature_pattern) -> bool:
    """Return True iff the diagnostic dump is enabled for this pattern."""
    if _MATCH_DIAG_PATH is None:
        return False
    if _MATCH_DIAG_PATTERN_FILTER is None:
        return True
    try:
        pid = getattr(signature_pattern, "id", "") or ""
    except Exception:
        pid = ""
    return _MATCH_DIAG_PATTERN_FILTER in pid


def _match_diag_write(line: str) -> None:
    """Append a line to the diagnostic file (opens on first call)."""
    global _MATCH_DIAG_FH
    if _MATCH_DIAG_PATH is None:
        return
    if _MATCH_DIAG_FH is None:
        _MATCH_DIAG_FH = open(_MATCH_DIAG_PATH, "w", encoding="utf-8")
        _MATCH_DIAG_FH.write(
            "# twin4build matcher diagnostic dump\n"
            f"# filter: pattern~={_MATCH_DIAG_PATTERN_FILTER!r}\n"
        )
    _MATCH_DIAG_FH.write(line)
    if not line.endswith("\n"):
        _MATCH_DIAG_FH.write("\n")
    _MATCH_DIAG_FH.flush()


def _diag_sm_name(obj) -> str:
    """Best-effort short name for an SM object (scalar or tuple)."""
    if obj is None:
        return "None"
    if isinstance(obj, (tuple, list)):
        if not obj:
            return "()"
        inner = [_diag_sm_name(o) for o in obj[:3]]
        suffix = ", ..." if len(obj) > 3 else ""
        return "(" + ", ".join(inner) + suffix + f")[{len(obj)}]"
    for attr in ("get_short_name",):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    uri = getattr(obj, "uri", None)
    if uri is not None:
        s = str(uri)
        return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    return str(obj)


def _diag_mapping_summary(mapping: Dict[Any, Any]) -> str:
    """Compact ``sp_id=sm_name`` summary of a partial mapping."""
    parts = []
    for sp_n, sm_n in mapping.items():
        if sm_n is None:
            continue
        try:
            sp_id = sp_n.id
        except Exception:
            sp_id = str(sp_n)
        parts.append(f"{sp_id}={_diag_sm_name(sm_n)}")
    return "{" + ", ".join(parts) + "}"


@autoreset_print
class Translator:
    r"""
    Class for ontology-driven automated model generation and calibration in building energy systems.

    Args:
        sim2sem_map: Dictionary mapping simulation model components to semantic model instances
        sem2sim_map: Dictionary mapping semantic model instances to simulation model components
        instance_to_group_map: Dictionary mapping simulation model components to their corresponding signature pattern groups

    This class implements a general methodology for translating semantic models of building systems into executable simulation models, as described in:

        Jakob Bjørnskov, Muhyiddine Jradi, Michael Wetter, "Automated model generation and parameter estimation of building energy models using an ontology-based framework," Energy and Buildings, Volume 329, 2025, 115228. https://doi.org/10.1016/j.enbuild.2024.115228

    Overview
    --------
    The Translator enables the automated generation and calibration of building energy simulation models by leveraging semantic models and a library of reusable component models. The approach is based on the following key concepts:

    - **Semantic Models**: Structured, machine-readable representations of building systems, including topology, equipment, and sensor placement, based on ontologies such as SAREF, SAREF4BLDG, SAREF4SYST, and FSO.
    - **Component Model Library**: Modular simulation components (e.g., fans, coils, controllers) each defined with a signature pattern that describes the semantic context in which the model applies.
    - **Signature Patterns**: Generalized graph patterns (subject-predicate-object triples) that specify how component models map to semantic model instances, including rules for optionality and traversal.
    - **Automated Model Generation**: The Translator searches the semantic model for matches to signature patterns, instantiates the corresponding component models, and connects them to form a complete simulation model.

    Pattern Matching Process
    ------------------------
    The core of the Translator is the pattern matching process, which identifies how signature patterns map to semantic model instances. This process involves:

    1. **Graph Representation**: Both the semantic model and signature patterns are represented as directed graphs with labeled nodes and edges.
    2. **Pattern Matching**: The Translator searches for subgraph isomorphisms between signature patterns and the semantic model.
    3. **Rule Application**: Different types of rules (Exact, SinglePath, MultiPath, Optional) determine how pattern elements map to semantic model elements.

    .. figure:: /_static/translator_semantic_model.png
       :alt: System overview showing components and their relationships
       :align: center
       :width: 60%

       **Example of a semantic model**: This diagram shows the relationships between various components in a building system, including fans, coils, sensors, meters, valves, and pumps. The different line styles represent different types of relationships (suppliesFluidTo, observes, hasValue, etc.).

    .. figure:: /_static/translator_signature_patterns.png
       :alt: Signature patterns showing different component configurations
       :align: center
       :width: 50%

       **Example of signature patterns**: This diagram illustrates five distinct patterns (p1-p5) of interconnected components, each representing different configurations or sub-systems within a larger model. The patterns show how generic component types (Fan, Sensor, Coil, etc.) can be arranged in different ways to match various system configurations.

    .. figure:: /_static/translator_pattern_matching.png
       :alt: Pattern matching process showing how signatures map to system components
       :align: center
       :width: 50%

       **Example of pattern matching**: This diagram shows how signature patterns are matched against the semantic model. The central graph represents the actual system components, while the surrounding "Match of signature pX" blocks show how generic pattern elements (n₁, n₂, etc.) map to specific system components. The dotted lines connect pattern elements to their corresponding system instances.

    Methodology
    -----------
    1. **Pattern Matching**: Signature patterns are matched against the semantic model using a graph search algorithm, identifying all valid contexts for each component model.
    2. **Model Instantiation**: For each match, the corresponding component model is instantiated and mapped to the relevant semantic model instances.
    3. **Model Assembly**: Components are connected according to the relationships defined in the semantic model and signature patterns, resulting in an executable simulation model.

    Mathematical Formulation
    -----------------------
    The task of searching for signature patterns in the semantic model is formulated as a subgraph isomorphism problem:

    Given the pattern signature represented by the graph :math:`p = (V_p, E_p, L_p)` and the semantic model represented by the graph :math:`G = (V_G, E_G, L_G)`, find the map :math:`f: V_p \rightarrow V_G` such that:

    .. math::

        L_G(f(u)) \subseteq L_p(u) \quad \forall u \in V_p

    .. math::

        L_p(u, v) = L_G(f(u), f(v)) \quad \forall (u, v) \in E_p

    .. math::

        (f(u), f(v)) \subseteq E_G \quad \forall (u, v) \in E_p

    Where:
      - :math:`L_G(f(u)) \subseteq L_p(u)` requires that the node label (ontology class) of the semantic model is a subset of the pattern node label
      - :math:`L_p(u, v) = L_G(f(u), f(v))` ensures that the edge label (ontology predicate) of the semantic model matches the pattern edge label
      - :math:`(f(u), f(v)) \subseteq E_G` ensures that the mapped pattern edge also exists in the semantic model

    For each match found, a map :math:`f_i` is generated, and the corresponding component model is instantiated.

    Rule Types
    ----------
    The Translator supports several types of rules for pattern matching:

    - **Exact**: Requires exact matches between pattern and semantic model elements
    - **SinglePath**: Allows traversal along a single path in the semantic model
    - **MultiPath**: Allows traversal along multiple paths in the semantic model
    - **Optional**: Makes pattern elements optional (may or may not be present)

    These rules are combined to create flexible signature patterns that can match various system configurations while maintaining the integrity of the model structure.

    Examples
    --------
    >>> import twin4build as tb
    >>> sem_model = tb.SemanticModel("path/to/semantic_model.ttl") # or web address
    >>> translator = tb.Translator()
    >>> sim_model = translator.translate(sem_model)
    >>> sim_model.visualize()

    """

    @property
    def sim2sem_map(self):
        return self._sim2sem_map

    @property
    def sem2sim_map(self):
        return self._sem2sim_map

    def __init__(self):
        self._sim2sem_map = {}
        self._sem2sim_map = {}
        self._instance_to_group_map = {}

    def translate(
        self,
        semantic_model: core.SemanticModel,
        systems_: List[core.System] = None,
        verbose=4,
        *,
        id: Optional[str] = None,
    ) -> "core.Model":
        """
        Translate a semantic model into a :class:`~twin4build.model.model.Model`.

        The translator produces three things that the caller needs:

        1. a runnable :class:`SimulationModel` (the computation graph);
        2. the input :class:`SemanticModel` (the ontology, unchanged); and
        3. the bidirectional ``sim2sem`` / ``sem2sim`` maps that link them.

        Returning a :class:`Model` keeps all three together in the single
        type designed to hold them, so downstream code (sensor unit
        conversion by Brick class, serialisation, visualisation, …) can
        cross between semantic and simulation sides without reaching into
        translator internals.

        Args:
            semantic_model: The semantic model to translate.
            systems_: List of system types to match against. ``None`` selects
                every ``core.System`` subclass with a ``.sp`` signature pattern.
            verbose: Verbosity level forwarded to the LOGGER.
            id: Optional id for the produced :class:`Model`. When ``None`` the
                model inherits ``semantic_model.id``. Useful when one semantic
                model is translated multiple times with different ``systems_``
                lists (e.g. a controls-only stage and a full physics stage):
                pass distinct ids so the resulting Models do not share
                ``generated_files/models/<id>/`` directories.

        Returns:
            A :class:`Model` wrapping the produced simulation graph, the
            input semantic model, and the translator (which carries the
            ``sim2sem`` / ``sem2sim`` maps).
        """
        LOGGER.verbose = verbose
        LOGGER.task("Applying translator")
        LOGGER.add_level()
        if semantic_model.count_triples() == 0:
            raise Exception(
                "Semantic model provided to translator appears to be empty."
            )

        if systems_ is None:
            systems_ = [
                cls[1]
                for cls in inspect.getmembers(systems, inspect.isclass)
                if (issubclass(cls[1], (core.System,)) and hasattr(cls[1], "sp"))
            ]

        # Match patterns
        complete_groups, incomplete_groups = self._match_patterns(
            systems_=systems_,
            semantic_model=semantic_model,
        )

        if len(complete_groups) > 0:
            # LOGGER("Found following matching candidate patterns:")

            for component_cls in complete_groups.keys():
                LOGGER.section("Class: %s", component_cls.__name__)
                LOGGER.add_level()

                for sp in complete_groups[component_cls].keys():
                    LOGGER.section(
                        "Signature pattern: %s, %d matches found",
                        sp.id,
                        len(complete_groups[component_cls][sp]),
                    )
                    LOGGER.add_level()
                    for i, group in enumerate(complete_groups[component_cls][sp]):
                        LOGGER.section("Group %d", i)
                        LOGGER.add_level()

                        for sp_subject, sm_subject in group.items():
                            # id_sp = str([str(s) for s in sp_subject.cls])
                            id_sp = sp_subject.id
                            # ``sm_subject`` may be a tuple under
                            # SetStepRule; ``_binding_short_name`` handles
                            # both scalar and tuple shapes.
                            id_m = Translator._binding_short_name(sm_subject)
                            LOGGER.result("%s: %s", id_sp, id_m)
                        LOGGER.remove_level()
                        LOGGER.ok("Group %d", i, change_status=True)
                    LOGGER.remove_level()
                    LOGGER.ok(
                        "Signature pattern: %s, %d matches found",
                        sp.id,
                        len(complete_groups[component_cls][sp]),
                        change_status=True,
                    )

                for sp in incomplete_groups[component_cls].keys():
                    LOGGER.section(
                        "Incomplete signature patterns: %s, %d",
                        sp.id,
                        len(incomplete_groups[component_cls][sp]),
                    )
                    LOGGER.add_level()
                    for i, group in enumerate(incomplete_groups[component_cls][sp]):
                        LOGGER.section("Group %d", i)
                        LOGGER.add_level()

                        for sp_subject, sm_subject in group.items():
                            # id_sp = str([str(s) for s in sp_subject.cls])
                            id_sp = sp_subject.id
                            id_m = Translator._binding_short_name(sm_subject)
                            LOGGER.result("%s: %s", id_sp, id_m)
                        LOGGER.remove_level()
                        LOGGER.ok("Group %d", i, change_status=True)
                    LOGGER.remove_level()
                    LOGGER.ok(
                        "Incomplete signature patterns: %s, %d",
                        sp.id,
                        len(incomplete_groups[component_cls][sp]),
                        change_status=True,
                    )

                LOGGER.remove_level()
                LOGGER.ok("Class: %s", component_cls.__name__, change_status=True)

        else:
            raise Exception("No matching patterns found.")

        # Create component instances
        self._instantiate_components(complete_groups, semantic_model)

        if len(self._sim2sem_map) == 0:
            raise Exception("No components instantiated.")

        result = self._solve_milp()
        if result["success"]:
            model_id = id if id is not None else semantic_model.id
            # Initialize simulation model
            sim_model = core.SimulationModel(id=model_id)

            # Connect components
            self._connect_components(result["connections"], sim_model)
            LOGGER.remove_level()
            LOGGER.ok("Applying translator", change_status=True)
        else:
            LOGGER.remove_level()
            LOGGER.error("Applying translator", change_status=True)
            sim_model = None
            raise Exception(f"MILP solver failed: {result['message']}")

        return core.Model.from_translation(
            id=model_id,
            semantic_model=semantic_model,
            simulation_model=sim_model,
            translator=self,
        )

    @staticmethod
    def _match_patterns(
        systems_: List[core.System], semantic_model: core.SemanticModel
    ) -> Tuple[Dict, Dict]:
        """
        Find all valid mappings from signature patterns to semantic model nodes.

        This implements a subgraph isomorphism algorithm that matches each component's
        signature pattern (a template graph) against the semantic model (the actual
        building system graph). The algorithm finds all ways that pattern nodes can
        be mapped to semantic model nodes while preserving the required relationships.

        Algorithm Overview:
        -------------------
        1. For each component class that defines signature patterns:
           - Iterate through each signature pattern (SP)

        2. For each SP node, find candidate semantic model (SM) nodes of matching type

        3. Use depth-first search (`_prune_recursive`) to validate mappings:
           - Traverse both graphs simultaneously
           - Apply pattern rules (Exact, SinglePath, MultiPath, Optional_)
           - Prune branches where relationships don't match

        4. Categorize matches as complete or incomplete:
           - Complete: All required SP nodes have SM matches
           - Incomplete: Only partial mappings found

        5. Attempt to merge incomplete groups:
           - Two partial mappings might combine into a complete match
           - Continue merging until no more combinations possible

        Key Data Structures:
        --------------------
        - sp_sm_map: Dict mapping SP nodes → SM nodes (the node mapping)
        - complete_groups: Mappings where all required nodes are matched
        - incomplete_groups: Partial mappings that might be merged
        - feasible: Tracks which SM nodes are feasible matches for each SP node
        - comparison_table: Tracks which SM nodes have been compared to each SP node

        Args:
            systems_: List of system classes that may have signature patterns defined
            semantic_model: The semantic model (graph) to match patterns against

        Returns:
            Tuple of (complete_groups, incomplete_groups) where each is a nested dict:
            {ComponentClass: {SignaturePattern: [list of sp_sm_map dicts]}}
        """
        def _match_single_pattern(
            component_cls, signature_pattern, complete_groups, incomplete_groups
        ):
            """
            Match a single signature pattern against the semantic model.

            This nested function handles the core matching logic for one pattern:
            1. Find candidate SM nodes for each SP node
            2. Recursively validate mappings
            3. Categorize as complete or incomplete
            4. Attempt to merge incomplete groups
            """
            # Initialize result containers for this pattern
            complete_groups[component_cls][signature_pattern] = []
            incomplete_groups[component_cls][signature_pattern] = []
            complete_matches = complete_groups[component_cls][signature_pattern]
            incomplete_matches = incomplete_groups[component_cls][signature_pattern]

            # ===================================================================
            # PHASE 1: Find candidate mappings using depth-first search
            # ===================================================================
            for sp_node in signature_pattern.nodes:
                candidate_sm_nodes = semantic_model.get_instances_of_type(sp_node.cls)

                for sm_node in candidate_sm_nodes:

                    # Initialize tracking structures for this DFS traversal
                    initial_map = {n: None for n in signature_pattern.nodes}
                    feasible = {n: set() for n in signature_pattern.nodes}
                    comparison_table = {n: set() for n in signature_pattern.nodes}
                    candidate_maps = [Translator._copy_nodemap(initial_map)]

                    # Skip if already compared (comparison_table is empty initially)
                    if sm_node not in comparison_table[sp_node]:
                        # signature_pattern.reset_ruleset()
                        _diag_p1 = _match_diag_enabled(signature_pattern)
                        if _diag_p1:
                            _match_diag_write(
                                f"[PHASE1] start pattern={signature_pattern.id} "
                                f"sp_node={sp_node.id} sm_node={_diag_sm_name(sm_node)}"
                            )
                        candidate_maps, _, _, is_pruned = Translator._prune_recursive(
                            sm_node,
                            sp_node,
                            candidate_maps,
                            feasible,
                            comparison_table,
                            signature_pattern,
                        )

                        # ===================================================================
                        # PHASE 2: Process valid (non-pruned) mappings
                        # ===================================================================
                        if not is_pruned:

                            # ===================================================================
                            # PHASE 3: Categorize as complete or incomplete
                            # ===================================================================
                            # Flatten the grouped mappings back to individual mappings
                            # candidate_maps_flattened = []
                            # for group in candidate_maps:
                            #     candidate_maps_flattened.extend(group)
                            # candidate_maps = candidate_maps_flattened

                            for mapping in candidate_maps:
                                is_complete = all(
                                    mapping[n] is not None
                                    for n in signature_pattern.required_nodes
                                )

                                if is_complete:
                                    if _diag_p1:
                                        _match_diag_write(
                                            f"[PHASE1]   COMPLETE from "
                                            f"sp_node={sp_node.id} "
                                            f"sm_node={_diag_sm_name(sm_node)} "
                                            f"mapping={_diag_mapping_summary(mapping)}"
                                        )
                                    LOGGER.info(
                                        "Match found: %s",
                                        signature_pattern.id,
                                    )
                                    LOGGER.add_level()
                                    LOGGER.info(
                                        lambda: Translator._get_maps_string(
                                            [mapping], LOGGER.info
                                        )
                                    )
                                    LOGGER.remove_level()
                                    complete_matches.append(mapping)
                                else:
                                    if _diag_p1:
                                        _match_diag_write(
                                            f"[PHASE1]   INCOMPLETE from "
                                            f"sp_node={sp_node.id} "
                                            f"sm_node={_diag_sm_name(sm_node)} "
                                            f"mapping={_diag_mapping_summary(mapping)}"
                                        )

                                    incomplete_matches = (
                                        Translator._try_merge_with_incomplete(
                                            mapping,
                                            incomplete_matches,
                                            complete_matches,
                                            signature_pattern,
                                        )
                                    )
                        elif _diag_p1:
                            _match_diag_write(
                                f"[PHASE1]   PRUNED from sp_node={sp_node.id} "
                                f"sm_node={_diag_sm_name(sm_node)}"
                            )

            # ===================================================================
            # PHASE 4: Merge incomplete groups with each other
            # ===================================================================
            incomplete_matches = Translator._merge_incomplete_groups(
                incomplete_matches,
                complete_matches,
                signature_pattern,
            )

            # ===================================================================
            # PHASE 5: Merge isolated optional-only groups with complete matches
            # ===================================================================
            # Isolated optional nodes (e.g. a floating weather-station sensor with no
            # structural triple connecting it to the main graph) produce incomplete groups
            # whose required nodes are all None.  They can never be merged by Phase 4
            # because all space/AHU groups are already complete by the time they appear.
            # Here we augment every complete match with whatever optional values these
            # isolated groups provide, then discard the isolated groups.
            #
            # Compatibility check: before transferring an optional ``sp_node ->
            # sm_node`` binding into a complete_group, verify that every rule
            # in the pattern's ruleset that mentions ``sp_node`` is still
            # satisfied given the complete_group's existing bindings.  Without
            # this check, an optional binding harvested from one structural
            # context (e.g. ``AHU01 hasPoint AHU01.Supply_Air_Temp_Setpoint``)
            # would silently leak into a complete_group rooted in a different
            # context (e.g. ``ahu = AHU02``), wiring AHU02's
            # ``supplyAirTemperatureSetpoint`` from AHU01's setpoint sensor.
            # That cross-contamination is the original "Issue #4" called out
            # in the AHU pattern comment block.
            for isolated_group in list(incomplete_matches):
                # An empty tuple binding (``()``) is produced by an
                # ``OptionalRule`` wrapping a ``SetStepRule`` that matched
                # nothing. Treat it as absent for required-node checks.
                if any(
                    len(Translator._iter_binding(isolated_group.get(n))) > 0
                    for n in signature_pattern.required_nodes
                ):
                    continue  # Still has unfilled required nodes — not purely optional
                for i, complete_group in enumerate(complete_matches):
                    merged = dict(complete_group)
                    for sp_node, sm_node in isolated_group.items():
                        if not (
                            len(Translator._iter_binding(sm_node)) > 0
                            and len(Translator._iter_binding(merged.get(sp_node))) == 0
                        ):
                            continue
                        if not Translator._optional_binding_compatible(
                            sp_node, sm_node, merged, signature_pattern
                        ):
                            continue
                        merged[sp_node] = sm_node
                    complete_matches[i] = merged
                incomplete_matches.remove(isolated_group)

            # ===================================================================
            # PHASE 6: Canonicalise set-bound tuples + deduplicate
            # ===================================================================
            # Two cleanup passes that must run in this order:
            #
            # 1. Canonicalisation. Each complete_match must satisfy the
            #    invariant "every element of every set-bound tuple has the
            #    required downstream edges to the scalars already bound in
            #    the same mapping". ``_match``'s dict-union merge strategy
            #    does not re-check this — if an upstream phase left a raw
            #    ``SetStepRule`` tuple in either operand (e.g. a VAV-start
            #    broadcast that filtered Heating_Mode via consensus on the
            #    scalar descendants but left the tuple itself unfiltered
            #    on a parallel incomplete branch), the merge carries the
            #    corruption forward. Filtering here is idempotent: tuples
            #    that already obey the invariant are untouched.
            #
            # 2. Deduplication. Phase-1 DFS starts from every (sp_node,
            #    sm_node) pair, so patterns whose nodes are mutually
            #    reachable via inverse predicates produce the same
            #    canonical mapping from several starts. Running dedup
            #    AFTER canonicalisation collapses the "same real match,
            #    different accidental tuple contents" case into one.
            canonicalised: List[Dict[Any, Any]] = []
            for m in complete_matches:
                canon = Translator._filter_set_bound_tuples(m, signature_pattern)
                if canon is not None:
                    canonicalised.append(canon)

            seen: set = set()
            deduped: List[Dict[Any, Any]] = []
            for m in canonicalised:
                key = Translator._canonical_mapping_key(m)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(m)
            if len(deduped) != len(complete_matches):
                LOGGER.debug(
                    "Canonicalised+deduped complete_matches for %s: %d -> %d",
                    signature_pattern.id,
                    len(complete_matches),
                    len(deduped),
                )
            complete_matches[:] = deduped

        # ===================================================================
        # MAIN LOOP: Process each component class and its patterns
        # ===================================================================
        complete_groups = {}
        incomplete_groups = {}

        # Filter to classes that have signature patterns defined
        classes_with_patterns = [
            cls for cls in systems_ if hasattr(cls, "sp") and cls.sp is not None
        ]

        for component_cls in classes_with_patterns:
            LOGGER.task("Processing component class: %s", component_cls.__name__)
            LOGGER.add_level()
            complete_groups[component_cls] = {}
            incomplete_groups[component_cls] = {}

            for signature_pattern in component_cls.sp:
                LOGGER.task("Matching signature pattern: %s", signature_pattern.id)
                LOGGER.add_level()
                # Ensure semantic model has all namespaces from the pattern
                semantic_model.add_namespaces(
                    signature_pattern.semantic_model.namespaces
                )

                # Match this pattern against the semantic model
                _match_single_pattern(
                    component_cls, signature_pattern, complete_groups, incomplete_groups
                )

                # NOTE: Equivalent pattern matching is disabled but preserved for future use.
                # This would allow falling back to alternative patterns if the main one fails.
                # if len(complete_groups[component_cls][signature_pattern]) == 0:
                #     for equivalent_pattern in signature_pattern.has_equivalent:
                #         _match_single_pattern(equivalent_pattern, complete_groups, incomplete_groups)
                #         if len(complete_groups[component_cls][equivalent_pattern]) > 0:
                #             # Apply semantic model transformations defined by the equivalent pattern
                #             for eq_group in complete_groups[component_cls][equivalent_pattern]:
                #                 new_node_map = equivalent_pattern.apply_changes(semantic_model, eq_group)
                #                 complete_groups[component_cls][signature_pattern].append(new_node_map)
                LOGGER.remove_level()
                LOGGER.ok(
                    "Matching signature pattern: %s",
                    signature_pattern.id,
                    change_status=True,
                )

            LOGGER.remove_level()
            LOGGER.ok(
                "Processing component class: %s",
                component_cls.__name__,
                change_status=True,
            )

        return complete_groups, incomplete_groups

    @staticmethod
    def _optional_binding_compatible(
        sp_node: "Node",
        sm_node: Any,
        complete_group: Dict["Node", Any],
        signature_pattern: "SignaturePattern",
    ) -> bool:
        """Decide whether an optional ``sp_node -> sm_node`` binding from an
        isolated incomplete group can safely be merged into ``complete_group``.

        For every rule in the pattern's ruleset that mentions ``sp_node``,
        verify that the proposed ``sm_node`` is consistent with the
        complete_group's existing bindings at the rule's other endpoint:

        * If ``sp_node`` is the rule's *subject* and the rule's object is
          already bound in ``complete_group``, the SM-side triple
          ``sm_node --pred--> complete_group[object]`` must exist for at
          least one predicate alternative in ``rule.predicate.preds``.
        * Symmetrically when ``sp_node`` is the rule's *object*.

        Rules whose other endpoint is not bound in ``complete_group``
        cannot constrain the merge and are skipped (this is the
        truly-isolated weather-station case from the Phase 5 docstring).

        Returns ``True`` when the binding is safe to merge, ``False``
        when at least one rule would be violated.  Conservative: any
        rule the helper cannot evaluate (e.g. the object is itself
        set-bound, ``OptionalRule`` on a chain etc.) is treated as
        non-constraining (returns ``True`` for that rule).
        """
        ruleset = getattr(signature_pattern, "_ruleset", {})

        def _triple_exists(sm_subject: Any, pred: "Predicate", sm_obj: Any) -> bool:
            """``sm_subject --pred--> sm_obj`` holds in the semantic graph?"""
            if sm_subject is None or sm_obj is None:
                return False
            if not hasattr(sm_subject, "get_predicate_object_pairs"):
                # Literal / unexpected — be permissive rather than reject.
                return True
            pred_objs = sm_subject.get_predicate_object_pairs()
            for p in getattr(pred, "preds", ()):
                if sm_obj in pred_objs.get(p, []):
                    return True
            return False

        # Walk every (subj, pred, obj) where sp_node is at one end.
        for (subj, pred, obj), rule in ruleset.items():
            if subj is sp_node:
                other = obj
                other_sm = complete_group.get(other)
                if other_sm is None:
                    continue
                # Scalar fast path.  If sp_node is being proposed as a scalar
                # (the common case for OptionalRule scalar objects) and the
                # other endpoint is also scalar in complete_group, check the
                # triple.  Set-bound endpoints are skipped (conservative).
                if isinstance(sm_node, tuple) or isinstance(other_sm, tuple):
                    continue
                if not _triple_exists(sm_node, pred, other_sm):
                    return False
            elif obj is sp_node:
                other = subj
                other_sm = complete_group.get(other)
                if other_sm is None:
                    continue
                if isinstance(sm_node, tuple) or isinstance(other_sm, tuple):
                    continue
                if not _triple_exists(other_sm, pred, sm_node):
                    return False
        return True

    @staticmethod
    def _try_merge_with_incomplete(
        new_mapping, incomplete_matches, complete_matches, signature_pattern
    ):
        """
        Attempt to merge a new mapping with existing incomplete groups.

        Args:
            new_mapping: The new SP→SM mapping to merge
            incomplete_matches: List of existing incomplete mappings (mutated)
            complete_matches: List of complete mappings (mutated if merge completes)
            signature_pattern: The signature pattern being matched

        Returns:
            Updated list of incomplete matches
        """
        LOGGER.debug("Trying to merge with incomplete")
        LOGGER.add_level()
        if not incomplete_matches:
            LOGGER.debug("No existing incomplete matches, adding new mapping")
            incomplete_matches.append(new_mapping)
            LOGGER.remove_level()
            return incomplete_matches

        updated_incomplete = incomplete_matches.copy()
        merge_found = False

        for existing_group in incomplete_matches:
            LOGGER.debug(
                "Checking merge with existing group (%d nodes)", len(existing_group)
            )
            LOGGER.add_level()
            if Translator._match(
                existing_group,
                new_mapping,
                signature_pattern,
                complete_matches,
                updated_incomplete,
            ):
                merge_found = True
                LOGGER.debug("Merge successful")
                LOGGER.remove_level()
                break
            LOGGER.remove_level()

        if not merge_found:
            LOGGER.debug("No merge found, adding as new incomplete group")
            updated_incomplete.append(new_mapping)

        LOGGER.remove_level()
        return updated_incomplete

    @staticmethod
    def _are_groups_compatible(group_a, group_b):
        """
        Quick compatibility check: two groups can merge only if they don't
        map the same SP node to different SM nodes.

        Returns:
            Tuple of (is_compatible: bool, has_new_contributions: bool)
        """
        has_new = False
        for sp_node, sm_a in group_a.items():
            sm_b = group_b.get(sp_node)
            if sm_a is not None and sm_b is not None:
                if sm_a != sm_b:
                    return False, False  # Incompatible
            elif sm_a is None and sm_b is not None:
                has_new = True
        return True, has_new

    @staticmethod
    def _build_compatibility_index(incomplete_matches):
        """
        Build an index to quickly find groups that might conflict.

        Returns:
            node_to_groups: dict mapping sp_node -> dict of {sm_node: set of group indices}
        """
        node_to_groups = {}
        for idx, group in enumerate(incomplete_matches):
            for sp_node, sm_node in group.items():
                if sm_node is not None:
                    if sp_node not in node_to_groups:
                        node_to_groups[sp_node] = {}
                    if sm_node not in node_to_groups[sp_node]:
                        node_to_groups[sp_node][sm_node] = set()
                    node_to_groups[sp_node][sm_node].add(idx)
        return node_to_groups

    @staticmethod
    def _find_compatible_candidates(group_idx, group, node_to_groups, num_groups):
        """
        Find indices of groups that are potentially compatible with the given group.
        Uses the index to eliminate groups with conflicting mappings.

        Returns:
            Set of candidate group indices (excludes self and incompatible groups)
        """
        # Start with all groups as candidates
        candidates = set(range(num_groups))
        candidates.discard(group_idx)

        # Remove groups that map any sp_node to a different sm_node
        for sp_node, sm_node in group.items():
            if sm_node is not None and sp_node in node_to_groups:
                # Find groups that map this sp_node to something different
                for other_sm, other_groups in node_to_groups[sp_node].items():
                    if other_sm != sm_node:
                        candidates -= other_groups
                        if not candidates:
                            return candidates  # Early exit if no candidates left

        return candidates

    @staticmethod
    def _merge_incomplete_groups(
        incomplete_matches, complete_matches, signature_pattern
    ):
        """
        Iteratively merge incomplete groups until no more progress.

        Uses indexing to reduce O(n²) comparisons:
        1. Builds an index mapping (sp_node, sm_node) -> group indices
        2. Uses index to find only compatible candidates (eliminates conflicting pairs)
        3. Tracks failed pairs to avoid redundant _match calls within an iteration

        Args:
            incomplete_matches: List of incomplete mappings (mutated)
            complete_matches: List of complete mappings (mutated)
            signature_pattern: The signature pattern being matched

        Returns:
            Updated list of incomplete matches
        """
        LOGGER.debug("Merging incomplete groups")
        LOGGER.add_level()

        if len(incomplete_matches) <= 1:
            LOGGER.remove_level()
            return incomplete_matches

        previous_count = float("inf")

        while len(incomplete_matches) < previous_count:
            LOGGER.debug("Iteration with %d incomplete groups", len(incomplete_matches))
            previous_count = len(incomplete_matches)
            updated_incomplete = incomplete_matches.copy()

            # Build index for fast compatibility checking
            node_to_groups = Translator._build_compatibility_index(incomplete_matches)

            # Track pairs that failed the expensive _match in this iteration
            # (keyed by sorted tuple of list indices to avoid duplicate checks)
            failed_pairs = set()

            merge_found = False
            for i, group_i in enumerate(incomplete_matches):
                if merge_found:
                    break

                # Get only compatible candidates using the index
                candidates = Translator._find_compatible_candidates(
                    i, group_i, node_to_groups, len(incomplete_matches)
                )

                for j in candidates:
                    # Create canonical pair key
                    pair_key = (i, j) if i < j else (j, i)
                    if pair_key in failed_pairs:
                        continue

                    group_j = incomplete_matches[j]

                    # Quick check: does group_j contribute new mappings?
                    _, has_new = Translator._are_groups_compatible(group_i, group_j)
                    if not has_new:
                        failed_pairs.add(pair_key)
                        continue

                    if Translator._match(
                        group_i,
                        group_j,
                        signature_pattern,
                        complete_matches,
                        updated_incomplete,
                    ):
                        merge_found = True
                        LOGGER.debug("Merge found, restarting iteration")
                        break
                    else:
                        failed_pairs.add(pair_key)

            incomplete_matches = updated_incomplete

        LOGGER.remove_level()
        return incomplete_matches

    def _solve_milp(self) -> Dict:
        """
        Solve a Mixed Integer Linear Programming problem to determine which components
        and connections to include in the simulation model.

        Variables:
        - Y_i: Binary variable indicating if component pair i is included
        - E_j: Binary variable indicating if connection j is active

        Objective: Maximize the number of included components

        Returns:
            Dictionary with results and selected components/connections
        """

        # TODO: Maybe we should have 2 modes. "Strict": generates the largest complete model "Loose": generates as many components as possible, where some components might miss connections.

        LOGGER.task("Solving MILP problem")
        LOGGER.add_level()

        def update_Y_mappings(component, Y_idx_to_component, Y_component_to_idx, N_Y):
            if component not in Y_component_to_idx:
                Y_idx_to_component[N_Y] = component
                Y_component_to_idx[component] = N_Y
                N_Y += 1
            return Y_idx_to_component, Y_component_to_idx, N_Y

        def update_E_mappings(conn, E_idx_to_conn, E_conn_to_idx, N_E):
            if conn not in E_conn_to_idx:
                E_idx_to_conn[N_E] = conn
                E_conn_to_idx[conn] = N_E
                N_E += 1
            return E_idx_to_conn, E_conn_to_idx, N_E

        def update_mappings(
            conn,
            Y_idx_to_component,
            Y_component_to_idx,
            N_Y,
            E_idx_to_conn,
            E_conn_to_idx,
            N_E,
        ):
            Y_idx_to_component, Y_component_to_idx, N_Y = update_Y_mappings(
                conn[0], Y_idx_to_component, Y_component_to_idx, N_Y
            )
            Y_idx_to_component, Y_component_to_idx, N_Y = update_Y_mappings(
                conn[1], Y_idx_to_component, Y_component_to_idx, N_Y
            )
            E_idx_to_conn, E_conn_to_idx, N_E = update_E_mappings(
                conn, E_idx_to_conn, E_conn_to_idx, N_E
            )
            return (
                Y_idx_to_component,
                Y_component_to_idx,
                N_Y,
                E_idx_to_conn,
                E_conn_to_idx,
                N_E,
            )

        def matprint(mat, fmt="g"):
            col_maxes = [
                max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T
            ]
            for x in mat:
                row = "  ".join(("{:" + str(col_maxes[i]) + fmt + "}").format(y) for i, y in enumerate(x))
                LOGGER.debug("%s", row)

        def resolve_port_indices(
            groups_source,
            groups_target,
            output_port_index,
            input_port_index,
            sm_for_index,
        ):
            """
            Resolve Node-based port indices to integer indices using group matching.

            Both output_port_index and input_port_index Nodes are from the TARGET's
            signature pattern. This function finds which *unique-value ordinal* of
            the indexing Node corresponds to the given semantic model instance.

            Using unique-value ordinal (rather than raw group index) means that
            combined patterns — where multiple nodes vary simultaneously across
            groups (e.g. sensors × setpoints × actuators cross-product) — produce
            the same correct slot indices as separate single-node patterns.

            Example with a combined pattern producing 8 groups
            (4 sensors × 2 setpoints, actuator fixed):
              Group 0: {sensors: Zone_Air_Temp, setpoints: SP1, actuators: Reheat}
              Group 1: {sensors: Zone_Air_Temp, setpoints: SP2, actuators: Reheat}
              Group 2: {sensors: Supply_Air_Temp, setpoints: SP1, actuators: Reheat}
              Group 3: {sensors: Supply_Air_Temp, setpoints: SP2, actuators: Reheat}
              ...

            input_port_index=sensors Node, sm_for_index=Zone_Air_Temp:
              Unique sensor values in order seen: [Zone_Air_Temp(0), Supply_Air_Temp(1), ...]
              → ordinal 0  ✓

            input_port_index=sensors Node, sm_for_index=Supply_Air_Temp:
              Raw group index would be 2 ✗  →  unique ordinal 1 ✓

            - input_port_index=Node (Scalar→Vector, Vector→Vector):
              Find the unique ordinal of sm_for_index among all distinct values
              that input_port_index takes across groups_target.

            - output_port_index=Node (Vector→Scalar):
              Find the unique ordinal of sm_for_index among all distinct values
              that any node takes (that equals sm_for_index) across groups_source.

            Returns:
                tuple: (resolved_output_port_index, resolved_input_port_index)
                    - For scalar connections: (None, None)
                    - For Vector→Scalar: (int, None) - unique ordinal in source output
                    - For Scalar→Vector: (None, int) - unique ordinal in target input
                    - (None, None) with a warning if the instance was not found
            """
            if not isinstance(output_port_index, Node) and not isinstance(
                input_port_index, Node
            ):
                # Scalar → Scalar: no group matching needed
                return output_port_index, input_port_index

            # Virtually expand tuple bindings into per-element scalar
            # groups so the unique-URI-ordinal logic below can run
            # unchanged. Parallel tuple bindings within a single group
            # are expected to be aligned element-wise (this invariant is
            # established by ``Translator.__broadcast_recurse``): every
            # set-bound node in a group has the same arity, so we expand
            # by that common arity and align by position. Mixed arities
            # fall back to per-node independent expansion (outer
            # product), which preserves the prior scalar-path output.
            def _expand(groups):
                expanded = []
                for group in groups:
                    tuple_items = [
                        (sp_n, sm_v)
                        for sp_n, sm_v in group.items()
                        if isinstance(sm_v, tuple)
                    ]
                    if not tuple_items:
                        expanded.append(group)
                        continue
                    lengths = {len(v) for _, v in tuple_items}
                    if len(lengths) == 1:
                        arity = lengths.pop()
                        for i in range(arity):
                            virt = dict(group)
                            for sp_n, sm_v in tuple_items:
                                virt[sp_n] = sm_v[i]
                            expanded.append(virt)
                    else:
                        # Mixed arities: expand independently so each
                        # tuple node contributes its elements without
                        # implying alignment. Uncommon — most tuple
                        # groups have consistent arity by construction.
                        current = [dict(group)]
                        for sp_n, sm_v in tuple_items:
                            next_batch = []
                            for base in current:
                                for elem in sm_v:
                                    copy = dict(base)
                                    copy[sp_n] = elem
                                    next_batch.append(copy)
                            current = next_batch
                        expanded.extend(current)
                return expanded

            groups_source = _expand(groups_source)
            groups_target = _expand(groups_target)

            # Resolve each side independently so Vector->Vector connections
            # (both ``output_port_index`` and ``input_port_index`` as Nodes) get
            # both ends resolved.  The earlier ``if/elif`` short-circuited and
            # returned ``None`` for the un-resolved side, which then tripped the
            # ``add_connection`` assertion "output port must be a scalar" for
            # patterns like AHU's
            # ``sp.add_connection(damper_cmds, "inputSignal", "supplyDamperPosition",
            #                     output_port_index=damper_cmds, input_port_index=spaces)``
            # where CITS.inputSignal is a Vector (one slot per actuator) and
            # AHU.supplyDamperPosition is a Vector (one slot per zone).
            resolved_output_idx: Optional[int] = (
                output_port_index if not isinstance(output_port_index, Node) else None
            )
            resolved_input_idx: Optional[int] = (
                input_port_index if not isinstance(input_port_index, Node) else None
            )

            if isinstance(input_port_index, Node):
                # Scalar->Vector or Vector->Vector input side: build ordered
                # list of unique sm values for the indexing Node across
                # groups_target (preserving first-seen order = natural slot
                # order) and look up the value of ``input_port_index`` in
                # the (expanded) target group whose bindings contain
                # ``sm_for_index``.
                #
                # Using the *aligned* SM value rather than ``sm_for_index``
                # itself matters when the source-side iteration key
                # (e.g. ``damper_cmds``) differs from the input-side
                # indexing key (e.g. ``spaces``).  The AHU pattern's rule
                # chain ``vavs --feeds--> spaces`` and
                # ``vavs --hasPart--> dampers --hasPoint--> damper_cmds``
                # guarantees ``damper_cmds[i]`` and ``spaces[i]`` are
                # aligned by their shared per-VAV broadcast position;
                # ``_expand`` above turns parallel tuples into per-position
                # scalar groups so this alignment is queryable.
                seen_uris: Dict[str, Any] = {}
                for group in groups_target:
                    v = group.get(input_port_index)
                    if v is not None:
                        key = str(v.uri)
                        if key not in seen_uris:
                            seen_uris[key] = v

                # Find ``sm_for_index`` (the source-side per-element URI)
                # in one of the target groups; from that group, read the
                # aligned input_port_index value.  Direct hit (when
                # ``input_port_index`` IS the source-side node) reduces to
                # ``sm_for_index == sm_for_index``.
                aligned_sm = None
                for group in groups_target:
                    for sp_n, sm_v in group.items():
                        if sm_v == sm_for_index:
                            aligned_sm = group.get(input_port_index)
                            break
                    if aligned_sm is not None:
                        break
                lookup_target = aligned_sm if aligned_sm is not None else sm_for_index

                resolved_input_idx = None
                for ordinal, (uri_key, v) in enumerate(seen_uris.items()):
                    if v == lookup_target:
                        resolved_input_idx = ordinal
                        break

                if resolved_input_idx is None:
                    LOGGER.warning(
                        "Could not resolve input_port_index: %s not found mapping to %s "
                        "(aligned target value: %s).",
                        input_port_index,
                        sm_for_index,
                        aligned_sm,
                    )
                    return None, None

            if isinstance(output_port_index, Node):
                # Vector->Scalar or Vector->Vector output side: find which
                # sp_node in groups_source maps to sm_for_index, then build
                # the unique-value ordinal for that sp_node across all groups.
                index_sp_node = None
                for group in groups_source:
                    for sp_node, sm_node in group.items():
                        if sm_node == sm_for_index:
                            index_sp_node = sp_node
                            break
                    if index_sp_node is not None:
                        break

                if index_sp_node is None:
                    LOGGER.warning(
                        "Could not resolve output_port_index: %s not found in source groups.",
                        sm_for_index,
                    )
                    return None, None

                seen_uris = {}
                for group in groups_source:
                    v = group.get(index_sp_node)
                    if v is not None:
                        key = str(v.uri)
                        if key not in seen_uris:
                            seen_uris[key] = v

                resolved_output_idx = None
                for ordinal, (uri_key, v) in enumerate(seen_uris.items()):
                    if v == sm_for_index:
                        resolved_output_idx = ordinal
                        break

                if resolved_output_idx is None:
                    LOGGER.warning(
                        "Could not resolve output_port_index: %s not found in source groups.",
                        sm_for_index,
                    )
                    return None, None

            return resolved_output_idx, resolved_input_idx

        # def print_problem(problem_info):
        #     print("Problem:")
        #     for info in problem_info:
        #         print(info)

        # Component and connection index mappings
        Y_idx_to_component = {}  # Maps component variable index to component
        Y_component_to_idx = {}  # Maps component to variable index
        E_idx_to_conn = {}  # Maps connection index to connection details
        E_conn_to_idx = {}  # Maps connection tuple to connection index
        self.E_conn_to_sp_group = {}  # Maps connection tuple to signature pattern group

        # Track required inputs for each component
        required_inputs = (
            {}
        )  # {component: {input_key: [(source_component, source_key), ...]}}

        N_Y = 0  # Number of component variables
        N_E = 0  # Number of connection variables

        # First pass: identify all components and their connections
        for component in self._sim2sem_map.keys():
            sps = self._sim2group_map.get(component, {})
            # Process each signature pattern for this component
            for sp, groups in sps.items():
                if component not in required_inputs:
                    required_inputs[component] = {}

                Y_idx_to_component, Y_component_to_idx, N_Y = update_Y_mappings(
                    component, Y_idx_to_component, Y_component_to_idx, N_Y
                )

                # Process required inputs for this component
                for key, (
                    sp_subject,
                    source_keys,
                    output_port_index,
                    input_port_index,
                ) in sp.inputs.items():

                    # _map_port_indices()

                    if key not in required_inputs[component]:
                        required_inputs[component][key] = []

                    # Get all potential source nodes for this input.
                    # ``sp_subject`` may be set-bound (tuple binding); in
                    # that case every element contributes an independent
                    # provider candidate, so flatten across groups.
                    match_nodes = {
                        sm
                        for group in groups
                        if sp_subject in group
                        for sm in Translator._iter_binding(group[sp_subject])
                    }

                    # Find all potential provider components
                    for sm_subject in match_nodes:

                        if sm_subject in self._sem2sim_map:
                            provider_components = self._sem2sim_map[
                                sm_subject
                            ]  # Get the provider component

                            for provider_component in provider_components:
                                p_nodes = self._sim2sem_map[provider_component]
                                p_sps = self._sim2group_map[provider_component]

                                # Check each signature pattern of the provider
                                for p_sp, p_groups in p_sps.items():
                                    Y_idx_to_component, Y_component_to_idx, N_Y = (
                                        update_Y_mappings(
                                            provider_component,
                                            Y_idx_to_component,
                                            Y_component_to_idx,
                                            N_Y,
                                        )
                                    )
                                    b = False
                                    # Find the appropriate source port/key from the provider.
                                    #
                                    # Two conditions must hold for a provider candidate to be
                                    # eligible for a pattern-declared connection:
                                    #   1. Its modeled semantic node must be an rdf-instance of
                                    #      ``source_class`` (so the pattern's RDF-class key selects
                                    #      the right port name for this provider).
                                    #   2. Its *simulation-system class* must actually expose
                                    #      ``source_key`` as an output port.
                                    #
                                    # Without (2) the MILP happily picks providers that don't
                                    # expose the declared output (e.g. when a pattern says
                                    # ``sp.add_connection(node, "inputSignal", ...)`` and a plain
                                    # SensorSystem -- not a Controller -- ends up modeling
                                    # ``node``). The resulting edge fails at
                                    # ``sim_model.add_connection`` with an ``AssertionError``
                                    # about an invalid output port.
                                    for source_class, source_key in source_keys.items():
                                        for modeled_match_node in p_nodes:
                                            if not modeled_match_node.isinstance(
                                                source_class
                                            ):
                                                continue
                                            if source_key not in provider_component.output:
                                                continue
                                            b = True
                                            break
                                        if b:
                                            break

                                    if b:
                                        # Resolve Node-based port indices to integer indices
                                        # Both output_port_index and input_port_index Nodes are from
                                        # the TARGET's signature pattern. We need to find their semantic
                                        # instances from the target groups to search in source/target groups.
                                        #
                                        # - output_port_index=Node (Vector→Scalar): find which slot in
                                        #   groups_source maps to the matching semantic instance
                                        # - input_port_index=Node (Scalar→Vector, Vector→Vector): find
                                        #   which slot in groups_target matches the semantic instance

                                        sm_for_index = sm_subject  # Default: sender's semantic instance

                                        # Only override ``sm_for_index`` from the
                                        # tuple binding when *neither* port-index
                                        # Node refers to ``sp_subject``.  When either
                                        # of them IS the sender node, the outer-loop
                                        # ``sm_subject`` (one per element of the
                                        # set-bound binding) is already the right
                                        # per-element key; overwriting it with
                                        # ``elements[0]`` of the other index's tuple
                                        # binding would collapse the fan-out for
                                        # patterns like AHU's
                                        # ``sp.add_connection(damper_cmds, "inputSignal",
                                        #                     "supplyDamperPosition",
                                        #                     output_port_index=damper_cmds,
                                        #                     input_port_index=spaces)``,
                                        # where ``damper_cmds`` IS the sender and
                                        # ``spaces`` indexes the receiver Vector.
                                        # ``resolve_port_indices`` does the
                                        # alignment lookup (find the aligned
                                        # ``spaces`` value from the group containing
                                        # the current ``damper_cmds`` URI) on the
                                        # input side, so passing ``sm_subject`` is
                                        # sufficient.
                                        out_is_subject = (
                                            isinstance(output_port_index, Node)
                                            and output_port_index is sp_subject
                                        )
                                        in_is_subject = (
                                            isinstance(input_port_index, Node)
                                            and input_port_index is sp_subject
                                        )
                                        if not (out_is_subject or in_is_subject):
                                            if isinstance(output_port_index, Node):
                                                for group in groups:
                                                    if output_port_index in group:
                                                        raw = group[output_port_index]
                                                        elements = Translator._iter_binding(raw)
                                                        if elements:
                                                            sm_for_index = elements[0]
                                                        break
                                            elif isinstance(input_port_index, Node):
                                                for group in groups:
                                                    if input_port_index in group:
                                                        raw = group[input_port_index]
                                                        elements = Translator._iter_binding(raw)
                                                        if elements:
                                                            sm_for_index = elements[0]
                                                        break

                                        resolved_output_idx, resolved_input_idx = (
                                            resolve_port_indices(
                                                p_groups,
                                                groups,
                                                output_port_index,
                                                input_port_index,
                                                sm_for_index,
                                            )
                                        )

                                        # Add this potential connection with resolved indices
                                        conn = (
                                            provider_component,
                                            component,
                                            source_key,
                                            key,
                                            resolved_output_idx,
                                            resolved_input_idx,
                                        )
                                        E_idx_to_conn, E_conn_to_idx, N_E = (
                                            update_E_mappings(
                                                conn, E_idx_to_conn, E_conn_to_idx, N_E
                                            )
                                        )
                                        self.E_conn_to_sp_group[conn] = (
                                            sp,
                                            groups,
                                            p_sp,
                                            p_groups,
                                        )
                                        if (
                                            provider_component,
                                            source_key,
                                            resolved_output_idx,
                                            resolved_input_idx,
                                        ) not in required_inputs[component][key]:
                                            required_inputs[component][key].append(
                                                (
                                                    provider_component,
                                                    source_key,
                                                    resolved_output_idx,
                                                    resolved_input_idx,
                                                )
                                            )
                                    # else: this provider candidate is not eligible --
                                    # either its modeled semantic node didn't match any
                                    # declared source_class, or it does not expose the
                                    # declared output port. Skip silently; other
                                    # providers (or the required-input MILP constraint)
                                    # will decide whether the connection is satisfiable.



        # Set up the constraints
        total_vars = N_E + N_Y + N_Y
        constraints_list = []
        constraint_info = []

        # 1. Required input constraints:
        # If a component is included, all its required inputs must be satisfied
        required_input_constraints = []
        for component, inputs in required_inputs.items():
            component_idx = Y_component_to_idx[component]

            for input_key, providers in inputs.items():
                if providers:  # No providers found for this input

                    # Create a constraint: Y_i <= (E_j1 + E_j2 + ... + E_jn)
                    # This means: If component i is included, at least one provider must be active
                    row = np.zeros(total_vars)
                    row[N_E + component_idx] = 1  # Coefficient for component i

                    edge_indices = []
                    for (
                        provider_component,
                        source_key,
                        output_port_index,
                        input_port_index,
                    ) in providers:
                        conn = (
                            provider_component,
                            component,
                            source_key,
                            input_key,
                            output_port_index,
                            input_port_index,
                        )
                        edge_idx = E_conn_to_idx[conn]
                        row[edge_idx] = -1  # Negative coefficient for the edge
                        edge_indices.append(edge_idx)

                    if edge_indices:
                        required_input_constraints.append(row)
                        edge_vars = [f"E_{idx}" for idx in edge_indices]
                        constraint_desc = f"Y_{component_idx} <= {' + '.join(edge_vars)}"
                        constraint_info.append(constraint_desc)

        # Convert to numpy array
        if required_input_constraints:
            A_required = np.vstack(required_input_constraints)
            b_required_l = np.full(
                len(required_input_constraints), -np.inf
            )  # Lower bound = -inf
            b_required_u = np.zeros(len(required_input_constraints))  # Upper bound = 0
            constraints_list.append(
                LinearConstraint(A_required, b_required_l, b_required_u)
            )

        # 2. Connection source constraints:
        # A connection can only exist if its source component is included
        conn_source_constraints = []
        for e_idx, (
            source_component,
            target_component,
            source_key,
            target_key,
            output_port_index,
            input_port_index,
        ) in E_idx_to_conn.items():
            source_idx = Y_component_to_idx[source_component]

            # Create constraint: E_j <= Y_i (connection j can only exist if source component i is included)
            row = np.zeros(total_vars)
            row[e_idx] = 1
            row[N_E + source_idx] = -1
            conn_source_constraints.append(row)
            constraint_desc = f"E_{e_idx} <= Y_{source_idx}"
            constraint_info.append(constraint_desc)

        # Convert to numpy array
        if conn_source_constraints:
            A_conn_source = np.vstack(conn_source_constraints)
            b_conn_source_l = np.full(
                len(conn_source_constraints), -np.inf
            )  # Lower bound = -inf
            b_conn_source_u = np.zeros(len(conn_source_constraints))  # Upper bound = 0
            constraints_list.append(
                LinearConstraint(A_conn_source, b_conn_source_l, b_conn_source_u)
            )

        # 3. Connection target constraints:
        # A connection can only exist if its target component is included
        conn_target_constraints = []
        for e_idx, (
            _,
            target_component,
            _,
            _,
            _,
            _,
        ) in E_idx_to_conn.items():
            target_idx = Y_component_to_idx[target_component]

            # Create constraint: E_j <= Y_i (connection j can only exist if target component i is included)
            row = np.zeros(total_vars)
            row[e_idx] = 1
            row[N_E + target_idx] = -1
            conn_target_constraints.append(row)
            constraint_desc = f"E_{e_idx} <= Y_{target_idx}"
            constraint_info.append(constraint_desc)

        # Convert to numpy array
        if conn_target_constraints:
            A_conn_target = np.vstack(conn_target_constraints)
            b_conn_target_l = np.full(
                len(conn_target_constraints), -np.inf
            )  # Lower bound = -inf
            b_conn_target_u = np.zeros(len(conn_target_constraints))  # Upper bound = 0
            constraints_list.append(
                LinearConstraint(A_conn_target, b_conn_target_l, b_conn_target_u)
            )

        # 4. One-input constraints: Each input slot can receive at most one connection
        # Group by (target_component, target_key, slot_index) where slot_index is:
        #   - None for scalar inputs
        #   - int for vector input slots
        conn_by_slot = (
            {}
        )  # {(target_component, target_key, slot_index): [edge_indices]}

        for e_idx, (
            _,
            target_component,
            _,
            target_key,
            output_port_index,
            input_port_index,
        ) in E_idx_to_conn.items():
            # input_port_index is either None (scalar) or int (vector slot)
            key = (target_component, target_key, input_port_index)
            if key not in conn_by_slot:
                conn_by_slot[key] = []
            conn_by_slot[key].append(e_idx)

        one_input_constraints = []
        for (
            target_comp,
            target_key,
            slot_idx,
        ), slot_connections in conn_by_slot.items():
            if (
                len(slot_connections) > 1
            ):  # Only need constraint if multiple potential connections to same slot
                row = np.zeros(total_vars)
                for e_idx in slot_connections:
                    row[e_idx] = 1
                one_input_constraints.append(row)
                edge_vars = [f"E_{idx}" for idx in slot_connections]
                constraint_desc = f"{' + '.join(edge_vars)} <= 1 (slot {slot_idx})"
                constraint_info.append(constraint_desc)

        # Convert to numpy array
        if one_input_constraints:
            A_one_input = np.vstack(one_input_constraints)
            b_one_input_l = np.full(
                len(one_input_constraints), -np.inf
            )  # Lower bound = -inf
            b_one_input_u = np.ones(len(one_input_constraints))  # Upper bound = 1
            constraints_list.append(
                LinearConstraint(A_one_input, b_one_input_l, b_one_input_u)
            )

        # 5. Modeled-identity mutex constraints.
        #
        # Two kinds of mutual exclusion apply here:
        #
        # (a) Per-node exclusivity for *singleton* modeled identities — the
        #     historical behaviour: a semantic node that is claimed as a
        #     singleton ``add_modeled_node(node)`` by multiple component
        #     candidates may be used by at most one of them.
        #
        # (b) Per-fingerprint exclusivity for ``ModeledNode`` groups — at
        #     most one component may claim a given ``(members + relations)``
        #     context, which is identified by its relational fingerprint.
        #
        # Member SM nodes of a ``ModeledNode`` group are intentionally
        # *non-exclusive*: other systems (e.g. a ``SensorSystem``) may
        # independently model a member semantic node on its own. Whether
        # such shared claims are compatible is decided by port-eligibility
        # filtering during edge enumeration, not here.
        node_to_components: Dict[Any, List[int]] = {}
        fingerprint_to_components: Dict[str, List[int]] = {}
        for component, modeled_nodes in self._sim2sem_map.items():
            if (
                component not in Y_component_to_idx
            ):  # Make sure component is in our variable list
                continue
            component_idx = Y_component_to_idx[component]
            group_members = self._sim_group_members.get(component, set())
            fp = self._sim_fingerprint.get(component)
            for node in modeled_nodes:
                # Skip per-node mutex for nodes claimed only via a
                # ``ModeledNode`` group (non-exclusive semantics).
                if node in group_members:
                    continue
                node_to_components.setdefault(node, []).append(component_idx)
            if fp is not None:
                fingerprint_to_components.setdefault(fp, []).append(component_idx)

        modeled_node_constraints = []
        # For each node that appears in multiple components
        for node, component_indices in node_to_components.items():
            if len(component_indices) > 1:
                # Create a constraint: sum(Y_i for all components containing this node) <= 1
                row = np.zeros(total_vars)
                for idx in component_indices:
                    row[N_E + idx] = 1
                modeled_node_constraints.append(row)
                components_str = " + ".join([f"Y_{idx}" for idx in component_indices])
                constraint_desc = f"{components_str} <= 1 (node mutex)"
                constraint_info.append(constraint_desc)

        for fp, component_indices in fingerprint_to_components.items():
            if len(component_indices) > 1:
                row = np.zeros(total_vars)
                for idx in component_indices:
                    row[N_E + idx] = 1
                modeled_node_constraints.append(row)
                components_str = " + ".join([f"Y_{idx}" for idx in component_indices])
                constraint_desc = (
                    f"{components_str} <= 1 (fingerprint {fp[:8]})"
                )
                constraint_info.append(constraint_desc)

        # Convert to numpy array and add to constraints
        if modeled_node_constraints:
            A_modeled_node = np.vstack(modeled_node_constraints)
            b_modeled_node_l = np.full(
                len(modeled_node_constraints), -np.inf
            )  # Lower bound = -inf
            b_modeled_node_u = np.ones(len(modeled_node_constraints))  # Upper bound = 1
            constraints_list.append(
                LinearConstraint(A_modeled_node, b_modeled_node_l, b_modeled_node_u)
            )

        # Balance the objective function - use a small weight for source nodes
        source_node_weight = 0  # 1.1#1.1  # Adjust this if needed - smaller weight means components are more important. We set it to 1.1 to make sure that the source nodes are not selected in isolation. However, if chosen, at least one additional component should be selected for it to be an advantage.

        c = np.zeros(total_vars)
        c[:N_E] = (
            -0.1
        )  # -1 works. Maximize the number of edges. We do this to favor more specialized components, e.g. BuildingSpace components with 1 adjacent space instead of 0 adjacent spaces.
        c[N_E + N_Y :] = source_node_weight  # Minimize source nodes

        # Modify the objective function to prefer complex components over multiple simple ones
        component_selection_cost = (
            0.9  # Fixed cost for selecting any component (< semantic_instance_benefit)
        )
        semantic_instance_benefit = 10  # Benefit per modeled semantic instance

        # Update the objective function coefficients.
        #
        # Coverage weight:
        #   - Singleton modeled identities contribute 1 each.
        #   - A multi-member ``ModeledNode`` group contributes a single unit
        #     (not ``len(members)``), so that e.g. a CITS claiming
        #     ``[vav, sensors, setpoints, actuators]`` as one implicit
        #     controller does not out-weigh four independent singleton
        #     components.
        for i in range(N_Y):
            component = Y_idx_to_component[i]
            if component in self._sim2sem_map:
                modeled_nodes = self._sim2sem_map[component]
                group_members = self._sim_group_members.get(component, set())
                fp = self._sim_fingerprint.get(component)
                # Count non-group-member SM nodes individually.
                node_count = sum(
                    1 for n in modeled_nodes if n not in group_members
                )
                # A group contributes +1 in total, regardless of member count.
                if fp is not None:
                    node_count += 1

                # Net contribution: cost - (benefit × node_count)
                c[N_E + i] = component_selection_cost - (
                    semantic_instance_benefit * node_count
                )

        # All variables are binary
        integrality = np.ones(total_vars)
        bounds = Bounds(lb=0, ub=1)

        LOGGER.section("Problem info")
        LOGGER.add_level()

        # LOGGER("Objective function")
        # LOGGER.add_level()
        # LOGGER(c)
        # LOGGER.remove_level()

        LOGGER.section("Variables")
        LOGGER.add_level()
        for i in range(N_Y):
            component = Y_idx_to_component[i]
            LOGGER.result("Y_%d: %s", i, component.id)
        LOGGER.remove_level()
        LOGGER.ok("Variables", change_status=True)

        LOGGER.section("Constraint info")
        LOGGER.add_level()
        for info in constraint_info:
            LOGGER.result("%s", info)
        LOGGER.remove_level()
        LOGGER.ok("Constraint info", change_status=True)
        LOGGER.remove_level()
        LOGGER.ok("Problem info", change_status=True)

        # Solve the MILP problem
        if not constraints_list:
            LOGGER.warning("No valid constraints.")
            LOGGER.remove_level()
            LOGGER.warning("Solving MILP problem", change_status=True)
            return {"success": False, "message": "No valid constraints"}

        res = milp(
            c=c, constraints=constraints_list, integrality=integrality, bounds=bounds
        )

        LOGGER.section("Solution")
        LOGGER.add_level()

        LOGGER.section("Active components")
        LOGGER.add_level()
        components = []
        for i in range(N_Y):
            if res.x[N_E + i] == 1:
                component = Y_idx_to_component[i]
                components.append(component)
                LOGGER.result(
                    "Y_%d: 1 (%s)%s", i, component.__class__.__name__, component.id
                )
        LOGGER.remove_level()
        LOGGER.ok("Active components", change_status=True)

        LOGGER.section("Active connections")
        LOGGER.add_level()
        connections = []
        # Collect all active connections and find max length for alignment
        active_conn_strings = []
        for i in range(N_E):
            if res.x[i] == 1:
                connections.append(E_idx_to_conn[i])
                (
                    source,
                    target,
                    source_key,
                    target_key,
                    output_port_index,
                    input_port_index,
                ) = E_idx_to_conn[i]
                left_part = f"  E_{i} = 1: ({source.__class__.__name__}){source.id}.{source_key}"
                right_part = f"({target.__class__.__name__}){target.id}.{target_key}"
                active_conn_strings.append((left_part, right_part))

        # Find max length of left parts
        if active_conn_strings:
            max_left_len = max(len(left) for left, _ in active_conn_strings)
            for left_part, right_part in active_conn_strings:
                LOGGER.result("Connection: %s -> %s", left_part, right_part)
        LOGGER.remove_level()
        LOGGER.ok("Active connections", change_status=True)

        LOGGER.section("Inactive components")
        LOGGER.add_level()
        for i in range(N_Y):
            if res.x[N_E + i] == 0:
                component = Y_idx_to_component[i]
                LOGGER.result(
                    "Y_%d: 0 (%s)%s", i, component.__class__.__name__, component.id
                )
        LOGGER.remove_level()
        LOGGER.ok("Inactive components", change_status=True)

        LOGGER.section("Inactive connections")
        LOGGER.add_level()
        # Collect all inactive connections and find max length for alignment
        inactive_conn_strings = []
        for i in range(N_E):
            if res.x[i] == 0:
                (
                    source,
                    target,
                    source_key,
                    target_key,
                    output_port_index,
                    input_port_index,
                ) = E_idx_to_conn[i]
                left_part = f"  E_{i} = 0: ({source.__class__.__name__}){source.id}.{source_key}"
                right_part = f"({target.__class__.__name__}){target.id}.{target_key}"
                inactive_conn_strings.append((left_part, right_part))

        # Find max length of left parts
        if inactive_conn_strings:
            max_left_len = max(len(left) for left, _ in inactive_conn_strings)
            for left_part, right_part in inactive_conn_strings:
                LOGGER.result(
                    "Connection: %s<%d> -> %s", left_part, max_left_len, right_part
                )
        LOGGER.remove_level()
        LOGGER.ok("Inactive connections", change_status=True)

        # if debug:
        #     print_problem(problem_info)
        LOGGER.remove_level()
        LOGGER.ok("Solution", change_status=True)

        LOGGER.remove_level()
        if res.success:
            LOGGER.ok("Solving MILP problem", change_status=True)
            return {
                "success": True,
                "message": "Optimization successful",
                "problem_info": constraint_info,
                "connections": connections,
            }
        LOGGER.warning("Solving MILP problem", change_status=True)
        return {"success": False, "message": res.message}

    def _instantiate_components(
        self, complete_groups: Dict, semantic_model: core.SemanticModel
    ) -> Dict:
        """
        Create component instances from matched groups

        Args:
            complete_groups: Dictionary of matched pattern groups

        Returns:
            Dictionary of instantiated components
        """

        def get_predicate_object_pairs(component):
            pairs = component.get_predicate_object_pairs()
            pairs_new = {}
            for key, value in pairs.items():
                key_ = semantic_model.get_instance(key).get_short_name()
                for value_ in value:
                    if isinstance(value_, core.SemanticLiteral):
                        pairs_new[key_] = value_.uri.value
            return pairs_new

        LOGGER.task("Instantiating components")
        LOGGER.add_level()

        # Component instantiation logic from _connect method
        class_to_instance_map = {}
        self._sim2sem_map = {}
        self._sem2sim_map = {}
        self._sim2group_map = (
            {}
        )  # Maps components to their matched signature patterns and groups
        # ``_sim_group_members[component]``: set of SM nodes that the
        # component claims *as members of a multi-member ModeledNode group*.
        # These are exempted from the per-node exclusive-mutex in the MILP
        # (non-exclusive semantics), because other systems may legitimately
        # also model them on their own.
        self._sim_group_members: Dict[core.System, set] = {}
        # ``_sim_fingerprint[component]``: relational fingerprint for
        # components backed by a multi-member ``ModeledNode`` group; ``None``
        # otherwise. Used as the MILP mutex bucket ("at most one component
        # per (members + relations) context").
        self._sim_fingerprint: Dict[core.System, Optional[str]] = {}
        # ``_context_to_component[fp]``: context-addressable lookup.
        self._context_to_component: Dict[str, core.System] = {}
        self.modeled_components = set()
        for i, (component_cls, sps) in enumerate(complete_groups.items()):
            LOGGER.section("Class: %s", component_cls.__name__)
            LOGGER.add_level()
            for sp, groups in sps.items():
                # Detect multi-member ModeledNode groups declared on this SP.
                mn_groups = [
                    mn
                    for mn in sp.modeled_nodes
                    if isinstance(mn, ModeledNode) and len(mn.members) > 1
                ]
                for group in groups:
                    # Expand modeled identities to the set of matched SM
                    # nodes, flattening any ModeledNode group to its
                    # members. We also record which matched SM nodes came
                    # from a multi-member group (for non-exclusive mutex).
                    modeled_match_nodes = set()
                    group_member_sm_nodes = set()
                    for sp_mn in sp.modeled_nodes:
                        if isinstance(sp_mn, ModeledNode):
                            for m in sp_mn.members:
                                # Members may be set-bound (tuple) when the
                                # group was produced by a SetStepRule chain.
                                # Flatten per-element so the mutex, id_
                                # composition, and base_kwargs merge see
                                # scalar SM nodes.
                                for sm in Translator._iter_binding(group[m]):
                                    modeled_match_nodes.add(sm)
                                    if len(sp_mn.members) > 1:
                                        group_member_sm_nodes.add(sm)
                        else:
                            for sm in Translator._iter_binding(group[sp_mn]):
                                modeled_match_nodes.add(sm)
                    self.modeled_components.update(modeled_match_nodes)  # Union/add set

                    # Compute the relational fingerprint when a multi-member
                    # ModeledNode group participates. For singleton matches
                    # we keep the existing single-IRI id for backwards
                    # compatibility with HTR, sensor, fan-coil etc. patterns.
                    component_fingerprint: Optional[str] = None
                    if mn_groups:
                        # Combine per-group fingerprints into a single stable
                        # id for this match; also used as the mutex bucket.
                        fp_parts = sorted(
                            resolve_fingerprint(mn, group) for mn in mn_groups
                        )
                        component_fingerprint = hashlib.blake2b(
                            "|".join(fp_parts).encode("utf-8"),
                            digest_size=16,
                        ).hexdigest()

                    if len(modeled_match_nodes) == 1:
                        component = next(iter(modeled_match_nodes))
                        id_ = core.sanitize_id(component.get_short_name())
                        base_kwargs = get_predicate_object_pairs(component)
                        extension_kwargs = {"id": id_}
                    else:
                        modeled_match_nodes_sorted = sorted(
                            modeled_match_nodes, key=lambda x: x.uri
                        )
                        if component_fingerprint is not None:
                            # Composite identity from a ModeledNode group:
                            # keep a *per-member* slice in the human-readable
                            # prefix so the id still hints at every
                            # participant, then suffix with the fingerprint
                            # so two groups sharing the same members but
                            # different relational shape get distinct ids.
                            # Naively concatenating full short names blows
                            # past Windows 260-char MAX_PATH when the id is
                            # used as a filename under
                            # ``model_parameters/<class>/<id>.json``; the
                            # fingerprint guarantees uniqueness so the slice
                            # is purely a debugging aid.
                            name_budget = 80
                            n_members = len(modeled_match_nodes_sorted)
                            per_member = max(
                                4, (name_budget // max(1, n_members)) - 2
                            )
                            tokens: List[str] = []
                            for n in modeled_match_nodes_sorted:
                                short = core.sanitize_id(n.get_short_name())
                                if len(short) > per_member:
                                    # Keep the tail: for most
                                    # naming schemes (e.g. Mortar
                                    # ``bldg1_ZONE_AHU01_RM115_Zone_...``)
                                    # the trailing segment carries the role
                                    # while the prefix is common boilerplate.
                                    short = short[-per_member:]
                                tokens.append(f"[{short}]")
                            id_ = "".join(tokens) + f"_{component_fingerprint[:16]}"
                        else:
                            # No fingerprint (non-composite ModeledNode path):
                            # fall back to the bracketed-members form.
                            id_ = "".join(
                                "[%s]" % core.sanitize_id(n.get_short_name())
                                for n in modeled_match_nodes_sorted
                            )
                        base_kwargs = {}
                        extension_kwargs = {
                            "id": id_,
                            "base_components": list(modeled_match_nodes_sorted),
                        }
                        for component in modeled_match_nodes_sorted:
                            kwargs = get_predicate_object_pairs(component)
                            base_kwargs.update(kwargs)

                    if (
                        component_cls not in class_to_instance_map
                        or id_ not in class_to_instance_map[component_cls]
                    ):  # Check if the instance is already created. For components with Multiple matches, the model might already have been created.
                        base_kwargs.update(extension_kwargs)
                        s = base_kwargs["id"]
                        LOGGER.info("Instantiating component: %s", s)
                        component = component_cls(**base_kwargs)

                        if component_cls not in class_to_instance_map:
                            class_to_instance_map[component_cls] = {}

                        assert (
                            component.id not in class_to_instance_map[component_cls]
                        ), f"Component {component.id} already exists in class {component_cls}"
                        class_to_instance_map[component_cls][component.id] = component

                        if len(sp.parameters) > 0:
                            LOGGER.section(
                                "Mapping parameters for component: %s", component.id
                            )
                            LOGGER.add_level()
                            # Get all parameters for the component
                            for key, node in sp.parameters.items():
                                if group[node] is not None:
                                    value = group[node]
                                    value = value.uri.value
                                    obj = rgetattr(component, key)
                                    LOGGER.config("%s: %s", key, value)

                                    if isinstance(obj, tps.Parameter):
                                        rsetattr(
                                            component,
                                            key,
                                            tps.Parameter(
                                                torch.tensor(
                                                    value, dtype=torch.float64
                                                ),
                                                requires_grad=False,
                                            ),
                                        )
                                    else:
                                        rsetattr(component, key, value)
                            LOGGER.remove_level()
                            LOGGER.ok(
                                "Mapping parameters for component: %s",
                                component.id,
                                change_status=True,
                            )

                        # Store matched groups for this signature pattern
                        if component not in self._sim2group_map:
                            self._sim2group_map[component] = {}
                        self._sim2group_map[component][sp] = [group]
                        self._sim2sem_map[component] = modeled_match_nodes
                        self._sim_group_members[component] = set(
                            group_member_sm_nodes
                        )
                        self._sim_fingerprint[component] = component_fingerprint
                        if component_fingerprint is not None:
                            self._context_to_component[
                                component_fingerprint
                            ] = component
                        for modeled_match_node in modeled_match_nodes:
                            if modeled_match_node not in self._sem2sim_map:
                                self._sem2sim_map[modeled_match_node] = set()
                            self._sem2sim_map[modeled_match_node].add(component)
                    else:
                        component = class_to_instance_map[component_cls][
                            id_
                        ]  # Get the existing component
                        # Merge matched signature patterns if component already exists
                        sps_new = self._sim2group_map[component]
                        if sp not in sps_new:
                            sps_new[sp] = []
                        sps_new[sp].append(group)
                        # Apply parameters from the new pattern, but only when the
                        # current value is still None (avoids overwriting set values).
                        if len(sp.parameters) > 0:
                            for key, node in sp.parameters.items():
                                if group[node] is not None:
                                    current_val = rgetattr(component, key)
                                    if current_val is None:
                                        value = group[node]
                                        value = value.uri.value
                                        obj = rgetattr(component, key)
                                        LOGGER.config(
                                            "Backfilling parameter %s=%s for %s",
                                            key,
                                            value,
                                            component.id,
                                        )
                                        if isinstance(obj, tps.Parameter):
                                            rsetattr(
                                                component,
                                                key,
                                                tps.Parameter(
                                                    torch.tensor(
                                                        value, dtype=torch.float64
                                                    ),
                                                    requires_grad=False,
                                                ),
                                            )
                                        else:
                                            rsetattr(component, key, value)
            LOGGER.remove_level()
            LOGGER.ok("Class: %s", component_cls.__name__, change_status=True)
        LOGGER.remove_level()
        LOGGER.ok("Instantiating components", change_status=True)

    def _connect_components(
        self,
        connections: List[Tuple[core.System, core.System, str, str]],
        sim_model: core.SimulationModel,
    ) -> None:
        """
        Connect instantiated components and add them to simulation model

        Args:
            connections: List of tuples of instantiated components and their connections
            sim_model: SimulationModel to add components to
        """
        LOGGER.task("Connecting components")
        LOGGER.add_level()
        # Extract the components that are actually used in connections
        new_E_conn_to_sp_group = {}
        used_components = set()
        for conn in connections:
            (
                source,
                target,
                source_key,
                target_key,
                output_port_index,
                input_port_index,
            ) = conn

            # sp, groups = self.E_conn_to_sp_group[conn]
            used_components.add(source)
            used_components.add(target)
            new_E_conn_to_sp_group[conn] = self.E_conn_to_sp_group[conn]
            # sim_model.add_connection(*conn) # NOTE: we need to update output_port_index and input_port_index first below
        self.E_conn_to_sp_group = new_E_conn_to_sp_group

        ### JUST ADDED ###
        # Find which signature patterns are actually used for this component
        new_sim2group_map = {}
        for conn in connections:
            (
                source,
                target,
                source_key,
                target_key,
                output_port_index,
                input_port_index,
            ) = conn
            sp_target, groups_target, sp_source, groups_source = (
                self.E_conn_to_sp_group[conn]
            )

            if target not in new_sim2group_map:
                new_sim2group_map[target] = {}
            if sp_target not in new_sim2group_map[target]:
                new_sim2group_map[target][sp_target] = groups_target
            else:
                assert (
                    groups_target == new_sim2group_map[target][sp_target]
                ), "Groups target do not match"

            if source not in new_sim2group_map:
                new_sim2group_map[source] = {}
            if sp_source not in new_sim2group_map[source]:
                new_sim2group_map[source][sp_source] = groups_source
            else:
                assert (
                    groups_source == new_sim2group_map[source][sp_source]
                ), "Groups source do not match"

            # new_sim2group_map[source]

            # used_sps.add(sp)
            # used_sps.add(p_sp)

        self._sim2group_map = new_sim2group_map

        for conn in connections:
            (
                source,
                target,
                source_key,
                target_key,
                output_port_index,
                input_port_index,
            ) = conn
            conn_str = f"({source.__class__.__name__}){source.id}.{source_key}[{output_port_index}] -> ({target.__class__.__name__}){target.id}.{target_key}[{input_port_index}]"
            LOGGER.info("Adding connection: %s", conn_str)

            # Indices are pre-resolved during MILP setup:
            # - None for scalar ports
            # - int for vector port slots
            sim_model.add_connection(
                source,
                target,
                source_key,
                target_key,
                output_port_index,
                input_port_index,
            )

        # 3. Update _sim2sem_map
        self._sim2sem_map = {
            component: nodes
            for component, nodes in self._sim2sem_map.items()
            if component in used_components
        }

        # 4. Update _sem2sim_map - this is more complex as it's inversely mapped
        new_sem2sim_map = {}
        for sem_node, sim_components in self._sem2sim_map.items():
            # Filter to only keep used components for each semantic node
            used_sim_components = {
                comp for comp in sim_components if comp in used_components
            }
            if used_sim_components:  # Only keep entries that still have components
                new_sem2sim_map[sem_node] = used_sim_components
        self._sem2sim_map = new_sem2sim_map

        # 5. Update modeled_components set
        self.modeled_components = {
            node
            for component in used_components
            for node in self._sim2sem_map.get(component, set())
        }
        LOGGER.remove_level()
        LOGGER.ok("Connecting components", change_status=True)

    @staticmethod
    def _copy_nodemap(nodemap: Dict[Node, Any]) -> Dict[Node, Any]:
        return {k: v for k, v in nodemap.items()}

    @staticmethod
    def _copy_nodemap_list(
        nodemap_list: List[Dict[Node, Any]],
    ) -> List[Dict[Node, Any]]:
        return [Translator._copy_nodemap(nodemap) for nodemap in nodemap_list]

    @staticmethod
    def _get_node_string(sp_subject: Optional[Node], sm_subject: Optional[Any]):
        """Format subject node pair for debug logging."""
        ss = Translator._binding_short_name(sm_subject)
        return f"{sp_subject.id}: {ss}"

    @staticmethod
    def _get_map_string(l, f=None):
        if f is None:
            f = LOGGER.debug
        LOGGER.add_level()
        for sp, sm in l.items():
            ss = Translator._binding_short_name(sm)
            f("%s: %s", sp.id, ss)
        LOGGER.remove_level()

    @staticmethod
    def _get_maps_string(maps, f=None):
        if f is None:
            f = LOGGER.debug
        for i, l in enumerate(maps):
            f("Map %d:", i)
            LOGGER.add_level()
            for sp, sm in l.items():
                ss = Translator._binding_short_name(sm)
                f("%s: %s", sp.id, ss)
            LOGGER.remove_level()

    @staticmethod
    def _iter_binding(value) -> Tuple[Any, ...]:
        """Return the binding as a tuple of scalar SM objects.

        Works transparently with both scalar bindings (as produced by
        :class:`StepRule`) and tuple bindings (as produced by
        :class:`SetStepRule`). Consumers that need to enumerate per-element
        semantic-model objects should always go through this helper, which
        keeps the rest of the Translator agnostic to the binding shape.

        - ``None``                 -> ``()``
        - scalar SM object ``o``   -> ``(o,)``
        - ``tuple`` or ``list``    -> ``tuple(value)``
        """
        if value is None:
            return ()
        if isinstance(value, (tuple, list)):
            return tuple(value)
        return (value,)

    @staticmethod
    def _binding_short_name(value) -> Optional[str]:
        """Format a binding value (scalar or tuple) for debug logging."""
        if value is None:
            return None
        elements = Translator._iter_binding(value)
        if len(elements) == 1:
            only = elements[0]
            return only.get_short_name() if hasattr(only, "get_short_name") else str(only)
        names = [
            (e.get_short_name() if hasattr(e, "get_short_name") else str(e))
            for e in elements[:3]
        ]
        suffix = "" if len(elements) <= 3 else ", ..."
        return "[" + ", ".join(names) + suffix + "]"

    @staticmethod
    def _element_satisfies_scalar_edges(
        elem: Any,
        sp_set_node: "Node",
        mapping: Dict[Any, Any],
        signature_pattern: "SignaturePattern",
    ) -> bool:
        """Check a candidate tuple element against the mapping's scalar bindings.

        For every required outgoing rule ``(sp_set_node, predicate, sp_target)``
        whose ``sp_target`` is bound in ``mapping``, verifies that
        ``elem --predicate--> mapping[sp_target]`` is present in the SM
        graph. This is the per-element well-formedness check the
        ``SetStepRule`` broadcast performs at DFS time; applying it again
        at the merge boundary is idempotent for already-filtered tuples
        and correctly rejects raw (unfiltered) tuples that leak in via
        alternative enumeration paths.

        ``NoStepRule`` edges are checked as non-existence (element must
        NOT have the predicate to the bound target). ``OptionalRule``
        wrappers are unwrapped and pass if the target is unbound.
        :class:`PathRule` / :class:`AnyPathRule` are multi-hop and not
        re-checked here — correctness for those paths is deferred to the
        DFS, and a failure mode would produce no match at all rather
        than a leaked tuple.
        """
        ruleset = signature_pattern.ruleset
        elem_pred_objs = elem.get_predicate_object_pairs()
        for (sp_s, sp_p, sp_o), rule in ruleset.items():
            if sp_s is not sp_set_node:
                continue
            # Only direct (one-hop) scalar edges are re-validated here.
            if not isinstance(rule, (StepRule, NoStepRule)):
                continue
            if isinstance(rule, OptionalRule):
                continue
            sm_target = mapping.get(sp_o)
            if sm_target is None:
                # Target unbound in this mapping ⇒ cannot constrain this
                # element on this edge. Skip; the missing binding will
                # be caught by the completeness check downstream.
                continue
            target_elements = Translator._iter_binding(sm_target)
            has_edge = False
            for predicate_obj in rule._predicate:
                for p in predicate_obj.preds:
                    objs = elem_pred_objs.get(p, [])
                    for t in target_elements:
                        if t in objs:
                            has_edge = True
                            break
                    if has_edge:
                        break
                if has_edge:
                    break
            if isinstance(rule, NoStepRule):
                if has_edge:
                    return False  # forbidden edge exists ⇒ reject
            else:
                if not has_edge:
                    return False  # required edge missing ⇒ reject
        return True

    @staticmethod
    def _filter_set_bound_tuples(
        mapping: Dict[Any, Any],
        signature_pattern: "SignaturePattern",
    ) -> Optional[Dict[Any, Any]]:
        """Return ``mapping`` with every set-bound tuple filtered against
        the mapping's scalar bindings.

        This enforces the matcher's central invariant at a single choke
        point: no complete or incomplete mapping may contain a tuple
        element that lacks the required downstream edges to the scalars
        already bound in that mapping. ``_match`` merges via dict-union
        and does not re-check tuple contents; this helper makes that
        re-check explicit and local.

        Returns ``None`` if any set-bound tuple is emptied by filtering
        (the whole mapping is infeasible) or if a required tuple binding
        goes to length 0 after filtering.

        A new dict is returned when any change is made; otherwise the
        input ``mapping`` is returned unchanged (identity-preserving for
        already-canonical inputs).
        """
        set_bound_nodes = getattr(signature_pattern, "_set_bound_nodes", set())
        required_nodes = getattr(signature_pattern, "required_nodes", set())
        filtered = mapping
        made_copy = False
        for sp_n in set_bound_nodes:
            t = mapping.get(sp_n)
            if not isinstance(t, tuple) or len(t) == 0:
                continue
            surviving = tuple(
                e
                for e in t
                if Translator._element_satisfies_scalar_edges(
                    e, sp_n, mapping, signature_pattern
                )
            )
            if len(surviving) == len(t):
                continue
            if not made_copy:
                filtered = dict(mapping)
                made_copy = True
            if not surviving:
                if sp_n in required_nodes:
                    return None
                filtered[sp_n] = ()
            else:
                filtered[sp_n] = surviving
        return filtered

    @staticmethod
    def _canonical_mapping_key(mapping: Dict[Any, Any]) -> Tuple[Tuple[Any, Any], ...]:
        """Build a hashable canonical key for a SP→SM mapping.

        Two mappings produced from different enumeration starting points
        (phase-1 DFS, phase-4 merge, phase-5 optional augmentation) are
        semantically identical when every non-``None`` SP node binds to
        the same SM object (or same tuple of SM objects). Without this
        canonicalisation, the matcher emits one ``complete_matches``
        entry per successful traversal path, which explodes into
        ``n_starting_points * n_real_matches`` downstream (16→32 for
        reheat, 16→24 for damper in ``translator_example_mortar_bldg1``).

        Keying rules:
          - ``None`` bindings are omitted (a mapping that binds a
            superset of another's nodes is considered distinct — it
            contributes more information).
          - Scalar bindings are keyed by ``sm_obj.uri`` (falling back to
            ``str(sm_obj)``).
          - Tuple bindings are keyed by a tuple of per-element keys in
            their existing order (order is already canonical per
            :meth:`SetStepRule.apply`).
          - SP nodes are keyed by ``sp_node.id`` (stable string).
        """
        items = []
        for sp_n, sm_v in mapping.items():
            if sm_v is None:
                continue
            try:
                sp_key = sp_n.id
            except Exception:
                sp_key = id(sp_n)
            if isinstance(sm_v, tuple):
                sm_key: Any = tuple(
                    str(getattr(e, "uri", None) or e) for e in sm_v
                )
            else:
                sm_key = str(getattr(sm_v, "uri", None) or sm_v)
            items.append((sp_key, sm_key))
        items.sort(key=lambda t: t[0])
        return tuple(items)

    @staticmethod
    def _prune_recursive(
        sm_subject,
        sp_subject,
        candidate_maps,
        feasible,
        comparison_table,
        signature_pattern,
        verbose=False,
    ):
        signature_pattern.reset_ruleset()
        return Translator.__prune_recursive(
            sm_subject,
            sp_subject,
            candidate_maps,
            feasible,
            comparison_table,
            signature_pattern,
            verbose=verbose,
            descendant_cache=None,
        )

    @staticmethod
    def __prune_recursive(
        sm_subject,
        sp_subject,
        candidate_maps,
        feasible,
        comparison_table,
        signature_pattern,
        verbose=False,
        descendant_cache=None,
    ):
        """
        Recursively match a signature pattern node against a semantic model node (DFS).

        Traverses the subject → predicate → object structure of both graphs simultaneously.
        For each predicate-object pair in sp_subject, checks if sm_subject has matching
        predicates and objects, then recursively matches the object nodes.

        Args:
            sm_subject: Current semantic model subject node being matched. May be a
                ``tuple`` when ``sp_subject`` is set-bound (see
                :class:`SetStepRule`); in that case the function broadcasts the
                recursion per-element and aggregates parallel tuples for every
                set-bound descendant.
            sp_subject: Current signature pattern subject node being matched
            candidate_maps: List of partial SP→SM mappings being built
            feasible: Tracks which SM nodes are feasible for each SP node
            comparison_table: Tracks which SM nodes have been compared
            signature_pattern: The full signature pattern
            verbose: Print debug info
            descendant_cache: Cache of descendant mappings from successful recursions.
                Maps (sp_node, sm_node) → {child_sp: child_sm, ...} so that the
                feasible shortcut can replay descendant values without re-recursing.

        Returns:
            (candidate_maps, feasible, comparison_table, is_pruned)
        """
        LOGGER.debug("Entering prune_recursive")
        LOGGER.add_level()
        LOGGER.debug(lambda: Translator._get_node_string(sp_subject, sm_subject))

        if descendant_cache is None:
            descendant_cache = {}

        # Broadcast entry: if ``sm_subject`` is a tuple, the caller has bound
        # ``sp_subject`` to a set of SM objects (via :class:`SetStepRule` or
        # closure propagation). Recurse per element and then aggregate
        # parallel tuples at every set-bound descendant before returning.
        if isinstance(sm_subject, tuple):
            result = Translator.__broadcast_recurse(
                sm_subject,
                sp_subject,
                candidate_maps,
                feasible,
                comparison_table,
                signature_pattern,
                verbose,
                descendant_cache,
            )
            LOGGER.remove_level()
            return result

        # Initialize tracking sets for current subject
        feasible.setdefault(sp_subject, set()).add(sm_subject)
        comparison_table.setdefault(sp_subject, set()).add(sm_subject)

        # Get predicate → objects mappings from both subject nodes
        sm_predicate_objects = sm_subject.get_predicate_object_pairs()
        sp_predicate_objects = sp_subject.predicate_object_pairs
        ruleset = signature_pattern.ruleset

        # Process each predicate-object pair required by the SP subject
        for sp_predicate, sp_objects in sp_predicate_objects.items():
            for sp_object in sp_objects:
                # Per-rule output bucket.  Resetting per rule (rather than
                # sharing one ``valid_maps`` across siblings) is what lets
                # ``candidate_maps = valid_maps`` below correctly *replace*
                # the in-flight maps with this rule's extended outputs;
                # otherwise the prior rule's pre-extension snapshots would
                # leak through and pollute downstream consumers (notably
                # :func:`__broadcast_recurse`, which only takes
                # ``child_maps[0]`` per element and would happily pick up
                # a stale partial map).
                valid_maps = []
                rule = ruleset[
                    (sp_subject, sp_predicate, sp_object)
                ]  # NOTE: Q: What happens if we have added multiple rules for the same subject, predicate, object? A: This would not be meaningful

                LOGGER.debug("Rule: %s", rule.__class__.__name__)

                # Collect SM objects from ALL predicates in the Predicate (cross-ontology matching)
                sm_objects = []
                for predicate in rule._predicate:  # Iterate tuple of Predicate objects
                    for (
                        pred
                    ) in predicate.preds:  # Iterate tuple of SemanticPredicate objects
                        pred_objects = sm_predicate_objects.get(pred, [])
                        sm_objects.extend(pred_objects)
                # Remove duplicates while preserving order
                seen = set()
                sm_objects = [x for x in sm_objects if not (x in seen or seen.add(x))]

                # Check if SM subject has any of the predicates with objects
                if sm_objects:
                    rule_pairs, _, _, ruleset = rule.apply(
                        sm_subject, sm_objects, ruleset, candidate_maps=candidate_maps
                    )

                    match_found = False
                    for (
                        maps_for_pair,
                        matched_sm_object,
                        matched_sp_object,
                        matched_type,
                        _,
                    ) in rule_pairs:
                        LOGGER.debug("Entered inner loop")
                        LOGGER.debug(
                            lambda: Translator._get_node_string(
                                matched_sp_object, matched_sm_object
                            )
                        )

                        # SetStepRule path: ``matched_sm_object`` is a tuple
                        # of all SM objects satisfying the predicate. Emit a
                        # single branch with tuple binding and broadcast
                        # downstream per element.
                        if isinstance(matched_sm_object, tuple):
                            feasible.setdefault(matched_sp_object, set())
                            comparison_table.setdefault(matched_sp_object, set())
                            for elem in matched_sm_object:
                                feasible[matched_sp_object].add(elem)
                                comparison_table[matched_sp_object].add(elem)

                            # Write the tuple binding into each candidate map
                            # so descendant broadcasting sees the full set.
                            for m in maps_for_pair:
                                m[matched_sp_object] = matched_sm_object

                            child_maps, feasible, comparison_table, is_pruned = (
                                Translator.__broadcast_recurse(
                                    matched_sm_object,
                                    matched_sp_object,
                                    maps_for_pair,
                                    feasible,
                                    comparison_table,
                                    signature_pattern,
                                    verbose,
                                    descendant_cache,
                                )
                            )
                            if not is_pruned:
                                valid_maps.extend(child_maps)
                                match_found = True
                            continue

                        # Initialize tracking for matched object
                        feasible.setdefault(matched_sp_object, set())
                        comparison_table.setdefault(matched_sp_object, set())

                        if matched_sm_object not in comparison_table[matched_sp_object]:
                            # New comparison - recurse (object becomes subject in next level)
                            comparison_table[matched_sp_object].add(matched_sm_object)
                            child_maps, feasible, comparison_table, is_pruned = (
                                Translator.__prune_recursive(
                                    matched_sm_object,
                                    matched_sp_object,
                                    maps_for_pair,
                                    feasible,
                                    comparison_table,
                                    signature_pattern,
                                    verbose,
                                    descendant_cache=descendant_cache,
                                )
                            )

                            if not is_pruned:
                                # Cache descendant mappings so the feasible
                                # shortcut can replay them without re-recursing.
                                if child_maps:
                                    ref = child_maps[0]
                                    descendants = {
                                        sp_n: sm_n
                                        for sp_n, sm_n in ref.items()
                                        if sm_n is not None
                                    }
                                    descendant_cache[
                                        (matched_sp_object, matched_sm_object)
                                    ] = descendants

                                # Early stop for SinglePath/MultiPath with Exact match
                                if (
                                    isinstance(rule, (PathRule, AnyPathRule))
                                    and rule.stop_early
                                    and matched_type == StepRule
                                ):
                                    valid_maps.extend(child_maps)
                                    match_found = True
                                    break

                                if match_found:
                                    LOGGER.debug(
                                        f'Multiple matches: "{sp_subject.id}" -> "{sm_subject.uri}"'
                                    )
                                valid_maps.extend(child_maps)
                                match_found = True

                        elif matched_sm_object in feasible[matched_sp_object]:
                            # Already matched and feasible - reuse result
                            # Also replay descendant values cached from the
                            # first successful recursion into this node.
                            cached = descendant_cache.get(
                                (matched_sp_object, matched_sm_object), {}
                            )
                            for m in maps_for_pair:
                                m[matched_sp_object] = matched_sm_object
                                for sp_n, sm_n in cached.items():
                                    if m.get(sp_n) is None:
                                        m[sp_n] = sm_n
                            valid_maps.extend(maps_for_pair)
                            match_found = True

                    # Prune if required rule had no match
                    if not match_found and not isinstance(rule, OptionalRule):
                        feasible[sp_subject].discard(sm_subject)
                        LOGGER.debug("Pruned (no match found)")
                        LOGGER.debug(
                            lambda: Translator._get_node_string(sp_subject, sm_subject)
                        )
                        LOGGER.remove_level()
                        return candidate_maps, feasible, comparison_table, True

                    # Only consume this rule's outputs when it actually
                    # matched.  ``OptionalRule`` may legitimately produce
                    # no pairs (e.g. the optional point is absent on this
                    # SM subject); in that case keep ``candidate_maps``
                    # as-is so subsequent siblings still see the prior
                    # bindings.
                    if match_found:
                        candidate_maps = valid_maps

                else:
                    # No predicates matched - prune if rule is required
                    if not isinstance(rule, OptionalRule):
                        feasible[sp_subject].discard(sm_subject)
                        LOGGER.debug("Pruned (missing predicate): %s", sp_predicate)
                        LOGGER.debug(
                            lambda: Translator._get_node_string(sp_subject, sm_subject)
                        )
                        LOGGER.remove_level()
                        return candidate_maps, feasible, comparison_table, True

        # Success - add current subject mapping
        if not candidate_maps:
            candidate_maps = [{n: None for n in signature_pattern.nodes}]

        candidate_maps = Translator._copy_nodemap_list(candidate_maps)
        for mapping in candidate_maps:
            mapping[sp_subject] = sm_subject

        LOGGER.debug("Returning from prune_recursive")
        LOGGER.add_level()
        LOGGER.debug(lambda: Translator._get_maps_string(candidate_maps))
        LOGGER.remove_level()

        LOGGER.remove_level()

        return candidate_maps, feasible, comparison_table, False

    @staticmethod
    def __broadcast_recurse(
        sm_tuple,
        sp_subject,
        candidate_maps,
        feasible,
        comparison_table,
        signature_pattern,
        verbose,
        descendant_cache,
    ):
        """Broadcast :meth:`__prune_recursive` over a set-bound subject.

        Runs one recursion per element of ``sm_tuple`` with ``sp_subject``
        as the SP side. For every downstream SP node that participates in
        the recursion and is marked set-bound on ``signature_pattern``,
        the per-element bindings are **aligned into a parallel tuple**
        (same length and order as ``sm_tuple``). Scalar descendants reachable
        from the set-bound subject are expected to agree across elements
        (shared resources); inconsistent scalars leave the node unbound.

        Element-level semantics
        -----------------------
        An element is **filtered out** of the broadcast's surviving tuple if
        its per-element recursion prunes (e.g. a downstream ``StepRule``
        requires a triple that the element lacks). This matches the idiomatic
        use of ``SetStepRule`` for collecting e.g. "all commands of this VAV
        that have a Brick timeseries reference" — logical-flag commands
        without timeseries are silently dropped rather than aborting the
        whole match. If **every** element prunes, the broadcast prunes as a
        whole.
        """
        set_bound_nodes = signature_pattern._set_bound_nodes
        aggregated_maps: List[Dict[Node, Any]] = []

        LOGGER.debug(
            "Broadcasting %s over %d elements",
            sp_subject.id,
            len(sm_tuple),
        )

        _diag = _match_diag_enabled(signature_pattern)
        if _diag:
            _match_diag_write(
                f"[BRCAST] enter pattern={signature_pattern.id} "
                f"sp_subject={sp_subject.id} n_elements={len(sm_tuple)} "
                f"elements={_diag_sm_name(sm_tuple)}"
            )

        # Mirror __prune_recursive's default: at least one candidate map
        # must exist for the per-element recursion to produce an output.
        if not candidate_maps:
            candidate_maps = [{n: None for n in signature_pattern.nodes}]

        for base_idx, base_map in enumerate(candidate_maps):
            # Snapshot of the parent map; each element recursion starts
            # from a copy and writes its scalar binding at sp_subject.
            # Elements whose recursion prunes are filtered out of the
            # surviving tuple (see docstring: filter semantics).
            per_elem_maps: List[Dict[Node, Any]] = []
            surviving_elements: List[Any] = []
            pruned_elements_repr: List[str] = []
            for elem in sm_tuple:
                elem_map = dict(base_map)
                elem_map[sp_subject] = elem
                elem_maps_input = [elem_map]
                child_maps, feasible, comparison_table, is_pruned = (
                    Translator.__prune_recursive(
                        elem,
                        sp_subject,
                        elem_maps_input,
                        feasible,
                        comparison_table,
                        signature_pattern,
                        verbose,
                        descendant_cache=descendant_cache,
                    )
                )
                if _diag:
                    if is_pruned:
                        _match_diag_write(
                            f"[BRCAST]   base={base_idx} elem={_diag_sm_name(elem)} "
                            f"FILTERED-OUT (downstream rule failed)"
                        )
                    else:
                        rep = child_maps[0] if child_maps else elem_map
                        _match_diag_write(
                            f"[BRCAST]   base={base_idx} elem={_diag_sm_name(elem)} "
                            f"OK child={_diag_mapping_summary(rep)}"
                        )
                if is_pruned:
                    pruned_elements_repr.append(_diag_sm_name(elem))
                    continue
                surviving_elements.append(elem)
                # If downstream recursion branched (StepRule on scalar
                # children under a set-bound parent shouldn't happen by
                # closure, but be defensive), take the first branch as the
                # representative for this element.
                per_elem_maps.append(child_maps[0] if child_maps else elem_map)

            # If EVERY element was filtered out, the broadcast as a whole
            # has no valid assertion and must prune.
            if not surviving_elements:
                if _diag:
                    _match_diag_write(
                        f"[BRCAST] WHOLE-BROADCAST PRUNED pattern={signature_pattern.id} "
                        f"sp_subject={sp_subject.id} "
                        f"all_elements_filtered={pruned_elements_repr}"
                    )
                return candidate_maps, feasible, comparison_table, True

            surviving_tuple = tuple(surviving_elements)

            merged = dict(base_map)
            # Canonical form: the subject binds to the tuple of elements
            # that actually survived the broadcast (filter semantics).
            merged[sp_subject] = surviving_tuple

            # Union of all SP nodes that received a value in any surviving
            # element's per-element recursion.
            touched_sp_nodes: set = set()
            for cm in per_elem_maps:
                for sp_n, v in cm.items():
                    if v is not None and sp_n is not sp_subject:
                        touched_sp_nodes.add(sp_n)

            for sp_n in touched_sp_nodes:
                # A value that was already bound in ``base_map`` is
                # INHERITED by every per-element map via ``dict(base_map)``
                # above. It must be preserved as-is — treating it as a
                # per-element contribution and re-aggregating would flatten
                # an inherited ``tuple_len_k`` into a ``tuple_len_{k * n_elem}``
                # (the "tuple doubling" bug). Only NEW contributions from
                # this broadcast's element recursions are aggregated.
                base_val = base_map.get(sp_n)
                if base_val is not None:
                    merged[sp_n] = base_val
                    continue

                values = [cm.get(sp_n) for cm in per_elem_maps]
                if sp_n in set_bound_nodes:
                    # Newly-contributed set-bound descendant: one scalar
                    # contribution per element (nested SetStepRule under a
                    # set-bound subject is out of scope per ``SetStepRule``
                    # docstring). Build a parallel tuple; defensive flatten
                    # for any stray tuple inputs.
                    flat: List[Any] = []
                    for v in values:
                        if v is None:
                            continue
                        if isinstance(v, tuple):
                            flat.extend(v)
                        else:
                            flat.append(v)
                    if flat:
                        merged[sp_n] = tuple(flat)
                else:
                    # Scalar descendant off a set-bound subject: require
                    # consensus across elements (shared resource).
                    non_none = [v for v in values if v is not None]
                    if non_none and all(v == non_none[0] for v in non_none):
                        merged[sp_n] = non_none[0]

            aggregated_maps.append(merged)

        if not aggregated_maps:
            aggregated_maps = [{n: None for n in signature_pattern.nodes}]
            for mapping in aggregated_maps:
                mapping[sp_subject] = sm_tuple

        if _diag:
            for mi, m in enumerate(aggregated_maps):
                _match_diag_write(
                    f"[BRCAST] OK pattern={signature_pattern.id} "
                    f"sp_subject={sp_subject.id} out={mi}/{len(aggregated_maps)} "
                    f"aggregated={_diag_mapping_summary(m)}"
                )

        return aggregated_maps, feasible, comparison_table, False

    @staticmethod
    def _has_sm_edge(sm_subj: Any, predicate: "Predicate", sm_obj: Any) -> bool:
        """Return ``True`` iff some element of ``sm_subj`` has an SM-side
        edge labelled by *any* ``SemanticPredicate`` in ``predicate.preds``
        landing on some element of ``sm_obj``.

        Tuple-bound endpoints follow the "any element suffices"
        semantics used elsewhere in the merge logic
        (cf. :meth:`_check_edge_connectivity`): a single matching edge
        between any pair of elements is enough to consider the SP edge
        honoured.  ``None`` on either side trivially means "no edge".
        """
        if sm_subj is None or sm_obj is None or predicate is None:
            return False
        obj_elems = set(Translator._iter_binding(sm_obj))
        if not obj_elems:
            return False
        for s_elem in Translator._iter_binding(sm_subj):
            s_pos = s_elem.get_predicate_object_pairs()
            for p in predicate.preds:
                children = s_pos.get(p, [])
                if any(c in obj_elems for c in children):
                    return True
        return False

    @staticmethod
    def _validate_binding_against_merged(
        merged_group: Dict[Node, Any],
        sp_node: Node,
        sm_node: Any,
        signature_pattern,
    ) -> bool:
        """Reject a merge candidate when the SP-edge structure links
        ``sp_node`` to an already-bound node but the SM doesn't honour
        the edge.

        Phase-4 merge has two paths that fill bindings into
        ``merged_group`` (connected and disconnected -- see
        :meth:`_match`).  The connected path's
        :meth:`_validate_merge` only re-runs :meth:`_prune_recursive`
        rooted *at* the new binding, which walks SP edges *downstream*
        from ``sp_node``.  The disconnected path adds bindings one at a
        time and likewise relies on the same downstream-only check.
        Neither path verifies upstream SP edges
        (``existing_subj --pred--> sp_node``), so an
        :class:`OptionalRule` slot can be filled from an unrelated SM
        neighbourhood -- e.g. AHU02's
        ``Supply_Air_Temperature_Setpoint`` getting filled with
        AHU01's setpoint URI from a partial rooted at the SAT
        sm-node, despite AHU02 having no ``hasPoint`` triple to that
        URI in the BMS graph.

        This validator iterates :attr:`SignaturePattern.ruleset` and,
        for every rule whose subject *or* object is already bound in
        ``merged_group`` and whose other endpoint equals ``sp_node``,
        requires :meth:`_has_sm_edge` between the bound endpoint and
        ``sm_node`` to hold.  Returns ``False`` (i.e. reject the
        binding) on the first violation.
        """
        for (subj, pred, obj), _rule in signature_pattern.ruleset.items():
            if pred is None:
                continue
            # ``_has_sm_edge`` only inspects a single direct triple, so
            # this validator can only soundly reject single-hop rules.
            # Multi-hop rules (``PathRule``, ``AnyPathRule``,
            # ``SetAnyPathRule``) are already validated by the main
            # ``_prune_recursive`` walk which performs the actual
            # transitive traversal; running the single-edge check on
            # them would falsely reject every legitimate multi-hop
            # binding (e.g. an SAREF-style ``Damper hasFluidSuppliedBy
            # Coil`` chain that goes damper → port → sensor → coil in
            # the BMS graph).  ``OptionalRule`` is also skipped here
            # because the prune walk treats it as best-effort and
            # demanding the upstream edge would defeat the optionality.
            if isinstance(_rule, (PathRule, AnyPathRule, SetAnyPathRule)):
                continue
            if not isinstance(_rule, (StepRule, NoStepRule)):
                continue
            if isinstance(_rule, OptionalRule):
                continue
            # Upstream: ``subj`` is already in merged_group, we're
            # binding ``obj == sp_node``.  Require an SM edge from the
            # bound subject to ``sm_node`` via ``pred``.
            if obj is sp_node and subj is not sp_node:
                subj_sm = merged_group.get(subj)
                if subj_sm is None:
                    continue
                if not Translator._has_sm_edge(subj_sm, pred, sm_node):
                    return False
            # Downstream: ``obj`` is already in merged_group, we're
            # binding ``subj == sp_node``.  Require an SM edge from
            # ``sm_node`` to the bound object via ``pred``.
            elif subj is sp_node and obj is not sp_node:
                obj_sm = merged_group.get(obj)
                if obj_sm is None:
                    continue
                if not Translator._has_sm_edge(sm_node, pred, obj_sm):
                    return False
        return True

    @staticmethod
    def _check_edge_connectivity(
        source_mapping: Dict[Node, Any],
        target_mapping: Dict[Node, Any],
        reverse: bool = False,
    ) -> bool:
        """
        Check if two partial mappings share a connecting edge in the semantic model.

        Args:
            source_mapping: Dict of SP→SM mappings (non-None values only)
            target_mapping: Dict to check connectivity against
            reverse: If True, check if target's nodes are in source's edges
                    If False, check if target's nodes exist in source's edges

        Returns:
            bool: True if mappings share a connecting edge
        """
        for sp_node, sm_node in source_mapping.items():
            # Broadcast tuple-bound source nodes: any element that
            # establishes the edge counts as connected. This keeps the
            # scalar fast-path untouched.
            sm_elements = Translator._iter_binding(sm_node)
            for sm_elem in sm_elements:
                sm_predicates = sm_elem.get_predicate_object_pairs()

                for predicate, sp_objects in sp_node.predicate_object_pairs.items():
                    sm_children = sm_predicates.get(predicate, [])
                    if not sm_children:
                        continue

                    for sp_object in sp_objects:
                        target_sm_node = target_mapping.get(sp_object)
                        if target_sm_node is None:
                            continue

                        # Target may itself be tuple-bound. Any single
                        # element satisfying the edge suffices; this
                        # matches the semantics of set-bound bindings
                        # ("all of these simultaneously, any of them
                        # covers").
                        target_elements = Translator._iter_binding(target_sm_node)

                        if reverse:
                            for t in target_elements:
                                if sm_children == t:
                                    return True
                        else:
                            for t in target_elements:
                                if t in sm_children:
                                    return True

        return False

    @staticmethod
    def _validate_merge(base_group, nodes_to_add, signature_pattern):
        """
        Validate that merging nodes into base_group respects pattern rules.

        Args:
            base_group: The base partial mapping
            nodes_to_add: Dict of SP→SM mappings to validate for merging
            signature_pattern: The signature pattern

        Returns:
            bool: True if merge is valid (no pruning occurred)
        """
        # feasible = {n: set() for n in signature_pattern.nodes}
        # comparison_table = {n: set() for n in signature_pattern.nodes}

        for sp_node, sm_node in nodes_to_add.items():
            if sm_node is None:
                continue

            feasible = {n: set() for n in signature_pattern.nodes}
            comparison_table = {n: set() for n in signature_pattern.nodes}

            _, _, _, is_pruned = Translator._prune_recursive(
                sm_node,
                sp_node,
                [],
                feasible,
                comparison_table,
                signature_pattern,
            )

            if is_pruned:
                return False

        return True

    # @profile
    @staticmethod
    def _match(
        group_a,
        group_b,
        signature_pattern,
        complete_matches,
        incomplete_matches,
    ):
        """
        Attempt to merge two partial SP→SM mappings.

        Tries three strategies:
        1. Forward: Check if group_b connects to group_a via edges
        2. Backward: Check if group_a connects to group_b via edges
        3. Disconnected: Merge compatible mappings without edge connectivity
           (for patterns with isolated subgraphs)

        If merge succeeds and is complete:
        - merged_group is added to complete_matches
        - For connected merges: both groups are removed from incomplete_matches
        - For disconnected merges: groups WITHOUT modeled_node filled are preserved

        Disconnected merges preserve groups that don't have the modeled_node filled,
        as these represent shared resources (like Weather_Station) that should be
        reusable across many instances of the modeled entity (like spaces).

        Args:
            group_a: First partial SP→SM mapping
            group_b: Second partial SP→SM mapping
            signature_pattern: The signature pattern being matched
            complete_matches: List of complete mappings (mutated if merge completes)
            incomplete_matches: List of incomplete mappings (mutated if merge completes)

        Returns:
            bool: True if merge was successful, False otherwise.
        """
        LOGGER.debug("Matching groups")
        LOGGER.add_level()
        LOGGER.debug(lambda: Translator._get_maps_string([group_a, group_b]))

        _diag = _match_diag_enabled(signature_pattern)
        if _diag:
            _match_diag_write(
                f"[MERGE] attempt pattern={signature_pattern.id} "
                f"A={_diag_mapping_summary(group_a)} "
                f"B={_diag_mapping_summary(group_b)}"
            )

        # Cache nodes list once (avoid repeated property access with assertion)
        sp_nodes = signature_pattern._nodes

        # Check compatibility and count new contributions in a single pass
        # Avoids multiple dict lookups per node and allows early exit
        is_compatible = True
        new_contributions = 0
        for n in sp_nodes:
            val_a = group_a.get(n)
            val_b = group_b.get(n)
            if val_a is not None and val_b is not None:
                if val_a != val_b:
                    is_compatible = False
                    break  # Early exit - incompatible means no merge possible
            elif val_a is None and val_b is not None:
                new_contributions += 1

        LOGGER.debug(
            "Compatibility check: %s", "pass" if is_compatible else "fail"
        )

        # Early exit if incompatible - no merge possible
        if not is_compatible:
            if _diag:
                _match_diag_write(
                    "[MERGE]   REJECT incompatible (conflicting bindings)"
                )
            LOGGER.remove_level()
            return False

        # Extract non-None mappings
        nodes_a = {k: v for k, v in group_a.items() if v is not None}
        nodes_b = {k: v for k, v in group_b.items() if v is not None}

        LOGGER.debug("Group B contributes %d new mappings", new_contributions)
        if new_contributions == 0:
            LOGGER.debug("No new contributions, skipping merge")
            if _diag:
                _match_diag_write(
                    "[MERGE]   REJECT no-new-contributions"
                )
            LOGGER.remove_level()
            return False

        merged_group = None

        # Strategy 1 & 2: Check edge connectivity in both directions
        # (is_compatible is guaranteed True here - we returned early if False)
        has_edge_b_to_a = Translator._check_edge_connectivity(
            nodes_b, group_a, reverse=False
        )
        has_edge_a_to_b = Translator._check_edge_connectivity(
            nodes_a, group_b, reverse=True
        )
        LOGGER.debug(
            "Edge connectivity - B->A: %s, A->B: %s", has_edge_b_to_a, has_edge_a_to_b
        )

        # Track whether we used a connected or disconnected merge strategy
        # and which groups contain only isolated nodes (can be reused for other merges)
        used_disconnected_merge = False
        groups_to_preserve = []  # Will contain groups with only isolated nodes

        if has_edge_b_to_a or has_edge_a_to_b:
            LOGGER.debug("Attempting connected merge validation")
            if Translator._validate_merge(group_a, nodes_b, signature_pattern):
                # Issue 4 guard: ``_validate_merge`` only walks SP edges
                # *downstream* of each new binding.  Reject the merge
                # if any new binding has an upstream SP edge to an
                # already-merged node that the SM doesn't honour.
                upstream_ok = all(
                    Translator._validate_binding_against_merged(
                        group_a, sp_n, sm_n, signature_pattern
                    )
                    for sp_n, sm_n in nodes_b.items()
                )
                if upstream_ok:
                    merged_group = {**group_a, **nodes_b}
                    LOGGER.debug("Connected merge successful")
                else:
                    LOGGER.debug(
                        "Connected merge rejected: upstream SP edge to "
                        "merged group lacks SM backing"
                    )
                    if _diag:
                        _match_diag_write(
                            "[MERGE]   REJECT connected upstream-edge "
                            "missing-in-SM"
                        )
            else:
                LOGGER.debug("Connected merge validation failed")

        # Strategy 3: Disconnected merge (for patterns with isolated subgraphs)
        if merged_group is None:
            LOGGER.debug("Attempting disconnected merge")
            merged_group = dict(group_a)
            used_disconnected_merge = True

            # For disconnected merges, only preserve groups that DON'T have the modeled_node filled.
            # Groups with modeled_node filled represent unique instances (like a specific space)
            # and should be consumed. Groups without it (like weather_station) represent shared
            # resources that can be reused across many spaces.
            # Expand any ModeledNode group to its plain SP-side members so
            # that ``group_a.get(node)`` actually finds the binding — a
            # ModeledNode composite is never a key in ``group_*``; its
            # members are.
            modeled_nodes = list(signature_pattern.iter_modeled_sp_nodes())
            groups_to_preserve = []

            # Check if group_a has any modeled_node filled - if not, it's a shared resource
            a_has_modeled = any(group_a.get(mn) is not None for mn in modeled_nodes)
            b_has_modeled = any(group_b.get(mn) is not None for mn in modeled_nodes)

            if not a_has_modeled:
                groups_to_preserve.append(group_a)
                LOGGER.debug("Group A has no modeled_node filled, preserving for reuse")
            if not b_has_modeled:
                groups_to_preserve.append(group_b)
                LOGGER.debug("Group B has no modeled_node filled, preserving for reuse")

            for sp_node, sm_node in nodes_b.items():
                # Use cached sp_nodes instead of signature_pattern.nodes
                feasible = {n: set() for n in sp_nodes}
                comparison_table = {n: set() for n in sp_nodes}
                result_maps, _, _, is_pruned = Translator._prune_recursive(
                    sm_node,
                    sp_node,
                    [],
                    feasible,
                    comparison_table,
                    signature_pattern,
                )

                if is_pruned:
                    merged_group = None
                    used_disconnected_merge = False
                    groups_to_preserve = []
                    break

                # Issue 4 guard: ``_prune_recursive`` walks SP edges
                # downstream of ``sp_node`` only.  The disconnected
                # fallback exists for genuinely independent
                # sub-graphs (e.g. one Weather_Station shared across
                # many spaces); it must NOT silently fill a slot whose
                # SP-edge structure ties it to an already-bound node
                # via an SM edge that doesn't actually exist (e.g.
                # AHU02 inheriting AHU01's
                # ``Supply_Air_Temperature_Setpoint``).
                if not Translator._validate_binding_against_merged(
                    merged_group, sp_node, sm_node, signature_pattern
                ):
                    if _diag:
                        _match_diag_write(
                            "[MERGE]   REJECT disconnected "
                            f"sp_node={sp_node.id} "
                            f"sm_node={_diag_sm_name(sm_node)} "
                            "upstream-edge missing-in-SM"
                        )
                    merged_group = None
                    used_disconnected_merge = False
                    groups_to_preserve = []
                    break

                # Only add if actually matched (not skipped by Optional_ rule)
                if result_maps and all(sm_node == m.get(sp_node) for m in result_maps):
                    merged_group[sp_node] = sm_node

        # If merge successful, check if complete
        if merged_group is not None:
            LOGGER.debug("Merge successful, checking completeness")

            # Enforce the set-bound-tuple invariant at the merge boundary:
            # every tuple element must have the required downstream edges
            # to whatever scalars are bound in the merged mapping. Raw
            # ``SetStepRule`` tuples that slipped through an alternative
            # enumeration path (e.g. via ``_check_edge_connectivity``'s
            # any-element semantics, or a parallel incomplete branch
            # whose scalar descendants disagreed and so didn't filter the
            # tuple) are caught and trimmed here.
            filtered_group = Translator._filter_set_bound_tuples(
                merged_group, signature_pattern
            )
            if filtered_group is None:
                if _diag:
                    _match_diag_write(
                        "[MERGE]   REJECT set-bound-infeasible "
                        "(tuple emptied by scalar-edge filter)"
                    )
                LOGGER.debug(
                    "Merge rejected: set-bound tuple infeasible against scalar edges"
                )
                LOGGER.remove_level()
                return False
            merged_group = filtered_group

            # Remove groups from incomplete_matches, but preserve groups from disconnected merges
            # (e.g., one Weather_Station subgraph can be reused for many spaces)
            if group_a in incomplete_matches and group_a not in groups_to_preserve:
                incomplete_matches.remove(group_a)
            if group_b in incomplete_matches and group_b not in groups_to_preserve:
                incomplete_matches.remove(group_b)

            # Ensure preserved groups are in incomplete_matches (they might not have been added yet,
            # e.g., new_mapping in _try_merge_with_incomplete)
            for group in groups_to_preserve:
                if group not in incomplete_matches:
                    incomplete_matches.append(group)
                    LOGGER.debug(
                        "Added preserved group to incomplete_matches for reuse"
                    )

            is_complete = all(
                merged_group.get(n) is not None
                for n in signature_pattern.required_nodes
            )
            LOGGER.debug(
                "Merged group is %s", "complete" if is_complete else "incomplete"
            )

            if is_complete:
                if _diag:
                    _match_diag_write(
                        f"[MERGE]   ACCEPT complete "
                        f"strategy={'disconnected' if used_disconnected_merge else 'connected'} "
                        f"merged={_diag_mapping_summary(merged_group)}"
                    )
                complete_matches.append(merged_group)
                # LOGGER.debug("Added to complete matches")
                LOGGER.info(
                    "Match found: %s",
                    signature_pattern.id,
                )
                LOGGER.add_level()
                LOGGER.info(
                    lambda: Translator._get_map_string(
                        merged_group, LOGGER.info
                    )
                )
                LOGGER.remove_level()
            else:
                if _diag:
                    _match_diag_write(
                        f"[MERGE]   ACCEPT incomplete "
                        f"strategy={'disconnected' if used_disconnected_merge else 'connected'} "
                        f"merged={_diag_mapping_summary(merged_group)}"
                    )
                incomplete_matches.append(merged_group)
                LOGGER.debug("Added to incomplete matches")

            LOGGER.remove_level()
            return True

        if _diag:
            _match_diag_write(
                "[MERGE]   REJECT no-valid-merge "
                "(connected validation failed and disconnected path produced no binding)"
            )
        LOGGER.remove_level()
        return False


class Node:
    node_instance_count = count()

    def __init__(
        self,
        cls: Union[Any, Tuple[Any, ...], List[Any], str],
        graph_name: Optional[str] = None,
        hash_: Optional[Any] = None,
    ) -> None:
        self._graph_name = graph_name
        if isinstance(cls, tuple) == False:
            if isinstance(cls, (list, set)):
                cls = tuple(cls)
            else:
                cls = (cls,)
        self.cls = cls
        self.predicate_object_pairs = {}
        self._signature_pattern = None
        self._id = self.make_id()

        if hash_ is not None:
            self._hash = hash(hash_)

    def __hash__(self):
        if hasattr(self, "_hash"):
            return self._hash
        return id(self)

    def __eq__(self, other):
        if isinstance(other, Node) and hasattr(self, "_hash") and hasattr(other, "_hash"):
            return self._hash == other._hash
        return self is other

    @property
    def id(self):
        return self._id

    @property
    def signature_pattern(self):
        return self._signature_pattern

    @property
    def semantic_model(self):
        """Get the semantic model associated with this node"""
        return self.signature_pattern.semantic_model

    def __str__(self):
        return self.id

    def __repr__(self):
        return f"Node({str(self.id)})"  # NOTE:

    def validate_cls(self):
        if self._signature_pattern is None:
            raise ValueError("No signature pattern set.")

        cls = self.cls
        if isinstance(cls, tuple) == False:
            cls = (cls,)

        cls_ = []
        for c in cls:
            if c is core.BlankNode:
                cls_.append(core.BlankNode)  # Sentinel — matched in SemanticInstance.isinstance()
            elif isinstance(c, core.SemanticType):
                cls_.append(c)
            elif isinstance(c, URIRef):
                cls_.append(core.SemanticType(c, self.signature_pattern.semantic_model))
            elif isinstance(c, str):
                cls_.append(
                    core.SemanticType(URIRef(c), self.signature_pattern.semantic_model)
                )
            else:
                raise ValueError(f"Invalid class type: {type(c)}")

        self.cls = tuple(cls_)  # Make immutable
        self._id = self.make_id()

    def make_id(self):
        # Join class URIs with underscore separator to create a valid URI identifier
        # This avoids creating invalid URIs like http://twin4build.org/['...', '...']
        # Extract local names (fragment or last path component) for URI-safe identifiers
        def get_local_name(uri_str):
            # Try fragment first (part after #)
            if "#" in uri_str:
                return uri_str.split("#")[-1]
            # Otherwise use last path component
            return uri_str.split("/")[-1]

        parts = []
        for s in self.cls:
            if s is core.BlankNode:
                parts.append("BlankNode")
            elif hasattr(s, "uri"):
                uri_str = str(s.uri)
                parts.append(get_local_name(uri_str))
            else:
                uri_str = str(s)
                parts.append(get_local_name(uri_str))

        return "_".join(parts)

    def set_signature_pattern(self, signature_pattern):
        """Set the signature pattern for this node"""
        self._signature_pattern = signature_pattern

    def get_type_attributes(self):
        attr = {}
        for c in self.cls:
            if c is not core.BlankNode:
                attr.update(c.get_type_attributes())
        return attr


class ModeledNode(Node):
    """Composite modeled identity: a group of SP-side ``Node``\\ s that jointly
    identify an entity which does not appear as a single instance in the
    semantic model (e.g. an implicit VAV controller that the BRICK graph
    describes only via its VAV, sensor, setpoint, and command points).

    Implicit registration
    ---------------------
    Constructing a ``ModeledNode`` from already-registered member ``Node``\\ s
    auto-binds it to the members' shared ``SignaturePattern``\\ . No explicit
    ``sp.add_modeled_node(...)`` call is needed, mirroring how plain ``Node``\\ s
    auto-register via ``add_triple``:

    >>> vav       = Node(cls=BRICK.VAV)
    >>> sensors   = Node(cls=BRICK.Zone_Temp_Sensor)
    >>> setpoints = Node(cls=BRICK.Zone_Temp_Setpoint)
    >>> actuators = Node(cls=BRICK.Valve_Command)
    >>> sp.add_triple(Exact(vav, sensors,   BRICK.hasPoint))
    >>> sp.add_triple(Exact(vav, setpoints, BRICK.hasPoint))
    >>> sp.add_triple(Exact(vav, actuators, BRICK.hasPoint))
    >>> ModeledNode([vav, sensors, setpoints, actuators])

    Relational fingerprint
    ----------------------
    At MILP match time, a ``ModeledNode`` resolves to a stable fingerprint
    that combines the matched member IRIs with the matched (subject,
    predicate, object) triples among those members. This gives a
    context-addressable identity: two groups with the same members but
    different relational shape receive different fingerprints.

    Mutex semantics
    ---------------
    Members of a ``ModeledNode`` group are **non-exclusive**: individual
    member SM nodes remain available to other systems (e.g. a ``SensorSystem``
    can claim the same ``BRICK.Command`` node that a controller's group
    already covers). Exclusivity is enforced per-fingerprint instead:
    at most one component per (members + relations) context.
    """

    def __init__(self, members: List["Node"]) -> None:
        assert len(members) > 0, "ModeledNode requires at least 1 member."

        sp = None
        for m in members:
            assert isinstance(
                m, Node
            ), f"All ModeledNode members must be Node instances, got {type(m).__name__}."
            assert not isinstance(m, ModeledNode), (
                "ModeledNode members must be plain Node instances, not nested "
                "ModeledNodes."
            )
            assert m._signature_pattern is not None, (
                f"ModeledNode member {m} has no SignaturePattern. Register it via "
                "sp.add_triple / sp.add_node before constructing a ModeledNode."
            )
            if sp is None:
                sp = m._signature_pattern
            else:
                assert m._signature_pattern is sp, (
                    "All ModeledNode members must share the same SignaturePattern."
                )

        self.members = list(members)

        # Aggregate cls as the ordered union of member cls tuples, so any
        # downstream code that introspects ``modeled_node.cls`` sees a
        # consistent type envelope.
        seen: List[Any] = []
        for m in members:
            for c in m.cls:
                if c not in seen:
                    seen.append(c)
        self.cls = tuple(seen)

        self._graph_name = None
        self.predicate_object_pairs = {}
        self._signature_pattern = sp

        # SP-space id: deterministic member-id concatenation. Only used for
        # identity inside the pattern; the *component* id at match time is
        # the relational fingerprint (see ``resolve_fingerprint``).
        self._id = "[" + "+".join(sorted(m.id for m in members)) + "]"

        # ``member_triples`` is populated by ``sp._register_modeled_node``.
        self.member_triples: List[Tuple["Node", "Predicate", "Node"]] = []

        sp._register_modeled_node(self)

    def set_signature_pattern(self, signature_pattern) -> None:
        # Pre-validated in __init__. Kept for API parity with ``Node``.
        self._signature_pattern = signature_pattern

    def validate_cls(self) -> None:
        # ModeledNode is never added via ``_add_node`` (members already are).
        # Its ``cls`` envelope is derived from the members, which validate
        # themselves; no-op here.
        pass

    def make_id(self) -> str:
        return self._id

    def __repr__(self) -> str:
        member_ids = ", ".join(m.id for m in self.members)
        return f"ModeledNode([{member_ids}])"


def resolve_fingerprint(modeled_node: "Node", sm_bindings: Dict["Node", Any]) -> str:
    """Compute a stable hex-digest fingerprint for a match of ``modeled_node``
    against the given ``sm_bindings`` (the MILP group / subgraph isomorphism).

    For a plain ``Node`` or a single-member ``ModeledNode``, the fingerprint
    collapses to a hash of the single matched IRI.

    For a multi-member ``ModeledNode`` group, the fingerprint hashes:

    1. The sorted matched IRIs of all members.
    2. The sorted matched ``(subj_iri, predicate_iri, obj_iri)`` triples for
       every ``member_triples`` edge (the "relational spine" of the group).

    This yields a context-addressable identity: two groups with the same
    members but different relational shape map to different fingerprints.
    """
    h = hashlib.blake2b(digest_size=16)

    def _iris_of(binding) -> List[str]:
        """Return sorted list of IRI strings from a scalar or tuple binding."""
        if binding is None:
            return []
        if isinstance(binding, tuple):
            return sorted(str(s.uri) for s in binding)
        return [str(binding.uri)]

    if isinstance(modeled_node, ModeledNode) and len(modeled_node.members) > 1:
        # Flatten each (possibly set-bound) member into its sorted IRIs,
        # then sort the union. Identical groups produce identical
        # fingerprints regardless of tuple vs. scalar shape on a
        # per-member basis.
        all_iris: List[str] = []
        for m in modeled_node.members:
            all_iris.extend(_iris_of(sm_bindings.get(m)))
        for iri in sorted(all_iris):
            h.update(b"M\x1f")
            h.update(iri.encode("utf-8"))

        triple_keys: List[str] = []
        for subj, pred, obj in modeled_node.member_triples:
            # A Predicate may hold multiple alternative SemanticPredicates for
            # cross-ontology matching. Sort their URIs for stable output.
            pred_iris = sorted(str(p.uri) for p in pred.preds)
            # Broadcast over tuple subjects/objects: every scalar (s, o)
            # pair in the cross-product contributes one triple key. For
            # aligned parallel tuples (the common case under
            # SetStepRule+closure), the cross-product collapses back to
            # the per-element pairs plus some extras; the sort+set
            # dedup normalizes.
            subj_iris = _iris_of(sm_bindings.get(subj))
            obj_iris = _iris_of(sm_bindings.get(obj))
            for s_iri in subj_iris:
                for o_iri in obj_iris:
                    triple_keys.append(
                        "{}\x1f{}\x1f{}".format(
                            s_iri, "|".join(pred_iris), o_iri
                        )
                    )
        for tk in sorted(set(triple_keys)):
            h.update(b"T\x1f")
            h.update(tk.encode("utf-8"))
    else:
        # Plain Node or 1-member ModeledNode: single matched IRI, or
        # flattened sorted IRIs for a 1-member set-bound ModeledNode.
        for iri in _iris_of(sm_bindings.get(modeled_node)):
            h.update(b"M\x1f")
            h.update(iri.encode("utf-8"))

    return h.hexdigest()


class Predicate:
    """
    Represents a predicate (relationship type) in a signature pattern graph.

    Mirrors the Node class structure but for predicates instead of types.
    Holds a tuple of SemanticPredicate objects for multi-predicate matching,
    enabling cross-ontology patterns where the same relationship might be
    expressed using different predicates (e.g., SAREF vs BRICK).

    Attributes
    ----------
    preds : Tuple[SemanticPredicate]
        All predicates this can match against (tuple for consistency with Node.cls)

    Examples
    --------
    Single predicate (standard usage):

    >>> pred = Predicate(core.namespace.FSO.suppliesFluidTo)
    >>> pred.preds  # Returns (FSO.suppliesFluidTo,)

    Multiple predicates (cross-ontology matching):

    >>> pred = Predicate((
    ...     core.namespace.FSO.suppliesFluidTo,  # SAREF/FSO
    ...     core.namespace.BRICK.feeds,           # BRICK
    ... ))
    >>> pred.preds  # Returns (FSO.suppliesFluidTo, BRICK.feeds)
    """

    predicate_instance_count = count()

    def __init__(
        self,
        preds: Union[Any, Tuple[Any, ...], List[Any], str],
        hash_: Optional[Any] = None,
    ) -> None:
        """
        Initialize a Predicate.

        Args:
            preds: A predicate or tuple of predicates (URIRef, str, or SemanticPredicate)
            hash_: Optional hash value for custom equality comparison
        """
        # Normalize to tuple (same as Node.cls normalization)
        if isinstance(preds, tuple) == False:
            if isinstance(preds, (list, set)):
                preds = tuple(preds)
            else:
                preds = (preds,)
        self.preds = preds
        self._signature_pattern = None
        self._id = self.make_id()

        if hash_ is not None:
            self._hash = hash(hash_)
            self.__hash__ = self.h
            self.__eq__ = self.eq

    @property
    def id(self):
        return self._id

    def h(self):
        return self._hash

    def eq(self, other):
        return self._hash == other._hash

    @property
    def signature_pattern(self):
        return self._signature_pattern

    @property
    def semantic_model(self):
        """Get the semantic model associated with this predicate"""
        return self.signature_pattern.semantic_model

    def __str__(self):
        return self.id

    def validate_preds(self):
        """
        Convert all predicates to SemanticPredicate objects.

        Similar to Node.validate_cls(), this converts URIRef and str predicates
        to SemanticPredicate objects using the signature pattern's semantic model.

        Raises:
            ValueError: If no signature pattern is set or invalid predicate type
        """
        if self._signature_pattern is None:
            raise ValueError("No signature pattern set.")

        preds = self.preds
        if isinstance(preds, tuple) == False:
            preds = (preds,)

        preds_ = []
        for p in preds:
            if isinstance(p, core.SemanticPredicate):
                preds_.append(p)
            elif isinstance(p, URIRef):
                preds_.append(
                    core.SemanticPredicate(p, self.signature_pattern.semantic_model)
                )
            elif isinstance(p, str):
                preds_.append(
                    core.SemanticPredicate(
                        URIRef(p), self.signature_pattern.semantic_model
                    )
                )
            else:
                raise ValueError(f"Invalid predicate type: {type(p)}")

        self.preds = tuple(preds_)  # Make immutable
        self._id = self.make_id()

    def make_id(self):
        """Create a unique identifier from predicate URIs."""

        def get_local_name(uri_str):
            if "#" in uri_str:
                return uri_str.split("#")[-1]
            return uri_str.split("/")[-1]

        parts = []
        for p in self.preds:
            if hasattr(p, "uri"):
                uri_str = str(p.uri)
            else:
                uri_str = str(p)
            parts.append(get_local_name(uri_str))

        return "_".join(parts)

    def set_signature_pattern(self, signature_pattern):
        """Set the signature pattern for this predicate"""
        self._signature_pattern = signature_pattern


@autoreset_print
class SignaturePattern:
    r"""
    A class for defining signature patterns that describe how component models map to semantic model instances.

    Signature patterns are the core mechanism by which the Translator identifies where and how component models
    should be instantiated within a semantic model. Each signature pattern defines a graph structure that
    specifies the semantic context required for a component model to be applicable.

    Overview
    --------
    A signature pattern consists of:
    - **Nodes**: Represent semantic model elements (components, properties, values)
    - **Edges**: Represent relationships between nodes (predicates)
    - **Rules**: Define how pattern elements map to semantic model elements
    - **Modeled Nodes**: Specify which nodes correspond to the actual component being modeled
    - **Parameters**: Define which nodes provide parameter values for the component
    - **Inputs**: Define which nodes provide input connections for the component

    Pattern Structure
    ----------------
    Signature patterns are defined using a graph-based approach where:

    - Each node represents a semantic model element (e.g., a Damper, Sensor, or Property)
    - Each edge represents a relationship between elements (e.g., "observes", "controls")
    - Rules determine how flexible the matching process is (Exact, SinglePath, MultiPath, Optional_)

    The pattern matching process finds subgraph isomorphisms between the signature pattern
    and the semantic model, allowing the Translator to identify valid contexts for component instantiation.

    Attributes
    ----------
    id : str
        Unique identifier for the signature pattern
    nodes : List[Node]
        List of nodes in the signature pattern
    required_nodes : List[Node]
        List of nodes that must be present for a match
    modeled_nodes : List[Node]
        List of nodes that correspond to the component being modeled
    parameters : Dict[str, Node]
        Dictionary mapping parameter names to nodes that provide values
    inputs : Dict[str, Tuple[Node, Dict]]
        Dictionary mapping input names to nodes and their source mappings
    ruleset : Dict[Tuple, Rule]
        Dictionary mapping (subject, predicate, object) tuples to rules

    Examples
    --------
    Basic damper control signature pattern (from actual damper system):

    >>> import twin4build.core as core
    >>> from twin4build.translator.translator import SignaturePattern, Node, Exact, Optional_
    >>>
    >>> def get_signature_pattern():
    ...     '''Create signature pattern for damper system'''
    ...     # Define nodes using real ontology classes
    ...     damper_node = Node(cls=core.namespace.S4BLDG.Damper)
    ...     controller_node = Node(cls=core.namespace.S4BLDG.Controller)
    ...     position_node = Node(cls=core.namespace.SAREF.OpeningPosition)
    ...     property_node = Node(cls=core.namespace.SAREF.Property)
    ...     flow_rate_node = Node(cls=core.namespace.S4BLDG.NominalAirFlowRate)
    ...     float_value = Node(cls=core.namespace.XSD.float)
    ...
    ...     # Create signature pattern with real parameters
    ...     sp = SignaturePattern(
    ...         semantic_model_=core.ontologies,
    ...     )
    ...
    ...     # Add required relationships using Exact rules
    ...     sp.add_triple(
    ...         Exact(subject=controller_node, object=position_node,
    ...               predicate=core.namespace.SAREF.controls)
    ...     )
    ...     sp.add_triple(
    ...         Exact(subject=position_node, object=damper_node,
    ...               predicate=core.namespace.SAREF.isPropertyOf)
    ...     )
    ...     sp.add_triple(
    ...         Exact(subject=controller_node, object=property_node,
    ...               predicate=core.namespace.SAREF.observes)
    ...     )
    ...
    ...     # Add optional parameter using Optional_ rule
    ...     sp.add_triple(
    ...         Optional_(subject=damper_node, object=flow_rate_node,
    ...                   predicate=core.namespace.SAREF.hasPropertyValue)
    ...     )
    ...
    ...     # Configure inputs and parameters
    ...     sp.add_input("damperPosition", controller_node, "inputSignal")
    ...     sp.add_parameter("nominalAirFlowRate", float_value)
    ...     sp.add_modeled_node(damper_node)
    ...
    ...     return sp

    PID controller pattern with exact relationships (from actual controller implementation):

    >>> def get_signature_pattern():
    ...     '''Create signature pattern for PID controller'''
    ...     # Define controller nodes using real ontology classes
    ...     controller_node = Node(cls=core.namespace.S4BLDG.SetpointController)
    ...     sensor_node = Node(cls=core.namespace.SAREF.Sensor)
    ...     property_node = Node(cls=core.namespace.SAREF.Property)
    ...     schedule_node = Node(cls=core.namespace.S4BLDG.Schedule)
    ...     reverse_node = Node(cls=core.namespace.XSD.boolean)
    ...
    ...     sp = SignaturePattern(
    ...         semantic_model_=core.ontologies,
    ...     )
    ...
    ...     # All relationships are exact for precise control logic
    ...     sp.add_triple(
    ...         Exact(subject=controller_node, object=property_node,
    ...               predicate=core.namespace.SAREF.observes)
    ...     )
    ...     sp.add_triple(
    ...         Exact(subject=sensor_node, object=property_node,
    ...               predicate=core.namespace.SAREF.observes)
    ...     )
    ...     sp.add_triple(
    ...         Exact(subject=controller_node, object=schedule_node,
    ...               predicate=core.namespace.SAREF.hasProfile)
    ...     )
    ...     sp.add_triple(
    ...         Exact(subject=controller_node, object=reverse_node,
    ...               predicate=core.namespace.S4BLDG.isReverse)
    ...     )
    ...
    ...     # Configure controller inputs and parameters
    ...     sp.add_input("actualValue", sensor_node, "measuredValue")
    ...     sp.add_input("setpointValue", schedule_node, "scheduleValue")
    ...     sp.add_parameter("isReverse", reverse_node)
    ...     sp.add_modeled_node(controller_node)
    ...
    ...     return sp

    Building space pattern with SinglePath for flexible connections (from building space system):

    >>> def get_signature_pattern():
    ...     '''Create signature pattern for building space system'''
    ...     # Define nodes for building space components
    ...     supply_damper = Node(cls=core.namespace.S4BLDG.Damper)  # supply damper
    ...     return_damper = Node(cls=core.namespace.S4BLDG.Damper)  # return damper
    ...     building_space = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    ...     space_heater = Node(cls=core.namespace.S4BLDG.SpaceHeater)
    ...     schedule = Node(cls=core.namespace.S4BLDG.Schedule)
    ...     outdoor_env = Node(cls=core.namespace.S4BLDG.OutdoorEnvironment)
    ...     supply_equipment = Node(cls=(
    ...         core.namespace.S4BLDG.Coil,
    ...         core.namespace.S4BLDG.AirToAirHeatRecovery,
    ...         core.namespace.S4BLDG.Fan,
    ...     ))
    ...
    ...     sp = SignaturePattern(
    ...         semantic_model_=core.ontologies,
    ...     )
    ...
    ...     # Exact relationships for system topology
    ...     sp.add_triple(
    ...         Exact(subject=supply_damper, object=building_space,
    ...               predicate=core.namespace.FSO.suppliesFluidTo)
    ...     )
    ...     sp.add_triple(
    ...         Exact(subject=return_damper, object=building_space,
    ...               predicate=core.namespace.FSO.hasFluidReturnedBy)
    ...     )
    ...     sp.add_triple(
    ...         Exact(subject=space_heater, object=building_space,
    ...               predicate=core.namespace.S4BLDG.isContainedIn)
    ...     )
    ...
    ...     # SinglePath allows flexible connection from damper to equipment
    ...     sp.add_triple(
    ...         SinglePath(subject=supply_damper, object=supply_equipment,
    ...                    predicate=core.namespace.FSO.hasFluidSuppliedBy)
    ...     )
    ...
    ...     # Configure inputs for the building space
    ...     sp.add_input("supplyAirFlowRate", supply_damper, "airFlowRate")
    ...     sp.add_input("exhaustAirFlowRate", return_damper, "airFlowRate")
    ...     sp.add_input("heatGain", space_heater, "Power")
    ...     sp.add_input("numberOfPeople", schedule, "scheduleValue")
    ...     sp.add_input("outdoorTemperature", outdoor_env, "outdoorTemperature")
    ...     sp.add_input("supplyAirTemperature", supply_equipment,
    ...                  ("outletAirTemperature", "primaryTemperatureOut"))
    ...
    ...     sp.add_modeled_node(building_space)
    ...     return sp

    BRICK ontology pattern (from damper BRICK system):

    >>> def get_signature_pattern_brick():
    ...     '''Create BRICK-specific signature pattern for damper'''
    ...     damper_node = Node(cls=core.namespace.BRICK.Damper)
    ...     position_setpoint = Node(cls=core.namespace.BRICK.Damper_Position_Setpoint)
    ...     position_sensor = Node(cls=core.namespace.BRICK.Damper_Position_Sensor)
    ...     flow_sensor = Node(cls=core.namespace.BRICK.Air_Flow_Sensor)
    ...     flow_setpoint = Node(cls=core.namespace.BRICK.Air_Flow_Setpoint)
    ...     float_value = Node(cls=core.namespace.XSD.float)
    ...
    ...     sp = SignaturePattern(
    ...         semantic_model_=core.ontologies,
    ...     )
    ...
    ...     # BRICK-specific relationships
    ...     sp.add_triple(
    ...         Exact(subject=position_setpoint, object=damper_node,
    ...               predicate=core.namespace.BRICK.isPointOf)
    ...     )
    ...     sp.add_triple(
    ...         Exact(subject=position_sensor, object=damper_node,
    ...               predicate=core.namespace.BRICK.isPointOf)
    ...     )
    ...     sp.add_triple(
    ...         Exact(subject=flow_sensor, object=damper_node,
    ...               predicate=core.namespace.BRICK.isPointOf)
    ...     )
    ...
    ...     # Optional flow rate parameter
    ...     sp.add_triple(
    ...         Optional_(subject=flow_setpoint, object=float_value,
    ...                   predicate=core.namespace.BRICK.hasValue)
    ...     )
    ...
    ...     sp.add_input("damperPosition", position_setpoint, "setpoint")
    ...     sp.add_parameter("nominalAirFlowRate", float_value)
    ...     sp.add_modeled_node(damper_node)
    ...
    ...     return sp

    Using signature patterns in component classes (from actual system implementation):

    >>> class DamperTorchSystem(core.System, nn.Module):
    ...     # Multiple signature patterns with different priorities
    ...     sp = [get_signature_pattern(), get_signature_pattern_brick()]
    ...
    ...     def __init__(self, a=1, nominalAirFlowRate=0.034, **kwargs):
    ...         super().__init__(**kwargs)
    ...         nn.Module.__init__(self)
    ...         # System implementation...

    Sensor signature patterns for space properties (from sensor system):

    >>> def get_space_temperature_signature_pattern():
    ...     '''Pattern for temperature sensors in building spaces'''
    ...     sensor_node = Node(cls=core.namespace.SAREF.Sensor)
    ...     temperature_node = Node(cls=core.namespace.SAREF.Temperature)
    ...     space_node = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    ...
    ...     sp = SignaturePattern(
    ...         semantic_model_=core.ontologies,
    ...     )
    ...
    ...     sp.add_triple(
    ...         Exact(subject=sensor_node, object=temperature_node,
    ...               predicate=core.namespace.SAREF.observes)
    ...     )
    ...     sp.add_triple(
    ...         Exact(subject=temperature_node, object=space_node,
    ...               predicate=core.namespace.SAREF.isPropertyOf)
    ...     )
    ...
    ...     sp.add_modeled_node(sensor_node)
    ...     return sp
    """

    _signatures = {}
    _signatures_reversed = {}
    _signature_instance_count = count()

    def __init__(self, id: Optional[str] = None) -> None:
        # if semantic_model_ is None:
        #     semantic_model_ = core.SemanticModel()

        # assert isinstance(
        #     semantic_model_, core.SemanticModel
        # ), 'The "semantic_model_" argument must be an instance of SemanticModel.'

        self.semantic_model = core.SemanticModel()

        if id is None:
            id = f"{str(__file__)}_{str(next(SignaturePattern._signature_instance_count))}"

        self.id = id
        SignaturePattern._signatures[id] = self
        SignaturePattern._signatures_reversed[self] = id
        self._nodes = []
        self._required_nodes = []
        self._program_registered_nodes = set()  # locked by add_triple (non-optional) / add_modeled_node
        self._user_registered_nodes = set()     # set explicitly by public add_node
        self._inputs = {}
        self._modeled_nodes = []
        # ``_modeled_node_groups`` maps each registered modeled identity to the
        # ordered list of plain SP-side ``Node`` members that constitute it.
        # A singleton (plain ``Node``) maps to ``[node]``; a ``ModeledNode``
        # group maps to its member list.
        self._modeled_node_groups: Dict[Node, List[Node]] = {}
        # ``_modeled_node_triples`` stores the "relational spine" of each
        # group — the subset of SP triples whose subject and object are both
        # members. Empty for singletons. Used at match time to compute the
        # relational fingerprint.
        self._modeled_node_triples: Dict[
            Node, List[Tuple[Node, "Predicate", Node]]
        ] = {}
        self._ruleset = {}
        self._rules = []
        self._parameters = {}
        # Nodes whose binding at match time is a *tuple* of semantic-model
        # objects rather than a single object. Registered by :class:`SetStepRule`
        # and propagated via :meth:`_propagate_set_bound` (fixed-point closure
        # over downstream rule subjects).
        self._set_bound_nodes: set = set()
        # self._pedantic = pedantic
        self._has_equivalent = []
        self._is_equivalent_of = []
        self._diff = None

        # if self._pedantic:
        #     self.semantic_model.parse_namespaces(
        #         self.semantic_model.graph, namespaces=self.semantic_model.namespaces
        #     )

    @property
    def has_equivalent(self):
        return self._has_equivalent

    @property
    def is_equivalent_of(self):
        return self._is_equivalent_of

    @property
    def parameters(self):
        return self._parameters

    @property
    def nodes(self):
        assert (
            len(self._nodes) > 0
        ), f"No nodes in the SignaturePattern {self.id}. It must contain at least 1 node."
        return self._nodes

    @property
    def required_nodes(self):
        return self._required_nodes

    @property
    def inputs(self):
        return self._inputs

    @property
    def ruleset(self):
        return self._ruleset

    @property
    def rules(self):
        return self._rules

    @property
    def modeled_nodes(self):
        assert (
            len(self._modeled_nodes) > 0
        ), f"No nodes has been marked as modeled in the SignaturePattern {self.id}. At least 1 node must be marked."
        return self._modeled_nodes

    def get_node_by_id(self, id):
        for node in self._nodes:
            if node.id == id:
                return node
        return None

    def add_rule(self, rule):
        """Register a pattern-matching rule (one triple or composite rule).

        This is the canonical registration method for :class:`Rule` objects
        — previously named :meth:`add_triple`. The two names mirror the
        taxonomy split: *triples* describe the data shape (subject,
        predicate, object), whereas *rules* describe how the matcher
        binds pattern nodes to semantic-model elements.

        Side effects
        ------------
        - Records ``rule`` in ``self._rules`` (deduplicated).
        - For each ``(subject, predicate, object)`` contributed by ``rule``
          (composite rules such as ``And``/``Or`` contribute multiple),
          installs the triple into ``self._ruleset`` and auto-registers
          the subject/object nodes via ``_add_node``.
        - If ``rule`` is a :class:`SetStepRule`, marks ``object`` as
          set-bound and propagates the status transitively via a
          fixed-point closure over downstream rule subjects.
        """
        assert isinstance(
            rule, Rule
        ), f'The "rule" argument must be a subclass of Rule - "{rule.__class__.__name__}" was provided.'

        subject = rule._subject
        object = rule._object
        predicate = rule._predicate
        # self._ruleset[(subject, predicate, object)] = rule
        if rule not in self._rules:
            self._rules.append(rule)

        for subj, obj, pred, rule_ in zip(subject, object, predicate, rule.rules):
            self._ruleset[(subj, pred, obj)] = rule

            assert (
                subj is not None and obj is not None and pred is not None
            ), "Rule must have subject, object, and predicate"

            assert isinstance(subj, Node) and isinstance(
                obj, Node
            ), '"subject" and "object" must be instances of class Node'

            assert isinstance(
                pred, Predicate
            ), '"predicate" must be an instance of class Predicate'

            self._add_node(subj, rule_)
            self._add_node(obj, rule_)

            # if any(isinstance(r, NoStepRule) for r in rule.sub_rules):
            #     self._remove_node(subj)
            #     self._remove_node(obj)

            # Set signature pattern on predicate and validate (converts to SemanticPredicate)
            pred.set_signature_pattern(self)
            pred.validate_preds()

            # if self._pedantic:
            #     attributes_a = subject.get_type_attributes()
            #     assert (
            #         predicate_obj in attributes_a
            #     ), f"The \"predicate\" argument must be one of the following: {', '.join(attributes_a)} - \"{predicate_obj}\" was provided."

            # Use Predicate object as key (like Node uses cls tuple)
            if (
                pred not in subj.predicate_object_pairs
            ):  # TODO: should maybe also be added to self.semantic_model.graph for visualization?
                subj.predicate_object_pairs[pred] = [obj]
            else:
                subj.predicate_object_pairs[pred].append(obj)

            subject_instance = core.namespace.T4B.__getitem__(subj.id)
            object_instance = core.namespace.T4B.__getitem__(obj.id)

            for cls_ in subj.cls:
                if cls_ is not core.BlankNode:
                    self.semantic_model.instance_graph.add(
                        (subject_instance, core.namespace.RDF.type, cls_.uri)
                    )
            for cls_ in obj.cls:
                if cls_ is not core.BlankNode:
                    self.semantic_model.instance_graph.add(
                        (object_instance, core.namespace.RDF.type, cls_.uri)
                    )

            # Add triples for all predicates (for visualization)
            for p in pred.preds:
                self.semantic_model.instance_graph.add(
                    (subject_instance, p.uri, object_instance)
                )

            # if isinstance(rule, OptionalRule) == False:
            #     if subj not in self._required_nodes:
            #         self._required_nodes.append(subj)

            # if isinstance(rule, OptionalRule) == False:
            #     if obj not in self._required_nodes:
            #         self._required_nodes.append(obj)

        # If any component-rule is a SetStepRule, mark its object(s) as
        # set-bound and propagate transitively. Propagation is a fixed-point
        # closure: any downstream rule whose subject is set-bound yields a
        # set-bound object (since the matcher will broadcast per element).
        for sub_rule in rule.rules:
            if isinstance(sub_rule, SetStepRule):
                self._set_bound_nodes.add(sub_rule.object)
        self._propagate_set_bound()

    def add_triple(self, rule):
        """Deprecated. Use :meth:`add_rule` instead.

        The method was renamed to reflect that a ``Rule`` object represents
        a pattern-matching *rule*, not a raw RDF triple (composite rules
        such as ``And``/``Or`` register multiple triples from a single
        rule).
        """
        warnings.warn(
            "SignaturePattern.add_triple() is deprecated. Use add_rule() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add_rule(rule)

    def _propagate_set_bound(self) -> None:
        """Fixed-point closure of set-bound status over registered rules.

        For any rule whose subject is already set-bound, mark the object
        set-bound as well — a tuple-bound subject forces the matcher to
        broadcast per element, so the object it produces is also a tuple
        per parent element, which we model as a single flattened tuple
        binding. Iterate until no new nodes are added.
        """
        changed = True
        while changed:
            changed = False
            for (subj, _pred, obj), _rule in self._ruleset.items():
                if subj in self._set_bound_nodes and obj not in self._set_bound_nodes:
                    self._set_bound_nodes.add(obj)
                    changed = True

    def add_connection(
        self,
        sender_node: Node,
        output_port: Union[str, Tuple[str]],
        input_port: str,
        output_port_index: Union[
            int, Tuple[int], Tuple[Node], Tuple[torch.Tensor]
        ] = None,
        input_port_index: Union[int, Node, torch.Tensor] = None,
    ):
        """
        Define a connection from a sender component to this component's input port.

        This method specifies how outputs from components matching the sender_node
        should be connected to this component's input. The connection can handle
        scalar-to-scalar, scalar-to-vector, vector-to-scalar, and vector-to-vector
        mappings using index parameters.

        Parameters
        ----------
        sender_node : Node
            The signature pattern Node representing the source component(s) that
            will send data to this component.

        output_port : str or tuple of str
            The name of the output port on the sender component. If the sender_node
            can match multiple classes, a tuple can be provided to specify different
            output port names for each class.

        input_port : str
            The name of the input port on this (target) component that will receive
            the connection.

        output_port_index : int, Node, torch.Tensor, or tuple thereof, optional
            Specifies which element(s) to select from the sender's output vector.

            - None: Use the full output (default for scalar outputs)
            - int: Select a single element at this index
            - Node: For Vector→Scalar connections. The Node must be from this
            (target) signature pattern and must map to semantic instances shared
            with the source. The index is determined by finding which position in
            the source's groups matches the semantic instance.
            - torch.Tensor: Select multiple elements at these indices

        input_port_index : int, Node, or torch.Tensor, optional
            Specifies which slot(s) in this component's input vector to fill.

            - None: Fill the entire input (default for scalar inputs)
            - int: Fill a single slot at this index
            - Node: For Scalar→Vector or Vector→Vector connections. The Node must
            be from this (target) signature pattern. The index/indices are
            determined by the ordering of groups matching this Node. This is the
            primary way to specify index mapping as it directly relates to the
            target component's structure.
            - torch.Tensor: Fill multiple slots at these indices

        Connection Type Summary
        -----------------------
        - Scalar→Scalar: Both indices are None
        - Scalar→Vector: Use input_port_index=Node to specify which slot receives the scalar
        - Vector→Scalar: Use output_port_index=Node to specify which element to pick
        - Vector→Vector: Use input_port_index=Node to specify the mapping from source to target

        Examples
        --------
        Simple scalar connection:

        >>> sp.add_connection(sensor_node, "temperature", "measuredTemperature")

        Vector connection where input is ordered by spaces:

        >>> sp.add_connection(spaces, "indoorTemperature", "exhaustTemperature",
        ...                   input_port_index=spaces)

        Vector to scalar - pick element corresponding to a specific space:

        >>> sp.add_connection(spaces, "indoorTemperature", "zoneTemperature",
        ...                   output_port_index=space_node)

        Notes
        -----
        When using a Node for indexing, the actual integer indices are resolved at
        translation time by the Translator._connect_components method, which examines
        the matched groups from the semantic model to determine the correct mapping.
        """
        self._add_node(sender_node, _lock=False)
        cls = list(sender_node.cls)
        assert (
            input_port not in self._inputs
        ), f'Input port "{input_port}" is already set for the SignaturePattern {self.id}.'

        if isinstance(output_port, str):
            output_port = {c: output_port for c in cls}
        elif isinstance(output_port, tuple):
            output_port_ = {}
            for c, output_port_key in zip(cls, output_port):
                output_port_[c] = output_port_key
            output_port = output_port_

        self._inputs[input_port] = (
            sender_node,
            output_port,
            output_port_index,
            input_port_index,
        )

    def add_input(self, key, node, source_keys=None):  # NOTE: Deprecated
        """
        Deprecated method. Use add_connection() instead.

        This method is deprecated and will be removed in a future version.
        Please use add_connection() method instead.
        """
        warnings.warn(
            "add_input() is deprecated and will be removed in a future version. "
            "Use add_connection() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Preprocess source_keys same as original method
        cls = list(node.cls)
        if source_keys is None:
            source_keys = {c: key for c in cls}
        elif isinstance(source_keys, str):
            source_keys = {c: source_keys for c in cls}
        elif isinstance(source_keys, tuple):
            source_keys_ = {}
            for c, source_key in zip(cls, source_keys):
                source_keys_[c] = source_key
            source_keys = source_keys_

        # Map old parameters to new method
        sender_node = node
        output_port = source_keys
        input_port = key

        # Call the new method with mapped parameters
        self.add_connection(
            sender_node=sender_node, output_port=output_port, input_port=input_port
        )

    def _add_node(self, node, rule=None, optional=False, _lock=True, _from_user=False):
        """Register a node and update its required/optional status according to caller priority.

        Priority (highest wins, lower cannot override higher):
          1. Structural  – ``add_triple`` (non-optional) / ``add_modeled_node``  → ``_lock=True``
          2. User        – public ``add_node``                                   → ``_from_user=True``
          3. Soft        – ``add_connection`` / ``add_parameter``                → both False

        A structural call locks the node in ``_program_registered_nodes``.
        A user call records it in ``_user_registered_nodes``.
        Soft calls are silently ignored when a higher-priority caller has already decided.
        """
        if not isinstance(rule, NoStepRule):
            if node not in self._nodes:
                self._nodes.append(node)
                node.set_signature_pattern(self)
                node.validate_cls()

            is_optional = isinstance(rule, OptionalRule) or optional

            # Determine whether this call is allowed to update the status.
            if _lock:
                can_update = True
                # Structural rules (``add_rule``/``add_modeled_node``)
                # lock the node into ``_program_registered_nodes``
                # regardless of optionality, so a subsequent soft call
                # (``add_connection``/``add_parameter``) cannot
                # silently upgrade an OptionalRule-registered node to
                # required.  The optional/required status is recorded
                # via ``_required_nodes`` below; the program-locked
                # set is only about *who decides* the status, not the
                # status itself.
                self._program_registered_nodes.add(node)
            elif _from_user:
                can_update = node not in self._program_registered_nodes
                if can_update:
                    self._user_registered_nodes.add(node)
            else:  # soft internal call
                can_update = (
                    node not in self._program_registered_nodes
                    and node not in self._user_registered_nodes
                )

            if can_update:
                if is_optional:
                    # Required wins over optional: a node that any
                    # non-optional rule marked required stays required
                    # even when an ``OptionalRule`` (or ``optional=True``
                    # ``add_node`` / ``add_parameter`` soft call) later
                    # registers it.  The previous "last writer wins"
                    # behaviour silently demoted nodes shared between
                    # required and optional rules -- e.g. an AHU node
                    # that is the subject of both ``SetAnyPathRule(ahu,
                    # vavs, feeds)`` (required) and ``OptionalRule(ahu,
                    # sat_setpoint, hasPoint)`` (optional) ended up
                    # absent from ``required_nodes``, so partial matches
                    # with ``AHU = None`` were accepted as complete by
                    # ``_try_merge_with_incomplete`` and produced
                    # spurious per-VAV "mini-AHU" components in Stage 2.
                    pass
                else:
                    if node not in self._required_nodes:
                        self._required_nodes.append(node)
        else:
            if node not in self._nodes:
                node.set_signature_pattern(self)
                node.validate_cls()
            if _lock:
                self._program_registered_nodes.add(node)

    def add_node(self, node, rule=None, optional=False):
        """Public user-facing method. Sets required/optional status unless the node is
        already locked by a structural rule (add_triple non-optional / add_modeled_node)."""
        self._add_node(node, rule=rule, optional=optional, _lock=False, _from_user=True)

    def _remove_node(self, node):
        if node in self._nodes:
            self._nodes.remove(node)

    def add_parameter(self, key, node):
        self._add_node(node, optional=True, _lock=False)
        assert node not in self._set_bound_nodes, (
            f"Cannot register set-bound node {node.id!r} as parameter {key!r}: "
            "parameter values are scalar, but this node resolves to a tuple of "
            "semantic-model instances (reachable from a SetStepRule chain). "
            "Refactor the pattern so the parameter node is scalar-bound, or "
            "promote the parameter to a per-member input via add_connection."
        )
        cls = list(node.cls)
        # allowed_classes = (float, int)
        allowed_classes = (
            core.namespace.XSD.float,
            core.namespace.XSD.int,
            core.namespace.XSD.boolean,
            core.namespace.XSD.string,
        )
        assert any(
            n.istype(allowed_classes) for n in cls
        ), f"The class of the \"node\" argument must be a subclass of {', '.join([c.get_short_name() for c in allowed_classes])} - {', '.join([c.get_short_name() for c in cls])} was provided."
        # assert any(issubclass(n, allowed_classes) for n in cls), f"The class of the \"node\" argument must be a subclass of {', '.join([c.__name__ for c in allowed_classes])} - {', '.join([c.__name__ for c in cls])} was provided."
        self._parameters[key] = node

    def add_modeled_node(self, node):
        """Register ``node`` as a modeled identity.

        Accepts a plain ``Node`` (singleton identity, historical behaviour) or
        a ``ModeledNode`` (composite identity — though ``ModeledNode`` already
        self-registers in its ``__init__``, so calling this explicitly is
        idempotent).
        """
        if isinstance(node, ModeledNode):
            # ModeledNode self-registers via _register_modeled_node in its
            # __init__. This explicit call is idempotent.
            if node not in self._modeled_nodes:
                self._register_modeled_node(node)
            return

        # Singleton identity: lock the plain Node as required.
        self._add_node(node)
        self._register_modeled_node(node)

    def _register_modeled_node(self, node):
        """Internal: append ``node`` to ``_modeled_nodes`` and capture group
        membership + ``member_triples``.

        For a plain ``Node``, the group is ``[node]`` and triples are empty.
        For a ``ModeledNode``, the group is its member list and triples are
        the subset of ``self._ruleset`` whose subject and object are both
        members (excluding ``NoStepRule`` entries).
        """
        if node in self._modeled_nodes:
            return
        self._modeled_nodes.append(node)

        if isinstance(node, ModeledNode):
            member_set = set(node.members)
            self._modeled_node_groups[node] = list(node.members)
            triples: List[Tuple[Node, Predicate, Node]] = []
            for (subj, pred, obj), rule in self._ruleset.items():
                if (
                    subj in member_set
                    and obj in member_set
                    and not isinstance(rule, NoStepRule)
                ):
                    triples.append((subj, pred, obj))
            # Deterministic order (by subj.id, pred.id, obj.id) so that
            # downstream fingerprinting is stable regardless of declaration
            # order.
            triples.sort(key=lambda t: (t[0].id, t[1].id, t[2].id))
            node.member_triples = triples
            self._modeled_node_triples[node] = triples
        else:
            self._modeled_node_groups[node] = [node]
            self._modeled_node_triples[node] = []

    def iter_modeled_sp_nodes(self):
        """Yield plain SP-side ``Node``\\ s for all modeled identities,
        expanding any ``ModeledNode`` group into its members.

        Used by match/merge logic that needs to check whether a group has
        bound its "unique" endpoints (vs a shared/reusable sub-pattern).
        """
        for mn in self._modeled_nodes:
            if isinstance(mn, ModeledNode):
                for m in mn.members:
                    yield m
            else:
                yield mn

    def remove_modeled_node(self, node):
        if node in self._modeled_nodes:
            self._modeled_nodes.remove(node)
        self._modeled_node_groups.pop(node, None)
        self._modeled_node_triples.pop(node, None)

    def reset_ruleset(self):
        for rule in self.rules:
            rule.reset()

    def add_namespace(self, namespace):
        self.semantic_model.graph.parse(namespace)

    def add_equivalent(self, sp, diff):
        """
        Add equivalent signature pattern and the corresponding diff between these.
        """
        self._has_equivalent.append(sp)
        sp._diff = diff
        sp._is_equivalent_of.append(self)
        assert (
            len(sp.has_equivalent) == 0
        ), "A signature cannot both be equivalent to and equivalent of another signature."
        assert (
            self.diff is None
        ), "A signature cannot be equivalent to another signature if it has a diff."

    def apply_changes(self, semantic_model, _eq_group):
        """
        Maps the SP of the candidate back to the original SP and applies the changes to the semantic model.
        """
        assert self._diff is not None, "A signature cannot map back if it has no diff."

        original_sp = self._is_equivalent_of[0]
        sp_sm_map = {sp_subject: None for sp_subject in original_sp.nodes}

        for sp_node, sm_node in sp_sm_map.items():
            if sp_node in _eq_group:
                sm_node = _eq_group[sp_node]
                sp_sm_map[sp_node] = sm_node

        for removal in self._diff.removals:
            subject, predicate, object = removal
            subject_instance = _eq_group[subject]
            object_instance = _eq_group[object]
            semantic_model.graph.remove(
                (subject_instance.uri, predicate, object_instance.uri)
            )

        # Locate uri already present in semantic model and reuse these. If not, create new ones.
        # The ones created should match the ones not mapped in sp_sm_map.
        for addition in self._diff.additions:
            subject, predicate, object = addition
            subject_type_uri = subject.cls[0].uri  # Node.SemanticType.uri
            object_type_uri = object.cls[0].uri  # Node.SemanticType.uri

            if subject in _eq_group:
                subject_instance_uri = _eq_group[subject].uri
            else:
                # Make new instance
                name = str(hash(subject))
                subject_instance_uri = semantic_model.T4B.__getitem__(
                    name
                )  # Define namespace
                semantic_model.graph.add(
                    (
                        subject_instance_uri,
                        core.namespace.RDF.type,
                        subject_type_uri,
                    )
                )

            if object in _eq_group:
                object_instance_uri = _eq_group[object].uri
            else:
                # Make new instance
                name = str(hash(object))
                object_instance_uri = semantic_model.T4B.__getitem__(
                    name
                )  # Define namespace
                semantic_model.graph.add(
                    (
                        object_instance_uri,
                        core.namespace.RDF.type,
                        object_type_uri,
                    )
                )

            semantic_model.graph.add(
                (subject_instance_uri, predicate, object_instance_uri)
            )
            sp_sm_map[subject] = semantic_model.get_instance(subject_instance_uri)
            sp_sm_map[object] = semantic_model.get_instance(object_instance_uri)

        assert all(
            sp_sm_map[sp_node] is not None for sp_node in original_sp.nodes
        ), "All nodes in the original SP must be mapped to a semantic model instance. Maybe the diff is not complete."

        return sp_sm_map


class Diff:
    def __init__(self):
        self.additions = []
        self.removals = []

    def add(self, subject, predicate, object):
        self.additions.append((subject, predicate, object))

    def remove(self, subject, predicate, object):
        self.removals.append((subject, predicate, object))


class Rule:
    r"""
    Base class for pattern matching rules that define how signature pattern elements map to semantic model elements.

    Rules are the fundamental building blocks of signature patterns, defining the constraints and flexibility
    of the pattern matching process. Each rule specifies how a relationship between two nodes in the signature
    pattern should be matched against the semantic model.

    Overview
    --------
    Rules define the mapping between signature pattern elements and semantic model elements through:
    - **Subject**: A Node representing the source
    - **Object**: A Node representing the target
    - **Predicate**: A Predicate object (can hold multiple predicates for cross-ontology matching)
    - **Priority**: The precedence level for rule application (higher values take precedence)

    Rule Types
    ----------
    The Translator supports several types of rules, each with different matching behavior:

    - **Exact**: Requires exact matches between pattern and semantic model elements
    - **SinglePath**: Allows traversal along a single path in the semantic model
    - **MultiPath**: Allows traversal along multiple paths in the semantic model
    - **Optional**: Makes pattern elements optional (may or may not be present)

    Cross-Ontology Matching
    -----------------------
    Rules support multiple predicates via the Predicate class (similar to how Node supports
    multiple types). When a Predicate has multiple values, the rule matches if ANY predicate matches:

    >>> # Cross-ontology matching using Predicate with multiple values
    >>> pred = Predicate((
    ...     core.namespace.FSO.suppliesFluidTo,  # SAREF/FSO
    ...     core.namespace.BRICK.feeds,           # BRICK
    ... ))
    >>> rule = Exact(subject=damper_node, object=space_node, predicate=pred)

    Rule Composition
    ---------------
    Rules can be combined using logical operators:
    - **And**: Both rules must be satisfied
    - **Or**: Either rule can be satisfied

    Examples
    --------
    >>> # Create nodes for a fan pattern
    >>> fan_node = Node(Fan)
    >>> meter_node = Node(Meter)
    >>> flow_node = Node(Flow)
    >>>
    >>> # Simple predicate (auto-wrapped in Predicate)
    >>> exact_rule = Exact(subject=meter_node, object=fan_node, predicate="observes")
    >>>
    >>> # Multi-predicate for cross-ontology matching
    >>> pred = Predicate((pred1, pred2))
    >>> exact_rule = Exact(subject=meter_node, object=fan_node, predicate=pred)
    >>>
    >>> # Combine rules
    >>> combined_rule = exact_rule & path_rule | optional_rule

    Attributes
    ----------
    subject : Node
        The source node in the signature pattern
    object : Node
        The target node in the signature pattern
    predicate : Predicate
        The predicate(s) for this rule (holds tuple of SemanticPredicate like Node holds tuple of SemanticType)
        The precedence level for rule application
    """

    def __init__(
        self,
        subject: Union[Node, Tuple[Node, ...]],
        object: Union[Node, Tuple[Node, ...]],
        predicate: Union[Predicate, Tuple[Predicate, ...]],
    ) -> None:
        """
        Initialize a Rule.

        Args:
            subject: The source Node
            object: The target Node
            predicate: A Predicate object, or value(s) to wrap in a Predicate
        """
        self.rules = (self,)
        if isinstance(subject, tuple):
            self._subject = subject
        else:
            self._subject = (subject,)
        if isinstance(object, tuple):
            self._object = object
        else:
            self._object = (object,)
        if isinstance(predicate, tuple):
            self._predicate = predicate
        else:
            self._predicate = (predicate,)

        new_predicate = []
        for pred in self._predicate:
            # Normalize predicate to Predicate class (similar to how we normalize cls to tuple in Node)
            if isinstance(pred, Predicate):
                new_predicate.append(pred)
            else:
                # Auto-wrap in Predicate (handles single value or tuple)
                new_predicate.append(Predicate(pred))
        self._predicate = tuple(new_predicate)

        assert (
            len(self._subject) == len(self._object) == len(self._predicate)
        ), "The number of subjects, objects, and predicates must be the same."

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    @property
    def subject(self):
        return self._subject

    @property
    def object(self):
        return self._object

    @property
    def predicate(self):
        return self._predicate


class And(Rule):
    """Logical AND of two rules - both must match."""

    def __init__(self, rule_a: Rule, rule_b: Rule) -> None:
        sp_subject = rule_a._subject + rule_b._subject
        sp_object = rule_a._object + rule_b._object
        sp_predicate = rule_a._predicate + rule_b._predicate
        super().__init__(subject=sp_subject, object=sp_object, predicate=sp_predicate)
        self.rule_a = rule_a
        self.rule_b = rule_b
        self.rules = rule_a.rules + rule_b.rules

    def apply(
        self,
        sm_subject: core.SemanticObject,
        sm_objects: List[core.SemanticObject],
        ruleset: Dict[Tuple[Node, Optional[Predicate], Node], Rule],
        candidate_maps: Optional[List[Optional[Any]]] = None,
        master_rule: Optional[Rule] = None,
    ) -> Tuple[
        List[Tuple[Optional[List], core.SemanticObject, Node, type]],
        bool,
        Dict[Tuple[Node, Optional[Predicate], Node], Rule],
    ]:
        LOGGER.debug("Applying %s", self.__class__.__name__)
        LOGGER.add_level()
        if master_rule is None:
            master_rule = self
        c = [ob.cls for ob in self.object]
        pairs_a, rule_applies_a, rule_applies_a_vec, ruleset_a = self.rule_a.apply(
            sm_subject,
            sm_objects,
            ruleset,
            candidate_maps=candidate_maps,
            master_rule=master_rule,
        )
        pairs_b, rule_applies_b, rule_applies_b_vec, ruleset_b = self.rule_b.apply(
            sm_subject,
            sm_objects,
            ruleset,
            candidate_maps=candidate_maps,
            master_rule=master_rule,
        )
        if rule_applies_a and rule_applies_b:
            mask = rule_applies_a_vec & rule_applies_b_vec
            indices = np.where(mask)[0]
            pairs = []
            pairs.extend([pair for pair in pairs_a if pair[4] in indices])
            pairs.extend([pair for pair in pairs_b if pair[4] in indices])
            ruleset_a.update(ruleset_b)
            for pair in pairs:
                LOGGER.debug(
                    "Matched: %s (%s) is (%s)",
                    pair[1].get_short_name(),
                    (mst.get_short_name() if (mst := pair[1].get_most_specific_type()) is not None else "None"),
                    c,
                )
            LOGGER.debug("Rule applies: %s", True)
            LOGGER.remove_level()
            return pairs, True, mask, ruleset_a

        LOGGER.debug("Rule applies: %s", False)
        LOGGER.remove_level()
        return [], False, np.array([False] * len(sm_objects)), ruleset

    def reset(self):
        self.rule_a.reset()
        self.rule_b.reset()


class Or(Rule):
    """Logical OR of two rules - either can match."""

    def __init__(self, rule_a: Rule, rule_b: Rule) -> None:
        # assert (
        #     rule_a.subject == rule_b.subject
        # ), "The subject of the two rules must be the same."
        # assert (
        #     rule_a.object == rule_b.object
        # ), "The object of the two rules must be the same."
        # assert (
        #     rule_a.predicate == rule_b.predicate
        # ), "The predicate of the two rules must be the same."
        sp_subject = rule_a._subject + rule_b._subject
        sp_object = rule_a._object + rule_b._object
        sp_predicate = rule_a._predicate + rule_b._predicate
        super().__init__(subject=sp_subject, object=sp_object, predicate=sp_predicate)
        self.rule_a = rule_a
        self.rule_b = rule_b
        self.rules = rule_a.rules + rule_b.rules

    def apply(
        self,
        sm_subject: core.SemanticObject,
        sm_objects: List[core.SemanticObject],
        ruleset: Dict[Tuple[Node, Optional[Predicate], Node], Rule],
        candidate_maps: Optional[List[Optional[Any]]] = None,
        master_rule: Optional[Rule] = None,
    ) -> Tuple[
        List[Tuple[Optional[List], core.SemanticObject, Node, type]],
        bool,
        Dict[Tuple[Node, Optional[Predicate], Node], Rule],
    ]:
        LOGGER.debug("Applying %s", self.__class__.__name__)
        LOGGER.add_level()
        if master_rule is None:
            master_rule = self
        pairs_a, rule_applies_a, rule_applies_a_vec, ruleset_a = self.rule_a.apply(
            sm_subject,
            sm_objects,
            ruleset,
            candidate_maps=candidate_maps,
            master_rule=master_rule,
        )
        pairs_b, rule_applies_b, rule_applies_b_vec, ruleset_b = self.rule_b.apply(
            sm_subject,
            sm_objects,
            ruleset,
            candidate_maps=candidate_maps,
            master_rule=master_rule,
        )
        c = [ob.cls for ob in self.object]
        if rule_applies_a and rule_applies_b:
            mask = rule_applies_a_vec | rule_applies_b_vec
            indices = np.where(mask)[0]
            pairs = []
            pairs.extend([pair for pair in pairs_a if pair[4] in indices])
            pairs.extend([pair for pair in pairs_b if pair[4] in indices])
            ruleset_a.update(ruleset_b)

            for pair in pairs:
                LOGGER.debug(
                    "Matched: %s (%s) is (%s)",
                    pair[1].get_short_name(),
                    (mst.get_short_name() if (mst := pair[1].get_most_specific_type()) is not None else "None"),
                    c,
                )
            LOGGER.debug("Rule applies: %s", True)
            LOGGER.remove_level()
            return pairs, True, rule_applies_a_vec | rule_applies_b_vec, ruleset_a

        elif rule_applies_a:
            for pair in pairs_a:
                LOGGER.debug(
                    "Matched: %s (%s) is (%s)",
                    pair[1].get_short_name(),
                    (mst.get_short_name() if (mst := pair[1].get_most_specific_type()) is not None else "None"),
                    c,
                )
            LOGGER.debug("Rule applies: %s", True)
            LOGGER.remove_level()
            return pairs_a, True, rule_applies_a_vec, ruleset_a

        elif rule_applies_b:
            for pair in pairs_b:
                LOGGER.debug(
                    "Matched: %s (%s) is (%s)",
                    pair[1].get_short_name(),
                    (mst.get_short_name() if (mst := pair[1].get_most_specific_type()) is not None else "None"),
                    c,
                )
            LOGGER.debug("Rule applies: %s", True)
            LOGGER.remove_level()
            return pairs_b, True, rule_applies_b_vec, ruleset_b

        LOGGER.debug("Rule applies: %s", False)
        LOGGER.remove_level()
        return [], False, np.array([False] * len(sm_objects)), ruleset

    def reset(self):
        self.rule_a.reset()
        self.rule_b.reset()


class NoStepRule(Rule):
    r"""
    One-hop negation rule — asserts the absence of a single edge.

    Name
    ----
    ``No`` prefix = negation of the default semantics; ``Step`` = one
    hop. Reads as "rule requiring the absence of this step (triple)".

    Asserts
    -------
    No match may contain the triple ``SM(s) --predicate--> SM(o)`` in
    the semantic-model graph. Whereas :class:`StepRule` *requires* the
    triple, ``NoStepRule`` *forbids* it.

    Matcher behavior
    ----------------
    Purely a veto. If an SM triple satisfying the pattern exists for a
    candidate subject/object pair, the branch is **pruned**. The rule
    never introduces a new SP→SM mapping — it only rejects.

    Binding produced
    ----------------
    None (``sp_object`` is never bound by a ``NoStepRule``).

    Composition
    -----------
    Typically pairs with another step/path rule to carve out the
    positive/negative shape of the intended subgraph (e.g. "space has a
    heating coil *but not* a cooling coil"). Composing ``NoStepRule``
    with a set-bound subject is supported: it fails if *any* element of
    the set has the forbidden edge.

    When to use vs. siblings
    ------------------------
    Use ``NoStepRule`` to disambiguate patterns whose scalar shape is
    otherwise a subset of a richer pattern (a typical case is
    differentiating a heating-only zone from a dual-duct zone).

    Example
    -------

    >>> sp.add_rule(NoStepRule(
    ...     subject=space,
    ...     object=cooling_coil,
    ...     predicate=core.namespace.BRICK.hasPart,
    ... ))
    """

    def __init__(self, **kwargs):
        Rule.__init__(self, **kwargs)

    @property
    def subject(self):
        assert len(self._subject) == 1, "The number of subjects must be 1."
        return self._subject[0]

    @property
    def object(self):
        assert len(self._object) == 1, "The number of objects must be 1."
        return self._object[0]

    @property
    def predicate(self):
        assert len(self._predicate) == 1, "The number of predicates must be 1."
        return self._predicate[0]

    def apply(
        self,
        sm_subject: core.SemanticObject,
        sm_objects: List[core.SemanticObject],
        ruleset: Dict[Tuple[Node, Optional[Predicate], Node], Rule],
        candidate_maps: Optional[List[Optional[Any]]] = None,
        master_rule: Optional[Rule] = None,
    ) -> Tuple[
        List[Tuple[Optional[List], core.SemanticObject, Node, type]],
        bool,
        Dict[Tuple[Node, Optional[Predicate], Node], Rule],
    ]:
        """ """
        LOGGER.debug("Applying %s", self.__class__.__name__)
        LOGGER.add_level()
        if master_rule is None:
            master_rule = self
        pairs = []
        rule_applies_vec = np.array([False] * len(sm_objects))

        if len(candidate_maps) == 0:
            candidate_maps = [None]

        for current_map in candidate_maps:
            excluded_sm_subjects = []
            excluded_sm_objects = []

            if current_map is not None:
                # Find SM objects to exclude (already matched to different SP objects)
                for (sp_subject, sp_predicate, sp_object), rule in ruleset.items():
                    if rule.predicate is not None and self.predicate is not None:
                        if (
                            sp_object in current_map
                            and rule.subject == self.subject
                            and rule.predicate == self.predicate
                            and rule.object != self.object
                        ):
                            excluded_sm_objects.append(current_map[sp_object])

                # Find SM subjects to exclude (already matched to different SP subjects)
                for (sp_subject, sp_predicate, sp_object), rule in ruleset.items():
                    if rule.predicate is not None and self.predicate is not None:
                        if (
                            sp_subject in current_map
                            and rule.object == self.object
                            and rule.predicate == self.predicate
                            and rule.subject != self.subject
                        ):
                            excluded_sm_subjects.append(current_map[sp_subject])
                maps_for_match = [current_map]
            else:
                maps_for_match = []

            # Check each candidate SM object
            for i, sm_object in enumerate(sm_objects):
                if (
                    sm_object.isinstance(self.object.cls) == False
                    and sm_subject not in excluded_sm_subjects
                    and sm_object not in excluded_sm_objects
                ):
                    # pairs.append((maps_for_match, sm_object, self.object, NoStepRule, i))
                    rule_applies_vec[i] = True
        rule_applies = np.any(rule_applies_vec)
        for pair in pairs:
            LOGGER.debug(
                "Matched: %s (%s) is %s",
                pair[1].get_short_name(),
                (mst.get_short_name() if (mst := pair[1].get_most_specific_type()) is not None else "None"),
                self.object.cls,
            )
        LOGGER.debug("Rule applies: %s", rule_applies)
        LOGGER.remove_level()

        return pairs, rule_applies, rule_applies_vec, ruleset

    def reset(self):
        pass


class StepRule(Rule):
    r"""
    One-hop required rule with scalar branching (the taxonomic default).

    Name
    ----
    ``Step`` — a single edge in the semantic-model graph (one ``(s, p, o)``
    triple). No prefix modifier means required presence with scalar binding,
    the default shape of the rule taxonomy. A :class:`PathRule` is a
    sequence of ``StepRule``-style edges; one step is the atom of a path.

    Asserts
    -------
    Every match contains the triple ``SM(s) --predicate--> SM(o)`` in the
    semantic-model graph. If no SM object satisfies the predicate, the
    branch is pruned (strict presence).

    Matcher behavior
    ----------------
    For each SM object satisfying ``self.object.cls``, the matcher emits a
    **separate branch** in ``candidate_maps``; each branch becomes its
    own complete match group with ``group[sp_object] = one_sm_object``.
    Sibling ``StepRule``s on the same subject produce a cross-product of
    branches — one group per Cartesian tuple.

    Binding produced
    ----------------
    ``SemanticObject`` (scalar) on ``sp_object``.

    Composition
    -----------
    Downstream rules on either endpoint extend each branch individually.
    If ``sp_subject`` is set-bound (reached from a :class:`SetStepRule`
    chain), ``StepRule`` is auto-broadcast per element by the matcher and
    the produced ``sp_object`` becomes set-bound via closure.

    When to use vs. siblings
    ------------------------
    - Use :class:`StepRule` when each matching SM object should beget its
      own simulation component (e.g. one ``SensorSystem`` per sensor).
    - Use :class:`SetStepRule` when many SM objects should be consumed
      jointly by one component.
    - Use :class:`NoStepRule` to express forbidden edges.
    - Use :class:`PathRule`/:class:`AnyPathRule` when the edge is a
      multi-hop traversal, not a single triple.

    Example
    -------

    >>> sp.add_rule(StepRule(
    ...     subject=vav,
    ...     object=reheat_valve,
    ...     predicate=core.namespace.BRICK.hasPart,
    ... ))
    """

    def __init__(self, **kwargs):
        Rule.__init__(self, **kwargs)

    @property
    def subject(self):
        assert len(self._subject) == 1, "The number of subjects must be 1."
        return self._subject[0]

    @property
    def object(self):
        assert len(self._object) == 1, "The number of objects must be 1."
        return self._object[0]

    @property
    def predicate(self):
        assert len(self._predicate) == 1, "The number of predicates must be 1."
        return self._predicate[0]

    def apply(
        self,
        sm_subject: core.SemanticObject,
        sm_objects: List[core.SemanticObject],
        ruleset: Dict[Tuple[Node, Optional[Predicate], Node], Rule],
        candidate_maps: Optional[List[Optional[Any]]] = None,
        master_rule: Optional[Rule] = None,
    ) -> Tuple[
        List[Tuple[Optional[List], core.SemanticObject, Node, type]],
        bool,
        Dict[Tuple[Node, Optional[Predicate], Node], Rule],
    ]:
        """
        Apply Exact rule to find matching SM objects.

        Args:
            sm_subject: The SM subject node (for exclusion checks)
            sm_objects: List of candidate SM object nodes to match against
            ruleset: Current ruleset dictionary
            candidate_maps: List of current SP→SM mappings
            master_rule: The top-level rule (for composite rules)

        Returns:
            Tuple of (pairs, rule_applies, ruleset) where pairs contains
            (maps, matched_sm_object, matched_sp_object, rule_type) tuples
        """
        LOGGER.debug("Applying %s", self.__class__.__name__)
        LOGGER.add_level()
        if master_rule is None:
            master_rule = self
        pairs = []
        rule_applies_vec = np.array([False] * len(sm_objects))

        if len(candidate_maps) == 0:
            candidate_maps = [None]

        for current_map in candidate_maps:
            excluded_sm_subjects = []
            excluded_sm_objects = []

            if current_map is not None:
                # Find SM objects to exclude (already matched to different SP objects)
                for (sp_subject, sp_predicate, sp_object), rule in ruleset.items():
                    if rule.predicate is not None and self.predicate is not None:
                        if (
                            sp_object in current_map
                            and rule.subject == self.subject
                            and rule.predicate == self.predicate
                            and rule.object != self.object
                        ):
                            excluded_sm_objects.append(current_map[sp_object])

                # Find SM subjects to exclude (already matched to different SP subjects)
                for (sp_subject, sp_predicate, sp_object), rule in ruleset.items():
                    if rule.predicate is not None and self.predicate is not None:
                        if (
                            sp_subject in current_map
                            and rule.object == self.object
                            and rule.predicate == self.predicate
                            and rule.subject != self.subject
                        ):
                            excluded_sm_subjects.append(current_map[sp_subject])
                maps_for_match = [current_map]
            else:
                maps_for_match = []

            # Check each candidate SM object
            for i, sm_object in enumerate(sm_objects):
                if (
                    sm_object.isinstance(self.object.cls)
                    and sm_subject not in excluded_sm_subjects
                    and sm_object not in excluded_sm_objects
                ):
                    pairs.append((maps_for_match, sm_object, self.object, StepRule, i))
                    rule_applies_vec[i] = True

        rule_applies = np.any(rule_applies_vec)
        for pair in pairs:
            LOGGER.debug(
                "Matched: %s (%s) is %s",
                pair[1].get_short_name(),
                (mst.get_short_name() if (mst := pair[1].get_most_specific_type()) is not None else "None"),
                self.object.cls,
            )
        LOGGER.debug("Rule applies: %s", rule_applies)
        LOGGER.remove_level()
        return pairs, rule_applies, rule_applies_vec, ruleset

    def reset(self):
        pass


class SetStepRule(StepRule):
    r"""
    Set-binding one-hop rule — binds the object to the tuple of all
    matching SM nodes reached by a single edge.

    Name
    ----
    ``Set`` prefix = set-valued binding (the default is scalar); ``Step``
    = one hop. Reads as "rule that binds the object to the set of SM
    nodes reached by this step (predicate)".

    Asserts
    -------
    Every match contains the triple ``SM(s) --predicate--> SM(o_i)`` for
    *every* ``SM(o_i)`` appearing in the bound tuple; the tuple
    enumerates all such SM objects reachable from ``SM(s)``. Zero matches
    prune the branch (same strictness as :class:`StepRule`).

    Matcher behavior
    ----------------
    Does **not** branch per SM match. All SM objects satisfying
    ``self.object.cls`` (after ``StepRule``-style sibling exclusion
    filtering) are collected into a single ``Tuple[SemanticObject, ...]``
    sorted by IRI and assigned to ``group[sp_object]`` in one branch.
    Where :class:`StepRule` emits ``N`` branches for ``N`` matches,
    ``SetStepRule`` emits exactly one.

    Binding produced
    ----------------
    ``Tuple[SemanticObject, ...]`` (set-valued) on ``sp_object``.

    Auto-broadcast
    --------------
    Any downstream rule whose subject is a set-bound SP node is
    automatically applied per element, and its object-side bindings are
    gathered into a parallel tuple (same length and order as the subject
    tuple). This closure is computed at
    :meth:`SignaturePattern.add_rule` time and stored in
    ``_set_bound_nodes``. Users do **not** annotate downstream hops —
    writing ``StepRule(subject=set_bound_node, object=..., predicate=...)``
    is idiomatic.

    Composition
    -----------
    - Can sit alongside :class:`StepRule`\ s on the same subject; the
      cross-product reduces to ``1 × (scalar sibling arities)``.
    - Downstream :class:`StepRule`/:class:`NoStepRule` on a set-bound
      subject are auto-broadcast.
    - Downstream ``SetStepRule`` on a set-bound subject is **not**
      supported in the initial implementation (would require nested-set
      semantics).
    - Composition with :class:`PathRule` / :class:`AnyPathRule`
      traversals is out of scope for the initial implementation — use
      ``SetStepRule`` only on direct (one-hop) edges.

    Restrictions
    ------------
    Set-bound nodes cannot be registered as ``add_parameter`` targets
    (parameters are scalar literals); see
    :meth:`SignaturePattern.add_parameter`.

    When to use vs. siblings
    ------------------------
    :class:`StepRule` when matches should beget separate components;
    ``SetStepRule`` when many SM objects jointly describe *one*
    simulation entity (e.g. a VAV-level controller whose inputs are all
    the sensors of a zone).

    Example
    -------

    >>> sp.add_rule(SetStepRule(
    ...     subject=vav,
    ...     object=sensors,
    ...     predicate=core.namespace.BRICK.hasPoint,
    ... ))
    >>> # ``sensors`` binds to the tuple of ALL sensor points of the matched
    >>> # VAV, in one group.
    """

    def __init__(self, **kwargs):
        StepRule.__init__(self, **kwargs)

    def apply(
        self,
        sm_subject: core.SemanticObject,
        sm_objects: List[core.SemanticObject],
        ruleset: Dict[Tuple[Node, Optional[Predicate], Node], Rule],
        candidate_maps: Optional[List[Optional[Any]]] = None,
        master_rule: Optional[Rule] = None,
    ) -> Tuple[
        List[Tuple[Optional[List], core.SemanticObject, Node, type]],
        bool,
        Dict[Tuple[Node, Optional[Predicate], Node], Rule],
    ]:
        """Apply SetStep rule: collect all StepRule matches into a single
        tuple-bound pair.

        Reuses the per-candidate-map exclusion bookkeeping of
        :class:`StepRule`. For each candidate map, all SM objects that
        satisfy the predicate are bundled into one tuple binding and
        emitted as a single pair ``(maps_for_match, tuple_of_sm_objects,
        self.object, SetStepRule, i0)`` where ``i0`` is the index of the
        first matched SM object (used solely to index ``rule_applies_vec``
        downstream).
        """
        LOGGER.debug("Applying %s", self.__class__.__name__)
        LOGGER.add_level()
        if master_rule is None:
            master_rule = self
        pairs: List[Tuple[Any, Any, Node, type, int]] = []
        rule_applies_vec = np.array([False] * len(sm_objects))

        if len(candidate_maps) == 0:
            candidate_maps = [None]

        for current_map in candidate_maps:
            excluded_sm_subjects: List[Any] = []
            excluded_sm_objects: List[Any] = []

            if current_map is not None:
                for (sp_subject, sp_predicate, sp_object), rule in ruleset.items():
                    if rule.predicate is not None and self.predicate is not None:
                        if (
                            sp_object in current_map
                            and rule.subject == self.subject
                            and rule.predicate == self.predicate
                            and rule.object != self.object
                        ):
                            excluded_sm_objects.append(current_map[sp_object])

                for (sp_subject, sp_predicate, sp_object), rule in ruleset.items():
                    if rule.predicate is not None and self.predicate is not None:
                        if (
                            sp_subject in current_map
                            and rule.object == self.object
                            and rule.predicate == self.predicate
                            and rule.subject != self.subject
                        ):
                            excluded_sm_subjects.append(current_map[sp_subject])
                maps_for_match = [current_map]
            else:
                maps_for_match = []

            matched: List[Tuple[int, Any]] = []
            for i, sm_object in enumerate(sm_objects):
                if (
                    sm_object.isinstance(self.object.cls)
                    and sm_subject not in excluded_sm_subjects
                    and sm_object not in excluded_sm_objects
                ):
                    matched.append((i, sm_object))
                    rule_applies_vec[i] = True

            if matched:
                # Deduplicate and canonicalize by IRI sort order so two
                # matches of the same SM subgraph produce identical tuple
                # bindings (cheap equality and hashability for
                # deduplication in the matcher and the MILP). Keep the
                # canonicalized ordering as the single source of truth —
                # the rest of the Translator assumes this shape.
                unique_by_obj: Dict[Any, Any] = {}
                for _, sm_object in matched:
                    unique_by_obj.setdefault(sm_object, sm_object)
                tuple_binding = tuple(
                    sorted(unique_by_obj.keys(), key=lambda o: str(o.uri))
                )
                first_idx = matched[0][0]
                pairs.append(
                    (
                        maps_for_match,
                        tuple_binding,
                        self.object,
                        SetStepRule,
                        first_idx,
                    )
                )

        rule_applies = np.any(rule_applies_vec)
        for pair in pairs:
            try:
                preview = ", ".join(
                    o.get_short_name() for o in pair[1][:3]
                )
                if len(pair[1]) > 3:
                    preview += ", ..."
            except Exception:
                preview = str(pair[1])
            LOGGER.debug(
                "Matched (set, %d): [%s] is %s",
                len(pair[1]),
                preview,
                self.object.cls,
            )
        LOGGER.debug("Rule applies: %s", rule_applies)
        LOGGER.remove_level()
        return pairs, rule_applies, rule_applies_vec, ruleset

    def reset(self):
        pass


class _SinglePath(Rule):
    """Internal rule for SinglePath traversal (creates intermediate SP nodes)."""

    def __init__(self, **kwargs):
        self.first_entry = True
        Rule.__init__(self, **kwargs)

    @property
    def subject(self):
        assert len(self._subject) == 1, "The number of subjects must be 1."
        return self._subject[0]

    @property
    def object(self):
        assert len(self._object) == 1, "The number of objects must be 1."
        return self._object[0]

    @property
    def predicate(self):
        assert len(self._predicate) == 1, "The number of predicates must be 1."
        return self._predicate[0]

    def apply(
        self,
        sm_subject: core.SemanticObject,
        sm_objects: List[core.SemanticObject],
        ruleset: Dict[Tuple[Node, Optional[Predicate], Node], Rule],
        candidate_maps: Optional[List[Optional[Any]]] = None,
        master_rule: Optional[Rule] = None,
    ) -> Tuple[
        List[Tuple[Optional[List], core.SemanticObject, Node, type]],
        bool,
        Dict[Tuple[Node, Optional[Predicate], Node], Rule],
    ]:
        """
        Apply SinglePath traversal rule.

        Creates intermediate SP subject nodes for path traversal. On first entry,
        accepts all SM objects. On subsequent entries, only accepts SM objects
        that have exactly one child for the predicate (single path constraint).
        """
        LOGGER.debug("Applying %s", self.__class__.__name__)
        LOGGER.add_level()
        if master_rule is None:
            master_rule = self
        pairs = []
        matched_sm_objects = []
        rule_applies_vec = np.array([False] * len(sm_objects))

        if self.first_entry:
            self.first_entry = False
            matched_sm_objects.extend(
                [(i, sm_object) for i, sm_object in enumerate(sm_objects)]
            )
            rule_applies_vec = np.array([True] * len(sm_objects))
        else:
            # Only allow single-path continuation
            if len(sm_objects) == 1:
                for i, sm_object in enumerate(sm_objects):
                    predicate_objects = sm_object.get_predicate_object_pairs()
                    # Check all predicates in the Predicate's tuple
                    for pred in self.predicate.preds:
                        if (
                            pred in predicate_objects
                            and len(predicate_objects[pred]) == 1
                        ):
                            matched_sm_objects.append((i, sm_object))
                            rule_applies_vec[i] = True
                            break  # Found a valid predicate, no need to check others
        rule_applies = np.any(rule_applies_vec)
        if rule_applies:
            for i, sm_object in matched_sm_objects:
                # Create intermediate SP subject node for continued traversal
                # Hash ensures uniqueness based on context (use predicate's preds tuple)
                intermediate_sp_subject = Node(
                    cls=sm_object.get_most_specific_type(allow_multiple_classes=True),
                    hash_=(sm_object, self.subject, self.predicate.preds, self.object),
                )
                intermediate_sp_subject.set_signature_pattern(
                    self.object.signature_pattern
                )
                intermediate_sp_subject.validate_cls()
                intermediate_sp_subject.predicate_object_pairs[self.predicate] = [
                    self.object
                ]
                ruleset[(intermediate_sp_subject, self.predicate, self.object)] = (
                    master_rule
                )
                pairs.append(
                    (candidate_maps, sm_object, intermediate_sp_subject, _SinglePath, i)
                )

        for pair in pairs:
            LOGGER.debug(
                "Matched: %s (%s) is %s",
                pair[1].get_short_name(),
                (mst.get_short_name() if (mst := pair[1].get_most_specific_type()) is not None else "None"),
                self.object.cls,
            )
        LOGGER.debug("Rule applies: %s", rule_applies)
        LOGGER.remove_level()
        return pairs, rule_applies, rule_applies_vec, ruleset

    def reset(self):
        self.first_entry = True


class PathRule(Rule):
    r"""
    Multi-hop traversal rule — a path of one or more steps with scalar
    binding on the endpoint.

    Name
    ----
    ``Path`` = traversal over one or more predicate hops; no prefix
    modifier = scalar branching. A path is a sequence of steps;
    :class:`StepRule` is the one-hop atom and ``PathRule`` is the
    multi-hop composite.

    Asserts
    -------
    Every match contains a single directed path from ``SM(s)`` to
    ``SM(o)`` whose edges all satisfy ``predicate``. The path may be one
    or more hops (so ``PathRule`` subsumes ``StepRule`` semantically,
    and in fact delegates to ``StepRule | _SinglePath``).

    Matcher behavior
    ----------------
    DFS from ``SM(s)``; each reached endpoint ``SM(o_candidate)``
    produces a separate branch with scalar binding
    ``group[sp_object] = SM(o_candidate)``. Intermediate SM nodes are
    represented as synthetic SP nodes so the subgraph shape remains
    validable. With ``stop_early=True`` (default), the traversal stops
    at the first direct-edge match when one exists.

    Binding produced
    ----------------
    ``SemanticObject`` (scalar) — the endpoint.

    Composition
    -----------
    From downstream rules' perspective, ``PathRule`` is indistinguishable
    from :class:`StepRule` — both produce scalar bindings on the
    endpoint node. Composition with :class:`SetStepRule` is out of scope
    in the initial implementation (a ``SetPathRule`` is a reserved name
    for a future extension).

    When to use vs. siblings
    ------------------------
    - :class:`StepRule` when the relationship is a single triple.
    - ``PathRule`` when the exact number of hops is irrelevant but
      there is a single canonical path.
    - :class:`AnyPathRule` when multiple structurally distinct paths
      between the same endpoints should all count as valid matches.

    Example
    -------

    >>> sp.add_rule(PathRule(
    ...     subject=ahu,
    ...     object=vav,
    ...     predicate=core.namespace.BRICK.feeds,
    ... ))
    >>> # ``vav`` is any node reachable from ``ahu`` via a feeds-path of
    >>> # one or more hops.
    """

    def __init__(self, stop_early=True, **kwargs):
        # Normalize predicate to Predicate class (similar to how we normalize cls to tuple in Node)
        super().__init__(**kwargs)
        if kwargs.get("predicate") is None:
            predicate = None
        elif isinstance(kwargs.get("predicate"), Predicate):
            predicate = kwargs.get("predicate")
        else:
            # Deprecation warning
            warnings.warn(
                "The 'predicate' argument is deprecated. Use the 'Predicate' class instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            # Auto-wrap in Predicate (handles single value or tuple)
            predicate = Predicate(kwargs.get("predicate"))
        kwargs["predicate"] = predicate
        self.rule = StepRule(**kwargs) | _SinglePath(**kwargs)  # This order
        self.stop_early = stop_early
        # super().__init__(**kwargs)

    def apply(
        self,
        sm_subject: core.SemanticObject,
        sm_objects: List[core.SemanticObject],
        ruleset: Dict[Tuple[Node, Optional[Predicate], Node], Rule],
        candidate_maps: Optional[List[Optional[Any]]] = None,
        master_rule: Optional[Rule] = None,
    ) -> Tuple[
        List[Tuple[Optional[List], core.SemanticObject, Node, type]],
        bool,
        Dict[Tuple[Node, Optional[Predicate], Node], Rule],
    ]:
        """Delegate to internal Exact | _SinglePath rule."""
        if master_rule is None:
            master_rule = self
        pairs, rule_applies, rule_applies_vec, ruleset = self.rule.apply(
            sm_subject,
            sm_objects,
            ruleset,
            candidate_maps=candidate_maps,
            master_rule=master_rule,
        )
        return pairs, rule_applies, rule_applies_vec, ruleset

    def reset(self):
        self.rule.first_entry = True


class _MultiPath(Rule):
    """Internal rule for MultiPath traversal (creates intermediate SP nodes)."""

    def __init__(self, **kwargs):
        self.first_entry = True
        super().__init__(**kwargs)

    @property
    def subject(self):
        assert len(self._subject) == 1, "The number of subjects must be 1."
        return self._subject[0]

    @property
    def object(self):
        assert len(self._object) == 1, "The number of objects must be 1."
        return self._object[0]

    @property
    def predicate(self):
        assert len(self._predicate) == 1, "The number of predicates must be 1."
        return self._predicate[0]

    def apply(
        self,
        sm_subject: core.SemanticObject,
        sm_objects: List[core.SemanticObject],
        ruleset: Dict[Tuple[Node, Optional[Predicate], Node], Rule],
        candidate_maps: Optional[List[Optional[Any]]] = None,
        master_rule: Optional[Rule] = None,
    ) -> Tuple[
        List[Tuple[Optional[List], core.SemanticObject, Node, type]],
        bool,
        Dict[Tuple[Node, Optional[Predicate], Node], Rule],
    ]:
        """
        Apply MultiPath traversal rule.

        Creates intermediate SP subject nodes for path traversal. On first entry,
        accepts all SM objects. On subsequent entries, accepts SM objects that
        have at least one child for the predicate (multi-path allows branching).
        """
        LOGGER.debug("Applying %s", self.__class__.__name__)
        LOGGER.add_level()
        if master_rule is None:
            master_rule = self
        pairs = []
        matched_sm_objects = []
        rule_applies_vec = np.array([False] * len(sm_objects))

        if self.first_entry:
            self.first_entry = False
            matched_sm_objects.extend(
                [(i, sm_object) for i, sm_object in enumerate(sm_objects)]
            )
            rule_applies_vec = np.array([True] * len(sm_objects))
        else:
            # Allow multi-path continuation (>= 1 child)
            if len(sm_objects) >= 1:
                for i, sm_object in enumerate(sm_objects):
                    predicate_objects = sm_object.get_predicate_object_pairs()
                    # Check all predicates in the Predicate's tuple
                    for pred in self.predicate.preds:
                        if (
                            pred in predicate_objects
                            and len(predicate_objects[pred]) >= 1
                        ):
                            matched_sm_objects.append((i, sm_object))
                            rule_applies_vec[i] = True
                            break  # Found a valid predicate, no need to check others

        rule_applies = np.any(rule_applies_vec)
        if rule_applies:

            for i, sm_object in matched_sm_objects:
                # Create intermediate SP subject node for continued traversal
                intermediate_sp_subject = Node(
                    cls=sm_object.get_most_specific_type(allow_multiple_classes=True),
                    hash_=(sm_object, self.subject, self.predicate.preds, self.object),
                )
                intermediate_sp_subject.set_signature_pattern(
                    self.object.signature_pattern
                )
                intermediate_sp_subject.validate_cls()
                intermediate_sp_subject.predicate_object_pairs[self.predicate] = [
                    self.object
                ]
                ruleset[(intermediate_sp_subject, self.predicate, self.object)] = (
                    master_rule
                )
                pairs.append(
                    (candidate_maps, sm_object, intermediate_sp_subject, _MultiPath, i)
                )

        for pair in pairs:
            LOGGER.debug(
                "Matched: %s (%s) is %s",
                pair[1].get_short_name(),
                (mst.get_short_name() if (mst := pair[1].get_most_specific_type()) is not None else "None"),
                self.object.cls,
            )
        LOGGER.debug("Rule applies: %s", rule_applies)
        LOGGER.remove_level()
        return pairs, rule_applies, rule_applies_vec, ruleset

    def reset(self):
        self.first_entry = True


class OptionalRule(Rule):
    r"""
    Conditional-presence decorator — turns any required rule into an
    optional one.

    Name
    ----
    ``Optional`` names the modality (conditional presence); the wrapped
    rule supplies the topology and binding shape. Reads as "rule whose
    assertion is evaluated but whose failure is tolerated".

    Asserts
    -------
    "If the inner rule can match, match and bind. If not, succeed
    without adding any binding." Thus ``OptionalRule`` never prunes a
    branch on its own — it only *augments* a match when possible.

    Matcher behavior
    ----------------
    Runs the inner rule; on zero matches it returns success with no
    binding added (rather than pruning the branch). When matches exist,
    it behaves like the inner rule — branches per scalar match, or
    emits a single tuple per set-bound inner.

    Binding produced
    ----------------
    Inherits the inner rule's binding shape when matched; ``None``
    (scalar inner) or ``()`` (set inner) when skipped.

    Composition
    -----------
    - ``OptionalRule(inner=StepRule(...))`` — the canonical "may or may
      not have this point" pattern.
    - ``OptionalRule(inner=SetStepRule(...))`` — optional set-binding:
      "this SP node *may* be set-bound, or empty". The
      :meth:`Translator.__broadcast_recurse` closure treats ``()``
      elsewhere as "no descendants reached".

    When to use vs. siblings
    ------------------------
    Use ``OptionalRule`` only for relationships whose absence should
    *not* fail the pattern. Required relationships stay bare
    (``StepRule``/``SetStepRule``/``PathRule``).

    Example
    -------

    >>> sp.add_rule(OptionalRule(inner=StepRule(
    ...     subject=vav,
    ...     object=co2_sensor,
    ...     predicate=core.namespace.BRICK.hasPoint,
    ... )))
    >>> # Match works whether or not the VAV has a CO2 sensor.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def subject(self):
        assert len(self._subject) == 1, "The number of subjects must be 1."
        return self._subject[0]

    @property
    def object(self):
        assert len(self._object) == 1, "The number of objects must be 1."
        return self._object[0]

    @property
    def predicate(self):
        assert len(self._predicate) == 1, "The number of predicates must be 1."
        return self._predicate[0]

    def apply(
        self,
        sm_subject: core.SemanticObject,
        sm_objects: List[core.SemanticObject],
        ruleset: Dict[Tuple[Node, Optional[Predicate], Node], Rule],
        candidate_maps: Optional[List[Optional[Any]]] = None,
        master_rule: Optional[Rule] = None,
    ) -> Tuple[
        List[Tuple[Optional[List], core.SemanticObject, Node, type]],
        bool,
        Dict[Tuple[Node, Optional[Predicate], Node], Rule],
    ]:
        """
        Apply Optional rule to find matching SM objects.

        Optional rules match if SM objects exist and have the correct type,
        but don't cause pruning if no match is found.
        """
        LOGGER.debug("Applying %s", self.__class__.__name__)
        LOGGER.add_level()
        if master_rule is None:
            master_rule = self
        pairs = []
        rule_applies_vec = np.array([False] * len(sm_objects))

        for i, sm_object in enumerate(sm_objects):
            if sm_object.isinstance(self.object.cls):
                pairs.append((candidate_maps, sm_object, self.object, OptionalRule, i))
                rule_applies_vec[i] = True

        rule_applies = np.any(rule_applies_vec)
        for pair in pairs:
            LOGGER.debug(
                "Matched: %s (%s) is %s",
                pair[1].get_short_name(),
                (mst.get_short_name() if (mst := pair[1].get_most_specific_type()) is not None else "None"),
                self.object.cls,
            )
        LOGGER.debug("Rule applies: %s", rule_applies)
        LOGGER.remove_level()
        return pairs, rule_applies, rule_applies_vec, ruleset

    def reset(self):
        pass


class AnyPathRule(Rule):
    r"""
    Any-of-alternatives traversal rule — a match succeeds if *any* of
    several structurally distinct paths between the endpoints exists.

    Name
    ----
    ``Any`` prefix = "any of several alternative paths satisfies"
    (scalar branching, multi-path traversal); ``Path`` = traversal.
    Reads as "rule satisfied by any path in this family".

    Asserts
    -------
    Every match contains at least one path from ``SM(s)`` to ``SM(o)``
    under the predicate expression — multiple structurally distinct SM
    paths are admissible and each admissible endpoint becomes its own
    scalar branch.

    Matcher behavior
    ----------------
    Like :class:`PathRule` but relaxes uniqueness of the traversal
    shape. Each endpoint still becomes its own scalar branch. Can be
    expensive and, in cyclic SM subgraphs, may blow up — the existing
    "use sparingly" warning stands. Typically prefer two explicit
    :class:`PathRule`\ s.

    Binding produced
    ----------------
    ``SemanticObject`` (scalar) — the endpoint. Does *not* collapse
    bindings into a set; for that, a future ``SetPathRule`` would be
    needed.

    Composition
    -----------
    Downstream consumers treat ``AnyPathRule`` and :class:`PathRule`
    identically (both produce scalar endpoints). Composition with
    :class:`SetStepRule` is out of scope in the initial implementation.

    When to use vs. siblings
    ------------------------
    Almost never in practice — prefer two explicit :class:`PathRule`\ s
    over one ``AnyPathRule``. Retained for patterns where enumerating
    alternatives is infeasible.

    Example
    -------

    >>> sp.add_rule(AnyPathRule(
    ...     subject=ahu,
    ...     object=terminal_unit,
    ...     predicate=core.namespace.BRICK.feeds,
    ... ))
    """

    def __init__(self, stop_early=True, endpoints_only=False, **kwargs):
        # Normalize predicate to Predicate class (similar to how we normalize cls to tuple in Node)
        if kwargs.get("predicate") is None:
            predicate = None
        elif isinstance(kwargs.get("predicate"), Predicate):
            predicate = kwargs.get("predicate")
        else:
            # Deprecation warning
            warnings.warn(
                "The 'predicate' argument is deprecated. Use the 'Predicate' class instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            # Auto-wrap in Predicate (handles single value or tuple)
            predicate = Predicate(kwargs.get("predicate"))
        kwargs["predicate"] = predicate
        self.rule = StepRule(**kwargs) | _MultiPath(**kwargs)
        self.stop_early = stop_early
        self.endpoints_only = endpoints_only
        super().__init__(**kwargs)

    @property
    def subject(self):
        assert len(self._subject) == 1, "The number of subjects must be 1."
        return self._subject[0]

    @property
    def object(self):
        assert len(self._object) == 1, "The number of objects must be 1."
        return self._object[0]

    @property
    def predicate(self):
        assert len(self._predicate) == 1, "The number of predicates must be 1."
        return self._predicate[0]

    def _apply_endpoints_only(self, sm_subject, sm_objects, ruleset, candidate_maps):
        """BFS from sm_objects following predicate edges, returning only target-type endpoints.

        Replaces the recursive Or(StepRule | _MultiPath) chain with an O(V+E) graph
        traversal. Only the (source, target) endpoint pairs are returned — no intermediate
        SP nodes are created, no ruleset mutations, no comparison_table interaction.
        """
        target_cls = self.object.cls
        preds = self.predicate.preds

        visited = set()
        queue = collections.deque(sm_objects)
        endpoints = []

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            if node.isinstance(target_cls):
                endpoints.append(node)
                continue

            pred_objects = node.get_predicate_object_pairs()
            for pred in preds:
                for child in pred_objects.get(pred, []):
                    if child not in visited:
                        queue.append(child)

        pairs = []
        for endpoint in endpoints:
            pairs.append((candidate_maps, endpoint, self.object, AnyPathRule, 0))

        rule_applies_vec = np.array([bool(endpoints)] * len(sm_objects))
        return pairs, bool(endpoints), rule_applies_vec, ruleset

    def apply(
        self,
        sm_subject: core.SemanticObject,
        sm_objects: List[core.SemanticObject],
        ruleset: Dict[Tuple[Node, Optional[Predicate], Node], Rule],
        candidate_maps: Optional[List[Optional[Any]]] = None,
        master_rule: Optional[Rule] = None,
    ) -> Tuple[
        List[Tuple[Optional[List], core.SemanticObject, Node, type]],
        bool,
        Dict[Tuple[Node, Optional[Predicate], Node], Rule],
    ]:
        if self.endpoints_only:
            return self._apply_endpoints_only(
                sm_subject, sm_objects, ruleset, candidate_maps
            )
        pairs, rule_applies, rule_applies_vec, ruleset = self.rule.apply(
            sm_subject,
            sm_objects,
            ruleset,
            candidate_maps=candidate_maps,
            master_rule=master_rule,
        )
        return pairs, rule_applies, rule_applies_vec, ruleset

    def reset(self):
        self.rule.first_entry = True


class SetAnyPathRule(SetStepRule):
    r"""
    Set-binding multi-hop traversal rule — bundles every endpoint
    reachable from ``SM(s)`` along the predicate expression into a
    single tuple binding on ``sp_object``.

    Name
    ----
    ``Set`` prefix = tuple-valued binding (the default is scalar);
    ``AnyPath`` = multi-hop alternative-path traversal (any directed
    path of one or more hops under the predicate). Reads as "rule that
    binds the object to the set of SM endpoints reachable from
    ``SM(s)`` via any path under this predicate". This fills the slot
    flagged by :class:`SetStepRule`'s docstring ("composition with
    :class:`AnyPathRule` traversals is out of scope for the initial
    implementation") and by the AHU pattern comment block in
    ``air_handling_unit_torch_system.py``.

    Asserts
    -------
    Every match contains *at least one* directed path from ``SM(s)`` to
    every ``SM(o_i)`` in the bound tuple, where each path is a sequence
    of one or more edges all satisfying ``predicate``. The tuple
    enumerates *all* such reachable endpoints (deduplicated and sorted
    by IRI). Zero reachable endpoints prune the branch (same strictness
    as :class:`SetStepRule`).

    Matcher behavior
    ----------------
    BFS from ``sm_objects`` (the direct predicate-children of
    ``sm_subject``) through the same predicate(s); any node whose type
    satisfies ``self.object.cls`` is collected as an endpoint and BFS
    does not descend past it. Endpoints are bundled into a single
    canonical tuple and emitted as one pair per ``current_map`` in
    ``candidate_maps`` — exactly the emission shape :meth:`SetStepRule.apply`
    uses, so the matcher's tuple-binding branch in :meth:`Translator._match`
    handles both rules identically.

    Where :class:`AnyPathRule` emits ``N`` scalar branches for ``N``
    reachable endpoints, ``SetAnyPathRule`` emits exactly one branch
    with a tuple of length ``N``.

    Binding produced
    ----------------
    ``Tuple[SemanticObject, ...]`` (set-valued) on ``sp_object``.

    Auto-broadcast
    --------------
    Inherits :class:`SetStepRule`'s auto-broadcast semantics: any
    downstream rule whose subject is set-bound (this rule's ``object``,
    or any node transitively reached from it) is automatically applied
    per element. The ``isinstance(sub_rule, SetStepRule)`` check in
    :meth:`SignaturePattern.add_rule` recognises this rule via
    inheritance, so no changes to the propagation closure are needed.

    Composition
    -----------
    - Use on indirect (multi-hop) edges where :class:`SetStepRule`'s
      single-hop restriction is too strict.
    - Downstream :class:`StepRule`/:class:`NoStepRule` on the set-bound
      object are auto-broadcast (same as :class:`SetStepRule`).
    - Composition with another :class:`SetStepRule` /
      ``SetAnyPathRule`` on the set-bound object is **not** supported
      (would require nested-set semantics, same restriction as
      :class:`SetStepRule`).

    When to use vs. siblings
    ------------------------
    - :class:`SetStepRule` when the relationship is a single triple and
      tuple binding is required.
    - :class:`AnyPathRule` when the relationship is multi-hop but each
      reachable endpoint should beget a separate component.
    - ``SetAnyPathRule`` when the relationship is multi-hop **and**
      every reachable endpoint jointly describes one simulation entity
      (e.g. an AHU and the set of all zones it ultimately feeds via a
      VAV cascade).

    Example
    -------

    >>> sp.add_rule(SetAnyPathRule(
    ...     subject=ahu,
    ...     object=spaces,
    ...     predicate=Predicate((BRICK.feeds, FSO.feedsFluidTo)),
    ... ))
    >>> # ``spaces`` binds to the tuple of every Room/HVAC_Zone reachable
    >>> # from ``ahu`` via any feeds-path (e.g. AHU -> VAV -> Space),
    >>> # in one group per AHU.
    """

    def __init__(self, stop_early=True, **kwargs):
        # Normalize predicate to Predicate class (mirrors AnyPathRule).
        if kwargs.get("predicate") is None:
            predicate = None
        elif isinstance(kwargs.get("predicate"), Predicate):
            predicate = kwargs.get("predicate")
        else:
            warnings.warn(
                "The 'predicate' argument is deprecated. Use the 'Predicate' class instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            predicate = Predicate(kwargs.get("predicate"))
        kwargs["predicate"] = predicate
        self.stop_early = stop_early
        super().__init__(**kwargs)

    def apply(
        self,
        sm_subject: core.SemanticObject,
        sm_objects: List[core.SemanticObject],
        ruleset: Dict[Tuple[Node, Optional[Predicate], Node], Rule],
        candidate_maps: Optional[List[Optional[Any]]] = None,
        master_rule: Optional[Rule] = None,
    ) -> Tuple[
        List[Tuple[Optional[List], core.SemanticObject, Node, type]],
        bool,
        Dict[Tuple[Node, Optional[Predicate], Node], Rule],
    ]:
        """Apply set-bound multi-hop rule.

        BFS from ``sm_objects`` through the rule's predicate(s),
        collecting every reachable node whose type satisfies
        ``self.object.cls`` as an endpoint. The endpoints are
        deduplicated, canonicalised by IRI sort, and emitted as a
        single tuple binding per ``candidate_map`` — the same emission
        shape :meth:`SetStepRule.apply` uses, so downstream tuple
        handling is shared.
        """
        LOGGER.debug("Applying %s", self.__class__.__name__)
        LOGGER.add_level()
        if master_rule is None:
            master_rule = self

        target_cls = self.object.cls
        preds = self.predicate.preds

        # BFS to collect every reachable endpoint of the target type.
        # ``sm_objects`` is the set of direct predicate-children of
        # ``sm_subject``; from each we walk further predicate edges
        # until we hit a node whose type satisfies ``target_cls``.
        visited: set = set()
        queue = collections.deque(sm_objects)
        endpoints: List[Any] = []
        endpoint_seen: set = set()

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            if node.isinstance(target_cls):
                if node not in endpoint_seen:
                    endpoint_seen.add(node)
                    endpoints.append(node)
                # Do not descend past an endpoint — same convention as
                # AnyPathRule._apply_endpoints_only.
                continue

            pred_objects = node.get_predicate_object_pairs()
            for pred in preds:
                for child in pred_objects.get(pred, []):
                    if child not in visited:
                        queue.append(child)

        rule_applies_vec = np.array([bool(endpoints)] * len(sm_objects))
        rule_applies = bool(endpoints)
        pairs: List[Tuple[Any, Any, Node, type, int]] = []

        if rule_applies:
            tuple_binding = tuple(sorted(endpoints, key=lambda o: str(o.uri)))

            if not candidate_maps:
                maps_iter: List[Optional[Any]] = [None]
            else:
                maps_iter = list(candidate_maps)

            for current_map in maps_iter:
                maps_for_match = [current_map] if current_map is not None else []
                pairs.append(
                    (
                        maps_for_match,
                        tuple_binding,
                        self.object,
                        SetStepRule,
                        0,
                    )
                )

            try:
                preview = ", ".join(o.get_short_name() for o in tuple_binding[:3])
                if len(tuple_binding) > 3:
                    preview += ", ..."
            except Exception:
                preview = str(tuple_binding)
            LOGGER.debug(
                "Matched (set-anypath, %d): [%s] is %s",
                len(tuple_binding),
                preview,
                self.object.cls,
            )

        LOGGER.debug("Rule applies: %s", rule_applies)
        LOGGER.remove_level()
        return pairs, rule_applies, rule_applies_vec, ruleset

    def reset(self):
        pass


# ---------------------------------------------------------------------------
# Backwards-compatibility aliases with deprecation warnings.
#
# The "Step"/"Path" taxonomy supersedes the earlier "Exact"/"SinglePath"
# /"MultiPath" taxonomy. Each old name now chains through its immediate
# predecessor so a single user call can surface multiple levels of
# deprecation (e.g. ``Exact -> ExactRule -> StepRule``) and migration is
# discoverable from the emitted warnings.
# ---------------------------------------------------------------------------
class ExactRule(StepRule):
    """Deprecated alias for :class:`StepRule`. Emits ``DeprecationWarning``."""

    def __init__(self, **kwargs):
        warnings.warn(
            "The 'ExactRule' class is deprecated. Use 'StepRule' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


class NoExactRule(NoStepRule):
    """Deprecated alias for :class:`NoStepRule`. Emits ``DeprecationWarning``."""

    def __init__(self, **kwargs):
        warnings.warn(
            "The 'NoExactRule' class is deprecated. Use 'NoStepRule' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


class UniPathRule(PathRule):
    """Deprecated alias for :class:`PathRule`. Emits ``DeprecationWarning``."""

    def __init__(self, **kwargs):
        warnings.warn(
            "The 'UniPathRule' class is deprecated. Use 'PathRule' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


class MultiPathRule(AnyPathRule):
    """Deprecated alias for :class:`AnyPathRule`. Emits ``DeprecationWarning``."""

    def __init__(self, **kwargs):
        warnings.warn(
            "The 'MultiPathRule' class is deprecated. Use 'AnyPathRule' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


class Exact(ExactRule):
    """Deprecated alias. Prefer :class:`StepRule`."""

    def __init__(self, **kwargs):
        warnings.warn(
            "The 'Exact' class is deprecated. Use 'StepRule' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


class SinglePath(UniPathRule):
    """Deprecated alias. Prefer :class:`PathRule`."""

    def __init__(self, **kwargs):
        warnings.warn(
            "The 'SinglePath' class is deprecated. Use 'PathRule' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


class MultiPath(MultiPathRule):
    """Deprecated alias. Prefer :class:`AnyPathRule`."""

    def __init__(self, **kwargs):
        warnings.warn(
            "The 'MultiPath' class is deprecated. Use 'AnyPathRule' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


class Optional_(OptionalRule):
    """Deprecated alias. Prefer :class:`OptionalRule`."""

    def __init__(self, **kwargs):
        warnings.warn(
            "The 'Optional_' class is deprecated. Use 'OptionalRule' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


# if __name__ == "__main__":

#     n1 = Node(cls=core.namespace.S4BLDG.Damper)
#     n2 = Node(cls=core.namespace.S4BLDG.Controller)
#     n3 = Node(cls=core.namespace.S4BLDG.Coil)
#     n4 = Node(cls=core.namespace.S4BLDG.AirToAirHeatRecovery)
#     n5 = Node(cls=core.namespace.S4BLDG.Valve)

#     r1 = NoStepRule(subject=n1, object=n2, predicate=core.namespace.SAREF.controls)

#     r2 = PathRule(subject=n3, object=n4, predicate=core.namespace.SAREF.feeds)

#     r3 = StepRule(subject=n2, object=n5, predicate=core.namespace.SAREF.controls)

#     r4 = And(r1, r2)

#     r5 = Or(r3, r4)

#     print(n1)
#     print(n2)
#     print(n3)
#     print(n4)
#     print(n5)

#     print(r5.rules)
#     print(r5.object)

#     sp = SignaturePattern()
#     sp.add_triple(r5)
#     print(sp.nodes)
