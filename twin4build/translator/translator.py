from __future__ import annotations

# Standard library imports
import inspect
import warnings
from dataclasses import dataclass
from itertools import count
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings


# Third party imports
import numpy as np
from sympy import I
import torch

# import twin4build.saref4syst.system as system
# import twin4build.model.simulation_model as simulation_model
# import twin4build.model.semantic_model.semantic_model as semantic_model
import torch.nn as nn
from rdflib import Literal, URIRef
from scipy.optimize import Bounds, LinearConstraint, milp

# Local application imports
import twin4build.core as core
import twin4build.systems as systems
import twin4build.utils.types as tps
from twin4build.utils.print_progress import LOGGER, autoreset_print
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rsetattr import rsetattr

import time ###
from line_profiler import profile

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
    ) -> core.SimulationModel:
        """
        Translate semantic model to simulation model using pattern matching

        Args:
            systems: List of system types to match against
            semantic_model: The semantic model to translate
            verbose: Verbosity level
        Returns:
            SimulationModel instance with matched components
        """
        LOGGER.verbose = verbose
        LOGGER("Applying translator", status="")
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
                LOGGER.info("Class: %s", component_cls.__name__)
                LOGGER.add_level()

                for sp in complete_groups[component_cls].keys():
                    LOGGER.info(
                        "Signature pattern: %s, %d matches found", sp.id, len(complete_groups[component_cls][sp])
                    )
                    LOGGER.add_level()
                    for i, group in enumerate(complete_groups[component_cls][sp]):
                        LOGGER.info("Group %d:", i)
                        LOGGER.add_level()

                        for sp_subject, sm_subject in group.items():
                            # id_sp = str([str(s) for s in sp_subject.cls])
                            id_sp = sp_subject.id
                            id_m = (
                                sm_subject.get_short_name()
                                if sm_subject is not None
                                else None
                            )  # Can be None if sp includes an Optional_ node.
                            LOGGER.info("%s: %s", id_sp, id_m)
                        LOGGER.remove_level()
                    LOGGER.remove_level()

                for sp in incomplete_groups[component_cls].keys():
                    LOGGER.info(
                        "INCOMPLETE signature patterns: %s, %d", sp.id, len(incomplete_groups[component_cls][sp])
                    )
                    LOGGER.add_level()
                    for i, group in enumerate(incomplete_groups[component_cls][sp]):
                        LOGGER.info("Group %d:", i)
                        LOGGER.add_level()

                        for sp_subject, sm_subject in group.items():
                            # id_sp = str([str(s) for s in sp_subject.cls])
                            id_sp = sp_subject.id
                            id_m = (
                                sm_subject.get_short_name()
                                if sm_subject is not None
                                else None
                            )
                            LOGGER.info("%s: %s", id_sp, id_m)
                        LOGGER.remove_level()
                    LOGGER.remove_level()

                LOGGER.remove_level()

        else:
            raise Exception("No matching patterns found.")

        # Create component instances
        self._instantiate_components(complete_groups, semantic_model)

        if len(self._sim2sem_map) == 0:
            raise Exception("No components instantiated.")

        result = self._solve_milp()
        if result["success"]:
            # Initialize simulation model
            sim_model = core.SimulationModel(id=semantic_model.id)

            # Connect components
            self._connect_components(result["connections"], sim_model)
            LOGGER.ok("Applying translator", change_status=True)
            LOGGER.remove_level()
        else:
            # This can happen if all components require no inputs. In this case, we can just return the simulation model with no connections. But this is probably not wanted behavior - better to raise an exception.
            LOGGER.error(result["message"])
            LOGGER.error("Applying translator", change_status=True)
            LOGGER.remove_level()
            sim_model = None
            raise Exception(f"MILP solver failed: {result['message']}")

        return sim_model

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
        LOGGER.debug("Matching patterns")
        LOGGER.add_level()

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
                        candidate_maps, _, _, is_pruned = Translator._prune_recursive(
                            sm_node, sp_node, candidate_maps,
                            feasible, comparison_table, signature_pattern,
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
                                    complete_matches.append(mapping)
                                else:
                                    
                                    incomplete_matches = Translator._try_merge_with_incomplete(
                                        mapping, incomplete_matches,
                                        complete_matches, signature_pattern,
                                    )

            # ===================================================================
            # PHASE 4: Merge incomplete groups with each other
            # ===================================================================
            incomplete_matches = Translator._merge_incomplete_groups(
                incomplete_matches, complete_matches, signature_pattern,
            )

            # ===================================================================
            # PHASE 5: Final check - move any newly complete groups
            # ===================================================================
            # Safety net: ensure no complete groups left in incomplete list
            # still_incomplete = [
            #     g for g in incomplete_matches
            #     if not all(g[n] is not None for n in signature_pattern.required_nodes)
            # ]
            # newly_complete = [
            #     g for g in incomplete_matches
            #     if all(g[n] is not None for n in signature_pattern.required_nodes)
            # ]
            # complete_matches.extend(newly_complete)
            # incomplete_matches[:] = still_incomplete


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
            LOGGER.debug(f"Processing component class: %s", component_cls.__name__)
            LOGGER.add_level()
            complete_groups[component_cls] = {}
            incomplete_groups[component_cls] = {}

            
            

            for signature_pattern in component_cls.sp:
                LOGGER.debug(f"Matching signature pattern: %s", signature_pattern.id)
                LOGGER.add_level()
                # Ensure semantic model has all namespaces from the pattern
                semantic_model.add_namespaces(signature_pattern.semantic_model.namespaces)

                # Match this pattern against the semantic model
                _match_single_pattern(component_cls, signature_pattern, complete_groups, incomplete_groups)

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


                

            LOGGER.remove_level()

        LOGGER.remove_level()
        return complete_groups, incomplete_groups

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
            LOGGER.debug("Checking merge with existing group (%d nodes)", len(existing_group))
            LOGGER.add_level()
            if Translator._match(
                existing_group, new_mapping, signature_pattern,
                complete_matches, updated_incomplete
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
        
        previous_count = float('inf')

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
                        group_i, group_j, signature_pattern,
                        complete_matches, updated_incomplete
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

        LOGGER.info("Solving MILP problem")
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
                for i, y in enumerate(x):
                    print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
                print("")

        def resolve_port_indices(groups_source, groups_target, output_port_index, input_port_index, sm_for_index):
            """
            Resolve Node-based port indices to integer indices using group matching.
            
            Both output_port_index and input_port_index Nodes are from the TARGET's
            signature pattern. This function finds which group index corresponds to
            the given semantic model instance.
            
            - input_port_index=Node (Scalar→Vector, Vector→Vector):
              Find which slot in groups_target contains the matching semantic instance.
              The Node specifies how the target's input vector is indexed.
              
            - output_port_index=Node (Vector→Scalar):
              Find which slot in groups_source contains the matching semantic instance.
              The Node specifies which element to pick from source's output vector.
            
            Args:
                groups_source: List of group dicts from the source component
                groups_target: List of group dicts from the target component
                output_port_index: Node instance or None (from target's pattern)
                input_port_index: Node instance or None (from target's pattern)
                sm_for_index: The semantic model instance to search for:
                    - For output_port_index: semantic instance from target group
                    - For input_port_index: semantic instance from target group
            
            Returns:
                tuple: (resolved_output_port_index, resolved_input_port_index)
                    - For scalar connections: (None, None)
                    - For Vector→Scalar: (int, None) - index into source's output
                    - For Scalar→Vector: (None, int) - index into target's input
            """
            if not isinstance(output_port_index, Node) and not isinstance(input_port_index, Node):
                # Scalar → Scalar: no group matching needed
                return output_port_index, input_port_index
            
            if isinstance(input_port_index, Node):
                # Scalar→Vector or Vector→Vector: find which target slot matches
                # Search groups_target for the group where input_port_index maps to sm_for_index
                for i_target, group_target in enumerate(groups_target):
                    if input_port_index in group_target and group_target[input_port_index] == sm_for_index:
                        return None, i_target
                # No match found - this shouldn't happen
                LOGGER.warning("Could not resolve input_port_index: %s not found mapping to %s", 
                             input_port_index, sm_for_index)
                return None, None
            
            elif isinstance(output_port_index, Node):
                # Vector→Scalar: find which source slot matches
                # Search groups_source for the group containing sm_for_index
                for i_source, group_source in enumerate(groups_source):
                    for sp_node, sm_node in group_source.items():
                        if sm_node == sm_for_index:
                            return i_source, None
                # No match found - this shouldn't happen
                LOGGER.warning("Could not resolve output_port_index: %s not found in source groups", 
                             sm_for_index)
                return None, None
            
            return output_port_index, input_port_index

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
                for key, (sp_subject, source_keys, output_port_index, input_port_index) in sp.inputs.items():

                    # _map_port_indices()



                    if key not in required_inputs[component]:
                        required_inputs[component][key] = []

                    # Get all potential source nodes for this input
                    match_nodes = {
                        group[sp_subject] for group in groups if sp_subject in group
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
                                    # Find the appropriate source port/key from the provider
                                    for source_class, source_key in source_keys.items():
                                        # Check if the provider has the required output
                                        for modeled_match_node in p_nodes:
                                            if modeled_match_node.isinstance(
                                                source_class
                                            ):
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
                                        
                                        if isinstance(output_port_index, Node):
                                            # Vector→Scalar: need semantic instance of output_port_index
                                            # from target groups to search in source groups
                                            for group in groups:
                                                if output_port_index in group:
                                                    sm_for_index = group[output_port_index]
                                                    break
                                        elif isinstance(input_port_index, Node):
                                            # Scalar→Vector: need semantic instance of input_port_index
                                            # from target groups to search in target groups
                                            # (Often input_port_index == sp_subject, so sm_subject is correct)
                                            for group in groups:
                                                if input_port_index in group:
                                                    sm_for_index = group[input_port_index]
                                                    break
                                        
                                        resolved_output_idx, resolved_input_idx = resolve_port_indices(
                                            p_groups, groups, output_port_index, input_port_index, sm_for_index
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
                                        self.E_conn_to_sp_group[conn] = (sp, groups, p_sp, p_groups)
                                        if (
                                            provider_component,
                                            source_key,
                                            resolved_output_idx,
                                            resolved_input_idx,
                                        ) not in required_inputs[component][key]:
                                            required_inputs[component][key].append(
                                                (provider_component, source_key, resolved_output_idx, resolved_input_idx)
                                            )
                                    else:
                                        raise Exception(
                                            "Provider does not have required output. This should not happen."
                                        )

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

                    # Create a constraint: Y_i ≤ (E_j1 + E_j2 + ... + E_jn)
                    # This means: If component i is included, at least one provider must be active
                    row = np.zeros(total_vars)
                    row[N_E + component_idx] = 1  # Coefficient for component i

                    edge_indices = []
                    for provider_component, source_key, output_port_index, input_port_index in providers:
                        conn = (provider_component, component, source_key, input_key, output_port_index, input_port_index)
                        edge_idx = E_conn_to_idx[conn]
                        row[edge_idx] = -1  # Negative coefficient for the edge
                        edge_indices.append(edge_idx)

                    if edge_indices:
                        required_input_constraints.append(row)
                        edge_vars = [f"E_{idx}" for idx in edge_indices]
                        constraint_desc = f"Y_{component_idx} ≤ {' + '.join(edge_vars)}"
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

            # Create constraint: E_j ≤ Y_i (connection j can only exist if source component i is included)
            row = np.zeros(total_vars)
            row[e_idx] = 1
            row[N_E + source_idx] = -1
            conn_source_constraints.append(row)
            constraint_desc = f"E_{e_idx} ≤ Y_{source_idx}"
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

            # Create constraint: E_j ≤ Y_i (connection j can only exist if target component i is included)
            row = np.zeros(total_vars)
            row[e_idx] = 1
            row[N_E + target_idx] = -1
            conn_target_constraints.append(row)
            constraint_desc = f"E_{e_idx} ≤ Y_{target_idx}"
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
        conn_by_slot = {}  # {(target_component, target_key, slot_index): [edge_indices]}
        
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
        for (target_comp, target_key, slot_idx), slot_connections in conn_by_slot.items():
            if (
                len(slot_connections) > 1
            ):  # Only need constraint if multiple potential connections to same slot
                row = np.zeros(total_vars)
                for e_idx in slot_connections:
                    row[e_idx] = 1
                one_input_constraints.append(row)
                edge_vars = [f"E_{idx}" for idx in slot_connections]
                constraint_desc = f"{' + '.join(edge_vars)} ≤ 1 (slot {slot_idx})"
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

        # 5. Add constraint that enforces that modeled nodes are only included in one component
        # Create a mapping from semantic model nodes to components that use them
        node_to_components = {}
        for component, modeled_nodes in self._sim2sem_map.items():
            if (
                component in Y_component_to_idx
            ):  # Make sure component is in our variable list
                component_idx = Y_component_to_idx[component]
                for node in modeled_nodes:
                    if node not in node_to_components:
                        node_to_components[node] = []
                    node_to_components[node].append(component_idx)

        modeled_node_constraints = []
        # For each node that appears in multiple components
        for node, component_indices in node_to_components.items():
            if len(component_indices) > 1:
                # Create a constraint: sum(Y_i for all components containing this node) ≤ 1
                row = np.zeros(total_vars)
                for idx in component_indices:
                    row[N_E + idx] = 1
                modeled_node_constraints.append(row)
                components_str = " + ".join([f"Y_{idx}" for idx in component_indices])
                constraint_desc = f"{components_str} ≤ 1"
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

        # Update the objective function coefficients
        for i in range(N_Y):
            component = Y_idx_to_component[i]
            if component in self._sim2sem_map:
                modeled_nodes = self._sim2sem_map[component]
                node_count = len(modeled_nodes)

                # Net contribution: cost - (benefit × node_count)
                c[N_E + i] = component_selection_cost - (
                    semantic_instance_benefit * node_count
                )

        # All variables are binary
        integrality = np.ones(total_vars)
        bounds = Bounds(lb=0, ub=1)

        LOGGER.info("Problem info")
        LOGGER.add_level()

        # LOGGER("Objective function")
        # LOGGER.add_level()
        # LOGGER(c)
        # LOGGER.remove_level()

        LOGGER.info("Variables")
        LOGGER.add_level()
        for i in range(N_Y):
            component = Y_idx_to_component[i]
            LOGGER.info("Y_%d: %s", i, component.id)
        LOGGER.remove_level()

        LOGGER.info("Constraint info")
        LOGGER.add_level()
        for info in constraint_info:
            LOGGER.info("%s", info)
        LOGGER.remove_level()
        LOGGER.remove_level()

        # Solve the MILP problem
        if not constraints_list:
            # print_problem(problem_info)
            return {"success": False, "message": "No valid constraints"}

        res = milp(
            c=c, constraints=constraints_list, integrality=integrality, bounds=bounds
        )

        LOGGER.info("Solution")
        LOGGER.add_level()

        LOGGER.info("Active components")
        LOGGER.add_level()
        components = []
        for i in range(N_Y):
            if res.x[N_E + i] == 1:
                component = Y_idx_to_component[i]
                components.append(component)
                # if debug:
                #     print(
                #         f"  Y_{i} = 1: ({component.__class__.__name__}){component.id}"
                #     )
                LOGGER.info(
                    "  Y_%d = 1: (%s)%s", i, component.__class__.__name__, component.id
                )
        LOGGER.remove_level()

        # if debug:
        #     print("=== Active connections ===")
        LOGGER.info("Active connections")
        LOGGER.add_level()
        connections = []
        # Collect all active connections and find max length for alignment
        active_conn_strings = []
        for i in range(N_E):
            if res.x[i] == 1:
                connections.append(E_idx_to_conn[i])
                source, target, source_key, target_key, output_port_index, input_port_index = E_idx_to_conn[i]
                left_part = f"  E_{i} = 1: ({source.__class__.__name__}){source.id}.{source_key}"
                right_part = f"({target.__class__.__name__}){target.id}.{target_key}"
                active_conn_strings.append((left_part, right_part))

        # Find max length of left parts
        if active_conn_strings:
            max_left_len = max(len(left) for left, _ in active_conn_strings)
            for left_part, right_part in active_conn_strings:
                LOGGER.info("%s → %s", left_part, right_part)
        LOGGER.remove_level()

        # if debug:
        #     print("=== Inactive components ===")
        LOGGER.info("Inactive components")
        LOGGER.add_level()
        for i in range(N_Y):
            if res.x[N_E + i] == 0:
                component = Y_idx_to_component[i]
                # if debug:
                #     print(
                #         f"  Y_{i} = 0: ({component.__class__.__name__}){component.id}"
                #     )
                LOGGER.info(
                    "  Y_%d = 0: (%s)%s", i, component.__class__.__name__, component.id
                )
        LOGGER.remove_level()

        # if debug:
        #     print("=== Inactive connections ===")
        LOGGER.info("Inactive connections")
        LOGGER.add_level()
        # Collect all inactive connections and find max length for alignment
        inactive_conn_strings = []
        for i in range(N_E):
            if res.x[i] == 0:
                source, target, source_key, target_key, output_port_index, input_port_index = E_idx_to_conn[i]
                left_part = f"  E_{i} = 0: ({source.__class__.__name__}){source.id}.{source_key}"
                right_part = f"({target.__class__.__name__}){target.id}.{target_key}"
                inactive_conn_strings.append((left_part, right_part))

        # Find max length of left parts
        if inactive_conn_strings:
            max_left_len = max(len(left) for left, _ in inactive_conn_strings)
            for left_part, right_part in inactive_conn_strings:
                LOGGER.info("%s<%d> → %s", left_part, max_left_len, right_part)
        LOGGER.remove_level()

        # if debug:
        #     print_problem(problem_info)
        LOGGER.remove_level()

        if res.success:
            return {
                "success": True,
                "message": "Optimization successful",
                "problem_info": constraint_info,
                "connections": connections,
            }
        else:
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

        LOGGER.info("Instantiating components")
        LOGGER.add_level()

        # Component instantiation logic from _connect method
        class_to_instance_map = {}
        self._sim2sem_map = {}
        self._sem2sim_map = {}
        self._sim2group_map = {}  # Maps components to their matched signature patterns and groups
        self.modeled_components = set()
        for i, (component_cls, sps) in enumerate(complete_groups.items()):
            LOGGER.info("Class: %s", component_cls.__name__)
            LOGGER.add_level()
            for sp, groups in sps.items():
                for group in groups:
                    modeled_match_nodes = {
                        group[sp_subject] for sp_subject in sp.modeled_nodes
                    }
                    self.modeled_components.update(modeled_match_nodes)  # Union/add set

                    if len(modeled_match_nodes) == 1:
                        component = next(iter(modeled_match_nodes))
                        id_ = component.get_short_name()
                        base_kwargs = get_predicate_object_pairs(component)
                        extension_kwargs = {"id": id_}
                    else:
                        id_ = ""
                        modeled_match_nodes_sorted = sorted(
                            modeled_match_nodes, key=lambda x: x.uri
                        )
                        for component in modeled_match_nodes_sorted:
                            id_ += "[%s]" % component.get_short_name()
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
                            LOGGER.add_level()
                            LOGGER.info(
                                "Mapping parameters for component: %s", component.id
                            )
                            LOGGER.add_level()
                            # Get all parameters for the component
                            for key, node in sp.parameters.items():
                                if group[node] is not None:
                                    value = group[node]
                                    value = value.uri.value
                                    obj = rgetattr(component, key)
                                    LOGGER.info("%s: %s", key, value)

                                    if isinstance(obj, tps.Parameter):
                                        rsetattr(
                                            component,
                                            key,
                                            tps.Parameter(
                                                torch.tensor(value, dtype=torch.float64),
                                                requires_grad=False,
                                            ),
                                        )
                                    else:
                                        rsetattr(component, key, value)
                            LOGGER.remove_level()

                        # Store matched groups for this signature pattern
                        if component not in self._sim2group_map:
                            self._sim2group_map[component] = {}
                        self._sim2group_map[component][sp] = [group]
                        self._sim2sem_map[component] = modeled_match_nodes
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
            LOGGER.remove_level()
        LOGGER.remove_level()

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
        LOGGER.info("Connecting components")
        LOGGER.add_level()
        # Extract the components that are actually used in connections
        new_E_conn_to_sp_group = {}
        used_components = set()
        for conn in connections:
            source, target, source_key, target_key, output_port_index, input_port_index = conn

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
            source, target, source_key, target_key, output_port_index, input_port_index = conn
            sp_target, groups_target, sp_source, groups_source = self.E_conn_to_sp_group[conn]

            if target not in new_sim2group_map:
                new_sim2group_map[target] = {}
            if sp_target not in new_sim2group_map[target]:
                new_sim2group_map[target][sp_target] = groups_target
            else:
                assert groups_target == new_sim2group_map[target][sp_target], "Groups target do not match"

            if source not in new_sim2group_map:
                new_sim2group_map[source] = {}
            if sp_source not in new_sim2group_map[source]:
                new_sim2group_map[source][sp_source] = groups_source
            else:
                assert groups_source == new_sim2group_map[source][sp_source], "Groups source do not match"

            # new_sim2group_map[source]

                # used_sps.add(sp)
                # used_sps.add(p_sp)

        self._sim2group_map = new_sim2group_map



        for conn in connections:
            source, target, source_key, target_key, output_port_index, input_port_index = conn
            conn_str = f"({source.__class__.__name__}){source.id}.{source_key}[{output_port_index}] → ({target.__class__.__name__}){target.id}.{target_key}[{input_port_index}]"
            LOGGER.info("Adding connection: %s", conn_str)

            # Indices are pre-resolved during MILP setup:
            # - None for scalar ports
            # - int for vector port slots
            sim_model.add_connection(source, target, source_key, target_key, output_port_index, input_port_index)
            



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

    @staticmethod
    def _copy_nodemap(nodemap: Dict[Node, Any]) -> Dict[Node, Any]:
        return {k: v for k, v in nodemap.items()}

    @staticmethod
    def _copy_nodemap_list(nodemap_list: List[Dict[Node, Any]]) -> List[Dict[Node, Any]]:
        return [Translator._copy_nodemap(nodemap) for nodemap in nodemap_list]

    @staticmethod
    def _get_node_string(sp_subject: Optional[Node], sm_subject: Optional[Any]):
        """Format subject node pair for debug logging."""
        ss = sm_subject.get_short_name() if sm_subject is not None else None
        return f"{sp_subject.id}: {ss}"

    @staticmethod
    def _get_map_string(l):
        LOGGER.add_level()
        for sp, sm in l.items():
            ss = sm.get_short_name() if sm is not None else None
            LOGGER.debug("%s: %s", sp.id, ss)
        LOGGER.remove_level()

    @staticmethod
    def _get_maps_string(maps):
        for i,l in enumerate(maps):
            LOGGER.debug("MAP %d:", i)
            LOGGER.add_level()
            for sp, sm in l.items():
                ss = sm.get_short_name() if sm is not None else None
                LOGGER.debug("%s: %s", sp.id, ss)
            LOGGER.remove_level()

    @staticmethod
    def _prune_recursive(sm_subject, sp_subject, candidate_maps, feasible, comparison_table,
        signature_pattern, verbose=False,):
        signature_pattern.reset_ruleset()
        return Translator.__prune_recursive(sm_subject, sp_subject, candidate_maps, feasible, comparison_table,
        signature_pattern, verbose=verbose)

    @staticmethod
    def __prune_recursive(
        sm_subject, sp_subject, candidate_maps, feasible, comparison_table,
        signature_pattern, verbose=False,
    ):
        """
        Recursively match a signature pattern node against a semantic model node (DFS).

        Traverses the subject → predicate → object structure of both graphs simultaneously.
        For each predicate-object pair in sp_subject, checks if sm_subject has matching
        predicates and objects, then recursively matches the object nodes.

        Args:
            sm_subject: Current semantic model subject node being matched
            sp_subject: Current signature pattern subject node being matched
            candidate_maps: List of partial SP→SM mappings being built
            feasible: Tracks which SM nodes are feasible for each SP node
            comparison_table: Tracks which SM nodes have been compared
            signature_pattern: The full signature pattern
            verbose: Print debug info

        Returns:
            (candidate_maps, feasible, comparison_table, is_pruned)
        """
        LOGGER.debug("Entering prune_recursive")
        LOGGER.add_level()
        LOGGER.debug(lambda: Translator._get_node_string(sp_subject, sm_subject))
        
        # Initialize tracking sets for current subject
        feasible.setdefault(sp_subject, set()).add(sm_subject)
        comparison_table.setdefault(sp_subject, set()).add(sm_subject)

        # Get predicate → objects mappings from both subject nodes
        sm_predicate_objects = sm_subject.get_predicate_object_pairs()
        sp_predicate_objects = sp_subject.predicate_object_pairs
        ruleset = signature_pattern.ruleset
        valid_maps = []

        # Process each predicate-object pair required by the SP subject
        for sp_predicate, sp_objects in sp_predicate_objects.items():
            for sp_object in sp_objects:
                rule = ruleset[(sp_subject, sp_predicate, sp_object)] # NOTE: Q: What happens if we have added multiple rules for the same subject, predicate, object? A: This would not be meaningful 

                LOGGER.debug("Rule: %s", rule.__class__.__name__)
                
                # Collect SM objects from ALL predicates in the Predicate (cross-ontology matching)
                sm_objects = []
                for predicate in rule._predicate: # Iterate tuple of Predicate objects
                    for pred in predicate.preds: # Iterate tuple of SemanticPredicate objects
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
                    for maps_for_pair, matched_sm_object, matched_sp_object, matched_type, _ in rule_pairs:
                        LOGGER.debug("Entered inner loop")
                        LOGGER.debug(lambda: Translator._get_node_string(matched_sp_object, matched_sm_object))
                        # Initialize tracking for matched object
                        feasible.setdefault(matched_sp_object, set())
                        comparison_table.setdefault(matched_sp_object, set())

                        if matched_sm_object not in comparison_table[matched_sp_object]:
                            # New comparison - recurse (object becomes subject in next level)
                            comparison_table[matched_sp_object].add(matched_sm_object)
                            child_maps, feasible, comparison_table, is_pruned = (
                                Translator.__prune_recursive(
                                    matched_sm_object, matched_sp_object, maps_for_pair,
                                    feasible, comparison_table, signature_pattern, verbose
                                )
                            )

                            if not is_pruned:
                                # Early stop for SinglePath/MultiPath with Exact match
                                if (isinstance(rule, (UniPathRule, MultiPathRule))
                                        and rule.stop_early and matched_type == ExactRule):
                                    valid_maps.extend(child_maps)
                                    match_found = True
                                    break

                                if match_found:
                                    warnings.warn(
                                        f'Multiple matches: "{sp_subject.id}" -> "{sm_subject.uri}"'
                                    )
                                valid_maps.extend(child_maps)
                                match_found = True

                        elif matched_sm_object in feasible[matched_sp_object]:
                            # Already matched and feasible - reuse result
                            for m in maps_for_pair:
                                m[matched_sp_object] = matched_sm_object
                            valid_maps.extend(maps_for_pair)
                            match_found = True

                    # Prune if required rule had no match
                    if not match_found and not isinstance(rule, OptionalRule):
                        feasible[sp_subject].discard(sm_subject)
                        LOGGER.debug("PRUNED (no match found)")
                        LOGGER.debug(lambda: Translator._get_node_string(sp_subject, sm_subject))
                        LOGGER.remove_level()
                        return candidate_maps, feasible, comparison_table, True

                    candidate_maps = valid_maps

                else:
                    # No predicates matched - prune if rule is required
                    if not isinstance(rule, OptionalRule):
                        feasible[sp_subject].discard(sm_subject)
                        LOGGER.debug("PRUNED (missing predicate): %s", sp_predicate)
                        LOGGER.debug(lambda: Translator._get_node_string(sp_subject, sm_subject))
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
    def _check_edge_connectivity(source_mapping: Dict[Node, Any], target_mapping: Dict[Node, Any], reverse: bool = False) -> bool:
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
            sm_predicates = sm_node.get_predicate_object_pairs()

            for predicate, sp_objects in sp_node.predicate_object_pairs.items():
                sm_children = sm_predicates.get(predicate, [])
                if not sm_children:
                    continue

                for sp_object in sp_objects:
                    target_sm_node = target_mapping.get(sp_object)
                    if target_sm_node is None:
                        continue

                    # Check connectivity based on direction
                    if reverse:
                        if sm_children == target_sm_node:
                            return True
                    else:
                        if target_sm_node in sm_children:
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
                sm_node, sp_node, [],
                feasible, comparison_table, signature_pattern,
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
        
        LOGGER.debug("Compatibility check: %s", 'PASS' if is_compatible else 'FAIL')
        
        # Early exit if incompatible - no merge possible
        if not is_compatible:
            LOGGER.remove_level()
            return False

        # Extract non-None mappings
        nodes_a = {k: v for k, v in group_a.items() if v is not None}
        nodes_b = {k: v for k, v in group_b.items() if v is not None}

        LOGGER.debug("Group B contributes %d new mappings", new_contributions)
        if new_contributions == 0:
            LOGGER.debug("No new contributions, skipping merge")
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
        LOGGER.debug("Edge connectivity - B->A: %s, A->B: %s", has_edge_b_to_a, has_edge_a_to_b)

        # Track whether we used a connected or disconnected merge strategy
        # and which groups contain only isolated nodes (can be reused for other merges)
        used_disconnected_merge = False
        groups_to_preserve = []  # Will contain groups with only isolated nodes
        
        if has_edge_b_to_a or has_edge_a_to_b:
            LOGGER.debug("Attempting connected merge validation")
            if Translator._validate_merge(group_a, nodes_b, signature_pattern):
                merged_group = {**group_a, **nodes_b}  # group_a is base, add nodes_b
                LOGGER.debug("Connected merge successful")
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
            modeled_nodes = signature_pattern.modeled_nodes
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
                    sm_node, sp_node, [],
                    feasible, comparison_table, signature_pattern,
                )

                if is_pruned:
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
                    LOGGER.debug("Added preserved group to incomplete_matches for reuse")


            is_complete = all(
                merged_group.get(n) is not None
                for n in signature_pattern.required_nodes
            )
            LOGGER.debug("Merged group is %s", 'COMPLETE' if is_complete else 'INCOMPLETE')

            if is_complete:
                complete_matches.append(merged_group)
                LOGGER.debug("Added to complete matches")
            else:
                incomplete_matches.append(merged_group)
                LOGGER.debug("Added to incomplete matches")

            LOGGER.remove_level()
            return True

        LOGGER.remove_level()
        return False


class Node:
    node_instance_count = count()

    def __init__(self, cls: Union[Any, Tuple[Any, ...], List[Any], str], graph_name: Optional[str] = None, hash_: Optional[Any] = None) -> None:
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
        """Get the semantic model associated with this node"""
        return self.signature_pattern.semantic_model

    def __str__(self):
        return self.id

    def __repr__(self):
        return f"Node({str(self.id)})" # NOTE: 

    def validate_cls(self):
        if self._signature_pattern is None:
            raise ValueError("No signature pattern set.")

        cls = self.cls
        if isinstance(cls, tuple) == False:
            cls = (cls,)

        cls_ = []
        for c in cls:
            if isinstance(c, core.SemanticType):
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
            if hasattr(s, "uri"):
                uri_str = str(s.uri)
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
            attr.update(c.get_type_attributes())
        return attr


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

    def __init__(self, preds: Union[Any, Tuple[Any, ...], List[Any], str], hash_: Optional[Any] = None) -> None:
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
                preds_.append(core.SemanticPredicate(p, self.signature_pattern.semantic_model))
            elif isinstance(p, str):
                preds_.append(
                    core.SemanticPredicate(URIRef(p), self.signature_pattern.semantic_model)
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
        self._inputs = {}
        self._modeled_nodes = []
        self._ruleset = {}
        self._rules = []
        self._parameters = {}
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

    def add_triple(self, rule):
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
            
        
            assert subj is not None and obj is not None and pred is not None, \
                "Rule must have subject, object, and predicate"
            
            assert isinstance(subj, Node) and isinstance(
                obj, Node
            ), '"subject" and "object" must be instances of class Node'
            
            assert isinstance(pred, Predicate), \
                '"predicate" must be an instance of class Predicate'

            self._add_node(subj, rule_)
            self._add_node(obj, rule_)

            # if any(isinstance(r, NoExactRule) for r in rule.sub_rules):
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
                self.semantic_model.instance_graph.add(
                    (subject_instance, core.namespace.RDF.type, cls_.uri)
                )
            for cls_ in obj.cls:
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


        
    def add_connection(self, 
                    sender_node: Node, 
                    output_port: Union[str, Tuple[str]], 
                    input_port: str, 
                    output_port_index: Union[int, Tuple[int], Tuple[Node], Tuple[torch.Tensor]] = None,
                    input_port_index: Union[int, Node, torch.Tensor] = None):
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
        self._add_node(sender_node)
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

        self._inputs[input_port] = (sender_node, output_port, output_port_index, input_port_index)

    


    def add_input(self, key, node, source_keys=None): # NOTE: Deprecated
        """
        Deprecated method. Use add_connection() instead.

        This method is deprecated and will be removed in a future version.
        Please use add_connection() method instead.
        """
        warnings.warn(
            "add_input() is deprecated and will be removed in a future version. "
            "Use add_connection() instead.",
            DeprecationWarning,
            stacklevel=2
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
            sender_node=sender_node,
            output_port=output_port,
            input_port=input_port
        )

    def _add_node(self, node, rule=None, optional=False):
        if isinstance(rule, NoExactRule)==False:
            if node not in self._nodes:
                self._nodes.append(node)
                node.set_signature_pattern(self)
                node.validate_cls()

            if isinstance(rule, OptionalRule) == False and optional == False:
                if node not in self._required_nodes:
                    self._required_nodes.append(node)
        else:
            if node not in self._nodes:
                node.set_signature_pattern(self)
                node.validate_cls()

    def _remove_node(self, node):
        if node in self._nodes:
            self._nodes.remove(node)

        

    def add_parameter(self, key, node):
        self._add_node(node, optional=True)
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
        self._add_node(node)
        if node not in self._modeled_nodes:
            self._modeled_nodes.append(node)

    def remove_modeled_node(self, node):
        if node in self._modeled_nodes:
            self._modeled_nodes.remove(node)

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

    def __init__(self, subject: Union[Node, Tuple[Node, ...]], object: Union[Node, Tuple[Node, ...]], predicate: Union[Predicate, Tuple[Predicate, ...]]) -> None:
        """
        Initialize a Rule.
        
        Args:
            subject: The source Node
            object: The target Node
            predicate: A Predicate object, or value(s) to wrap in a Predicate
        """
        self.rules = (self, )
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

        assert len(self._subject) == len(self._object) == len(self._predicate), "The number of subjects, objects, and predicates must be the same."


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
        master_rule: Optional[Rule] = None
    ) -> Tuple[List[Tuple[Optional[List], core.SemanticObject, Node, type]], bool, Dict[Tuple[Node, Optional[Predicate], Node], Rule]]:
        LOGGER.debug(f"Applying {self.__class__.__name__}")
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
                LOGGER.debug("MATCHED: %s (%s) IS (%s)", pair[1].get_short_name(), pair[1].get_most_specific_type().get_short_name(), c)
            LOGGER.debug("Rule applies: %s", True)
            LOGGER.remove_level()
            return pairs, True, mask, ruleset_a


        
        LOGGER.debug("Rule applies: %s", False)
        LOGGER.remove_level()
        return [], False, np.array([False]*len(sm_objects)), ruleset

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
        master_rule: Optional[Rule] = None
    ) -> Tuple[List[Tuple[Optional[List], core.SemanticObject, Node, type]], bool, Dict[Tuple[Node, Optional[Predicate], Node], Rule]]:
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
                LOGGER.debug("MATCHED: %s (%s) IS (%s)", pair[1].get_short_name(), pair[1].get_most_specific_type().get_short_name(), c)
            LOGGER.debug("Rule applies: %s", True)
            LOGGER.remove_level()
            return pairs, True, rule_applies_a_vec | rule_applies_b_vec, ruleset_a

        elif rule_applies_a:
            for pair in pairs_a:
                LOGGER.debug("MATCHED: %s (%s) IS (%s)", pair[1].get_short_name(), pair[1].get_most_specific_type().get_short_name(), c)
            LOGGER.debug("Rule applies: %s", True)
            LOGGER.remove_level()
            return pairs_a, True, rule_applies_a_vec, ruleset_a
            
        elif rule_applies_b:
            for pair in pairs_b:
                LOGGER.debug("MATCHED: %s (%s) IS (%s)", pair[1].get_short_name(), pair[1].get_most_specific_type().get_short_name(), c)
            LOGGER.debug("Rule applies: %s", True)
            LOGGER.remove_level()
            return pairs_b, True, rule_applies_b_vec, ruleset_b

        LOGGER.debug("Rule applies: %s", False)
        LOGGER.remove_level()
        return [], False, np.array([False]*len(sm_objects)), ruleset

    def reset(self):
        self.rule_a.reset()
        self.rule_b.reset()



class NoExactRule(Rule):
    r"""

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
        master_rule: Optional[Rule] = None
    ) -> Tuple[List[Tuple[Optional[List], core.SemanticObject, Node, type]], bool, Dict[Tuple[Node, Optional[Predicate], Node], Rule]]:
        """
        
        """
        LOGGER.debug(f"Applying {self.__class__.__name__}")
        LOGGER.add_level()
        if master_rule is None:
            master_rule = self
        pairs = []
        rule_applies_vec = np.array([False]*len(sm_objects))

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
                    sm_object.isinstance(self.object.cls)==False
                    and sm_subject not in excluded_sm_subjects
                    and sm_object not in excluded_sm_objects
                ):
                    # pairs.append((maps_for_match, sm_object, self.object, NoExactRule, i))
                    rule_applies_vec[i] = True
        rule_applies = np.any(rule_applies_vec)
        for pair in pairs:
            LOGGER.debug("MATCHED: %s (%s) IS %s", pair[1].get_short_name(), pair[1].get_most_specific_type().get_short_name(), self.object.cls)
        LOGGER.debug("Rule applies: %s", rule_applies)
        LOGGER.remove_level()
        
        return pairs, rule_applies, rule_applies_vec, ruleset

    def reset(self):
        pass

class ExactRule(Rule):
    r"""
    Rule that requires exact matches between pattern and semantic model elements.

    The Exact rule is the most restrictive rule type, requiring that the semantic model
    contains exactly the same relationship as specified in the signature pattern. This rule
    is used when you need precise control over the pattern matching process.

    Behavior
    --------
    - Requires that the semantic model contains the exact relationship specified
    - No traversal or flexibility in matching
    - Used for critical relationships that must be present exactly as specified

    Examples
    --------
    Controller-property relationships (from PID controller system):

    >>> # Define controller nodes using real ontology classes
    >>> controller_node = Node(cls=core.namespace.S4BLDG.SetpointController)
    >>> sensor_node = Node(cls=core.namespace.SAREF.Sensor)
    >>> property_node = Node(cls=core.namespace.SAREF.Property)
    >>>
    >>> # Define exact relationships for precise control logic
    >>> controller_observes = Exact(
    ...     subject=controller_node,
    ...     object=property_node,
    ...     predicate=core.namespace.SAREF.observes
    ... )
    >>> sensor_observes = Exact(
    ...     subject=sensor_node,
    ...     object=property_node,
    ...     predicate=core.namespace.SAREF.observes
    ... )

    Damper control relationships (from damper system):

    >>> # Define damper control nodes
    >>> damper_node = Node(cls=core.namespace.S4BLDG.Damper)
    >>> controller_node = Node(cls=core.namespace.S4BLDG.Controller)
    >>> position_node = Node(cls=core.namespace.SAREF.OpeningPosition)
    >>>
    >>> # Controller must directly control the opening position
    >>> control_relationship = Exact(
    ...     subject=controller_node,
    ...     object=position_node,
    ...     predicate=core.namespace.SAREF.controls
    ... )
    >>>
    >>> # Position must be property of the damper
    >>> property_relationship = Exact(
    ...     subject=position_node,
    ...     object=damper_node,
    ...     predicate=core.namespace.SAREF.isPropertyOf
    ... )

    Building space topology (from building space system):

    >>> # Define building space nodes
    >>> supply_damper = Node(cls=core.namespace.S4BLDG.Damper)
    >>> return_damper = Node(cls=core.namespace.S4BLDG.Damper)
    >>> building_space = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    >>> space_heater = Node(cls=core.namespace.S4BLDG.SpaceHeater)
    >>>
    >>> # Exact fluid supply relationships
    >>> supply_relationship = Exact(
    ...     subject=supply_damper,
    ...     object=building_space,
    ...     predicate=core.namespace.FSO.suppliesFluidTo
    ... )
    >>> return_relationship = Exact(
    ...     subject=return_damper,
    ...     object=building_space,
    ...     predicate=core.namespace.FSO.hasFluidReturnedBy
    ... )
    >>>
    >>> # Space heater containment
    >>> containment_relationship = Exact(
    ...     subject=space_heater,
    ...     object=building_space,
    ...     predicate=core.namespace.S4BLDG.isContainedIn
    ... )

    BRICK ontology relationships (from BRICK damper system):

    >>> # Define BRICK nodes
    >>> damper_node = Node(cls=core.namespace.BRICK.Damper)
    >>> position_setpoint = Node(cls=core.namespace.BRICK.Damper_Position_Setpoint)
    >>> position_sensor = Node(cls=core.namespace.BRICK.Damper_Position_Sensor)
    >>> flow_sensor = Node(cls=core.namespace.BRICK.Air_Flow_Sensor)
    >>>
    >>> # BRICK-specific exact relationships
    >>> setpoint_relationship = Exact(
    ...     subject=position_setpoint,
    ...     object=damper_node,
    ...     predicate=core.namespace.BRICK.isPointOf
    ... )
    >>> sensor_relationship = Exact(
    ...     subject=position_sensor,
    ...     object=damper_node,
    ...     predicate=core.namespace.BRICK.isPointOf
    ... )
    >>> flow_relationship = Exact(
    ...     subject=flow_sensor,
    ...     object=damper_node,
    ...     predicate=core.namespace.BRICK.isPointOf
    ... )

    Sensor-property relationships (from sensor system):

    >>> # Define sensor nodes
    >>> sensor_node = Node(cls=core.namespace.SAREF.Sensor)
    >>> temperature_node = Node(cls=core.namespace.SAREF.Temperature)
    >>> space_node = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    >>>
    >>> # Sensor must observe the temperature property
    >>> sensor_observes = Exact(
    ...     subject=sensor_node,
    ...     object=temperature_node,
    ...     predicate=core.namespace.SAREF.observes
    ... )
    >>>
    >>> # Temperature must be property of the space
    >>> temperature_property = Exact(
    ...     subject=temperature_node,
    ...     object=space_node,
    ...     predicate=core.namespace.SAREF.isPropertyOf
    ... )
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
        master_rule: Optional[Rule] = None
    ) -> Tuple[List[Tuple[Optional[List], core.SemanticObject, Node, type]], bool, Dict[Tuple[Node, Optional[Predicate], Node], Rule]]:
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
        rule_applies_vec = np.array([False]*len(sm_objects))

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
                    pairs.append((maps_for_match, sm_object, self.object, ExactRule, i))
                    rule_applies_vec[i] = True

        rule_applies = np.any(rule_applies_vec)
        for pair in pairs:
            LOGGER.debug("MATCHED: %s (%s) IS %s", pair[1].get_short_name(), pair[1].get_most_specific_type().get_short_name(), self.object.cls)
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
        master_rule: Optional[Rule] = None
    ) -> Tuple[List[Tuple[Optional[List], core.SemanticObject, Node, type]], bool, Dict[Tuple[Node, Optional[Predicate], Node], Rule]]:
        """
        Apply SinglePath traversal rule.

        Creates intermediate SP subject nodes for path traversal. On first entry,
        accepts all SM objects. On subsequent entries, only accepts SM objects
        that have exactly one child for the predicate (single path constraint).
        """
        LOGGER.debug(f"Applying {self.__class__.__name__}")
        LOGGER.add_level()
        if master_rule is None:
            master_rule = self
        pairs = []
        matched_sm_objects = []
        rule_applies_vec = np.array([False]*len(sm_objects))

        if self.first_entry:
            self.first_entry = False
            matched_sm_objects.extend([(i,sm_object) for i, sm_object in enumerate(sm_objects)])
            rule_applies_vec = np.array([True]*len(sm_objects))
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
                            matched_sm_objects.append((i,sm_object))
                            rule_applies_vec[i] = True
                            break  # Found a valid predicate, no need to check others
        rule_applies = np.any(rule_applies_vec)
        if rule_applies:
            for i, sm_object in matched_sm_objects:
                # Create intermediate SP subject node for continued traversal
                # Hash ensures uniqueness based on context (use predicate's preds tuple)
                intermediate_sp_subject = Node(
                    cls=sm_object.get_most_specific_type(),
                    hash_=(sm_object, self.subject, self.predicate.preds, self.object),
                )
                intermediate_sp_subject.set_signature_pattern(self.object.signature_pattern)
                intermediate_sp_subject.validate_cls()
                intermediate_sp_subject.predicate_object_pairs[self.predicate] = [self.object]
                ruleset[(intermediate_sp_subject, self.predicate, self.object)] = master_rule
                pairs.append((candidate_maps, sm_object, intermediate_sp_subject, _SinglePath, i))

        for pair in pairs:
            LOGGER.debug("MATCHED: %s (%s) IS %s", pair[1].get_short_name(), pair[1].get_most_specific_type().get_short_name(), self.object.cls)
        LOGGER.debug("Rule applies: %s", rule_applies)
        LOGGER.remove_level()
        return pairs, rule_applies, rule_applies_vec, ruleset

    def reset(self):
        self.first_entry = True


class UniPathRule(Rule):
    r"""
    Rule that allows traversal along a single path in the semantic model.

    The SinglePath rule is more flexible than Exact, allowing the pattern matcher to traverse
    through intermediate nodes in the semantic model to find a path between the subject and object.
    This is useful when the semantic model has additional intermediate elements that aren't
    part of the core pattern.

    Priority: 1 (lower priority than Exact)

    Behavior
    --------
    - Allows traversal through intermediate nodes in the semantic model
    - Finds a single path between subject and object
    - More flexible than Exact but still constrained to one path
    - Can stop early if stop_early=True (default)

    Examples
    --------
    Building space equipment connections (from building space system):

    >>> # Define building space nodes using real ontology classes
    >>> supply_damper = Node(cls=core.namespace.S4BLDG.Damper)
    >>> supply_equipment = Node(cls=(
    ...     core.namespace.S4BLDG.Coil,
    ...     core.namespace.S4BLDG.AirToAirHeatRecovery,
    ...     core.namespace.S4BLDG.Fan,
    ... ))
    >>>
    >>> # SinglePath allows flexible connection from damper to upstream equipment
    >>> # This can traverse through intermediate components like ducts or junctions
    >>> equipment_connection = SinglePath(
    ...     subject=supply_damper,
    ...     object=supply_equipment,
    ...     predicate=core.namespace.FSO.hasFluidSuppliedBy
    ... )
    >>>
    >>> # This will match even if the semantic model has:
    >>> # damper -> duct_section -> coil
    >>> # damper -> junction -> fan
    >>> # damper -> heat_recovery_unit

    BRICK ontology flexible connections (from BRICK building space system):

    >>> # Define BRICK nodes
    >>> vav_node = Node(cls=core.namespace.BRICK.VAV)  # Variable Air Volume unit
    >>> ahu_node = Node(cls=core.namespace.BRICK.AHU)  # Air Handling Unit
    >>>
    >>> # SinglePath allows traversal through BRICK equipment hierarchy
    >>> ahu_connection = SinglePath(
    ...     subject=vav_node,
    ...     object=ahu_node,
    ...     predicate=core.namespace.BRICK.isFedBy
    ... )
    >>>
    >>> # This can match complex BRICK hierarchies:
    >>> # VAV -> Terminal_Unit -> Zone_Equipment -> AHU
    >>> # VAV -> Duct_System -> AHU

    Sensor connections after equipment (from sensor system):

    >>> # Define sensor nodes for temperature measurement after coil
    >>> sensor_node = Node(cls=core.namespace.SAREF.Sensor)
    >>> temperature_node = Node(cls=core.namespace.SAREF.Temperature)
    >>> coil_air_side = Node(cls=core.namespace.S4BLDG.Coil)
    >>> system_after = Node(cls=core.namespace.S4SYST.System)
    >>>
    >>> # Exact relationship for sensor observation
    >>> sensor_observes = Exact(
    ...     subject=sensor_node,
    ...     object=temperature_node,
    ...     predicate=core.namespace.SAREF.observes
    ... )
    >>>
    >>> # SinglePath allows flexible connection from coil to sensor location
    >>> coil_to_sensor = SinglePath(
    ...     subject=coil_air_side,
    ...     object=sensor_node,
    ...     predicate=core.namespace.FSO.suppliesFluidTo
    ... )
    >>>
    >>> # This matches various sensor placements:
    >>> # coil -> duct_section -> sensor
    >>> # coil -> mixing_box -> sensor
    >>> # coil -> damper -> sensor

    Multi-type node connections (common pattern):

    >>> # Node that can match multiple equipment types
    >>> equipment_node = Node(cls=(
    ...     core.namespace.S4BLDG.Pump,
    ...     core.namespace.S4BLDG.Fan,
    ...     core.namespace.S4BLDG.Compressor
    ... ))
    >>> pipe_or_duct = Node(cls=(
    ...     core.namespace.S4BLDG.Pipe,
    ...     core.namespace.S4BLDG.Duct
    ... ))
    >>>
    >>> # SinglePath for flexible fluid/air distribution
    >>> distribution_path = SinglePath(
    ...     subject=equipment_node,
    ...     object=pipe_or_duct,
    ...     predicate=core.namespace.FSO.suppliesFluidTo
    ... )
    >>>
    >>> # This allows matching:
    >>> # pump -> valve -> pipe
    >>> # fan -> damper -> duct
    >>> # compressor -> expansion_valve -> pipe

    Flexible system topology traversal:

    >>> # Building space connections with intermediate zones
    >>> building_space1 = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    >>> building_space2 = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    >>>
    >>> # SinglePath for adjacent zone connections through shared systems
    >>> adjacent_connection = SinglePath(
    ...     subject=building_space1,
    ...     object=building_space2,
    ...     predicate=core.namespace.S4SYST.connectedTo
    ... )
    >>>
    >>> # This can traverse:
    >>> # space1 -> shared_duct_system -> space2
    >>> # space1 -> common_equipment -> space2
    >>> # space1 -> thermal_bridge -> space2
    """
    def __init__(self, stop_early=True, **kwargs):
        # Normalize predicate to Predicate class (similar to how we normalize cls to tuple in Node)
        super().__init__(**kwargs)
        if kwargs.get('predicate') is None:
            predicate = None
        elif isinstance(kwargs.get('predicate'), Predicate):
            predicate = kwargs.get('predicate')
        else:
            # Deprecation warning
            warnings.warn("The 'predicate' argument is deprecated. Use the 'Predicate' class instead.", DeprecationWarning, stacklevel=2)

            # Auto-wrap in Predicate (handles single value or tuple)
            predicate = Predicate(kwargs.get('predicate'))
        kwargs['predicate'] = predicate
        self.rule = ExactRule(**kwargs) | _SinglePath(**kwargs)  # This order
        self.stop_early = stop_early
        # super().__init__(**kwargs)

    def apply(
        self,
        sm_subject: core.SemanticObject,
        sm_objects: List[core.SemanticObject],
        ruleset: Dict[Tuple[Node, Optional[Predicate], Node], Rule],
        candidate_maps: Optional[List[Optional[Any]]] = None,
        master_rule: Optional[Rule] = None
    ) -> Tuple[List[Tuple[Optional[List], core.SemanticObject, Node, type]], bool, Dict[Tuple[Node, Optional[Predicate], Node], Rule]]:
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
        master_rule: Optional[Rule] = None
    ) -> Tuple[List[Tuple[Optional[List], core.SemanticObject, Node, type]], bool, Dict[Tuple[Node, Optional[Predicate], Node], Rule]]:
        """
        Apply MultiPath traversal rule.

        Creates intermediate SP subject nodes for path traversal. On first entry,
        accepts all SM objects. On subsequent entries, accepts SM objects that
        have at least one child for the predicate (multi-path allows branching).
        """
        LOGGER.debug(f"Applying {self.__class__.__name__}")
        LOGGER.add_level()
        if master_rule is None:
            master_rule = self
        pairs = []
        matched_sm_objects = []
        rule_applies_vec = np.array([False]*len(sm_objects))

        if self.first_entry:
            self.first_entry = False
            matched_sm_objects.extend([(i,sm_object) for i, sm_object in enumerate(sm_objects)])
            rule_applies_vec = np.array([True]*len(sm_objects))
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
                            matched_sm_objects.append((i,sm_object))
                            rule_applies_vec[i] = True
                            break  # Found a valid predicate, no need to check others

        rule_applies = np.any(rule_applies_vec)
        if rule_applies:
            
            for i, sm_object in matched_sm_objects:
                # Create intermediate SP subject node for continued traversal
                intermediate_sp_subject = Node(cls=sm_object.get_most_specific_type())
                intermediate_sp_subject.set_signature_pattern(self.object.signature_pattern)
                intermediate_sp_subject.validate_cls()
                intermediate_sp_subject.predicate_object_pairs[self.predicate] = [self.object]
                ruleset[(intermediate_sp_subject, self.predicate, self.object)] = master_rule
                pairs.append((candidate_maps, sm_object, intermediate_sp_subject, _MultiPath, i))

        for pair in pairs:
            LOGGER.debug("MATCHED: %s (%s) IS %s", pair[1].get_short_name(), pair[1].get_most_specific_type().get_short_name(), self.object.cls)
        LOGGER.debug("Rule applies: %s", rule_applies)
        LOGGER.remove_level()
        return pairs, rule_applies, rule_applies_vec, ruleset

    def reset(self):
        self.first_entry = True


class OptionalRule(Rule):
    r"""
    Rule that makes pattern elements optional (may or may not be present).

    The Optional_ rule allows signature patterns to include elements that may or may not be
    present in the semantic model. This is useful for creating flexible patterns that can
    match a variety of system configurations.

    Priority: 1 (lowest priority)

    Behavior
    --------
    - Makes the relationship optional - it may or may not exist in the semantic model
    - If the relationship exists, it must match the specified pattern
    - If the relationship doesn't exist, the pattern can still match
    - Used to create flexible patterns that accommodate variations in system configurations

    Examples
    --------
    Optional damper parameters (from damper system):

    >>> # Define damper nodes using real ontology classes
    >>> damper_node = Node(cls=core.namespace.S4BLDG.Damper)
    >>> property_value = Node(cls=core.namespace.SAREF.PropertyValue)
    >>> float_value = Node(cls=core.namespace.XSD.float)
    >>> flow_rate_node = Node(cls=core.namespace.S4BLDG.NominalAirFlowRate)
    >>>
    >>> # Optional parameter relationships - damper may have nominal flow rate
    >>> optional_value = Optional_(
    ...     subject=property_value,
    ...     object=float_value,
    ...     predicate=core.namespace.SAREF.hasValue
    ... )
    >>> optional_property = Optional_(
    ...     subject=property_value,
    ...     object=flow_rate_node,
    ...     predicate=core.namespace.SAREF.isValueOfProperty
    ... )
    >>> optional_damper_param = Optional_(
    ...     subject=damper_node,
    ...     object=property_value,
    ...     predicate=core.namespace.SAREF.hasPropertyValue
    ... )
    >>>
    >>> # Pattern matches whether or not flow rate is specified:
    >>> # - If flow rate exists: must match the pattern structure
    >>> # - If flow rate doesn't exist: pattern still matches

    Optional BRICK values (from BRICK damper system):

    >>> # Define BRICK nodes
    >>> flow_setpoint = Node(cls=core.namespace.BRICK.Air_Flow_Setpoint)
    >>> float_value = Node(cls=core.namespace.XSD.float)
    >>>
    >>> # Optional BRICK value - flow setpoint may have a numeric value
    >>> optional_brick_value = Optional_(
    ...     subject=flow_setpoint,
    ...     object=float_value,
    ...     predicate=core.namespace.BRICK.hasValue
    ... )
    >>>
    >>> # This allows the pattern to match BRICK models with or without
    >>> # explicit setpoint values configured

    Optional building space components (example pattern extension):

    >>> # Define building space nodes
    >>> building_space = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    >>> heat_recovery = Node(cls=core.namespace.S4BLDG.AirToAirHeatRecovery)
    >>> humidity_sensor = Node(cls=core.namespace.SAREF.Sensor)
    >>> humidity_property = Node(cls=core.namespace.SAREF.Humidity)
    >>>
    >>> # Optional heat recovery system
    >>> optional_heat_recovery = Optional_(
    ...     subject=building_space,
    ...     object=heat_recovery,
    ...     predicate=core.namespace.S4BLDG.contains
    ... )
    >>>
    >>> # Optional humidity monitoring
    >>> optional_humidity_sensor = Optional_(
    ...     subject=humidity_sensor,
    ...     object=humidity_property,
    ...     predicate=core.namespace.SAREF.observes
    ... )
    >>> optional_humidity_in_space = Optional_(
    ...     subject=humidity_property,
    ...     object=building_space,
    ...     predicate=core.namespace.SAREF.isPropertyOf
    ... )
    >>>
    >>> # Pattern works for various building space configurations:
    >>> # - Basic space without heat recovery or humidity sensing
    >>> # - Space with heat recovery but no humidity sensing
    >>> # - Space with humidity sensing but no heat recovery
    >>> # - Fully equipped space with both features

    Optional controller parameters (common in control systems):

    >>> # Define controller nodes
    >>> controller_node = Node(cls=core.namespace.S4BLDG.SetpointController)
    >>> deadband_node = Node(cls=core.namespace.S4BLDG.Deadband)
    >>> gain_node = Node(cls=core.namespace.S4BLDG.ProportionalGain)
    >>> integral_time = Node(cls=core.namespace.S4BLDG.IntegralTime)
    >>>
    >>> # Optional controller tuning parameters
    >>> optional_deadband = Optional_(
    ...     subject=controller_node,
    ...     object=deadband_node,
    ...     predicate=core.namespace.SAREF.hasProperty
    ... )
    >>> optional_gain = Optional_(
    ...     subject=controller_node,
    ...     object=gain_node,
    ...     predicate=core.namespace.SAREF.hasProperty
    ... )
    >>> optional_integral = Optional_(
    ...     subject=controller_node,
    ...     object=integral_time,
    ...     predicate=core.namespace.SAREF.hasProperty
    ... )
    >>>
    >>> # Controller pattern matches various configurations:
    >>> # - Basic on/off controller (no tuning parameters)
    >>> # - P controller (proportional gain only)
    >>> # - PI controller (proportional + integral)
    >>> # - Full PID controller with deadband

    Flexible sensor configurations (from sensor system patterns):

    >>> # Define sensor nodes for position measurement
    >>> sensor_node = Node(cls=core.namespace.SAREF.Sensor)
    >>> position_node = Node(cls=core.namespace.SAREF.OpeningPosition)
    >>> valve_or_damper = Node(cls=(
    ...     core.namespace.S4BLDG.Valve,
    ...     core.namespace.S4BLDG.Damper,
    ... ))
    >>> controller_node = Node(cls=core.namespace.S4BLDG.Controller)
    >>>
    >>> # Required: sensor observes position
    >>> sensor_observes = Exact(
    ...     subject=sensor_node,
    ...     object=position_node,
    ...     predicate=core.namespace.SAREF.observes
    ... )
    >>>
    >>> # Required: position belongs to valve/damper
    >>> position_property = Exact(
    ...     subject=position_node,
    ...     object=valve_or_damper,
    ...     predicate=core.namespace.SAREF.isPropertyOf
    ... )
    >>>
    >>> # Optional: controller controls the position
    >>> optional_control = Optional_(
    ...     subject=controller_node,
    ...     object=position_node,
    ...     predicate=core.namespace.SAREF.controls
    ... )
    >>>
    >>> # Pattern matches:
    >>> # - Manual valve with position sensor (no controller)
    >>> # - Automated valve with controller and position feedback
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
        master_rule: Optional[Rule] = None
    ) -> Tuple[List[Tuple[Optional[List], core.SemanticObject, Node, type]], bool, Dict[Tuple[Node, Optional[Predicate], Node], Rule]]:
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
        rule_applies_vec = np.array([False]*len(sm_objects))

        for i, sm_object in enumerate(sm_objects):
            if sm_object.isinstance(self.object.cls):
                pairs.append((candidate_maps, sm_object, self.object, OptionalRule, i))
                rule_applies_vec[i] = True

        rule_applies = np.any(rule_applies_vec)
        for pair in pairs:
            LOGGER.debug("MATCHED: %s (%s) IS %s", pair[1].get_short_name(), pair[1].get_most_specific_type().get_short_name(), self.object.cls)
        LOGGER.debug("Rule applies: %s", rule_applies)
        LOGGER.remove_level()
        return pairs, rule_applies, rule_applies_vec, ruleset

    def reset(self):
        pass


class MultiPathRule(Rule):
    r"""
    Rule that allows traversal along multiple paths in the semantic model.

    The MultiPath rule is the most flexible rule type, allowing the pattern matcher to explore
    multiple paths between the subject and object in the semantic model. This is useful when
    there are multiple valid ways to connect components or when the semantic model has complex
    relationship structures.

    Priority: 1 (lower priority than Exact)

    Behavior
    --------
    - Allows traversal through multiple paths in the semantic model
    - Finds all possible paths between subject and object
    - Most flexible rule type
    - Can stop early if stop_early=True (default)

    **Note**: MultiPath rules can cause infinite recursion in some complex semantic models
    and are used sparingly in practice. Consider using SinglePath for most flexible matching needs.

    Examples
    --------
    Complex building space connections (theoretical usage):

    >>> # Define building space nodes using real ontology classes
    >>> building_space1 = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    >>> building_space2 = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    >>>
    >>> # MultiPath for complex adjacent zone relationships
    >>> # Note: This is commented out in real systems due to recursion issues
    >>> # adjacent_connection = MultiPath(
    >>> #     subject=building_space1,
    >>> #     object=building_space2,
    >>> #     predicate=core.namespace.S4SYST.connectedTo
    >>> # )
    >>>
    >>> # This could theoretically match multiple connection types:
    >>> # space1 -> shared_hvac_system -> space2
    >>> # space1 -> structural_connection -> space2
    >>> # space1 -> thermal_bridge -> space2
    >>> # space1 -> common_corridor -> space2

    Equipment network traversal (theoretical usage):

    >>> # Define HVAC equipment nodes
    >>> chiller_node = Node(cls=core.namespace.S4BLDG.Chiller)
    >>> cooling_tower = Node(cls=core.namespace.S4BLDG.CoolingTower)
    >>> heat_exchanger = Node(cls=core.namespace.S4BLDG.HeatExchanger)
    >>>
    >>> # MultiPath for complex chilled water systems with multiple paths
    >>> # cooling_network = MultiPath(
    >>> #     subject=chiller_node,
    >>> #     object=cooling_tower,
    >>> #     predicate=core.namespace.FSO.suppliesFluidTo
    >>> # )
    >>>
    >>> # Could match various cooling system configurations:
    >>> # chiller -> primary_loop -> heat_exchanger -> secondary_loop -> cooling_tower
    >>> # chiller -> bypass_valve -> direct_connection -> cooling_tower
    >>> # chiller -> buffer_tank -> distribution_system -> cooling_tower

    BRICK equipment hierarchies (theoretical usage):

    >>> # Define BRICK nodes for air handling systems
    >>> ahu_node = Node(cls=core.namespace.BRICK.AHU)
    >>> terminal_unit = Node(cls=core.namespace.BRICK.Terminal_Unit)
    >>>
    >>> # MultiPath for complex BRICK hierarchies
    >>> # Note: Use with caution due to potential performance issues
    >>> # brick_hierarchy = MultiPath(
    >>> #     subject=ahu_node,
    >>> #     object=terminal_unit,
    >>> #     predicate=core.namespace.BRICK.feeds
    >>> # )
    >>>
    >>> # Could traverse multiple BRICK relationship paths:
    >>> # AHU -> VAV_Box -> Terminal_Unit
    >>> # AHU -> Duct_System -> Zone_Equipment -> Terminal_Unit
    >>> # AHU -> Distribution_System -> End_Use_Equipment -> Terminal_Unit

    Practical alternatives to MultiPath:

    >>> # Instead of MultiPath, consider using multiple SinglePath rules
    >>> # or combining Optional_ rules for specific known alternatives
    >>>
    >>> # Define equipment nodes
    >>> supply_equipment = Node(cls=(
    ...     core.namespace.S4BLDG.Coil,
    ...     core.namespace.S4BLDG.Fan,
    ...     core.namespace.S4BLDG.HeatExchanger
    ... ))
    >>> distribution_node = Node(cls=(
    ...     core.namespace.S4BLDG.Duct,
    ...     core.namespace.S4BLDG.Pipe
    ... ))
    >>>
    >>> # Primary connection path
    >>> primary_path = SinglePath(
    ...     subject=supply_equipment,
    ...     object=distribution_node,
    ...     predicate=core.namespace.FSO.suppliesFluidTo
    ... )
    >>>
    >>> # Alternative: Use Optional_ for specific alternative connections
    >>> bypass_valve = Node(cls=core.namespace.S4BLDG.Valve)
    >>> optional_bypass = Optional_(
    ...     subject=supply_equipment,
    ...     object=bypass_valve,
    ...     predicate=core.namespace.FSO.suppliesFluidTo
    ... )
    >>>
    >>> # This approach provides controlled flexibility without recursion risks

    **Best Practice**: In most real-world implementations, use SinglePath for flexible
    connections and Optional_ for alternative configurations rather than MultiPath,
    which can cause performance issues in complex semantic models.
    """
    def __init__(self, stop_early=True, **kwargs):
        # Normalize predicate to Predicate class (similar to how we normalize cls to tuple in Node)
        if kwargs.get('predicate') is None:
            predicate = None
        elif isinstance(kwargs.get('predicate'), Predicate):
            predicate = kwargs.get('predicate')
        else:
            # Deprecation warning
            warnings.warn("The 'predicate' argument is deprecated. Use the 'Predicate' class instead.", DeprecationWarning, stacklevel=2)

            # Auto-wrap in Predicate (handles single value or tuple)
            predicate = Predicate(kwargs.get('predicate'))
        kwargs['predicate'] = predicate
        self.rule = ExactRule(**kwargs) | _MultiPath(**kwargs)
        self.stop_early = stop_early
        super().__init__(**kwargs)

    def apply(
        self,
        sm_subject: core.SemanticObject,
        sm_objects: List[core.SemanticObject],
        ruleset: Dict[Tuple[Node, Optional[Predicate], Node], Rule],
        candidate_maps: Optional[List[Optional[Any]]] = None,
        master_rule: Optional[Rule] = None
    ) -> Tuple[List[Tuple[Optional[List], core.SemanticObject, Node, type]], bool, Dict[Tuple[Node, Optional[Predicate], Node], Rule]]:
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


# Backwards compatibility classes with deprecation warnings
class Exact(ExactRule):
    def __init__(self, **kwargs):
        warnings.warn(
            "The 'Exact' class is deprecated. Use 'ExactRule' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(**kwargs)


class SinglePath(UniPathRule):
    def __init__(self, **kwargs):
        warnings.warn(
            "The 'SinglePath' class is deprecated. Use 'UniPathRule' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(**kwargs)


class MultiPath(MultiPathRule):
    def __init__(self, **kwargs):
        warnings.warn(
            "The 'MultiPath' class is deprecated. Use 'MultiPathRule' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(**kwargs)


class Optional_(OptionalRule):
    def __init__(self, **kwargs):
        warnings.warn(
            "The 'Optional_' class is deprecated. Use 'OptionalRule' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(**kwargs)







# if __name__ == "__main__":

#     n1 = Node(cls=core.namespace.S4BLDG.Damper)
#     n2 = Node(cls=core.namespace.S4BLDG.Controller)
#     n3 = Node(cls=core.namespace.S4BLDG.Coil)
#     n4 = Node(cls=core.namespace.S4BLDG.AirToAirHeatRecovery)
#     n5 = Node(cls=core.namespace.S4BLDG.Valve)

#     r1 = NoExactRule(subject=n1, object=n2, predicate=core.namespace.SAREF.controls)

#     r2 = UniPathRule(subject=n3, object=n4, predicate=core.namespace.SAREF.feeds)

#     r3 = ExactRule(subject=n2, object=n5, predicate=core.namespace.SAREF.controls)

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