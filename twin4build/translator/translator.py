from __future__ import annotations

# Standard library imports
import inspect
import warnings
from dataclasses import dataclass
from itertools import count
from typing import Dict, List, Tuple

# Third party imports
import numpy as np
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
from twin4build.utils.print_progress import PRINTPROGRESS
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rsetattr import rsetattr


class Translator:
    r"""
    Class for ontology-driven automated model generation and calibration in building energy systems.

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

    def __init__(self):
        self.sim2sem_map = {}
        self.sem2sim_map = {}
        self.instance_to_group_map = {}

    def translate(
        self, semantic_model: core.SemanticModel, systems_: List[core.System] = None
    ) -> core.SimulationModel:
        """
        Translate semantic model to simulation model using pattern matching

        Args:
            systems: List of system types to match against
            semantic_model: The semantic model to translate

        Returns:
            SimulationModel instance with matched components
        """

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
            # PRINTPROGRESS("Found following matching candidate patterns:")

            for component_cls in complete_groups.keys():
                PRINTPROGRESS(f"Component: {component_cls.__name__}")
                PRINTPROGRESS.add_level()

                for sp in complete_groups[component_cls].keys():
                    PRINTPROGRESS(
                        f"Signature pattern: {sp.id}, {len(complete_groups[component_cls][sp])} matches found"
                    )
                PRINTPROGRESS.remove_level()

        else:
            raise Exception("No matching patterns found.")

        # Create component instances
        self._instantiate_components(complete_groups, semantic_model)

        if len(self.sim2sem_map) == 0:
            raise Exception("No components instantiated.")

        result = self._solve_milp()
        if result["success"]:
            # Initialize simulation model
            dir_conf = semantic_model.dir_conf.copy()
            dir_conf[-1] = "simulation_model"
            sim_model = core.SimulationModel(id="simulation_model", dir_conf=dir_conf)

            # Connect components
            self._connect_components(result["connections"], sim_model)
        else:
            # This can happen if all components require no inputs. In this case, we can just return the simulation model with no connections. But this is probably not wanted behavior - better to raise an exception.
            print(result["message"])
            raise Exception(f"MILP solver failed: {result['message']}")

        return sim_model

    @staticmethod
    def _match_patterns(
        systems_: List[core.System], semantic_model: core.SemanticModel
    ) -> Tuple[Dict, Dict]:
        """
        Match signature patterns against semantic model nodes

        Args:
            semantic_model: The semantic model to match against
            systems: List of system types with signature patterns

        Returns:
            Tuple of (complete_groups, incomplete_groups) dictionaries
        """
        complete_groups = {}
        incomplete_groups = {}

        # Get classes with signature patterns
        classes = [cls for cls in systems_ if hasattr(cls, "sp")]
        for component_cls in classes:
            complete_groups[component_cls] = {}
            incomplete_groups[component_cls] = {}

            for sp in component_cls.sp:
                # print(f"\n=== Starting new signature pattern match for class {component_cls} ===")
                # Initialize groups for this signature pattern
                complete_groups[component_cls][sp] = []
                incomplete_groups[component_cls][sp] = []
                cg = complete_groups[component_cls][sp]
                ig = incomplete_groups[component_cls][sp]
                feasible_map = {}
                comparison_table_map = {}
                for sp_subject in sp.nodes:
                    match_nodes = semantic_model.get_instances_of_type(sp_subject.cls)
                    # print(f"MATCH NODES: {match_nodes}")
                    # print(f"({component_cls}) TESTING SP_SUBJECT: {[s.uri for s in sp_subject.cls]}")
                    for sm_subject in match_nodes:
                        sp_sm_map = {sp_subject: None for sp_subject in sp.nodes}
                        feasible = {sp_subject: set() for sp_subject in sp.nodes}
                        comparison_table = {
                            sp_subject: set() for sp_subject in sp.nodes
                        }
                        sp_sm_map_list = [Translator._copy_nodemap(sp_sm_map)]
                        prune = True
                        if sm_subject not in comparison_table[sp_subject]:
                            sp.reset_ruleset()

                            # print("====================== ENTERING PRUNE RECURSIVE ======================")
                            # id_sp = str([str(s) for s in sp_subject.cls])
                            # id_sp = sp_subject.id
                            # id_sp = id_sp.replace(r"\n", "")
                            # mn = sm_subject.uri if sm_subject is not None else None
                            # id_m = [str(mn)]
                            # print(id_sp, id_m)

                            sp_sm_map_list, feasible, comparison_table, prune = (
                                Translator._prune_recursive(
                                    sm_subject,
                                    sp_subject,
                                    sp_sm_map_list,
                                    feasible,
                                    comparison_table,
                                    sp,
                                    verbose=False,
                                )
                            )

                            for sp_sm_map_ in sp_sm_map_list:
                                feasible_map[id(sp_sm_map_)] = feasible
                                comparison_table_map[id(sp_sm_map_)] = comparison_table

                        # elif sm_subject in feasible[sp_subject]:
                        #     sp_sm_map[sp_subject] = sm_subject
                        #     sp_sm_map_list = [sp_sm_map]
                        #     prune = False

                        if prune == False:
                            # print(f"\nProcessing match for {sp_subject.id}")
                            # print(f"Current sp_sm_map_list length: {len(sp_sm_map_list)}")

                            # We check that the obtained sp_sm_map_list contains node maps with different modeled nodes.
                            # If an SP does not contain a MultiPath rule, we can prune the sp_sm_map_list to only contain node maps with different modeled nodes.
                            modeled_nodes = []
                            for sp_sm_map_ in sp_sm_map_list:
                                node_map_set = set()
                                for sp_modeled_node in sp.modeled_nodes:
                                    node_map_set.add(sp_sm_map_[sp_modeled_node])
                                modeled_nodes.append(node_map_set)

                            node_map_list_new = []
                            for i, (sp_sm_map_, node_map_set) in enumerate(
                                zip(sp_sm_map_list, modeled_nodes)
                            ):
                                active_set = node_map_set
                                passive_set = set().union(
                                    *[v for k, v in enumerate(modeled_nodes) if k != i]
                                )
                                if (
                                    len(active_set.intersection(passive_set)) > 0
                                    and any(
                                        [
                                            isinstance(v, MultiPath)
                                            for v in sp._ruleset.values()
                                        ]
                                    )
                                    == False
                                ):
                                    warnings.warn(
                                        f"Multiple matches found for {sp_subject.id} and {sp_subject.cls}."
                                    )
                                node_map_list_new.append(
                                    sp_sm_map_
                                )  # This constraint has been removed to allow for multiple matches. Note that multiple
                            sp_sm_map_list = node_map_list_new

                            # Cross matching could maybe stop early if a match is found. For SP with multiple allowed matches it might be necessary to check all matches
                            for sp_sm_map_ in sp_sm_map_list:
                                # print("\nCROSS MATCHING AGAINST INCOMPLETE GROUPS: ")
                                # for sp_subject___, sm_subject___ in sp_sm_map_.items():
                                #     id_sp = sp_subject___.id
                                #     id_sp = id_sp.replace(r"\n", "")
                                #     mn = sm_subject___.uri if sm_subject___ is not None else None
                                #     id_m = [str(mn)]
                                #     print(id_sp, id_m)

                                if all(
                                    [
                                        sp_sm_map_[sp_subject] is not None
                                        for sp_subject in sp.required_nodes
                                    ]
                                ):
                                    cg.append(sp_sm_map_)
                                else:
                                    if (
                                        len(ig) == 0
                                    ):  # If there are no groups in the incomplete group list, add the node map
                                        ig.append(sp_sm_map_)
                                    else:
                                        new_ig = ig.copy()
                                        is_match_ = False
                                        for (
                                            group
                                        ) in ig:  # Iterate over incomplete groups
                                            is_match, group, cg, new_ig = (
                                                Translator._match(
                                                    group,
                                                    sp_sm_map_,
                                                    sp,
                                                    cg,
                                                    new_ig,
                                                    feasible_map,
                                                    comparison_table_map,
                                                )
                                            )
                                            if is_match:
                                                is_match_ = True
                                        if is_match_ == False:
                                            new_ig.append(sp_sm_map_)
                                        ig = new_ig

                ig_len = np.inf
                while len(ig) < ig_len:
                    ig_len = len(ig)
                    new_ig = ig.copy()
                    is_match = False
                    for group_i in ig:
                        for group_j in ig:
                            if group_i != group_j:
                                is_match, group, cg, new_ig = Translator._match(
                                    group_i,
                                    group_j,
                                    sp,
                                    cg,
                                    new_ig,
                                    feasible_map,
                                    comparison_table_map,
                                )
                            if is_match:
                                break
                        if is_match:
                            break
                    ig = new_ig

                # # # if True:#component_cls is components.BuildingSpace1AdjBoundaryOutdoorFMUSystem:
                # print("INCOMPLETE GROUPS================================================================================")
                # for group in ig:
                #     print("GROUP------------------------------")
                #     for sp_subject___, sm_subject___ in group.items():
                #         id_sp = sp_subject___.id
                #         id_sp = id_sp.replace(r"\n", "")
                #         mn = sm_subject___.uri if sm_subject___ is not None else None
                #         id_m = [str(mn)]
                #         print(id_sp, id_m)

                # print("COMPLETE GROUPS================================================================================")
                # for group in cg:
                #     print("GROUP------------------------------")
                #     for sp_subject___, sm_subject___ in group.items():
                #         id_sp = sp_subject___.id
                #         id_sp = id_sp.replace(r"\n", "")
                #         mn = sm_subject___.uri if sm_subject___ is not None else None
                #         id_m = [str(mn)]
                #         print(id_sp, id_m)

                new_ig = ig.copy()
                for group in ig:  # Iterate over incomplete groups
                    if all(
                        [
                            group[sp_subject] is not None
                            for sp_subject in sp.required_nodes
                        ]
                    ):  # CHANGED: Check for None instead of empty sets
                        cg.append(group)
                        new_ig.remove(group)
                ig = new_ig

        return complete_groups, incomplete_groups

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

        def update_Y_mappings(component, Y_idx_to_component, Y_component_to_idx, N_Y):
            if component not in Y_component_to_idx:
                Y_idx_to_component[N_Y] = component
                Y_component_to_idx[component] = N_Y
                # print(f"    Added component Y_{N_Y}: {component.id}")
                N_Y += 1
            return Y_idx_to_component, Y_component_to_idx, N_Y

        def update_E_mappings(conn, E_idx_to_conn, E_conn_to_idx, N_E):
            if conn not in E_conn_to_idx:
                E_idx_to_conn[N_E] = conn
                E_conn_to_idx[conn] = N_E
                # print(f"    Added connection E_{N_E}: {conn[0].id}.{conn[2]} -> {conn[1].id}.{conn[3]}")
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

        def print_problem(problem_info):
            print("Problem:")
            for info in problem_info:
                print(info)

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
        # print("\n=== DEBUG: Starting first pass to identify components and connections ===")
        for component, (
            modeled_match_nodes,
            (component_cls, sps),
        ) in self.instance_to_group_map.items():
            # print(f"DEBUG: Processing component: {component.id} (class: {component_cls})")
            # Process each signature pattern for this component
            for sp, groups in sps.items():
                # print(f"  DEBUG: Processing signature pattern: {sp.ownedBy}")
                if component not in required_inputs:
                    required_inputs[component] = {}

                Y_idx_to_component, Y_component_to_idx, N_Y = update_Y_mappings(
                    component, Y_idx_to_component, Y_component_to_idx, N_Y
                )

                # Process required inputs for this component
                for key, (sp_subject, source_keys) in sp.inputs.items():
                    # print(f"    DEBUG: Processing input key: {key}, sp_subject: {sp_subject}")
                    if key not in required_inputs[component]:
                        required_inputs[component][key] = []

                    # Get all potential source nodes for this input
                    match_nodes = {
                        group[sp_subject] for group in groups if sp_subject in group
                    }
                    # print(f"    DEBUG: Match nodes for input {key}: {match_nodes}")

                    # Skip if these nodes aren't in the modeled components
                    # if match_nodes.issubset(self.modeled_components):

                    # Find all potential provider components
                    for sm_subject in match_nodes:

                        if sm_subject in self.sem2sim_map:
                            provider_components = self.sem2sim_map[
                                sm_subject
                            ]  # Get the provider component

                            for provider_component in provider_components:
                                (p_nodes, (p_cls, p_sps)) = self.instance_to_group_map[
                                    provider_component
                                ]  # Find the provider's signature patterns

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
                                        # Add this potential connection
                                        conn = (
                                            provider_component,
                                            component,
                                            source_key,
                                            key,
                                        )
                                        E_idx_to_conn, E_conn_to_idx, N_E = (
                                            update_E_mappings(
                                                conn, E_idx_to_conn, E_conn_to_idx, N_E
                                            )
                                        )
                                        self.E_conn_to_sp_group[conn] = (sp, groups)
                                        if (
                                            provider_component,
                                            source_key,
                                        ) not in required_inputs[component][key]:
                                            required_inputs[component][key].append(
                                                (provider_component, source_key)
                                            )
                                    else:
                                        raise Exception(
                                            "Provider does not have required output. This should not happen."
                                        )

        # Set up the constraints
        total_vars = N_E + N_Y + N_Y
        constraints_list = []
        problem_info = []

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
                    for provider_component, source_key in providers:
                        conn = (provider_component, component, source_key, input_key)
                        edge_idx = E_conn_to_idx[conn]
                        row[edge_idx] = -1  # Negative coefficient for the edge
                        edge_indices.append(edge_idx)

                    if edge_indices:
                        required_input_constraints.append(row)
                        edge_vars = [f"E_{idx}" for idx in edge_indices]
                        constraint_desc = f"Y_{component_idx} ≤ {' + '.join(edge_vars)}"
                        problem_info.append(constraint_desc)

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
        ) in E_idx_to_conn.items():
            source_idx = Y_component_to_idx[source_component]

            # Create constraint: E_j ≤ Y_i (connection j can only exist if source component i is included)
            row = np.zeros(total_vars)
            row[e_idx] = 1
            row[N_E + source_idx] = -1
            conn_source_constraints.append(row)
            constraint_desc = f"E_{e_idx} ≤ Y_{source_idx}"
            problem_info.append(constraint_desc)

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
            source_component,
            target_component,
            source_key,
            target_key,
        ) in E_idx_to_conn.items():
            target_idx = Y_component_to_idx[target_component]

            # Create constraint: E_j ≤ Y_i (connection j can only exist if target component i is included)
            row = np.zeros(total_vars)
            row[e_idx] = 1
            row[N_E + target_idx] = -1
            conn_target_constraints.append(row)
            constraint_desc = f"E_{e_idx} ≤ Y_{target_idx}"
            problem_info.append(constraint_desc)

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

        # 4. One-input constraints: Each input port can receive at most one connection
        # Group connections by target component and target port
        conn_by_target = {}  # {(target_component, target_key): [edge_indices]}

        for e_idx, (
            source_component,
            target_component,
            source_key,
            target_key,
        ) in E_idx_to_conn.items():
            key = (target_component, target_key)
            if key not in conn_by_target:
                conn_by_target[key] = []
            conn_by_target[key].append(e_idx)

        one_input_constraints = []
        for (target_component, target_key), input_connections in conn_by_target.items():
            if (
                len(input_connections) > 1
            ):  # Only need constraint if multiple potential connections
                row = np.zeros(total_vars)
                for e_idx in input_connections:
                    row[e_idx] = 1
                one_input_constraints.append(row)
                edge_vars = [f"E_{idx}" for idx in input_connections]
                constraint_desc = f"{' + '.join(edge_vars)} ≤ 1"
                problem_info.append(constraint_desc)

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
        for component, modeled_nodes in self.sim2sem_map.items():
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
                problem_info.append(constraint_desc)

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

        # Add N_Y binary variables for source nodes (Z variables)
        # N_Z = N_Y

        # # Group connections by target component
        # incoming_connections = {}  # {component_idx: [edge_indices]}
        # for e_idx, (source_component, target_component, source_key, target_key) in E_idx_to_conn.items():
        #     target_idx = Y_component_to_idx[target_component]
        #     if target_idx not in incoming_connections:
        #         incoming_connections[target_idx] = []
        #     incoming_connections[target_idx].append(e_idx)

        # print(f"\n----- Identified incoming connections for {len(incoming_connections)} components -----")

        # # Source node constraints:
        # # Z_i = 1 iff component i is selected (Y_i = 1) AND has no incoming active connections
        # source_node_constraints = []
        # source_node_rhs = []  # right-hand side values

        # print("\n----- Creating source node constraints -----")

        # for y_idx in range(N_Y):
        #     component = Y_idx_to_component[y_idx]
        #     print(f"\nProcessing component Y_{y_idx}: {component.id}")

        #     # Constraint 1: Z_i ≤ Y_i (source node indicator can only be 1 if component is selected)
        #     row1 = np.zeros(total_vars)
        #     row1[N_E + N_Y + y_idx] = 1  # Z_i
        #     row1[N_E + y_idx] = -1       # -Y_i
        #     source_node_constraints.append(row1)
        #     source_node_rhs.append(0)  # Z_i - Y_i ≤ 0
        #     print(f"  Added constraint 1: Z_{y_idx} ≤ Y_{y_idx} (source node only if component is selected)")

        #     # Constraint 2: For components with possible incoming connections
        #     if y_idx in incoming_connections and incoming_connections[y_idx]:
        #         incoming_edges = incoming_connections[y_idx]
        #         print(f"  Component has {len(incoming_edges)} potential incoming connections")

        #         # Y_i - sum(incoming_edges) - Z_i ≤ 0
        #         # This ensures that if Y_i = 1 and sum(incoming_edges) = 0, then Z_i must be 1
        #         row2 = np.zeros(total_vars)
        #         row2[N_E + y_idx] = 1         # Y_i
        #         row2[N_E + N_Y + y_idx] = -1  # -Z_i
        #         for e_idx in incoming_edges:
        #             row2[e_idx] = -1          # -E_j for each incoming edge
        #         source_node_constraints.append(row2)
        #         source_node_rhs.append(0)

        #         edge_str = " + ".join([f"E_{e}" for e in incoming_edges])
        #         print(f"  Added constraint 2a: Y_{y_idx} - ({edge_str}) - Z_{y_idx} ≤ 0")
        #         print(f"    (If component has no incoming connections active, it must be a source node)")

        #         # If any incoming edge is active, Z_i must be 0
        #         for e_idx in incoming_edges:
        #             conn = E_idx_to_conn[e_idx]
        #             print(f"  Processing edge E_{e_idx}: {conn[0].id}.{conn[2]} → {conn[1].id}.{conn[3]}")

        #             row3 = np.zeros(total_vars)
        #             row3[e_idx] = 1                # E_j
        #             row3[N_E + N_Y + y_idx] = 1    # Z_i
        #             source_node_constraints.append(row3)
        #             source_node_rhs.append(1)  # E_j + Z_i ≤ 1

        #             print(f"  Added constraint 2b: E_{e_idx} + Z_{y_idx} ≤ 1")
        #             print(f"    (If this edge is active, component cannot be a source node)")

        #     # Constraint 3: If component has no incoming connections, Z_i = Y_i
        #     else:
        #         print(f"  Component has no potential incoming connections, it's always a source if selected")

        #         # Y_i - Z_i = 0  (combined with Z_i ≤ Y_i from constraint 1, forces Z_i = Y_i)
        #         row2 = np.zeros(total_vars)
        #         row2[N_E + y_idx] = 1         # Y_i
        #         row2[N_E + N_Y + y_idx] = -1  # -Z_i
        #         source_node_constraints.append(row2)
        #         source_node_rhs.append(0)  # Y_i - Z_i ≤ 0 (combined with constraint 1 makes Z_i = Y_i)

        #         print(f"  Added constraint 3: Y_{y_idx} - Z_{y_idx} ≤ 0 (with previous constraint, Z_{y_idx} = Y_{y_idx})")

        # # Convert to numpy array
        # if source_node_constraints:
        #     A_source_node = np.vstack(source_node_constraints)
        #     # These constraints are all of the form ≤ rhs
        #     b_source_node_l = np.full(len(source_node_constraints), -np.inf)
        #     b_source_node_u = np.array(source_node_rhs)
        #     constraints_list.append(LinearConstraint(A_source_node, b_source_node_l, b_source_node_u))
        #     print(f"\n----- Added {len(source_node_constraints)} source node constraints -----")
        #     print(f"A_source_node shape: {A_source_node.shape}")

        #     # if A_source_node.shape[0] < 20:  # Only print small matrices
        #         # print("Source node constraint matrix:")
        #         # matprint(A_source_node)
        #         # print("Upper bounds:")
        #         # print(b_source_node_u)

        # Balance the objective function - use a small weight for source nodes
        source_node_weight = 0  # 1.1#1.1  # Adjust this if needed - smaller weight means components are more important. We set it to 1.1 to make sure that the source nodes are not selected in isolation. However, if chosen, at least one additional component should be selected for it to be an advantage.

        c = np.zeros(total_vars)
        c[:N_E] = (
            -0.1
        )  # -1 works. Maximize the number of edges. We do this to favor more specific components, e.g. BuildingSpace components with 1 adjacent space instead of 0 adjacent spaces.
        c[N_E + N_Y :] = source_node_weight  # Minimize source nodes

        # Modify the objective function to prefer complex components over multiple simple ones
        component_selection_cost = (
            0.9  # Fixed cost for selecting any component (< semantic_instance_benefit)
        )
        semantic_instance_benefit = 10  # Benefit per modeled semantic instance

        # Update the objective function coefficients
        for i in range(N_Y):
            component = Y_idx_to_component[i]
            if component in self.sim2sem_map:
                modeled_nodes = self.sim2sem_map[component]
                node_count = len(modeled_nodes)

                # Net contribution: cost - (benefit × node_count)
                c[N_E + i] = component_selection_cost - (
                    semantic_instance_benefit * node_count
                )

        # All variables are binary
        integrality = np.ones(total_vars)
        bounds = Bounds(lb=0, ub=1)

        # Solve the MILP problem
        if not constraints_list:
            print_problem(problem_info)
            return {"success": False, "message": "No valid constraints"}

        res = milp(
            c=c, constraints=constraints_list, integrality=integrality, bounds=bounds
        )

        debug = False

        if debug:
            print("=== Active components ===")
        components = []
        for i in range(N_Y):
            if res.x[N_E + i] == 1:
                component = Y_idx_to_component[i]
                components.append(component)
                if debug:
                    print(
                        f"  Y_{i} = 1: ({component.__class__.__name__}){component.id}"
                    )

        if debug:
            print("=== Active connections ===")
        connections = []
        for i in range(N_E):
            if res.x[i] == 1:
                connections.append(E_idx_to_conn[i])
                source, target, source_key, target_key = E_idx_to_conn[i]
                if debug:
                    print(
                        f"  E_{i} = 1: ({source.__class__.__name__}){source.id}.{source_key} → ({target.__class__.__name__}){target.id}.{target_key}"
                    )

        if debug:
            print("=== Inactive components ===")
        for i in range(N_Y):
            if res.x[N_E + i] == 0:
                component = Y_idx_to_component[i]
                if debug:
                    print(
                        f"  Y_{i} = 0: ({component.__class__.__name__}){component.id}"
                    )

        if debug:
            print("=== Inactive connections ===")
        for i in range(N_E):
            if res.x[i] == 0:
                source, target, source_key, target_key = E_idx_to_conn[i]
                if debug:
                    print(
                        f"  E_{i} = 0: ({source.__class__.__name__}){source.id}.{source_key} → ({target.__class__.__name__}){target.id}.{target_key}"
                    )

        if debug:
            print_problem(problem_info)

        if res.success:
            return {
                "success": True,
                "message": "Optimization successful",
                "problem_info": problem_info,
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
                    if value_.is_literal:
                        pairs_new[key_] = value_.uri.value
            return pairs_new

        # Component instantiation logic from _connect method
        class_to_instance_map = {}
        self.sim2sem_map = {}
        self.sem2sim_map = {}
        self.instance_to_group_map = {}
        self.modeled_components = set()
        for i, (component_cls, sps) in enumerate(complete_groups.items()):
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
                            id_ += f"[{component.get_short_name()}]"
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
                        component = component_cls(**base_kwargs)

                        if component_cls not in class_to_instance_map:
                            class_to_instance_map[component_cls] = {}

                        assert (
                            component.id not in class_to_instance_map[component_cls]
                        ), f"Component {component.id} already exists in class {component_cls}"
                        class_to_instance_map[component_cls][component.id] = component

                        # Get all parameters for the component
                        for key, node in sp.parameters.items():
                            if group[node] is not None:
                                value = group[node]
                                value = value.uri.value
                                obj = rgetattr(component, key)
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
                        sps_new = {sp: [group]}
                        self.instance_to_group_map[component] = (
                            modeled_match_nodes,
                            (component_cls, sps_new),
                        )
                        self.sim2sem_map[component] = modeled_match_nodes
                        for modeled_match_node in modeled_match_nodes:
                            if modeled_match_node not in self.sem2sim_map:
                                self.sem2sim_map[modeled_match_node] = set()
                            self.sem2sim_map[modeled_match_node].add(component)
                    else:
                        component = class_to_instance_map[component_cls][
                            id_
                        ]  # Get the existing component
                        (modeled_match_nodes_, (_, sps_new)) = (
                            self.instance_to_group_map[component]
                        )
                        assert (
                            modeled_match_nodes_ == modeled_match_nodes
                        ), "The modeled_match_nodes are not the same"
                        if sp not in sps_new:
                            sps_new[sp] = []
                        sps_new[sp].append(group)
                        self.instance_to_group_map[component] = (
                            modeled_match_nodes,
                            (component_cls, sps_new),
                        )

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
        # Extract the components that are actually used in connections
        new_E_conn_to_sp_group = {}
        used_components = set()
        for conn in connections:
            source, target, source_key, target_key = conn
            used_components.add(source)
            used_components.add(target)
            new_E_conn_to_sp_group[conn] = self.E_conn_to_sp_group[conn]
            sim_model.add_connection(*conn)
        self.E_conn_to_sp_group = new_E_conn_to_sp_group

        # Clean up the maps to only include used components
        # 1. Update instance_to_group_map
        self.instance_to_group_map = {
            component: group_info
            for component, group_info in self.instance_to_group_map.items()
            if component in used_components
        }

        # 2. Update sim2sem_map
        self.sim2sem_map = {
            component: nodes
            for component, nodes in self.sim2sem_map.items()
            if component in used_components
        }

        # 3. Update sem2sim_map - this is more complex as it's inversely mapped
        new_sem2sim_map = {}
        for sem_node, sim_components in self.sem2sim_map.items():
            # Filter to only keep used components for each semantic node
            used_sim_components = {
                comp for comp in sim_components if comp in used_components
            }
            if used_sim_components:  # Only keep entries that still have components
                new_sem2sim_map[sem_node] = used_sim_components
        self.sem2sim_map = new_sem2sim_map

        # 4. Update modeled_components set
        self.modeled_components = {
            node
            for component in used_components
            for node in self.sim2sem_map.get(component, set())
        }

    @staticmethod
    def _copy_nodemap(nodemap):
        return {k: v for k, v in nodemap.items()}

    @staticmethod
    def _copy_nodemap_list(nodemap_list):
        return [Translator._copy_nodemap(nodemap) for nodemap in nodemap_list]

    @staticmethod
    def _prune_recursive(
        sm_subject,
        sp_subject,
        sp_sm_map_list,
        feasible,
        comparison_table,
        sp,
        verbose=False,
    ):
        """
        Performs a depth-first search that simultaniously traverses and compares sp_subject in the signature pattern with sm_subject in the semantic model.
        """
        if sp_subject not in feasible:
            feasible[sp_subject] = set()
        if sp_subject not in comparison_table:
            comparison_table[sp_subject] = set()
        feasible[sp_subject].add(sm_subject)
        comparison_table[sp_subject].add(sm_subject)
        sm_predicate_object_pairs = sm_subject.get_predicate_object_pairs()
        sp_predicate_object_pairs = sp_subject.predicate_object_pairs
        ruleset = sp.ruleset

        print("\nENTERED RECURSIVE") if verbose else None
        print("sm_predicate_object_pairs") if verbose else None
        for p, o in sm_predicate_object_pairs.items():
            for v in o:
                print(p, str(v)) if verbose else None
        print("sp_predicate_object_pairs") if verbose else None
        for p, o in sp_predicate_object_pairs.items():
            for v in o:
                print(p, str(v)) if verbose else None
        print("\n") if verbose else None

        id_sp = sp_subject.id
        id_sp = id_sp.replace(r"\n", "")
        mn = sm_subject.uri if sm_subject is not None else None
        id_m = [str(mn)]
        print(id_sp, id_m) if verbose else None
        new_node_map_list = []
        for (
            sp_predicate,
            sp_object,
        ) in (
            sp_predicate_object_pairs.items()
        ):  # iterate the required attributes/predicates of the signature node
            print("SP_PREDICATE: ", sp_predicate) if verbose else None
            (
                print(
                    "sm_predicate_object_pairs keys: ", sm_predicate_object_pairs.keys()
                )
                if verbose
                else None
            )
            id_sp = sp_subject.id
            id_sp = id_sp.replace(r"\n", "")
            mn = sm_subject.uri if sm_subject is not None else None
            id_m = [str(mn)]
            print("FOR pair: ", id_sp, id_m) if verbose else None
            if (
                sp_predicate in sm_predicate_object_pairs
            ):  # is there a match with the semantic node?
                sm_object = sm_predicate_object_pairs[sp_predicate]
                if sm_object is not None:
                    for sp_object_ in sp_object:
                        rule = ruleset[(sp_subject, sp_predicate, sp_object_)]
                        pairs, rule_applies, ruleset = rule.apply(
                            sm_subject,
                            sm_object,
                            ruleset,
                            sp_sm_map_list=sp_sm_map_list,
                        )
                        found = False
                        for (
                            sp_sm_map_list__,
                            filtered_sm_object,
                            filtered_sp_object,
                            filtered_ruletype,
                        ) in pairs:
                            print("\n") if verbose else None
                            print("TESTING") if verbose else None
                            id_sp = filtered_sp_object.id
                            id_sp = id_sp.replace(r"\n", "")
                            mn = (
                                filtered_sm_object.uri
                                if filtered_sm_object is not None
                                else None
                            )
                            id_m = [str(mn)]
                            print(id_sp, id_m) if verbose else None

                            if filtered_sp_object not in comparison_table:
                                comparison_table[filtered_sp_object] = set()
                            if filtered_sp_object not in feasible:
                                feasible[filtered_sp_object] = set()

                            if (
                                filtered_sm_object
                                not in comparison_table[filtered_sp_object]
                            ):  # sp_object_
                                comparison_table[filtered_sp_object].add(
                                    filtered_sm_object
                                )  # sp_object_
                                sp_sm_map_list_, feasible, comparison_table, prune = (
                                    Translator._prune_recursive(
                                        filtered_sm_object,
                                        filtered_sp_object,
                                        sp_sm_map_list__,
                                        feasible,
                                        comparison_table,
                                        sp,
                                        verbose=verbose,
                                    )
                                )

                                if prune == False:
                                    if (
                                        isinstance(rule, (SinglePath, MultiPath))
                                        and rule.stop_early
                                    ):
                                        if (
                                            filtered_ruletype == Exact
                                        ):  # TODO: Check if this is correct. Assumes specific order?
                                            new_node_map_list.extend(sp_sm_map_list_)
                                            found = True
                                            print("STOPPING EARLY") if verbose else None
                                            break

                                if found and prune == False:

                                    # name = sm_subject.id if "id" in get_object_attributes(sm_subject) else sm_subject.__class__.__name__
                                    warnings.warn(
                                        f'Multiple matches found for context signature node "{sp_subject.id}" and semantic model node "{sm_subject.uri}".'
                                    )

                                if prune == False:
                                    new_node_map_list.extend(sp_sm_map_list_)
                                    found = True

                            elif (
                                filtered_sm_object in feasible[filtered_sp_object]
                            ):  # sp_object_

                                # print("FOUND IN FEASIBLE")
                                for sp_sm_map__ in sp_sm_map_list__:
                                    sp_sm_map__[filtered_sp_object] = (
                                        filtered_sm_object  # sp_object_
                                    )
                                new_node_map_list.extend(sp_sm_map_list__)
                                found = True

                        if found == False and isinstance(rule, Optional_) == False:
                            feasible[sp_subject].discard(sm_subject)
                            print("PRUNED #1:") if verbose else None
                            id_sp = sp_subject.id
                            id_sp = id_sp.replace(r"\n", "")
                            mn = sm_subject.uri if sm_subject is not None else None
                            id_m = [str(mn)]
                            print(id_sp, id_m) if verbose else None
                            print("\n") if verbose else None
                            return sp_sm_map_list, feasible, comparison_table, True
                        else:
                            sp_sm_map_list = new_node_map_list

                            print("\nCURRENT list: ") if verbose else None
                            for l in sp_sm_map_list:
                                (
                                    print("GROUP------------------------------")
                                    if verbose
                                    else None
                                )
                                for sp_subject___, sm_subject___ in l.items():
                                    id_sp = sp_subject___.id
                                    id_sp = id_sp.replace(r"\n", "")
                                    mn = (
                                        sm_subject___.uri
                                        if sm_subject___ is not None
                                        else None
                                    )
                                    id_m = [str(mn)]
                                    print(id_sp, id_m) if verbose else None

                else:
                    for sp_object_ in sp_object:
                        rule = ruleset[(sp_subject, sp_predicate, sp_object_)]
                        if isinstance(rule, Optional_) == False:
                            feasible[sp_subject].discard(sm_subject)
                            print("PRUNED #2") if verbose else None
                            return sp_sm_map_list, feasible, comparison_table, True
            else:
                for sp_object_ in sp_object:
                    rule = ruleset[(sp_subject, sp_predicate, sp_object_)]
                    if isinstance(rule, Optional_) == False:
                        feasible[sp_subject].discard(sm_subject)
                        print("PRUNED #3") if verbose else None
                        return sp_sm_map_list, feasible, comparison_table, True

        if len(sp_sm_map_list) == 0:
            sp_sm_map = {sp_subject: None for sp_subject in sp.nodes}
            sp_sm_map_list = [sp_sm_map]

        sp_sm_map_list = Translator._copy_nodemap_list(sp_sm_map_list)
        for sp_sm_map__ in sp_sm_map_list:
            sp_sm_map__[sp_subject] = sm_subject

        print("\nRETURNING list: ") if verbose else None
        for l in sp_sm_map_list:
            print("GROUP------------------------------") if verbose else None
            for sp_subject___, sm_subject___ in l.items():
                id_sp = sp_subject___.id
                id_sp = id_sp.replace(r"\n", "")
                mn = sm_subject___.uri if sm_subject___ is not None else None
                id_m = [str(mn)]
                print(id_sp, id_m) if verbose else None

        return sp_sm_map_list, feasible, comparison_table, False

    @staticmethod
    def _match(group, sp_sm_map, sp, cg, new_ig, feasible_map, comparison_table_map):
        can_match = all(
            [
                (
                    group[sp_subject] == sp_sm_map[sp_subject]
                    if group[sp_subject] is not None
                    and sp_sm_map[sp_subject] is not None
                    else True
                )
                for sp_subject in sp.nodes
            ]
        )
        is_match = False
        if can_match:
            node_map_no_None = {
                sp_subject: sm_subject
                for sp_subject, sm_subject in sp_sm_map.items()
                if sm_subject is not None
            }

            for sp_subject, match_node_nm in node_map_no_None.items():
                for attr, sp_object in sp_subject.predicate_object_pairs.items():
                    predicate_object_pairs = match_node_nm.get_predicate_object_pairs()
                    if (
                        attr in predicate_object_pairs
                        and len(predicate_object_pairs[attr]) != 0
                    ):
                        node_map_child = predicate_object_pairs[attr]
                        for sp_object_ in sp_object:
                            group_child = group[sp_object_]
                            if group_child is not None and len(node_map_child) != 0:
                                if group_child in node_map_child:
                                    is_match = True
                                    break
                    if is_match:
                        break
                if is_match:
                    break

            if is_match:
                for sp_subject, sm_subject_ in node_map_no_None.items():
                    feasible = {sp_subject: set() for sp_subject in sp.nodes}
                    comparison_table = {sp_subject: set() for sp_subject in sp.nodes}
                    sp.reset_ruleset()
                    group_prune = Translator._copy_nodemap(group)
                    group_prune = {
                        sp_node___: group_prune[sp_node___] for sp_node___ in sp.nodes
                    }
                    l, _, _, prune = Translator._prune_recursive(
                        sm_subject_,
                        sp_subject,
                        [group_prune],
                        feasible,
                        comparison_table,
                        sp,
                    )
                    if prune:
                        is_match = False
                        break

            if is_match:
                for sp_node__, match_node__ in node_map_no_None.items():
                    group[sp_node__] = match_node__
                if all(
                    [group[sp_subject] is not None for sp_subject in sp.required_nodes]
                ):
                    cg.append(group)
                    new_ig.remove(group)

        if not is_match:
            group_no_None = {
                sp_subject: sm_subject
                for sp_subject, sm_subject in group.items()
                if sm_subject is not None
            }
            for sp_subject, match_node_group in group_no_None.items():
                for (
                    sp_predicate,
                    sp_object,
                ) in sp_subject.predicate_object_pairs.items():
                    predicate_object_pairs = (
                        match_node_group.get_predicate_object_pairs()
                    )
                    if (
                        sp_predicate in predicate_object_pairs
                        and len(predicate_object_pairs[sp_predicate]) != 0
                    ):
                        group_child = predicate_object_pairs[sp_predicate]
                        for sp_object_ in sp_object:
                            node_map_child_ = sp_sm_map[sp_object_]
                            if node_map_child_ is not None and group_child is not None:
                                if group_child == node_map_child_:
                                    is_match = True
                                    break
                    if is_match:
                        break
                if is_match:
                    break

            if is_match:
                for sp_subject, sm_subject_ in node_map_no_None.items():
                    feasible = {sp_subject: set() for sp_subject in sp.nodes}
                    comparison_table = {sp_subject: set() for sp_subject in sp.nodes}
                    sp.reset_ruleset()
                    group_prune = Translator._copy_nodemap(group)
                    group_prune = {
                        sp_node___: group_prune[sp_node___] for sp_node___ in sp.nodes
                    }
                    l, _, _, prune = Translator._prune_recursive(
                        sm_subject_,
                        sp_subject,
                        [group_prune],
                        feasible,
                        comparison_table,
                        sp,
                    )

                    if prune:
                        is_match = False
                        break

            if is_match:
                for sp_node__, match_node__ in node_map_no_None.items():
                    group[sp_node__] = match_node__
                if all(
                    [group[sp_subject] is not None for sp_subject in sp.required_nodes]
                ):
                    cg.append(group)
                    new_ig.remove(group)
        return is_match, group, cg, new_ig


class Node:
    node_instance_count = count()

    def __init__(self, cls, graph_name=None, hash_=None):
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

    def h(self):
        return self._hash

    def eq(self, other):
        return self._hash == other._hash

    @property
    def signature_pattern(self):
        return self._signature_pattern

    @property
    def id(self):
        return self._id

    # @property
    # def graph_name(self):
    #     if self._graph_name is None:
    #         graph_name = "<"
    #         n = len(self.cls)

    #         for i, c in enumerate(self.cls):
    #             graph_name += c.get_short_name()
    #             if i < n-1:
    #                 id += ", "
    #         graph_name += f"\nn<SUB>{str(next(Node.node_instance_count))}</SUB>>"
    #         self._graph_name = graph_name
    #     return self._graph_name

    @property
    def semantic_model(self):
        """Get the semantic model associated with this node"""
        return self.signature_pattern.semantic_model

    def __str__(self):
        return self.id

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
        return str([str(s) for s in self.cls])

    def set_signature_pattern(self, signature_pattern):
        """Set the signature pattern for this node"""
        self._signature_pattern = signature_pattern

    def get_type_attributes(self):
        attr = {}
        for c in self.cls:
            attr.update(c.get_type_attributes())
        return attr


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
    ownedBy : str
        Name of the component class that owns this pattern
    priority : int
        Priority level for pattern matching (higher values take precedence)
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
    ...         ownedBy="DamperSystem",
    ...         priority=0
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
    ...         ownedBy="PIControllerFMUSystem"
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
    ...         ownedBy="BuildingSpaceTorchSystem",
    ...         priority=510,
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
    ...         ownedBy="DamperSystemBrick",
    ...         priority=1
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
    ...         ownedBy="SensorSystem"
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

    def __init__(
        self, semantic_model_=None, id=None, ownedBy=None, priority=0, pedantic=False
    ):
        assert isinstance(
            ownedBy, (str,)
        ), 'The "ownedBy" argument must be a class.'  # from type to str
        if semantic_model_ is None:
            semantic_model_ = core.SemanticModel()

        assert isinstance(
            semantic_model_, core.SemanticModel
        ), 'The "semantic_model_" argument must be an instance of SemanticModel.'
        self.semantic_model = semantic_model_

        if id is None:
            id = f"{ownedBy}_{str(next(SignaturePattern._signature_instance_count))}"

        self.id = id
        SignaturePattern._signatures[id] = self
        SignaturePattern._signatures_reversed[self] = id
        self.ownedBy = ownedBy
        self._nodes = []
        self._required_nodes = []
        self._inputs = {}
        self.p_inputs = []
        self._modeled_nodes = []
        self._ruleset = {}
        self._priority = priority
        self._parameters = {}
        self._pedantic = pedantic

        if self._pedantic:
            self.semantic_model.parse_namespaces(
                self.semantic_model.graph, namespaces=self.semantic_model.namespaces
            )

    @property
    def parameters(self):
        return self._parameters

    @property
    def priority(self):
        return self._priority

    @property
    def nodes(self):
        assert (
            len(self._nodes) > 0
        ), f"No nodes in the SignaturePattern owned by {self.ownedBy}. It must contain at least 1 node."
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
    def modeled_nodes(self):
        assert (
            len(self._modeled_nodes) > 0
        ), f"No nodes has been marked as modeled in the SignaturePattern owned by {self.ownedBy}. At least 1 node must be marked."
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
        subject = rule.subject
        object = rule.object
        predicate = rule.predicate
        assert isinstance(subject, Node) and isinstance(
            object, Node
        ), '"a" and "b" must be instances of class Node'
        self._add_node(subject, rule)
        self._add_node(object, rule)

        subject.set_signature_pattern(self)
        object.set_signature_pattern(self)
        subject.validate_cls()
        object.validate_cls()

        if self._pedantic:
            attributes_a = subject.get_type_attributes()
            assert (
                predicate in attributes_a
            ), f"The \"predicate\" argument must be one of the following: {', '.join(attributes_a)} - \"{predicate}\" was provided."

        if (
            predicate not in subject.predicate_object_pairs
        ):  # TODO: should maybe also be added to self.semantic_model.graph for visualization?
            subject.predicate_object_pairs[predicate] = [object]
        else:
            subject.predicate_object_pairs[predicate].append(object)
        self._ruleset[(subject, predicate, object)] = rule

    def add_input(self, key, node, source_keys=None):
        cls = list(node.cls)
        assert (
            key not in self._inputs
        ), f'Input key "{key}" already exists in the SignaturePattern owned by {self.ownedBy}.'

        if source_keys is None:
            source_keys = {c: key for c in cls}
        elif isinstance(source_keys, str):
            source_keys = {c: source_keys for c in cls}
        elif isinstance(source_keys, tuple):
            source_keys_ = {}
            for c, source_key in zip(cls, source_keys):
                source_keys_[c] = source_key
            source_keys = source_keys_

        self._inputs[key] = (node, source_keys)
        self.p_inputs.append(f"{node.id} | {key}")

    def _add_node(self, node, rule):
        if node not in self._nodes:
            self._nodes.append(node)

        if isinstance(rule, Optional_) == False:
            if node not in self._required_nodes:
                self._required_nodes.append(node)

    def add_parameter(self, key, node):
        cls = list(node.cls)
        # allowed_classes = (float, int)
        allowed_classes = (
            core.namespace.XSD.float,
            core.namespace.XSD.int,
            core.namespace.XSD.boolean,
        )
        assert any(
            n.istype(allowed_classes) for n in cls
        ), f"The class of the \"node\" argument must be a subclass of {', '.join([c.__name__ for c in allowed_classes])} - {', '.join([c.__name__ for c in cls])} was provided."
        # assert any(issubclass(n, allowed_classes) for n in cls), f"The class of the \"node\" argument must be a subclass of {', '.join([c.__name__ for c in allowed_classes])} - {', '.join([c.__name__ for c in cls])} was provided."
        self._parameters[key] = node

    def add_modeled_node(self, node):
        if node not in self._modeled_nodes:
            self._modeled_nodes.append(node)
        if node not in self._nodes:
            self._nodes.append(node)

    def remove_modeled_node(self, node):
        self._modeled_nodes.remove(node)

    def reset_ruleset(self):
        for rule in self._ruleset.values():
            rule.reset()

    def add_namespace(self, namespace):
        self.semantic_model.graph.parse(namespace)


class Rule:
    r"""
    Base class for pattern matching rules that define how signature pattern elements map to semantic model elements.

    Rules are the fundamental building blocks of signature patterns, defining the constraints and flexibility
    of the pattern matching process. Each rule specifies how a relationship between two nodes in the signature
    pattern should be matched against the semantic model.

    Overview
    --------
    Rules define the mapping between signature pattern elements and semantic model elements through:
    - **Subject**: The source node in the signature pattern
    - **Object**: The target node in the signature pattern
    - **Predicate**: The relationship type between subject and object
    - **Priority**: The precedence level for rule application (higher values take precedence)

    Rule Types
    ----------
    The Translator supports several types of rules, each with different matching behavior:

    - **Exact**: Requires exact matches between pattern and semantic model elements
    - **SinglePath**: Allows traversal along a single path in the semantic model
    - **MultiPath**: Allows traversal along multiple paths in the semantic model
    - **Optional**: Makes pattern elements optional (may or may not be present)

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
    >>> # Define relationships with different rule types
    >>> exact_rule = Exact(meter_node, fan_node, "observes")
    >>> path_rule = SinglePath(meter_node, flow_node, "hasValue")
    >>> optional_rule = Optional_(fan_node, flow_node, "hasProperty")
    >>>
    >>> # Combine rules
    >>> combined_rule = exact_rule & path_rule | optional_rule

    Attributes
    ----------
    subject : Node
        The source node in the signature pattern
    object : Node
        The target node in the signature pattern
    predicate : str
        The relationship type between subject and object
    PRIORITY : int
        The precedence level for rule application
    """

    def __init__(self, subject=None, object=None, predicate=None):
        self.subject = subject
        self.object = object
        self.predicate = predicate

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)


class And(Rule):
    def __init__(self, rule_a, rule_b):
        super().__init__()
        self.rule_a = rule_a
        self.rule_b = rule_b

    def apply(
        self, sm_subject, sm_object, ruleset, sp_sm_map_list=None, master_rule=None
    ):  # a is match node and b is pattern node
        if master_rule is None:
            master_rule = self
        pairs_a, rule_applies_a, ruleset_a = self.rule_a.apply(
            sm_subject, sm_object, ruleset, master_rule=master_rule
        )
        pairs_b, rule_applies_b, ruleset_b = self.rule_b.apply(
            sm_subject, sm_object, ruleset, master_rule=master_rule
        )
        return self.rule_a.get_match_nodes(sm_object).intersect(
            self.rule_b.get_matching_nodes(sm_object)
        )

    def get_sp_node(self):
        return self.object


class Or(Rule):
    def __init__(self, rule_a, rule_b):
        assert (
            rule_a.subject == rule_b.subject
        ), "The subject of the two rules must be the same."
        assert (
            rule_a.object == rule_b.object
        ), "The object of the two rules must be the same."
        assert (
            rule_a.predicate == rule_b.predicate
        ), "The predicate of the two rules must be the same."
        subject = rule_a.subject
        object = rule_a.object
        predicate = rule_a.predicate
        super().__init__(subject=subject, object=object, predicate=predicate)
        self.rule_a = rule_a
        self.rule_b = rule_b

    def apply(
        self, sm_subject, sm_object, ruleset, sp_sm_map_list=None, master_rule=None
    ):
        if master_rule is None:
            master_rule = self
        pairs_a, rule_applies_a, ruleset_a = self.rule_a.apply(
            sm_subject,
            sm_object,
            ruleset,
            sp_sm_map_list=sp_sm_map_list,
            master_rule=master_rule,
        )
        pairs_b, rule_applies_b, ruleset_b = self.rule_b.apply(
            sm_subject,
            sm_object,
            ruleset,
            sp_sm_map_list=sp_sm_map_list,
            master_rule=master_rule,
        )
        if rule_applies_a and rule_applies_b:
            pairs_a.extend(pairs_b)
            ruleset_a.update(ruleset_b)
            return pairs_a, True, ruleset_a

        elif rule_applies_a:
            self.PRIORITY = self.rule_a.PRIORITY
            return pairs_a, True, ruleset_a
        elif rule_applies_b:
            self.PRIORITY = self.rule_b.PRIORITY
            return pairs_b, True, ruleset_b

        return [], False, ruleset

    def reset(self):
        self.rule_a.reset()
        self.rule_b.reset()


class Exact(Rule):
    r"""
    Rule that requires exact matches between pattern and semantic model elements.

    The Exact rule is the most restrictive rule type, requiring that the semantic model
    contains exactly the same relationship as specified in the signature pattern. This rule
    is used when you need precise control over the pattern matching process.

    Priority: 10 (highest priority)

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

    PRIORITY = 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply(
        self, sm_subject, sm_object, ruleset, sp_sm_map_list=None, master_rule=None
    ):  # a is potential match nodes and b is pattern node
        # print("ENTERED EXACT")
        if master_rule is None:
            master_rule = self
        pairs = []
        rule_applies = False

        if len(sp_sm_map_list) == 0:
            sp_sm_map_list = [None]

        for sp_sm_map in sp_sm_map_list:
            sm_subject_no_match = []
            sm_object_no_match = []

            if sp_sm_map is not None:
                for (sp_subject, sp_predicate, sp_object), rule in ruleset.items():
                    if (
                        sp_object in sp_sm_map
                        and sp_subject == self.subject
                        and sp_predicate == self.predicate
                        and sp_object != self.object
                    ):
                        sm_object_no_match.append(sp_sm_map[sp_object])

                for (sp_subject, sp_predicate, sp_object), rule in ruleset.items():
                    if (
                        sp_subject in sp_sm_map
                        and sp_object == self.object
                        and sp_predicate == self.predicate
                        and sp_subject != self.subject
                    ):
                        sm_subject_no_match.append(sp_sm_map[sp_subject])
                sp_sm_map_list_ = [sp_sm_map]
            else:
                sp_sm_map_list_ = []

            for sm_object_ in sm_object:
                if (
                    sm_object_.isinstance(self.object.cls)
                    and sm_subject not in sm_subject_no_match
                    and sm_object_ not in sm_object_no_match
                ):
                    pairs.append((sp_sm_map_list_, sm_object_, self.object, Exact))
                    rule_applies = True

        return pairs, rule_applies, ruleset

    def reset(self):
        pass


class _SinglePath(Rule):
    PRIORITY = 2

    def __init__(self, **kwargs):
        self.first_entry = True
        super().__init__(**kwargs)

    def apply(
        self, sm_subject, sm_object, ruleset, sp_sm_map_list=None, master_rule=None
    ):
        if master_rule is None:
            master_rule = self
        pairs = []
        sm_objects = []
        rule_applies = False
        if self.first_entry:
            # print("FIRST ENTRY")
            self.first_entry = False
            sm_objects.extend(sm_object)
            rule_applies = True
        else:
            if len(sm_object) == 1:
                for sm_object_ in sm_object:
                    predicate_object_pairs = sm_object_.get_predicate_object_pairs()
                    if (
                        self.predicate in predicate_object_pairs
                        and len(predicate_object_pairs[self.predicate]) == 1
                    ):
                        sm_objects.append(sm_object_)
                        rule_applies = True

        if rule_applies:
            for sm_object_ in sm_objects:
                # The hash is provided to make sure the added key in the sm_sp_map is recognizeable by its context and not its
                subject = Node(
                    cls=sm_object_.type,
                    hash_=(sm_object_, self.subject, self.predicate, self.object),
                )
                subject.set_signature_pattern(self.object.signature_pattern)
                subject.validate_cls()
                subject.predicate_object_pairs[self.predicate] = [self.object]
                ruleset[(subject, self.predicate, self.object)] = master_rule
                pairs.append((sp_sm_map_list, sm_object_, subject, _SinglePath))
        else:
            subject = None
        return pairs, rule_applies, ruleset

    def reset(self):
        self.first_entry = True


class SinglePath(Rule):
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

    PRIORITY = 1

    def __init__(self, stop_early=True, **kwargs):
        self.rule = Exact(**kwargs) | _SinglePath(**kwargs)  # This order
        self.stop_early = stop_early
        super().__init__(**kwargs)

    def apply(
        self, sm_subject, sm_object, ruleset, sp_sm_map_list=None, master_rule=None
    ):
        if master_rule is None:
            master_rule = self  ###################################
        pairs, rule_applies, ruleset = self.rule.apply(
            sm_subject,
            sm_object,
            ruleset,
            sp_sm_map_list=sp_sm_map_list,
            master_rule=master_rule,
        )
        return pairs, rule_applies, ruleset

    def reset(self):
        self.rule.first_entry = True


class _MultiPath(Rule):
    PRIORITY = 2

    def __init__(self, **kwargs):
        self.first_entry = True
        super().__init__(**kwargs)

    def apply(
        self, sm_subject, sm_object, ruleset, sp_sm_map_list=None, master_rule=None
    ):
        if master_rule is None:
            master_rule = self
        pairs = []
        sm_objects = []
        rule_applies = False
        if self.first_entry:
            self.first_entry = False
            sm_objects.extend(sm_object)
            rule_applies = True
        else:
            if len(sm_object) >= 1:
                for sm_object_ in sm_object:
                    attributes = sm_object_.get_predicate_object_pairs()
                    if (
                        self.predicate in attributes
                        and len(attributes[self.predicate]) >= 1
                    ):
                        sm_objects.append(sm_object_)
                        rule_applies = True

        if rule_applies:
            for sm_object_ in sm_objects:
                subject = Node(cls=sm_object_.type)
                subject.set_signature_pattern(self.object.signature_pattern)
                subject.validate_cls()
                subject.predicate_object_pairs[self.predicate] = [self.object]
                ruleset[(subject, self.predicate, self.object)] = master_rule
                pairs.append((sp_sm_map_list, sm_object_, subject, _MultiPath))
        else:
            subject = None
        return pairs, rule_applies, ruleset

    def reset(self):
        self.first_entry = True


class Optional_(Rule):
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

    PRIORITY = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply(
        self, sm_subject, sm_object, ruleset, sp_sm_map_list=None, master_rule=None
    ):
        if master_rule is None:
            master_rule = self
        pairs = []
        rule_applies = False
        for sm_object_ in sm_object:
            if sm_object_.isinstance(self.object.cls):
                pairs.append((sp_sm_map_list, sm_object_, self.object, Optional_))
                rule_applies = True

        return pairs, rule_applies, ruleset

    def reset(self):
        pass


class MultiPath(Rule):
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

    PRIORITY = 1

    def __init__(self, stop_early=True, **kwargs):
        self.rule = Exact(**kwargs) | _MultiPath(**kwargs)
        self.stop_early = stop_early
        super().__init__(**kwargs)

    def apply(
        self, sm_subject, sm_object, ruleset, sp_sm_map_list=None, master_rule=None
    ):
        pairs, rule_applies, ruleset = self.rule.apply(
            sm_subject,
            sm_object,
            ruleset,
            sp_sm_map_list=sp_sm_map_list,
            master_rule=master_rule,
        )
        return pairs, rule_applies, ruleset

    def reset(self):
        self.rule.first_entry = True
