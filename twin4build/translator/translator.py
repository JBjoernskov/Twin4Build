from rdflib import URIRef
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple
import inspect
import pandas as pd
import sys
import numpy as np
# Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 3)
    sys.path.append(file_path)

# import twin4build.base as base
# from twin4build.utils.rgetattr import rgetattr
# from twin4build.utils.rsetattr import rsetattr
from itertools import count
import twin4build.systems as systems
import warnings
import twin4build.saref4syst.system as system
import twin4build.model.simulation_model as simulation_model
import twin4build.model.semantic_model as semantic_model
import twin4build.base as base
from urllib.parse import urldefrag, urljoin, urlparse

import twin4build.systems as systems




class Translator:
    def __init__(self):
        self.instance_map = {}
        self.instance_map_reversed = {}
        self.instance_to_group_map = {}

    def translate(self, 
                 systems_: List[system.System], 
                 semantic_model: semantic_model.SemanticModel) -> simulation_model.SimulationModel:
        """
        Translate semantic model to simulation model using pattern matching
        
        Args:
            systems: List of system types to match against
            semantic_model: The semantic model to translate
            
        Returns:
            SimulationModel instance with matched components
        """
        # Initialize simulation model
        sim_model = simulation_model.SimulationModel(
            id="simulation_model",
            saveSimulationResult=False
        )

        # Match patterns
        complete_groups, incomplete_groups = self._match_patterns(
            systems_=systems_,
            semantic_model=semantic_model,
        )

        # Create component instances
        components = self._instantiate_components(complete_groups)
        
        # Connect components
        self._connect_components(components, sim_model)

        return sim_model

    @staticmethod
    def _match_patterns(systems_: List[system.System], semantic_model: semantic_model.SemanticModel) -> Tuple[Dict, Dict]:
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
                # Initialize groups for this signature pattern
                complete_groups[component_cls][sp] = []
                incomplete_groups[component_cls][sp] = []
                cg = complete_groups[component_cls][sp]
                ig = incomplete_groups[component_cls][sp]
                for sp_node in sp.nodes:
                    match_nodes = semantic_model.get_instances_of_type(sp_node.cls)
                    for match_node in match_nodes:
                        node_map = {sp_node_: None for sp_node_ in sp.nodes}
                        feasible = {sp_node: set() for sp_node in sp.nodes}
                        comparison_table = {sp_node: set() for sp_node in sp.nodes}
                        node_map_list = [Translator.copy_nodemap(node_map)]
                        prune = True
                        if match_node not in comparison_table[sp_node]:
                            sp.reset_ruleset()
                            node_map_list, node_map, feasible, comparison_table, prune = Translator._prune_recursive(match_node, sp_node, node_map, node_map_list, feasible, comparison_table, sp.ruleset)

                        elif match_node in feasible[sp_node]:
                            node_map[sp_node] = match_node
                            node_map_list = [node_map]
                            prune = False
                        
                        if prune==False:
                            # We check that the obtained node_map_list contains node maps with different modeled nodes.
                            # If an SP does not contain a MultipleMatches rule, we can prune the node_map_list to only contain node maps with different modeled nodes.
                            modeled_nodes = []
                            for node_map_ in node_map_list:
                                node_map_set = set()
                                for sp_modeled_node in sp.modeled_nodes:
                                    node_map_set.add(node_map_[sp_modeled_node])
                                modeled_nodes.append(node_map_set)
                            node_map_list_new = []
                            for i,(node_map_, node_map_set) in enumerate(zip(node_map_list, modeled_nodes)):
                                active_set = node_map_set
                                passive_set = set().union(*[v for k,v in enumerate(modeled_nodes) if k!=i])
                                if len(active_set.intersection(passive_set))>0 and any([isinstance(v, MultipleMatches) for v in sp._ruleset.values()])==False:
                                    warnings.warn(f"Multiple matches found for {sp_node.id} and {sp_node.cls}.")
                                node_map_list_new.append(node_map_) # This constraint has been removed to allow for multiple matches. Note that multiple
                            node_map_list = node_map_list_new
                            
                            # Cross matching could maybe stop early if a match is found. For SP with multiple allowed matches it might be necessary to check all matches 
                            for node_map_ in node_map_list:
                                if all([node_map_[sp_node_] is not None for sp_node_ in sp.nodes]):
                                    cg.append(node_map_)
                                else:
                                    if len(ig)==0: #If there are no groups in the incomplete group list, add the node map
                                        ig.append(node_map_)
                                    else:
                                        new_ig = ig.copy()
                                        is_match_ = False
                                        for group in ig: #Iterate over incomplete groups
                                            is_match, group, cg, new_ig = Translator._match(group, node_map_, sp, cg, new_ig)
                                            if is_match:
                                                is_match_ = True
                                        if is_match_==False:
                                            new_ig.append(node_map_)
                                        ig = new_ig
                
                ig_len = np.inf
                while len(ig)<ig_len:
                    ig_len = len(ig)
                    new_ig = ig.copy()
                    is_match = False
                    for group_i in ig:
                        for group_j in ig:
                            if group_i!=group_j:
                                is_match, group, cg, new_ig = Translator._match(group_i, group_j, sp, cg, new_ig)
                            if is_match:
                                break
                        if is_match:
                            break
                    ig = new_ig
                    
                
                # if True:#component_cls is components.BuildingSpace1AdjBoundaryOutdoorFMUSystem:
                print("INCOMPLETE GROUPS================================================================================")
                for group in ig:
                    print("GROUP------------------------------")
                    for sp_node_, match_node in group.items():
                        id_sp = str([str(s) for s in sp_node_.cls])
                        id_sp = sp_node_.id
                        id_sp = id_sp.replace(r"\n", "")
                        mn = match_node.uri if match_node is not None else None
                        id_m = [str(mn)]
                        print(id_sp, id_m)


                print("COMPLETE GROUPS================================================================================")
                for group in cg:
                    print("GROUP------------------------------")
                    for sp_node_, match_node in group.items():
                        id_sp = str([str(s) for s in sp_node_.cls])
                        id_sp = sp_node_.id
                        id_sp = id_sp.replace(r"\n", "")
                        mn = match_node.uri if match_node is not None else None
                        id_m = [str(mn)]
                        print(id_sp, id_m)
                
                
                new_ig = ig.copy()
                for group in ig: #Iterate over incomplete groups
                    if all([group[sp_node_] is not None for sp_node_ in sp.required_nodes]):  # CHANGED: Check for None instead of empty sets
                        cg.append(group)
                        new_ig.remove(group)
                ig = new_ig
                    
        return complete_groups, incomplete_groups

    def _instantiate_components(self, complete_groups: Dict) -> Dict:
        """
        Create component instances from matched groups
        
        Args:
            complete_groups: Dictionary of matched pattern groups
            
        Returns:
            Dictionary of instantiated components
        """
        # Sort groups by priority
        for component_cls, sps in complete_groups.items():
            complete_groups[component_cls] = {
                sp: groups for sp, groups in sorted(
                    complete_groups[component_cls].items(), 
                    key=lambda item: item[0].priority, 
                    reverse=True
                )
            }
        
        complete_groups = {
            k: v for k, v in sorted(
                complete_groups.items(),
                key=lambda item: max(sp.priority for sp in item[1]),
                reverse=True
            )
        }
        
        # Component instantiation logic from _connect method
        self.instance_map = {}
        self.instance_map_reversed = {}
        self.instance_to_group_map = {} ############### if changed to self.instance_to_group_map, it cannot be pickled
        self.modeled_components = set()
        for i, (component_cls, sps) in enumerate(complete_groups.items()):
            for sp, groups in sps.items():
                for group in groups:
                    modeled_match_nodes = {group[sp_node] for sp_node in sp.modeled_nodes} # CHANGED: Access single node directly
                    if len(self.modeled_components.intersection(modeled_match_nodes))==0 or any([isinstance(v, MultipleMatches) for v in sp._ruleset.values()]):
                        self.modeled_components |= modeled_match_nodes #Union/add set
                        if len(modeled_match_nodes)==1:
                            component = next(iter(modeled_match_nodes))
                            id_ = component.get_short_name()
                            base_kwargs = component.get_object_attributes()
                            extension_kwargs = {"id": id_}
                        else:
                            id_ = ""
                            modeled_match_nodes_sorted = sorted(modeled_match_nodes, key=lambda x: x.id)
                            for component in modeled_match_nodes_sorted:
                                id_ += f"[{component.get_short_name()}]"
                            base_kwargs = {}
                            extension_kwargs = {"id": id_,
                                                "base_components": list(modeled_match_nodes_sorted)}
                            for component in modeled_match_nodes_sorted:
                                kwargs = component.get_object_attributes()
                                base_kwargs.update(kwargs)

                        if id_ not in [c.id for c in self.instance_map.keys()]: #Check if the instance is already created. For components with Multiple matches, the model might already have been created.
                            base_kwargs.update(extension_kwargs)
                            component = component_cls(**base_kwargs)
                            self.instance_to_group_map[component] = (modeled_match_nodes, (component_cls, sp, [group]))
                            self.instance_map[component] = modeled_match_nodes
                            for modeled_match_node in modeled_match_nodes:
                                self.instance_map_reversed[modeled_match_node] = component
                        else:
                            component = self.instance_map_reversed[next(iter(modeled_match_nodes))] # Just index with the first element in the set as all elements should return the same component
                            (modeled_match_nodes_, (_, _, groups)) = self.instance_to_group_map[component]
                            modeled_match_nodes_ |= modeled_match_nodes
                            groups.append(group)
                            self.instance_to_group_map[component] = (modeled_match_nodes_, (component_cls, sp, groups))
                            self.instance_map[component] = modeled_match_nodes_
                            for modeled_match_node in modeled_match_nodes_:
                                self.instance_map_reversed[modeled_match_node] = component
        
        return self.instance_map

    def _connect_components(self, 
                            components: Dict, 
                            sim_model: simulation_model.SimulationModel) -> None:
        """
        Connect instantiated components and add them to simulation model
        
        Args:
            components: Dictionary of instantiated components
            sim_model: SimulationModel to add components to
        """
        for component, (modeled_match_nodes, (component_cls, sp, groups)) in self.instance_to_group_map.items():
            # Get all required inputs for the component
            for key, (sp_node, source_keys) in sp.inputs.items():
                match_node_list = [group[sp_node] for group in groups]  # CHANGED: Access single node directly
                match_node_set = {group[sp_node] for group in groups}
                if match_node_set.issubset(self.modeled_components):
                    for match_node in match_node_list:
                        component_inner = self.instance_map_reversed[match_node]
                        source_key = [source_key for c, source_key in source_keys.items() if isinstance(component_inner, c)][0]
                        sim_model.add_connection(component_inner, component, source_key, key)
                else:
                    for match_node in match_node_list:
                        warnings.warn(f"\nThe component with class \"{match_node.__class__.__name__}\" and id \"{match_node.id}\" is not modeled. The input \"{key}\" of the component with class \"{component_cls.__name__}\" and id \"{component.id}\" is not connected.\n")
            
            # Get all parameters for the component
            for key, node in sp.parameters.items():
                if groups[0][node] is not None:
                    value = groups[0][node]
                    rsetattr(component, key, value)
            
            # Add components to simulation model
            for component in components.keys():
                sim_model.components[component.id] = component

    @staticmethod
    def copy_nodemap(nodemap):
        return {k: v for k, v in nodemap.items()}

    @staticmethod
    def copy_nodemap_list(nodemap_list):
        return [Translator.copy_nodemap(nodemap) for nodemap in nodemap_list]


    @staticmethod
    def _prune_recursive(match_node, sp_node, node_map, node_map_list, feasible, comparison_table, ruleset):
        """
        Performs a depth-first search that simultaniously traverses and compares sp_node in the signature pattern with match_node in the semantic model.
        """
        if sp_node not in feasible: feasible[sp_node] = set()
        if sp_node not in comparison_table: comparison_table[sp_node] = set()
        feasible[sp_node].add(match_node)
        comparison_table[sp_node].add(match_node)
        match_name_attributes = match_node.get_object_attributes()
        sp_node_pairs = sp_node.attributes
        
        for sp_attr_name, sp_node_child in sp_node_pairs.items(): #iterate the required attributes/predicates of the signature node
            if sp_attr_name in match_name_attributes: #is there a match with the semantic node?
                match_node_child = match_name_attributes[sp_attr_name]
                if match_node_child is not None:
                    for sp_node_child_ in sp_node_child:
                        rule = ruleset[(sp_node, sp_attr_name, sp_node_child_)]
                        pairs, rule_applies, ruleset = rule.apply(match_node, match_node_child, ruleset, node_map_list=node_map_list)
                        found = False
                        new_node_map_list = []
                        for node_map_list__, filtered_match_node_child, filtered_sp_node_child in pairs:
                            if filtered_match_node_child not in comparison_table[sp_node_child_]:
                                comparison_table[sp_node_child_].add(filtered_match_node_child)
                                node_map_list_, node_map_, feasible, comparison_table, prune = Translator._prune_recursive(filtered_match_node_child, filtered_sp_node_child, node_map, node_map_list__, feasible, comparison_table, ruleset)
                                    
                                if found and prune==False:
                                    # name = match_node.id if "id" in get_object_attributes(match_node) else match_node.__class__.__name__
                                    warnings.warn(f"Multiple matches found for context signature node \"{sp_node.id}\" and semantic model node \"{match_node.uri}\".")
                                
                                if prune==False:
                                    new_node_map_list.extend(node_map_list_)
                                    found = True

                            elif filtered_match_node_child in feasible[sp_node_child_]:
                                for node_map__ in node_map_list__:
                                    node_map__[sp_node_child_] = filtered_match_node_child
                                new_node_map_list.extend(node_map_list__)
                                found = True

                        if found==False and isinstance(rule, Optional_)==False:
                            feasible[sp_node].discard(match_node)
                            return node_map_list, node_map, feasible, comparison_table, True
                        else:
                            node_map_list = new_node_map_list

                else:
                    # if isinstance(sp_node_child, list):
                    for sp_node_child_ in sp_node_child:
                        rule = ruleset[(sp_node, sp_attr_name, sp_node_child_)]
                        if isinstance(rule, Optional_)==False:
                            feasible[sp_node].discard(match_node)
                            return node_map_list, node_map, feasible, comparison_table, True
                    # else:
                    #     rule = ruleset[(sp_node, sp_attr_name, sp_node_child)]
                    #     if isinstance(rule, Optional_)==False:
                    #         feasible[sp_node].discard(match_node)
                    #         return node_map_list, node_map, feasible, comparison_table, True
            else:
                # if isinstance(sp_node_child, list):
                for sp_node_child_ in sp_node_child:
                    rule = ruleset[(sp_node, sp_attr_name, sp_node_child_)]
                    if isinstance(rule, Optional_)==False:
                        feasible[sp_node].discard(match_node)
                        return node_map_list, node_map, feasible, comparison_table, True
                # else:
                #     rule = ruleset[(sp_node, sp_attr_name, sp_node_child)]
                #     if isinstance(rule, Optional_)==False:
                #         feasible[sp_node].discard(match_node)
                #         return node_map_list, node_map, feasible, comparison_table, True
        if len(node_map_list)==0:
            node_map_list = [node_map]

        node_map_list = Translator.copy_nodemap_list(node_map_list)
        for node_map__ in node_map_list:
            node_map__[sp_node] = match_node
        
        return node_map_list, node_map, feasible, comparison_table, False


    @staticmethod
    def _match(group, node_map_, sp, cg, new_ig):
        can_match = all([group[sp_node_] == node_map_[sp_node_]
                        if group[sp_node_] is not None and node_map_[sp_node_] is not None
                        else True for sp_node_ in sp.nodes])
        is_match = False
        if can_match:
            node_map_no_None = {sp_node_: match_node
                                for sp_node_, match_node in node_map_.items()
                                if match_node is not None}

            for sp_node_, match_node_nm in node_map_no_None.items():
                attributes = sp_node_.attributes
                for attr, object in attributes.items():
                    # node_map_child = getattr(match_node_nm, attr)
                    node_map_child = match_node_nm.get_object_attributes()[attr]
                    if node_map_child is not None and (isinstance(node_map_child, list) and len(node_map_child) == 0) == False:
                        if isinstance(node_map_child, list) == False:
                            node_map_child_ = [node_map_child]
                        else:
                            node_map_child_ = node_map_child
                        if isinstance(object, list) == False:
                            subject_ = [object]
                        else:
                            subject_ = object

                        for subject__ in subject_:
                            group_child = group[subject__]
                            if group_child is not None and len(node_map_child_) != 0:
                                if group_child in node_map_child_:
                                    is_match = True
                                    break
                    if is_match:
                        break
                if is_match:
                    break

            if is_match:
                for sp_node_, match_node_ in node_map_no_None.items():
                    feasible = {sp_node: set() for sp_node in sp.nodes}
                    comparison_table = {sp_node: set() for sp_node in sp.nodes}
                    sp.reset_ruleset()
                    group_prune = Translator.copy_nodemap(group)
                    group_prune = {sp_node___: group_prune[sp_node___] for sp_node___ in sp.nodes}
                    _, _, _, _, prune = Translator._prune_recursive(match_node_, sp_node_, Translator.copy_nodemap(group_prune), [group_prune], feasible, comparison_table, sp.ruleset)
                    if prune:
                        is_match = False
                        break

                if is_match:
                    for sp_node__, match_node__ in node_map_no_None.items():
                        group[sp_node__] = match_node__  # CHANGED: Direct assignment instead of set operations
                    if all([group[sp_node_] is not None for sp_node_ in sp.nodes]):  # CHANGED: Check for None instead of empty sets
                        cg.append(group)
                        new_ig.remove(group)

        if not is_match:
            group_no_None = {sp_node_: match_node for sp_node_, match_node in group.items() if match_node is not None}
            for sp_node_, match_node_group in group_no_None.items():
                attributes = sp_node_.attributes
                for attr, object in attributes.items():
                    group_child = match_node_group.get_object_attributes()[attr]
                    if group_child is not None and (isinstance(group_child, list) and len(group_child) == 0) == False:
                        if isinstance(group_child, list) == False:
                            group_child_ = [group_child]
                        else:
                            group_child_ = group_child
                        if isinstance(object, list) == False:
                            subject_ = [object]
                        else:
                            subject_ = object

                        for subject__ in subject_:
                            node_map_child_ = node_map_[subject__]
                            if node_map_child_ is not None and group_child_ is not None:
                                if group_child_ == node_map_child_:
                                    is_match = True
                                    break
                    if is_match:
                        break
                if is_match:
                    break

            if is_match:
                for sp_node_, match_node_ in node_map_no_None.items():
                    feasible = {sp_node: set() for sp_node in sp.nodes}
                    comparison_table = {sp_node: set() for sp_node in sp.nodes}
                    sp.reset_ruleset()
                    group_prune = Translator.copy_nodemap(group)
                    group_prune = {sp_node___: group_prune[sp_node___] for sp_node___ in sp.nodes}
                    _, _, _, _, prune = Translator._prune_recursive(match_node_, sp_node_, Translator.copy_nodemap(group_prune), [group_prune], feasible, comparison_table, sp.ruleset)
                    if prune:
                        is_match = False
                        break

                if is_match:
                    for sp_node__, match_node__ in node_map_no_None.items():
                        group[sp_node__] = match_node__  # CHANGED: Direct assignment instead of set operations
                    if all([group[sp_node_] is not None for sp_node_ in sp.nodes]):  # CHANGED: Check for None instead of empty sets
                        cg.append(group)
                        new_ig.remove(group)

        return is_match, group, cg, new_ig



class Node:
    node_instance_count = count()

    def __init__(self, cls, graph_name=None):
        self._graph_name = graph_name
        if isinstance(cls, tuple)==False:
            cls = (cls,)
        self.cls = cls
        self.attributes = {}
        self._signature_pattern = None
        self.id = str([str(s) for s in self.cls])



    @property
    def signature_pattern(self):
        return self._signature_pattern
    
    @property
    def graph_name(self):
        if self._graph_name is None:
            graph_name = "<"
            n = len(self.cls)

            for i, c in enumerate(self.cls):
                graph_name += c.get_short_name()
                if i < n-1:
                    id += ", "
            graph_name += f"\nn<SUB>{str(next(Node.node_instance_count))}</SUB>>"
            self._graph_name = graph_name
        return self._graph_name

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
        if isinstance(cls, tuple)==False:
            cls = (cls,)

        cls_ = []
        for c in cls:
            if isinstance(c, semantic_model.SemanticType):
                cls_.append(c)
            elif isinstance(c, URIRef):
                cls_.append(semantic_model.SemanticType(c, self.signature_pattern.semantic_model.graph))
            elif isinstance(c, str):
                cls_.append(semantic_model.SemanticType(URIRef(c), self.signature_pattern.semantic_model.graph))
            else:
                raise ValueError(f"Invalid class type: {type(c)}")

        self.cls = tuple(cls_)  # Make immutable
    
    def set_signature_pattern(self, signature_pattern):
        """Set the signature pattern for this node"""
        self._signature_pattern = signature_pattern

    def get_type_attributes(self):
        attr = {}
        for c in self.cls:
            attr.update(c.get_type_attributes())
        return attr
    

class SignaturePattern():
    signatures = {}
    signatures_reversed = {}
    signature_instance_count = count()
    def __init__(self, semantic_model, id=None, ownedBy=None, priority=0):
        assert isinstance(ownedBy, (type, )), "The \"ownedBy\" argument must be a class."

        assert isinstance(semantic_model, semantic_model.SemanticModel), "The \"semantic_model\" argument must be an instance of SemanticModel."
        self.semantic_model = semantic_model

        if id is None:
            id = f"{ownedBy.__name__}_{str(next(SignaturePattern.signature_instance_count))}"
        self.id = id
        SignaturePattern.signatures[id] = self
        SignaturePattern.signatures_reversed[self] = id
        self.ownedBy = ownedBy
        self._nodes = []
        self._required_nodes = []
        self.p_edges = []
        self._inputs = {}
        self.p_inputs = []
        self._modeled_nodes = []
        self._ruleset = {}
        self._priority = priority
        self._parameters = {}

    @property
    def parameters(self):
        return self._parameters

    @property
    def priority(self):
        return self._priority

    @property
    def nodes(self):
        assert len(self._nodes)>0, f"No nodes in the SignaturePattern owned by {self.ownedBy}. It must contain at least 1 node."
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
        assert len(self._modeled_nodes)>0, f"No nodes has been marked as modeled in the SignaturePattern owned by {self.ownedBy}. At least 1 node must be marked."
        return self._modeled_nodes
    
    def get_node_by_id(self, id):
        for node in self._nodes:
            if node.id==id:
                return node
        return None

    def add_relation(self, rule):
        assert isinstance(rule, Rule), f"The \"rule\" argument must be a subclass of Rule - \"{rule.__class__.__name__}\" was provided."
        subject = rule.subject
        object = rule.object
        predicate = rule.predicate
        assert isinstance(subject, Node) and isinstance(object, Node), "\"a\" and \"b\" must be instances of class Node"
        self._add_node(subject, rule)
        self._add_node(object, rule)

        subject.set_signature_pattern(self)
        object.set_signature_pattern(self)
        subject.validate_cls()
        object.validate_cls()
        

        attributes_a = subject.get_type_attributes()
        assert predicate in attributes_a, f"The \"predicate\" argument must be one of the following: {', '.join(attributes_a)} - \"{predicate}\" was provided."
        if predicate not in subject.attributes:
            subject.attributes[predicate] = [object]
        else:
            subject.attributes[predicate].append(object)
        self._ruleset[(subject, predicate, object)] = rule
        
        self.p_edges.append(f"{subject.id} ----{predicate}---> {object.id}")

    def add_input(self, key, node, source_keys=None):
        cls = list(node.cls)
        assert key not in self._inputs, f"Input key \"{key}\" already exists in the SignaturePattern owned by {self.ownedBy}."

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

        if isinstance(rule, Optional_)==False:
            if node not in self._required_nodes:
                self._required_nodes.append(node)

    def add_parameter(self, key, node):
        cls = list(node.cls)
        allowed_classes = (float, int)
        assert any(issubclass(n, allowed_classes) for n in cls), f"The class of the \"node\" argument must be a subclass of {', '.join([c.__name__ for c in allowed_classes])} - {', '.join([c.__name__ for c in cls])} was provided."
        self._parameters[key] = node

    def add_modeled_node(self, node):
        if node not in self._modeled_nodes:
            self._modeled_nodes.append(node)
        if node not in self._nodes:
            self._nodes.append(node)

    def remove_modeled_node(self, node):
        self._modeled_nodes.remove(node)

    def print_edges(self):
        print("")
        print("===== EDGES =====")
        for e in self.p_edges:
            print(f"     {e}")
        print("=================")

    def print_inputs(self):
        print("")
        print("===== INPUTS =====")
        print("  Node  |  Input") 
        # print("_________________")
        for i in self.p_inputs:
            print(f"      {i}")
        print("==================")

    def reset_ruleset(self):
        for rule in self._ruleset.values():
            rule.reset()

class Rule:
    def __init__(self,
                 subject=None,
                 object=None,
                 predicate=None):
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

    def apply(self, match_node, match_node_child, ruleset, node_map_list=None, master_rule=None): #a is match node and b is pattern node
        if master_rule is None: master_rule = self
        pairs_a, rule_applies_a, ruleset_a = self.rule_a.apply(match_node, match_node_child, ruleset, master_rule=master_rule)
        pairs_b, rule_applies_b, ruleset_b = self.rule_b.apply(match_node, match_node_child, ruleset, master_rule=master_rule)
        return self.rule_a.get_match_nodes(match_node_child).intersect(self.rule_b.get_matching_nodes(match_node_child))

    def get_sp_node(self):
        return self.object

class Or(Rule):
    def __init__(self, rule_a, rule_b):
        assert rule_a.subject==rule_b.subject, "The subject of the two rules must be the same."
        assert rule_a.object==rule_b.object, "The object of the two rules must be the same."
        assert rule_a.predicate==rule_b.predicate, "The predicate of the two rules must be the same."
        subject = rule_a.subject
        object = rule_a.object
        predicate = rule_a.predicate
        super().__init__(subject=subject,
                        object=object,
                        predicate=predicate)
        self.rule_a = rule_a
        self.rule_b = rule_b
    
    def apply(self, match_node, match_node_child, ruleset, node_map_list=None, master_rule=None):
        if master_rule is None: master_rule = self
        pairs_a, rule_applies_a, ruleset_a = self.rule_a.apply(match_node, match_node_child, ruleset, node_map_list=node_map_list, master_rule=master_rule)
        pairs_b, rule_applies_b, ruleset_b = self.rule_b.apply(match_node, match_node_child, ruleset, node_map_list=node_map_list, master_rule=master_rule)
        if rule_applies_a and rule_applies_b:
            if self.rule_a.PRIORITY==self.rule_b.PRIORITY:
                self.PRIORITY = self.rule_a.PRIORITY
                return pairs_a.union(pairs_b), True, ruleset_a
            elif self.rule_a.PRIORITY > self.rule_b.PRIORITY:
                self.PRIORITY = self.rule_a.PRIORITY
                return pairs_a, True, ruleset_a
            else:
                self.PRIORITY = self.rule_b.PRIORITY
                return pairs_b, True, ruleset_b

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
    PRIORITY = 10
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def apply(self, match_node, match_node_child, ruleset, node_map_list=None, master_rule=None): #a is potential match nodes and b is pattern node
        # print("ENTERED EXACT")
        if master_rule is None: master_rule = self
        pairs = []
        rule_applies = False

        if len(node_map_list)==0:
            node_map_list = [None]

        
        for node_map in node_map_list:
            match_node_no_match = []
            match_node_child_no_match = []

            if node_map is not None:
                for (sp_node, sp_attr_name, sp_node_child_), rule in ruleset.items():
                    if sp_node_child_ in node_map and sp_node==self.subject and sp_attr_name==self.predicate and sp_node_child_!=self.object:
                        match_node_child_no_match.append(node_map[sp_node_child_])
            
                for (sp_node, sp_attr_name, sp_node_child_), rule in ruleset.items():
                    if sp_node in node_map and sp_node_child_==self.object and sp_attr_name==self.predicate and sp_node!=self.subject:
                        match_node_no_match.append(node_map[sp_node])
                node_map_list_ = [node_map]
            else:
                node_map_list_ = []
            
            # print("match_node_child_no_match", [m.id if "id" in get_object_attributes(m) else m.__class__.__name__ + str(id(m)) for m in match_node_child_no_match])
            # print("match_node_no_match", [m.id if "id" in get_object_attributes(m) else m.__class__.__name__ + str(id(m)) for m in match_node_no_match])
            # print("match_node", match_node.id if "id" in get_object_attributes(match_node) else match_node.__class__.__name__ + str(id(match_node)))

            # print("LIST")
            for match_node_child_ in match_node_child:
                # print("match_node_child", match_node_child_.id if "id" in get_object_attributes(match_node_child_) else match_node_child_.__class__.__name__ + str(id(match_node_child_)))
                if match_node_child_.isinstance(self.object.cls) and match_node not in match_node_no_match and match_node_child_ not in match_node_child_no_match:
                    pairs.append((node_map_list_, match_node_child_, self.object))
                    rule_applies = True

        # print(f"RULE APPLIES: {rule_applies}")
        return pairs, rule_applies, ruleset
    
    def reset(self):
        pass


class SinglePath(Rule):
    PRIORITY = 2
    def __init__(self, **kwargs):
        self.first_entry = True
        super().__init__(**kwargs)

    def apply(self, match_node, match_node_child, ruleset, node_map_list=None, master_rule=None): #a is potential match nodes and b is pattern node
        # print("ENTERED IGNORE")
        if master_rule is None: master_rule = self
        pairs = []
        match_nodes_child = []
        rule_applies = False
        if self.first_entry:
            # print("FIRST ENTRY")
            self.first_entry = False
            match_nodes_child.extend(match_node_child)
            rule_applies = True
        else:
            if len(match_node_child)==1:
                for match_node_child_ in match_node_child:
                    # print("---")
                    # print(f"attr :", self.predicate)
                    # print(f"value: ", rgetattr(match_node_child_, self.predicate))
                    # print(match_node_child_.get_object_attributes())
                    attributes = match_node_child_.get_object_attributes()
                    if self.predicate in attributes and len(attributes[self.predicate])==1:
                        match_nodes_child.append(match_node_child_)
                        rule_applies = True
        
        if rule_applies:
            for match_node_child_ in match_nodes_child:
                subject = Node(cls=(match_node_child_.type, ))
                subject.attributes[self.predicate] = [self.object]
                ruleset[(subject, self.predicate, self.object)] = master_rule
                pairs.append((node_map_list, match_node_child_, subject))
        else:
            subject = None
        # print(f"RULE APPLIES: {rule_applies}")
        return pairs, rule_applies, ruleset
    
    def reset(self):
        self.first_entry = True

class IgnoreIntermediateNodes(Rule):
    PRIORITY = 1
    def __init__(self, **kwargs):
        self.rule = Exact(**kwargs) | SinglePath(**kwargs)
        super().__init__(**kwargs)

    def apply(self, match_node, match_node_child, ruleset, node_map_list=None, master_rule=None):
        pairs, rule_applies, ruleset = self.rule.apply(match_node, match_node_child, ruleset, node_map_list=node_map_list, master_rule=master_rule)
        return pairs, rule_applies, ruleset
    
    def reset(self):
        self.rule.first_entry = True

class MultiPath(Rule):
    PRIORITY = 2
    def __init__(self, **kwargs):
        self.first_entry = True
        super().__init__(**kwargs)

    def apply(self, match_node, match_node_child, ruleset, node_map_list=None, master_rule=None): #a is potential match nodes and b is pattern node
        # print("ENTERED IGNORE")
        if master_rule is None: master_rule = self
        pairs = []
        match_nodes_child = []
        rule_applies = False
        if self.first_entry:
            # print("FIRST ENTRY")
            self.first_entry = False
            match_nodes_child.extend(match_node_child)

            rule_applies = True
        else:
            if len(match_node_child)>=1:
                for match_node_child_ in match_node_child:
                    attributes = match_node_child_.get_object_attributes()
                    if self.predicate in attributes and len(attributes[self.predicate])>=1:
                        match_nodes_child.append(match_node_child_)
                        rule_applies = True

        
        if rule_applies:
            for match_node_child_ in match_nodes_child:
                subject = Node(cls=(match_node_child_.type, ))
                subject.attributes[self.predicate] = [self.object]
                ruleset[(subject, self.predicate, self.object)] = master_rule
                pairs.append((node_map_list, match_node_child_, subject))
        else:
            subject = None
        # print(f"RULE APPLIES: {rule_applies}")
        return pairs, rule_applies, ruleset
    
    def reset(self):
        self.first_entry = True

class Optional_(Rule):
    PRIORITY = 1
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def apply(self, match_node, match_node_child, ruleset, node_map_list=None, master_rule=None): #a is potential match nodes and b is pattern node
        if master_rule is None: master_rule = self
        pairs = []
        rule_applies = False
        for match_node_child_ in match_node_child:
            if isinstance(match_node_child_, self.object.cls):
                pairs.append((node_map_list, match_node_child_, self.object))
                rule_applies = True
        return pairs, rule_applies, ruleset
    
    def reset(self):
        pass


class MultipleMatches(Rule):
    PRIORITY = 1
    def __init__(self, **kwargs):
        self.rule = Exact(**kwargs) | MultiPath(**kwargs)
        super().__init__(**kwargs)
        
    def apply(self, match_node, match_node_child, ruleset, node_map_list=None, master_rule=None): #a is potential match nodes and b is pattern node
        pairs, rule_applies, ruleset = self.rule.apply(match_node, match_node_child, ruleset, node_map_list=node_map_list, master_rule=master_rule)
        return pairs, rule_applies, ruleset
    
    def reset(self):
        self.rule.first_entry = True
        





################################
# Usage example:
if __name__ == "__main__":
    # Create model from a turtle file (from URL)

    print(__name__)
    turtle_file = "https://github.com/BrickSchema/Brick/blob/master/examples/soda_brick.ttl?raw=true"
    turtle_file = "https://brickschema.org/ttl/mortar/bldg8.ttl"


    print("CREATING SEMANTIC MODEL")
    turtle_file = r"C:\Users\jabj\Documents\python\Twin4build-Case-Studies\hoeje_taastrup\HTR full graph cantine.ttl"
    model = semantic_model.SemanticModel(turtle_file, parse_namespaces=True)
    model.graph.parse("https://brickschema.org/schema/1.3/Brick.ttl", format="turtle")
    model.graph.parse("https://alikucukavci.github.io/FSO/fso.ttl", format="turtle")

    print("MISSING NAMESPACES")
    print("\n".join(model.missing_namespaces))

    print("CREATING BRICK MODEL")
    brick_file = "https://brickschema.org/schema/1.4.1/Brick.ttl"
    brick_model = semantic_model.SemanticModel(brick_file, format='turtle')

    # Print discovered namespaces
    print("\nDiscovered namespaces:")
    for prefix, namespace in model.namespaces.items():
        print(f"{prefix}: {namespace}")

    print("\nLooking for class:")
    for s, p, o in model.graph.triples((None, None, None)):
        if "Damper" in str(s):
            print(f"Found class: {s}")

    # aa

    
    q = """
    CONSTUCT ?s ?p ?o
    WHERE {
        ?s ?p ?o
    }
    """

    print("VISUALIZING SEMANTIC MODEL")
    # model.visualize(class_filter=(model.BRICK.VAV, model.BRICK.AHU, model.FSO.Component, model.BOT.Space, model.FSO.MechanicalDamper), predicate_filter=(model.FSO.feedsFluidTo), filter_rule="AND")
    model.visualize(predicate_filter=(model.FSO.feedsFluidTo), filter_query=q)



    # https://w3id.org/rec#Room



    print("VISUALIZING BRICK MODEL")
    # brick_model.visualize()
    Node.set_default_graph(brick_model)
    
    # sp = SignaturePattern(brick_model, ownedBy=systems.DamperSystem)

    # node1 = Node(cls=brick_model.BRICK.VAV)
    # node2 = Node(cls=model.BOT.Space)
    # node3 = Node(cls=brick_model.BRICK.Supply_Air_Flow_Sensor)
    # node4 = Node(cls=brick_model.BRICK.Air_Handler_Unit)
    # node5 = Node(cls=brick_model.BRICK.Room)

    # sp.add_relation(IgnoreIntermediateNodes(subject=node4, object=node1, predicate="feeds"))
    # sp.add_relation(IgnoreIntermediateNodes(subject=node1, object=node2, predicate="feeds"))
    # sp.add_relation(Exact(subject=node1, object=node3, predicate="hasPoint"))
    # sp.add_relation(Exact(subject=node2, object=node5, predicate="hasPart"))
    # sp.add_modeled_node(node2)




    ###

    node0 = Node(cls=base.Damper, id="<n<SUB>1</SUB>(Damper)>") #supply damper
    node1 = Node(cls=base.Damper, id="<n<SUB>2</SUB>(Damper)>") #return damper
    node2 = Node(cls=base.BuildingSpace, id="<n<SUB>3</SUB>(BuildingSpace)>")
    node3 = Node(cls=base.Valve, id="<n<SUB>4</SUB>(Valve)>") #supply valve
    node4 = Node(cls=base.SpaceHeater, id="<n<SUB>5</SUB>(SpaceHeater)>")
    node5 = Node(cls=base.Schedule, id="<n<SUB>6</SUB>(Schedule)>") #return valve
    node6 = Node(cls=base.OutdoorEnvironment, id="<n<SUB>7</SUB>(OutdoorEnvironment)>")
    node7 = Node(cls=base.Sensor, id="<n<SUB>8</SUB>(Sensor)>")
    node8 = Node(cls=base.Temperature, id="<n<SUB>9</SUB>(Temperature)>")
    sp = SignaturePattern(ownedBy="BuildingSpace0AdjBoundaryOutdoorFMUSystem", priority=60)

    sp.add_edge(Exact(object=node0, subject=node2, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node1, subject=node2, predicate="hasFluidReturnedBy"))
    sp.add_edge(Exact(object=node3, subject=node2, predicate="isContainedIn"))
    sp.add_edge(Exact(object=node4, subject=node2, predicate="isContainedIn"))
    sp.add_edge(Exact(object=node3, subject=node4, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node2, subject=node5, predicate="hasProfile"))
    sp.add_edge(Exact(object=node2, subject=node6, predicate="connectedTo"))
    sp.add_edge(IgnoreIntermediateNodes(object=node7, subject=node0, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node7, subject=node8, predicate="observes"))


    sp.add_input("airFlowRate", node0)
    sp.add_input("waterFlowRate", node3)
    sp.add_input("numberOfPeople", node5, "scheduleValue") ##############################
    sp.add_input("outdoorTemperature", node6, "outdoorTemperature")
    sp.add_input("outdoorCo2Concentration", node6, "outdoorCo2Concentration")
    sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input("supplyAirTemperature", node7, "measuredValue")

    sp.add_modeled_node(node4)
    sp.add_modeled_node(node2)

    ###

    # node1 = Node(cls=brick_model.BRICK.VAV)
    # node2 = Node(cls=brick_model.BRICK.Air_Handler_Unit)
    # sp.add_relation(IgnoreIntermediateNodes(subject=node2, object=node1, predicate="feedsFluidTo"))
    # sp.add_modeled_node(node2)

    ss = [systems.BuildingSpace0AdjBoundaryOutdoorFMUSystem]
    ss[0].sp = [sp]

    print("TRANSLATING")
    translator = Translator()
    translator.translate(ss, model)

    # classes = [cls[1] for cls in inspect.getmembers(systems, inspect.isclass) if (issubclass(cls[1], (system.System, )) and hasattr(cls[1], "sp"))]
    
################################
