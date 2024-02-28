
from twin4build.saref4syst.system import System
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.get_object_attributes import get_object_attributes
from itertools import count
class NodeBase:
    NODE_INSTANCE_COUNT = count()
    def __init__(self):
        pass

def Node(cls, **kwargs):
    if not isinstance(cls, tuple):
        cls = (cls, )
    cls = cls + (NodeBase, )
    
    class Node_(*cls):
        def __init__(self, cls=None, **kwargs):
            if any([issubclass(c, (System, )) for c in cls]):
                if "id" not in kwargs:
                    kwargs["id"] = str(next(NodeBase.NODE_INSTANCE_COUNT))
            else:
                self.id = str(next(NodeBase.NODE_INSTANCE_COUNT))
            self.cls = cls
            self.attributes = set()
            super().__init__(**kwargs)
    node = Node_(cls, **kwargs)
    return node



class SignaturePattern():
    def __init__(self, ownedBy=None, priority=0):
        assert isinstance(ownedBy, (str, )), "The \"ownedBy\" argument must be a string."
        self.ownedBy = ownedBy
        self._nodes = []
        self.p_edges = []
        self._inputs = {}
        self.p_inputs = []
        self._modeled_nodes = set()
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
    def inputs(self):
        return self._inputs

    @property
    def ruleset(self):
        return self._ruleset

    @property
    def modeled_nodes(self):
        assert len(self._modeled_nodes)>0, f"No nodes has been marked as modeled in the SignaturePattern owned by {self.ownedBy}. At least 1 node must be marked."
        return self._modeled_nodes

    def add_edge(self, rule):
        assert isinstance(rule, Rule), f"The \"rule\" argument must be a subclass of Rule - \"{rule.__class__.__name__}\" was provided."
        object = rule.object
        subject = rule.subject
        predicate = rule.predicate
        assert isinstance(object, NodeBase) and isinstance(subject, NodeBase), "\"a\" and \"b\" must be instances of class Node"
        if object not in self._nodes:
            self._nodes.append(object)
        if subject not in self._nodes:
            self._nodes.append(subject)
        attributes_a = get_object_attributes(object)
        assert predicate in attributes_a, f"The \"predicate\" argument must be one of the following: {', '.join(attributes_a)} - \"{predicate}\" was provided."
        attr = rgetattr(object, predicate)
        if isinstance(attr, list):
            attr.append(subject)
        else:
            rsetattr(object, predicate, subject)
        self._ruleset[(object, subject, predicate)] = rule
        object.attributes.add(predicate)
        self.p_edges.append(f"{object.id} ----{predicate}---> {subject.id}")

    def add_input(self, key, node, source_keys=None):
        cls = list(node.cls)
        cls.remove(NodeBase)
        assert all(issubclass(t, System) for t in cls), f"All classes of \"node\" argument must be an instance of class System - {', '.join([c.__name__ for c in cls])} was provided."

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

    def add_parameter(self, key, node, source_keys=None):
        cls = list(node.cls)
        cls.remove(NodeBase)
        assert all(issubclass(t, System) for t in cls), f"All classes of \"node\" argument must be an instance of class System - {', '.join([c.__name__ for c in cls])} was provided."

        if source_keys is None:
            source_keys = {c: key for c in cls}
        elif isinstance(source_keys, str):
            source_keys = {c: source_keys for c in cls}
        elif isinstance(source_keys, tuple):
            source_keys_ = {}
            for c, source_key in zip(cls, source_keys):
                source_keys_[c] = source_key
            source_keys = source_keys_
        
        self._parameters[key] = (node, source_keys)

    def add_modeled_node(self, node):
        self._modeled_nodes.add(node)
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
                 object=None,
                 subject=None,
                 predicate=None):
        self.object = object
        self.subject = subject
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

    def apply(self, match_node, ruleset, master_rule=None): #a is match node and b is pattern node
        if master_rule is None: master_rule = self
        pairs_a, rule_applies_a, ruleset_a = self.rule_a.apply(match_node, ruleset, master_rule)
        pairs_b, rule_applies_b, ruleset_b = self.rule_b.apply(match_node, ruleset, master_rule)
        
        
        return self.rule_a.get_match_nodes(match_node).intersect(self.rule_b.get_matching_nodes(match_node))

    def get_sp_node(self):
        return self.subject

class Or(Rule):
    def __init__(self, rule_a, rule_b):
        assert rule_a.object==rule_b.object, "The object of the two rules must be the same."
        assert rule_a.subject==rule_b.subject, "The subject of the two rules must be the same."
        assert rule_a.predicate==rule_b.predicate, "The predicate of the two rules must be the same."
        object = rule_a.object
        subject = rule_a.subject
        predicate = rule_a.predicate
        super().__init__(object=object,
                        subject=subject,
                        predicate=predicate)
        self.rule_a = rule_a
        self.rule_b = rule_b
    
    def apply(self, match_node, ruleset, master_rule=None):
        print("------")
        print("ENTERED APPLY FUNCTION OF OR")
        if master_rule is None: master_rule = self
        pairs_a, rule_applies_a, ruleset_a = self.rule_a.apply(match_node, ruleset, master_rule)
        pairs_b, rule_applies_b, ruleset_b = self.rule_b.apply(match_node, ruleset, master_rule)

        print("rules__")
        print("rule_applies_a: ", rule_applies_a)
        print("rule_applies_b: ", rule_applies_b)
        if rule_applies_a and rule_applies_b:
            if self.rule_a.PRIORITY==self.rule_b.PRIORITY:
                self.PRIORITY = self.rule_a.PRIORITY
                # assert ruleset_a==ruleset_b, "The ruleset of the two rules must be the same."
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
        
        return set(), False, ruleset

    def reset(self):
        self.rule_a.reset()
        self.rule_b.reset()


class Exact(Rule):
    PRIORITY = 10
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def apply(self, match_node, ruleset, master_rule=None): #a is potential match nodes and b is pattern node
        print("------")
        print("ENTERED APPLY FUNCTION OF EXACT")
        if master_rule is None: master_rule = self
        print("subject class: ", self.subject.cls)
        pairs = set()
        rule_applies = False
        print("isinstance(match_node, list)", isinstance(match_node, list))
        if isinstance(match_node, list): #both are list
            for match_node_ in match_node:
                print("MATCH NODE EXACT: ", match_node_.id if "id" in get_object_attributes(match_node_) else match_node_.__class__.__name__)
                if isinstance(match_node_, self.subject.cls):
                    pairs.add((match_node_, self.subject))
                    rule_applies = True
        else:
            print("MATCH NODE EXACT: ", match_node.id if "id" in get_object_attributes(match_node) else match_node.__class__.__name__)
            if isinstance(match_node, self.subject.cls):
                pairs.add((match_node, self.subject))
                rule_applies = True
        return pairs, rule_applies, ruleset
    
    def reset(self):
        pass
    
    

class IgnoreIntermediateNodes(Rule):
    
    PRIORITY = 1
    def __init__(self, **kwargs):
        self.first_entry = True
        super().__init__(**kwargs)

    def apply(self, match_node, ruleset, master_rule=None): #a is potential match nodes and b is pattern node
        print("------")
        print("ENTERED APPLY FUNCTION OF IgnoreIntermediateNodes")
        print("first_entry: ", self.first_entry)
        if master_rule is None: master_rule = self
        pairs = set()
        match_nodes = set()
        rule_applies = False
        if self.first_entry:
            self.first_entry = False
            if isinstance(match_node, list): #both are list
                match_nodes.update(match_node)
            else:
                match_nodes.add(match_node)
            rule_applies = True
        else:
            if isinstance(match_node, list): #both are list
                print("len(match_node): ", len(match_node))
                
                if len(match_node)==1:
                    for match_node_ in match_node:
                        print("MATCH NODE IGNORE: ", match_node_.id if "id" in get_object_attributes(match_node_) else match_node_.__class__.__name__)
                        print("len(rgetattr(match_node_, self.predicate)): ", len(rgetattr(match_node_, self.predicate)))
                        if len(rgetattr(match_node_, self.predicate))==1:
                            match_nodes.add(match_node_)
                            rule_applies = True

                


                # if len(match_node)==1:
                #     match_nodes.update(match_node)
                #     rule_applies = True
            else:
                match_nodes.add(match_node)
                rule_applies = True
        
        if rule_applies:
            for match_node_ in match_nodes:
                object = Node(cls=(match_node_.__class__, ))
                attr = rgetattr(object, self.predicate)
                if isinstance(attr, list):
                    attr.append(self.subject)
                else:
                    rsetattr(object, self.predicate, self.subject)
                ruleset[(object, self.subject, self.predicate)] = master_rule
                object.attributes.add(self.predicate)
                pairs.add((match_node_, object))
        else:
            object = None

        return pairs, rule_applies, ruleset
    
    def reset(self):
        self.first_entry = True




# class AcceptMultipleMatches(Rule):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def get_match_nodes(self, match_node):
#         if isinstance(match_node, list): #both are list
#             if len(match_node)==1:
#                 return set(match_node)
#             else:
#                 return set()
#         else:
#             return set(match_node)
    
#     def get_sp_node(self):
#         return self.subject
        