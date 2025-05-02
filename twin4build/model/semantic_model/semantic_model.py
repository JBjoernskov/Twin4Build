import io
import brickschema.brickify.src.handlers.Handler.TableHandler as table_handler
import pydotplus
import shutil
import subprocess
import warnings
import logging
import os
import sys
import pandas as pd
from openpyxl import load_workbook
from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal
from typing import List, Dict, Any, Set, Optional, Tuple, Type, Union
from urllib.error import HTTPError
from rdflib.tools.rdf2dot import rdf2dot
import rdflib.tools.csv2rdf
from bs4 import BeautifulSoup
from twin4build.utils.mkdir_in_root import mkdir_in_root
from twin4build.utils.uppath import uppath
import twin4build.core as core
from pathlib import Path
import typer


class SemanticProperty:
    """Represents an ontology property"""
    def __init__(self, uri: Union[str, URIRef], graph: Graph):
        # Convert string URI to URIRef if needed
        self.uri = URIRef(uri) if isinstance(uri, str) else uri
        self.graph = graph
        
        # Property types to check for
        property_types = {
            URIRef("http://www.w3.org/2002/07/owl#ObjectProperty"),
            URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            URIRef("http://www.w3.org/2002/07/owl#DatatypeProperty"),
            URIRef("http://www.w3.org/2002/07/owl#AnnotationProperty"),
            URIRef("http://www.w3.org/2002/07/owl#FunctionalProperty")
        }
        
        # Check if URI represents a valid property
        is_property = any((self.uri, RDF.type, prop_type) in self.graph for prop_type in property_types)
        is_used_as_predicate = any(self.graph.triples((None, self.uri, None)))
        
        if not (is_property or is_used_as_predicate):
            raise ValueError(f"URI '{self.uri}' is not a valid property in the ontology")
        
        self._domain = None
        self._range = None
    
    @property
    def domain(self) -> List[URIRef]:
        """Get the domain (valid subject types) of this property"""
        if self._domain is None:
            self._domain = set(self.graph.objects(self.uri, RDFS.domain))
        return self._domain
    
    @property
    def range(self) -> List[URIRef]:
        """Get the range (valid object types) of this property"""
        if self._range is None:
            self._range = set(self.graph.objects(self.uri, RDFS.range))
        return self._range
    
    def __str__(self):
        return str(self.uri)
    
    def get_short_name(self):
        return str(self.uri).split('#')[-1]
    
    def isproperty(self, cls: Union[str, "SemanticProperty", Tuple[Union[str, "SemanticProperty"], ...], List[Union[str, "SemanticProperty"]]]) -> bool:
        """Check if this instance is of any of the given property types (including inheritance)
        
        Args:
            property: Single property or tuple/list of properties to check against
            
        Returns:
            True if instance matches any of the specified properties
        """
        if not isinstance(cls, (tuple, list)):
            cls = (cls,)
        
        # Check each class in the tuple against all instance types
        for c in cls:
            if str(c) == str(self.uri):
                return True
        return False


class SemanticType:
    """Represents an ontology class with inheritance"""
    def __init__(self, uri: Union[str, URIRef], graph: Graph, validate=False):
        # Convert string URI to URIRef if needed
        self.uri = URIRef(uri) if isinstance(uri, str) else uri
        self.graph = graph

        if validate:
        
            # Built-in RDF/RDFS classes that are always valid
            BUILT_IN_CLASSES = {
                URIRef("http://www.w3.org/2000/01/rdf-schema#Resource"),
                URIRef("http://www.w3.org/2000/01/rdf-schema#Class"),
                URIRef("http://www.w3.org/2002/07/owl#Class")
            }
            
            # Debug: Print all type triples for this URI
            # print(f"\nDebug - Checking type declarations for {self.uri}")
            # print("All triples where this URI is subject:")
            # for s, p, o in self.graph.triples((self.uri, None, None)):
            #     print(f"  {p} -> {o}")
            # print("All triples where this URI is object:")
            # for s, p, o in self.graph.triples((None, None, self.uri)):
            #     print(f"  {s} -> {p}")
            
            # Check if URI represents a valid type
            is_owl_class = (self.uri, RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class")) in self.graph
            is_rdfs_class = (self.uri, RDF.type, RDFS.Class) in self.graph
            is_built_in = self.uri in BUILT_IN_CLASSES
            
            # Additional checks for class-like behavior
            has_subclass = any(self.graph.triples((None, RDFS.subClassOf, self.uri)))
            is_subclass = any(self.graph.triples((self.uri, RDFS.subClassOf, None)))
            has_instances = any(self.graph.triples((None, RDF.type, self.uri)))
            
            # Check if it's a property (which should not be treated as a class)
            property_types = {
                URIRef("http://www.w3.org/2002/07/owl#ObjectProperty"),
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
                URIRef("http://www.w3.org/2002/07/owl#DatatypeProperty"),
                URIRef("http://www.w3.org/2002/07/owl#AnnotationProperty"),
                URIRef("http://www.w3.org/2002/07/owl#FunctionalProperty")
            }
            
            is_property = any((self.uri, RDF.type, prop_type) in self.graph for prop_type in property_types)
            is_used_as_predicate = any(self.graph.triples((None, self.uri, None)))
            
            # Check if it's used in domain/range declarations (suggesting it's a class)
            is_in_domain = any(self.graph.triples((None, RDFS.domain, self.uri)))
            is_in_range = any(self.graph.triples((None, RDFS.range, self.uri)))
            
            # print(f"Debug - Class checks for {self.uri}:")
            # print(f"  is_owl_class: {is_owl_class}")
            # print(f"  is_rdfs_class: {is_rdfs_class}")
            # print(f"  is_built_in: {is_built_in}")
            # print(f"  has_subclass: {has_subclass}")
            # print(f"  is_subclass: {is_subclass}")
            # print(f"  has_instances: {has_instances}")
            # print(f"  is_property: {is_property}")
            # print(f"  is_used_as_predicate: {is_used_as_predicate}")
            # print(f"  is_in_domain: {is_in_domain}")
            # print(f"  is_in_range: {is_in_range}")
            
            if is_property or is_used_as_predicate:
                raise ValueError(f"URI '{self.uri}' is a property, not a class")
            
            # Consider it a valid class if any of these conditions are true
            if not (is_owl_class or is_rdfs_class or is_built_in or has_subclass or 
                    is_subclass or has_instances or is_in_domain or is_in_range):
                raise ValueError(f"URI '{self.uri}' is not declared as a valid class/type in the ontology")
        
        self._parent_classes = None
        self._attributes = None
        
    @property
    def parent_classes(self) -> Set[str]:
        """Get all parent classes (including indirect) using RDFS reasoning"""
        if self._parent_classes is None:
            self._parent_classes = set()
            for parent in self.graph.transitive_objects(self.uri, RDFS.subClassOf):
                self._parent_classes.add(str(parent))
        return self._parent_classes
    
    def get_type_attributes(self) -> Dict[str, List[Any]]:
        """Find all possible attributes (properties) that can be used with instances of this class.
        
        This method looks for all ObjectProperties defined in the ontology that could be used
        with instances of this class (self.uri). These are properties that either:
        1. Have no domain restrictions (can be used with any class)
        2. Have this class or any of its parent classes in their domain
        
        Returns:
            Dictionary mapping property names to lists of their allowed range values
        """
        if self._attributes is None:
            self._attributes = {}
            
            # Find all ObjectProperties in the ontology
            for prop, _, _ in self.graph.triples((None, RDF.type, URIRef("http://www.w3.org/2002/07/owl#ObjectProperty"))): # We are looking explicitly for ObjectProperties. This could maybe be generalized?
                # pred_name = str(prop).split('#')[-1]
                
                # Get the domains (if any) for this property
                domains = list(self.graph.objects(prop, RDFS.domain))
                
                # Property is valid if it has no domain restrictions or if this class/parents are in its domain
                if not domains or any(str(domain) in self.parent_classes or domain == self.uri for domain in domains):
                    # Get the ranges for this property
                    ranges = list(self.graph.objects(prop, RDFS.range))
                    self._attributes[prop] = ranges
                        
        return self._attributes
    
    def __str__(self):
        return str(self.uri)
    
    def get_short_name(self):
        return str(self.uri).split('#')[-1]
    
    def istype(self, cls: Union[str, "SemanticType", Tuple[Union[str, "SemanticType"], ...], List[Union[str, "SemanticType"]]]) -> bool:
        """Check if this instance is of any of the given class types (including inheritance)
        
        Args:
            cls: Single class or tuple/list of classes to check against
            
        Returns:
            True if instance matches any of the specified classes
        """
        # Convert single class to tuple for consistent handling
        if not isinstance(cls, (tuple, list)):
            cls = (cls,)
        
        # Check each class in the tuple against all instance types
        for c in cls:
            if str(c) == str(self.uri):
                return True
            elif str(c) in self.parent_classes:
                return True
        return False
    
    def has_subclasses(self) -> bool:
        """Check if this type has any subclasses"""
        return any(self.graph.triples((None, RDFS.subClassOf, self.uri)))


class SemanticObject:
    """Class to represent an ontology instance or literal"""
    def __init__(self, value: Union[str, URIRef, Literal], model: "SemanticModel", datatype: Optional[URIRef] = None, lang: Optional[str] = None):
        """
        Initialize a semantic object representing either a URI resource or a literal
        
        Args:
            value: The URI or literal value
            model: The semantic model this object belongs to
            datatype: Optional datatype URI for literals. If provided (not None), the value is treated as a literal.
            lang: Optional language tag for literals. If provided (not None), the value is treated as a literal.
        """
        self.model = model
        
        # Handle different input types
        if isinstance(value, Literal):
            # Already a Literal object
            self.uri = value
            self.is_literal = True
        elif datatype is not None or lang is not None:
            # String that should be treated as a literal
            self.uri = Literal(value, datatype=datatype, lang=lang)
            self.is_literal = True
        else:
            # URI reference
            self.uri = URIRef(value) if isinstance(value, str) else value
            self.is_literal = False
            
        self._types = None
        self._attributes = None
    
    @property
    def type(self) -> List[SemanticType]:
        """Get all types of this instance"""
        if self.is_literal:
            # For literals, return the datatype
            if self._types is None:
                if self.uri.datatype:
                    self._types = {self.model.get_type(self.uri.datatype)}
                else:
                    # Plain literals without datatype
                    self._types = {self.model.get_type(RDFS.Literal)}
            return self._types
            
        if self._types is None:
            # First check direct RDF.type assertions
            direct_types = set(self.model.graph.objects(self.uri, RDF.type))
            
            # Then check for type through owl:sameAs relations
            same_as = set(self.model.graph.objects(self.uri, URIRef("http://www.w3.org/2002/07/owl#sameAs")))
            for same_as_uri in same_as:
                same_as_types = set(self.model.graph.objects(same_as_uri, RDF.type))
                direct_types.update(same_as_types)
            
            # Convert all direct types to SemanticType objects and include their parent types
            types_with_parents = set()
            for t in direct_types:
                # Add the direct type
                type_obj = self.model.get_type(t)
                types_with_parents.add(type_obj)
                
                # Add parent types
                for parent_uri in type_obj.parent_classes:
                    types_with_parents.add(self.model.get_type(parent_uri))
                
            self._types = types_with_parents
        return self._types
    
    def get_predicate_object_pairs(self) -> Dict[str, Any]:
        """Return all attributes of this instance"""
        if self.is_literal:
            # Literals don't have properties
            return {}
            
        if self._attributes is None:
            self._attributes = {}
            for pred, obj in self.model.graph.predicate_objects(self.uri):

                is_class = any(self.model.graph.triples((obj, RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class")))) or \
                              any(self.model.graph.triples((obj, RDF.type, RDFS.Class)))
                
                if is_class:
                    obj_instance = self.model.get_type(obj)
                else:
                    obj_instance = self.model.get_instance(obj)


                # if pred != RDF.type: #It is not necessary to include the type triples when used with signature pattern as the type explicitly defined in the signature pattern
                    # Handle both URI and literal objects
                    
                    
                if pred in self._attributes:
                    self._attributes[pred].append(obj_instance)
                else:
                    self._attributes[pred] = [obj_instance]
        return self._attributes
    
    def isinstance(self, cls: Union[str, SemanticType, Tuple[Union[str, SemanticType], ...], List[Union[str, SemanticType]]]) -> bool:
        """Check if this instance is of any of the given class types (including inheritance)
        
        Args:
            cls: Single class or tuple/list of classes to check against
            
        Returns:
            True if instance matches any of the specified classes
        """
        # Convert single class to tuple for consistent handling
        if not isinstance(cls, (tuple, list)):
            cls = (cls,)

        # Handle literals differently
        if self.is_literal:
            # Check each class in the tuple against the literal's datatype
            for c in cls:
                c_str = str(c)
                # Direct datatype match
                if self.uri.datatype and c_str == str(self.uri.datatype):
                    return True
                # Check for XSD.string match with language-tagged literals
                if self.uri.language and c_str == str(self.model.XSD.string):
                    return True
                # Plain literals (no datatype) match with XSD.string
                if not self.uri.datatype and not self.uri.language and c_str == str(self.model.XSD.string):
                    return True
            return False

        if self.type:
            # Check each class in the tuple against all instance types
            for c in cls:
                for instance_type in self.type:
                    if str(c) == str(instance_type.uri):
                        return True
                    elif str(c) in instance_type.parent_classes:
                        return True
        return False

    def __str__(self):
        return str(self.uri)
    
    def get_short_name(self):
        for namespace in self.model.namespaces.values():
            if namespace in str(self.uri):
                return str(self.uri).split(namespace)[-1]
        return None
    
    def get_namespace(self):
        for namespace in self.model.namespaces.values():
            for instance_type in self.type:
                if namespace in str(instance_type.uri):
                    return namespace
        return None

class SemanticModel:
    def __init__(self, 
                 rdf_file: Optional[str] = None,
                 namespaces: Optional[Dict[str, str]] = None,
                 format: Optional[str] = None,
                 parse_namespaces=False,
                 verbose=False,
                 id: str="semantic_model",
                 dir_conf: List[str] = None):
        """
        Initialize the ontology model
        Args:
            rdf_file: Path or URL to the ontology file
            namespaces: Optional additional namespace prefix-URI pairs
            format: Optional format specification ('xml', 'turtle', 'n3', 'nt', 'json-ld', etc.)
        """
        self.id = id
        self.rdf_file = rdf_file
        if namespaces is None:
            self.namespaces = {}
        else:
            assert isinstance(namespaces, dict), "The \"namespaces\" argument must be a dictionary."
            self.namespaces = {}
            for prefix, uri in namespaces.items():
                self.namespaces[prefix.upper()] = Namespace(uri)
                setattr(self, prefix.upper(), self.namespaces[prefix.upper()])
        self.format = format
        self.parsed_namespaces = set()
        self.error_namespaces = set()

        # Cache for instances
        self._instances = {}
        self._types = {}
        self._properties = {}


        if dir_conf is None:
            self.dir_conf = ["generated_files", "models", self.id]
        else:
            self.dir_conf = dir_conf

        if rdf_file is not None:
            if verbose:
                self._init(namespaces=namespaces, 
                           parse_namespaces=parse_namespaces)
            else:
                logging.disable(sys.maxsize) # https://stackoverflow.com/questions/2266646/how-to-disable-logging-on-the-standard-error-stream
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._init(namespaces=namespaces, 
                               parse_namespaces=parse_namespaces)
        else:
            self.graph = Graph()
            for prefix, namespace in self.namespaces.items():
                self.graph.bind(prefix.lower(), namespace)
            # logging.disable(logging.NOTSET)
        

    def _init(self, 
            namespaces: Optional[Dict[str, str]] = None,
            parse_namespaces=False):
        # Load and parse the RDF file
        self.graph = self.get_graph(self.rdf_file, self.format)
        
        # Common namespaces
        self.RDF = RDF
        self.RDFS = RDFS
                
        
        if parse_namespaces:
            self.parse_namespaces(self.graph, namespaces=namespaces)
            
        # Extract namespaces from the graph
        for prefix, uri in self.graph.namespaces():
            if prefix:  # Skip empty prefix
                self.namespaces[prefix.upper()] = Namespace(uri)
                setattr(self, prefix.upper(), self.namespaces[prefix.upper()])


        for prefix, namespace in self.namespaces.items():
            self.graph.bind(prefix.lower(), namespace)
        
        

    def get_graph_copy(self):
        """Create a complete copy of the graph including namespace bindings.
        
        Returns:
            Graph: A new graph instance with all triples and namespace bindings copied
        """
        new_graph = Graph()
        
        # Copy all namespace bindings
        for prefix, namespace in self.graph.namespaces():
            new_graph.bind(prefix, namespace)
            
        # Copy all triples
        for s, p, o in self.graph.triples((None, None, None)):
            new_graph.add((s, p, o))
        return new_graph

    def parse_namespaces(self, graph, namespaces=None):
        if namespaces is None:
            namespaces = graph.namespaces()

        for prefix, uri in namespaces.items():
            try:
                if graph==self.graph:
                    if uri not in self.parsed_namespaces and uri not in self.error_namespaces:
                        graph.parse(uri)
                        self.parsed_namespaces.add(uri)
                        self.namespaces[prefix.upper()] = Namespace(uri)
                        setattr(self, prefix.upper(), self.namespaces[prefix.upper()])
                else:
                    graph.parse(uri)
            except HTTPError as err:
                warnings.warn(f"Cannot parse namespace {uri}.")
                if graph==self.graph:
                    if uri not in self.parsed_namespaces and uri not in self.error_namespaces:
                        if prefix.upper() in core.ontologies.namespaces:
                            uri_ = core.ontologies.namespaces[prefix.upper()]
                            warnings.warn(f"Found namespace in core.ontologies.namespaces: {uri_}")
                            graph.parse(uri_)
                        else:
                            self.error_namespaces.add(uri)
                else:
                    if prefix.upper() in core.ontologies.namespaces:
                        uri_ = core.ontologies.namespaces[prefix.upper()]
                        warnings.warn(f"Found namespace in core.ontologies.namespaces: {uri_}")
                        graph.parse(uri_)
                    else:
                        self.error_namespaces.add(uri)
            except Exception as err:
                warnings.warn(f"Cannot parse namespace {uri}.")
                if graph==self.graph:
                    if uri not in self.parsed_namespaces and uri not in self.error_namespaces:
                        if prefix.upper() in core.ontologies.namespaces:
                            uri_ = core.ontologies.namespaces[prefix.upper()]
                            warnings.warn(f"Found namespace in core.ontologies.namespaces: {uri_}")
                            graph.parse(uri_)
                        else:
                            self.error_namespaces.add(uri)
                else:
                    if prefix.upper() in core.ontologies.namespaces:
                        uri_ = core.ontologies.namespaces[prefix.upper()]
                        warnings.warn(f"Found namespace in core.ontologies.namespaces: {uri_}")
                        graph.parse(uri_)
                    else:
                        self.error_namespaces.add(uri)
        # else:
        #     for prefix, uri in self.graph.namespaces():
        #         if isinstance(uri, str):
        #             uri = URIRef(uri)

        #         if graph==self.graph:
        #             if uri not in self.parsed_namespaces and uri not in self.error_namespaces:
        #                 graph.parse(uri)
        #                 self.parsed_namespaces.add(uri)
        #         else:
        #             graph.parse(uri)

    def get_dir(self, folder_list: List[str] = None, filename: Optional[str] = None) -> Tuple[str, bool]:
        """
        Get the directory path for storing model-related files.

        Args:
            folder_list (List[str]): List of folder names to create.
            filename (Optional[str]): Name of the file to create.

        Returns:
            Tuple[str, bool]: The full path to the directory or file, and a boolean indicating if the file exists.
        """
        if folder_list is None:
            folder_list = []
        folder_list_ = self.dir_conf.copy()
        folder_list_.extend(folder_list)
        filename, isfile = mkdir_in_root(folder_list=folder_list_, filename=filename)
        return filename, isfile

    def get_graph(self, filename: str, format: Optional[str] = None, mappings_dir: Optional[str] = None) -> Graph:
        if os.path.isfile(filename):
            # Get extension
            # ext = os.path.splitext(filename)[1][1:].lower()
            graph = self.parse_spreadsheet(filename, mappings_dir=mappings_dir)
        else:
            graph = Graph()
            if format is None:
                graph.parse(filename)
            else:
                graph.parse(filename, format=format)

        return graph
    
    def filter_graph(self, query: str) -> Graph:
        """Filter the graph based on class and predicate filters.
        The filtering is done using OR(class_filter) and OR(predicate_filter).
        
        Args:
            class_filter: List of class URIs to include (None = no class filtering)
            predicate_filter: List of predicates to include (None = no predicate filtering)
            
        Returns:
            Filtered graph
        """
        
        new_graph = self.get_graph_copy()
        #self.get_graph(self.rdf_file, self.format)
        keep_triples = set()

        if query is not None:
            if query.strip().upper().startswith('CONSTRUCT'):
                result = self.graph.query(query)
                # Create new graph with only the constructed triples
                for row in result:
                    keep_triples.add(row)
            else:
                raise ValueError("Query must start with CONSTRUCT")
            for s, p, o in new_graph.triples((None, None, None)):
                if (s, p, o) not in keep_triples:
                    new_graph.remove((s, p, o))
        return new_graph
        
    def get_instance(self, value: Union[str, URIRef, Literal], datatype: Optional[URIRef] = None, lang: Optional[str] = None) -> SemanticObject:
        """
        Get a specific instance by URI or create a literal
        
        Args:
            value: The URI or literal value
            is_literal: Flag to indicate if the value should be treated as a literal
            datatype: Optional datatype URI for literals
            lang: Optional language tag for literals
            
        Returns:
            SemanticObject representing the URI or Literal
        """
        # Handle literals
        if isinstance(value, Literal) or datatype is not None or lang is not None:
            # Create a new literal object (we don't cache literals)
            if isinstance(value, Literal):
                return SemanticObject(value, self)
            else:
                return SemanticObject(value, self, datatype=datatype, lang=lang)
            
        # Handle URIs
        uri = URIRef(value) if isinstance(value, str) else value
        if uri not in self._instances:
            self._instances[uri] = SemanticObject(uri, self)
        return self._instances[uri]
    
    def get_type(self, uri: str) -> SemanticType:
        """Get a specific type by URI"""
        uri = URIRef(uri) if isinstance(uri, str) else uri
        if uri not in self._types:
            self._types[uri] = SemanticType(uri, self.graph)
        return self._types[uri]
    
    def get_property(self, uri: str) -> SemanticProperty:
        """Get a specific property by URI"""
        uri = URIRef(uri) if isinstance(uri, str) else uri
        if uri not in self._properties:
            self._properties[uri] = SemanticProperty(uri, self.graph)
        return self._properties[uri]
    
    def get_instances_of_type(self, class_uris: Union[str, URIRef, SemanticType, Tuple, List]) -> List[SemanticObject]:
        """
        Get all instances that match any of the specified types (including subtypes)
        
        Args:
            class_uris: Single URI or tuple/list of URIs representing the types to match
            
        Returns:
            List of SemanticObject instances that match any of the specified types
        """
        # Convert single class_uri to tuple for consistent handling
        if not isinstance(class_uris, tuple):
            if isinstance(class_uris, (list, set)):
                class_uris = tuple(class_uris)
            else:
                class_uris = (class_uris,)

        # print("GET INSTANCES OF TYPE")

        # Check if any of the requested types are XSD datatypes
        # If so, we need to find literals with those datatypes
        xsd_datatypes = []
        uri_types = []
        
        for class_uri in class_uris:
            if isinstance(class_uri, str):
                uri = URIRef(class_uri)
            elif isinstance(class_uri, SemanticType):
                uri = class_uri.uri
            elif isinstance(class_uri, URIRef):
                uri = class_uri
            else:
                raise ValueError(f"Invalid class URI: {class_uri}")
                
            # Check if this is an XSD datatype
            if str(uri).startswith("http://www.w3.org/2001/XMLSchema#"):
                xsd_datatypes.append(uri)
            else:
                uri_types.append(uri)
        
        instances = []
        processed_instances = set()  # To avoid duplicates
        
        # First, handle regular URI instances
        if uri_types:
            # Process each type in the tuple
            for uri in uri_types:            
                # Get the class and all its subclasses
                subclasses = set([uri])
                for subclass in self.graph.transitive_subjects(RDFS.subClassOf, uri):
                    subclasses.add(subclass)
                            
                # Get instances of the class and its subclasses
                for subclass in subclasses:
                    # print(f"SUBCLASS: {subclass}")
                    # First check direct type assertions
                    for instance in self.graph.subjects(RDF.type, subclass):
                        if instance not in processed_instances:
                            inst_obj = self.get_instance(instance)
                            instances.append(inst_obj)
                            processed_instances.add(instance)
                    
                    # Then check for indirect type assertions through owl:sameAs
                    for instance in self.graph.subjects(RDF.type, subclass):
                        for same_as in self.graph.objects(instance, URIRef("http://www.w3.org/2002/07/owl#sameAs")):
                            if same_as not in processed_instances:
                                inst_obj = self.get_instance(same_as)
                                instances.append(inst_obj)
                                processed_instances.add(same_as)
        
        # Then, handle literals with the specified datatypes
        if xsd_datatypes:
            # Find all literals in the graph with the specified datatypes
            for s, p, o in self.graph.triples((None, None, None)):
                if isinstance(o, Literal) and o.datatype in xsd_datatypes:
                    # Create a SemanticObject for this literal
                    literal_obj = self.get_instance(o)
                    instances.append(literal_obj)
        
        return instances
    
    def count_instances(self) -> int:
        return len(list(self.graph.subjects(RDF.type, None)))

    def count_triples(self) -> int:
        return len(list(self.graph.triples((None, None, None))))

    def visualize(self, query=None):
        """
        Visualize RDF graph with optional class and predicate filtering.
        The filter acts as an OR filter.
        
        Args:
            class_filter: List of class URIs to include (None = no class filtering)
            predicate_filter: List of predicates to include (None = no predicate filtering)
        """
        # Omit rdf:type triples by default
        if query is None:
            query = """
            CONSTRUCT {
                ?s ?p ?o 
            }
            WHERE {
                ?s ?p ?o .
                FILTER (?p != rdf:type && 
                        ?p != rdfs:subClassOf)
            }
            """


        light_black = "#3B3838"
        dark_blue = "#44546A"
        orange = "#DC8665"#"#C55A11"
        red = "#873939"
        grey = "#666666"
        light_grey = "#71797E"
        light_blue = "#8497B0"
        green = "#83AF9B"#"#BF9000"
        buttercream = "#B89B72"
        green = "#83AF9B"
        white = "#FFFFFF"

        # Here we set the attributes for the tables.
        # The lists are indexed by the row number.
        # The first element is the attribute for the first row, etc.
        # The last element is None, which means that the default value is kept.
        
        fill_color_map = {core.S4BLDG.BuildingSpace: [light_black, light_black, None, None],
                          core.S4BLDG.Controller: [orange, orange, None, None],
                          core.S4BLDG.AirToAirHeatRecovery: [dark_blue, dark_blue, None, None],
                          core.S4BLDG.Coil: [red, red, None, None],
                          core.S4BLDG.Damper: [dark_blue, dark_blue, None, None],
                          core.S4BLDG.Valve: [red, red, None, None],
                          core.S4BLDG.Fan: [dark_blue, dark_blue, None, None],
                          core.S4BLDG.SpaceHeater: [red, red, None, None],
                          core.SAREF.Sensor: [green, green, None, None],
                          core.SAREF.Meter: [green, green, None, None],
                          core.S4BLDG.Schedule: [grey, grey, None, None],
                          core.S4BLDG.Pump: [red, red, None, None],
                          core.S4SYST.System: [green, green, None, None],
                          core.S4SYST.Connection: [light_blue, light_blue, None, None],
                          core.S4SYST.ConnectionPoint: [light_blue, light_blue, None, None]}
        
        font_color_map = {core.S4BLDG.BuildingSpace: [white, white, None, None],
                          core.S4BLDG.Controller: [white, white, None, None],
                          core.S4BLDG.AirToAirHeatRecovery: [white, white, None, None],
                          core.S4BLDG.Coil: [white, white, None, None],
                          core.S4BLDG.Damper: [white, white, None, None],
                          core.S4BLDG.Valve: [white, white, None, None],
                          core.S4BLDG.Fan: [white, white, None, None],
                          core.S4BLDG.SpaceHeater: [white, white, None, None],
                          core.SAREF.Sensor: [white, white, None, None],
                          core.SAREF.Meter: [white, white, None, None],
                          core.S4BLDG.Schedule: [white, white, None, None],
                          core.S4BLDG.Pump: [white, white, None, None],
                          core.S4SYST.System: [white, white, None, None],
                          core.S4SYST.Connection: [white, white, None, None],
                          core.S4SYST.ConnectionPoint: [white, white, None, None]}
        
        font_size_map = {core.S4BLDG.BuildingSpace: [10, 8, 8, 6],
                         core.S4BLDG.Controller: [10, 8, 8, 6],
                         core.S4BLDG.AirToAirHeatRecovery: [10, 8, 8, 6],
                         core.S4BLDG.Coil: [10, 8, 8, 6],
                         core.S4BLDG.Damper: [10, 8, 8, 6],
                         core.S4BLDG.Valve: [10, 8, 8, 6],
                         core.S4BLDG.Fan: [10, 8, 8, 6],
                         core.S4BLDG.SpaceHeater: [10, 8, 8, 6],
                         core.SAREF.Sensor: [10, 8, 8, 6],
                         core.SAREF.Meter: [10, 8, 8, 6],
                         core.S4BLDG.Schedule: [10, 8, 8, 6],
                         core.S4BLDG.Pump: [10, 8, 8, 6],
                         core.S4SYST.System: [10, 8, 8, 6],
                         core.S4SYST.Connection: [7, 6, 6, 5],
                         core.S4SYST.ConnectionPoint: [7, 6, 6, 5]}
        

        font_bold_map = {core.S4BLDG.BuildingSpace: [True, True, None, None],
                         core.S4BLDG.Controller: [True, True, None, None],
                         core.S4BLDG.AirToAirHeatRecovery: [True, True, None, None],
                         core.S4BLDG.Coil: [True, True, None, None],
                         core.S4BLDG.Damper: [True, True, None, None],
                         core.S4BLDG.Valve: [True, True, None, None],
                         core.S4BLDG.Fan: [True, True, None, None],
                         core.S4BLDG.SpaceHeater: [True, True, None, None],
                         core.SAREF.Sensor: [True, True, None, None],
                         core.SAREF.Meter: [True, True, None, None],
                         core.S4BLDG.Schedule: [True, True, None, None],
                         core.S4BLDG.Pump: [True, True, None, None],
                         core.S4SYST.System: [True, True, None, None],
                         core.S4SYST.Connection: [False, False, None, None],
                         core.S4SYST.ConnectionPoint: [False, False, None, None]}

        # Filter graph
        graph = self.filter_graph(query)
        stream = io.StringIO()
        rdf2dot(graph, stream)

        dg = pydotplus.graph_from_dot_data(stream.getvalue())

        # Add class type to node labels
        for node in dg.get_nodes():
            if 'label' in node.obj_dict['attributes']:
                html_str = node.obj_dict['attributes']['label']
                soup = BeautifulSoup(html_str, 'html.parser')
                soup.table.attrs.update({"border":"2", "width":"100%"})
                row = soup.find_all('tr')[1]
                col = row.find_all('td')[0]
                uri = col.string
                inst = self.get_instance(uri)
                type_ = inst.type

                z_ = {e for e in type_ if e.has_subclasses()==False}
                z = {e.uri.n3(self.graph.namespace_manager) for e in z_}
                if len(z)==0:
                    z = {"Unknown class"}

                z = " | ".join(z)# data
                b = soup.new_tag('b', attrs={})
                b.string = z
                new_col = soup.new_tag('td', attrs={"bgcolor": "grey", "colspan": "2"})
                new_col.append(b)
                new_row = soup.new_tag("tr", attrs={"border": "1px solid black"})
                new_row.append(new_col)
                first_row = soup.find_all("tr")[0]
                first_row.insert_before(new_row)


                


                for i, row in enumerate(soup.find_all("tr")):#[:-1]: #All except the last row, which is the full inst URI (small blue text)
                    for col in row.find_all("td"):
                        attrs = {}
                        bold = False

                        for t in type_:
                            if t.uri in fill_color_map:
                                # print(col.attrs)
                                if t.uri in fill_color_map:
                                    if fill_color_map[t.uri][i] is not None:
                                        col.attrs['bgcolor'] = fill_color_map[t.uri][i]


                                if t.uri in font_color_map:
                                    if font_color_map[t.uri][i] is not None:
                                        attrs['color'] = font_color_map[t.uri][i]
                                
                                if t.uri in font_size_map:
                                    if font_size_map[t.uri][i] is not None:
                                        attrs['point-size'] = font_size_map[t.uri][i]

                                if t.uri in font_bold_map:
                                    if font_bold_map[t.uri][i] is not None:
                                        bold = font_bold_map[t.uri][i]

                    
                        font = soup.new_tag('font', attrs=attrs)
                                        
    
                        if col.string:
                            s = col.string
                            new_col = soup.new_tag('td', attrs=col.attrs)
                            col.replace_with(new_col)
                            new_col.append(font)


                        elif col.find("b"):
                            s = col.find("b").string
                            col.find("b").replace_with(font)


                        if bold:
                            s_ = soup.new_tag('b', attrs={})
                            s_.string = s
                            s = s_
                        font.append(s)
                node.obj_dict['attributes']['label'] = str(soup).replace("&lt;", "<").replace("&gt;", ">")

        def del_dir(dirname):
            for filename in os.listdir(dirname):
                file_path = os.path.join(dirname, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        dirname,_ = self.get_dir(folder_list=["graphs", "temp"])

        # Delete all files in dirname
        del_dir(dirname)

        
        dot_filename = os.path.join(dirname, "object_graph.dot")
        dg.write(dot_filename)


        ### ccomps ###
        dirname_ccomps,_ = self.get_dir(folder_list=["graphs", "temp", "ccomps"])
        dot_filename_ccomps = os.path.join(dirname_ccomps, "object_graph_ccomps.dot")
        del_dir(dirname_ccomps)
        app_path = shutil.which("ccomps")
        args = [app_path,
                "-x",
                f"-o{dot_filename_ccomps}",
                f"{dot_filename}"]
        subprocess.run(args=args)

        ### dot ###
        # Get all filenames generated in the folder dirname
        app_path = shutil.which("dot")
        filenames = []
        for filename in os.listdir(dirname):
            file_path = os.path.join(dirname, filename)
            if os.path.isfile(file_path):
                dot_filename_ccomps = file_path
                dot_filename_dot = os.path.join(dirname, filename.replace("ccomps", "dot"))
                args = [app_path,
                        f"-o{dot_filename_dot}",
                        f"{dot_filename_ccomps}"]
                subprocess.run(args=args)

                filenames.append(dot_filename_dot)


        dot_filename_ccomps = os.path.join(dirname, "object_graph_ccomps_joined.dot")
        with open(dot_filename_ccomps,'wb') as wfd:
            for f in filenames:
                with open(f,'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
                    # wfd.write(b"\n")

        
        ### gvpack ###
        dot_filename_gvpack = os.path.join(dirname, "object_graph_gvpack.dot")
        app_path = shutil.which("gvpack")
        args = [app_path,
                "-array3",
                f"-o{dot_filename_gvpack}",
                f"{dot_filename_ccomps}"]
        subprocess.run(args=args)

        ### neato ###
        semantic_model_png,_ = self.get_dir(folder_list=["graphs"], filename="semantic_model.png")
        app_path = shutil.which("neato")
        args = [app_path,
                "-Tpng",
                "-n2",
                "-Gsize=10!",
                "-Gdpi=2000",
                # "-v", # verbose
                f"-o{semantic_model_png}",
                f"{dot_filename_gvpack}"]
        subprocess.run(args=args)

    def parse_spreadsheet(self, spreadsheet, mappings_dir=None):
        """Parse spreadsheet into RDF graph using brickify tool"""

        # Overwrite typer progress bar to prevent it from printing to the console.
        class Overwriter:
            def __init__(self, iterable, *args, **kwargs):
                self.iterable = iterable
                
            def __enter__(self):
                return self.iterable
                
            def __exit__(self, exc_type, exc_value, traceback):
                pass
        def overwriter(iterable_object,*args,**kwargs):
            return Overwriter(iterable_object, *args, **kwargs)
        table_handler.progressbar = overwriter

        graph = Graph()
        
        # Get directories for storing files
        dirname, _ = self.get_dir(folder_list=["mappings"])
        if mappings_dir is None:
            mappings_dir = os.path.join(uppath(os.path.abspath(__file__), 1), "mappings")
        
        # Read common macros content
        common_macros_path = os.path.join(mappings_dir, "common_macros.yml")
        with open(common_macros_path, 'r') as fd:
            common_macros_content = fd.read()
        
        # Load workbook and process each sheet
        wb = load_workbook(spreadsheet)

        


        # # temp #
        # for sheet in wb.sheetnames:
        #     csv_file = os.path.join(dirname, f"{sheet}.csv")
        #     df = pd.read_excel(spreadsheet, sheet_name=sheet)
        #     df.to_csv(csv_file, index=False)
        # ########
        # aa

        for sheet in wb.sheetnames:
            # Check if there's a corresponding config file
            config_file = os.path.join(mappings_dir, f"{sheet}.yml")
            if os.path.exists(config_file):
                # Create new config file with macros in dirname
                new_config_path = os.path.join(dirname, f"{sheet}.yml")
                with open(config_file,'rb') as fd:
                    with open(new_config_path,'wb') as wfd:
                        shutil.copyfileobj(fd, wfd)
                        wfd.write(b"\n")
                        wfd.write(common_macros_content.encode())
                
                # Convert sheet to CSV
                csv_file = os.path.join(dirname, f"{sheet}.csv")
                df = pd.read_excel(spreadsheet, sheet_name=sheet)
                df.to_csv(csv_file, index=False)
                
                # Run brickify with new config file
                output_file = os.path.join(dirname, f"{sheet}.ttl")
                
                # cmd = [
                #     "brickify",
                #     csv_file,
                #     "--output", output_file,
                #     "--input-type", "csv",
                #     "--config", new_config_path,
                #     "--building-prefix", "bldg",
                #     "--building-namespace", "http://example.org/building#"
                # ]


                # Parse the output file into our graph
                handler = table_handler.TableHandler(
                    source=csv_file,
                    input_format="csv",
                    config_file=new_config_path,
                )
                
                # Convert using the handler
                result_graph = handler.convert("bldg", "http://example.org/building#", 
                                                "site", "https://example.com/site#")
                
                # Serialize to the output file
                result_graph.serialize(destination=str(output_file), format="turtle")
                
                try:
                    # subprocess.run(cmd, check=True)



                    
                    
                    # Parse the output file into our graph
                    graph.parse(output_file, format='turtle')
                except subprocess.CalledProcessError as e:
                    print(f"Error running brickify for sheet {sheet}: {e}")
                except Exception as e:
                    print(f"Error processing sheet {sheet}: {e}")
        
        return graph

    def reason(self, namespaces=None):
        """Perform RDFS and OWL reasoning to infer additional triples. Currently, we infer inverse, symmetric, and transitive properties.
        """
        if namespaces is None:
            # ontology_graph = self.graph
            namespaces = self.namespaces

        
        ontology_graph = Graph()
        self.parse_namespaces(ontology_graph, namespaces=namespaces)

        # Track new triples to add
        new_triples = set()
        
        # Handle inverse properties
        for s, p, o in ontology_graph.triples((None, URIRef("http://www.w3.org/2002/07/owl#inverseOf"), None)):
            # Find and add inverse relationships
            for subj, _, obj in self.graph.triples((None, s, None)):
                new_triple = (obj, o, subj)
                new_triples.add(new_triple)
            
            for subj, _, obj in self.graph.triples((None, o, None)):
                new_triple = (obj, s, subj)
                new_triples.add(new_triple)

        # Handle symmetric properties
        symmetric_props = set(ontology_graph.subjects(RDF.type, URIRef("http://www.w3.org/2002/07/owl#SymmetricProperty")))
        for prop in symmetric_props:
            for subj, _, obj in self.graph.triples((None, prop, None)):
                new_triples.add((obj, prop, subj))
        
        # Handle transitive properties
        transitive_props = set(ontology_graph.subjects(RDF.type, URIRef("http://www.w3.org/2002/07/owl#TransitiveProperty")))
        for prop in transitive_props:
            # Find all pairs connected by this property
            pairs = list(self.graph.triples((None, prop, None)))
            # For each pair, look for additional connections
            for s1, _, o1 in pairs:
                for s2, _, o2 in pairs:
                    if o1 == s2:  # Found a chain
                        new_triples.add((s1, prop, o2))

        # Add all new triples to the graph
        for s, p, o in new_triples:
            self.graph.add((s, p, o))
                


