import io
import pydotplus
import shutil
import subprocess
import warnings
import logging
import os
import sys
from rdflib import Graph, Namespace, RDF, RDFS, URIRef
from typing import List, Dict, Any, Set, Optional, Tuple, Type, Union
from urllib.error import HTTPError
from rdflib.tools.rdf2dot import rdf2dot
from bs4 import BeautifulSoup
from twin4build.utils.mkdir_in_root import mkdir_in_root
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
                pred_name = str(prop).split('#')[-1]
                
                # Get the domains (if any) for this property
                domains = list(self.graph.objects(prop, RDFS.domain))
                
                # Property is valid if it has no domain restrictions or if this class/parents are in its domain
                if not domains or any(str(domain) in self.parent_classes or domain == self.uri for domain in domains):
                    # Get the ranges for this property
                    ranges = list(self.graph.objects(prop, RDFS.range))
                    self._attributes[pred_name] = ranges
                        
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
    """Class to represent an ontology instance"""
    def __init__(self, uri: Union[str, URIRef], model: "SemanticModel"):
        self.uri = URIRef(uri) if isinstance(uri, str) else uri
        self.model = model
        self._types = None
        self._attributes = None
    
    @property
    def type(self) -> List[SemanticType]:
        """Get all types of this instance"""
        if self._types is None:
            # First check direct RDF.type assertions
            types = set(self.model.graph.objects(self.uri, RDF.type))
            
            # Then check for type through owl:sameAs relations
            same_as = set(self.model.graph.objects(self.uri, URIRef("http://www.w3.org/2002/07/owl#sameAs")))
            
            for same_as_uri in same_as:
                same_as_types = set(self.model.graph.objects(same_as_uri, RDF.type))
                types = types.union(same_as_types)
            
            # Convert all types to SemanticType objects
            self._types = set([self.model.get_type(t) for t in types])
        return self._types
    
    def get_object_attributes(self) -> Dict[str, Any]:
        """Return all attributes of this instance"""
        if self._attributes is None:
            self._attributes = {}
            for pred, obj in self.model.graph.predicate_objects(self.uri):
                if pred != RDF.type:
                    pred_name = str(pred).split('#')[-1]
                    if pred_name in self._attributes:
                        self._attributes[pred_name].append(self.model.get_instance(obj))
                    else:
                        self._attributes[pred_name] = [self.model.get_instance(obj)]
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
        return str(self.uri).split('#')[-1]

class SemanticModel:
    def __init__(self, 
                 rdf_file: str,
                 additional_namespaces: Optional[Dict[str, str]] = None, 
                 format: Optional[str] = None,
                 parse_namespaces=False,
                 verbose=False,
                 id: str="semantic_model",
                 dir_conf: List[str] = None):
        """
        Initialize the ontology model
        Args:
            rdf_file: Path or URL to the ontology file
            additional_namespaces: Optional additional namespace prefix-URI pairs
            format: Optional format specification ('xml', 'turtle', 'n3', 'nt', 'json-ld', etc.)
        """
        self.id = id

        if dir_conf is None:
            self.dir_conf = ["generated_files", "models", self.id]
        else:
            self.dir_conf = dir_conf

        if verbose:
            self._init(rdf_file=rdf_file,
                        additional_namespaces=additional_namespaces, 
                        format=format,
                        parse_namespaces=parse_namespaces)
        else:
            logging.disable(sys.maxsize) # https://stackoverflow.com/questions/2266646/how-to-disable-logging-on-the-standard-error-stream
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._init(rdf_file=rdf_file,
                        additional_namespaces=additional_namespaces, 
                        format=format,
                        parse_namespaces=parse_namespaces)
            # logging.disable(logging.NOTSET)
        

    def _init(self, 
            rdf_file: str,
            additional_namespaces: Optional[Dict[str, str]] = None, 
            format: Optional[str] = None,
            parse_namespaces=False):
        # Load and parse the RDF file
        self.graph = SemanticModel.get_graph(rdf_file, format)
        self.rdf_file = rdf_file
        self.format = format
        
        
        # Common namespaces
        self.RDF = RDF
        self.RDFS = RDFS
        
        # Extract namespaces from the graph
        self.namespaces = {}
        for prefix, uri in self.graph.namespaces():
            if prefix:  # Skip empty prefix
                self.namespaces[prefix.upper()] = Namespace(uri)
                setattr(self, prefix.upper(), self.namespaces[prefix.upper()])
        
        # Add any additional namespaces
        if additional_namespaces:
            for prefix, uri in additional_namespaces.items():
                if prefix not in self.namespaces:
                    self.namespaces[prefix] = Namespace(uri)
                    setattr(self, prefix, self.namespaces[prefix])
        self.missing_namespaces = []
        if parse_namespaces:
            for namespace in self.namespaces.values():
                try:
                    print(f"Parsing {namespace.title}")
                    self.graph.parse(namespace, format='turtle')
                except HTTPError as err:
                    print(f"The provided address does not exist (404).\n")
                    self.missing_namespaces.append(namespace)
                except Exception as err:
                    print(str(err) + "\n")
                    self.missing_namespaces.append(namespace)
        
        
        # Cache for instances
        self._instances = {}
        self._types = {}
        self._properties = {}



    def get_dir(self, folder_list: List[str] = [], filename: Optional[str] = None) -> Tuple[str, bool]:
        """
        Get the directory path for storing model-related files.

        Args:
            folder_list (List[str]): List of folder names to create.
            filename (Optional[str]): Name of the file to create.

        Returns:
            Tuple[str, bool]: The full path to the directory or file, and a boolean indicating if the file exists.
        """
        
        self.dir_conf.extend(folder_list)
        folder_list = self.dir_conf
        filename, isfile = mkdir_in_root(folder_list=folder_list, filename=filename)
        return filename, isfile


    @staticmethod
    def get_graph(rdf_file: str, format: Optional[str] = None) -> Graph:
        """Get the graph from an RDF file"""

        graph = Graph()
        # Determine format if not specified
        if format is None:
            # Get file extension
            if rdf_file.startswith('http'):
                # For URLs, get the file extension from the path
                ext = rdf_file.split('.')[-1].lower()
            else:
                ext = os.path.splitext(rdf_file)[1][1:].lower()
            
            # Map common extensions to formats
            format_map = {
                'ttl': 'turtle',
                'n3': 'n3',
                'nt': 'nt',
                'jsonld': 'json-ld',
                'json': 'json-ld',
                'xml': 'xml',
                'rdf': 'xml',
                'owl': 'xml'
            }
            format = format_map.get(ext, 'xml')  # Default to XML if unknown
        
        try:
            graph.parse(rdf_file, format=format)
        except Exception as e:
            print(f"Failed to parse with format {format}, trying alternatives...")
            # Try common formats if the specified one fails
            for alt_format in ['xml', 'turtle', 'n3', 'json-ld']:
                if alt_format != format:
                    try:
                        graph.parse(rdf_file, format=alt_format)
                        print(f"Successfully parsed with format: {alt_format}")
                        break
                    except:
                        continue
            else:
                raise ValueError(f"Could not parse file {rdf_file} with any known format")

        
        return graph
    
    def filter_graph(self, class_filter: Optional[Tuple] = None, predicate_filter: Optional[Tuple] = None, filter_rule: str = "OR") -> Graph:
        """Filter the graph based on class and predicate filters.
        The filtering is done using OR(class_filter) and OR(predicate_filter).
        
        Args:
            class_filter: List of class URIs to include (None = no class filtering)
            predicate_filter: List of predicates to include (None = no predicate filtering)
            
        Returns:
            Filtered graph
        """
        assert filter_rule in ["OR", "AND"], "Filter rule must be either OR or AND"
        new_graph = SemanticModel.get_graph(self.rdf_file, self.format)
        keep_triples = set()
        
        if class_filter is not None:
            instances = self.get_instances_of_type(class_filter)
            for s, p, o in self.graph.triples((None, None, None)):
                if self.get_instance(s) in instances or self.get_instance(o) in instances:
                    keep_triples.add((s, p, o))

        if predicate_filter is not None:
            keep_triples_new = set()
            if class_filter is not None:
                for s, p, o in keep_triples:
                    for predicate in predicate_filter:
                        if self.get_property(p).isproperty(predicate):
                            keep_triples_new.add((s, p, o))
                            break
            else:
                for s, p, o in self.graph.triples((None, None, None)):
                    for predicate in predicate_filter:
                        if self.get_property(p).isproperty(predicate):
                            keep_triples_new.add((s, p, o))
                            break
            keep_triples = keep_triples_new

        if class_filter is not None or predicate_filter is not None:
            for s, p, o in new_graph.triples((None, None, None)):
                if (s, p, o) not in keep_triples:
                    new_graph.remove((s, p, o))
        return new_graph
        
    def get_instance(self, uri: str) -> SemanticObject:
        """Get a specific instance by URI"""
        uri = URIRef(uri) if isinstance(uri, str) else uri
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
    
    def get_instances_of_type(self, class_uris: Tuple) -> List[SemanticObject]:
        """
        Get all instances that match any of the specified types (including subtypes)
        
        Args:
            class_uri: Single URI or tuple of URIs representing the types to match
            
        Returns:
            List of SemanticObject instances that match any of the specified types
        """
        # Convert single class_uri to tuple for consistent handling
        if not isinstance(class_uris, tuple):
            class_uris = (class_uris,)

        class_uris_new = []
        for class_uri in class_uris:
            if isinstance(class_uri, str):
                class_uris_new.append(URIRef(class_uri))
            elif isinstance(class_uri, SemanticType):
                class_uris_new.append(class_uri.uri)
            elif isinstance(class_uri, URIRef):
                class_uris_new.append(class_uri)
            else:
                raise ValueError(f"Invalid class URI: {class_uri}")
        class_uris = tuple(class_uris_new)
                
        instances = []
        processed_instances = set()  # To avoid duplicates
        
        # Process each type in the tuple
        for uri in class_uris:            
            # Get the class and all its subclasses
            subclasses = set([uri])
            for subclass in self.graph.transitive_subjects(RDFS.subClassOf, uri):
                subclasses.add(subclass)
                        
            # Get instances of the class and its subclasses
            for subclass in subclasses:
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
        return instances

    def visualize(self, class_filter=None, predicate_filter=None, filter_rule="OR"):
        """
        Visualize RDF graph with optional class and predicate filtering.
        The filter acts as an OR filter.
        
        Args:
            class_filter: List of class URIs to include (None = no class filtering)
            predicate_filter: List of predicates to include (None = no predicate filtering)
        """

        if class_filter is not None:
            if isinstance(class_filter, tuple)==False:
                class_filter = (class_filter,)

        if predicate_filter is not None:
            if isinstance(predicate_filter, tuple)==False:
                predicate_filter = (predicate_filter,)


        # Get a copy of the graph
        # graph = self.graph.copy()
        # graph = SemanticModel.get_graph(self.rdf_file, self.format)


        # Filter graph
        graph = self.filter_graph(class_filter, predicate_filter, filter_rule)

        stream = io.StringIO()
        print("RDF2DOT")
        rdf2dot(graph, stream)

        # print(f"Writing to file object_graph.dot")
        # with open('object_graph.dot', 'w') as fd:
            # stream.seek(0)
        #     shutil.copyfileobj(stream, fd)

        print("PYDOTPLUS")
        dg = pydotplus.graph_from_dot_data(stream.getvalue())
        print("PYDOTPLUS DONE")

        # Add class type to node labels
        for node in dg.get_nodes():
            if 'label' in node.obj_dict['attributes']:
                # print("======================================================")
                html_str = node.obj_dict['attributes']['label']
                soup = BeautifulSoup(html_str, 'html.parser')
                # print("BEFORE")
                # print(soup.prettify())

                soup.table.attrs.update({"border":"2", "width":"100%"})


                type_ = self.get_instance(soup.find('href')).type
                row = soup.find_all('tr')[1]
                col = row.find_all('td')[0]
                uri = col.string
                inst = self.get_instance(uri)
                type_ = inst.type

                z = {e for e in type_ if e.has_subclasses()==False}



                type_set = set(type_)

                if class_filter is not None:
                    class_filter_set = set([self.get_type(c) for c in class_filter])
                    if len(z)==0:
                        z = type_set.intersection(class_filter_set)

                z = {e.uri.fragment for e in z}

                if len(z)==0:
                    z = {"Unknown class"}

                # z = [str(s) for s in z]
                z = " | ".join(z)# data

                # print("z", z)
                # print("type", type(z))

                #Add as many td (data) you want.
                b = soup.new_tag('b', attrs={})
                b.string = z
                new_col = soup.new_tag('td', attrs={"bgcolor": "grey", "colspan": "2"})
                new_col.append(b)
                new_row = soup.new_tag("tr", attrs={"border": "1px solid black"})
                new_row.append(new_col)

                #Add whole 'tr'(row) to table.

                first_row = soup.find_all("tr")[0]
                first_row.insert_before(new_row)


                a = node.obj_dict['attributes']['label']


                node.obj_dict['attributes']['label'] = str(soup).replace("&lt;", "<").replace("&gt;", ">")
                # print("AFTER")
                # print(soup.prettify())

                # print("BEFORE STRING")
                # print(a)

                # print("AFTER STRING")
                # print(node.obj_dict['attributes']['label'])



        dot_filename = "object_graph.dot"
        dg.write(dot_filename)
        dirname,_ = self.get_dir(folder_list=["graphs", "temp"])

        # Delete all files in dirname
        for filename in os.listdir(dirname):
            file_path = os.path.join(dirname, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)


        ### ccomps ###
        dot_filename_ccomps = os.path.join(dirname, "object_graph_ccomps.dot")
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
                "-v",
                f"-o{semantic_model_png}",
                f"{dot_filename_gvpack}"]
        subprocess.run(args=args)

        print(f"Number of nodes: {len(dg.get_nodes())}")
        print(f"Number of edges: {len(dg.get_edges())}")
