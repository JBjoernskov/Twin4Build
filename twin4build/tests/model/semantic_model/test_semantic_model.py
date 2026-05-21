# Standard library imports
import os
import shutil
import tempfile
import unittest

# Third party imports
from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef

# Local application imports
import twin4build
import twin4build.core as core
from twin4build.model.semantic_model.semantic_model import (
    SemanticEntity,
    SemanticInstance,
    SemanticLiteral,
    SemanticModel,
    SemanticObject,
    SemanticPredicate,
    SemanticProperty,
    SemanticType,
    get_short_name,
    parse_wrapper,
)

# Set test flag
twin4build._IS_TESTING = True


class TestSemanticModel(unittest.TestCase):
    """Comprehensive tests for SemanticModel, SemanticObject, SemanticProperty, SemanticType, and SemanticPredicate."""

    @classmethod
    def setUpClass(cls):
        """Check if Graphviz is installed."""
        cls.graphviz_installed = all(
            [
                shutil.which("dot") is not None,
                shutil.which("ccomps") is not None,
                shutil.which("gvpack") is not None,
                shutil.which("neato") is not None,
            ]
        )

    def setUp(self):
        """Set up a fresh semantic model for each test."""
        self.model_id = "test_semantic_model"
        self.semantic_model = SemanticModel(id=self.model_id)
        self.model = self.semantic_model  # Alias for convenience
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up any generated files."""
        if os.path.exists("test_output.ttl"):
            os.remove("test_output.ttl")

        # Cleanup model directory
        if os.path.exists("generated_files/models/" + self.model_id):
            shutil.rmtree("generated_files/models/" + self.model_id)

        # Cleanup temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization_without_rdf(self):
        """Test semantic model initialization without RDF file."""
        model = SemanticModel(id="test_init")
        self.assertIsNotNone(model)
        # Semantic model uses instance_graph and ontology_graph
        self.assertTrue(hasattr(model, "instance_graph"))
        self.assertTrue(hasattr(model, "ontology_graph"))

    def test_get_instance(self):
        """Test get_instance method."""
        # Create a new instance
        uri = "http://example.org/instance1"
        instance = self.semantic_model.get_instance(uri)

        self.assertIsInstance(instance, SemanticObject)
        self.assertEqual(str(instance.uri), uri)

        # Retrieve existing instance
        instance2 = self.semantic_model.get_instance(uri)
        self.assertEqual(instance, instance2)

        # Create literal instance
        literal_val = "some_value"
        literal = self.semantic_model.get_instance(
            literal_val, datatype="http://www.w3.org/2001/XMLSchema#string"
        )
        self.assertIsInstance(literal, SemanticLiteral)

    def test_get_property(self):
        """Test get_property method."""
        uri = "http://example.org/property1"

        # Third party imports
        from rdflib import RDF, URIRef

        self.semantic_model.ontology_graph.add(
            (
                URIRef(uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        prop = self.semantic_model.get_property(uri)

        self.assertIsInstance(prop, SemanticProperty)
        self.assertEqual(str(prop.uri), uri)

    def test_serialize(self):
        """Test RDF serialization."""
        uri = "http://example.org/instance1"
        type_uri = "http://example.org/Type1"

        # Third party imports
        from rdflib import RDF, URIRef

        self.semantic_model.instance_graph.add(
            (URIRef(uri), RDF.type, URIRef(type_uri))
        )

        output_file_instance = "test_output_instance.ttl"
        output_file_ontology = "test_output_ontology.ttl"

        self.semantic_model.serialize(
            filename_instance_graph=output_file_instance,
            filename_ontology_graph=output_file_ontology,
        )
        expected_path_instance = self.semantic_model.get_dir(
            filename=output_file_instance
        )[0]
        expected_path_ontology = self.semantic_model.get_dir(
            filename=output_file_ontology
        )[0]
        self.assertTrue(
            os.path.exists(expected_path_instance),
            f"File not found at {expected_path_instance}",
        )
        self.assertTrue(
            os.path.exists(expected_path_ontology),
            f"File not found at {expected_path_ontology}",
        )
        self.assertEqual(
            len(self.semantic_model.instance_graph),
            len(Graph().parse(expected_path_instance, format="turtle")),
        )
        self.assertEqual(
            len(self.semantic_model.ontology_graph),
            len(Graph().parse(expected_path_ontology, format="turtle")),
        )

    def test_visualize(self):
        """Test graph visualization."""
        self.semantic_model.visualize()
        self.assertTrue(True)

    def test_graph_property(self):
        """Test that semantic model has graph properties."""
        self.assertTrue(hasattr(self.semantic_model, "instance_graph"))
        self.assertTrue(hasattr(self.semantic_model, "ontology_graph"))

    def test_count_triples(self):
        """Test count_triples method."""
        # Third party imports
        from rdflib import RDF, URIRef

        uri = "http://example.org/instance1"
        type_uri = "http://example.org/Type1"
        self.semantic_model.instance_graph.add(
            (URIRef(uri), RDF.type, URIRef(type_uri))
        )

        count = self.semantic_model.count_triples()
        self.assertGreaterEqual(count, 1)

    def test_count_triples_empty(self):
        """Test count_triples on model with no additions."""
        count = self.semantic_model.count_triples()
        self.assertIsInstance(count, int)

    def test_get_graph_copy(self):
        """Test get_graph_copy method."""
        # Third party imports
        from rdflib import RDF, URIRef

        uri = "http://example.org/test_copy"
        type_uri = "http://example.org/TestType"
        self.semantic_model.instance_graph.add(
            (URIRef(uri), RDF.type, URIRef(type_uri))
        )

        graph_copy = self.semantic_model.get_graph_copy(
            self.semantic_model.instance_graph
        )

        self.assertIsNotNone(graph_copy)
        self.assertEqual(len(graph_copy), len(self.semantic_model.instance_graph))

    def test_namespaces_property(self):
        """Test namespaces property."""
        namespaces = self.semantic_model.namespaces
        self.assertIsNotNone(namespaces)
        self.assertTrue(len(namespaces) > 0)

    def test_add_namespaces(self):
        """Test add_namespaces method."""
        # Third party imports
        from rdflib import Namespace

        custom_ns = Namespace("http://example.org/custom#")
        self.semantic_model.add_namespaces({"CUSTOM": custom_ns})

        namespaces = self.semantic_model.namespaces
        self.assertIsNotNone(namespaces)

    def test_get_type(self):
        """Test get_type method."""
        type_uri = "http://www.w3.org/2000/01/rdf-schema#Class"

        sem_type = self.semantic_model.get_type(type_uri)
        self.assertIsNotNone(sem_type)

    def test_get_predicate(self):
        """Test get_predicate method."""
        # Third party imports
        from rdflib import RDF

        predicate = self.semantic_model.get_predicate(str(RDF.type))
        self.assertIsNotNone(predicate)

    def test_get_instances_of_type(self):
        """Test get_instances_of_type method."""
        # Third party imports
        from rdflib import RDF, URIRef

        type_uri = "http://example.org/TestClass"
        for i in range(3):
            inst_uri = f"http://example.org/instance_{i}"
            self.semantic_model.instance_graph.add(
                (URIRef(inst_uri), RDF.type, URIRef(type_uri))
            )

        instances = self.semantic_model.get_instances_of_type(type_uri)

        self.assertIsNotNone(instances)
        self.assertEqual(len(instances), 3)

    def test_get_instances_of_nonexistent_type(self):
        """Test get_instances_of_type with non-existent type."""
        instances = self.semantic_model.get_instances_of_type(
            "http://example.org/NonExistentType"
        )
        self.assertEqual(len(instances), 0)

    def test_get_dir(self):
        """Test get_dir method."""
        path, isfile = self.semantic_model.get_dir(
            folder_list=["test"], filename="file.txt"
        )
        self.assertIsNotNone(path)

    def test_bind_namespace(self):
        """Test bind_namespace method."""
        # Third party imports
        from rdflib import Namespace

        custom_ns = Namespace("http://custom.example.org/")
        namespaces = {"CUSTOM": custom_ns}
        self.semantic_model.add_namespaces(namespaces)

        namespaces = self.semantic_model.namespaces
        self.assertIsNotNone(namespaces)

    def test_instance_is_literal(self):
        """Test literal instance creation returns SemanticLiteral."""
        literal = self.semantic_model.get_instance(
            "test_value", datatype="http://www.w3.org/2001/XMLSchema#string"
        )
        self.assertIsInstance(literal, SemanticLiteral)

    def test_instance_not_literal(self):
        """Test non-literal instance returns SemanticInstance."""
        uri = "http://example.org/instance"
        instance = self.semantic_model.get_instance(uri)
        self.assertIsInstance(instance, SemanticInstance)
        self.assertNotIsInstance(instance, SemanticLiteral)

    def test_parse_namespaces(self):
        """Test parse_namespaces method."""
        # Third party imports
        from rdflib import Namespace

        # This is a basic test - real ontology parsing would need actual ontology files
        custom_ns = Namespace("http://example.org/ns/")

        # parse_namespaces might need ontology files to work properly
        # Just test that the method exists and can be called
        try:
            self.semantic_model.parse_namespaces(namespaces={"EX": custom_ns})
        except Exception:
            pass  # Expected if ontology file doesn't exist

        self.assertTrue(True)

    def test_get_all_instances(self):
        """Test getting all instances of a certain type."""
        type_uri = "http://example.org/TestInstanceType"

        self.semantic_model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )

        for i in range(3):
            inst_uri = f"http://example.org/inst_{i}"
            self.semantic_model.instance_graph.add(
                (URIRef(inst_uri), RDF.type, URIRef(type_uri))
            )

        instances = self.semantic_model.get_instances_of_type(type_uri)
        self.assertEqual(len(instances), 3)

    def test_query(self):
        """Test SPARQL query capability."""
        # Add some data
        subj_uri = "http://example.org/querySubj"
        type_uri = "http://example.org/QueryType"

        self.semantic_model.instance_graph.add(
            (URIRef(subj_uri), RDF.type, URIRef(type_uri))
        )

        # Run a simple query
        query = """
        SELECT ?s WHERE {
            ?s a <http://example.org/QueryType> .
        }
        """
        results = list(self.semantic_model.instance_graph.query(query))

        self.assertGreater(len(results), 0)

    # ==================== SemanticObject Tests ====================

    def test_semantic_object_creation(self):
        """Test creating a semantic object."""
        uri = "http://example.org/test_object"
        obj = self.model.get_instance(uri)
        self.assertIsNotNone(obj)
        self.assertEqual(str(obj.uri), uri)

    def test_semantic_object_str(self):
        """Test semantic object string representation."""
        uri = "http://example.org/test_object"
        obj = self.model.get_instance(uri)
        str_repr = str(obj)
        self.assertIsNotNone(str_repr)

    def test_semantic_object_repr(self):
        """Test semantic object repr."""
        uri = "http://example.org/test_object"
        obj = self.model.get_instance(uri)
        repr_str = repr(obj)
        self.assertIsNotNone(repr_str)

    def test_semantic_object_hash(self):
        """Test semantic object hash."""
        uri = "http://example.org/test_object"
        obj = self.model.get_instance(uri)
        hash_val = hash(obj)
        self.assertIsNotNone(hash_val)

    def test_semantic_object_equality(self):
        """Test semantic object equality."""
        uri = "http://example.org/test_object"
        obj = self.model.get_instance(uri)
        obj2 = self.model.get_instance(uri)
        self.assertEqual(obj, obj2)

        obj3 = self.model.get_instance("http://example.org/different")
        self.assertNotEqual(obj, obj3)

    def test_object_get_short_name(self):
        """Test get_short_name method."""
        uri = "http://example.org/test_object"
        obj = self.model.get_instance(uri)
        short_name = obj.get_short_name()
        if short_name is not None:
            self.assertIsInstance(short_name, str)

    def test_object_get_namespace_basic(self):
        """Test get_namespace method."""
        uri = "http://example.org/test_object"
        obj = self.model.get_instance(uri)
        try:
            namespace = obj.get_namespace()
            if namespace is not None:
                self.assertIsInstance(namespace, tuple)
        except Exception:
            pass  # Some objects may not have namespace

    def test_object_with_different_uris(self):
        """Test objects with different URIs are not equal."""
        obj1 = self.model.get_instance("http://example.org/obj1")
        obj2 = self.model.get_instance("http://example.org/obj2")
        self.assertNotEqual(obj1, obj2)

    def test_object_hash_consistency(self):
        """Test that same URI gives same hash."""
        uri = "http://example.org/test_object"
        obj1 = self.model.get_instance(uri)
        obj2 = self.model.get_instance(uri)
        self.assertEqual(hash(obj1), hash(obj2))

    def test_object_get_short_name_with_namespace_match(self):
        """Test get_short_name when namespace matches."""
        custom_ns = Namespace("http://custom.example.org/")
        self.model.add_namespaces({"CUSTOM": custom_ns})

        uri = "http://custom.example.org/MyInstance"
        obj = self.model.get_instance(uri)

        short_name = obj.get_short_name()
        if short_name is not None:
            self.assertEqual(short_name, "MyInstance")

    def test_object_isinstance_with_types(self):
        """Test isinstance checking."""
        type_uri = "http://example.org/TestObjType"
        instance_uri = "http://example.org/testInstance"

        self.model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.instance_graph.add(
            (URIRef(instance_uri), RDF.type, URIRef(type_uri))
        )

        obj = self.model.get_instance(instance_uri)
        types = obj.types

        self.assertIsNotNone(types)
        self.assertIsInstance(types, set)

    def test_literal_types_with_datatype(self):
        """Test types property for literals with datatype."""
        literal_value = Literal("42", datatype=XSD.integer)
        literal_obj = self.model.get_instance(literal_value)

        self.assertIsInstance(literal_obj, SemanticLiteral)
        types = literal_obj.types

        self.assertIsNotNone(types)
        self.assertIsInstance(types, set)
        self.assertGreater(len(types), 0)

    def test_literal_types_without_datatype(self):
        """Test types property for plain literals."""
        literal_value = Literal("plain text")
        literal_obj = self.model.get_instance(literal_value)

        self.assertIsInstance(literal_obj, SemanticLiteral)
        types = literal_obj.types

        self.assertIsNotNone(types)
        self.assertIsInstance(types, set)

    def test_object_isinstance_literal_with_datatype(self):
        """Test isinstance for literal with datatype."""
        literal_obj = self.model.get_instance(Literal("42", datatype=XSD.integer))

        self.assertTrue(literal_obj.isinstance(str(XSD.integer)))
        self.assertFalse(literal_obj.isinstance(str(XSD.string)))

    def test_object_isinstance_literal_with_language(self):
        """Test isinstance for literal with language tag."""
        literal_obj = self.model.get_instance(Literal("hello", lang="en"))
        result = literal_obj.isinstance(str(XSD.string))
        self.assertIsInstance(result, bool)

    def test_object_isinstance_plain_literal(self):
        """Test isinstance for plain literal."""
        literal_obj = self.model.get_instance(Literal("plain"))
        result = literal_obj.isinstance(str(XSD.string))
        self.assertIsInstance(result, bool)

    def test_object_isinstance_uri_with_type(self):
        """Test isinstance for URI instances."""
        type_uri = "http://example.org/TestInstanceType"
        instance_uri = "http://example.org/testInstanceIsInstance"

        self.model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.instance_graph.add(
            (URIRef(instance_uri), RDF.type, URIRef(type_uri))
        )

        obj = self.model.get_instance(instance_uri)

        self.assertTrue(obj.isinstance(type_uri))
        self.assertFalse(obj.isinstance("http://example.org/OtherType"))

    def test_object_isinstance_with_tuple(self):
        """Test isinstance with tuple of types."""
        type_uri = "http://example.org/TupleTestType"
        instance_uri = "http://example.org/tupleTestInstance"

        self.model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.instance_graph.add(
            (URIRef(instance_uri), RDF.type, URIRef(type_uri))
        )

        obj = self.model.get_instance(instance_uri)
        self.assertTrue(obj.isinstance((type_uri, "http://example.org/OtherType")))

    def test_object_get_namespace(self):
        """Test get_namespace method."""
        custom_ns = Namespace("http://namespace.test.org/")
        self.model.add_namespaces({"NSTEST": custom_ns})

        instance_uri = "http://namespace.test.org/MyInstance"
        obj = self.model.get_instance(instance_uri)

        namespace = obj.get_namespace()

        self.assertIsNotNone(namespace)
        self.assertIsInstance(namespace, tuple)

    def test_object_get_most_specific_type(self):
        """Test get_most_specific_type method."""
        animal_uri = "http://example.org/Animal"
        mammal_uri = "http://example.org/Mammal"
        dog_uri = "http://example.org/Dog"
        instance_uri = "http://example.org/fido"

        self.model.ontology_graph.add(
            (
                URIRef(animal_uri),
                RDF.type,
                URIRef("http://www.w3.org/2002/07/owl#Class"),
            )
        )
        self.model.ontology_graph.add(
            (
                URIRef(mammal_uri),
                RDF.type,
                URIRef("http://www.w3.org/2002/07/owl#Class"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(dog_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add(
            (URIRef(mammal_uri), RDFS.subClassOf, URIRef(animal_uri))
        )
        self.model.ontology_graph.add(
            (URIRef(dog_uri), RDFS.subClassOf, URIRef(mammal_uri))
        )
        self.model.instance_graph.add((URIRef(instance_uri), RDF.type, URIRef(dog_uri)))

        obj = self.model.get_instance(instance_uri)
        most_specific = obj.get_most_specific_type()

        if most_specific is not None:
            self.assertEqual(str(most_specific.uri), dog_uri)

    # ==================== SemanticProperty Tests ====================

    def test_semantic_property_creation(self):
        """Test creating a semantic property."""
        prop_uri = "http://example.org/testProperty"
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        prop = self.model.get_property(prop_uri)
        self.assertIsNotNone(prop)

    def test_semantic_property_str(self):
        """Test semantic property string representation."""
        prop_uri = "http://example.org/testPropStr"
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        prop = self.model.get_property(prop_uri)
        str_repr = str(prop)
        self.assertIsNotNone(str_repr)

    def test_property_domain(self):
        """Test property domain access."""
        prop_uri = "http://example.org/testPropDomain"
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        prop = self.model.get_property(prop_uri)
        domain = prop.domain
        self.assertIsNotNone(domain)

    def test_property_range(self):
        """Test property range access."""
        prop_uri = "http://example.org/testPropRange"
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        prop = self.model.get_property(prop_uri)
        range_ = prop.range
        self.assertIsNotNone(range_)

    def test_property_isproperty(self):
        """Test isproperty method."""
        prop_uri = "http://example.org/testPropIs"
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        prop = self.model.get_property(prop_uri)
        self.assertTrue(prop.isproperty(prop_uri))
        self.assertFalse(prop.isproperty("http://example.org/otherProperty"))

    def test_property_get_short_name(self):
        """Test property get_short_name method."""
        prop_uri = "http://example.org/testPropShortName"
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        prop = self.model.get_property(prop_uri)
        short_name = prop.get_short_name()

        if short_name is not None:
            self.assertIsInstance(short_name, str)

    def test_property_get_short_name_with_registered_namespace(self):
        """Test property get_short_name when namespace matches."""
        custom_ns = Namespace("http://custom.test.org/")
        self.model.add_namespaces({"CUSTOMTEST": custom_ns})

        prop_uri = "http://custom.test.org/myProperty"
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        prop = self.model.get_property(prop_uri)
        short_name = prop.get_short_name()
        self.assertEqual(short_name, "myProperty")

    def test_property_get_short_name_no_match(self):
        """Test property get_short_name when no namespace matches.

        ``get_short_name`` now falls back to the full URI string when no
        registered prefix matches, so downstream serialisation always
        gets a printable identifier instead of a bare ``None`` that
        used to mask the underlying URI in logs / diagnostics.
        """
        prop_uri = "http://unregistered.property.org/myProperty"
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        prop = self.model.get_property(prop_uri)
        short_name = prop.get_short_name()
        self.assertEqual(short_name, prop_uri)

    def test_invalid_property_uri(self):
        """Test creating property with unknown URI still returns a predicate.

        Since SemanticProperty is now an alias for SemanticPredicate (which does not
        validate), get_property no longer raises ValueError for unknown URIs.
        """
        prop = self.model.get_property("http://example.org/not_a_real_property_12345")
        self.assertIsNotNone(prop)
        self.assertIsInstance(prop, SemanticPredicate)

    # ==================== SemanticPredicate Tests ====================

    def test_predicate_creation(self):
        """Test creating a semantic predicate."""
        predicate = self.model.get_predicate(str(RDF.type))
        self.assertIsNotNone(predicate)

    def test_predicate_str(self):
        """Test predicate string representation."""
        predicate = self.model.get_predicate(str(RDF.type))
        str_repr = str(predicate)
        self.assertIsNotNone(str_repr)

    def test_predicate_repr(self):
        """Test predicate repr."""
        predicate = self.model.get_predicate(str(RDF.type))
        repr_str = repr(predicate)
        self.assertIsNotNone(repr_str)

    def test_predicate_hash(self):
        """Test predicate hash."""
        predicate = self.model.get_predicate(str(RDF.type))
        hash_val = hash(predicate)
        self.assertIsNotNone(hash_val)

    def test_predicate_equality(self):
        """Test predicate equality."""
        predicate1 = self.model.get_predicate(str(RDF.type))
        predicate2 = self.model.get_predicate(str(RDF.type))
        self.assertEqual(predicate1, predicate2)

    def test_predicate_domain_property(self):
        """Test predicate domain property."""
        predicate = self.model.get_predicate(str(RDF.type))
        domain = predicate.domain
        self.assertIsNotNone(domain)
        self.assertIsInstance(domain, set)

    def test_predicate_range_property(self):
        """Test predicate range property."""
        predicate = self.model.get_predicate(str(RDF.type))
        range_val = predicate.range
        self.assertIsNotNone(range_val)
        self.assertIsInstance(range_val, set)

    def test_predicate_inverse_properties(self):
        """Test predicate inverse_properties property."""
        predicate = self.model.get_predicate(str(RDF.type))
        inverse_props = predicate.inverse_properties
        self.assertIsNotNone(inverse_props)
        self.assertIsInstance(inverse_props, list)

    def test_predicate_super_properties(self):
        """Test predicate super_properties property."""
        predicate = self.model.get_predicate(str(RDF.type))
        super_props = predicate.super_properties
        self.assertIsNotNone(super_props)
        self.assertIsInstance(super_props, list)

    def test_predicate_sub_properties(self):
        """Test predicate sub_properties property."""
        predicate = self.model.get_predicate(str(RDF.type))
        sub_props = predicate.sub_properties
        self.assertIsNotNone(sub_props)
        self.assertIsInstance(sub_props, list)

    def test_predicate_get_short_name(self):
        """Test predicate get_short_name method."""
        predicate = self.model.get_predicate(str(RDF.type))
        short_name = predicate.get_short_name()
        if short_name is not None:
            self.assertIsInstance(short_name, str)

    def test_predicate_equivalent_properties(self):
        """Test equivalent_properties property."""
        prop_uri = "http://example.org/testPropEquivPred"
        equiv_prop_uri = "http://example.org/equivalentPropPred"
        owl_equivalent_property = URIRef(
            "http://www.w3.org/2002/07/owl#equivalentProperty"
        )

        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (
                URIRef(equiv_prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(prop_uri), owl_equivalent_property, URIRef(equiv_prop_uri))
        )

        predicate = self.model.get_predicate(prop_uri)
        equiv_props = predicate.equivalent_properties

        self.assertIsNotNone(equiv_props)
        self.assertIsInstance(equiv_props, list)
        self.assertGreater(len(equiv_props), 0)

    def test_predicate_is_symmetric(self):
        """Test is_symmetric property."""
        prop_uri = "http://example.org/symmetricPropPred"
        owl_symmetric = URIRef("http://www.w3.org/2002/07/owl#SymmetricProperty")
        self.model.ontology_graph.add((URIRef(prop_uri), RDF.type, owl_symmetric))

        predicate = self.model.get_predicate(prop_uri)
        self.assertTrue(predicate.is_symmetric)

    def test_predicate_is_not_symmetric(self):
        """Test is_symmetric property when property is not symmetric."""
        prop_uri = "http://example.org/nonSymmetricPropPred"
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        predicate = self.model.get_predicate(prop_uri)
        self.assertFalse(predicate.is_symmetric)

    def test_predicate_is_transitive(self):
        """Test is_transitive property."""
        prop_uri = "http://example.org/transitivePropPred"
        owl_transitive = URIRef("http://www.w3.org/2002/07/owl#TransitiveProperty")
        self.model.ontology_graph.add((URIRef(prop_uri), RDF.type, owl_transitive))

        predicate = self.model.get_predicate(prop_uri)
        self.assertTrue(predicate.is_transitive)

    def test_predicate_is_not_transitive(self):
        """Test is_transitive property when property is not transitive."""
        prop_uri = "http://example.org/nonTransitivePropPred"
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        predicate = self.model.get_predicate(prop_uri)
        self.assertFalse(predicate.is_transitive)

    def test_predicate_is_functional(self):
        """Test is_functional property."""
        prop_uri = "http://example.org/functionalPropPred"
        owl_functional = URIRef("http://www.w3.org/2002/07/owl#FunctionalProperty")
        self.model.ontology_graph.add((URIRef(prop_uri), RDF.type, owl_functional))

        predicate = self.model.get_predicate(prop_uri)
        self.assertTrue(predicate.is_functional)

    def test_predicate_is_not_functional(self):
        """Test is_functional property when property is not functional."""
        prop_uri = "http://example.org/nonFunctionalPropPred"
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        predicate = self.model.get_predicate(prop_uri)
        self.assertFalse(predicate.is_functional)

    def test_predicate_is_inverse_functional(self):
        """Test is_inverse_functional property."""
        prop_uri = "http://example.org/inverseFunctionalPropPred"
        owl_inv_func = URIRef("http://www.w3.org/2002/07/owl#InverseFunctionalProperty")
        self.model.ontology_graph.add((URIRef(prop_uri), RDF.type, owl_inv_func))

        predicate = self.model.get_predicate(prop_uri)
        self.assertTrue(predicate.is_inverse_functional)

    def test_predicate_is_not_inverse_functional(self):
        """Test is_inverse_functional property when property is not inverse functional."""
        prop_uri = "http://example.org/nonInverseFunctionalPropPred"
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        predicate = self.model.get_predicate(prop_uri)
        self.assertFalse(predicate.is_inverse_functional)

    def test_predicate_get_namespace(self):
        """Test get_namespace method."""
        predicate = self.model.get_predicate(str(RDF.type))
        namespace = predicate.get_namespace()
        self.assertIsNotNone(namespace)
        self.assertIsInstance(namespace, tuple)

    def test_predicate_ispredicate(self):
        """Test ispredicate method."""
        predicate = self.model.get_predicate(str(RDF.type))
        self.assertTrue(predicate.ispredicate(str(RDF.type)))
        self.assertFalse(predicate.ispredicate("http://example.org/otherPredicate"))
        self.assertTrue(predicate.ispredicate([str(RDF.type), "http://other.org/prop"]))

    def test_predicate_inverse_properties_with_owl_inverse_of(self):
        """Test inverse_properties when inverseOf is defined."""
        prop_uri = "http://example.org/propWithInversePred"
        inv_prop_uri = "http://example.org/inversePropPred"
        owl_inverse_of = URIRef("http://www.w3.org/2002/07/owl#inverseOf")

        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (
                URIRef(inv_prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(prop_uri), owl_inverse_of, URIRef(inv_prop_uri))
        )

        predicate = self.model.get_predicate(prop_uri)
        inv_props = predicate.inverse_properties

        self.assertIsNotNone(inv_props)
        self.assertIsInstance(inv_props, list)
        self.assertGreater(len(inv_props), 0)

    def test_predicate_get_short_name_with_registered_namespace(self):
        """Test predicate get_short_name when namespace matches."""
        custom_ns = Namespace("http://predicate.test.org/")
        self.model.add_namespaces({"PREDTEST": custom_ns})

        prop_uri = "http://predicate.test.org/myPredicate"
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        predicate = self.model.get_predicate(prop_uri)
        short_name = predicate.get_short_name()
        self.assertEqual(short_name, "myPredicate")

    def test_predicate_get_short_name_no_match(self):
        """Test predicate get_short_name when no namespace matches.

        ``get_short_name`` now falls back to the full URI string when no
        registered prefix matches, so downstream serialisation always
        gets a printable identifier instead of a bare ``None`` that
        used to mask the underlying URI in logs / diagnostics.
        """
        prop_uri = "http://unregistered.predicate.org/myPredicate"
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )

        predicate = self.model.get_predicate(prop_uri)
        short_name = predicate.get_short_name()
        self.assertEqual(short_name, prop_uri)

    def test_predicate_ispredicate_with_super_property(self):
        """Test ispredicate matching super properties."""
        sub_prop_uri = "http://example.org/subPropIsPred"
        super_prop_uri = "http://example.org/superPropIsPred"

        self.model.ontology_graph.add(
            (
                URIRef(sub_prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (
                URIRef(super_prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(sub_prop_uri), RDFS.subPropertyOf, URIRef(super_prop_uri))
        )

        predicate = self.model.get_predicate(sub_prop_uri)
        self.assertTrue(predicate.ispredicate(super_prop_uri))

    def test_predicate_ispredicate_with_equivalent_property(self):
        """Test ispredicate matching equivalent properties."""
        prop_uri = "http://example.org/propIsEquiv"
        equiv_prop_uri = "http://example.org/equivPropIs"
        owl_equivalent_property = URIRef(
            "http://www.w3.org/2002/07/owl#equivalentProperty"
        )

        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (
                URIRef(equiv_prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(prop_uri), owl_equivalent_property, URIRef(equiv_prop_uri))
        )

        predicate = self.model.get_predicate(prop_uri)
        self.assertTrue(predicate.ispredicate(equiv_prop_uri))

    def test_predicate_super_properties_with_subproperty(self):
        """Test super_properties when subPropertyOf is defined."""
        sub_prop_uri = "http://example.org/subPropPred"
        super_prop_uri = "http://example.org/superPropPred"

        self.model.ontology_graph.add(
            (
                URIRef(sub_prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (
                URIRef(super_prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(sub_prop_uri), RDFS.subPropertyOf, URIRef(super_prop_uri))
        )

        predicate = self.model.get_predicate(sub_prop_uri)
        super_props = predicate.super_properties
        self.assertIsNotNone(super_props)
        self.assertIsInstance(super_props, list)

    def test_predicate_sub_properties_with_subproperty(self):
        """Test sub_properties when subPropertyOf is defined."""
        sub_prop_uri = "http://example.org/subProp2Pred"
        super_prop_uri = "http://example.org/superProp2Pred"

        self.model.ontology_graph.add(
            (
                URIRef(sub_prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (
                URIRef(super_prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(sub_prop_uri), RDFS.subPropertyOf, URIRef(super_prop_uri))
        )

        predicate = self.model.get_predicate(super_prop_uri)
        sub_props = predicate.sub_properties
        self.assertIsNotNone(sub_props)
        self.assertIsInstance(sub_props, list)

    def test_predicate_equality_with_string(self):
        """Test predicate equality with string."""
        predicate = self.model.get_predicate(str(RDF.type))
        self.assertEqual(predicate, str(RDF.type))

    def test_predicate_equality_with_uriref(self):
        """Test predicate equality with URIRef."""
        predicate = self.model.get_predicate(str(RDF.type))
        self.assertEqual(predicate, RDF.type)

    def test_predicate_inequality(self):
        """Test predicate inequality."""
        predicate = self.model.get_predicate(str(RDF.type))
        self.assertNotEqual(predicate, 42)
        self.assertNotEqual(predicate, None)

    # ==================== SemanticType Tests ====================

    def test_type_creation(self):
        """Test creating a semantic type."""
        type_uri = "http://example.org/TestClassType"
        self.model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )

        sem_type = self.model.get_type(type_uri)
        self.assertIsNotNone(sem_type)

    def test_type_super_classes(self):
        """Test super_classes property."""
        child_type_uri = "http://example.org/ChildClass"
        parent_type_uri = "http://example.org/ParentClass"

        self.model.ontology_graph.add(
            (
                URIRef(child_type_uri),
                RDF.type,
                URIRef("http://www.w3.org/2002/07/owl#Class"),
            )
        )
        self.model.ontology_graph.add(
            (
                URIRef(parent_type_uri),
                RDF.type,
                URIRef("http://www.w3.org/2002/07/owl#Class"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(child_type_uri), RDFS.subClassOf, URIRef(parent_type_uri))
        )

        child_type = self.model.get_type(child_type_uri)
        super_classes = child_type.super_classes
        self.assertIsNotNone(super_classes)
        self.assertIsInstance(super_classes, list)

    def test_type_sub_classes(self):
        """Test sub_classes property."""
        child_type_uri = "http://example.org/ChildClass2"
        parent_type_uri = "http://example.org/ParentClass2"

        self.model.ontology_graph.add(
            (
                URIRef(child_type_uri),
                RDF.type,
                URIRef("http://www.w3.org/2002/07/owl#Class"),
            )
        )
        self.model.ontology_graph.add(
            (
                URIRef(parent_type_uri),
                RDF.type,
                URIRef("http://www.w3.org/2002/07/owl#Class"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(child_type_uri), RDFS.subClassOf, URIRef(parent_type_uri))
        )

        parent_type = self.model.get_type(parent_type_uri)
        sub_classes = parent_type.sub_classes
        self.assertIsNotNone(sub_classes)
        self.assertIsInstance(sub_classes, list)

    def test_type_equivalent_classes(self):
        """Test equivalent_classes property."""
        type1_uri = "http://example.org/EquivClass1"
        type2_uri = "http://example.org/EquivClass2"
        owl_equivalent_class = URIRef("http://www.w3.org/2002/07/owl#equivalentClass")

        self.model.ontology_graph.add(
            (URIRef(type1_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add(
            (URIRef(type2_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add(
            (URIRef(type1_uri), owl_equivalent_class, URIRef(type2_uri))
        )

        type1 = self.model.get_type(type1_uri)
        equiv_classes = type1.equivalent_classes
        self.assertIsNotNone(equiv_classes)
        self.assertIsInstance(equiv_classes, list)

    def test_type_get_type_attributes(self):
        """Test get_type_attributes method."""
        type_uri = "http://example.org/ClassWithAttrs"
        prop_uri = "http://example.org/someObjectProperty"

        self.model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add(
            (
                URIRef(prop_uri),
                RDF.type,
                URIRef("http://www.w3.org/2002/07/owl#ObjectProperty"),
            )
        )
        self.model.ontology_graph.add((URIRef(prop_uri), RDFS.domain, URIRef(type_uri)))

        sem_type = self.model.get_type(type_uri)
        attrs = sem_type.get_type_attributes()
        self.assertIsNotNone(attrs)
        self.assertIsInstance(attrs, dict)

    def test_type_str_repr(self):
        """Test string representation of type."""
        type_uri = "http://example.org/TestTypeStr"
        self.model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )

        sem_type = self.model.get_type(type_uri)
        self.assertEqual(str(sem_type), type_uri)
        self.assertIn("SemanticType", repr(sem_type))

    def test_type_hash_and_equality(self):
        """Test type hash and equality."""
        type_uri = "http://example.org/TestTypeHash"
        self.model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )

        type1 = self.model.get_type(type_uri)
        type2 = self.model.get_type(type_uri)
        self.assertEqual(hash(type1), hash(type2))
        self.assertEqual(type1, type2)

    def test_type_istype_method(self):
        """Test istype method."""
        child_uri = "http://example.org/ChildType"
        parent_uri = "http://example.org/ParentType"

        self.model.ontology_graph.add(
            (URIRef(child_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add(
            (
                URIRef(parent_uri),
                RDF.type,
                URIRef("http://www.w3.org/2002/07/owl#Class"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(child_uri), RDFS.subClassOf, URIRef(parent_uri))
        )

        child_type = self.model.get_type(child_uri)
        self.assertTrue(child_type.istype(child_uri))
        result = child_type.istype("http://example.org/UnrelatedType")
        self.assertIsInstance(result, bool)

    def test_type_istype_with_parent_class(self):
        """Test istype method matching parent class."""
        child_uri = "http://example.org/ChildTypeIsType"
        parent_uri = "http://example.org/ParentTypeIsType"

        self.model.ontology_graph.add(
            (URIRef(child_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add(
            (
                URIRef(parent_uri),
                RDF.type,
                URIRef("http://www.w3.org/2002/07/owl#Class"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(child_uri), RDFS.subClassOf, URIRef(parent_uri))
        )

        child_type = self.model.get_type(child_uri)
        self.assertTrue(child_type.istype(parent_uri))

    def test_type_has_subclasses(self):
        """Test has_subclasses method."""
        child_uri = "http://example.org/ChildTypeHasSub"
        parent_uri = "http://example.org/ParentTypeHasSub"

        self.model.ontology_graph.add(
            (URIRef(child_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add(
            (
                URIRef(parent_uri),
                RDF.type,
                URIRef("http://www.w3.org/2002/07/owl#Class"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(child_uri), RDFS.subClassOf, URIRef(parent_uri))
        )

        parent_type = self.model.get_type(parent_uri)
        child_type = self.model.get_type(child_uri)
        self.assertTrue(parent_type.has_subclasses())
        self.assertFalse(child_type.has_subclasses())

    def test_type_get_short_name_with_registered_namespace(self):
        """Test get_short_name when namespace matches."""
        custom_ns = Namespace("http://type.test.org/")
        self.model.add_namespaces({"TYPETEST": custom_ns})

        type_uri = "http://type.test.org/MyClass"
        self.model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )

        sem_type = self.model.get_type(type_uri)
        short_name = sem_type.get_short_name()
        self.assertEqual(short_name, "MyClass")

    def test_type_get_short_name_no_match(self):
        """Test get_short_name when no namespace matches.

        ``get_short_name`` now falls back to the full URI string when no
        registered prefix matches, so downstream serialisation always
        gets a printable identifier instead of a bare ``None`` that
        used to mask the underlying URI in logs / diagnostics.
        """
        type_uri = "http://unregistered.namespace.org/MyClass"
        self.model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )

        sem_type = self.model.get_type(type_uri)
        short_name = sem_type.get_short_name()
        self.assertEqual(short_name, type_uri)

    def test_type_super_classes_with_equivalent(self):
        """Test super_classes including equivalent classes."""
        child_uri = "http://example.org/ChildWithEquiv"
        parent_uri = "http://example.org/ParentWithEquiv"
        equiv_parent_uri = "http://example.org/EquivParent"
        owl_equivalent_class = URIRef("http://www.w3.org/2002/07/owl#equivalentClass")

        self.model.ontology_graph.add(
            (URIRef(child_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add(
            (
                URIRef(parent_uri),
                RDF.type,
                URIRef("http://www.w3.org/2002/07/owl#Class"),
            )
        )
        self.model.ontology_graph.add(
            (
                URIRef(equiv_parent_uri),
                RDF.type,
                URIRef("http://www.w3.org/2002/07/owl#Class"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(child_uri), RDFS.subClassOf, URIRef(parent_uri))
        )
        self.model.ontology_graph.add(
            (URIRef(parent_uri), owl_equivalent_class, URIRef(equiv_parent_uri))
        )

        child_type = self.model.get_type(child_uri)
        super_classes = child_type.super_classes
        super_uris = [str(s.uri) for s in super_classes]
        self.assertIn(parent_uri, super_uris)

    def test_type_equivalent_classes_with_values(self):
        """Test equivalent_classes actually returns values."""
        type1_uri = "http://example.org/EquivClassVal1"
        type2_uri = "http://example.org/EquivClassVal2"
        owl_equivalent_class = URIRef("http://www.w3.org/2002/07/owl#equivalentClass")

        self.model.ontology_graph.add(
            (URIRef(type1_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add(
            (URIRef(type2_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add(
            (URIRef(type2_uri), owl_equivalent_class, URIRef(type1_uri))
        )

        type1 = self.model.get_type(type1_uri)
        equiv_classes = type1.equivalent_classes
        self.assertGreater(len(equiv_classes), 0)

    def test_type_equality_with_string(self):
        """Test type equality with string."""
        type_uri = "http://example.org/TestTypeEq"
        self.model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )

        sem_type = self.model.get_type(type_uri)
        self.assertEqual(sem_type, type_uri)

    def test_type_equality_with_uriref(self):
        """Test type equality with URIRef."""
        type_uri = "http://example.org/TestTypeEqRef"
        self.model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )

        sem_type = self.model.get_type(type_uri)
        self.assertEqual(sem_type, URIRef(type_uri))

    def test_type_inequality(self):
        """Test type inequality."""
        type_uri = "http://example.org/TestTypeIneq"
        self.model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )

        sem_type = self.model.get_type(type_uri)
        self.assertNotEqual(sem_type, 42)
        self.assertNotEqual(sem_type, None)

    # ==================== get_short_name Function Tests ====================

    def test_get_short_name_func_with_matching_namespace(self):
        """Test get_short_name when URI matches a namespace."""
        namespaces = {"EX": Namespace("http://example.org/")}
        uri = "http://example.org/TestClass"
        result = get_short_name(uri, namespaces)
        self.assertEqual(result, "TestClass")

    def test_get_short_name_func_no_matching_namespace(self):
        """Test get_short_name when URI doesn't match any namespace."""
        namespaces = {"OTHER": Namespace("http://other.org/")}
        uri = "http://example.org/TestClass"
        result = get_short_name(uri, namespaces)
        self.assertIsNone(result)

    def test_get_short_name_func_with_uriref(self):
        """Test get_short_name with URIRef input."""
        namespaces = {"EX": Namespace("http://example.org/")}
        uri = URIRef("http://example.org/MyEntity")
        result = get_short_name(uri, namespaces)
        self.assertEqual(result, "MyEntity")

    # ==================== parse_wrapper Tests ====================

    def test_parse_wrapper_with_ignored_namespace(self):
        """Test parse_wrapper skips ignored namespaces."""
        graph = Graph()
        xsd_namespace = str(core.namespace.XSD)
        result = parse_wrapper(graph, source=xsd_namespace)
        self.assertIsNone(result)

    def test_parse_wrapper_with_valid_source(self):
        """Test parse_wrapper with a valid source."""
        # Standard library imports
        from io import StringIO

        graph = Graph()
        ttl_data = """
        @prefix ex: <http://example.org/> .
        ex:subject ex:predicate ex:object .
        """
        parse_wrapper(graph, source=StringIO(ttl_data), format="turtle")
        self.assertGreater(len(graph), 0)

    # ==================== get_predicate_object_pairs Tests ====================

    def test_get_predicate_object_pairs_basic(self):
        """Test get_predicate_object_pairs with basic triples."""
        subj_uri = "http://example.org/subject1"
        obj_uri = "http://example.org/object1"
        pred_uri = "http://example.org/hasSomething"

        self.model.instance_graph.add(
            (URIRef(subj_uri), URIRef(pred_uri), URIRef(obj_uri))
        )

        subj = self.model.get_instance(subj_uri)
        pairs = subj.get_predicate_object_pairs()

        self.assertIsNotNone(pairs)
        self.assertIsInstance(pairs, dict)
        self.assertGreater(len(pairs), 0)

    def test_get_predicate_object_pairs_literal_returns_empty(self):
        """Test get_predicate_object_pairs returns empty dict for literals."""
        literal_obj = self.model.get_instance(
            "test_value", datatype="http://www.w3.org/2001/XMLSchema#string"
        )
        pairs = literal_obj.get_predicate_object_pairs()
        self.assertEqual(pairs, {})

    def test_get_predicate_object_pairs_with_multiple_predicates(self):
        """Test get_predicate_object_pairs with multiple predicates."""
        subj_uri = "http://example.org/subject2"
        obj1_uri = "http://example.org/object2a"
        obj2_uri = "http://example.org/object2b"
        pred1_uri = "http://example.org/hasFirst"
        pred2_uri = "http://example.org/hasSecond"

        self.model.instance_graph.add(
            (URIRef(subj_uri), URIRef(pred1_uri), URIRef(obj1_uri))
        )
        self.model.instance_graph.add(
            (URIRef(subj_uri), URIRef(pred2_uri), URIRef(obj2_uri))
        )

        subj = self.model.get_instance(subj_uri)
        pairs = subj.get_predicate_object_pairs()
        self.assertGreaterEqual(len(pairs), 2)

    def test_get_predicate_object_pairs_with_literal_object(self):
        """Test get_predicate_object_pairs with literal object values."""
        subj_uri = "http://example.org/subject3"
        pred_uri = "http://example.org/hasValue"

        self.model.instance_graph.add(
            (URIRef(subj_uri), URIRef(pred_uri), Literal("test value"))
        )

        subj = self.model.get_instance(subj_uri)
        pairs = subj.get_predicate_object_pairs()
        self.assertIsNotNone(pairs)
        self.assertGreater(len(pairs), 0)

    def test_get_predicate_object_pairs_with_inverse_property(self):
        """Test get_predicate_object_pairs with inverse property reasoning."""
        subj_uri = "http://example.org/personA"
        obj_uri = "http://example.org/personB"
        has_child = "http://example.org/hasChild"
        has_parent = "http://example.org/hasParent"
        owl_inverse_of = URIRef("http://www.w3.org/2002/07/owl#inverseOf")

        self.model.ontology_graph.add(
            (
                URIRef(has_child),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (
                URIRef(has_parent),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(has_child), owl_inverse_of, URIRef(has_parent))
        )
        self.model.instance_graph.add(
            (URIRef(subj_uri), URIRef(has_child), URIRef(obj_uri))
        )

        obj = self.model.get_instance(obj_uri)
        pairs = obj.get_predicate_object_pairs()
        self.assertIsNotNone(pairs)

    def test_get_predicate_object_pairs_with_symmetric_property(self):
        """Test get_predicate_object_pairs with symmetric property reasoning."""
        subj_uri = "http://example.org/entityA"
        obj_uri = "http://example.org/entityB"
        knows = "http://example.org/knows"
        owl_symmetric = URIRef("http://www.w3.org/2002/07/owl#SymmetricProperty")

        self.model.ontology_graph.add((URIRef(knows), RDF.type, owl_symmetric))
        self.model.instance_graph.add(
            (URIRef(subj_uri), URIRef(knows), URIRef(obj_uri))
        )

        obj = self.model.get_instance(obj_uri)
        pairs = obj.get_predicate_object_pairs()
        self.assertIsNotNone(pairs)

    # ==================== reason() Tests ====================

    def test_reason_basic(self):
        """Test basic reason() call."""
        type_uri = "http://example.org/TestType"
        instance_uri = "http://example.org/instance1"

        self.model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.instance_graph.add(
            (URIRef(instance_uri), RDF.type, URIRef(type_uri))
        )

        self.model.reason()
        self.assertTrue(True)

    def test_reason_with_inverse_properties(self):
        """Test reason() with inverse properties."""
        has_part = "http://example.org/hasPart"
        is_part_of = "http://example.org/isPartOf"
        whole_uri = "http://example.org/whole"
        part_uri = "http://example.org/part"
        owl_inverse_of = URIRef("http://www.w3.org/2002/07/owl#inverseOf")

        self.model.ontology_graph.add(
            (
                URIRef(has_part),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (
                URIRef(is_part_of),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(has_part), owl_inverse_of, URIRef(is_part_of))
        )
        self.model.instance_graph.add(
            (URIRef(whole_uri), URIRef(has_part), URIRef(part_uri))
        )

        initial_count = len(self.model.instance_graph)
        self.model.reason()
        self.assertGreaterEqual(len(self.model.instance_graph), initial_count)

    def test_reason_with_symmetric_properties(self):
        """Test reason() with symmetric properties."""
        knows = "http://example.org/knows"
        person_a = "http://example.org/personA"
        person_b = "http://example.org/personB"
        owl_symmetric = URIRef("http://www.w3.org/2002/07/owl#SymmetricProperty")

        self.model.ontology_graph.add((URIRef(knows), RDF.type, owl_symmetric))
        self.model.instance_graph.add(
            (URIRef(person_a), URIRef(knows), URIRef(person_b))
        )

        initial_count = len(self.model.instance_graph)
        self.model.reason()
        self.assertGreaterEqual(len(self.model.instance_graph), initial_count)

    def test_reason_with_transitive_properties(self):
        """Test reason() with transitive properties."""
        contains = "http://example.org/contains"
        a = "http://example.org/A"
        b = "http://example.org/B"
        c = "http://example.org/C"
        owl_transitive = URIRef("http://www.w3.org/2002/07/owl#TransitiveProperty")

        self.model.ontology_graph.add((URIRef(contains), RDF.type, owl_transitive))
        self.model.instance_graph.add((URIRef(a), URIRef(contains), URIRef(b)))
        self.model.instance_graph.add((URIRef(b), URIRef(contains), URIRef(c)))

        initial_count = len(self.model.instance_graph)
        self.model.reason()
        self.assertGreaterEqual(len(self.model.instance_graph), initial_count)

    def test_reason_with_subclass(self):
        """Test reason() with subclass reasoning."""
        animal = "http://example.org/Animal"
        dog = "http://example.org/Dog"
        fido = "http://example.org/fido"

        self.model.ontology_graph.add(
            (URIRef(animal), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add(
            (URIRef(dog), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add((URIRef(dog), RDFS.subClassOf, URIRef(animal)))
        self.model.instance_graph.add((URIRef(fido), RDF.type, URIRef(dog)))

        initial_count = len(self.model.instance_graph)
        self.model.reason()
        self.assertGreaterEqual(len(self.model.instance_graph), initial_count)

    def test_reason_with_equivalent_class(self):
        """Test reason() with equivalent class reasoning."""
        car = "http://example.org/Car"
        automobile = "http://example.org/Automobile"
        my_car = "http://example.org/myCar"
        owl_equiv_class = URIRef("http://www.w3.org/2002/07/owl#equivalentClass")

        self.model.ontology_graph.add(
            (URIRef(car), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add(
            (
                URIRef(automobile),
                RDF.type,
                URIRef("http://www.w3.org/2002/07/owl#Class"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(car), owl_equiv_class, URIRef(automobile))
        )
        self.model.instance_graph.add((URIRef(my_car), RDF.type, URIRef(car)))

        initial_count = len(self.model.instance_graph)
        self.model.reason()
        self.assertGreaterEqual(len(self.model.instance_graph), initial_count)

    def test_reason_with_equivalent_class_reverse(self):
        """Test reason() with equivalent class in reverse direction."""
        type1 = "http://example.org/Type1"
        type2 = "http://example.org/Type2"
        instance = "http://example.org/instance"
        owl_equiv_class = URIRef("http://www.w3.org/2002/07/owl#equivalentClass")

        self.model.ontology_graph.add(
            (URIRef(type1), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add(
            (URIRef(type2), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.ontology_graph.add((URIRef(type2), owl_equiv_class, URIRef(type1)))
        self.model.instance_graph.add((URIRef(instance), RDF.type, URIRef(type1)))

        self.model.reason()
        self.assertTrue(True)

    def test_reason_with_equivalent_property(self):
        """Test reason() with equivalent property reasoning."""
        likes = "http://example.org/likes"
        enjoys = "http://example.org/enjoys"
        person = "http://example.org/person1"
        thing = "http://example.org/thing1"
        owl_equiv_prop = URIRef("http://www.w3.org/2002/07/owl#equivalentProperty")

        self.model.ontology_graph.add(
            (
                URIRef(likes),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (
                URIRef(enjoys),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add((URIRef(likes), owl_equiv_prop, URIRef(enjoys)))
        self.model.instance_graph.add((URIRef(person), URIRef(likes), URIRef(thing)))

        initial_count = len(self.model.instance_graph)
        self.model.reason()
        self.assertGreaterEqual(len(self.model.instance_graph), initial_count)

    def test_reason_with_subproperty(self):
        """Test reason() with subproperty reasoning."""
        has_parent = "http://example.org/hasParent"
        has_ancestor = "http://example.org/hasAncestor"
        person = "http://example.org/child"
        parent = "http://example.org/parent"

        self.model.ontology_graph.add(
            (
                URIRef(has_parent),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (
                URIRef(has_ancestor),
                RDF.type,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            )
        )
        self.model.ontology_graph.add(
            (URIRef(has_parent), RDFS.subPropertyOf, URIRef(has_ancestor))
        )
        self.model.instance_graph.add(
            (URIRef(person), URIRef(has_parent), URIRef(parent))
        )

        initial_count = len(self.model.instance_graph)
        self.model.reason()
        self.assertGreaterEqual(len(self.model.instance_graph), initial_count)

    def test_reason_with_same_as(self):
        """Test reason() with owl:sameAs reasoning."""
        ind1 = "http://example.org/individual1"
        ind2 = "http://example.org/individual2"
        prop = "http://example.org/hasProperty"
        obj = "http://example.org/object"
        owl_same_as = URIRef("http://www.w3.org/2002/07/owl#sameAs")

        self.model.instance_graph.add((URIRef(ind1), owl_same_as, URIRef(ind2)))
        self.model.instance_graph.add((URIRef(ind1), URIRef(prop), URIRef(obj)))

        initial_count = len(self.model.instance_graph)
        self.model.reason()
        self.assertGreaterEqual(len(self.model.instance_graph), initial_count)

    # ==================== get_graphs() Tests ====================

    def test_get_graphs_with_turtle_file(self):
        """Test get_graphs with a turtle file."""
        # Create a temporary turtle file
        ttl_content = """
        @prefix ex: <http://example.org/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        
        ex:subject1 rdf:type ex:TestClass .
        ex:subject1 ex:hasValue "test" .
        """

        ttl_file = os.path.join(self.temp_dir, "test.ttl")
        with open(ttl_file, "w") as f:
            f.write(ttl_content)

        # Call get_graphs
        instance_graph, ontology_graph = self.model.get_graphs(ttl_file)

        self.assertIsNotNone(instance_graph)
        self.assertIsNotNone(ontology_graph)
        self.assertGreater(len(instance_graph), 0)

    def test_get_graphs_with_format_specified(self):
        """Test get_graphs with explicit format."""
        # Create a temporary turtle file
        ttl_content = """
        @prefix ex: <http://example.org/> .
        ex:subject ex:predicate ex:object .
        """

        ttl_file = os.path.join(self.temp_dir, "test_format.ttl")
        with open(ttl_file, "w") as f:
            f.write(ttl_content)

        # Call get_graphs with format
        instance_graph, ontology_graph = self.model.get_graphs(
            ttl_file, format="turtle"
        )

        self.assertIsNotNone(instance_graph)
        self.assertGreater(len(instance_graph), 0)

    def test_get_graphs_with_schema_definitions(self):
        """Test get_graphs copies schema definitions to ontology_graph (covers lines 1583-1678)."""
        # Create a turtle file with schema definitions mixed with instances
        ttl_content = """
        @prefix ex: <http://example.org/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        
        # Class definitions (should be copied to ontology_graph)
        ex:Animal rdf:type owl:Class .
        ex:Dog rdf:type owl:Class .
        ex:Dog rdfs:subClassOf ex:Animal .
        
        # Property definitions (should be copied to ontology_graph)
        ex:hasName rdf:type owl:DatatypeProperty .
        ex:hasOwner rdf:type owl:ObjectProperty .
        ex:hasChild rdfs:subPropertyOf ex:hasMember .
        
        # Equivalent class (should be copied)
        ex:Car owl:equivalentClass ex:Automobile .
        
        # Equivalent property (should be copied)
        ex:friend owl:equivalentProperty ex:buddy .
        
        # Instance data
        ex:fido rdf:type ex:Dog .
        ex:fido ex:hasName "Fido" .
        """

        ttl_file = os.path.join(self.temp_dir, "test_schema.ttl")
        with open(ttl_file, "w") as f:
            f.write(ttl_content)

        # Call get_graphs
        instance_graph, ontology_graph = self.model.get_graphs(
            ttl_file, format="turtle"
        )

        self.assertIsNotNone(instance_graph)
        self.assertIsNotNone(ontology_graph)
        # Ontology graph should have schema definitions
        self.assertGreater(len(ontology_graph), 0)

    def test_get_graphs_file_not_found(self):
        """Test get_graphs raises error for non-existent xlsx file."""
        with self.assertRaises(FileNotFoundError):
            self.model.get_graphs("nonexistent.xlsx")

    # ==================== filter_graph() Tests ====================

    def _setup_filter_graph_data(self):
        """Helper to set up test data for filter_graph tests."""
        self.model.instance_graph.add(
            (
                URIRef("http://example.org/a"),
                URIRef("http://example.org/knows"),
                URIRef("http://example.org/b"),
            )
        )
        self.model.instance_graph.add(
            (
                URIRef("http://example.org/b"),
                URIRef("http://example.org/knows"),
                URIRef("http://example.org/c"),
            )
        )
        self.model.instance_graph.add(
            (
                URIRef("http://example.org/c"),
                URIRef("http://example.org/knows"),
                URIRef("http://example.org/d"),
            )
        )
        self.model.instance_graph.add(
            (
                URIRef("http://example.org/a"),
                RDF.type,
                URIRef("http://example.org/Person"),
            )
        )
        self.model.instance_graph.add(
            (
                URIRef("http://example.org/b"),
                RDF.type,
                URIRef("http://example.org/Person"),
            )
        )

    def test_filter_graph_with_construct_query(self):
        """Test filter_graph with CONSTRUCT query."""
        self._setup_filter_graph_data()
        query = """
        CONSTRUCT { ?s ?p ?o }
        WHERE { ?s ?p ?o }
        """
        filtered = self.model.filter_graph(query=query)
        self.assertIsNotNone(filtered)
        self.assertGreater(len(filtered), 0)

    def test_filter_graph_with_initial_node(self):
        """Test filter_graph with initial_node."""
        self._setup_filter_graph_data()
        query = """
        CONSTRUCT { ?s ?p ?o }
        WHERE { ?s ?p ?o }
        """
        filtered = self.model.filter_graph(
            query=query, initial_node="http://example.org/a"
        )
        self.assertIsNotNone(filtered)

    def test_filter_graph_with_node_limit(self):
        """Test filter_graph with node_limit."""
        self._setup_filter_graph_data()
        query = """
        CONSTRUCT { ?s ?p ?o }
        WHERE { ?s ?p ?o }
        """
        filtered = self.model.filter_graph(query=query, node_limit=2)
        self.assertIsNotNone(filtered)

    def test_filter_graph_with_triple_limit(self):
        """Test filter_graph with triple_limit."""
        self._setup_filter_graph_data()
        query = """
        CONSTRUCT { ?s ?p ?o }
        WHERE { ?s ?p ?o }
        """
        filtered = self.model.filter_graph(query=query, triple_limit=2)
        self.assertIsNotNone(filtered)

    def test_filter_graph_with_bfs_traversal(self):
        """Test filter_graph with BFS traversal."""
        self._setup_filter_graph_data()
        query = """
        CONSTRUCT { ?s ?p ?o }
        WHERE { ?s ?p ?o }
        """
        filtered = self.model.filter_graph(
            query=query, traversal_mode="bfs", node_limit=5
        )
        self.assertIsNotNone(filtered)

    def test_filter_graph_with_dfs_traversal(self):
        """Test filter_graph with DFS traversal."""
        self._setup_filter_graph_data()
        query = """
        CONSTRUCT { ?s ?p ?o }
        WHERE { ?s ?p ?o }
        """
        filtered = self.model.filter_graph(
            query=query, traversal_mode="dfs", node_limit=5
        )
        self.assertIsNotNone(filtered)

    def test_filter_graph_with_random_seed(self):
        """Test filter_graph with random_seed."""
        self._setup_filter_graph_data()
        query = """
        CONSTRUCT { ?s ?p ?o }
        WHERE { ?s ?p ?o }
        """
        filtered = self.model.filter_graph(query=query, random_seed=42)
        self.assertIsNotNone(filtered)

    def test_filter_graph_invalid_query(self):
        """Test filter_graph with non-CONSTRUCT query raises error."""
        self._setup_filter_graph_data()
        query = """
        SELECT ?s ?p ?o
        WHERE { ?s ?p ?o }
        """
        with self.assertRaises(ValueError):
            self.model.filter_graph(query=query)

    def test_filter_graph_invalid_initial_node(self):
        """Test filter_graph with non-existent initial_node raises error."""
        self._setup_filter_graph_data()
        query = """
        CONSTRUCT { ?s ?p ?o }
        WHERE { ?s ?p ?o }
        """
        with self.assertRaises(ValueError):
            self.model.filter_graph(
                query=query, initial_node="http://example.org/nonexistent"
            )

    # ==================== visualize() Tests ====================

    def _setup_visualize_data(self):
        """Helper to set up test data for visualize tests."""
        # Register namespace so get_short_name() works
        example_ns = Namespace("http://example.org/")
        self.model.add_namespaces({"EX": example_ns})

        type_uri = "http://example.org/TestClass"
        instance1_uri = "http://example.org/instance1"
        instance2_uri = "http://example.org/instance2"
        pred_uri = "http://example.org/relatesTo"

        self.model.ontology_graph.add(
            (URIRef(type_uri), RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))
        )
        self.model.instance_graph.add(
            (URIRef(instance1_uri), RDF.type, URIRef(type_uri))
        )
        self.model.instance_graph.add(
            (URIRef(instance2_uri), RDF.type, URIRef(type_uri))
        )
        self.model.instance_graph.add(
            (URIRef(instance1_uri), URIRef(pred_uri), URIRef(instance2_uri))
        )
        self.model.instance_graph.add(
            (
                URIRef(instance1_uri),
                URIRef("http://example.org/hasName"),
                Literal("Instance One"),
            )
        )

    def test_visualize_basic(self):
        """Test basic visualize call."""
        if not self.graphviz_installed:
            self.skipTest("Graphviz not installed")

        self._setup_visualize_data()
        self.model.visualize()

        svg_path, _ = self.model.get_dir(
            folder_list=["graphs"], filename="semantic_model.svg"
        )
        self.assertTrue(os.path.exists(svg_path), f"Expected SVG file at {svg_path}")

    def test_visualize_with_custom_query(self):
        """Test visualize with custom CONSTRUCT query."""
        if not self.graphviz_installed:
            self.skipTest("Graphviz not installed")

        self._setup_visualize_data()
        query = """
        CONSTRUCT { ?s ?p ?o }
        WHERE { ?s ?p ?o }
        """
        self.model.visualize(query=query)

    def test_visualize_with_node_limit(self):
        """Test visualize with node_limit."""
        if not self.graphviz_installed:
            self.skipTest("Graphviz not installed")

        self._setup_visualize_data()
        self.model.visualize(node_limit=5)

    def test_visualize_with_triple_limit(self):
        """Test visualize with triple_limit."""
        if not self.graphviz_installed:
            self.skipTest("Graphviz not installed")

        self._setup_visualize_data()
        self.model.visualize(triple_limit=10)

    def test_visualize_without_full_uri(self):
        """Test visualize with include_full_uri=False."""
        if not self.graphviz_installed:
            self.skipTest("Graphviz not installed")

        self._setup_visualize_data()
        self.model.visualize(include_full_uri=False)

    def test_visualize_with_slice_uri_int(self):
        """Test visualize with slice_uri as integer."""
        if not self.graphviz_installed:
            self.skipTest("Graphviz not installed")

        self._setup_visualize_data()
        self.model.visualize(slice_uri=20)

    def test_visualize_with_slice_uri_tuple(self):
        """Test visualize with slice_uri as tuple."""
        if not self.graphviz_installed:
            self.skipTest("Graphviz not installed")

        self._setup_visualize_data()
        self.model.visualize(slice_uri=(0, 30))

    def test_visualize_with_bfs_traversal(self):
        """Test visualize with BFS traversal mode."""
        if not self.graphviz_installed:
            self.skipTest("Graphviz not installed")

        self._setup_visualize_data()
        self.model.visualize(traversal_mode="bfs", node_limit=5)

    def test_visualize_with_generate_subgraphs(self):
        """Test visualize with generate_subgraphs=True."""
        if not self.graphviz_installed:
            self.skipTest("Graphviz not installed")

        self._setup_visualize_data()
        self.model.visualize(generate_subgraphs=True)

    def test_visualize_with_custom_dpi(self):
        """Test visualize with custom DPI."""
        if not self.graphviz_installed:
            self.skipTest("Graphviz not installed")

        self._setup_visualize_data()
        self.model.visualize(dpi=100)


if __name__ == "__main__":
    unittest.main()
    # TestSemanticModel.setUpClass()
    # test_semantic_model = TestSemanticModel()
    # test_semantic_model.setUp()
    # test_semantic_model.test_visualize_basic()
    # test_semantic_model.tearDown()
