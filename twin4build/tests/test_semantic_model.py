import unittest
import os
import shutil
from twin4build.model.semantic_model.semantic_model import SemanticModel


class TestSemanticModel(unittest.TestCase):
    def setUp(self):
        """Set up a fresh semantic model for each test."""
        self.semantic_model = SemanticModel()

    def tearDown(self):
        """Clean up any generated files."""
        if os.path.exists("test_output.ttl"):
            os.remove("test_output.ttl")

    def test_initialization_without_rdf(self):
        """Test semantic model initialization without RDF file."""
        model = SemanticModel()
        self.assertIsNotNone(model)
        # Semantic model uses instance_graph and ontology_graph
        self.assertTrue(hasattr(model, 'instance_graph'))
        self.assertTrue(hasattr(model, 'ontology_graph'))

    def test_initialization_with_rdf(self):
        """Test semantic model initialization with RDF file."""
        # This test requires an actual RDF file
        # Skip if no test RDF file is available
        # Try to find an example RDF file
        from twin4build.utils.uppath import uppath
        test_file = os.path.join(
            uppath(os.path.abspath(__file__), 1),
            "test_instance_graph.ttl"
        )
        
        model = SemanticModel(rdf_file=test_file)
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.instance_graph)
        self.assertIsNotNone(model.ontology_graph)

    def test_get_instances_of_type(self):
        """Test querying instances by type."""
        # First add some instances
        instance_type = "http://example.org/Damper"
        
        instances = self.semantic_model.get_instances_of_type(instance_type)
        
        # Should return a list (possibly empty)
        self.assertIsInstance(instances, (list, set))

    def test_serialize(self):
        """Test RDF serialization."""
        output_file, _ = self.semantic_model.get_dir(filename="instance_graph.ttl")
        
        # Add some data to the model
        # (This is a basic test; actual implementation may vary)
        
        self.semantic_model.serialize(filename_instance_graph=output_file)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_file))

    def test_visualize(self):
        """Test graph visualization."""
        # Just test that the method exists and doesn't crash
        # Actual visualization testing would require more setup
        self.semantic_model.visualize()
        self.assertTrue(True)

    def test_graph_property(self):
        """Test that semantic model has graph properties."""
        self.assertTrue(hasattr(self.semantic_model, 'instance_graph'))
        self.assertTrue(hasattr(self.semantic_model, 'ontology_graph'))

    def test_query_basic(self):
        """Test basic SPARQL querying if supported."""
        # Try a simple SPARQL query
        query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
        results = self.semantic_model.instance_graph.query(query)
        
        # Should return results (possibly empty)
        self.assertIsNotNone(results)


if __name__ == '__main__':
    unittest.main()

