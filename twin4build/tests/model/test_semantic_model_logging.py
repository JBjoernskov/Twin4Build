"""Building a semantic model leaves the application's logging alone.

``SemanticModel(rdf_file=...)`` with the default ``verbose`` used to call
``logging.disable(sys.maxsize)`` and never restore it: a process-wide
floor that silenced every logger of the embedding application from that
moment on (#138).  The RDF libraries are quieted for the parse only now,
and their loggers come back at their previous level.
"""

# Standard library imports
import logging
import os
import shutil
import tempfile
import unittest

# Local application imports
import twin4build
from twin4build.model.semantic_model.semantic_model import SemanticModel

twin4build._IS_TESTING = True

TTL = """@prefix brick: <https://brickschema.org/schema/Brick#> .
@prefix ex: <http://example.org/> .
ex:room a brick:Room .
"""


class TestSemanticModelLogging(unittest.TestCase):
    MODEL_ID = "test_semantic_model_logging"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ttl = os.path.join(self.tmp, "graph.ttl")
        with open(self.ttl, "w", encoding="utf-8") as f:
            f.write(TTL)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        path = os.path.join("generated_files", "models", self.MODEL_ID)
        if os.path.exists(path):
            shutil.rmtree(path)

    def test_application_logging_survives(self):
        app = logging.getLogger("test_semantic_model_logging.app")
        app.setLevel(logging.INFO)
        rdflib_logger = logging.getLogger("rdflib")
        before = rdflib_logger.level
        self.assertTrue(app.isEnabledFor(logging.INFO))

        SemanticModel(rdf_file=self.ttl, id=self.MODEL_ID)

        self.assertEqual(logging.root.manager.disable, 0)
        self.assertTrue(app.isEnabledFor(logging.INFO))
        self.assertEqual(rdflib_logger.level, before)


if __name__ == "__main__":
    unittest.main()
