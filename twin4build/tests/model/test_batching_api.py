# Standard library imports
import unittest

# Local application imports
from twin4build.model.model import Model


class TestBatchingApi(unittest.TestCase):
    def test_batching_api_is_hard_renamed(self):
        self.assertTrue(callable(Model.batch_components))
        self.assertTrue(callable(Model.get_batched_component_info))
        self.assertTrue(callable(Model.get_batch_id_for_component))
        self.assertFalse(hasattr(Model, "build_compiled_model"))
        self.assertFalse(hasattr(Model, "get_compiled_component_info"))
        self.assertFalse(hasattr(Model, "get_block_id_for_component"))


if __name__ == "__main__":
    unittest.main()
