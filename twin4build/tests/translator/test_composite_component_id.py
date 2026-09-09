"""Regression test: composite (multi-member ``ModeledNode``) component ids
must stay short enough to be used as filenames.

The prefix used to shrink each member slice to a minimum of four
characters but never dropped members, so an AHU pattern binding hundreds
of VAVs produced a multi-kilobyte id and ``model_parameters/<class>/<id>.json``
could not be written (Windows ``MAX_PATH``).
"""

# Standard library imports
import unittest

# Local application imports
from twin4build.translator.translator import Translator

FP = "0123456789abcdef0123456789abcdef"


class TestCompositeComponentId(unittest.TestCase):
    def test_small_group_keeps_legacy_form(self):
        names = ["HTR9_VEN02", "R08.01_VAV01_1", "R08_01_VAV01_1"]
        id_ = Translator._composite_component_id(names, FP)
        self.assertEqual(
            id_, "[HTR9_VEN02][R08_01_VAV01_1][R08_01_VAV01_1]_0123456789abcdef"
        )

    def test_large_group_is_capped(self):
        names = ["HTR9_VEN02"] + [f"R00.{i:02d}_VAV01_1" for i in range(700)]
        id_ = Translator._composite_component_id(names, FP)
        budget = Translator.COMPOSITE_ID_NAME_BUDGET
        prefix, _, fp = id_.rpartition("_")
        self.assertEqual(fp, FP[:16])
        # prefix = bracketed slices + "+N" marker; must respect the budget
        self.assertLessEqual(len(prefix), budget + len("+700"))
        self.assertIn("+", prefix)
        self.assertLess(len(id_), 120)

    def test_distinct_fingerprints_stay_distinct(self):
        names = [f"member_{i}" for i in range(300)]
        a = Translator._composite_component_id(names, "a" * 32)
        b = Translator._composite_component_id(names, "b" * 32)
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
