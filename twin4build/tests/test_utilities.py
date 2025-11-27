import unittest
import datetime
import os
import pytz
from twin4build.utils.uppath import uppath
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.rhasattr import rhasattr
from twin4build.utils.rdelattr import rdelattr
from twin4build.utils.validate_period import validate_period


class TestUppath(unittest.TestCase):
    def test_uppath_basic(self):
        """Test basic uppath functionality."""
        # Use OS-specific separator
        path = os.path.join("C:", "Example", "test", "path", "file")
        result = uppath(path, 2)
        # Result should contain the parent directories
        self.assertIn("test", result)

    def test_uppath_single_level(self):
        """Test uppath with single level removal."""
        path = os.path.join("home", "user", "documents", "file.txt")
        result = uppath(path, 1)
        self.assertIn("documents", result)

    def test_uppath_multiple_levels(self):
        """Test uppath with multiple level removal."""
        path = os.path.join("a", "b", "c", "d", "e", "f")
        result = uppath(path, 3)
        self.assertIn("c", result)

    def test_uppath_zero_levels(self):
        """Test uppath with zero level removal."""
        path = os.path.join("a", "b", "c")
        result = uppath(path, 0)
        self.assertEqual(result, path)

    def test_uppath_with_windows_path(self):
        """Test uppath with Windows-style path."""
        path = "C:\\Users\\test\\file.txt"
        result = uppath(path, 1)
        # Should handle OS-specific separators
        self.assertIn("test", result)


class TestRecursiveAttributeOperations(unittest.TestCase):
    def setUp(self):
        """Create a test object with nested attributes."""
        class Inner:
            def __init__(self):
                self.value = 42
                self.name = "inner"

        class Outer:
            def __init__(self):
                self.inner = Inner()
                self.simple = "test"

        self.obj = Outer()

    def test_rgetattr_simple(self):
        """Test rgetattr with simple attribute."""
        result = rgetattr(self.obj, "simple")
        self.assertEqual(result, "test")

    def test_rgetattr_nested(self):
        """Test rgetattr with nested attribute."""
        result = rgetattr(self.obj, "inner.value")
        self.assertEqual(result, 42)

    def test_rgetattr_deep_nested(self):
        """Test rgetattr with deeply nested attribute."""
        result = rgetattr(self.obj, "inner.name")
        self.assertEqual(result, "inner")

    def test_rgetattr_with_default(self):
        """Test rgetattr with default value for missing attribute."""
        result = rgetattr(self.obj, "nonexistent", "default")
        self.assertEqual(result, "default")

    def test_rsetattr_simple(self):
        """Test rsetattr with simple attribute."""
        rsetattr(self.obj, "simple", "modified")
        self.assertEqual(self.obj.simple, "modified")

    def test_rsetattr_nested(self):
        """Test rsetattr with nested attribute."""
        rsetattr(self.obj, "inner.value", 99)
        self.assertEqual(self.obj.inner.value, 99)

    def test_rsetattr_nested_string(self):
        """Test rsetattr with nested string attribute."""
        rsetattr(self.obj, "inner.name", "modified_inner")
        self.assertEqual(self.obj.inner.name, "modified_inner")

    def test_rhasattr_existing(self):
        """Test rhasattr with existing attribute."""
        self.assertTrue(rhasattr(self.obj, "simple"))
        self.assertTrue(rhasattr(self.obj, "inner.value"))

    def test_rhasattr_nonexistent(self):
        """Test rhasattr with non-existent attribute."""
        self.assertFalse(rhasattr(self.obj, "nonexistent"))
        self.assertFalse(rhasattr(self.obj, "inner.nonexistent"))

    def test_rdelattr_simple(self):
        """Test rdelattr with simple attribute."""
        rdelattr(self.obj, "simple")
        self.assertFalse(hasattr(self.obj, "simple"))

    def test_rdelattr_nested(self):
        """Test rdelattr with nested attribute."""
        rdelattr(self.obj, "inner.value")
        self.assertFalse(hasattr(self.obj.inner, "value"))


class TestValidatePeriod(unittest.TestCase):
    def test_validate_period_single_datetime(self):
        """Test validate_period with single datetime objects."""
        start = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end = datetime.datetime(2023, 1, 2, 0, 0, 0, tzinfo=pytz.UTC)
        step = 600
        
        start_out, end_out, step_out = validate_period(start, end, step)
        
        self.assertEqual(len(start_out), 1)
        self.assertEqual(len(end_out), 1)
        self.assertEqual(len(step_out), 1)
        self.assertEqual(start_out[0], start)
        self.assertEqual(end_out[0], end)
        self.assertEqual(step_out[0], step)

    def test_validate_period_list_datetime(self):
        """Test validate_period with list of datetime objects."""
        starts = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            datetime.datetime(2023, 1, 2, 0, 0, 0, tzinfo=pytz.UTC),
        ]
        ends = [
            datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC),
            datetime.datetime(2023, 1, 2, 1, 0, 0, tzinfo=pytz.UTC),
        ]
        steps = [600, 300]
        
        start_out, end_out, step_out = validate_period(starts, ends, steps)
        
        self.assertEqual(len(start_out), 2)
        self.assertEqual(len(end_out), 2)
        self.assertEqual(len(step_out), 2)
        self.assertEqual(start_out, starts)
        self.assertEqual(end_out, ends)
        self.assertEqual(step_out, steps)

    def test_validate_period_list_with_single_step(self):
        """Test validate_period with list of datetimes but single step size."""
        starts = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            datetime.datetime(2023, 1, 2, 0, 0, 0, tzinfo=pytz.UTC),
        ]
        ends = [
            datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC),
            datetime.datetime(2023, 1, 2, 1, 0, 0, tzinfo=pytz.UTC),
        ]
        step = 600
        
        start_out, end_out, step_out = validate_period(starts, ends, step)
        
        self.assertEqual(len(start_out), 2)
        self.assertEqual(len(end_out), 2)
        self.assertEqual(len(step_out), 2)
        self.assertEqual(step_out, [600, 600])

    def test_validate_period_invalid_start_type(self):
        """Test validate_period with invalid start time type."""
        with self.assertRaises(AssertionError):
            validate_period("invalid", datetime.datetime.now(pytz.UTC), 600)

    def test_validate_period_invalid_end_type(self):
        """Test validate_period with invalid end time type."""
        with self.assertRaises(AssertionError):
            validate_period(datetime.datetime.now(pytz.UTC), "invalid", 600)

    def test_validate_period_invalid_step_type(self):
        """Test validate_period with invalid step size type."""
        with self.assertRaises(AssertionError):
            validate_period(
                datetime.datetime.now(pytz.UTC),
                datetime.datetime.now(pytz.UTC),
                "invalid"
            )

    def test_validate_period_mismatched_list_lengths(self):
        """Test validate_period with mismatched list lengths."""
        starts = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        ends = [
            datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC),
            datetime.datetime(2023, 1, 2, 1, 0, 0, tzinfo=pytz.UTC),
        ]
        steps = [600]
        
        with self.assertRaises(AssertionError):
            validate_period(starts, ends, steps)

    def test_validate_period_mixed_types(self):
        """Test validate_period with mixed types (should fail)."""
        start = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        ends = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)]
        step = 600
        
        with self.assertRaises(AssertionError):
            validate_period(start, ends, step)


class TestDataLoaders(unittest.TestCase):
    def test_sample_from_df_imports(self):
        """Test that data loader functions can be imported."""
        from twin4build.utils.data_loaders.load import sample_from_df
        self.assertIsNotNone(sample_from_df)

    def test_parseDateStr_imports(self):
        """Test that parseDateStr can be imported."""
        from twin4build.utils.data_loaders.load import parseDateStr
        self.assertIsNotNone(parseDateStr)


class TestPlotUtilities(unittest.TestCase):
    def test_plot_imports(self):
        """Test that plot utilities can be imported."""
        from twin4build.utils.plot import plot, plot_component, Entry, Colors
        self.assertIsNotNone(plot)
        self.assertIsNotNone(plot_component)
        self.assertIsNotNone(Entry)
        self.assertIsNotNone(Colors)


if __name__ == '__main__':
    unittest.main()

