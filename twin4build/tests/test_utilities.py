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

    def test_parseDateStr_valid(self):
        """Test parseDateStr with valid date string."""
        import numpy as np
        from twin4build.utils.data_loaders.load import parseDateStr
        
        result = parseDateStr("2023-01-15T10:30:00")
        self.assertIsNotNone(result)
        self.assertFalse(np.isnat(result))

    def test_parseDateStr_empty(self):
        """Test parseDateStr with empty string."""
        import numpy as np
        from twin4build.utils.data_loaders.load import parseDateStr
        
        result = parseDateStr("")
        self.assertTrue(np.isnat(result))

    def test_parseDateStr_invalid(self):
        """Test parseDateStr with invalid date string."""
        import numpy as np
        from twin4build.utils.data_loaders.load import parseDateStr
        
        result = parseDateStr("not_a_date")
        self.assertTrue(np.isnat(result))

    def test_sample_from_df_basic(self):
        """Test sample_from_df with basic DataFrame."""
        import pandas as pd
        from twin4build.utils.data_loaders.load import sample_from_df
        
        # Create test DataFrame
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=10,
            freq='1h'
        )
        df = pd.DataFrame({
            'date_time': dates,
            'value': [i * 10.0 for i in range(10)]
        })
        
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 5, 0, 0, tzinfo=pytz.UTC)
        
        result = sample_from_df(
            df,
            datecolumn=0,
            valuecolumn=1,
            step_size=3600,
            start_time=start_time,
            end_time=end_time,
            resample=True,
            clip=True,
            tz="UTC"
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)  # 5 hours from 0 to 4

    def test_sample_from_df_constant_resample(self):
        """Test sample_from_df with constant resampling."""
        import pandas as pd
        from twin4build.utils.data_loaders.load import sample_from_df
        
        # Create test DataFrame
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=5,
            freq='2h'
        )
        df = pd.DataFrame({
            'date_time': dates,
            'value': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 6, 0, 0, tzinfo=pytz.UTC)
        
        result = sample_from_df(
            df,
            datecolumn=0,
            valuecolumn=1,
            step_size=3600,
            start_time=start_time,
            end_time=end_time,
            resample=True,
            resample_method="constant",
            clip=True,
            tz="UTC"
        )
        
        self.assertIsNotNone(result)
        # Check that constant resampling forward-fills values
        self.assertEqual(result.iloc[0].item(), 10.0)
        self.assertEqual(result.iloc[1].item(), 10.0)  # Forward-filled from previous

    def test_sample_from_df_no_resample(self):
        """Test sample_from_df without resampling."""
        import pandas as pd
        from twin4build.utils.data_loaders.load import sample_from_df
        
        # Create test DataFrame
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=5,
            freq='1h'
        )
        df = pd.DataFrame({
            'date_time': dates,
            'value': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 3, 0, 0, tzinfo=pytz.UTC)
        
        result = sample_from_df(
            df,
            datecolumn=0,
            valuecolumn=1,
            step_size=3600,
            start_time=start_time,
            end_time=end_time,
            resample=False,
            clip=True,
            tz="UTC"
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)  # 3 data points


class TestPlotUtilities(unittest.TestCase):
    def test_plot_imports(self):
        """Test that plot utilities can be imported."""
        from twin4build.utils.plot import plot, plot_component, Entry, Colors
        self.assertIsNotNone(plot)
        self.assertIsNotNone(plot_component)
        self.assertIsNotNone(Entry)
        self.assertIsNotNone(Colors)


class TestUnitConverters(unittest.TestCase):
    def test_do_nothing(self):
        from twin4build.utils.unit_converters.functions import _do_nothing
        self.assertEqual(_do_nothing(5), 5)
        self.assertEqual(_do_nothing(-3.14), -3.14)

    def test_change_sign(self):
        from twin4build.utils.unit_converters.functions import change_sign
        self.assertEqual(change_sign(5), -5)
        self.assertEqual(change_sign(-3.14), 3.14)

    def test_temperature_conversions(self):
        from twin4build.utils.unit_converters.functions import to_degC_from_degK, to_degK_from_degC
        
        # 0 degC = 273.15 K
        self.assertAlmostEqual(to_degC_from_degK(273.15), 0)
        self.assertAlmostEqual(to_degK_from_degC(0), 273.15)
        
        # 100 degC = 373.15 K
        self.assertAlmostEqual(to_degC_from_degK(373.15), 100)
        self.assertAlmostEqual(to_degK_from_degC(100), 373.15)

    def test_multiply_const(self):
        from twin4build.utils.unit_converters.functions import multiply_const
        
        converter = multiply_const(2.5)
        self.assertEqual(converter(4), 10.0)
        self.assertEqual(converter.call(4), 10.0)

    def test_regularize(self):
        from twin4build.utils.unit_converters.functions import regularize
        
        converter = regularize(0)
        self.assertEqual(converter(5), 5)
        self.assertEqual(converter(-5), 0)
        self.assertEqual(converter(0), 0)

    def test_add_attr(self):
        from twin4build.utils.unit_converters.functions import add_attr
        
        class TestObj:
            def __init__(self):
                self.offset = 10
        
        obj = TestObj()
        converter = add_attr(obj, "offset")
        
        self.assertEqual(converter(5), 15)


class TestGetObjAttr(unittest.TestCase):
    def test_get_obj_attr_normal(self):
        """Test get_obj_attr with normal object."""
        from twin4build.utils.get_obj_attr import get_obj_attr
        
        class TestObj:
            def __init__(self):
                self.a = 1
                self.b = 2
                self._private = 3
        
        obj = TestObj()
        attrs = get_obj_attr(obj)
        
        self.assertIn('a', attrs)
        self.assertIn('b', attrs)
        self.assertIn('_private', attrs)
        self.assertEqual(attrs['a'], 1)
        self.assertEqual(attrs['b'], 2)

    def test_get_obj_attr_inverse(self):
        """Test get_obj_attr with inverse mapping."""
        from twin4build.utils.get_obj_attr import get_obj_attr
        
        class TestObj:
            def __init__(self):
                self.a = 1
                self.b = 2
        
        obj = TestObj()
        attrs = get_obj_attr(obj, inverse=True)
        
        self.assertIn(1, attrs)
        self.assertIn(2, attrs)
        self.assertEqual(attrs[1], 'a')
        self.assertEqual(attrs[2], 'b')


class TestDictUtils(unittest.TestCase):
    def test_compare_dict_structure_same(self):
        """Test compare_dict_structure with same structure."""
        from twin4build.utils.dict_utils import compare_dict_structure
        
        dict1 = {'a': 1, 'b': {'c': 2}}
        dict2 = {'a': 3, 'b': {'c': 4}}
        
        result = compare_dict_structure(dict1, dict2)
        self.assertTrue(result['structures_match'])

    def test_compare_dict_structure_different(self):
        """Test compare_dict_structure with different structure."""
        from twin4build.utils.dict_utils import compare_dict_structure
        
        dict1 = {'a': 1, 'b': {'c': 2}}
        dict2 = {'a': 3, 'x': {'y': 4}}
        
        result = compare_dict_structure(dict1, dict2)
        self.assertFalse(result['structures_match'])
        self.assertIn('b', result['missing_in_2'])
        self.assertIn('x', result['missing_in_1'])

    def test_flatten_dict(self):
        """Test flatten_dict function returns list of tuples."""
        from twin4build.utils.dict_utils import flatten_dict
        
        # Create a simple object for testing
        class TestObj:
            pass
        
        obj = TestObj()
        nested = {'a': 1, 'b': 2}
        flattened = flatten_dict(nested, obj)
        
        self.assertIsInstance(flattened, list)
        # Should return tuples with (key, value)
        self.assertEqual(len(flattened), 2)


class TestMkdirInRoot(unittest.TestCase):
    def test_mkdir_in_root_basic(self):
        """Test mkdir_in_root creates directories."""
        from twin4build.utils.mkdir_in_root import mkdir_in_root
        import tempfile
        import shutil
        
        # Use a temp directory as root
        with tempfile.TemporaryDirectory() as tmpdir:
            result, isfile = mkdir_in_root(
                folder_list=["test_folder", "subfolder"],
                filename="test.txt",
                root=tmpdir
            )
            
            self.assertIsNotNone(result)
            self.assertTrue(result.endswith("test.txt"))


if __name__ == '__main__':
    unittest.main()

