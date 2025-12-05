# Standard library imports
import datetime
import os
import unittest

# Third party imports
import pytz

# Local application imports
from twin4build.utils.rdelattr import rdelattr
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rhasattr import rhasattr
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.uppath import uppath
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
                "invalid",
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
        # Local application imports
        from twin4build.utils.data_loaders.load import sample_from_df

        self.assertIsNotNone(sample_from_df)

    def test_parseDateStr_imports(self):
        """Test that parseDateStr can be imported."""
        # Local application imports
        from twin4build.utils.data_loaders.load import parseDateStr

        self.assertIsNotNone(parseDateStr)

    def test_parseDateStr_valid(self):
        """Test parseDateStr with valid date string."""
        # Third party imports
        import numpy as np

        # Local application imports
        from twin4build.utils.data_loaders.load import parseDateStr

        result = parseDateStr("2023-01-15T10:30:00")
        self.assertIsNotNone(result)
        self.assertFalse(np.isnat(result))

    def test_parseDateStr_empty(self):
        """Test parseDateStr with empty string."""
        # Third party imports
        import numpy as np

        # Local application imports
        from twin4build.utils.data_loaders.load import parseDateStr

        result = parseDateStr("")
        self.assertTrue(np.isnat(result))

    def test_parseDateStr_invalid(self):
        """Test parseDateStr with invalid date string."""
        # Third party imports
        import numpy as np

        # Local application imports
        from twin4build.utils.data_loaders.load import parseDateStr

        result = parseDateStr("not_a_date")
        self.assertTrue(np.isnat(result))

    def test_sample_from_df_basic(self):
        """Test sample_from_df with basic DataFrame."""
        # Third party imports
        import pandas as pd

        # Local application imports
        from twin4build.utils.data_loaders.load import sample_from_df

        # Create test DataFrame
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=10,
            freq="1h",
        )
        df = pd.DataFrame({"date_time": dates, "value": [i * 10.0 for i in range(10)]})

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
            tz="UTC",
        )

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)  # 5 hours from 0 to 4

    def test_sample_from_df_constant_resample(self):
        """Test sample_from_df with constant resampling."""
        # Third party imports
        import pandas as pd

        # Local application imports
        from twin4build.utils.data_loaders.load import sample_from_df

        # Create test DataFrame
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=5,
            freq="2h",
        )
        df = pd.DataFrame({"date_time": dates, "value": [10.0, 20.0, 30.0, 40.0, 50.0]})

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
            tz="UTC",
        )

        self.assertIsNotNone(result)
        # Check that constant resampling forward-fills values
        self.assertEqual(result.iloc[0].item(), 10.0)
        self.assertEqual(result.iloc[1].item(), 10.0)  # Forward-filled from previous

    def test_sample_from_df_no_resample(self):
        """Test sample_from_df without resampling."""
        # Third party imports
        import pandas as pd

        # Local application imports
        from twin4build.utils.data_loaders.load import sample_from_df

        # Create test DataFrame
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=5,
            freq="1h",
        )
        df = pd.DataFrame({"date_time": dates, "value": [10.0, 20.0, 30.0, 40.0, 50.0]})

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
            tz="UTC",
        )

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)  # 3 data points


class TestPlotUtilities(unittest.TestCase):
    def test_plot_imports(self):
        """Test that plot utilities can be imported."""
        # Local application imports
        from twin4build.utils.plot import Colors, Entry, plot, plot_component

        self.assertIsNotNone(plot)
        self.assertIsNotNone(plot_component)
        self.assertIsNotNone(Entry)
        self.assertIsNotNone(Colors)


class TestUnitConverters(unittest.TestCase):
    def test_do_nothing(self):
        # Local application imports
        from twin4build.utils.unit_converters.functions import _do_nothing

        self.assertEqual(_do_nothing(5), 5)
        self.assertEqual(_do_nothing(-3.14), -3.14)

    def test_change_sign(self):
        # Local application imports
        from twin4build.utils.unit_converters.functions import change_sign

        self.assertEqual(change_sign(5), -5)
        self.assertEqual(change_sign(-3.14), 3.14)

    def test_temperature_conversions(self):
        # Local application imports
        from twin4build.utils.unit_converters.functions import (
            to_degC_from_degK,
            to_degK_from_degC,
        )

        # 0 degC = 273.15 K
        self.assertAlmostEqual(to_degC_from_degK(273.15), 0)
        self.assertAlmostEqual(to_degK_from_degC(0), 273.15)

        # 100 degC = 373.15 K
        self.assertAlmostEqual(to_degC_from_degK(373.15), 100)
        self.assertAlmostEqual(to_degK_from_degC(100), 373.15)

    def test_multiply_const(self):
        # Local application imports
        from twin4build.utils.unit_converters.functions import multiply_const

        converter = multiply_const(2.5)
        self.assertEqual(converter(4), 10.0)
        self.assertEqual(converter.call(4), 10.0)

    def test_regularize(self):
        # Local application imports
        from twin4build.utils.unit_converters.functions import regularize

        converter = regularize(0)
        self.assertEqual(converter(5), 5)
        self.assertEqual(converter(-5), 0)
        self.assertEqual(converter(0), 0)

    def test_add_attr(self):
        # Local application imports
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
        # Local application imports
        from twin4build.utils.get_obj_attr import get_obj_attr

        class TestObj:
            def __init__(self):
                self.a = 1
                self.b = 2
                self._private = 3

        obj = TestObj()
        attrs = get_obj_attr(obj)

        self.assertIn("a", attrs)
        self.assertIn("b", attrs)
        self.assertIn("_private", attrs)
        self.assertEqual(attrs["a"], 1)
        self.assertEqual(attrs["b"], 2)

    def test_get_obj_attr_inverse(self):
        """Test get_obj_attr with inverse mapping."""
        # Local application imports
        from twin4build.utils.get_obj_attr import get_obj_attr

        class TestObj:
            def __init__(self):
                self.a = 1
                self.b = 2

        obj = TestObj()
        attrs = get_obj_attr(obj, inverse=True)

        self.assertIn(1, attrs)
        self.assertIn(2, attrs)
        self.assertEqual(attrs[1], "a")
        self.assertEqual(attrs[2], "b")


class TestGetObjectProperties(unittest.TestCase):
    def test_get_object_properties(self):
        """Test get_object_properties returns all properties."""
        # Local application imports
        from twin4build.utils.get_object_properties import get_object_properties

        class TestObj:
            def __init__(self):
                self.a = 1
                self.b = "test"
                self.c = [1, 2, 3]

        obj = TestObj()
        props = get_object_properties(obj)

        self.assertIsInstance(props, dict)
        self.assertEqual(props["a"], 1)
        self.assertEqual(props["b"], "test")
        self.assertEqual(props["c"], [1, 2, 3])

    def test_get_object_properties_empty(self):
        """Test get_object_properties with object without properties."""
        # Local application imports
        from twin4build.utils.get_object_properties import get_object_properties

        class EmptyObj:
            pass

        obj = EmptyObj()
        props = get_object_properties(obj)

        self.assertIsInstance(props, dict)
        # Should be empty or contain only internal properties
        self.assertEqual(len([k for k in props.keys() if not k.startswith("_")]), 0)


class TestDeprecation(unittest.TestCase):
    def test_deprecate_args_with_position(self):
        """Test deprecate_args with positional replacement."""
        # Local application imports
        from twin4build.utils.deprecation import deprecate_args

        kwargs = {"old_arg": "value1"}
        with self.assertWarns(DeprecationWarning):
            value_map = deprecate_args(
                deprecated_args=["old_arg"],
                new_args=["new_arg"],
                positions=[0],
                kwargs=kwargs,
            )

        self.assertIn("new_arg", value_map)
        self.assertEqual(value_map["new_arg"], "value1")
        self.assertNotIn("old_arg", kwargs)

    def test_deprecate_args_without_position(self):
        """Test deprecate_args without position (None)."""
        # Local application imports
        from twin4build.utils.deprecation import deprecate_args

        kwargs = {"old_param": 42}
        with self.assertWarns(DeprecationWarning):
            value_map = deprecate_args(
                deprecated_args=["old_param"],
                new_args=["new_param"],
                positions=[None],
                kwargs=kwargs,
            )

        self.assertIn("new_param", value_map)
        self.assertEqual(value_map["new_param"], 42)
        self.assertNotIn("old_param", kwargs)

    def test_deprecate_args_invalid_position(self):
        """Test deprecate_args with invalid position raises ValueError."""
        # Local application imports
        from twin4build.utils.deprecation import deprecate_args

        kwargs = {"old_arg": "value"}
        with self.assertRaises(ValueError):
            deprecate_args(
                deprecated_args=["old_arg"],
                new_args=["new_arg"],
                positions=["invalid"],  # Invalid position type
                kwargs=kwargs,
            )

    def test_deprecate_args_no_deprecated_args(self):
        """Test deprecate_args when no deprecated args are in kwargs."""
        # Local application imports
        from twin4build.utils.deprecation import deprecate_args

        kwargs = {"normal_arg": "value"}
        value_map = deprecate_args(
            deprecated_args=["old_arg"],
            new_args=["new_arg"],
            positions=[0],
            kwargs=kwargs,
        )

        # Should return empty map and not modify kwargs
        self.assertEqual(len(value_map), 0)
        self.assertIn("normal_arg", kwargs)

    def test_deprecate_args_multiple(self):
        """Test deprecate_args with multiple deprecated arguments."""
        # Local application imports
        from twin4build.utils.deprecation import deprecate_args

        kwargs = {"old1": "val1", "old2": "val2"}
        with self.assertWarns(DeprecationWarning):
            value_map = deprecate_args(
                deprecated_args=["old1", "old2"],
                new_args=["new1", "new2"],
                positions=[0, 1],
                kwargs=kwargs,
            )

        self.assertEqual(value_map["new1"], "val1")
        self.assertEqual(value_map["new2"], "val2")
        self.assertNotIn("old1", kwargs)
        self.assertNotIn("old2", kwargs)


class TestDictUtils(unittest.TestCase):
    def test_compare_dict_structure_same(self):
        """Test compare_dict_structure with same structure."""
        # Local application imports
        from twin4build.utils.dict_utils import compare_dict_structure

        dict1 = {"a": 1, "b": {"c": 2}}
        dict2 = {"a": 3, "b": {"c": 4}}

        result = compare_dict_structure(dict1, dict2)
        self.assertTrue(result["structures_match"])

    def test_compare_dict_structure_different(self):
        """Test compare_dict_structure with different structure."""
        # Local application imports
        from twin4build.utils.dict_utils import compare_dict_structure

        dict1 = {"a": 1, "b": {"c": 2}}
        dict2 = {"a": 3, "x": {"y": 4}}

        result = compare_dict_structure(dict1, dict2)
        self.assertFalse(result["structures_match"])
        self.assertIn("b", result["missing_in_2"])
        self.assertIn("x", result["missing_in_1"])

    def test_compare_dict_structure_nested_diff(self):
        """Test compare_dict_structure with deeply nested differences."""
        # Local application imports
        from twin4build.utils.dict_utils import compare_dict_structure

        dict1 = {"a": {"b": {"c": 1, "d": 2}}, "e": 3}
        dict2 = {"a": {"b": {"c": 1}}, "e": 3}

        result = compare_dict_structure(dict1, dict2)
        self.assertFalse(result["structures_match"])
        self.assertIn("a.b.d", result["missing_in_2"])

    def test_compare_dict_structure_non_dict(self):
        """Test compare_dict_structure with non-dict values."""
        # Local application imports
        from twin4build.utils.dict_utils import compare_dict_structure

        dict1 = "not a dict"
        dict2 = {"a": 1}

        result = compare_dict_structure(dict1, dict2)
        self.assertTrue(result["structures_match"])
        self.assertEqual(len(result["missing_in_2"]), 0)

    def test_get_dict_differences_basic(self):
        """Test get_dict_differences with basic differences."""
        # Local application imports
        from twin4build.utils.dict_utils import get_dict_differences

        dict1 = {"a": 1, "b": 2}
        dict2 = {"a": 1, "c": 3}

        result = get_dict_differences(dict1, dict2)
        self.assertTrue(result["structure_mismatch"])
        self.assertIn("b", result["missing_in_2"])
        self.assertIn("c", result["missing_in_1"])

    def test_get_dict_differences_nested(self):
        """Test get_dict_differences with nested dictionaries."""
        # Local application imports
        from twin4build.utils.dict_utils import get_dict_differences

        dict1 = {"a": {"b": 1, "c": 2}, "d": 3}
        dict2 = {"a": {"b": 1}, "e": 4}

        result = get_dict_differences(dict1, dict2)
        self.assertTrue(result["structure_mismatch"])
        self.assertIn("d", result["missing_in_2"])
        self.assertIn("e", result["missing_in_1"])
        self.assertIn("a.c", result["missing_in_2"])

    def test_get_dict_differences_non_dict(self):
        """Test get_dict_differences with non-dict inputs."""
        # Local application imports
        from twin4build.utils.dict_utils import get_dict_differences

        result = get_dict_differences("not a dict", {"a": 1})
        self.assertFalse(result["structure_mismatch"])

    def test_merge_dicts_standard(self):
        """Test merge_dicts with standard merge."""
        # Local application imports
        from twin4build.utils.dict_utils import merge_dicts

        dict1 = {"a": 1, "b": {"c": 2}}
        dict2 = {"b": {"d": 3}, "e": 4}

        result = merge_dicts(dict1, dict2)
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"]["c"], 2)
        self.assertEqual(result["b"]["d"], 3)
        self.assertEqual(result["e"], 4)

    def test_merge_dicts_prioritize_dict1(self):
        """Test merge_dicts with dict1 priority."""
        # Local application imports
        from twin4build.utils.dict_utils import merge_dicts

        dict1 = {"a": 1, "b": None, "c": {"d": 2}}
        dict2 = {"a": 10, "b": 20, "c": {"d": 30, "e": 40}, "f": 50}

        result = merge_dicts(dict1, dict2, prioritize="dict1")
        # a should keep dict1 value (not None)
        self.assertEqual(result["a"], 1)
        # b should take dict2 value (dict1 is None)
        self.assertEqual(result["b"], 20)
        # c.d should keep dict1 value
        self.assertEqual(result["c"]["d"], 2)
        # c.e should take dict2 value (doesn't exist in dict1)
        # BUT with dict1 priority, keys not in dict1 should NOT be added
        self.assertNotIn("e", result["c"])
        # f should NOT be in result (not in dict1)
        self.assertNotIn("f", result)

    def test_merge_dicts_prioritize_dict2(self):
        """Test merge_dicts with dict2 priority."""
        # Local application imports
        from twin4build.utils.dict_utils import merge_dicts

        dict1 = {"a": 10, "b": 20, "c": {"d": 30}}
        dict2 = {"a": 1, "b": None, "c": {"d": 2}}

        result = merge_dicts(dict1, dict2, prioritize="dict2")
        # a should keep dict2 value (not None)
        self.assertEqual(result["a"], 1)
        # b should take dict1 value (dict2 is None)
        self.assertEqual(result["b"], 20)
        # c.d should keep dict2 value
        self.assertEqual(result["c"]["d"], 2)

    def test_merge_dicts_nested_dicts(self):
        """Test merge_dicts with deeply nested dictionaries."""
        # Local application imports
        from twin4build.utils.dict_utils import merge_dicts

        dict1 = {"a": {"b": {"c": 1}}}
        dict2 = {"a": {"b": {"d": 2}}}

        result = merge_dicts(dict1, dict2)
        self.assertEqual(result["a"]["b"]["c"], 1)
        self.assertEqual(result["a"]["b"]["d"], 2)

    def test_flatten_dict(self):
        """Test flatten_dict function returns list of tuples."""
        # Local application imports
        from twin4build.utils.dict_utils import flatten_dict

        # Create a simple object for testing
        class TestObj:
            pass

        obj = TestObj()
        nested = {"a": 1, "b": 2}
        flattened = flatten_dict(nested, obj)

        self.assertIsInstance(flattened, list)
        # Should return tuples with (key, value)
        self.assertEqual(len(flattened), 2)

    def test_flatten_dict_nested(self):
        """Test flatten_dict with nested dictionaries."""
        # Local application imports
        from twin4build.utils.dict_utils import flatten_dict

        class TestObj:
            def __init__(self):
                self.x = 1
                self.y = 2

        obj = TestObj()
        nested = {"a": 1, "b": {"x": 10, "y": 20}}
        flattened = flatten_dict(nested, obj)

        # Should flatten nested dictionaries
        self.assertIsInstance(flattened, list)
        # Check that x and y are included (they exist in obj)
        keys = [item[0] for item in flattened]
        self.assertIn("x", keys)
        self.assertIn("y", keys)

    def test_flatten_dict_empty(self):
        """Test flatten_dict with empty dictionary."""
        # Local application imports
        from twin4build.utils.dict_utils import flatten_dict

        class TestObj:
            pass

        obj = TestObj()
        flattened = flatten_dict({}, obj)
        self.assertEqual(len(flattened), 0)

    def test_flatten_dict_non_dict(self):
        """Test flatten_dict with non-dictionary input."""
        # Local application imports
        from twin4build.utils.dict_utils import flatten_dict

        class TestObj:
            pass

        obj = TestObj()
        flattened = flatten_dict("not a dict", obj)
        self.assertEqual(len(flattened), 0)


class TestMkdirInRoot(unittest.TestCase):
    def test_mkdir_in_root_basic(self):
        """Test mkdir_in_root creates directories."""
        # Standard library imports
        import shutil
        import tempfile

        # Local application imports
        from twin4build.utils.mkdir_in_root import mkdir_in_root

        # Use a temp directory as root
        with tempfile.TemporaryDirectory() as tmpdir:
            result, isfile = mkdir_in_root(
                folder_list=["test_folder", "subfolder"],
                filename="test.txt",
                root=tmpdir,
            )

            self.assertIsNotNone(result)
            self.assertTrue(result.endswith("test.txt"))


class TestGetMainDir(unittest.TestCase):
    def test_get_main_dir(self):
        """Test get_main_dir returns a valid directory."""
        # Standard library imports
        import os

        # Local application imports
        from twin4build.utils.get_main_dir import get_main_dir

        # Should return the main directory of the project
        main_dir = get_main_dir()
        self.assertIsNotNone(main_dir)
        self.assertIsInstance(main_dir, str)
        # Should be an existing directory
        self.assertTrue(os.path.isdir(main_dir))

    def test_get_main_dir_contains_twin4build(self):
        """Test get_main_dir path contains twin4build."""
        # Local application imports
        from twin4build.utils.get_main_dir import get_main_dir

        main_dir = get_main_dir()
        # The main directory should contain twin4build somewhere in the path
        self.assertIn("Twin4Build", main_dir) or self.assertIn(
            "twin4build", main_dir.lower()
        )


class TestPrintProgress(unittest.TestCase):
    def test_print_progress_initialization(self):
        """Test PrintProgress initialization."""
        # Local application imports
        from twin4build.utils.print_progress import PrintProgress

        p = PrintProgress()
        self.assertIsNotNone(p)
        self.assertFalse(p.is_active)
        self.assertEqual(p.verbose, 3)
        # Auto-disabled in test environments
        self.assertFalse(p.enabled)

    def test_print_progress_enable_disable(self):
        """Test PrintProgress enable/disable functionality."""
        # Local application imports
        from twin4build.utils.print_progress import PrintProgress

        p = PrintProgress()
        # Auto-disabled in test environments
        self.assertFalse(p.enabled)

        p.enable()
        self.assertTrue(p.enabled)

        p.disable()
        self.assertFalse(p.enabled)

    def test_print_progress_verbose_setting(self):
        """Test PrintProgress verbose setting."""
        # Local application imports
        from twin4build.utils.print_progress import PrintProgress

        p = PrintProgress()
        self.assertEqual(p.verbose, 3)

        p.verbose = 2
        self.assertEqual(p.verbose, 2)

        p.verbose = 0
        self.assertEqual(p.verbose, 0)

    def test_print_progress_add_line(self):
        """Test PrintProgress add_line method."""
        # Local application imports
        from twin4build.utils.print_progress import PrintProgress

        p = PrintProgress()
        p.add_line(indent="  ", message="Test message", status="OK")

        self.assertTrue(p.is_active)
        self.assertEqual(len(p.message), 1)
        self.assertEqual(p.message[0], "Test message")
        self.assertEqual(p.status[0], "OK")

    def test_print_progress_get_char_level(self):
        """Test PrintProgress get_char_level method."""
        # Local application imports
        from twin4build.utils.print_progress import PrintProgress

        p = PrintProgress()

        # Test with a line containing vertical bars
        line = "|__|test|data"
        char_levels = p.get_char_level(line)

        self.assertIsNotNone(char_levels)
        self.assertEqual(len(char_levels), len(line))

    def test_print_progress_current_level(self):
        """Test PrintProgress current_level property."""
        # Local application imports
        from twin4build.utils.print_progress import PrintProgress

        p = PrintProgress()
        self.assertEqual(p.current_level, 0)

    def test_print_progress_add_remove_level(self):
        """Test PrintProgress add_level and remove_level."""
        # Standard library imports
        from unittest.mock import patch

        # Local application imports
        from twin4build.utils.print_progress import PrintProgress

        p = PrintProgress()
        p.enable()  # Enable to test add_level functionality

        # Mock only print_lines to prevent terminal manipulation
        # but allow _add_level to run so internal state is correct
        with patch.object(p, "print_lines"):
            p.add_level(n=2)
            self.assertTrue(p.added_level)

            p.remove_level()
            self.assertFalse(p.added_level)

        p.disable()

    def test_print_progress_call(self):
        """Test PrintProgress __call__ method."""
        # Standard library imports
        from unittest.mock import patch

        # Local application imports
        from twin4build.utils.print_progress import PrintProgress

        p = PrintProgress()
        p.enable()  # Enable for this test (auto-disabled in test environments)
        p.verbose = 3

        # Mock print_lines to prevent any terminal manipulation
        with patch.object(p, "print_lines"):
            p("Test message", status="INFO")

        p.disable()

        self.assertTrue(p.is_active)
        self.assertIn("Test message", p.message)

    def test_print_progress_call_disabled(self):
        """Test PrintProgress __call__ when disabled."""
        # Local application imports
        from twin4build.utils.print_progress import PrintProgress

        p = PrintProgress()
        # Instance is auto-disabled in test environments
        self.assertFalse(p.enabled)

        # Call with a message - should do nothing when disabled
        p("Test message", status="INFO")

        # Should not have added any lines
        self.assertEqual(len(p.message), 0)

    def test_print_progress_context_manager(self):
        """Test PrintProgress as context manager."""
        # Local application imports
        from twin4build.utils.print_progress import PrintProgress

        # Instance is auto-disabled in test environments
        with PrintProgress() as p:
            self.assertIsNotNone(p)
            self.assertFalse(p.enabled)

    def test_print_progress_is_interactive(self):
        """Test PrintProgress is_interactive method."""
        # Local application imports
        from twin4build.utils.print_progress import PrintProgress

        p = PrintProgress()
        # This should return True or False depending on environment
        result = p.is_interactive()
        self.assertIsInstance(result, bool)


class TestSimpleCycle(unittest.TestCase):
    def test_simple_cycles(self):
        """Test simple_cycles function for detecting cycles in a graph."""
        # Local application imports
        from twin4build.utils.simple_cycle import simple_cycles

        # Create a simple graph with a cycle: A -> B -> C -> A
        graph = {"A": {"B"}, "B": {"C"}, "C": {"A"}}

        cycles = list(simple_cycles(graph))
        self.assertEqual(len(cycles), 1)
        self.assertEqual(len(cycles[0]), 3)

    def test_no_cycles(self):
        """Test simple_cycles with acyclic graph."""
        # Local application imports
        from twin4build.utils.simple_cycle import simple_cycles

        # Acyclic graph: A -> B -> C
        graph = {"A": {"B"}, "B": {"C"}, "C": set()}

        cycles = list(simple_cycles(graph))
        self.assertEqual(len(cycles), 0)

    def test_multiple_cycles(self):
        """Test simple_cycles with multiple cycles."""
        # Local application imports
        from twin4build.utils.simple_cycle import simple_cycles

        # Graph with two cycles
        graph = {"A": {"B"}, "B": {"A", "C"}, "C": {"D"}, "D": {"C"}}

        cycles = list(simple_cycles(graph))
        self.assertGreaterEqual(len(cycles), 1)


class TestConstants(unittest.TestCase):
    def test_constants_import(self):
        """Test that constants can be imported."""
        # Local application imports
        import twin4build.utils.constants as constants

        self.assertIsNotNone(constants.ABSOLUTE_ZERO_CELSIUS)
        self.assertAlmostEqual(constants.ABSOLUTE_ZERO_CELSIUS, -273.15, places=2)


if __name__ == "__main__":
    unittest.main()
