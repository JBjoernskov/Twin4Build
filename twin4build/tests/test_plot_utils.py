import unittest
import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import tempfile
import os



class TestEntry(unittest.TestCase):
    def test_entry_initialization_with_numpy(self):
        """Test Entry initialization with numpy array."""
        from twin4build.utils.plot import Entry
        
        data = np.array([1.0, 2.0, 3.0, 4.0])
        entry = Entry(data=data, label="Test Data")
        
        self.assertIsNotNone(entry)
        self.assertEqual(entry.label, "Test Data")
        np.testing.assert_array_equal(entry.data, data)
        self.assertEqual(entry.axis, 1)  # Default axis
    
    def test_entry_initialization_with_list(self):
        """Test Entry initialization with list (converts to numpy)."""
        from twin4build.utils.plot import Entry
        
        data = [1.0, 2.0, 3.0, 4.0]
        entry = Entry(data=data, label="Test Data")
        
        self.assertIsInstance(entry.data, np.ndarray)
        np.testing.assert_array_equal(entry.data, np.array(data))
    
    def test_entry_initialization_with_tensor(self):
        """Test Entry initialization with torch tensor."""
        from twin4build.utils.plot import Entry
        
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        entry = Entry(data=data, label="Test Data")
        
        self.assertIsInstance(entry.data, np.ndarray)
        np.testing.assert_array_almost_equal(entry.data, data.numpy())
    
    def test_entry_initialization_with_pandas_series(self):
        """Test Entry initialization with pandas Series."""
        from twin4build.utils.plot import Entry
        
        data = pd.Series([1.0, 2.0, 3.0, 4.0])
        entry = Entry(data=data, label="Test Data")
        
        self.assertIsInstance(entry.data, np.ndarray)
        np.testing.assert_array_equal(entry.data, data.values)
    
    def test_entry_with_custom_styling(self):
        """Test Entry with custom styling parameters."""
        from twin4build.utils.plot import Entry
        
        data = np.array([1.0, 2.0, 3.0])
        entry = Entry(
            data=data,
            label="Styled Data",
            color="red",
            fmt="--",
            axis=2,
            linewidth=3
        )
        
        self.assertEqual(entry.label, "Styled Data")
        self.assertEqual(entry.color, "red")
        self.assertEqual(entry.fmt, "--")
        self.assertEqual(entry.axis, 2)
        self.assertEqual(entry.linewidth, 3)
    
    def test_entry_missing_data_raises_error(self):
        """Test that Entry raises ValueError when data is None."""
        from twin4build.utils.plot import Entry
        
        with self.assertRaises(ValueError) as context:
            Entry(data=None, label="No Data")
        
        self.assertIn("data", str(context.exception).lower())
    
    def test_entry_missing_label_raises_error(self):
        """Test that Entry raises AssertionError when label is missing."""
        from twin4build.utils.plot import Entry
        
        data = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(AssertionError):
            Entry(data=data)
    
    def test_entry_deprecated_attribute_parameter(self):
        """Test Entry with deprecated 'attribute' parameter."""
        from twin4build.utils.plot import Entry
        
        data = np.array([1.0, 2.0, 3.0])
        with self.assertWarns(DeprecationWarning):
            entry = Entry(data=data, attribute="Old Style Label")
        
        self.assertEqual(entry.label, "Old Style Label")
    
    def test_entry_deprecated_linestyle_parameter(self):
        """Test Entry with deprecated 'linestyle' parameter."""
        from twin4build.utils.plot import Entry
        
        data = np.array([1.0, 2.0, 3.0])
        with self.assertWarns(DeprecationWarning):
            entry = Entry(data=data, label="Test", linestyle="--")
        
        self.assertEqual(entry.fmt, "--")
    
    def test_entry_deprecated_component_parameters(self):
        """Test Entry with deprecated component-based parameters."""
        from twin4build.utils.plot import Entry
        
        data = np.array([1.0, 2.0, 3.0])
        with self.assertWarns(DeprecationWarning):
            entry = Entry(data=data, label="Test", component="some_component")


class TestColors(unittest.TestCase):
    def test_colors_attributes_exist(self):
        """Test that Colors class has expected color attributes."""
        from twin4build.utils.plot.plot import Colors
        
        self.assertIsNotNone(Colors.blue)
        self.assertIsNotNone(Colors.orange)
        self.assertIsNotNone(Colors.green)
        self.assertIsNotNone(Colors.red)
        self.assertIsNotNone(Colors.purple)
        self.assertIsNotNone(Colors.brown)
        self.assertIsNotNone(Colors.pink)
        self.assertIsNotNone(Colors.grey)
        self.assertIsNotNone(Colors.black)
    
    def test_colors_black_is_string(self):
        """Test that black color is explicitly 'black' string."""
        from twin4build.utils.plot.plot import Colors
        
        self.assertEqual(Colors.black, "black")
    
    def test_colors_from_seaborn(self):
        """Test that colors are tuples (from seaborn)."""
        from twin4build.utils.plot.plot import Colors
        
        # Seaborn colors should be tuples of RGB values
        self.assertIsInstance(Colors.blue, tuple)
        self.assertIsInstance(Colors.red, tuple)
        self.assertIsInstance(Colors.green, tuple)


class TestPlotSettings(unittest.TestCase):
    def test_plot_settings_attributes(self):
        """Test PlotSettings has expected attributes."""
        from twin4build.utils.plot.plot import PlotSettings
        
        self.assertIsNotNone(PlotSettings.legend_loc)
        self.assertIsNotNone(PlotSettings.x)
        self.assertIsNotNone(PlotSettings.left_y)
        self.assertIsNotNone(PlotSettings.right_y_first)
        self.assertIsNotNone(PlotSettings.right_y_second)
        self.assertIsNotNone(PlotSettings.outward)
    
    def test_plot_settings_legend_loc_format(self):
        """Test legend_loc is a tuple of coordinates."""
        from twin4build.utils.plot.plot import PlotSettings
        
        self.assertIsInstance(PlotSettings.legend_loc, tuple)
        self.assertEqual(len(PlotSettings.legend_loc), 2)
    
    def test_plot_settings_save_folder(self):
        """Test save_folder method returns a valid path."""
        from twin4build.utils.plot.plot import PlotSettings
        
        folder = PlotSettings.save_folder()
        self.assertIsNotNone(folder)
        self.assertIsInstance(folder, str)
        # Should be a valid directory
        self.assertTrue(os.path.isdir(folder))


class TestPlotHelperFunctions(unittest.TestCase):
    def test_get_file_name_basic(self):
        """Test get_file_name converts names correctly."""
        from twin4build.utils.plot.plot import get_file_name
        
        result = get_file_name("Test Plot")
        self.assertEqual(result, "plot_test_plot")
    
    def test_get_file_name_with_spaces(self):
        """Test get_file_name handles multiple spaces."""
        from twin4build.utils.plot.plot import get_file_name
        
        result = get_file_name("My Test   Plot Name")
        self.assertEqual(result, "plot_my_test___plot_name")
    
    def test_get_file_name_case_insensitive(self):
        """Test get_file_name converts to lowercase."""
        from twin4build.utils.plot.plot import get_file_name
        
        result = get_file_name("UPPERCASE")
        self.assertEqual(result, "plot_uppercase")
    
    def test_bar_plot_line_format_hourly(self):
        """Test bar_plot_line_format with hourly data."""
        from twin4build.utils.plot.plot import bar_plot_line_format
        
        # Create a timestamp
        timestamp = pd.Timestamp('2023-01-01 14:30:00')
        result = bar_plot_line_format(timestamp, "H")
        
        self.assertEqual(result, "14")
    
    def test_bar_plot_line_format_hourly_midnight(self):
        """Test bar_plot_line_format at midnight includes day."""
        from twin4build.utils.plot.plot import bar_plot_line_format
        
        timestamp = pd.Timestamp('2023-01-02 00:00:00')  # Monday
        result = bar_plot_line_format(timestamp, "H")
        
        self.assertIn("00", result)
        self.assertIn("Mon", result)
    
    def test_bar_plot_line_format_daily(self):
        """Test bar_plot_line_format with daily data."""
        from twin4build.utils.plot.plot import bar_plot_line_format
        
        timestamp = pd.Timestamp('2023-01-04 12:00:00')  # Wednesday
        result = bar_plot_line_format(timestamp, "D")
        
        self.assertEqual(result, "Wed")
    
    def test_bar_plot_line_format_daily_monday(self):
        """Test bar_plot_line_format on Monday includes week."""
        from twin4build.utils.plot.plot import bar_plot_line_format
        
        timestamp = pd.Timestamp('2023-01-02 12:00:00')  # Monday
        result = bar_plot_line_format(timestamp, "D")
        
        self.assertIn("Mon", result)
        self.assertIn("week", result.lower())
    
    def test_bar_plot_line_format_weekly(self):
        """Test bar_plot_line_format with weekly data."""
        from twin4build.utils.plot.plot import bar_plot_line_format
        
        timestamp = pd.Timestamp('2023-01-15 12:00:00')
        result = bar_plot_line_format(timestamp, "W")
        
        # Should return week number as 2-digit string
        week_num = timestamp.isocalendar()[1]
        self.assertIn(f"{week_num:02d}", result)
    
    def test_bar_plot_line_format_monthly(self):
        """Test bar_plot_line_format with monthly data."""
        from twin4build.utils.plot.plot import bar_plot_line_format
        
        timestamp = pd.Timestamp('2023-03-15 12:00:00')
        result = bar_plot_line_format(timestamp, "M")
        
        self.assertEqual(result, "Mar")
    
    def test_bar_plot_line_format_monthly_january(self):
        """Test bar_plot_line_format in January includes year."""
        from twin4build.utils.plot.plot import bar_plot_line_format
        
        timestamp = pd.Timestamp('2023-01-15 12:00:00')
        result = bar_plot_line_format(timestamp, "M")
        
        self.assertIn("Jan", result)
        self.assertIn("2023", result)
    
    def test_bar_plot_line_format_annual(self):
        """Test bar_plot_line_format with annual data."""
        from twin4build.utils.plot.plot import bar_plot_line_format
        
        timestamp = pd.Timestamp('2023-06-15 12:00:00')
        result = bar_plot_line_format(timestamp, "A")
        
        # Annual format returns month name
        self.assertEqual(result, "Jun")


class TestAlignYAxes(unittest.TestCase):
    def test_calculate_ticks_basic(self):
        """Test calculate_ticks with basic parameters."""
        from twin4build.utils.plot.align_y_axes import calculate_ticks
        
        # Create a mock axis
        fig, ax = plt.subplots()
        ax.set_ylim(0, 100)
        
        ticks = calculate_ticks(ax, nticks=5)
        
        self.assertIsInstance(ticks, np.ndarray)
        self.assertEqual(len(ticks), 5)
        self.assertTrue(ticks[0] <= 0)
        self.assertTrue(ticks[-1] >= 100)
        
        plt.close(fig)
    
    def test_calculate_ticks_with_round_to(self):
        """Test calculate_ticks with explicit round_to parameter."""
        from twin4build.utils.plot.align_y_axes import calculate_ticks
        
        fig, ax = plt.subplots()
        ax.set_ylim(0, 50)
        
        ticks = calculate_ticks(ax, nticks=6, round_to=10)
        
        self.assertIsInstance(ticks, np.ndarray)
        # Ticks should be multiples of 10
        for tick in ticks:
            self.assertAlmostEqual(tick % 10, 0, places=5)
        
        plt.close(fig)
    
    def test_calculate_ticks_zero_crossing(self):
        """Test calculate_ticks with data crossing zero."""
        from twin4build.utils.plot.align_y_axes import calculate_ticks
        
        fig, ax = plt.subplots()
        ax.set_ylim(-50, 50)
        
        ticks = calculate_ticks(ax, nticks=5, zero_tick_idx=2)
        
        self.assertIsInstance(ticks, np.ndarray)
        self.assertEqual(len(ticks), 5)
        # Check that one tick is at or near zero
        self.assertTrue(any(abs(tick) < 1e-10 for tick in ticks))
        
        plt.close(fig)
    
    def test_calculate_ticks_negative_range(self):
        """Test calculate_ticks with entirely negative range."""
        from twin4build.utils.plot.align_y_axes import calculate_ticks
        
        fig, ax = plt.subplots()
        ax.set_ylim(-100, -10)
        
        ticks = calculate_ticks(ax, nticks=5)
        
        self.assertIsInstance(ticks, np.ndarray)
        self.assertTrue(all(tick <= -10 for tick in ticks[:-1] if not np.isclose(tick, -10)))
        self.assertTrue(ticks[0] <= -100)
        
        plt.close(fig)
    
    def test_alignYaxes_two_axes(self):
        """Test alignYaxes with two axes."""
        from twin4build.utils.plot.align_y_axes import alignYaxes
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        # Set some data ranges
        ax1.set_ylim(0, 100)
        ax2.set_ylim(0, 1000)
        
        alignYaxes([ax1, ax2], [5, 5], [None, None], [None, 0.1])
        
        # Both axes should now have 5 ticks
        self.assertEqual(len(ax1.get_yticks()), 5)
        self.assertEqual(len(ax2.get_yticks()), 5)
        
        plt.close(fig)
    
    def test_alignYaxes_three_axes(self):
        """Test alignYaxes with three axes."""
        from twin4build.utils.plot.align_y_axes import alignYaxes
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        
        # Offset the third axis
        ax3.spines['right'].set_position(('outward', 60))
        
        # Set some data ranges
        ax1.set_ylim(0, 100)
        ax2.set_ylim(0, 1000)
        ax3.set_ylim(-50, 50)
        
        alignYaxes([ax1, ax2, ax3], [5, 5, 5], [None, None, None], [None, None, 0.1])
        
        # All axes should have 5 ticks
        self.assertEqual(len(ax1.get_yticks()), 5)
        self.assertEqual(len(ax2.get_yticks()), 5)
        self.assertEqual(len(ax3.get_yticks()), 5)
        
        plt.close(fig)
    
    def test_alignYaxes_all_none_yoffsets_raises_assertion(self):
        """Test that alignYaxes raises AssertionError when all yoffsets are None."""
        from twin4build.utils.plot.align_y_axes import alignYaxes
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        # Set some data ranges
        ax1.set_ylim(0, 100)
        ax2.set_ylim(0, 1000)
        
        # Should raise AssertionError when all yoffsets are None
        with self.assertRaises(AssertionError) as context:
            alignYaxes([ax1, ax2], [5, 5], [None, None], [None, None])
                
        plt.close(fig)
    
    def test_alignYaxes_with_some_yoffsets_not_none(self):
        """Test that alignYaxes works correctly when at least one yoffset is not None."""
        from twin4build.utils.plot.align_y_axes import alignYaxes
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        # Set some data ranges
        ax1.set_ylim(0, 100)
        ax2.set_ylim(0, 1000)
        
        # Should not raise when at least one yoffset is not None
        # If it raises, the test will fail automatically
        alignYaxes([ax1, ax2], [5, 5], [None, None], [None, 0.1])
        
        plt.close(fig)


class TestLoadParams(unittest.TestCase):
    def test_load_params_executes(self):
        """Test that load_params executes without error."""
        from twin4build.utils.plot.plot import load_params
        
        # Should not raise any exception
        try:
            load_params()
            success = True
        except Exception as e:
            success = False
            print(f"load_params raised: {e}")
        
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()

