# Standard library imports
import datetime
import os
import tempfile
import unittest

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Local application imports
# Set test flag
import twin4build

twin4build._IS_TESTING = True


class TestEntry(unittest.TestCase):
    def test_entry_initialization_with_numpy(self):
        """Test Entry initialization with numpy array."""
        # Local application imports
        from twin4build.utils.plot import Entry

        data = np.array([1.0, 2.0, 3.0, 4.0])
        entry = Entry(data=data, label="Test Data")

        self.assertIsNotNone(entry)
        self.assertEqual(entry.label, "Test Data")
        np.testing.assert_array_equal(entry.data, data)
        self.assertEqual(entry.axis, 1)  # Default axis

    def test_entry_initialization_with_list(self):
        """Test Entry initialization with list (converts to numpy)."""
        # Local application imports
        from twin4build.utils.plot import Entry

        data = [1.0, 2.0, 3.0, 4.0]
        entry = Entry(data=data, label="Test Data")

        self.assertIsInstance(entry.data, np.ndarray)
        np.testing.assert_array_equal(entry.data, np.array(data))

    def test_entry_initialization_with_tensor(self):
        """Test Entry initialization with torch tensor."""
        # Local application imports
        from twin4build.utils.plot import Entry

        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        entry = Entry(data=data, label="Test Data")

        self.assertIsInstance(entry.data, np.ndarray)
        np.testing.assert_array_almost_equal(entry.data, data.numpy())

    def test_entry_initialization_with_pandas_series(self):
        """Test Entry initialization with pandas Series."""
        # Local application imports
        from twin4build.utils.plot import Entry

        data = pd.Series([1.0, 2.0, 3.0, 4.0])
        entry = Entry(data=data, label="Test Data")

        self.assertIsInstance(entry.data, np.ndarray)
        np.testing.assert_array_equal(entry.data, data.values)

    def test_entry_with_custom_styling(self):
        """Test Entry with custom styling parameters."""
        # Local application imports
        from twin4build.utils.plot import Entry

        data = np.array([1.0, 2.0, 3.0])
        entry = Entry(
            data=data, label="Styled Data", color="red", fmt="--", axis=2, linewidth=3
        )

        self.assertEqual(entry.label, "Styled Data")
        self.assertEqual(entry.color, "red")
        self.assertEqual(entry.fmt, "--")
        self.assertEqual(entry.axis, 2)
        self.assertEqual(entry.linewidth, 3)

    def test_entry_missing_data_raises_error(self):
        """Test that Entry raises ValueError when data is None."""
        # Local application imports
        from twin4build.utils.plot import Entry

        with self.assertRaises(ValueError) as context:
            Entry(data=None, label="No Data")

        self.assertIn("data", str(context.exception).lower())

    def test_entry_missing_label_raises_error(self):
        """Test that Entry raises AssertionError when label is missing."""
        # Local application imports
        from twin4build.utils.plot import Entry

        data = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(AssertionError):
            Entry(data=data)

    def test_entry_deprecated_attribute_parameter(self):
        """Test Entry with deprecated 'attribute' parameter."""
        # Local application imports
        from twin4build.utils.plot import Entry

        data = np.array([1.0, 2.0, 3.0])
        with self.assertWarns(DeprecationWarning):
            entry = Entry(data=data, attribute="Old Style Label")

        self.assertEqual(entry.label, "Old Style Label")

    def test_entry_deprecated_linestyle_parameter(self):
        """Test Entry with deprecated 'linestyle' parameter."""
        # Local application imports
        from twin4build.utils.plot import Entry

        data = np.array([1.0, 2.0, 3.0])
        with self.assertWarns(DeprecationWarning):
            entry = Entry(data=data, label="Test", linestyle="--")

        self.assertEqual(entry.fmt, "--")

    def test_entry_deprecated_component_parameters(self):
        """Test Entry with deprecated component-based parameters."""
        # Local application imports
        from twin4build.utils.plot import Entry

        data = np.array([1.0, 2.0, 3.0])
        with self.assertWarns(DeprecationWarning):
            entry = Entry(data=data, label="Test", component="some_component")


class TestColors(unittest.TestCase):
    def test_colors_attributes_exist(self):
        """Test that Colors class has expected color attributes."""
        # Local application imports
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
        # Local application imports
        from twin4build.utils.plot.plot import Colors

        self.assertEqual(Colors.black, "black")

    def test_colors_from_seaborn(self):
        """Test that colors are tuples (from seaborn)."""
        # Local application imports
        from twin4build.utils.plot.plot import Colors

        # Seaborn colors should be tuples of RGB values
        self.assertIsInstance(Colors.blue, tuple)
        self.assertIsInstance(Colors.red, tuple)
        self.assertIsInstance(Colors.green, tuple)


class TestPlotSettings(unittest.TestCase):
    def test_plot_settings_attributes(self):
        """Test PlotSettings has expected attributes."""
        # Local application imports
        from twin4build.utils.plot.plot import PlotSettings

        self.assertIsNotNone(PlotSettings.legend_loc)
        self.assertIsNotNone(PlotSettings.x)
        self.assertIsNotNone(PlotSettings.left_y)
        self.assertIsNotNone(PlotSettings.right_y_first)
        self.assertIsNotNone(PlotSettings.right_y_second)
        self.assertIsNotNone(PlotSettings.outward)

    def test_plot_settings_legend_loc_format(self):
        """Test legend_loc is a tuple of coordinates."""
        # Local application imports
        from twin4build.utils.plot.plot import PlotSettings

        self.assertIsInstance(PlotSettings.legend_loc, tuple)
        self.assertEqual(len(PlotSettings.legend_loc), 2)

    def test_plot_settings_save_folder(self):
        """Test save_folder method returns a valid path."""
        # Local application imports
        from twin4build.utils.plot.plot import PlotSettings

        folder = PlotSettings.save_folder()
        self.assertIsNotNone(folder)
        self.assertIsInstance(folder, str)
        # Should be a valid directory
        self.assertTrue(os.path.isdir(folder))


class TestPlotHelperFunctions(unittest.TestCase):
    def test_get_file_name_basic(self):
        """Test get_file_name converts names correctly."""
        # Local application imports
        from twin4build.utils.plot.plot import get_file_name

        result = get_file_name("Test Plot")
        self.assertEqual(result, "plot_test_plot")

    def test_get_file_name_with_spaces(self):
        """Test get_file_name handles multiple spaces."""
        # Local application imports
        from twin4build.utils.plot.plot import get_file_name

        result = get_file_name("My Test   Plot Name")
        self.assertEqual(result, "plot_my_test___plot_name")

    def test_get_file_name_case_insensitive(self):
        """Test get_file_name converts to lowercase."""
        # Local application imports
        from twin4build.utils.plot.plot import get_file_name

        result = get_file_name("UPPERCASE")
        self.assertEqual(result, "plot_uppercase")

    def test_bar_plot_line_format_hourly(self):
        """Test bar_plot_line_format with hourly data."""
        # Local application imports
        from twin4build.utils.plot.plot import bar_plot_line_format

        # Create a timestamp
        timestamp = pd.Timestamp("2023-01-01 14:30:00")
        result = bar_plot_line_format(timestamp, "H")

        self.assertEqual(result, "14")

    def test_bar_plot_line_format_hourly_midnight(self):
        """Test bar_plot_line_format at midnight includes day."""
        # Local application imports
        from twin4build.utils.plot.plot import bar_plot_line_format

        timestamp = pd.Timestamp("2023-01-02 00:00:00")  # Monday
        result = bar_plot_line_format(timestamp, "H")

        self.assertIn("00", result)
        self.assertIn("Mon", result)

    def test_bar_plot_line_format_daily(self):
        """Test bar_plot_line_format with daily data."""
        # Local application imports
        from twin4build.utils.plot.plot import bar_plot_line_format

        timestamp = pd.Timestamp("2023-01-04 12:00:00")  # Wednesday
        result = bar_plot_line_format(timestamp, "D")

        self.assertEqual(result, "Wed")

    def test_bar_plot_line_format_daily_monday(self):
        """Test bar_plot_line_format on Monday includes week."""
        # Local application imports
        from twin4build.utils.plot.plot import bar_plot_line_format

        timestamp = pd.Timestamp("2023-01-02 12:00:00")  # Monday
        result = bar_plot_line_format(timestamp, "D")

        self.assertIn("Mon", result)
        self.assertIn("week", result.lower())

    def test_bar_plot_line_format_weekly(self):
        """Test bar_plot_line_format with weekly data."""
        # Local application imports
        from twin4build.utils.plot.plot import bar_plot_line_format

        timestamp = pd.Timestamp("2023-01-15 12:00:00")
        result = bar_plot_line_format(timestamp, "W")

        # Should return week number as 2-digit string
        week_num = timestamp.isocalendar()[1]
        self.assertIn(f"{week_num:02d}", result)

    def test_bar_plot_line_format_monthly(self):
        """Test bar_plot_line_format with monthly data."""
        # Local application imports
        from twin4build.utils.plot.plot import bar_plot_line_format

        timestamp = pd.Timestamp("2023-03-15 12:00:00")
        result = bar_plot_line_format(timestamp, "M")

        self.assertEqual(result, "Mar")

    def test_bar_plot_line_format_monthly_january(self):
        """Test bar_plot_line_format in January includes year."""
        # Local application imports
        from twin4build.utils.plot.plot import bar_plot_line_format

        timestamp = pd.Timestamp("2023-01-15 12:00:00")
        result = bar_plot_line_format(timestamp, "M")

        self.assertIn("Jan", result)
        self.assertIn("2023", result)

    def test_bar_plot_line_format_annual(self):
        """Test bar_plot_line_format with annual data."""
        # Local application imports
        from twin4build.utils.plot.plot import bar_plot_line_format

        timestamp = pd.Timestamp("2023-06-15 12:00:00")
        result = bar_plot_line_format(timestamp, "A")

        # Annual format returns month name
        self.assertEqual(result, "Jun")


class TestAlignYAxes(unittest.TestCase):
    def test_calculate_ticks_basic(self):
        """Test calculate_ticks with basic parameters."""
        # Local application imports
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
        # Local application imports
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
        # Local application imports
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
        # Local application imports
        from twin4build.utils.plot.align_y_axes import calculate_ticks

        fig, ax = plt.subplots()
        ax.set_ylim(-100, -10)

        ticks = calculate_ticks(ax, nticks=5)

        self.assertIsInstance(ticks, np.ndarray)
        self.assertTrue(
            all(tick <= -10 for tick in ticks[:-1] if not np.isclose(tick, -10))
        )
        self.assertTrue(ticks[0] <= -100)

        plt.close(fig)

    def test_alignYaxes_two_axes(self):
        """Test alignYaxes with two axes."""
        # Local application imports
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
        # Local application imports
        from twin4build.utils.plot.align_y_axes import alignYaxes

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()

        # Offset the third axis
        ax3.spines["right"].set_position(("outward", 60))

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
        # Local application imports
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
        # Local application imports
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
        # Local application imports
        from twin4build.utils.plot.plot import load_params

        # Should not raise any exception
        try:
            load_params()
            success = True
        except Exception as e:
            success = False
            print(f"load_params raised: {e}")

        self.assertTrue(success)


class TestOnPick(unittest.TestCase):
    """Tests for on_pick event handler."""

    def test_on_pick_toggle_visibility(self):
        """Test on_pick toggles line visibility."""
        # Standard library imports
        from unittest.mock import MagicMock, Mock

        # Local application imports
        from twin4build.utils.plot.plot import on_pick

        # Create mock figure and line
        fig = Mock()
        fig.canvas = Mock()
        fig.canvas.draw_idle = Mock()

        # Create mock line
        mock_line = Mock()
        mock_line.get_visible.return_value = True
        mock_line.set_visible = Mock()

        # Create mock legend item
        mock_legend = Mock()
        mock_legend.set_alpha = Mock()

        # Create graphs dict mapping legend to line
        graphs = {mock_legend: mock_line}

        # Create mock event
        event = Mock()
        event.artist = mock_legend

        # Call on_pick
        on_pick(event, fig, graphs)

        # Line should be toggled to not visible
        mock_line.set_visible.assert_called_with(False)
        mock_legend.set_alpha.assert_called_with(0.2)
        fig.canvas.draw_idle.assert_called_once()

    def test_on_pick_make_visible(self):
        """Test on_pick makes hidden line visible."""
        # Standard library imports
        from unittest.mock import Mock

        # Local application imports
        from twin4build.utils.plot.plot import on_pick

        fig = Mock()
        fig.canvas = Mock()
        fig.canvas.draw_idle = Mock()

        mock_line = Mock()
        mock_line.get_visible.return_value = False
        mock_line.set_visible = Mock()

        mock_legend = Mock()
        mock_legend.set_alpha = Mock()

        graphs = {mock_legend: mock_line}

        event = Mock()
        event.artist = mock_legend

        on_pick(event, fig, graphs)

        # Line should be toggled to visible
        mock_line.set_visible.assert_called_with(True)
        mock_legend.set_alpha.assert_called_with(1)


class TestFilterNans(unittest.TestCase):
    """Tests for filter_nans function."""

    def test_filter_nans_no_nans(self):
        """Test filter_nans with no NaN values."""
        # Local application imports
        from twin4build.utils.plot.plot import filter_nans

        time = np.array([1.0, 2.0, 3.0, 4.0])
        data = np.array([10.0, 20.0, 30.0, 40.0])

        filtered_time, filtered_data = filter_nans(time, data)

        np.testing.assert_array_equal(filtered_time, time)
        np.testing.assert_array_equal(filtered_data, data)

    def test_filter_nans_with_nans_in_time(self):
        """Test filter_nans with NaN values in time array."""
        # Local application imports
        from twin4build.utils.plot.plot import filter_nans

        time = np.array([1.0, np.nan, 3.0, 4.0])
        data = np.array([10.0, 20.0, 30.0, 40.0])

        filtered_time, filtered_data = filter_nans(time, data)

        np.testing.assert_array_equal(filtered_time, np.array([1.0, 3.0, 4.0]))
        np.testing.assert_array_equal(filtered_data, np.array([10.0, 30.0, 40.0]))

    def test_filter_nans_with_pandas_nat(self):
        """Test filter_nans with pandas NaT values."""
        # Local application imports
        from twin4build.utils.plot.plot import filter_nans

        time = pd.Series(
            [pd.Timestamp("2023-01-01"), pd.NaT, pd.Timestamp("2023-01-03")]
        )
        data = np.array([10.0, 20.0, 30.0])

        filtered_time, filtered_data = filter_nans(time.values, data)

        self.assertEqual(len(filtered_time), 2)
        self.assertEqual(len(filtered_data), 2)

    def test_filter_nans_all_nans(self):
        """Test filter_nans when all time values are NaN."""
        # Local application imports
        from twin4build.utils.plot.plot import filter_nans

        time = np.array([np.nan, np.nan, np.nan])
        data = np.array([10.0, 20.0, 30.0])

        filtered_time, filtered_data = filter_nans(time, data)

        self.assertEqual(len(filtered_time), 0)
        self.assertEqual(len(filtered_data), 0)


class TestGetData(unittest.TestCase):
    """Tests for get_data function."""

    def test_get_data_with_entry_object(self):
        """Test get_data with Entry object."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, get_data

        data = np.array([1.0, 2.0, 3.0])
        entry = Entry(data=data, label="Test", fmt="--", axis=2)

        parsed_data, fmt, axis, kwargs = get_data(entry)

        self.assertIsNotNone(parsed_data)
        self.assertEqual(fmt, "--")
        self.assertEqual(axis, 2)
        self.assertEqual(kwargs.get("label"), "Test")

    def test_get_data_with_tuple_deprecated(self):
        """Test get_data with deprecated tuple format."""
        # Local application imports
        from twin4build.utils.plot.plot import get_data

        data = np.array([1.0, 2.0, 3.0])
        t = (data, "Test Label")

        with self.assertWarns(DeprecationWarning):
            parsed_data, fmt, axis, kwargs = get_data(t)

        self.assertIsNotNone(parsed_data)
        self.assertIsNone(fmt)
        self.assertIsNone(axis)
        self.assertEqual(kwargs.get("label"), "Test Label")

    def test_get_data_with_invalid_type(self):
        """Test get_data with invalid type raises error."""
        # Local application imports
        from twin4build.utils.plot.plot import get_data

        with self.assertRaises(ValueError):
            get_data("invalid")

    def test_get_data_with_tuple_wrong_length(self):
        """Test get_data with tuple of wrong length returns None."""
        # Local application imports
        from twin4build.utils.plot.plot import get_data

        # Tuple with more than 2 elements returns None for legacy component processing
        t = (np.array([1.0]), "label", "extra")

        with self.assertWarns(DeprecationWarning):
            data, fmt, axis, kwargs = get_data(t)

        self.assertIsNone(data)

    def test_get_data_converts_list_to_numpy(self):
        """Test get_data converts list to numpy array."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, get_data

        data = [1.0, 2.0, 3.0, 4.0]
        entry = Entry(data=data, label="List Data")

        parsed_data, _, _, _ = get_data(entry)

        self.assertIsInstance(parsed_data, np.ndarray)

    def test_get_data_converts_tensor_to_numpy(self):
        """Test get_data converts torch tensor to numpy array."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, get_data

        data = torch.tensor([1.0, 2.0, 3.0])
        entry = Entry(data=data, label="Tensor Data")

        parsed_data, _, _, _ = get_data(entry)

        self.assertIsInstance(parsed_data, np.ndarray)

    def test_get_data_converts_series_to_numpy(self):
        """Test get_data converts pandas Series to numpy array."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, get_data

        data = pd.Series([1.0, 2.0, 3.0])
        entry = Entry(data=data, label="Series Data")

        parsed_data, _, _, _ = get_data(entry)

        self.assertIsInstance(parsed_data, np.ndarray)

    def test_get_data_reshapes_1d_array(self):
        """Test get_data reshapes 1D array to 2D."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, get_data

        data = np.array([1.0, 2.0, 3.0])
        entry = Entry(data=data, label="1D Data")

        parsed_data, _, _, _ = get_data(entry)

        self.assertEqual(parsed_data.ndim, 2)
        self.assertEqual(parsed_data.shape, (1, 3))


class TestPlotFunction(unittest.TestCase):
    """Tests for the main plot function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample time and data
        self.time = pd.date_range(start="2023-01-01", periods=24, freq="h")
        self.data1 = np.sin(np.linspace(0, 2 * np.pi, 24))
        self.data2 = np.cos(np.linspace(0, 2 * np.pi, 24))
        self.data3 = np.linspace(0, 100, 24)

    def tearDown(self):
        """Clean up after tests."""
        plt.close("all")

    def test_plot_single_entry(self):
        """Test plot with a single Entry on axis 1."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        entry = Entry(data=self.data1, label="Sine Wave")
        fig, axes = plot(time=self.time, entries=[entry], show=False)

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), 1)

        plt.close(fig)

    def test_plot_multiple_entries_axis1(self):
        """Test plot with multiple entries on axis 1."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        entry1 = Entry(data=self.data1, label="Sine Wave")
        entry2 = Entry(data=self.data2, label="Cosine Wave")

        fig, axes = plot(
            time=self.time,
            entries=[entry1, entry2],
            ylabel_1axis="Amplitude",
            show=False,
        )

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), 1)

        plt.close(fig)

    def test_plot_with_two_axes(self):
        """Test plot with entries on two y-axes."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        entry1 = Entry(data=self.data1, label="Sine Wave", axis=1)
        entry2 = Entry(data=self.data3, label="Linear Data", axis=2)

        fig, axes = plot(
            time=self.time,
            entries=[entry1, entry2],
            ylabel_1axis="Amplitude",
            ylabel_2axis="Value",
            show=False,
        )

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), 2)

        plt.close(fig)

    def test_plot_with_three_axes(self):
        """Test plot with entries on three y-axes."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        entry1 = Entry(data=self.data1, label="Sine Wave", axis=1)
        entry2 = Entry(data=self.data2, label="Cosine Wave", axis=2)
        entry3 = Entry(data=self.data3, label="Linear Data", axis=3)

        fig, axes = plot(
            time=self.time,
            entries=[entry1, entry2, entry3],
            ylabel_1axis="Amplitude 1",
            ylabel_2axis="Amplitude 2",
            ylabel_3axis="Value",
            show=False,
        )

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), 3)

        plt.close(fig)

    def test_plot_with_ylim(self):
        """Test plot with y-axis limits."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        entry = Entry(data=self.data1, label="Sine Wave")

        fig, axes = plot(
            time=self.time,
            entries=[entry],
            ylim_1axis=(-2, 2),
            show=False,
        )

        self.assertIsInstance(fig, Figure)
        ylim = axes[0].get_ylim()
        self.assertLessEqual(ylim[0], -2)
        self.assertGreaterEqual(ylim[1], 2)

        plt.close(fig)

    def test_plot_with_title(self):
        """Test plot with title."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        entry = Entry(data=self.data1, label="Sine Wave")

        fig, axes = plot(
            time=self.time,
            entries=[entry],
            title="Test Plot Title",
            show=False,
        )

        self.assertIsInstance(fig, Figure)
        # Check suptitle exists
        self.assertEqual(fig._suptitle.get_text(), "Test Plot Title")

        plt.close(fig)

    def test_plot_with_custom_styling(self):
        """Test plot with custom Entry styling."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        entry = Entry(
            data=self.data1,
            label="Styled Data",
            color="red",
            fmt="--",
            linewidth=3,
        )

        fig, axes = plot(time=self.time, entries=[entry], show=False)

        self.assertIsInstance(fig, Figure)

        plt.close(fig)

    def test_plot_with_numpy_time(self):
        """Test plot with numpy datetime array."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        time = np.array(self.time)
        entry = Entry(data=self.data1, label="Test Data")

        fig, axes = plot(time=time, entries=[entry], show=False)

        self.assertIsInstance(fig, Figure)

        plt.close(fig)

    def test_plot_with_list_time(self):
        """Test plot with list of times (batch mode)."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        time = [self.time]
        data = self.data1.reshape(1, -1)
        entry = Entry(data=data, label="Batch Data")

        fig, axes = plot(time=time, entries=[entry], show=False)

        self.assertIsInstance(fig, Figure)

        plt.close(fig)

    def test_plot_with_single_entry_not_list(self):
        """Test plot with single Entry (not in list)."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        entry = Entry(data=self.data1, label="Single Entry")

        fig, axes = plot(time=self.time, entries=entry, show=False)

        self.assertIsInstance(fig, Figure)

        plt.close(fig)

    def test_plot_missing_time_raises_error(self):
        """Test plot raises error when time is missing."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        entry = Entry(data=self.data1, label="Test Data")

        with self.assertRaises(AssertionError):
            plot(time=None, entries=[entry], show=False)

    def test_plot_missing_entries_raises_error(self):
        """Test plot raises error when entries is missing."""
        # Local application imports
        from twin4build.utils.plot.plot import plot

        with self.assertRaises(AssertionError):
            plot(time=self.time, entries=None, show=False)

    def test_plot_no_axis1_entries_raises_error(self):
        """Test plot raises error when no entries for axis 1."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        entry = Entry(data=self.data1, label="Axis 2 Only", axis=2)

        with self.assertRaises(ValueError) as context:
            plot(time=self.time, entries=[entry], show=False)

        self.assertIn("axis=1", str(context.exception))

    def test_plot_with_roundto_parameters(self):
        """Test plot with roundto parameters."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        entry1 = Entry(data=self.data1, label="Data 1", axis=1)
        entry2 = Entry(data=self.data3, label="Data 2", axis=2)

        fig, axes = plot(
            time=self.time,
            entries=[entry1, entry2],
            ylabel_1axis="Amp",
            ylabel_2axis="Val",
            roundto_1axis=0.1,
            roundto_2axis=10,
            show=False,
        )

        self.assertIsInstance(fig, Figure)

        plt.close(fig)

    def test_plot_with_yoffset_parameters(self):
        """Test plot with yoffset parameters."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        entry1 = Entry(data=self.data1, label="Data 1", axis=1)
        entry2 = Entry(data=self.data3, label="Data 2", axis=2)

        fig, axes = plot(
            time=self.time,
            entries=[entry1, entry2],
            ylabel_1axis="Amp",
            ylabel_2axis="Val",
            yoffset_1axis=0.1,
            yoffset_2axis=5,
            show=False,
        )

        self.assertIsInstance(fig, Figure)

        plt.close(fig)

    def test_plot_with_align_zero_false(self):
        """Test plot with align_zero=False."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        entry1 = Entry(data=self.data1, label="Data 1", axis=1)
        entry2 = Entry(data=self.data3, label="Data 2", axis=2)

        fig, axes = plot(
            time=self.time,
            entries=[entry1, entry2],
            ylabel_1axis="Amp",
            ylabel_2axis="Val",
            yoffset_2axis=5,
            align_zero=False,
            show=False,
        )

        self.assertIsInstance(fig, Figure)

        plt.close(fig)

    def test_plot_with_ylim_on_multiple_axes(self):
        """Test plot with y-limits on multiple axes."""
        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot

        entry1 = Entry(data=self.data1, label="Data 1", axis=1)
        entry2 = Entry(data=self.data2, label="Data 2", axis=2)
        entry3 = Entry(data=self.data3, label="Data 3", axis=3)

        fig, axes = plot(
            time=self.time,
            entries=[entry1, entry2, entry3],
            ylabel_1axis="Amp 1",
            ylabel_2axis="Amp 2",
            ylabel_3axis="Value",
            ylim_1axis=(-1.5, 1.5),
            ylim_2axis=(-1.5, 1.5),
            ylim_3axis=(0, 120),
            show=False,
        )

        self.assertIsInstance(fig, Figure)

        plt.close(fig)


class TestGetFigAxes(unittest.TestCase):
    """Tests for get_fig_axes function."""

    def tearDown(self):
        """Clean up after tests."""
        plt.close("all")

    def test_get_fig_axes_single_plot(self):
        """Test get_fig_axes with single plot."""
        # Local application imports
        from twin4build.utils.plot.plot import get_fig_axes

        fig, axes = get_fig_axes("Single Plot", n_plots=1)

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), 1)
        self.assertIsInstance(axes[0], Axes)

        plt.close(fig)

    def test_get_fig_axes_multiple_plots(self):
        """Test get_fig_axes with multiple plots."""
        # Local application imports
        from twin4build.utils.plot.plot import get_fig_axes

        fig, axes = get_fig_axes("Multiple Plots", n_plots=4, cols=2)

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), 4)

        plt.close(fig)

    def test_get_fig_axes_with_custom_size(self):
        """Test get_fig_axes with custom size."""
        # Local application imports
        from twin4build.utils.plot.plot import get_fig_axes

        fig, axes = get_fig_axes("Custom Size", n_plots=1, size_inches=(10, 8))

        self.assertIsInstance(fig, Figure)
        size = fig.get_size_inches()
        self.assertEqual(tuple(size), (10, 8))

        plt.close(fig)

    def test_get_fig_axes_with_custom_offset(self):
        """Test get_fig_axes with custom offset."""
        # Local application imports
        from twin4build.utils.plot.plot import get_fig_axes

        fig, axes = get_fig_axes("Custom Offset", n_plots=1, offset=(0.15, 0.2))

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), 1)

        plt.close(fig)

    def test_get_fig_axes_with_multiple_rows(self):
        """Test get_fig_axes with multiple rows."""
        # Local application imports
        from twin4build.utils.plot.plot import get_fig_axes

        fig, axes = get_fig_axes("Multiple Rows", n_plots=6, cols=2)

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), 6)

        plt.close(fig)

    def test_get_fig_axes_fewer_plots_than_grid(self):
        """Test get_fig_axes with fewer plots than grid cells."""
        # Local application imports
        from twin4build.utils.plot.plot import get_fig_axes

        # 3 plots in a 2x2 grid
        fig, axes = get_fig_axes("Partial Grid", n_plots=3, cols=2)

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), 3)

        plt.close(fig)

    def test_get_fig_axes_title_set(self):
        """Test that get_fig_axes sets the title correctly."""
        # Local application imports
        from twin4build.utils.plot.plot import get_fig_axes

        fig, axes = get_fig_axes("My Test Title", n_plots=1)

        self.assertEqual(fig._suptitle.get_text(), "My Test Title")

        plt.close(fig)


class TestPlotComponentDeprecated(unittest.TestCase):
    """Tests for deprecated plot_component function."""

    def setUp(self):
        """Set up test fixtures."""
        self.time = pd.date_range(start="2023-01-01", periods=24, freq="h")
        self.data1 = np.sin(np.linspace(0, 2 * np.pi, 24))
        self.data2 = np.cos(np.linspace(0, 2 * np.pi, 24))

    def tearDown(self):
        """Clean up after tests."""
        plt.close("all")

    def test_plot_component_with_entry_objects(self):
        """Test plot_component with Entry objects."""
        # Standard library imports
        from unittest.mock import Mock

        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot_component

        # Create mock simulator
        mock_simulator = Mock()
        mock_simulator.date_time_steps = self.time

        entry1 = Entry(data=self.data1, label="Test Data")

        with self.assertWarns(DeprecationWarning):
            fig, axes = plot_component(
                simulator=mock_simulator,
                components_1axis=[entry1],
                show=False,
            )

        self.assertIsInstance(fig, Figure)

        plt.close(fig)

    def test_plot_component_with_direct_data_tuples(self):
        """Test plot_component with direct data tuples (deprecated format)."""
        # Standard library imports
        from unittest.mock import Mock

        # Local application imports
        from twin4build.utils.plot.plot import plot_component

        mock_simulator = Mock()
        mock_simulator.date_time_steps = self.time

        # Direct data tuple format: (data, label)
        with self.assertWarns(DeprecationWarning):
            fig, axes = plot_component(
                simulator=mock_simulator,
                components_1axis=[(self.data1, "Direct Data")],
                show=False,
            )

        self.assertIsInstance(fig, Figure)

        plt.close(fig)

    def test_plot_component_multiple_axes_with_entries(self):
        """Test plot_component with entries on multiple axes."""
        # Standard library imports
        from unittest.mock import Mock

        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot_component

        mock_simulator = Mock()
        mock_simulator.date_time_steps = self.time

        entry1 = Entry(data=self.data1, label="Axis 1")
        entry2 = Entry(data=self.data2, label="Axis 2")

        with self.assertWarns(DeprecationWarning):
            fig, axes = plot_component(
                simulator=mock_simulator,
                components_1axis=[entry1],
                components_2axis=[entry2],
                ylabel_1axis="Amp 1",
                ylabel_2axis="Amp 2",
                show=False,
            )

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), 2)

        plt.close(fig)

    def test_plot_component_three_axes_with_entries(self):
        """Test plot_component with entries on three axes."""
        # Standard library imports
        from unittest.mock import Mock

        # Local application imports
        from twin4build.utils.plot.plot import Entry, plot_component

        mock_simulator = Mock()
        mock_simulator.date_time_steps = self.time

        data3 = np.linspace(0, 100, 24)

        entry1 = Entry(data=self.data1, label="Axis 1")
        entry2 = Entry(data=self.data2, label="Axis 2")
        entry3 = Entry(data=data3, label="Axis 3")

        with self.assertWarns(DeprecationWarning):
            fig, axes = plot_component(
                simulator=mock_simulator,
                components_1axis=[entry1],
                components_2axis=[entry2],
                components_3axis=[entry3],
                ylabel_1axis="Amp 1",
                ylabel_2axis="Amp 2",
                ylabel_3axis="Value",
                show=False,
            )

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), 3)

        plt.close(fig)

    def test_plot_component_direct_tuples_multiple_axes(self):
        """Test plot_component with direct data tuples on multiple axes."""
        # Standard library imports
        from unittest.mock import Mock

        # Local application imports
        from twin4build.utils.plot.plot import plot_component

        mock_simulator = Mock()
        mock_simulator.date_time_steps = self.time

        with self.assertWarns(DeprecationWarning):
            fig, axes = plot_component(
                simulator=mock_simulator,
                components_1axis=[(self.data1, "Axis 1")],
                components_2axis=[(self.data2, "Axis 2")],
                ylabel_1axis="Amp 1",
                ylabel_2axis="Amp 2",
                show=False,
            )

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), 2)

        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
