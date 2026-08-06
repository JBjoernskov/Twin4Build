# Standard library imports
import atexit
import functools
import inspect
import os
import sys
import threading
import time
import warnings

# Third party imports
import __main__
import numpy as np
from dateutil import tz

# Curses TUI removed in twin4build 2.0 in favor of dual ANSI stdout + plain logfile.
CURSES_AVAILABLE = False


def _print_color_palette(stdscr):
    # Clear the screen first to ensure we start fresh
    stdscr.clear()

    curses.start_color()
    curses.use_default_colors()

    # Determine the maximum number of color pairs we can safely use
    # COLOR_PAIRS includes pair 0, so we can use pairs 1 to COLOR_PAIRS-1
    max_pairs = min(curses.COLOR_PAIRS - 1, curses.COLORS)

    # Initialize color pairs with default background (-1)
    for i in range(0, max_pairs):
        curses.init_pair(i + 1, i, -1)

    # Add a header
    stdscr.addstr(0, 0, f"Available colors (showing {max_pairs} colors):\n\n")

    # Display colors with their numbers
    row = 2
    col = 0
    for i in range(0, max_pairs + 1):
        color_text = f"{i:3d} "

        # Move to next row if we reach the right edge
        if col + len(color_text) >= curses.COLS:
            row += 1
            col = 0
            if row >= curses.LINES - 1:  # Leave room for instructions
                break

        stdscr.addstr(row, col, color_text, curses.color_pair(i))
        col += len(color_text)

    # Add instructions at the bottom
    stdscr.addstr(curses.LINES - 1, 0, "Press any key to exit...")

    stdscr.refresh()
    stdscr.getch()


def print_color_palette():
    warnings.warn(
        "print_color_palette was removed with the curses logger TUI in twin4build 2.0.",
        DeprecationWarning,
        stacklevel=2,
    )


class Logger:
    """Custom logging system for Twin4Build with tree-structured output.

    LOGGER Usage Standard
    =====================

    Badge system
    ------------
    Work & control flow:
        - ``LOGGER.task(msg)``    -> ``[TASK]``    -- Unit of work (can self-nest).

    Data & structure:
        - ``LOGGER.section(msg)`` -> ``[SECTION]`` -- Data-organizing container (can self-nest).
        - ``LOGGER.config(msg)``  -> ``[CONFIG]``  -- Setup/input value (leaf).
        - ``LOGGER.result(msg)``  -> ``[RESULT]``  -- Reported result/output data (can self-nest).
        - ``LOGGER.iter(msg)``    -> ``[ITER]``    -- Iteration metrics (leaf).

    Informational:
        - ``LOGGER.info(msg)``    -> ``[INFO]``    -- General information, discoveries.
        - ``LOGGER.debug(msg)``   -> ``[DEBUG]``   -- Verbose detail, filtered by default.

    Feedback (dual mode):
        - ``LOGGER.ok(msg)``      -> ``[OUTCOME]`` green  -- or recolor with ``change_status=True``.
        - ``LOGGER.warning(msg)`` -> ``[OUTCOME]`` yellow -- or recolor with ``change_status=True``.
        - ``LOGGER.error(msg)``   -> ``[OUTCOME]`` red    -- or recolor with ``change_status=True``.

    Nesting rules
    -------------
    ``[TASK]`` children -- any badge is allowed.

    ``[SECTION]`` and ``[CONFIG]`` children -- only data badges:
    ``[SECTION]``, ``[INFO]``, ``[ITER]``, ``[RESULT]``. Never ``[TASK]``,
    ``[OUTCOME]``, or ``[DEBUG]``.

    Outcome dual mode
    -----------------
    Without ``change_status=True``: prints a **new** ``[OUTCOME]`` line colored
    green / yellow / red.
    With ``change_status=True``: **recolors** an existing line's badge
    (text stays unchanged; only the color changes).

    Message formatting
    ------------------
    Always use ``%``-style formatting for dynamic messages::

        LOGGER.info("Found %d items", n)

    Never use f-strings -- ``%``-style enables lazy evaluation (skipping
    interpolation when the message is filtered out).

    Message styling
    ---------------
    - Sentence case (capitalize first word only).
    - No terminal punctuation, except warning/error outcomes (full sentences
      ending with a period).
    - ``Label: value`` for config/result.
    - ``key=value | key=value`` for iteration metrics.
    - ``component_id.attribute`` for identifiers.

    Example
    -------
    ::

        LOGGER.task("Starting estimation")
        LOGGER.add_level()
        LOGGER.config("Method: %s", method)
        LOGGER.task("Initializing model")
        model.initialize()
        LOGGER.ok("Initializing model", change_status=True)
        LOGGER.iter("eval=%d | obj=%.6f | elapsed=%.1fs", n, obj, elapsed)
        LOGGER.ok("Converged: iterations=%d", nit)
        LOGGER.remove_level()
        LOGGER.ok("Starting estimation", change_status=True)
    """

    def __init__(self) -> None:
        self.level_indent = []  # level as function of line number
        self.level = []
        self.indent = []
        self.message = []
        self.status = []
        self.location = []
        self.added_level = False
        self._pending_levels = 0
        self._phantom_levels = 0
        self.removed_level = False
        self.level_stack = [0]
        self.has_printed = False
        self._verbose = 3
        self._current_level_indent = 0
        self._block_count = 0
        self.logfile = None
        self._last_file_content = ""  # Cache for atomic file updates
        self._is_active = False
        self._log_flush_size = 50  # Flush to file every N lines
        self._log_buffer = []  # Pending formatted lines not yet written
        self._flushed_line_count = 0  # Number of _curses_lines already written to disk
        # File mode is buffered; ensure we flush on exit/crash
        self._file_flush_registered = False
        self._file_flush_logfile_path = None
        self._file_excepthook_installed = False
        self.call_depth = 0
        self._scroll_offset = 0
        self._lock = threading.Lock()
        self._stop_thread = threading.Event()
        self._display_thread = None
        self._scroll_step = 4  # Number of lines to scroll per tick
        self._paused = False  # Pause state for curses display
        self._pause_event = threading.Event()  # Event for pausing execution
        self._pause_event.set()  # Initially not paused (set = can proceed)
        # Warning capture
        self._warning_handler_installed = False
        self._original_showwarning = None
        # Curses-related attributes
        # Disable curses in CI environments (GitHub Actions, Jenkins, etc.)
        is_ci = (
            os.getenv("CI")
            or os.getenv("GITHUB_ACTIONS")
            or os.getenv("JENKINS_HOME")
            or os.getenv("TRAVIS")
        )
        self._use_curses = False  # curses TUI removed in 2.0
        self._use_threading = True  # Whether to use background thread for scrolling
        self._curses_mode = False
        self._stdscr = None
        self._curses_lines = []  # Store lines for curses display
        self._persist_on_exit = (
            True  # Whether to show final output after curses cleanup
        )
        self._atexit_registered = False  # Track if we've registered cleanup
        self.VERT = "|"
        self.HOR = "_" * 3
        self.SPACE = " " * 3

        # Color pair indices (used by curses)
        self.COLOR_PAIR_LEVEL_CYCLE = [8, 4]  # Alternating colors for tree levels
        self.OK_COLOR_PAIR = 3  # Green
        self.ERROR_COLOR_PAIR = 5  # Red
        self.WARNING_COLOR_PAIR = 7  # Yellow
        self.INFO_COLOR_PAIR = 2  # Blue
        self.LOCATION_COLOR_PAIR = 6  # Cyan/Magenta for file:line locations

        # Single source of truth: map color pair index to ANSI code
        # This is used for both curses-to-ANSI conversion and direct ANSI output
        self.COLOR_PAIR_TO_ANSI = {
            2: "34",  # Blue
            3: "32",  # Green
            4: "36",  # Cyan
            5: "31",  # Red
            6: "35",  # Magenta
            7: "33",  # Yellow
            8: "37",  # White
        }

        self._enabled = True
        # Allow explicit opt-in to progress output while tests run
        self._allow_in_tests = False
        self._show_location = True

        # Status filtering
        self._status_filters = {
            "debug": False,
            "warning": True,
            "error": True,
            "ok": True,
            "info": True,
            "task": True,
            "section": True,
            "config": True,
            "result": True,
            "iter": True,
            "outcome": True,
            "default": True,
        }  # True = show, False = hide

        # Caller filtering
        self._caller_filters = set()  # Set of caller function names
        self._caller_filter_mode = "whitelist"  # "whitelist" or "blacklist"
        self._caller_filter_include_stack = True  # Whether to check entire call stack

        # Once-only warning deduplication
        self._warned_once_messages: set = set()

    def _install_file_flush_handlers(self, logfile_path: str):
        """Install flush handlers so buffered file logs aren't lost on crashes."""
        if not logfile_path:
            return

        # Flush at normal interpreter exit
        if (not self._file_flush_registered) or (self._file_flush_logfile_path != logfile_path):
            atexit.register(self._flush_log_buffer, logfile_path)
            self._file_flush_registered = True
            self._file_flush_logfile_path = logfile_path

        # Flush on unhandled exceptions (file mode doesn't initialize curses)
        if self._file_excepthook_installed:
            return
        original_excepthook = sys.excepthook

        def _file_exception_handler(exc_type, exc_value, exc_traceback):
            try:
                self._flush_log_buffer(self._file_flush_logfile_path)
            finally:
                original_excepthook(exc_type, exc_value, exc_traceback)

        sys.excepthook = _file_exception_handler
        self._file_excepthook_installed = True

    def __enter__(self):
        """Context manager entry - ensures proper cleanup on exceptions"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - cleanup curses if active"""
        if self._curses_mode:
            self._cleanup_curses(preserve_output=True)
        return False  # Don't suppress exceptions

    @property
    def allow_in_tests(self):
        return self._allow_in_tests

    @property
    def enabled(self):
        # Optimized: avoid repeated imports by caching module reference
        if not hasattr(self, "_twin4build_module"):
            try:
                # Local application imports
                import twin4build

                self._twin4build_module = twin4build
            except ImportError:
                self._twin4build_module = None
                return False

        if self._twin4build_module is None:
            return False

        if not self._twin4build_module._IMPORT_COMPLETE:
            return False
        if self._twin4build_module._IS_TESTING and not self.allow_in_tests:
            return False
        return self._enabled

    def enable(self):
        self._enabled = True
        # When explicitly enabled, permit use even under _IS_TESTING

    def disable(self):
        self._enabled = False

    def enable_threading(self):
        """Enable background threading for scrolling (curses mode only).

        When enabled, a background thread handles input and display updates,
        allowing smooth scrolling and interactive features in curses mode.
        This is the default behavior.

        Example:
            LOGGER.enable_threading()  # Enable smooth scrolling
        """
        self._use_threading = True

    def disable_threading(self):
        """Disable background threading - will still use curses but without scrolling.

        When disabled, the display updates synchronously without a background thread.
        This can be useful for debugging, reducing resource usage, or avoiding
        threading-related issues. Scrolling and pause features will still work
        but will only update when new messages are logged.

        Example:
            LOGGER.disable_threading()  # Disable background thread
        """
        self._use_threading = False
        # If already in curses mode with threading, stop the thread
        if self._display_thread is not None and self._display_thread.is_alive():
            self._stop_thread.set()
            self._display_thread.join(timeout=1.0)
            self._display_thread = None

    def show_status(self, status_type):
        """Show messages of the given status type.

        Args:
            status_type: One of "debug", "info", "warning", "error", "ok", "success", "default"
        """
        self._status_filters[status_type.lower()] = True

    def hide_status(self, status_type):
        """Hide messages of the given status type.

        Args:
            status_type: One of "debug", "info", "warning", "error", "ok", "success", "default"
        """
        self._status_filters[status_type.lower()] = False

    def show_all_status(self):
        """Show all status types."""
        for status in self._status_filters:
            self._status_filters[status] = True

    def hide_all_status(self):
        """Hide all status types."""
        for status in self._status_filters:
            self._status_filters[status] = False

    def show_all_callers(self):
        """Remove caller filter — show messages from all callers (default)."""
        self._caller_filters.clear()
        self._caller_filter_mode = "whitelist"

    def show_caller(self, caller_name, include_stack=True, hide_other_callers=False):
        """Show messages from call stacks containing caller_name (whitelist mode).

        Args:
            caller_name: Name of the function whose messages to show
            include_stack: If True, also show messages from functions called by it
            hide_other_callers: If True (default), clear all previous caller filters so
                                only this caller is shown. Set to False to add to an
                                existing whitelist without clearing it.
        """
        self._caller_filter_mode = "whitelist"
        if hide_other_callers:
            self._caller_filters.clear()
        self._caller_filter_include_stack = include_stack
        self._caller_filters.add(caller_name)

    def hide_caller(self, caller_name, include_stack=True):
        """Hide messages from call stacks containing caller_name (blacklist mode).

        Args:
            caller_name: Name of the function whose messages to hide
            include_stack: If True, also hide messages from functions called by it
        """
        self._caller_filter_mode = "blacklist"
        self._caller_filters.clear()
        self._caller_filter_include_stack = include_stack
        self._caller_filters.add(caller_name)

    def _get_caller_function_name(self):
        """Get the name of the function that called LOGGER"""
        frame = inspect.currentframe()
        # Walk back: current -> _get_caller_function_name -> __call__ -> caller...
        if frame is not None:
            frame = frame.f_back  # __call__ or internal
        while frame is not None:
            func_name = frame.f_code.co_name
            # Skip internal methods, the __call__ method, and convenience wrappers
            if func_name not in [
                "__call__",
                "_get_caller_function_name",
                "_should_filter_message",
                "wait_if_paused",
                "wrapper",
                "debug",
                "info",
                "warning",
                "error",
                "ok",
                "success",
            ]:
                return func_name
            frame = frame.f_back
        return None

    def _is_caller_in_stack(self, caller_set):
        """Check if any function in the current call stack is in the caller_set"""
        frame = inspect.currentframe()
        # Walk back: current -> _is_caller_in_stack -> _should_filter_message -> __call__ -> caller...
        if frame is not None:
            frame = frame.f_back  # _should_filter_message
        if frame is not None:
            frame = frame.f_back  # __call__
        while frame is not None:
            func_name = frame.f_code.co_name
            # Skip internal methods
            if func_name not in [
                "__call__",
                "_get_caller_function_name",
                "_should_filter_message",
                "_is_caller_in_stack",
                "wait_if_paused",
                "wrapper",
            ]:
                if func_name in caller_set:
                    return True
            frame = frame.f_back

        return False

    def _should_filter_message(self, caller_name=None):
        """Check if a message should be filtered out based on caller filters.

        Note: Status filtering is handled by the public methods (debug, info, etc.)
        before calling __call__, so we don't need to parse status strings here.
        """
        # Check caller filter only
        if self._caller_filters:
            if self._caller_filter_mode == "whitelist":
                if self._caller_filter_include_stack:
                    # Whitelist mode: only show if any caller in stack is in the set
                    if not self._is_caller_in_stack(self._caller_filters):
                        return True
                else:
                    # Whitelist mode: only show if immediate caller is in the set
                    if caller_name not in self._caller_filters:
                        return True
            elif self._caller_filter_mode == "blacklist":
                if self._caller_filter_include_stack:
                    # Blacklist mode: hide if any caller in stack is in the set
                    if self._is_caller_in_stack(self._caller_filters):
                        return True
                else:
                    # Blacklist mode: hide if immediate caller is in the set
                    if caller_name in self._caller_filters:
                        return True

        return False

    def wait_if_paused(self, timeout=None):
        """
        Wait if execution is paused. Call this in your main execution loop
        to respect pause state from curses display.

        Args:
            timeout: Maximum time to wait in seconds. None means wait indefinitely.

        Returns:
            True if resumed normally, False if timeout occurred
        """
        if self._curses_mode:
            return self._pause_event.wait(timeout=timeout)
        return True  # If not in curses mode, never pause

    @property
    def is_paused(self):
        """Check if execution is currently paused"""
        return self._paused

    @property
    def _caller_whitelist_active(self):
        """True when a caller whitelist is set — status filters should not block early."""
        return bool(self._caller_filters) and self._caller_filter_mode == "whitelist"

    @property
    def threading_enabled(self):
        """Check if background threading is enabled"""
        return self._use_threading

    @property
    def is_active(self):
        return self._is_active

    @property
    def verbose(self):
        # return int(self._verbose)
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        assert isinstance(value, (int)), "verbose must be an integer or None"
        self._verbose = value

    @property
    def current_level(self):
        return len(self.level_stack) - 1

    @property
    def log_flush_size(self):
        return self._log_flush_size

    @log_flush_size.setter
    def log_flush_size(self, value):
        self._log_flush_size = value

    def _get_logfile_path(self):
        """Get the plain (non-ANSI) logfile path.

        Always returns a path so LLM-friendly plain logs are written. Human-facing
        ANSI output still goes to stdout in parallel.
        """
        if self.logfile is not None:
            return self.logfile
        return "progress.log"

    def get_log(self):
        logfile = self._get_logfile_path()
        if logfile is not None:
            f = open(logfile, "w")
        else:
            f = None
        return f

    def _get_indices(self, line):
        return [i for i, ltr in enumerate(line) if ltr == self.VERT]

    def get_char_level(self, line):
        indices = self._get_indices(line)
        char_level = np.zeros(len(line), dtype=int)
        if len(indices) == 0:
            return char_level
        for l, (prev, next) in enumerate(zip(indices[:-1], indices[1:])):
            char_level[prev:next] = l + 1
        char_level[indices[-1] :] = len(indices)
        return char_level

    def _get_caller_location(self):
        """Return 'filename:lineno' for the caller of LOGGER"""
        current_file = os.path.abspath(__file__)
        frame = inspect.currentframe()
        # Walk back: current -> _get_caller_location -> __call__ -> caller...
        if frame is not None:
            frame = frame.f_back  # __call__ or internal
        while frame is not None:
            fname = os.path.abspath(frame.f_code.co_filename)
            if fname != current_file:
                return f"{os.path.basename(fname)}:{frame.f_lineno}"
            frame = frame.f_back
        return None

    # ANSI escape codes
    ANSI_RESET = "\033[0m"

    def _get_ansi_color(self, color_pair_idx):
        """Get ANSI escape sequence for a color pair index"""
        ansi_code = self.COLOR_PAIR_TO_ANSI.get(color_pair_idx)
        if ansi_code:
            return f"\033[{ansi_code}m"
        return ""

    def _get_status_color_pair(self, status):
        """Get the color pair index for a status string.

        This is the single source of truth for status -> color mapping.
        Color suffixes (e.g., [TASK:green], [TASK:red], [OUTCOME:yellow])
        take priority. Badges without a color suffix use default colors.
        """
        status_lower = status.lower()
        if ":green]" in status_lower or "[ok]" in status_lower or "[success]" in status_lower:
            return self.OK_COLOR_PAIR
        elif ":red]" in status_lower or "[error]" in status_lower or "[failed]" in status_lower:
            return self.ERROR_COLOR_PAIR
        elif ":yellow]" in status_lower or "[warning]" in status_lower or "[warn]" in status_lower:
            return self.WARNING_COLOR_PAIR
        elif "[debug]" in status_lower:
            return self.INFO_COLOR_PAIR
        else:
            return self.INFO_COLOR_PAIR

    def _get_status_ansi_color(self, status):
        """Get ANSI color escape sequence for a status string"""
        return self._get_ansi_color(self._get_status_color_pair(status))

    @staticmethod
    def _get_status_display_text(status):
        """Get the display text for a status, stripping internal color suffixes.

        [TASK:green] -> [TASK], [OUTCOME:yellow] -> [OUTCOME], etc.
        Any status of the form [TAG:color] is rendered as [TAG].
        """
        if ":" in status and status.endswith("]"):
            return status.split(":")[0] + "]"
        return status

    @staticmethod
    def _apply_color_to_status(existing_status, incoming_status):
        """Preserve existing badge text, apply only the color from incoming status.

        Used by change_status=True to recolor a line without replacing its badge.
        e.g. existing=[TASK], incoming=[OUTCOME:green] -> [TASK:green]
             existing=[TASK], incoming=[OUTCOME:red]   -> [TASK:red]
        """
        for color in ("green", "yellow", "red"):
            if color in incoming_status.lower():
                base = existing_status.split(":")[0].rstrip("]")
                return "%s:%s]" % (base, color)
        return incoming_status

    def _format_message(self, message, location, use_ansi_colors=False):
        if location:
            if use_ansi_colors:
                loc_color = self._get_ansi_color(self.LOCATION_COLOR_PAIR)
                return f"{message} {loc_color}({location}){self.ANSI_RESET}"
            else:
                return f"{message} ({location})"
        return message

    def _format_line_ansi(self, indent, message, status, level, location):
        """Format a complete log line with ANSI colors for tree, message, location, and status."""
        char_levels = self.get_char_level(indent)

        colored_output = ""
        prev_pair_idx = None
        for i, char in enumerate(indent):
            pair_idx = self._get_color_pair_idx(char_levels[i])
            if pair_idx != prev_pair_idx:
                if prev_pair_idx is not None:
                    colored_output += self.ANSI_RESET
                colored_output += self._get_ansi_color(pair_idx)
                prev_pair_idx = pair_idx
            colored_output += char
        if prev_pair_idx is not None:
            colored_output += self.ANSI_RESET

        if status:
            status_color = self._get_status_ansi_color(status)
            display_status = self._get_status_display_text(status)
            colored_output += f"{status_color}{display_status}{self.ANSI_RESET} "

        colored_output += message

        if location:
            loc_color = self._get_ansi_color(self.LOCATION_COLOR_PAIR)
            colored_output += f" {loc_color}({location}){self.ANSI_RESET}"

        return colored_output

    def add_line(self, indent="", message="", status="", location=None):
        self.indent.append(indent)
        self.message.append(message)
        self.status.append(status)
        self.location.append(location)
        self.level_indent.append(self._current_level_indent)
        self.level.append(self.current_level)
        self._is_active = True

    def _compute_bar_visibility(self):
        """Bottom-up pass: keep a vertical bar only when it connects to a future message.

        A bar at ancestor level A on line i is shown iff there is a later
        line j at level A+1 (a sibling) with no intervening line at level <= A
        (which would break the scope).  This is the ``├`` vs ``└`` distinction
        in standard tree renderers.
        """
        n = len(self.indent)
        if n == 0:
            return []

        max_level = max(self.level) if self.level else 0
        has_sibling = [False] * (max_level + 2)

        rendered = [None] * n
        for i in range(n - 1, -1, -1):
            indent = self.indent[i]
            level = self.level[i]
            is_separator = self.message[i] == ""

            if indent and level > 0:
                bar_positions = [j for j, c in enumerate(indent) if c == self.VERT]
                chars = list(indent)
                last = len(bar_positions) - 1
                for A, pos in enumerate(bar_positions):
                    if A == last and not is_separator:
                        break  # connector bar is always shown on message lines
                    if not has_sibling[A]:
                        chars[pos] = " "
                rendered[i] = "".join(chars)
            else:
                rendered[i] = indent

            if not is_separator:
                for A in range(level, len(has_sibling)):
                    has_sibling[A] = False
                if level > 0:
                    has_sibling[level - 1] = True

        return rendered

    def _update_lines(self):
        """Update the canonical _curses_lines data structure from current state.

        This is the single source of truth for all output modes
        (curses display, file, stdout).
        """
        with self._lock:
            if self._paused:
                return
            current_len = len(self._curses_lines)
            rendered_indents = self._compute_bar_visibility()
            temp_lines = []
            for indent, message, status, level, location in zip(
                rendered_indents, self.message, self.status, self.level, self.location
            ):
                temp_lines.append((indent, message, status, level, location))
            self._curses_lines = temp_lines
            new_len = len(self._curses_lines)
            diff = new_len - current_len
            if self._scroll_offset > 0 and diff > 0:
                self._scroll_offset += diff

    def _flush_log_buffer(self, logfile_path=None):
        """Append buffered log lines to the log file and clear the buffer."""
        if not self._log_buffer:
            return
        if logfile_path is None:
            logfile_path = self._get_logfile_path()
        if logfile_path is None:
            return
        content = "\n".join(self._log_buffer) + "\n"
        try:
            with open(logfile_path, "a", encoding="utf-8") as f:
                f.write(content)
        except OSError:
            pass
        self._flushed_line_count += len(self._log_buffer)
        self._log_buffer.clear()

    def print_lines(self):
        logfile_path = self._get_logfile_path()
        is_file_mode = logfile_path is not None

        # Initialize curses if needed and we're not logging to file
        if self._use_curses and not is_file_mode and not self._curses_mode:
            self._init_curses()

        # Always update canonical line data
        self._update_lines()

        if self._curses_mode and not is_file_mode:
            # Curses display - thread handles rendering, or update directly
            if not self._use_threading:
                with self._lock:
                    self._handle_input()
                    self._update_curses_display()
        elif is_file_mode:
            self._install_file_flush_handlers(logfile_path)
            # Append-only buffered file output.
            # Rebuild the buffer for all lines not yet flushed — this captures
            # status updates (e.g. ...[OK]) for lines still in the buffer.
            self._log_buffer = []
            for indent, message, status, level, location in self._curses_lines[
                self._flushed_line_count :
            ]:
                display_status = self._get_status_display_text(status)
                _status = display_status + " " if status != "" else ""
                display_message = self._format_message(
                    message, location, use_ansi_colors=False
                )
                self._log_buffer.append(indent + _status + display_message)

            if len(self._log_buffer) >= self._log_flush_size:
                self._flush_log_buffer(logfile_path)
        else:
            # Stdout output
            if self.has_printed:
                self.clear_lines(self.n_printed)

            use_ansi = sys.stdout.isatty()

            self.n_printed = 0
            for indent, message, status, level, location in self._curses_lines:
                if use_ansi:
                    s = self._format_line_ansi(indent, message, status, level, location)
                else:
                    display_status = self._get_status_display_text(status)
                    _status = display_status + " " if status != "" else ""
                    display_message = self._format_message(
                        message, location, use_ansi_colors=False
                    )
                    s = indent + _status + display_message
                print(s, flush=True)
                self.n_printed += 1

        self.has_printed = True

    def is_interactive(self):
        return not hasattr(__main__, "__file__")

    def _init_curses(self):
        """Initialize curses mode, optionally using alternate screen buffer"""
        if not self._use_curses or self._curses_mode:
            return False

        # print("DEBUG: Starting curses initialization", file=sys.stderr)

        # Optionally enter alternate screen buffer (like vim does)
        # This preserves the current terminal content and creates a separate "window"
        # When disabled, curses draws over the current terminal content
        # sys.stdout.write("\033[?1049h")  # Enter alternate screen
        # sys.stdout.write("\033[49m")     # Set default background color (transparent)
        # sys.stdout.flush()
        # print("DEBUG: Alternate screen buffer entered", file=sys.stderr)

        # Now start curses
        self._stdscr = curses.initscr()
        # print("DEBUG: curses.initscr() completed", file=sys.stderr)
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)  # Hide cursor
        self._stdscr.keypad(True)  # Enable keypad for scrolling keys
        self._stdscr.nodelay(True)  # Non-blocking input

        # Enable color if available
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
            except Exception:
                pass

            # Set background to default/transparent (-1)
            try:
                curses.assume_default_colors(-1, -1)  # Available in python 3.14
            except Exception:
                pass

            max_pairs = min(curses.COLOR_PAIRS - 1, curses.COLORS)
            # print(f"DEBUG: Initializing {max_pairs} color pairs", file=sys.stderr)
            for i in range(0, max_pairs):
                try:
                    curses.init_pair(i + 1, i, -1)
                except Exception:
                    # Fallback for terminals that might not support -1 background
                    curses.init_pair(i + 1, i, 0)

        # Clear the screen to start fresh (with default background)
        self._stdscr.bkgd(" ", curses.color_pair(0))
        self._stdscr.clear()

        # Start the display thread only if threading is enabled
        if self._use_threading:
            self._stop_thread.clear()
            self._display_thread = threading.Thread(
                target=self._display_loop, daemon=True
            )
            self._display_thread.start()

        # Capture warnings so they are replayed after curses teardown
        self._install_warning_handler()

        # Register cleanup function to run at exit (only once)
        if not self._atexit_registered:
            atexit.register(self._cleanup_curses, preserve_output=True)
            self._atexit_registered = True

        # Install exception handler to ensure curses cleanup on crashes
        self._install_exception_handler()

        # Ensure pause state is properly initialized
        self._paused = False
        self._pause_event.set()

        self._curses_mode = True
        # print("DEBUG: Curses initialization completed successfully", file=sys.stderr)
        return True

    def _display_loop(self):
        """Background thread loop for handling input and updating display"""
        while not self._stop_thread.is_set():
            with self._lock:
                if self._stdscr and self._curses_mode:
                    self._handle_input()
                    # Always update display to show pause status and handle scrolling
                    self._update_curses_display()
            time.sleep(0.05)  # Update at ~20Hz

    def _install_exception_handler(self):
        """Install a global exception handler to cleanup curses on crashes"""
        if hasattr(self, "_exception_handler_installed"):
            return  # Already installed

        original_excepthook = sys.excepthook

        def curses_exception_handler(exc_type, exc_value, exc_traceback):
            # Clean up curses first if we're in curses mode
            if self._curses_mode and self._stdscr is not None:
                # print(f"DEBUG: Exception occurred, cleaning up curses: {exc_type.__name__}", file=sys.stderr)
                # Try normal cleanup first (preserves output)
                self._cleanup_curses(preserve_output=True)
                # print("DEBUG: Curses cleanup completed after exception", file=sys.stderr)

            print()
            original_excepthook(exc_type, exc_value, exc_traceback)

        sys.excepthook = curses_exception_handler
        self._exception_handler_installed = True

    def _install_warning_handler(self):
        """Redirect warnings to LOGGER during curses mode so they persist"""
        if self._warning_handler_installed:
            return

        self._original_showwarning = warnings.showwarning

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            # While in curses, push warnings into the progress log so they are replayed after teardown.
            if self._curses_mode:
                # Use the filename and lineno provided by Python's warning system
                # These are the actual location where warnings.warn() was called
                loc = f"{os.path.basename(filename)}:{lineno}"
                self(str(message), status="[OUTCOME:yellow]", location=loc)

            # Forward to the original handler when not in curses, or when an explicit
            # target file is provided (e.g., logging to file).
            if (not self._curses_mode) or file is not None:
                self._original_showwarning(
                    message, category, filename, lineno, file, line
                )

        warnings.showwarning = _showwarning
        self._warning_handler_installed = True

    def _restore_warning_handler(self):
        """Restore default warnings.showwarning"""
        if self._warning_handler_installed and self._original_showwarning is not None:
            warnings.showwarning = self._original_showwarning
        self._warning_handler_installed = False
        self._original_showwarning = None

    def _capture_curses_screen(self):
        """Capture the current curses screen content with colors"""
        if not self._stdscr:
            return []

        captured_lines = []
        height, width = self._stdscr.getmaxyx()

        for y in range(height):
            # Get the entire line with character and color information
            line_chars = []
            for x in range(width):
                char_attr = self._stdscr.inch(y, x)
                char = char_attr & 0xFF  # Get character
                attr = char_attr & ~0xFF  # Get attributes (including color)

                if char == 0:  # Null character
                    char = ord(" ")

                line_chars.append((chr(char), attr))

            # Convert to colored string
            colored_line = self._convert_line_to_ansi(line_chars)

            # Remove trailing spaces but keep the line structure
            if (
                colored_line.strip() or y < height - 5
            ):  # Keep empty lines except at the very end
                captured_lines.append(colored_line.rstrip())

        # Remove trailing empty lines
        while captured_lines and not captured_lines[-1].strip():
            captured_lines.pop()

        return captured_lines

    def _convert_line_to_ansi(self, line_chars):
        """Convert a line of (char, attr) tuples to ANSI colored string"""
        if not line_chars:
            return ""

        result = ""
        current_color_pair = None
        current_attrs = None

        for char, attr in line_chars:
            # Extract color pair and other attributes
            color_pair = curses.pair_number(attr)
            other_attrs = attr & ~(curses.A_COLOR)

            # Check if we need to change colors/attributes
            if color_pair != current_color_pair or other_attrs != current_attrs:
                # Reset previous formatting
                if current_color_pair is not None or current_attrs is not None:
                    result += self.ANSI_RESET

                # Apply new formatting using the shared method
                result += self._convert_curses_attr_to_ansi(color_pair, other_attrs)

                current_color_pair = color_pair
                current_attrs = other_attrs

            result += char

        # Reset at the end of line
        if current_color_pair is not None or current_attrs is not None:
            result += self.ANSI_RESET

        return result

    def _convert_curses_attr_to_ansi(self, color_pair, other_attrs=0):
        """Convert curses color pair and attributes to ANSI sequence.

        This is the single source of truth for curses-to-ANSI conversion.
        """
        ansi_codes = []

        # Handle color pair using centralized mapping
        ansi_color = self.COLOR_PAIR_TO_ANSI.get(color_pair)
        if ansi_color:
            ansi_codes.append(ansi_color)

        # Handle attributes safely
        if other_attrs and "curses" in globals() and curses is not None:
            if other_attrs & curses.A_BOLD:
                ansi_codes.append("1")
            if other_attrs & curses.A_DIM:
                ansi_codes.append("2")
            if other_attrs & curses.A_UNDERLINE:
                ansi_codes.append("4")
            if other_attrs & curses.A_BLINK:
                ansi_codes.append("5")
            if other_attrs & curses.A_REVERSE:
                ansi_codes.append("7")

        if ansi_codes:
            return f"\033[{';'.join(ansi_codes)}m"
        return ""

    def _cleanup_curses(self, preserve_output=None):
        """Clean up curses resources and display complete progress history with correct colors"""
        if preserve_output is None:
            preserve_output = self._persist_on_exit

        # Stop the display thread
        if self._display_thread is not None and self._display_thread.is_alive():
            self._stop_thread.set()
            self._display_thread.join(timeout=1.0)

        # Restore warning handler regardless of curses state
        self._restore_warning_handler()

        if self._stdscr is not None:
            if "curses" in globals() and curses is not None:
                curses.curs_set(1)
                curses.nocbreak()
                curses.echo()
                curses.endwin()

            # Replay the complete progress history with ANSI colors.
            # endwin() restores the pre-curses terminal state, so the curses
            # content is no longer visible — replay is needed to preserve it.
            # NOTE: do NOT send \033[?1049l here — alternate screen is never
            # entered, and sending the exit escape corrupts the terminal.
            if preserve_output and self._curses_lines:
                print()
                for indent, message, status, level, location in self._curses_lines:
                    print(
                        self._format_line_ansi(indent, message, status, level, location),
                        flush=True,
                    )

            self._stdscr = None
            self._curses_mode = False

    def _get_status_color(self, status):
        """Get curses color pair for status text"""
        if not curses.has_colors():
            return 0
        return curses.color_pair(self._get_status_color_pair(status))

    def _get_color_pair_idx(self, level):
        idx_ = level % len(self.COLOR_PAIR_LEVEL_CYCLE)
        idx = self.COLOR_PAIR_LEVEL_CYCLE[idx_]
        return idx

    def _get_color_pair(self, level):
        return curses.color_pair(self._get_color_pair_idx(level))

    def _handle_input(self):
        """Handle user input for scrolling and pause"""
        if not self._stdscr:
            return

        while True:
            # try:
            key = self._stdscr.getch()
            if key == curses.ERR:
                break

            height, _ = self._stdscr.getmaxyx()
            max_lines = height - 1
            total_lines = len(self._curses_lines)
            max_scroll = max(0, total_lines - max_lines)

            if key == curses.KEY_UP:
                self._scroll_offset = min(
                    self._scroll_offset + self._scroll_step, max_scroll
                )
            elif key == curses.KEY_DOWN:
                self._scroll_offset = max(self._scroll_offset - self._scroll_step, 0)
            elif key == curses.KEY_PPAGE:  # Page Up
                self._scroll_offset = min(self._scroll_offset + max_lines, max_scroll)
            elif key == curses.KEY_NPAGE:  # Page Down
                self._scroll_offset = max(self._scroll_offset - max_lines, 0)
            elif key == curses.KEY_HOME:
                self._scroll_offset = max_scroll
            elif key == curses.KEY_END:
                self._scroll_offset = 0
            elif key == ord("p") or key == ord("P"):  # Toggle pause
                self._paused = not self._paused
                if self._paused:
                    self._pause_event.clear()  # Block execution
                else:
                    self._pause_event.set()  # Resume execution
            # except curses.error:
            #     break

    def _update_curses_display(self):
        """Update the curses display with current lines"""
        if not self._curses_mode or self._stdscr is None:
            return

        # Input handling is now done in the display thread via _handle_input

        self._stdscr.clear()
        height, width = self._stdscr.getmaxyx()

        # Calculate visible lines
        max_lines = height - 1  # Reserve one line for status
        total_lines = len(self._curses_lines)

        # Logic for scroll offset:
        # 0 means "stick to bottom" (auto-scroll)
        # >0 means "show lines offset from bottom"

        # Clamp scroll offset to valid range
        max_scroll = max(0, total_lines - max_lines)
        self._scroll_offset = min(self._scroll_offset, max_scroll)

        # Calculate start line based on scroll offset
        start_line = max(0, total_lines - max_lines - self._scroll_offset)
        end_line = min(total_lines, start_line + max_lines)

        # Display lines
        for i in range(start_line, end_line):
            if i >= len(self._curses_lines):
                break

            indent, message, status, level, location = self._curses_lines[i]
            display_row = i - start_line

            display_status = self._get_status_display_text(status)
            _status = display_status + " " if status != "" else ""

            location_text = f" ({location})" if location else ""

            char_levels = self.get_char_level(indent)

            col_pos = 0
            for j, char in enumerate(indent):
                if col_pos >= width - 1:
                    break
                char_level = char_levels[j]
                char_color = self._get_color_pair(char_level)
                self._stdscr.addstr(display_row, col_pos, char, char_color)
                col_pos += 1

            if status and col_pos + len(_status) <= width:
                status_color = self._get_status_color(status)
                self._stdscr.addstr(display_row, col_pos, _status, status_color)
                col_pos += len(_status)

            if col_pos + len(message) <= width:
                self._stdscr.addstr(display_row, col_pos, message)
                col_pos += len(message)
            elif col_pos < width - 1:
                self._stdscr.addstr(display_row, col_pos, message[: width - 1 - col_pos])
                col_pos = width - 1

            if location and col_pos + len(location_text) <= width:
                location_color = curses.color_pair(self.LOCATION_COLOR_PAIR)
                self._stdscr.addstr(display_row, col_pos, location_text, location_color)
                col_pos += len(location_text)

        # Add scroll indicator if needed
        if total_lines > max_lines:
            scroll_msg = f"Lines {start_line + 1}-{end_line} of {total_lines}"
            if self._scroll_offset > 0:
                scroll_msg += " (SCROLLED)"
            if self._paused:
                scroll_msg += " [PAUSED - Press 'p' to resume]"
            self._stdscr.addstr(height - 1, 0, scroll_msg[: width - 1])
        elif self._paused:
            # Show pause status even when all lines fit on screen
            scroll_msg = "[PAUSED - Press 'p' to resume]"
            self._stdscr.addstr(height - 1, 0, scroll_msg[: width - 1])

        self._stdscr.refresh()

    def clear_lines(self, n_lines):
        # Only used for stdout mode (curses and file modes handle clearing differently)
        pass

    def _remove_level(self):
        indent = self._get_indent(remove_level=True)
        self.add_line(indent=indent)

    def remove_level(self):
        if self.verbose == 0 or self.enabled is False:
            return

        if self._phantom_levels > 0:
            self._phantom_levels -= 1
            return

        if self._block_count > 0:
            self._block_count -= 1
            return

        # Pending levels (added but no message printed yet) — pop without visual changes
        if self._pending_levels > 0:
            self._current_level_indent = (
                self._current_level_indent - self.level_stack[-1]
            )
            self.level_stack.pop()
            self._pending_levels -= 1
            if self._pending_levels == 0:
                self.added_level = False
            self.removed_level = False
            return

        if not self.level or self.level[-1] == 0:
            return

        self._current_level_indent = self._current_level_indent - self.level_stack[-1]
        self.level_stack.pop()
        self.added_level = False
        self.removed_level = True

    def _add_level(self):
        pass

    def add_level(self, n=2):
        assert n >= 0, "Cannot add negative number of levels"
        if self.verbose == 0 or self.enabled is False:
            return

        if self.added_level:
            # Consecutive add_level with no visible message in between.
            # Treat as phantom (no visual indent); track so remove_level balances.
            self._phantom_levels += 1
            return

        if self.current_level + 2 > self.verbose:
            self._block_count += 1
            return

        self.level_stack.append(n)
        self._current_level_indent += n
        self._pending_levels += 1
        self.added_level = True

    def _get_line(self, s):
        match_idx = []
        for i, (indent, message, status) in enumerate(
            zip(self.indent, self.message, self.status)
        ):
            if message == s:
                match_idx.append(i)
        return match_idx

    def _get_indent(self, add_level=False, remove_level=False):
        assert not (
            add_level and remove_level
        ), "Cannot add and remove level at the same time"
        indent = ""
        _indent = ""
        for i in range(1, len(self.level_stack)):
            _indent += self.SPACE * self.level_stack[i - 1] + self.VERT

        if self._current_level_indent >= 1:
            if remove_level:
                indent = _indent
            elif add_level:
                indent = _indent
            else:
                indent = _indent + self.HOR * (self.level_stack[-1])
        return indent

    def __call__(
        self,
        message=None,
        status="",
        change_status=False,
        ignore_no_match=False,
        location=None,
    ):
        # Early bailout BEFORE any expensive operations
        if self.verbose == 0 or self.enabled is False:
            return

        assert message is None or isinstance(
            message, str
        ), "Message must be a string or None"

        # Wait if paused (blocks until resumed)
        self.wait_if_paused()

        # Check if message should be filtered by caller name (only if we have filters)
        # Note: Status filtering is handled by public methods (debug(), info(), etc.)
        if self._caller_filters:
            caller_name = self._get_caller_function_name()
            if self._should_filter_message(caller_name):
                return

        if change_status:
            if self._block_count > 0:
                ignore_no_match = True
            assert message is not None, "Cannot change status of None"
            match_idx = self._get_line(message)
            if len(match_idx) == 0:
                # Hard-raising on a missing line is too costly: a
                # mismatched format-string between ``LOGGER.task(...)``
                # and the closing ``LOGGER.ok(..., change_status=True)``
                # crashes long-running simulations / optimisations
                # purely because of a logging hiccup.  Instead we
                # silently no-op (the underlying work has already run)
                # so the user keeps their results, and surface the
                # mismatch via the env var so we can still find these
                # in development if we want to.
                if ignore_no_match or os.environ.get(
                    "TWIN4BUILD_LOGGER_STRICT", ""
                ).lower() not in ("1", "true", "yes"):
                    pass
                else:
                    raise ValueError(
                        f"Line not found: '{message}'"
                        + f"current level: {self.current_level}"
                        + f"verbose: {self.verbose}"
                    )
            elif len(match_idx) >= 1:
                idx = match_idx[-1]
                self.status[idx] = self._apply_color_to_status(
                    self.status[idx], status
                )
                self.print_lines()
        else:
            if self._block_count > 0:
                return

            if message is not None:
                if self._pending_levels > 0:
                    self._pending_levels = 0

                if self.removed_level and self._current_level_indent >= 1:
                    sep = ""
                    for si in range(1, len(self.level_stack)):
                        sep += self.SPACE * self.level_stack[si - 1] + self.VERT
                    self.add_line(indent=sep, message="", status="")

                if location is None and self._show_location:
                    location = self._get_caller_location()
                indent = self._get_indent()
                self.add_line(
                    indent=indent, message=message, status=status, location=location
                )
                self.print_lines()
                self.added_level = False
                self.removed_level = False
            else:
                pass

    def is_enabled_for(self, status_type):
        """Check if a status type is enabled. Use this to avoid expensive string operations.

        Args:
            status_type: One of "debug", "info", "warning", "error", "ok",
                         "phase", "step", "config", "iter", "outcome", "default"

        Returns:
            bool: True if messages of this type will be logged

        Example:
            if LOGGER.is_enabled_for("debug"):
                LOGGER.debug("Expensive calculation: %s", expensive_func())
        """
        if self.verbose == 0 or self.enabled is False:
            return False
        return self._status_filters.get(status_type.lower(), True)

    def debug(
        self, message, *args, change_status=False, ignore_no_match=False, location=None
    ):
        """Log a debug message. Filtered out if debug filter is disabled.

        Args:
            message: Message string, format string (if args provided), or callable returning string or None
            *args: Arguments for string formatting (lazy evaluation). Callables are invoked only if logging is enabled.
            change_status: Whether to change the status of an existing message
            ignore_no_match: Ignore if no matching message found for status change
            location: Optional caller location override

        Examples:
            LOGGER.debug("Simple message")
            LOGGER.debug("Value: %s", expensive_func())  # expensive_func() only called if debug enabled
            LOGGER.debug("Name: %s, Age: %d", name, age)
            LOGGER.debug("Result: %s", lambda: expensive_calculation())  # lambda only called if debug enabled
            LOGGER.debug(lambda: f"Complex: {expensive_func()}")  # entire message built only if debug enabled
            LOGGER.debug(lambda: LOGGER.debug("Nested call") or None)  # callable can execute Logger statements
        """
        # Fast filter check - avoid expensive operations if filtered; but let a
        # caller whitelist override so whitelisted callers still see this status type
        if not self._status_filters.get("debug", True) and not self._caller_whitelist_active:
            return
        # Evaluate callable message if needed
        if callable(message):
            message = message()
            # If callable returns None, it has already logged (or intentionally skipped)
            if message is None:
                return
        # Lazy string formatting - only format if not filtered
        if args:
            # Evaluate any callable arguments
            evaluated_args = tuple(arg() if callable(arg) else arg for arg in args)
            message = message % evaluated_args
        self(
            message,
            status="[DEBUG]",
            change_status=change_status,
            ignore_no_match=ignore_no_match,
            location=location,
        )

    def info(
        self, message, *args, change_status=False, ignore_no_match=False, location=None
    ):
        """Log an info message. Filtered out if info filter is disabled.

        Args:
            message: Message string, format string (if args provided), or callable returning string or None
            *args: Arguments for string formatting (lazy evaluation). Callables are invoked only if logging is enabled.
            change_status: Whether to change the status of an existing message
            ignore_no_match: Ignore if no matching message found for status change
            location: Optional caller location override
        """
        # Fast filter check - avoid expensive operations if filtered; but let a
        # caller whitelist override so whitelisted callers still see this status type
        if not self._status_filters.get("info", True) and not self._caller_whitelist_active:
            return
        # Evaluate callable message if needed
        if callable(message):
            message = message()
            # If callable returns None, it has already logged (or intentionally skipped)
            if message is None:
                return
        if args:
            # Evaluate any callable arguments
            evaluated_args = tuple(arg() if callable(arg) else arg for arg in args)
            message = message % evaluated_args
        self(
            message,
            status="[INFO]",
            change_status=change_status,
            ignore_no_match=ignore_no_match,
            location=location,
        )

    def warning(
        self, message, *args, change_status=False, ignore_no_match=False, location=None, warn_once=False
    ):
        """Log a warning. Dual mode:
        - change_status=False (default): prints new yellow [OUTCOME] line.
        - change_status=True: recolors existing line yellow.
        """
        if not self._status_filters.get("warning", True) and not self._caller_whitelist_active:
            return
        if callable(message):
            message = message()
            if message is None:
                return
        if args:
            evaluated_args = tuple(arg() if callable(arg) else arg for arg in args)
            message = message % evaluated_args
        if warn_once:
            if message in self._warned_once_messages:
                return
            self._warned_once_messages.add(message)
        self(
            message,
            status="[OUTCOME:yellow]" if not change_status else "[OUTCOME:yellow]",
            change_status=change_status,
            ignore_no_match=ignore_no_match,
            location=location,
        )

    def error(
        self, message, *args, change_status=False, ignore_no_match=False, location=None
    ):
        """Log an error. Dual mode:
        - change_status=False (default): prints new red [OUTCOME] line.
        - change_status=True: recolors existing line red.
        """
        if not self._status_filters.get("error", True) and not self._caller_whitelist_active:
            return
        if callable(message):
            message = message()
            if message is None:
                return
        if args:
            evaluated_args = tuple(arg() if callable(arg) else arg for arg in args)
            message = message % evaluated_args
        self(
            message,
            status="[OUTCOME:red]" if not change_status else "[OUTCOME:red]",
            change_status=change_status,
            ignore_no_match=ignore_no_match,
            location=location,
        )

    def ok(
        self, message, *args, change_status=False, ignore_no_match=False, location=None
    ):
        """Log success. Dual mode:
        - change_status=False (default): prints new green [OUTCOME] line.
        - change_status=True: recolors existing line green.
        """
        if not self._status_filters.get("ok", True) and not self._caller_whitelist_active:
            return
        if callable(message):
            message = message()
            if message is None:
                return
        if args:
            evaluated_args = tuple(arg() if callable(arg) else arg for arg in args)
            message = message % evaluated_args
        self(
            message,
            status="[OUTCOME:green]" if not change_status else "[OUTCOME:green]",
            change_status=change_status,
            ignore_no_match=ignore_no_match,
            location=location,
        )

    def task(
        self, message, *args, location=None
    ):
        """Log a task entry. Tasks are units of work that can self-nest.

        Always prints a new [TASK] line. Use add_level() after to indent children.
        Close with remove_level() and ok/warning/error(msg, change_status=True)
        to propagate color.
        """
        if not self._status_filters.get("task", True) and not self._caller_whitelist_active:
            return
        if callable(message):
            message = message()
            if message is None:
                return
        if args:
            evaluated_args = tuple(arg() if callable(arg) else arg for arg in args)
            message = message % evaluated_args
        self(
            message,
            status="[TASK]",
            location=location,
        )

    def section(
        self, message, *args, location=None
    ):
        """Log a section entry. Sections are data-organizing containers.

        Use for structural grouping (not temporal workflow). Children should only
        be [SECTION], [INFO], [ITER], or [RESULT] -- never [TASK], [OUTCOME],
        or [DEBUG].
        """
        if not self._status_filters.get("section", True) and not self._caller_whitelist_active:
            return
        if callable(message):
            message = message()
            if message is None:
                return
        if args:
            evaluated_args = tuple(arg() if callable(arg) else arg for arg in args)
            message = message % evaluated_args
        self(
            message,
            status="[SECTION]",
            location=location,
        )

    def result(
        self, message, *args, location=None
    ):
        """Log a result/output data line. Can self-nest for hierarchical results.

        Use for reported results from computation, not for input/setup values
        (use config() for those).
        """
        if not self._status_filters.get("result", True) and not self._caller_whitelist_active:
            return
        if callable(message):
            message = message()
            if message is None:
                return
        if args:
            evaluated_args = tuple(arg() if callable(arg) else arg for arg in args)
            message = message % evaluated_args
        self(
            message,
            status="[RESULT]",
            location=location,
        )

    def config(
        self, message, *args, location=None
    ):
        """Log a configuration value. Format: 'Label: value'."""
        if not self._status_filters.get("config", True) and not self._caller_whitelist_active:
            return
        if callable(message):
            message = message()
            if message is None:
                return
        if args:
            evaluated_args = tuple(arg() if callable(arg) else arg for arg in args)
            message = message % evaluated_args
        self(
            message,
            status="[CONFIG]",
            location=location,
        )

    def iter(
        self, message, *args, location=None
    ):
        """Log iteration metrics. Format: 'Eval N: key=value | key=value (Xs)'."""
        if not self._status_filters.get("iter", True) and not self._caller_whitelist_active:
            return
        if callable(message):
            message = message()
            if message is None:
                return
        if args:
            evaluated_args = tuple(arg() if callable(arg) else arg for arg in args)
            message = message % evaluated_args
        self(
            message,
            status="[ITER]",
            location=location,
        )

    def finalize(self):
        """Finalize the progress display and ensure output persists"""
        if self._curses_mode:
            self._cleanup_curses(preserve_output=True)

    def reset(self):
        # Clean up curses before resetting
        self._cleanup_curses(preserve_output=True)

        self._flush_log_buffer()  # Write any remaining buffered lines before reset
        self.level_indent = []  # level as function of line number
        self.level = []
        self.indent = []
        self.message = []
        self.status = []
        self.location = []
        self.added_level = False
        self._pending_levels = 0
        self._phantom_levels = 0
        self.removed_level = False
        self.level_stack = [0]
        self.has_printed = False
        self._current_level_indent = 0
        self._block_count = 0
        self._last_file_content = ""
        self._is_active = False
        self._log_buffer = []
        self._flushed_line_count = 0
        self._curses_lines = []
        self._paused = False
        self._pause_event.set()  # Ensure not paused after reset
        # Don't reset user-configured settings — only printing history.
        # Preserved: _verbose, logfile, _status_filters, _caller_filters,
        #            _caller_filter_mode, _warned_once_messages, _show_location,
        #            _log_flush_size, _atexit_registered

    def __del__(self):
        """Destructor to ensure curses cleanup"""
        self._cleanup_curses(preserve_output=True)


def reset_print(f):
    """
    Decorator that resets LOGGER state when call depth returns to 0.

    IMPORTANT: This decorator breaks profiling tools (cProfile) due to identity
    collision. All wrapped functions share the same wrapper identity, causing
    incorrect cumulative time attribution in profiler output.

    To disable for accurate profiling, set environment variable:
        DISABLE_AUTORESET_PRINT=1

    The decorator overhead is negligible (~microseconds), so disabling it for
    profiling does not significantly change performance characteristics.

    Examples:
        PowerShell: $env:DISABLE_AUTORESET_PRINT='1'; python script.py
        CMD:        set DISABLE_AUTORESET_PRINT=1 && python script.py
        Bash:       DISABLE_AUTORESET_PRINT=1 python script.py
    """
    # Check if decorator should be disabled (for profiling)
    if os.environ.get("DISABLE_AUTORESET_PRINT", "0") == "1":
        return f  # Return function unwrapped - transparent to profiler

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        LOGGER.call_depth += 1
        try:
            result = f(*args, **kwargs)
        finally:
            LOGGER.call_depth -= 1
            if LOGGER.call_depth == 0:
                LOGGER.reset()
        return result

    return wrapper


def autoreset_print(cls):
    """
    Class decorator that applies @reset_print to all methods in the class.
    This ensures that LOGGER context is managed correctly even when
    methods are called independently.
    """
    if os.environ.get("DISABLE_AUTORESET_PRINT", "0") == "1":
        return cls  # Return class unmodified - transparent to profiler

    for name, attr in cls.__dict__.items():
        if isinstance(attr, staticmethod):
            setattr(cls, name, staticmethod(reset_print(attr.__func__)))
        elif isinstance(attr, classmethod):
            setattr(cls, name, classmethod(reset_print(attr.__func__)))
        elif callable(attr):
            setattr(cls, name, reset_print(attr))
    return cls


LOGGER = Logger()
# if is_testing():
#     LOGGER.disable()
# else:
#     LOGGER.enable()

if __name__ == "__main__":

    # print_color_palette()

    for i in range(1, 51):
        print(i)

    # # Create a new model
    # model = tb.Model(id="estimator_example")

    # Load the model from semantic file
    # filename_simulation = utils.get_path(["estimator_example", "semantic_model.ttl"])

    # logfile = r"C:\Users\jabj\Documents\python\Twin4Build\twin4build\utils\log.txt"
    # model.load(simulation_model_filename=filename_simulation, verbose=0, logfile=None)

    p = Logger()
    p.verbose = 50

    # DEBUG: Test get_char_level method
    test_line1 = "LOADING"
    test_line2 = "|______Hello, world!"
    test_line3 = "|      |______Deep nesting"
    test_line4 = "|      |      |______Deep deep nesting"

    print(f"Testing get_char_level:")
    print(f"Line1: '{test_line1}' -> {p.get_char_level(test_line1)}")
    print(f"  _get_indices: {p._get_indices(test_line1)}")
    print(f"Line2: '{test_line2}' -> {p.get_char_level(test_line2)}")
    print(f"  _get_indices: {p._get_indices(test_line2)}")
    print(f"Line3: '{test_line3}' -> {p.get_char_level(test_line3)}")
    print(f"  _get_indices: {p._get_indices(test_line3)}")
    print(f"Line4: '{test_line4}' -> {p.get_char_level(test_line4)}")
    print(f"  _get_indices: {p._get_indices(test_line4)}")
    print()

    # p.add_level(5)

    p("Level 0", status="[OK]")
    # time.sleep(3)
    # time.sleep(3)

    # print(p.level)
    # aa

    p.add_level()
    p("Level 1", status="[ERROR]")
    # p("Level 1")

    # time.sleep(1)
    p.add_level()
    p("Level 2", status="[WARNING]")
    # p("Level 2")
    # time.sleep(1)
    p.add_level(3)
    p("Level 3")
    p.add_level()
    p("Level 4")
    p.add_level(5)
    p("Level 5")
    p.add_level(1)
    p("Level 6")

    time.sleep(5)
