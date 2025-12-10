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

# Try to import curses (not available on all systems)
try:
    # Standard library imports
    import curses

    CURSES_AVAILABLE = True
except ImportError:
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
    curses.wrapper(_print_color_palette)


class PrintProgress:
    def __init__(self) -> None:
        self.level_indent = []  # level as function of line number
        self.level = []
        self.indent = []
        self.message = []
        self.status = []
        self.location = []
        self.added_level = False
        self.removed_level = False
        self.level_stack = [0]
        self.has_printed = False
        self._verbose = 3
        self._current_level_indent = 0
        self._block_count = 0
        self.logfile = None
        self._is_active = False
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
        self._use_curses = CURSES_AVAILABLE and not self.is_interactive() and not is_ci
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
        self.OK_COLOR_PAIR = 3       # Green
        self.ERROR_COLOR_PAIR = 5    # Red
        self.WARNING_COLOR_PAIR = 7  # Yellow
        self.INFO_COLOR_PAIR = 2     # Blue
        self.LOCATION_COLOR_PAIR = 6 # Cyan/Magenta for file:line locations
        
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
        self._show_location = True

    def __enter__(self):
        """Context manager entry - ensures proper cleanup on exceptions"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - cleanup curses if active"""
        if self._curses_mode:
            self._cleanup_curses(preserve_output=True)
        return False  # Don't suppress exceptions

    @property
    def enabled(self):
        # Check _IS_TESTING dynamically to handle cases where it's set after import
        from twin4build import _IS_TESTING, _IMPORT_COMPLETE
        if not _IMPORT_COMPLETE:
            return False
        if _IS_TESTING:
            return False
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

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
    def is_active(self):
        return self._is_active

    @property
    def verbose(self):
        return int(self._verbose)

    @verbose.setter
    def verbose(self, value):
        assert isinstance(value, (int)), "verbose must be an integer or None"
        self._verbose = value

    @property
    def current_level(self):
        return len(self.level_stack) - 1

    def get_log(self):
        if self.logfile is not None:
            f = open(self.logfile, "w")
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
        """Return 'filename:lineno' for the caller of PRINTPROGRESS"""
        try:
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
        except Exception:
            pass
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
        """
        status_lower = status.lower()
        if "[ok]" in status_lower or "[success]" in status_lower:
            return self.OK_COLOR_PAIR
        elif "[error]" in status_lower or "[failed]" in status_lower:
            return self.ERROR_COLOR_PAIR
        elif "[warning]" in status_lower or "[warn]" in status_lower:
            return self.WARNING_COLOR_PAIR
        else:
            return self.INFO_COLOR_PAIR

    def _get_status_ansi_color(self, status):
        """Get ANSI color escape sequence for a status string"""
        return self._get_ansi_color(self._get_status_color_pair(status))

    def _format_message(self, message, location, use_ansi_colors=False):
        if location:
            if use_ansi_colors:
                loc_color = self._get_ansi_color(self.LOCATION_COLOR_PAIR)
                return f"{message} {loc_color}({location}){self.ANSI_RESET}"
            else:
                return f"{message} ({location})"
        return message

    def add_line(self, indent="", message="", status="", location=None):
        self.indent.append(indent)
        self.message.append(message)
        self.status.append(status)
        self.location.append(location)
        self.level_indent.append(self._current_level_indent)
        self.level.append(self.current_level)
        self._is_active = True

    def print_lines(self):
        f = self.get_log()

        # Initialize curses if needed and we're not logging to file
        if self._use_curses and f is None and not self._curses_mode:
            self._init_curses()

        if self._curses_mode and f is None:
            # In curses mode, update the data structure. The display thread handles drawing.
            # This simulates the clearing behavior of traditional printing
            with self._lock:
                # Skip updating if paused
                if not self._paused:
                    # Store current lines to calculate diff for anti-fighting logic
                    current_len = len(self._curses_lines)

                    # Rebuild the list of lines
                    temp_lines = []
                    for indent, message, status, level, location in zip(
                        self.indent, self.message, self.status, self.level, self.location
                    ):
                        temp_lines.append((indent, message, status, level, location))

                    self._curses_lines = temp_lines
                    new_len = len(self._curses_lines)
                    diff = new_len - current_len

                    # Anti-fighting logic: if scrolled up, maintain relative position
                    if self._scroll_offset > 0 and diff > 0:
                        self._scroll_offset += diff
        else:
            # Use traditional printing
            if self.has_printed:
                self.clear_lines(self.n_printed)

            # Use ANSI colors if outputting to terminal (not to file)
            use_ansi = f is None and sys.stdout.isatty()
            
            self.n_printed = 0
            for indent, message, status, level, location in zip(
                self.indent, self.message, self.status, self.level, self.location
            ):
                # if level+1 <= self.verbose:
                _status = "..." + status if status != "" else ""
                display_message = self._format_message(message, location, use_ansi_colors=use_ansi)
                s = indent + display_message + _status
                print(s, flush=True, file=f)
                self.n_printed += 1

                # time.sleep(0.2)

            # Also update curses lines for potential final output
            # This ensures the final state matches what's currently visible
            if self._use_curses:
                with self._lock:
                    # Skip updating if paused
                    if not self._paused:
                        current_len = len(self._curses_lines)

                        temp_lines = []
                        for indent, message, status, level, location in zip(
                            self.indent, self.message, self.status, self.level, self.location
                        ):
                            temp_lines.append((indent, message, status, level, location))

                        self._curses_lines = temp_lines
                        new_len = len(self._curses_lines)
                        diff = new_len - current_len

                        # Anti-fighting logic
                        if self._scroll_offset > 0 and diff > 0:
                            self._scroll_offset += diff

        if f is not None:
            f.close()
        self.has_printed = True

        # time.sleep(0.7)

        # self.added_level = False
        # self.removed_level = False

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
                curses.assume_default_colors(-1, -1) # Available in python 3.14
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
        self._stdscr.bkgd(' ', curses.color_pair(0))
        self._stdscr.clear()

        # Start the display thread
        self._stop_thread.clear()
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
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
                    try:
                        self._handle_input()
                        # Always update display to show pause status and handle scrolling
                        self._update_curses_display()
                    except Exception:
                        pass
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

            # Then call the original exception handler to show the traceback
            original_excepthook(exc_type, exc_value, exc_traceback)

        sys.excepthook = curses_exception_handler
        self._exception_handler_installed = True

    def _install_warning_handler(self):
        """Redirect warnings to PRINTPROGRESS during curses mode so they persist"""
        if self._warning_handler_installed:
            return

        self._original_showwarning = warnings.showwarning

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            # While in curses, push warnings into the progress log so they are replayed after teardown.
            if self._curses_mode:
                try:
                    # Use the filename and lineno provided by Python's warning system
                    # These are the actual location where warnings.warn() was called
                    loc = f"{os.path.basename(filename)}:{lineno}"
                    self(str(message), status="[WARNING]", location=loc)
                except Exception:
                    pass

            # Forward to the original handler when not in curses, or when an explicit
            # target file is provided (e.g., logging to file).
            if (not self._curses_mode) or file is not None:
                try:
                    self._original_showwarning(message, category, filename, lineno, file, line)
                except Exception:
                    pass

        warnings.showwarning = _showwarning
        self._warning_handler_installed = True

    def _restore_warning_handler(self):
        """Restore default warnings.showwarning"""
        if self._warning_handler_installed and self._original_showwarning is not None:
            try:
                warnings.showwarning = self._original_showwarning
            except Exception:
                pass
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
            try:
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
            except AttributeError:
                pass

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
            # Clean up curses
            # Check if curses is available (it might be None during interpreter shutdown)
            if "curses" in globals() and curses is not None:
                try:
                    curses.curs_set(1)
                    curses.nocbreak()
                    curses.echo()
                    curses.endwin()
                except Exception:
                    pass

            # Exit alternate screen buffer if we used it
            # This restores the original terminal content
            try:
                sys.stdout.write("\033[?1049l")  # Exit alternate screen
                sys.stdout.flush()
            except Exception:
                pass

            # Display the COMPLETE progress history with colors
            # Only needed when using alternate screen (otherwise output is already visible)
            if preserve_output and self._curses_lines:
                print()  # Add some spacing
                for indent, message, status, level, location in self._curses_lines:
                    # Build the main text (indent + message, without location)
                    main_text = indent + message
                    
                    # Build location text if present
                    location_text = f" ({location})" if location else ""

                    # Get character-level colors using numpy method
                    char_levels = self.get_char_level(main_text)

                    # Build colored output character by character
                    colored_output = ""
                    for i, char in enumerate(main_text):
                        char_level = char_levels[i]
                        color_pair_idx = self._get_color_pair_idx(char_level)
                        char_color = self._get_ansi_color(color_pair_idx)
                        colored_output += f"{char_color}{char}{self.ANSI_RESET}"

                    # Add location with its own color
                    if location:
                        loc_color = self._get_ansi_color(self.LOCATION_COLOR_PAIR)
                        colored_output += f"{loc_color}{location_text}{self.ANSI_RESET}"

                    # Add status with appropriate color
                    _status = ""
                    if status:
                        status_color = self._get_status_ansi_color(status)
                        _status = f"...{status_color}{status}{self.ANSI_RESET}"

                    # Print the complete colored line
                    try:
                        print(f"{colored_output}{_status}", flush=True)
                    except Exception:
                        pass

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
            try:
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
                    self._scroll_offset = max(
                        self._scroll_offset - self._scroll_step, 0
                    )
                elif key == curses.KEY_PPAGE:  # Page Up
                    self._scroll_offset = min(
                        self._scroll_offset + max_lines, max_scroll
                    )
                elif key == curses.KEY_NPAGE:  # Page Down
                    self._scroll_offset = max(self._scroll_offset - max_lines, 0)
                elif key == curses.KEY_HOME:
                    self._scroll_offset = max_scroll
                elif key == curses.KEY_END:
                    self._scroll_offset = 0
                elif key == ord('p') or key == ord('P'):  # Toggle pause
                    self._paused = not self._paused
                    if self._paused:
                        self._pause_event.clear()  # Block execution
                    else:
                        self._pause_event.set()  # Resume execution
            except curses.error:
                break

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

            # Create status text
            _status = "..." + status if status != "" else ""

            # Build the main text (indent + message, without location)
            main_text = indent + message
            
            # Build location text if present
            location_text = f" ({location})" if location else ""

            # Get character-level colors using numpy method (for indent coloring)
            char_levels = self.get_char_level(main_text)

            # Display main text character by character with appropriate colors
            col_pos = 0
            for j, char in enumerate(main_text):
                if col_pos >= width - 1:
                    break

                # Get the level for this character
                char_level = char_levels[j]
                char_color = self._get_color_pair(char_level)
                self._stdscr.addstr(display_row, col_pos, char, char_color)
                col_pos += 1

            # Add location with its own color if there's room
            if location and col_pos + len(location_text) <= width:
                location_color = curses.color_pair(self.LOCATION_COLOR_PAIR)
                self._stdscr.addstr(display_row, col_pos, location_text, location_color)
                col_pos += len(location_text)

            # Add colored status if there's room
            if status and col_pos + len(_status) <= width:
                status_color = self._get_status_color(status)
                self._stdscr.addstr(display_row, col_pos, _status, status_color)

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
        # time.sleep(3)
        if self.is_interactive() and self.get_log() is None:
            # Third party imports
            from IPython.display import clear_output

            clear_output()
        elif self._curses_mode:
            # Curses handles clearing automatically - no need to do anything
            pass
        else:
            f = self.get_log()

            if f is not None:
                f.close()

        # self.added_level = False
        # self.removed_level = False

    def _remove_level(self):
        indent = self._get_indent(remove_level=True)
        self.add_line(indent=indent)

    def remove_level(self):
        if self.verbose == 0 or self.enabled is False:
            return

        if self.level[-1] == 0:
            return  # "Already at the root level. Cannot remove level."

        if self._block_count > 0:
            self._block_count -= 1
            return

        self._current_level_indent = self._current_level_indent - self.level_stack[-1]

        if self.added_level:  # Undo add_level
            for _ in range(self.level_stack[-1]):
                self.level.pop()
                self.level_indent.pop()
                self.indent.pop()
                self.message.pop()
                self.status.pop()
                self.location.pop()
            self.level_stack.pop()
            self.removed_level = False
        else:
            if self.removed_level:
                self.level.pop()
                self.level_indent.pop()
                self.indent.pop()
                self.message.pop()
                self.status.pop()
                self.location.pop()
            self.level_stack.pop()
            self._remove_level()
            self.removed_level = True

        self.added_level = False

    def _add_level(self):
        indent = self._get_indent(add_level=True)
        if indent != "":
            self.add_line(indent=indent)

    def add_level(self, n=2):
        assert n >= 0, "Cannot add negative number of levels"
        if self.verbose == 0 or self.enabled is False:
            return
        if self.current_level + 2 > self.verbose:  # +2 because of the added level
            self._block_count += 1
            return

        if self.added_level:  # changed_level?
            self.level_stack[-1] += n
        else:
            self.level_stack.append(n)  # what about if we just removed a level?
        self._current_level_indent += n
        for _ in range(n):
            self._add_level()
        self.added_level = True
        self.removed_level = False

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
        assert message is None or isinstance(
            message, str
        ), "Message must be a string or None"
        if self.verbose == 0 or self.enabled is False:
            return
        
        # Wait if paused (blocks until resumed)
        self.wait_if_paused()
        
        # change_status = False
        if change_status:
            if self._block_count > 0:
                ignore_no_match = True
            assert message is not None, "Cannot change status of None"
            match_idx = self._get_line(message)
            if len(match_idx) == 0:
                if ignore_no_match:
                    pass
                else:
                    raise ValueError(
                        f"Line not found: '{message}'"
                        + f"current level: {self.current_level}"
                        + f"verbose: {self.verbose}"
                    )
            elif len(match_idx) > 1:
                # Multiple lines found, use the last one
                self.status[match_idx[-1]] = status
                self.print_lines()
                # raise ValueError("Multiple lines found")
            elif len(match_idx) == 1:
                self.status[match_idx[0]] = status
                self.print_lines()
        else:
            if self._block_count > 0:
                return

            if message is not None:
                if location is None and self._show_location:
                    location = self._get_caller_location()
                indent = self._get_indent()
                self.add_line(indent=indent, message=message, status=status, location=location)
                self.print_lines()
                self.added_level = False
                self.removed_level = False
            else:
                pass

    def finalize(self):
        """Finalize the progress display and ensure output persists"""
        if self._curses_mode:
            self._cleanup_curses(preserve_output=True)

    def reset(self):
        # Clean up curses before resetting
        self._cleanup_curses(preserve_output=True)

        self.level_indent = []  # level as function of line number
        self.level = []
        self.indent = []
        self.message = []
        self.status = []
        self.location = []
        self.added_level = False
        self.removed_level = False
        self.level_stack = [0]
        self.has_printed = False
        # self._verbose = 3 # dont reset verbose level
        self._current_level_indent = 0
        self._block_count = 0
        self.logfile = None
        self._is_active = False
        self._curses_lines = []
        self._paused = False
        self._pause_event.set()  # Ensure not paused after reset
        # Note: We don't reset _atexit_registered so cleanup remains registered

    def __del__(self):
        """Destructor to ensure curses cleanup"""
        self._cleanup_curses(preserve_output=True)


def reset_print(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        PRINTPROGRESS.call_depth += 1
        try:
            result = f(*args, **kwargs)
        finally:
            PRINTPROGRESS.call_depth -= 1
            if PRINTPROGRESS.call_depth == 0:
                PRINTPROGRESS.reset()
        return result

    return wrapper


def autoreset_print(cls):
    """
    Class decorator that applies @reset_print to all methods in the class.
    This ensures that PRINTPROGRESS context is managed correctly even when
    methods are called independently.
    """
    for name, attr in cls.__dict__.items():
        if isinstance(attr, staticmethod):
            setattr(cls, name, staticmethod(reset_print(attr.__func__)))
        elif isinstance(attr, classmethod):
            setattr(cls, name, classmethod(reset_print(attr.__func__)))
        elif callable(attr):
            setattr(cls, name, reset_print(attr))
    return cls


PRINTPROGRESS = PrintProgress()
# if is_testing():
#     PRINTPROGRESS.disable()
# else:
#     PRINTPROGRESS.enable()

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

    p = PrintProgress()
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