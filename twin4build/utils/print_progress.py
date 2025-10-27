# Standard library imports
import datetime
import time
import sys
import os
import atexit
from itertools import cycle

# Third party imports
import __main__
from tkinter.constants import S
from dateutil import tz
import numpy as np

# Try to import curses (not available on all systems)
try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False

# Local application imports
# %pip install twin4build # Uncomment in google colab
import twin4build as tb
import twin4build.examples.utils as utils


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
        curses.init_pair(i+1, i, -1)
    
    # Add a header
    stdscr.addstr(0, 0, f"Available colors (showing {max_pairs} colors):\n\n")
    
    try:
        # Display colors with their numbers
        row = 2
        col = 0
        for i in range(0, max_pairs+1):
            color_text = f"{i:3d} "
            
            # Move to next row if we reach the right edge
            if col + len(color_text) >= curses.COLS:
                row += 1
                col = 0
                if row >= curses.LINES - 1:  # Leave room for instructions
                    break
            
            stdscr.addstr(row, col, color_text, curses.color_pair(i))
            col += len(color_text)
    
    except curses.ERR:
        # End of screen reached
        pass
    
    # Add instructions at the bottom
    try:
        stdscr.addstr(curses.LINES-1, 0, "Press any key to exit...")
    except curses.ERR:
        pass
    
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
        self.added_level = False
        self.removed_level = False
        self.level_stack = [0]
        self.has_printed = False
        self._verbose = 3
        self._current_level_indent = 0
        self._block_count = 0
        self.logfile = None
        self._is_active = False
        # Curses-related attributes
        self._use_curses = CURSES_AVAILABLE and not self.is_interactive()
        self._curses_mode = False
        self._stdscr = None
        self._curses_lines = []  # Store lines for curses display
        self._persist_on_exit = True  # Whether to show final output after curses cleanup
        self._atexit_registered = False  # Track if we've registered cleanup
        self.VERT = "|"
        self.HOR = "_"*3
        self.SPACE = " "*3
        self.COLOR_PAIR_LEVEL_CYCLE = [8, 4]#, c2]
        self.OK_COLOR_PAIR = 3
        self.ERROR_COLOR_PAIR = 5
        self.WARNING_COLOR_PAIR = 7
        self.INFO_COLOR_PAIR = 2
    
    def __enter__(self):
        """Context manager entry - ensures proper cleanup on exceptions"""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - cleanup curses if active"""
        if self._curses_mode:
            try:
                self._cleanup_curses(preserve_output=True)
            except:
                # Emergency cleanup
                try:
                    curses.endwin()
                    sys.stdout.write('\033[?1049l')
                    sys.stdout.flush()
                except:
                    pass
        return False  # Don't suppress exceptions

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
            char_level[prev:next] = l+1
        char_level[indices[-1]:] = len(indices)
        return char_level


    def add_line(self, indent="", message="", status=""):
        self.indent.append(indent)
        self.message.append(message)
        self.status.append(status)
        self.level_indent.append(self._current_level_indent)
        self.level.append(self.current_level)
        self._is_active = True

    def print_lines(self):
        f = self.get_log()
        
        # Initialize curses if needed and we're not logging to file
        if self._use_curses and f is None and not self._curses_mode:
            self._init_curses()
        
        if self._curses_mode and f is None:
            # In curses mode, clear the previous display and update with current lines
            # This simulates the clearing behavior of traditional printing
            self._curses_lines = []
            for indent, message, status, level in zip(
                self.indent, self.message, self.status, self.level, strict=True
            ):
                self._curses_lines.append((indent, message, status, level))
            
            # Use curses display
            self._update_curses_display()
        else:
            # Use traditional printing
            if self.has_printed:
                self.clear_lines(self.n_printed)

            self.n_printed = 0
            for indent, message, status, level in zip(
                self.indent, self.message, self.status, self.level, strict=True
            ):
                # if level+1 <= self.verbose:
                _status = "..." + status if status != "" else ""
                s = indent + message + _status
                print(s, flush=True, file=f)
                self.n_printed += 1

                # time.sleep(0.2)
            
            # Also update curses lines for potential final output
            # This ensures the final state matches what's currently visible
            if self._use_curses:
                self._curses_lines = []
                for indent, message, status, level in zip(
                    self.indent, self.message, self.status, self.level, strict=True
                ):
                    self._curses_lines.append((indent, message, status, level))
        
        if f is not None:
            f.close()
        self.has_printed = True

        # time.sleep(0.7)

        # self.added_level = False
        # self.removed_level = False

    def is_interactive(self):
        return not hasattr(__main__, "__file__")
    
    def _init_curses(self):
        """Initialize curses mode using alternate screen buffer (like vim)"""
        if not self._use_curses or self._curses_mode:
            return False
        
        try:
            print("DEBUG: Starting curses initialization", file=sys.stderr)
            # Enter alternate screen buffer first (like vim does)
            # This preserves the current terminal content and creates a separate "window"
            sys.stdout.write('\033[?1049h')  # Enter alternate screen
            sys.stdout.flush()
            print("DEBUG: Alternate screen buffer entered", file=sys.stderr)
            
            # Now start curses in the alternate screen
            self._stdscr = curses.initscr()
            print("DEBUG: curses.initscr() completed", file=sys.stderr)
            curses.noecho()
            curses.cbreak()
            curses.curs_set(0)  # Hide cursor
            
            # Clear the alternate screen to start fresh
            self._stdscr.clear()
            
            # Enable color if available
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                max_pairs = min(curses.COLOR_PAIRS - 1, curses.COLORS)
                print(f"DEBUG: Initializing {max_pairs} color pairs", file=sys.stderr)
                for i in range(0, max_pairs):
                    curses.init_pair(i+1, i, -1)
                    

                
            
            # Register cleanup function to run at exit (only once)
            if not self._atexit_registered:
                atexit.register(self._cleanup_curses, preserve_output=True)
                self._atexit_registered = True
            
            # Install exception handler to ensure curses cleanup on crashes
            self._install_exception_handler()
            
            self._curses_mode = True
            print("DEBUG: Curses initialization completed successfully", file=sys.stderr)
            return True
        except Exception as e:
            # Log the specific error for debugging
            print(f"DEBUG: Exception in curses setup: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            
            # If curses setup fails, clean up and fall back to regular terminal
            self._cleanup_curses()
            return False
    
    def _install_exception_handler(self):
        """Install a global exception handler to cleanup curses on crashes"""
        if hasattr(self, '_exception_handler_installed'):
            return  # Already installed
        
        original_excepthook = sys.excepthook
        
        def curses_exception_handler(exc_type, exc_value, exc_traceback):
            # Clean up curses first if we're in curses mode
            if self._curses_mode and self._stdscr is not None:
                try:
                    print(f"DEBUG: Exception occurred, cleaning up curses: {exc_type.__name__}", file=sys.stderr)
                    # Try normal cleanup first (preserves output)
                    self._cleanup_curses(preserve_output=True)
                    print("DEBUG: Curses cleanup completed after exception", file=sys.stderr)
                except Exception as cleanup_e:
                    # Emergency cleanup if normal cleanup fails
                    print(f"DEBUG: Cleanup failed, attempting emergency cleanup: {cleanup_e}", file=sys.stderr)
                    try:
                        curses.endwin()
                        sys.stdout.write('\033[?1049l')  # Exit alternate screen
                        sys.stdout.flush()
                        print("DEBUG: Emergency cleanup completed", file=sys.stderr)
                    except Exception as emergency_e:
                        print(f"DEBUG: Emergency cleanup also failed: {emergency_e}", file=sys.stderr)

            
            # Then call the original exception handler to show the traceback
            original_excepthook(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = curses_exception_handler
        self._exception_handler_installed = True
    
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
                    char = ord(' ')
                
                line_chars.append((chr(char), attr))

            
            # Convert to colored string
            colored_line = self._convert_line_to_ansi(line_chars)
            
            # Remove trailing spaces but keep the line structure
            if colored_line.strip() or y < height - 5:  # Keep empty lines except at the very end
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
                    result += "\033[0m"
                
                # Apply new formatting
                ansi_codes = []
                
                # Handle color pair
                if color_pair > 0:
                    # Map specific color pairs to the correct ANSI colors based on your scheme
                    color_mapping = {
                        2: "34",   # Blue (for status and levels)
                        3: "32",   # Green (for [OK] status)
                        4: "36",   # Cyan (for level 1, 4, etc.)
                        5: "31",   # Red (for level 2, 5, etc.)
                        7: "33",   # Yellow (for warnings)
                        8: "37",   # White (for level 0, 3, 6, etc.)
                    }
                    
                    ansi_color = color_mapping.get(color_pair)
                    if ansi_color:
                        ansi_codes.append(ansi_color)
                
                # Handle attributes
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
                
                # Apply the ANSI codes
                if ansi_codes:
                    result += f"\033[{';'.join(ansi_codes)}m"
                
                current_color_pair = color_pair
                current_attrs = other_attrs
            
            result += char
        
        # Reset at the end of line
        if current_color_pair is not None or current_attrs is not None:
            result += "\033[0m"
        
        return result

    def _sample_curses_colors(self):
        """Sample actual colors from curses display to build accurate color mapping"""
        if not self._stdscr:
            return {}
        
        color_mapping = {}
        height, width = self._stdscr.getmaxyx()
        
        # Sample a few positions to determine actual color pairs being used
        for y in range(height):  # Sample first 20 lines
            for x in range(width):  # Sample first all chars per line
                char_attr = self._stdscr.inch(y, x)
                char = char_attr & 0xFF
                attr = char_attr & ~0xFF
                
                if char != ord(' ') and char != 0:  # Skip spaces and nulls
                    color_pair = curses.pair_number(attr)
                    other_attrs = attr & ~(curses.A_COLOR)
                    
                    # Only process valid color pair numbers (0-255 range)
                    if 0 <= color_pair <= 255:
                        # Store the mapping from color_pair to ANSI
                        if color_pair not in color_mapping:
                            color_mapping[color_pair] = self._convert_curses_attr_to_ansi(color_pair, other_attrs)

                # Check if we have all the colors we need
                needed_pairs = set(self.COLOR_PAIR_LEVEL_CYCLE + [self.OK_COLOR_PAIR, self.ERROR_COLOR_PAIR, self.WARNING_COLOR_PAIR, self.INFO_COLOR_PAIR])
                if needed_pairs.issubset(set(color_mapping.keys())):
                    return color_mapping

        # Add OK, ERROR, WARNING, INFO color pairs
        other_attrs = curses.color_pair(self.OK_COLOR_PAIR) & ~(curses.A_COLOR)
        color_mapping[self.OK_COLOR_PAIR] = self._convert_curses_attr_to_ansi(self.OK_COLOR_PAIR, other_attrs)

        other_attrs = curses.color_pair(self.ERROR_COLOR_PAIR) & ~(curses.A_COLOR)
        color_mapping[self.ERROR_COLOR_PAIR] = self._convert_curses_attr_to_ansi(self.ERROR_COLOR_PAIR, other_attrs)
        
        other_attrs = curses.color_pair(self.WARNING_COLOR_PAIR) & ~(curses.A_COLOR)
        color_mapping[self.WARNING_COLOR_PAIR] = self._convert_curses_attr_to_ansi(self.WARNING_COLOR_PAIR, other_attrs)
        
        other_attrs = curses.color_pair(self.INFO_COLOR_PAIR) & ~(curses.A_COLOR)
        color_mapping[self.INFO_COLOR_PAIR] = self._convert_curses_attr_to_ansi(self.INFO_COLOR_PAIR, other_attrs)
        
        # # Ensure we have at least the basic color pairs, add defaults if missing
        # for pair in self.COLOR_PAIR_LEVEL_CYCLE:
        #     if pair not in color_mapping:
        #         color_mapping[pair] = self._convert_curses_attr_to_ansi(pair, 0)
        
        return color_mapping

    def _convert_curses_attr_to_ansi(self, color_pair, other_attrs):
        """Convert curses color pair and attributes to ANSI sequence"""
        ansi_codes = []
        
        # Handle color pair using our mapping
        color_mapping = {
            2: "34",   # Blue
            3: "32",   # Green  
            4: "36",   # Cyan
            5: "31",   # Red
            7: "33",   # Yellow
            8: "37",   # White
        }
        
        ansi_color = color_mapping.get(color_pair)
        if ansi_color:
            ansi_codes.append(ansi_color)
        
        # Handle attributes
        if other_attrs & curses.A_BOLD:
            ansi_codes.append("1")
        if other_attrs & curses.A_DIM:
            ansi_codes.append("2")
        if other_attrs & curses.A_UNDERLINE:
            ansi_codes.append("4")
        
        if ansi_codes:
            return f"\033[{';'.join(ansi_codes)}m"
        else:
            return "\033[0m"

    def _cleanup_curses(self, preserve_output=None):
        """Clean up curses resources and display complete progress history with correct colors"""
        if preserve_output is None:
            preserve_output = self._persist_on_exit
            
        # Sample the actual colors from curses before cleanup
        sampled_colors = {}
        if self._stdscr is not None and preserve_output:
            sampled_colors = self._sample_curses_colors()
        
        if self._stdscr is not None:
            try:
                # Clean up curses
                print(f"DEBUG: Starting curses cleanup, curses_mode={self._curses_mode}", file=sys.stderr)
                curses.curs_set(1)
                curses.nocbreak()
                curses.echo()
                curses.endwin()
                print("DEBUG: curses.endwin() completed", file=sys.stderr)
                
                # Exit alternate screen buffer - this restores the original terminal content
                sys.stdout.write('\033[?1049l')  # Exit alternate screen
                sys.stdout.flush()
                print("DEBUG: Alternate screen buffer exited", file=sys.stderr)
                
                # Display the COMPLETE progress history with sampled colors
                if preserve_output and self._curses_lines:
                    print("DEBUG: Starting to display progress history", file=sys.stderr)
                    print()  # Add some spacing
                    for indent, message, status, level in self._curses_lines:
                        _status = "..." + status if status != "" else ""
                        
                        # Build the full line
                        main_text = indent + message
                        full_line = main_text + _status
                        
                        # Get character-level colors using your numpy method
                        char_levels = self.get_char_level(full_line)
                        
                        # Build colored output character by character using actual curses colors
                        colored_output = ""
                        
                        # Color the main text character by character
                        for i, char in enumerate(main_text):
                            char_level = char_levels[i]
                            
                            # Map level to curses color pair (same logic as _get_color_pair)
                            color_pair_idx = self._get_color_pair_idx(char_level)
                            
                            # Use sampled color if available, otherwise fallback
                            # if color_pair in sampled_colors:
                            char_color = sampled_colors[color_pair_idx]
                            # else:
                            #     char_color = self._get_ansi_level_color(char_level)
                            
                            colored_output += f"{char_color}{char}\033[0m"
                        
                        # Add status color
                        if status:
                            status_lower = status.lower()
                            if '[ok]' in status_lower or '[success]' in status_lower:
                                status_color = "\033[32m"  # Green
                            elif '[error]' in status_lower or '[failed]' in status_lower:
                                status_color = "\033[31m"  # Red
                            elif '[warning]' in status_lower or '[warn]' in status_lower:
                                status_color = "\033[33m"  # Yellow
                            else:
                                status_color = "\033[34m"  # Blue
                            _status = f"...{status_color}{status}\033[0m"
                        
                        # Print the complete colored line
                        print(f"{colored_output}{_status}", flush=True)
                    print("DEBUG: Progress history display completed", file=sys.stderr)
            except Exception as e:
                # Log the specific error for debugging
                print(f"DEBUG: Exception in cleanup: {type(e).__name__}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                
                # Emergency cleanup if normal cleanup fails
                try:
                    print("DEBUG: Attempting emergency cleanup", file=sys.stderr)
                    curses.endwin()
                    sys.stdout.write('\033[?1049l')  # Exit alternate screen
                    sys.stdout.flush()
                    print("DEBUG: Emergency cleanup completed", file=sys.stderr)
                except Exception as emergency_e:
                    print(f"DEBUG: Emergency cleanup also failed: {emergency_e}", file=sys.stderr)
                
                # Re-raise the original exception so we can see what went wrong
                raise e
            finally:
                print("DEBUG: Setting cleanup flags", file=sys.stderr)
                self._stdscr = None
                self._curses_mode = False
    
    def _get_status_color(self, status):
        """Get color for status text in curses mode"""
        if not curses.has_colors():
            return 0
            
        status_lower = status.lower()
        if '[ok]' in status_lower or '[success]' in status_lower:
            return curses.color_pair(self.OK_COLOR_PAIR)  # Green
        elif '[error]' in status_lower or '[failed]' in status_lower:
            return curses.color_pair(self.ERROR_COLOR_PAIR)  # Red  
        elif '[warning]' in status_lower or '[warn]' in status_lower:
            return curses.color_pair(self.WARNING_COLOR_PAIR)  # Yellow
        else:
            return curses.color_pair(self.INFO_COLOR_PAIR)  # Blue

    def _get_color_pair_idx(self, level):
        idx_ = level % len(self.COLOR_PAIR_LEVEL_CYCLE)
        idx = self.COLOR_PAIR_LEVEL_CYCLE[idx_]
        return idx

    def _get_color_pair(self, level):
        return curses.color_pair(self._get_color_pair_idx(level))
    
    def _update_curses_display(self):
        """Update the curses display with current lines"""
        if not self._curses_mode or self._stdscr is None:
            return
            
        self._stdscr.clear()
        height, width = self._stdscr.getmaxyx()
        
        # Calculate visible lines
        max_lines = height - 1  # Reserve one line for status
        total_lines = len(self._curses_lines)
        
        # Auto-scroll to show latest lines
        start_line = max(0, total_lines - max_lines) if total_lines > max_lines else 0
        end_line = min(total_lines, start_line + max_lines)
        
        # Display lines
        for i in range(start_line, end_line):
            if i >= len(self._curses_lines):
                break
                
            indent, message, status, level = self._curses_lines[i]
            display_row = i - start_line
            
            # Create status text
            _status = "..." + status if status != "" else ""
        
            # Build the full line text
            main_text = indent + message
            full_line = main_text + _status
            
            # Get character-level colors using your numpy method
            char_levels = self.get_char_level(full_line)
            
            # Display main text character by character with appropriate colors
            col_pos = 0
            for i, char in enumerate(main_text):
                if col_pos >= width - 1:
                    break
                    
                # Get the level for this character
                char_level = char_levels[i]
                char_color = self._get_color_pair(char_level)
                self._stdscr.addstr(display_row, col_pos, char, char_color)
                col_pos += 1
            
            # Add colored status if there's room
            if status and col_pos + len(_status) <= width:
                status_color = self._get_status_color(status)
                self._stdscr.addstr(display_row, col_pos, _status, status_color)
                    

        
        # Add scroll indicator if needed
        if total_lines > max_lines:
            scroll_info = f"Lines {start_line + 1}-{end_line} of {total_lines}"
            try:
                self._stdscr.addstr(height - 1, 0, scroll_info[:width - 1])
            except curses.error:
                pass
        
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
        if self.verbose==0:
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
            self.level_stack.pop()
            self.removed_level = False
        else:
            if self.removed_level:
                self.level.pop()
                self.level_indent.pop()
                self.indent.pop()
                self.message.pop()
                self.status.pop()
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
        if self.verbose==0:
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
        self, message=None, status="", change_status=False, ignore_no_match=False
    ):
        assert message is None or isinstance(message, str), "Message must be a string or None"
        if self.verbose==0:
            return
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
                raise ValueError("Multiple lines found")
            elif len(match_idx) == 1:
                self.status[match_idx[0]] = status
                self.print_lines()
        else:
            if self._block_count > 0:
                return

            if message is not None:
                indent = self._get_indent()
                self.add_line(indent=indent, message=message, status=status)
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
        self._cleanup_curses(preserve_output=False)
        
        self.level_indent = []  # level as function of line number
        self.level = []
        self.indent = []
        self.message = []
        self.status = []
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
        # Note: We don't reset _atexit_registered so cleanup remains registered
    
    def __del__(self):
        """Destructor to ensure curses cleanup"""
        try:
            self._cleanup_curses(preserve_output=True)
        except:
            pass

def reset_print(f):
    def wrapper(*args, **kwargs):
        if not PRINTPROGRESS.is_active:
            reset_PRINTPROGRESS = True
        else:
            reset_PRINTPROGRESS = False
        f(*args, **kwargs)
        if reset_PRINTPROGRESS:
            PRINTPROGRESS.reset()
    return wrapper


PRINTPROGRESS = PrintProgress()

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
