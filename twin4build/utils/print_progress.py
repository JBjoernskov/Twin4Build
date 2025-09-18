# Standard library imports
import datetime
import time

# Third party imports
import __main__
from dateutil import tz

# Local application imports
# %pip install twin4build # Uncomment in google colab
import twin4build as tb
import twin4build.examples.utils as utils


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
        self._verbose = 0
        self._current_level_indent = 0
        self._block_count = 0
        self.logfile = None
        self._is_active = False

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

    def add_line(self, indent="", message="", status=""):
        self.indent.append(indent)
        self.message.append(message)
        self.status.append(status)
        self.level_indent.append(self._current_level_indent)
        self.level.append(self.current_level)
        self._is_active = True

    def print_lines(self):
        if self.has_printed:
            self.clear_lines(self.n_printed)

        f = self.get_log()
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
        if f is not None:
            f.close()
        self.has_printed = True

        # time.sleep(0.7)

        # self.added_level = False
        # self.removed_level = False

    def is_interactive(self):
        return not hasattr(__main__, "__file__")

    def clear_lines(self, n_lines):
        # time.sleep(3)
        if self.is_interactive() and self.get_log() is None:
            # Third party imports
            from IPython.display import clear_output

            clear_output()
        else:
            f = self.get_log()
            LINE_UP = "\033[1A"
            LINE_CLEAR = "\x1b[2K"
            for _ in range(n_lines):
                print(LINE_UP, end=LINE_CLEAR, flush=True, file=f)
            if f is not None:
                f.close()

        # self.added_level = False
        # self.removed_level = False

    def _remove_level(self):
        indent = self._get_indent(remove_level=True)
        self.add_line(indent=indent)

    def remove_level(self):
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

    def get_line(self, s):
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
            _indent += "   " * self.level_stack[i - 1] + "|"

        if self._current_level_indent >= 1:
            if remove_level:
                indent = _indent
            elif add_level:
                indent = _indent
            else:
                indent = _indent + "___" * (self.level_stack[-1])
        return indent

    def __call__(
        self, message=None, status="", change_status=False, ignore_no_match=False
    ):

        # change_status = False
        if change_status:
            if self._block_count > 0:
                ignore_no_match = True
            assert message is not None, "Cannot change status of None"
            match_idx = self.get_line(message)
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

    def reset(self):
        self.level_indent = []  # level as function of line number
        self.level = []
        self.indent = []
        self.message = []
        self.status = []
        self.added_level = False
        self.removed_level = False
        self.level_stack = [0]
        self.has_printed = False
        self._verbose = 0
        self._current_level_indent = 0
        self._block_count = 0
        self.logfile = None
        self._is_active = False


PRINTPROGRESS = PrintProgress()

if __name__ == "__main__":
    for i in range(1, 51):
        print(i)

    # # Create a new model
    model = tb.Model(id="estimator_example")

    # Load the model from semantic file
    filename_simulation = utils.get_path(["estimator_example", "semantic_model.ttl"])

    logfile = r"C:\Users\jabj\Documents\python\Twin4Build\twin4build\utils\log.txt"
    model.load(simulation_model_filename=filename_simulation, verbose=0, logfile=None)

    # p = PrintProgress()
    # p.verbose = 50

    # # p.add_level(5)

    # p("LOADING", status="[OK]")
    # # time.sleep(3)

    # # print(p.level)
    # # aa

    # p.add_level()
    # p("Hello, world!1")
    # p.add_level()
    # p.remove_level()
    # p("Hello, world!2")
    # p.add_level()
    # p.remove_level()

    # # time.sleep(3)
    # p.add_level(3)
    # p("Hello, world!333")
    # p("Hellasdasdadas")
    # p.add_level()
    # p.remove_level()
    # p.remove_level()
    # p("Hellasdasdadas", status="[OK]", change_status=True)

    # p.remove_level()
    # p("Hello, world!333", status="[DSJAKLLSJDKDKJLA]", change_status=True)
    # p("Hello, world!3")
    # p("aaa")
    # p("Hello, world!2", status="[CHANGE]", change_status=True)
    # # p("Hello, world!3", status="[changed]", change_status=True) ################
    # # p.remove_level()
    # p("DAKLSDKLMSKLM")
    # # p.remove_level()
    # p("Hello, world!4")
