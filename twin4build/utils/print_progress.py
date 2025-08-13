class PrintProgress:
    def __init__(self) -> None:
        self.level = 0
        self.has_printed = False

        self.has_printed_status = False
        self.has_printed_after_add_level = False
        self.has_printed_after_remove_level = False
        self.plain = False
        self.saved_text = ""

    def remove_level(self):
        if (
            self.has_printed_after_add_level == False
            or self.has_printed_after_remove_level == False
        ):
            self.saved_text = ""

        if self.has_printed_status == False and self.plain == False:
            print(self.status)
            self.has_printed_status = True

        indent = self._get_indent(remove_level=True)
        if self.level > 0:
            self.level -= 1
        self.saved_text += indent + "\n"
        self.has_printed_after_remove_level = False

    def add_level(self):
        indent = self._get_indent(add_level=True)
        self.level += 1
        if indent != "":
            self.saved_text += indent + "\n"
        self.has_printed_after_add_level = False

    def _get_indent(self, add_level=False, remove_level=False):
        assert not (
            add_level and remove_level
        ), "Cannot add and remove level at the same time"
        indent = ""
        if self.level >= 1:
            if remove_level:
                indent = "   |" * (self.level - 1)
            elif add_level:
                indent = "   |" * (self.level + 1)
            else:
                indent = "   |" * self.level + "___"
        return indent

    def __call__(self, s=None, plain=False, status="[OK]"):
        if s is not None:
            if self.level == 0:
                print("")

            if (
                self.has_printed
                and self.has_printed_status == False
                and self.plain == False
            ):
                print(self.status)
                self.has_printed_status = True

            if self.saved_text != "":
                print(self.saved_text, end="")
                self.saved_text = ""

            indent = self._get_indent()
            if plain == False:
                print(indent + s + "...", end="", flush=True)
                self.has_printed_status = False
            else:
                print(indent + s)
            self.has_printed = True
            self.has_printed_after_remove_level = True
            self.has_printed_after_add_level = True
            self.plain = plain
            self.status = status
        else:
            if self.has_printed and self.plain == False:
                print(self.status)
            else:
                print("")


PRINTPROGRESS = PrintProgress()

if __name__ == "__main__":
    # Standard library imports
    import time

    p = PrintProgress()
    p.add_level()
    p("Hello, world!", status="[OK]")
    time.sleep(3)
    p.add_level()
    p("Hello, world!", status="[KKK]")
    time.sleep(4)
    p.add_level()
    p("Hello, world!", status="[TTT]")
    p.remove_level()
    p.remove_level()
    # p.remove_level()
    p("Hello, world!", status="[QQQ]")
    p.remove_level()
    p("Hello, world!", status="[DDD]")
    # p()
