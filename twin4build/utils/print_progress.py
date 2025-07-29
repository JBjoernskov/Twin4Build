class PrintProgress:
    def __init__(self) -> None:
        self.level = 0
        self.has_printed = False
        self.plain = False
        self.saved_text = ""

    def remove_level(self):
        indent = self._get_indent(remove_level=True)
        if self.level > 0:
            self.level -= 1
        self.saved_text += indent + "\n"

    def add_level(self):
        indent = self._get_indent(add_level=True)
        self.level += 1
        if indent != "":
            self.saved_text += indent + "\n"
        # if self.has_printed and self.plain==False:
        #     print("")
        # self.has_printed = False

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

            if self.has_printed and self.plain == False:
                print(status)
            if self.saved_text != "":
                print(self.saved_text, end="")
                self.saved_text = ""

            indent = self._get_indent()
            if plain == False:
                print(indent + s + "...", end="")
            else:
                print(indent + s)
            self.has_printed = True
            self.plain = plain
            self.status = status
        else:
            print("")


PRINTPROGRESS = PrintProgress()
