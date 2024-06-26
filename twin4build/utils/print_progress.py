
class PrintProgress:
    def __init__(self) -> None:
        self.level = 0
        self.has_printed = False

    def remove_level(self):
        if self.level>0:
            self.level -= 1

    def add_level(self):
        self.level += 1
        print("")
        self.has_printed = False
        
    def __call__(self, s=None):
        if s is not None:
            if self.level==0:
                print("")
            indent = "   |___"*self.level
            if self.has_printed:
                print("done")
            print(indent+s+"...", end="")
            self.has_printed = True
        else:
            print("")

