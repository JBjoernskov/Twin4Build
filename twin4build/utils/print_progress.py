
class PrintProgress:
    def __init__(self) -> None:
        self.level = 0
        self.has_printed = False
        self.plain = False

    def remove_level(self):
        if self.level>0:
            self.level -= 1

    def add_level(self):
        self.level += 1
        print("")
        self.has_printed = False
        
    def __call__(self, s=None, plain=False):
        if s is not None:
            if self.level==0:
                print("")
            indent = ""
            if self.level==1:
                if plain==False:
                    indent = "   |___"
                else:
                    indent = "|   "
            elif self.level>1:
                if plain==False:
                    indent = "    "*self.level + "|___"
                else:
                    indent = "    "*(self.level-1) + "|   "
            if self.has_printed and self.plain==False:
                print("done")

            if plain==False:
                print(indent+s+"...", end="")
            else:
                print(indent+s)
            self.has_printed = True
            self.plain = plain
        else:
            print("")

