from __future__ import annotations
from typing import Union
import twin4build.saref.command.command as command
class Function:
    def __init__(self,
                hasCommand: Union[command.Command, None]=None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(hasCommand, command.Command) or hasCommand is None, "Attribute \"hasCommand\" is of type \"" + str(type(hasCommand)) + "\" but must be of type \"" + str(command.Command) + "\""
        self.hasCommand = hasCommand
