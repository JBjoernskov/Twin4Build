from __future__ import annotations
from typing import Union
import twin4build.saref.function.function as function
import twin4build.saref.state.state as state
class Command:
    def __init__(self,
                actsUpon: Union[state.State, None]=None,
                isCommandOf: Union[function.Function, None]=None,
                **kwargs):
        
        assert isinstance(actsUpon, state.State) or actsUpon is None, "Attribute \"actsUpon\" is of type \"" + str(type(actsUpon)) + "\" but must be of type \"" + str(state.State) + "\""
        assert isinstance(isCommandOf, function.Function) or isCommandOf is None, "Attribute \"isCommandOf\" is of type \"" + str(type(isCommandOf)) + "\" but must be of type \"" + str(function.Function) + "\""

