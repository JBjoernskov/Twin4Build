import twin4build.core as core
from typing import Union
class SetpointController(core.Controller):
    def __init__(self,
                 isReverse: Union[bool, None] = None,
                **kwargs):
        assert isinstance(isReverse, bool) or isReverse is None, "Attribute \"isReverse\" is of type \"" + str(type(isReverse)) + "\" but must be of type \"" + str(bool) + "\""
        self.isReverse = isReverse
        super().__init__(**kwargs)