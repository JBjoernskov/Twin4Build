import twin4build.base as base
from typing import Union
from twin4build.logger.Logging import Logging
from twin4build.base import Schedule
from twin4build.utils.time_series_input import TimeSeriesInputSystem
logger = Logging.get_logger("ai_logfile")
logger.info("RulebasedController file")
class SetpointController(base.Controller):
    def __init__(self,
                 isReverse: Union[bool, None] = None,
                #  hasSetpointSchedule: Union[Schedule, TimeSeriesInputSystem, None] = None,
                **kwargs):
        assert isinstance(isReverse, bool) or isReverse is None, "Attribute \"isReverse\" is of type \"" + str(type(isReverse)) + "\" but must be of type \"" + str(bool) + "\""
        self.isReverse = isReverse
        super().__init__(**kwargs)
        # assert isinstance(hasSetpointSchedule, Schedule) or isinstance(hasSetpointSchedule, TimeSeriesInputSystem) or hasSetpointSchedule is None, "Attribute \"hasSetpointSchedule\" is of type \"" + str(type(hasSetpointSchedule)) + "\" but must be of either type \"" + Schedule.__name__ + "\" or type \"" + TimeSeriesInputSystem.__name__ + "\""
        # self.hasSetpointSchedule = hasSetpointSchedule