

from __future__ import annotations
from typing import Union
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import twin4build.saref.task.task as task
    import twin4build.saref.device.device as device
    import twin4build.saref.property_.property_ as property_
    import twin4build.saref.function.function as function
    import twin4build.saref.profile.profile as profile
    import twin4build.saref.state.state as state
    import twin4build.saref.commodity.commodity as commodity
    import twin4build.saref.measurement.measurement as measurement
    import twin4build.saref.service.service as service

import os 
import sys 

uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 4)
sys.path.append(file_path)

import twin4build.saref4bldg.physical_object.physical_object as physical_object

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class Device(physical_object.PhysicalObject):
    def __init__(self,
                accomplishes: Union(task.Task, None)=None,
                consistsOf: Union(device.Device, None)=None,
                controlsProperty: Union(property_.Property, None)=None,
                actuatesProperty: Union(property_.Property, None)=None, ##################
                hasFunction: Union(function.Function, None)=None,
                hasManufacturer: Union(str, None)=None,
                hasModel: Union(str, None)=None,
                hasProfile: Union(profile.Profile, None)=None,
                hasState: Union(state.State, None)=None,
                isUsedFor: Union(commodity.Commodity, None)=None,
                makesMeasurement: Union(measurement.Measurement, None)=None,
                measuresProperty: Union(property_.Property, None)=None,
                offers: Union(service.Service, None)=None,
                **kwargs):
        
        logger.info("[Saref.Device Class] : Entered in Inititalise Function")
        
        super().__init__(**kwargs)
        import twin4build.saref.task.task as task
        import twin4build.saref.device.device as device
        import twin4build.saref.property_.property_ as property_
        import twin4build.saref.function.function as function
        import twin4build.saref.profile.profile as profile
        import twin4build.saref.state.state as state
        import twin4build.saref.commodity.commodity as commodity
        import twin4build.saref.measurement.measurement as measurement
        import twin4build.saref.service.service as service
        assert isinstance(accomplishes, task.Task) or accomplishes is None, "Attribute \"accomplishes\" is of type \"" + str(type(accomplishes)) + "\" but must be of type \"" + str(task.Task) + "\""
        assert isinstance(consistsOf, device.Device) or consistsOf is None, "Attribute \"consistsOf\" is of type \"" + str(type(consistsOf)) + "\" but must be of type \"" + str(device.Device) + "\""
        assert isinstance(controlsProperty, property_.Property) or controlsProperty is None, "Attribute \"controlsProperty\" is of type \"" + str(type(controlsProperty)) + "\" but must be of type \"" + str(property_.Property) + "\""
        assert isinstance(actuatesProperty, property_.Property) or actuatesProperty is None, "Attribute \"actuatesProperty\" is of type \"" + str(type(actuatesProperty)) + "\" but must be of type \"" + str(property_.Property) + "\""
        assert isinstance(hasFunction, function.Function) or hasFunction is None, "Attribute \"hasFunction\" is of type \"" + str(type(hasFunction)) + "\" but must be of type \"" + str(function.Function) + "\""
        assert isinstance(hasManufacturer, measurement.Measurement) or hasManufacturer is None, "Attribute \"hasManufacturer\" is of type \"" + str(type(hasManufacturer)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(hasModel, measurement.Measurement) or hasModel is None, "Attribute \"hasModel\" is of type \"" + str(type(hasModel)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(hasProfile, profile.Profile) or hasProfile is None, "Attribute \"hasProfile\" is of type \"" + str(type(hasProfile)) + "\" but must be of type \"" + str(profile.Profile) + "\""
        assert isinstance(hasState, state.State) or hasState is None, "Attribute \"hasState\" is of type \"" + str(type(hasState)) + "\" but must be of type \"" + str(state.State) + "\""
        assert isinstance(isUsedFor, commodity.Commodity) or isUsedFor is None, "Attribute \"isUsedFor\" is of type \"" + str(type(isUsedFor)) + "\" but must be of type \"" + str(commodity.Commodity) + "\""
        assert isinstance(makesMeasurement, measurement.Measurement) or makesMeasurement is None, "Attribute \"makesMeasurement\" is of type \"" + str(type(makesMeasurement)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(measuresProperty, property_.Property) or measuresProperty is None, "Attribute \"measuresProperty\" is of type \"" + str(type(measuresProperty)) + "\" but must be of type \"" + str(property_.Property) + "\""
        assert isinstance(offers, service.Service) or offers is None, "Attribute \"offers\" is of type \"" + str(type(offers)) + "\" but must be of type \"" + str(service.Service) + "\""
        self.accomplishes = accomplishes
        self.consistsOf = consistsOf
        self.controlsProperty = controlsProperty
        self.actuatesProperty = actuatesProperty ####
        self.hasFunction = hasFunction
        self.hasManufacturer = hasManufacturer
        self.hasModel = hasModel
        self.hasProfile = hasProfile
        self.hasState = hasState
        self.isUsedFor = isUsedFor
        self.makesMeasurement = makesMeasurement
        self.measuresProperty = measuresProperty
        self.offers = offers

        logger.info("[Saref.Device Class] : Exited from Inititalise Function")
        


