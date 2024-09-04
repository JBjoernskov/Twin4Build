

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

class Device(physical_object.PhysicalObject):
    def __init__(self,
                accomplishes: Union[task.Task, None]=None,
                consistsOf: Union[device.Device, None]=None,
                observes: Union[property_.Property, None]=None,
                controls: Union[list, None]=None, ##################
                hasFunction: Union[function.Function, None]=None,
                hasManufacturer: Union[str, None]=None,
                hasModel: Union[str, None]=None,
                hasProfile: Union[profile.Profile, None]=None,
                hasState: Union[state.State, None]=None,
                isUsedFor: Union[commodity.Commodity, None]=None,
                makesMeasurement: Union[measurement.Measurement, None]=None,
                # observes: Union(property_.Property, None)=None,
                offers: Union[service.Service, None]=None,
                **kwargs):
        
        
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
        assert isinstance(observes, property_.Property) or observes is None, "Attribute \"observes\" is of type \"" + str(type(observes)) + "\" but must be of type \"" + str(property_.Property) + "\""
        assert isinstance(controls, list) or controls is None, "Attribute \"controls\" is of type \"" + str(type(controls)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(hasFunction, function.Function) or hasFunction is None, "Attribute \"hasFunction\" is of type \"" + str(type(hasFunction)) + "\" but must be of type \"" + str(function.Function) + "\""
        assert isinstance(hasManufacturer, measurement.Measurement) or hasManufacturer is None, "Attribute \"hasManufacturer\" is of type \"" + str(type(hasManufacturer)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(hasModel, measurement.Measurement) or hasModel is None, "Attribute \"hasModel\" is of type \"" + str(type(hasModel)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(hasProfile, profile.Profile) or hasProfile is None, "Attribute \"hasProfile\" is of type \"" + str(type(hasProfile)) + "\" but must be of type \"" + str(profile.Profile) + "\""
        assert isinstance(hasState, state.State) or hasState is None, "Attribute \"hasState\" is of type \"" + str(type(hasState)) + "\" but must be of type \"" + str(state.State) + "\""
        assert isinstance(isUsedFor, commodity.Commodity) or isUsedFor is None, "Attribute \"isUsedFor\" is of type \"" + str(type(isUsedFor)) + "\" but must be of type \"" + str(commodity.Commodity) + "\""
        assert isinstance(makesMeasurement, measurement.Measurement) or makesMeasurement is None, "Attribute \"makesMeasurement\" is of type \"" + str(type(makesMeasurement)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        # assert isinstance(observes, property_.Property) or observes is None, "Attribute \"observes\" is of type \"" + str(type(observes)) + "\" but must be of type \"" + str(property_.Property) + "\""
        assert isinstance(offers, service.Service) or offers is None, "Attribute \"offers\" is of type \"" + str(type(offers)) + "\" but must be of type \"" + str(service.Service) + "\""
        self.accomplishes = accomplishes
        self.consistsOf = consistsOf
        self.observes = observes
        if controls is None:
            controls = []
        self.controls = controls ####
        self.hasFunction = hasFunction
        self.hasManufacturer = hasManufacturer
        self.hasModel = hasModel
        self.hasProfile = hasProfile
        self.hasState = hasState
        self.isUsedFor = isUsedFor
        self.makesMeasurement = makesMeasurement
        # self.observes = observes
        self.offers = offers
        


