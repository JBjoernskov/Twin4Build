import twin4build.systems.utils.fmu.fmu_component as fmu_component
from twin4build.utils.uppath import uppath
import os
from twin4build.utils.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing
import twin4build.core as core


class SpaceHeaterSystem(fmu_component.FMUComponent):
    def __init__(self,
                 waterFlowRateMax=None,
                **kwargs):
        super().__init__(**kwargs)
        self.start_time = 0
        fmu_filename = "Radiator.FMU"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = fmu_component.unzip_fmu(self.fmu_path)

        self.input = {"supplyWaterTemperature": None,
                      "waterFlowRate": None,
                      "indoorTemperature": None}
        self.output = {"outletWaterTemperature": None,
                       "PowerToRadiator": None,
                       "EnergyToRadiator": None}
        
        #Used in finite difference jacobian approximation for uncertainty analysis.
        self.inputLowerBounds = {"valvePosition": 0}
        self.inputUpperBounds = {"valvePosition": 1}


        self.FMUinputMap = {"supplyWaterTemperature": "supplyWaterTemperature",
                            "waterFlowRate": "waterFlowRate",
                            "indoorTemperature": "indoorTemperature"}
        self.FMUoutputMap = {"outletWaterTemperature": "outletWaterTemperature",
                            "PowerToRadiator": "PowerToRadiator",
                            "EnergyToRadiator": "EnergyToRadiator"}

        self.FMUparameterMap = {"self.outputCapacity": "Q_flow_nominal",
                                "nominalRoomTemperature": "Kv",
                                "nominalSupplyTemperature": "dpFixed_nominal",
                                "nominalReturnTemperature": "dpFixed_nominal",
                                "nominalSupplyTemperature": "dpFixed_nominal"} ####################################################
        
        
        self.input_conversion = {"valvePosition": do_nothing}
        self.output_conversion = {"waterFlowRate": do_nothing,
                                       "valvePosition": do_nothing}

        self.INITIALIZED = False

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass
        
    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None,
                    model=None):
        '''
            This function initializes the FMU component by setting the start_time and fmu_filename attributes, 
            and then sets the parameters for the FMU model.
        '''
        if self.INITIALIZED:
            self.reset()
        else:
            self.initialize_fmu()
            self.INITIALIZED = True


        