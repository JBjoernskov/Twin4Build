import Saref4Syst

import ControllerModel
import ValveModel
import SpaceHeaterModel
import CoilModel
import AirToAirHeatRecoveryModel
import FanModel
import DamperModel
import FlowMeterModel

import math


class PhysicalObject(Saref4Syst.System, Saref4Syst.Connection):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        


class BuildingSpace(Saref4Syst.System):
    def __init__(self,
                hasSpace = None,
                isSpaceOf = None,
                contains = None,
                **kwargs):
        super().__init__(**kwargs)
        self.hasSpace = hasSpace
        self.isSpaceOf = isSpaceOf
        self.contains = contains

    def update_output(self):
        self.output["indoorTemperature"] = 23
        self.output["co2Concentration"] = 550

        





###
class Device(PhysicalObject):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)



###
class BuildingDevice(Device):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)


###
class DistributionDevice(BuildingDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        






###
class DistributionControlDevice(DistributionDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

class DistributionFlowDevice(DistributionDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)


###
class Controller(DistributionControlDevice, ControllerModel.ControllerModel):
    def __init__(self,
                isTemperatureController = None,
                isCo2Controller = None,
                **kwargs):
        super().__init__(**kwargs)
        self.isTemperatureController = isTemperatureController
        self.isCo2Controller = isCo2Controller

        


###
class FlowController(DistributionFlowDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

class FlowTerminal(DistributionFlowDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

class EnergyConversionDevice(DistributionFlowDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

class FlowMovingDevice(DistributionFlowDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)





###
class Valve(FlowController, ValveModel.ValveModel):
    def __init__(self,
                closeOffRating = None, 
                flowCoefficient = None, 
                size = None, 
                testPressure = None, 
                valveMechanism = None, 
                valveOperation = None, 
                valvePattern = None, 
                workingPressure = None,
                waterFlowRateMax = None, ###
                valveAuthority = None,
                save = False,
                **kwargs):
        super().__init__(**kwargs)

        self.closeOffRating = closeOffRating
        self.flowCoefficient = flowCoefficient
        self.size = size
        self.testPressure = testPressure
        self.valveMechanism = valveMechanism
        self.valveOperation = valveOperation
        self.valvePattern = valvePattern
        self.workingPressure = workingPressure
        self.waterFlowRateMax = waterFlowRateMax ###
        self.valveAuthority = valveAuthority ###
        self.save = save ###




class SpaceHeater(FlowTerminal, SpaceHeaterModel.SpaceHeaterModel):
    def __init__(self,
                bodyMass = None, 
                energySource = None, 
                heatTransferDimension = None, 
                heatTransferMedium = None, 
                numberOfPanels = None, 
                numberOfSections = None, 
                outputCapacity = None, 
                placementType = None, 
                temperatureClassification = None, 
                thermalEfficiency = None, 
                thermalMassHeatCapacity = None, 
                specificHeatCapacityWater = None, ###
                timeStep = None, ###
                save = False,
                **kwargs):
        super().__init__(**kwargs)

        self.bodyMass = bodyMass
        self.energySource = energySource
        self.heatTransferDimension = heatTransferDimension
        self.heatTransferMedium = heatTransferMedium
        self.numberOfPanels = numberOfPanels
        self.numberOfSections = numberOfSections
        self.outputCapacity = outputCapacity
        self.placementType = placementType
        self.temperatureClassification = temperatureClassification
        self.thermalEfficiency = thermalEfficiency
        self.thermalMassHeatCapacity = thermalMassHeatCapacity
        self.specificHeatCapacityWater = specificHeatCapacityWater ###
        self.heatTransferCoefficient = self.outputCapacity/(self.input["supplyWaterTemperature"]*0.8-self.output["radiatorOutletTemperature"]) ###
        self.timeStep = timeStep ###
        self.save = save ###




class Coil(EnergyConversionDevice, CoilModel.CoilModel):
    def __init__(self,
                airFlowRateMax = None,
                airFlowRateMin = None,
                nominalLatentCapacity = None,
                nominalSensibleCapacity = None,
                nominalUa = None,
                operationTemperatureMax = None,
                operationTemperatureMin = None,
                placementType = None,
                specificHeatCapacityAir = None,
                isHeatingCoil = None,
                isCoolingCoil = None,
                save = False,
                **kwargs):
        super().__init__(**kwargs)
        self.airFlowRateMax = airFlowRateMax
        self.airFlowRateMin = airFlowRateMin
        self.nominalLatentCapacity = nominalLatentCapacity
        self.nominalSensibleCapacity = nominalSensibleCapacity
        self.nominalUa = nominalUa
        self.operationTemperatureMax = operationTemperatureMax
        self.operationTemperatureMin = operationTemperatureMin
        self.placementType = placementType
        self.specificHeatCapacityAir = specificHeatCapacityAir ###
        self.isHeatingCoil = isHeatingCoil ###
        self.isCoolingCoil = isCoolingCoil ###
        self.save = save ###


class AirToAirHeatRecovery(EnergyConversionDevice, AirToAirHeatRecoveryModel.AirToAirHeatRecoveryModel):
    def __init__(self,
                hasDefrost = None,
                heatTransferTypeEnum = None,
                operationTemperatureMax = None,
                operationTemperatureMin = None,
                primaryAirFlowRateMax = None,
                primaryAirFlowRateMin = None,
                secondaryAirFlowRateMax = None,
                secondaryAirFlowRateMin = None,
                specificHeatCapacityAir = None,
                save = False,
                systemId = None,
                **kwargs):
        super().__init__(**kwargs)
        self.hasDefrost = hasDefrost
        self.heatTransferTypeEnum = heatTransferTypeEnum
        self.operationTemperatureMax = operationTemperatureMax
        self.operationTemperatureMin = operationTemperatureMin
        self.primaryAirFlowRateMax = primaryAirFlowRateMax
        self.primaryAirFlowRateMin = primaryAirFlowRateMin
        self.secondaryAirFlowRateMax = secondaryAirFlowRateMax
        self.secondaryAirFlowRateMin = secondaryAirFlowRateMin
        self.specificHeatCapacityAir = specificHeatCapacityAir ###
        self.eps_75_h = 0.8 ###
        self.eps_75_c = 0.8 ###
        self.eps_100_h = 0.8 ###
        self.eps_100_c = 0.8 ###
        self.save = save ###



        


class Fan(FlowMovingDevice, FanModel.FanModel):
    def __init__(self,
                capacityControlType = None,
                motorDriveType = None,
                nominalAirFlowRate = None,
                nominalPowerRate = None,
                nominalRotationSpeed = None,
                nominalStaticPressure = None,
                nominalTotalPressure = None,
                operationTemperatureMax = None,
                operationTemperatureMin = None,
                operationalRiterial = None,
                isSupplyFan = None,
                isReturnFan = None,
                save = False,
                **kwargs):
        super().__init__(**kwargs)
        self.capacityControlType = capacityControlType
        self.motorDriveType = motorDriveType
        self.nominalAirFlowRate = nominalAirFlowRate
        self.nominalPowerRate = nominalPowerRate
        self.nominalRotationSpeed = nominalRotationSpeed
        self.nominalStaticPressure = nominalStaticPressure
        self.nominalTotalPressure = nominalTotalPressure
        self.operationTemperatureMax = operationTemperatureMax
        self.operationTemperatureMin = operationTemperatureMin
        self.operationalRiterial = operationalRiterial
        self.c1 = 0 ###
        self.c2 = 0 ###
        self.c3 = -0.3 ###
        self.c4 = 1.3 ###
        self.isSupplyFan = isSupplyFan ###
        self.isReturnFan = isReturnFan ###
        self.save = save ###



class Damper(FlowController, DamperModel.DamperModel):
    def __init__(self,
                airFlowRateMax = None,
                bladeAction = None,
                bladeEdge = None,
                bladeShape = None,
                bladeThickness = None,
                closeOffRating = None,
                faceArea = None,
                frameDepth = None,
                frameThickness = None,
                frameType = None,
                leakageFullyClosed = None,
                nominalAirFlowRate = None,
                numberOfBlades = None,
                openPressureDrop = None,
                operation = None,
                operationTemperatureMax = None,
                operationTemperatureMin = None,
                orientation = None,
                temperatureRating = None,
                workingPressureMax = None,
                isSupplyDamper = None,
                isReturnDamper = None,
                save = False,
                **kwargs):
        super().__init__(**kwargs)
        self.airFlowRateMax = airFlowRateMax
        self.bladeAction = bladeAction
        self.bladeEdge = bladeEdge
        self.bladeShape = bladeShape
        self.bladeThickness = bladeThickness
        self.closeOffRating = closeOffRating
        self.faceArea = faceArea
        self.frameDepth = frameDepth
        self.frameThickness = frameThickness
        self.frameType = frameType
        self.leakageFullyClosed = leakageFullyClosed
        self.nominalAirFlowRate = nominalAirFlowRate
        self.numberOfBlades = numberOfBlades
        self.openPressureDrop = openPressureDrop
        self.operation = operation
        self.operationTemperatureMax = operationTemperatureMax
        self.operationTemperatureMin = operationTemperatureMin
        self.orientation = orientation
        self.temperatureRating = temperatureRating
        self.workingPressureMax = workingPressureMax
        self.a = 5 ###
        self.c = -self.a ###
        self.b = math.log((self.nominalAirFlowRate-self.c)/self.a) ###
        self.isSupplyDamper = isSupplyDamper###
        self.isReturnDamper = isReturnDamper ###
        self.save = save ###


        if self.isSupplyDamper:
            self.DamperSignalName = "supplyDamperSignal"
            self.AirFlowRateName = "supplyAirFlowRate" + str(self.systemId)
        elif self.isReturnDamper:
            self.DamperSignalName = "returnDamperSignal"
            self.AirFlowRateName = "returnAirFlowRate" + str(self.systemId)
        else:
            raise Exception("Damper is neither defined as supply or return. Set either \"isSupplyDamper\" or \"isReturnDamper\" to True")
    



class FlowMeter(FlowController, FlowMeterModel.FlowMeterModel):
    def __init__(self,
                readOutType = None,
                remoteReading = None,
                isSupplyFlowMeter = None,
                isReturnFlowMeter = None,
                **kwargs):
        super().__init__(**kwargs)
        self.readOutType = readOutType
        self.remoteReading = remoteReading
        self.isSupplyFlowMeter = isSupplyFlowMeter ###
        self.isReturnFlowMeter = isReturnFlowMeter ###
        
