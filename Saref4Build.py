import saref4syst.Saref4Syst as Saref4Syst

import ControllerModel
import ValveModel
import SpaceHeaterModel
import CoilModel
import AirToAirHeatRecoveryModel
import FanModel
import DamperModel
import FlowMeterModel
import BuildingSpaceModel

import math


class PhysicalObject(Saref4Syst.System):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        


class BuildingSpace(BuildingSpaceModel.BuildingSpaceModel):
    def __init__(self,
                hasSpace = None,
                isSpaceOf = None,
                contains = None,
                **kwargs):
        super().__init__(**kwargs)
        self.hasSpace = hasSpace
        self.isSpaceOf = isSpaceOf
        self.contains = contains

        

    

        





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
                **kwargs):
        super().__init__(**kwargs)
        

        


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
                **kwargs):
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
        super().__init__(**kwargs)

        





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
                **kwargs):
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
        super().__init__(**kwargs)

    



class FlowMeter(FlowController, FlowMeterModel.FlowMeterModel):
    def __init__(self,
                readOutType = None,
                remoteReading = None,
                **kwargs):
        super().__init__(**kwargs)
        self.readOutType = readOutType
        self.remoteReading = remoteReading
        
