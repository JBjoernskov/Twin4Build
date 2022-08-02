from ..flow_moving_device import FlowMovingDevice
class Fan(FlowMovingDevice):
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