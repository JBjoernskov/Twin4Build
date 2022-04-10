import Saref4Build
import Saref4Syst
import WeatherStation
from dateutil.tz import tzutc
import datetime
import matplotlib.pyplot as plt


class BuildingEnergyModel:
    def __init__(self,
                startPeriod = None,
                endPeriod = None):
        self.startPeriod = startPeriod
        self.endPeriod = endPeriod

    def add_connection(self, from_obj, to_obj, connection_type):
        from_obj_connection = Saref4Syst.Connection(connectsSystem = from_obj, connectionType = connection_type)
        from_obj.connectedThrough.append(from_obj_connection)
        to_obj_connection_point = Saref4Syst.ConnectionPoint(connectionPointOf = to_obj, connectsSystemThrough = from_obj_connection)
        from_obj_connection.connectsSystemAt = to_obj_connection_point
        to_obj.connectsAt.append(to_obj_connection_point)
    
    def load_model(self):

        hvac_system = Saref4Build.DistributionDevice(subSystemOf = [], hasSubSystem = [])
        heating_system = Saref4Build.DistributionDevice(subSystemOf = [hvac_system], hasSubSystem = [])
        ventilation_system = Saref4Build.DistributionDevice(subSystemOf = [hvac_system], hasSubSystem = [])
        cooling_system = Saref4Build.DistributionDevice(subSystemOf = [hvac_system], hasSubSystem = [])

        ventilation_system_dict = {"VentilationSystem1": Saref4Build.DistributionDevice(subSystemOf = [hvac_system])}
        heating_system_dict = {"HeatingSystem1": Saref4Build.DistributionDevice(subSystemOf = [hvac_system])}
        cooling_system_dict = {"CoolingSystem1": Saref4Build.DistributionDevice(subSystemOf = [hvac_system])}

        
        

        ### One per 

        weather_station = WeatherStation.WeatherStation(startPeriod = self.startPeriod,
                                                        endPeriod = self.endPeriod,
                                                        input = {},
                                                        output = {},
                                                        savedInput = {},
                                                        savedOutput = {},
                                                        createReport = True,
                                                        connectedThrough = [],
                                                        connectsAt = [])

        air_to_air_heat_recovery = Saref4Build.AirToAirHeatRecovery(primaryAirFlowRateMax = 1,
                                                                    secondaryAirFlowRateMax = 1,
                                                                    specificHeatCapacityAir = 1000,
                                                                    subSystemOf = [ventilation_system],
                                                                    input = {},
                                                                    output = {},
                                                                    savedInput = {},
                                                                    savedOutput = {},
                                                                    createReport = True,
                                                                    connectedThrough = [],
                                                                    connectsAt = [])
        ventilation_system.hasSubSystem.append(air_to_air_heat_recovery)

        heating_coil = Saref4Build.Coil(isHeatingCoil = True,
                                        specificHeatCapacityAir = 1000,
                                        subSystemOf = [heating_system, ventilation_system],
                                        input = {"supplyAirTemperatureSetpoint": 23},
                                        output = {},
                                        savedInput = {},
                                        savedOutput = {},
                                        createReport = True,
                                        connectedThrough = [],
                                        connectsAt = [])
        heating_system.hasSubSystem.append(heating_coil)
        ventilation_system.hasSubSystem.append(heating_coil)

        cooling_coil = Saref4Build.Coil(isCoolingCoil = True,
                                        specificHeatCapacityAir = 1000,
                                        subSystemOf = [cooling_system, ventilation_system],
                                        input = {"supplyAirTemperatureSetpoint": 23},
                                        output = {},
                                        savedInput = {},
                                        savedOutput = {},
                                        connectedThrough = [],
                                        connectsAt = [])
        heating_system.hasSubSystem.append(cooling_coil)
        ventilation_system.hasSubSystem.append(cooling_coil)

        supply_fan = Saref4Build.Fan(isSupplyFan = True,
                                    nominalAirFlowRate = 25000/3600*1.225,
                                    nominalPowerRate = 10000,
                                    subSystemOf = [ventilation_system],
                                    input = {},
                                    output = {},
                                    savedInput = {},
                                    savedOutput = {},
                                    connectedThrough = [],
                                    connectsAt = [])
        ventilation_system.hasSubSystem.append(supply_fan)

        return_fan = Saref4Build.Fan(isReturnFan = True,
                                    nominalAirFlowRate = 25000/3600*1.225,
                                    nominalPowerRate = 10000,
                                    subSystemOf = [ventilation_system],
                                    input = {},
                                    output = {},
                                    savedInput = {},
                                    savedOutput = {},
                                    connectedThrough = [],
                                    connectsAt = [])
        ventilation_system.hasSubSystem.append(return_fan)

        supply_flowmeter = Saref4Build.FlowMeter(isSupplyFlowMeter = True, 
                                                subSystemOf = [ventilation_system],
                                                input = {},
                                                output = {},
                                                connectedThrough = [],
                                    connectsAt = [])
        ventilation_system.hasSubSystem.append(supply_flowmeter)

        return_flowmeter = Saref4Build.FlowMeter(isReturnFlowMeter = True, 
                                                subSystemOf = [ventilation_system],
                                                input = {},
                                                output = {},
                                                savedInput = {},
                                                savedOutput = {},
                                                connectedThrough = [],
                                                connectsAt = [])
        ventilation_system.hasSubSystem.append(return_flowmeter)


        self.initComponents = []
        for i in range(100):
            space_heater = Saref4Build.SpaceHeater(outputCapacity = 1000,
                                                    thermalMassHeatCapacity = 30000,
                                                    specificHeatCapacityWater = 4180,
                                                    timeStep = 600, 
                                                    subSystemOf = [heating_system],
                                                    input = {"supplyWaterTemperature": 60},
                                                    output = {"radiatorOutletTemperature": 22},
                                                    savedInput = {},
                                                    savedOutput = {},
                                                    connectedThrough = [],
                                                    connectsAt = [])
            heating_system.hasSubSystem.append(space_heater)

            valve = Saref4Build.Valve(waterFlowRateMax = 0.1,
                                        valveAuthority = 0.8, 
                                        subSystemOf = [heating_system],
                                        input = {},
                                        output = {},
                                        savedInput = {},
                                        savedOutput = {},
                                        connectedThrough = [],
                                        connectsAt = [])
            heating_system.hasSubSystem.append(valve)

            temperature_controller = Saref4Build.Controller(isTemperatureController = True,
                                                            subSystemOf = [heating_system],
                                                            input = {},
                                                            output = {"valveSignal": 0},
                                                            savedInput = {},
                                                            savedOutput = {},
                                                            connectedThrough = [],
                                                            connectsAt = [])
            heating_system.hasSubSystem.append(temperature_controller)

            co2_controller = Saref4Build.Controller(isCo2Controller = True, 
                                                    subSystemOf = [ventilation_system],
                                                    input = {},
                                                    output = {"supplyDamperSignal": 0,
                                                            "returnDamperSignal": 0},
                                                    savedInput = {},
                                                    savedOutput = {},
                                                    connectedThrough = [],
                                                    connectsAt = [])
            ventilation_system.hasSubSystem.append(co2_controller)

            supply_damper = Saref4Build.Damper(isSupplyDamper = True,
                                                nominalAirFlowRate = 250/3600*1.225,
                                                subSystemOf = [ventilation_system],
                                                input = {},
                                                output = {},
                                                savedInput = {},
                                                savedOutput = {},
                                                connectedThrough = [],
                                                connectsAt = [])
            ventilation_system.hasSubSystem.append(supply_damper)

            return_damper = Saref4Build.Damper(isReturnDamper = True,
                                                nominalAirFlowRate = 250/3600*1.225,
                                                subSystemOf = [ventilation_system],
                                                input = {},
                                                output = {},
                                                savedInput = {},
                                                savedOutput = {},
                                                connectedThrough = [],
                                                connectsAt = [])
            ventilation_system.hasSubSystem.append(return_damper)

            space = Saref4Build.BuildingSpace(input = {},
                                                output = {"indoorTemperature": 22},
                                                savedInput = {},
                                                savedOutput = {},
                                                connectedThrough = [],
                                                connectsAt = [])



            

            self.add_connection(space, temperature_controller, "indoorTemperature")
            self.add_connection(space, co2_controller, "co2Concentration")

            self.add_connection(supply_damper, supply_flowmeter, supply_damper.AirFlowRateName)
            self.add_connection(return_damper, return_flowmeter, return_damper.AirFlowRateName)
        
            self.add_connection(space, space_heater, "indoorTemperature")
            self.add_connection(valve, space_heater, "waterFlowRate")

            self.add_connection(temperature_controller, valve, "valveSignal")
            self.add_connection(temperature_controller, space, "valveSignal")

            self.add_connection(co2_controller, space, "supplyDamperSignal")
            self.add_connection(co2_controller, space, "returnDamperSignal")

            self.add_connection(co2_controller, supply_damper, "supplyDamperSignal")
            self.add_connection(co2_controller, return_damper, "returnDamperSignal")

            self.initComponents.append(space)

        self.add_connection(weather_station, air_to_air_heat_recovery, "outdoorTemperature")

        self.add_connection(space, air_to_air_heat_recovery, "indoorTemperature") #########################
        self.add_connection(supply_flowmeter, air_to_air_heat_recovery, "supplyAirFlowRate")
        self.add_connection(return_flowmeter, air_to_air_heat_recovery, "returnAirFlowRate")
        
        self.add_connection(air_to_air_heat_recovery, heating_coil, "supplyAirTemperature")
        self.add_connection(air_to_air_heat_recovery, cooling_coil, "supplyAirTemperature")

        self.add_connection(supply_flowmeter, heating_coil, "supplyAirFlowRate")
        self.add_connection(supply_flowmeter, cooling_coil, "supplyAirFlowRate")

        self.add_connection(supply_flowmeter, supply_fan, "supplyAirFlowRate")
        self.add_connection(return_flowmeter, return_fan, "returnAirFlowRate")

        self.initComponents.append(weather_station)
        self.activeComponents = self.initComponents



    def simulate(self):
        
        self.find_path()

        dt = 10
        time = self.startPeriod
        time_list = []
        while time < self.endPeriod:
            for component in self.component_order:
                # print(component.__class__.__name__)
                #Gather all needed inputs for the component through all ingoing connections
                for connection_point in component.connectsAt:
                    connection = connection_point.connectsSystemThrough
                    connected_component = connection.connectsSystem
                    component.input[connection.connectionType] = connected_component.output[connection.connectionType] 

                component.update_output()
                component.update_report()

            time_list.append(time)
            time += datetime.timedelta(minutes=dt)
            


        print("-------")
        for component in self.component_order:
            if component.createReport:
                component.plot_report(time_list)
        plt.show()









        
    def find_path(self):
        self.component_order = []
        self.component_order.extend(self.initComponents)
        while len(self.activeComponents)>0:
            print("YYYYYYYYYYYY")
            self.traverse()


    def traverse(self):
        activeComponentsNew = []
        for component in self.activeComponents:
            for connection in component.connectedThrough:
                connection_point = connection.connectsSystemAt
                connected_component = connection_point.connectionPointOf
                if connected_component.connectionVisits is None:
                    connected_component.connectionVisits = [connection.connectionType]
                else:
                    connected_component.connectionVisits.append(connection.connectionType)

                has_connections = True
                for ingoing_connection_point in connected_component.connectsAt:
                    ingoing_connection = ingoing_connection_point.connectsSystemThrough
                    if ingoing_connection.connectionType not in connected_component.connectionVisits or isinstance(connected_component, Saref4Build.BuildingSpace):
                        has_connections = False
                        break
                
                if has_connections:
                    self.component_order.append(connected_component)

                    if connected_component.connectedThrough is not None:
                        activeComponentsNew.append(connected_component)

                # print("---")
                # print(component.__class__.__name__)
                # print(connected_component.__class__.__name__)
                # print(connection.connectionType)

                # print(connected_component.connectionVisits)

        

        self.activeComponents = activeComponentsNew



    def get_leaf_subsystems(self, system):
        for sub_system in system.hasSubSystem:
            if sub_system.hasSubSystem is None:
                self.leaf_subsystems.append(sub_system)
            else:
                self.get_leaf_subsystems(sub_system)




startPeriod = datetime.datetime(year=2018, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
endPeriod = datetime.datetime(year=2018, month=1, day=30, hour=0, minute=0, second=0, tzinfo=tzutc())
model = BuildingEnergyModel(startPeriod = startPeriod,
                            endPeriod = endPeriod)
model.load_model()
model.simulate()



