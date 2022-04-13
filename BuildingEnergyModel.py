import Saref4Build
import Saref4Syst
import WeatherStation
import Schedule


from dateutil.tz import tzutc
import datetime
import matplotlib.pyplot as plt

from SpaceDataCollection import SpaceDataCollection


class BuildingEnergyModel:
    def __init__(self,
                timeStep = None,
                startPeriod = None,
                endPeriod = None):
        self.timeStep = timeStep
        self.startPeriod = startPeriod
        self.endPeriod = endPeriod

    def add_connection(self, from_obj, to_obj, from_connection_name, to_connection_name):
        from_obj_connection = Saref4Syst.Connection(connectsSystem = from_obj, fromConnectionName = from_connection_name, toConnectionName = to_connection_name)
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

        

        weather_station = WeatherStation.WeatherStation(startPeriod = self.startPeriod,
                                                        endPeriod = self.endPeriod,
                                                        input = {},
                                                        output = {},
                                                        savedInput = {},
                                                        savedOutput = {},
                                                        createReport = True,
                                                        connectedThrough = [],
                                                        connectsAt = [])

        occupancy_schedule = Schedule.Schedule(startPeriod = self.startPeriod,
                                                timeStep = self.timeStep,
                                                rulesetDict = {
                                                    "ruleset_default_value": 0,
                                                    "ruleset_start_minute": [0,0,0,0,0],
                                                    "ruleset_end_minute": [0,0,0,0,0],
                                                    "ruleset_start_hour": [0,5,8,12,18],
                                                    "ruleset_end_hour": [6,8,12,18,22],
                                                    "ruleset_value": [0,35,35,35,0]},
                                                input = {},
                                                output = {},
                                                savedInput = {},
                                                savedOutput = {},
                                                createReport = False,
                                                connectedThrough = [],
                                                connectsAt = [])
        temperature_setpoint_schedule = Schedule.Schedule(startPeriod = self.startPeriod,
                                                timeStep = self.timeStep,
                                                rulesetDict = {
                                                    "ruleset_default_value": 20,
                                                    "ruleset_start_minute": [0,0],
                                                    "ruleset_end_minute": [0,0],
                                                    "ruleset_start_hour": [0,6],
                                                    "ruleset_end_hour": [6,18],
                                                    "ruleset_value": [20,22.5]},
                                                input = {},
                                                output = {},
                                                savedInput = {},
                                                savedOutput = {},
                                                createReport = False,
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
                                        createReport = True,
                                        connectedThrough = [],
                                        connectsAt = [])
        heating_system.hasSubSystem.append(cooling_coil)
        ventilation_system.hasSubSystem.append(cooling_coil)

        supply_fan = Saref4Build.Fan(isSupplyFan = True,
                                    nominalAirFlowRate = 250/3600*1.225,
                                    nominalPowerRate = 10000,
                                    subSystemOf = [ventilation_system],
                                    input = {},
                                    output = {},
                                    savedInput = {},
                                    savedOutput = {},
                                    createReport = True,
                                    connectedThrough = [],
                                    connectsAt = [])
        ventilation_system.hasSubSystem.append(supply_fan)

        return_fan = Saref4Build.Fan(isReturnFan = True,
                                    nominalAirFlowRate = 250/3600*1.225,
                                    nominalPowerRate = 10000,
                                    subSystemOf = [ventilation_system],
                                    input = {},
                                    output = {},
                                    savedInput = {},
                                    savedOutput = {},
                                    createReport = True,
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
        self.initComponents.append(temperature_setpoint_schedule)
        self.initComponents.append(weather_station)
        self.initComponents.append(occupancy_schedule)
        for i in range(1):
            space_heater = Saref4Build.SpaceHeater(outputCapacity = 1000,
                                                    thermalMassHeatCapacity = 30000,
                                                    specificHeatCapacityWater = 4180,
                                                    timeStep = self.timeStep, 
                                                    subSystemOf = [heating_system],
                                                    input = {"supplyWaterTemperature": 60},
                                                    output = {"radiatorOutletTemperature": 22,
                                                                "Energy": 0},
                                                    savedInput = {},
                                                    savedOutput = {},
                                                    createReport = True,
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
                                        createReport = True,
                                        connectedThrough = [],
                                        connectsAt = [])
            heating_system.hasSubSystem.append(valve)

            temperature_controller = Saref4Build.Controller(isTemperatureController = True,
                                                            k_p = 8,
                                                            k_i = 0,
                                                            k_d = 0,
                                                            subSystemOf = [heating_system],
                                                            input = {},
                                                            output = {"valveSignal": 0},
                                                            savedInput = {},
                                                            savedOutput = {},
                                                            connectedThrough = [],
                                                            connectsAt = [])
            heating_system.hasSubSystem.append(temperature_controller)

            co2_controller = Saref4Build.Controller(isCo2Controller = True, 
                                                    k_p = 0.01,
                                                    k_i = 0,
                                                    k_d = 0,
                                                    subSystemOf = [ventilation_system],
                                                    input = {"indoorCo2ConcentrationSetpoint": 600},
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
                                                createReport = True,
                                                connectedThrough = [],
                                                connectsAt = [])
            ventilation_system.hasSubSystem.append(supply_damper)
            supply_damper.output[supply_damper.AirFlowRateName] = 0 ########################

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
            return_damper.output[return_damper.AirFlowRateName] = 0 ########################

            space = Saref4Build.BuildingSpace(densityAir = 1.225,
                                                airVolume = 50,
                                                timeStep = self.timeStep,
                                                input = {"generationCo2Concentration": 0.06,
                                                        "outdoorCo2Concentration": 500,
                                                        "shadesSignal": 0},
                                                output = {"indoorTemperature": 21.5,
                                                        "indoorCo2Concentration": 500},
                                                savedInput = {},
                                                savedOutput = {},
                                                createReport = True,
                                                connectedThrough = [],
                                                connectsAt = [])



            
            self.add_connection(temperature_setpoint_schedule, temperature_controller, "scheduleValue", "indoorTemperatureSetpoint")
            self.add_connection(space, temperature_controller, "indoorTemperature", "indoorTemperature")
            self.add_connection(space, co2_controller, "indoorCo2Concentration", "indoorCo2Concentration")

            self.add_connection(supply_damper, supply_flowmeter, supply_damper.AirFlowRateName, supply_damper.AirFlowRateName)
            self.add_connection(return_damper, return_flowmeter, return_damper.AirFlowRateName, return_damper.AirFlowRateName)
        
            self.add_connection(space, space_heater, "indoorTemperature", "indoorTemperature")
            self.add_connection(valve, space_heater, "waterFlowRate", "waterFlowRate")

            self.add_connection(temperature_controller, valve, "valveSignal", "valveSignal")
            self.add_connection(temperature_controller, space, "valveSignal", "valveSignal")

            self.add_connection(co2_controller, space, "supplyDamperSignal", "supplyDamperSignal")
            self.add_connection(co2_controller, space, "returnDamperSignal", "returnDamperSignal")

            self.add_connection(supply_damper, space, supply_damper.AirFlowRateName, "supplyAirFlowRate")
            self.add_connection(return_damper, space, return_damper.AirFlowRateName, "returnAirFlowRate")
            self.add_connection(occupancy_schedule, space, "scheduleValue", "numberOfPeople")

            self.add_connection(weather_station, space, "directRadiation", "directRadiation")
            self.add_connection(weather_station, space, "diffuseRadiation", "diffuseRadiation")
            self.add_connection(weather_station, space, "outdoorTemperature", "outdoorTemperature")


            self.add_connection(co2_controller, supply_damper, "supplyDamperSignal", "supplyDamperSignal")
            self.add_connection(co2_controller, return_damper, "returnDamperSignal", "returnDamperSignal")

            self.initComponents.append(space)

        self.add_connection(weather_station, air_to_air_heat_recovery, "outdoorTemperature", "outdoorTemperature")

        self.add_connection(space, air_to_air_heat_recovery, "indoorTemperature", "indoorTemperature") #########################
        self.add_connection(supply_flowmeter, air_to_air_heat_recovery, "supplyAirFlowRate", "supplyAirFlowRate")
        self.add_connection(return_flowmeter, air_to_air_heat_recovery, "returnAirFlowRate", "returnAirFlowRate")
        
        self.add_connection(air_to_air_heat_recovery, heating_coil, "supplyAirTemperature", "supplyAirTemperature")
        self.add_connection(air_to_air_heat_recovery, cooling_coil, "supplyAirTemperature", "supplyAirTemperature")

        self.add_connection(supply_flowmeter, heating_coil, "supplyAirFlowRate", "supplyAirFlowRate")
        self.add_connection(supply_flowmeter, cooling_coil, "supplyAirFlowRate", "supplyAirFlowRate")

        self.add_connection(supply_flowmeter, supply_fan, "supplyAirFlowRate", "supplyAirFlowRate")
        self.add_connection(return_flowmeter, return_fan, "returnAirFlowRate", "returnAirFlowRate")

        
        self.activeComponents = self.initComponents



    def simulate(self):
        
        self.find_path()

        dt = 10
        time = self.startPeriod
        time_list = []
        while time < self.endPeriod:
            for component in self.component_order:
                # print("----")
                # print(component.__class__.__name__)
                #Gather all needed inputs for the component through all ingoing connections
                for connection_point in component.connectsAt:
                    connection = connection_point.connectsSystemThrough
                    connected_component = connection.connectsSystem
                    # print("--h--")
                    # print(connected_component.__class__.__name__)
                    # print(connection.toConnectionName)
                    # print(connection.fromConnectionName)
                    # print(connected_component.output)
                    component.input[connection.toConnectionName] = connected_component.output[connection.fromConnectionName]

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
                    connected_component.connectionVisits = [connection.toConnectionName]
                else:
                    connected_component.connectionVisits.append(connection.toConnectionName)

                has_connections = True
                for ingoing_connection_point in connected_component.connectsAt:
                    ingoing_connection = ingoing_connection_point.connectsSystemThrough
                    if ingoing_connection.toConnectionName not in connected_component.connectionVisits or isinstance(connected_component, Saref4Build.BuildingSpace):
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



timeStep = 600
startPeriod = datetime.datetime(year=2019, month=1, day=29, hour=0, minute=0, second=0, tzinfo=tzutc())
endPeriod = datetime.datetime(year=2019, month=1, day=30, hour=0, minute=0, second=0, tzinfo=tzutc())
model = BuildingEnergyModel(timeStep = timeStep,
                            startPeriod = startPeriod,
                            endPeriod = endPeriod)
model.load_model()
model.simulate()



