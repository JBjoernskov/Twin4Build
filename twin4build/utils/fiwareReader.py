import warnings
import collections.abc
from twin4build.saref.commodity.commodity import Commodity
from twin4build.saref.device.meter.meter import Meter
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.function.metering_function.metering_function import MeteringFunction
from twin4build.saref.function.sensing_function.sensing_function import SensingFunction
from twin4build.saref.property_.property_ import Property
from twin4build.saref.measurement.measurement import Measurement
from twin4build.saref4bldg.building_space.building_space import BuildingSpace
from twin4build.saref4bldg.building_space.building_space_model import NoSpaceModelException
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.controller import Controller
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_device import DistributionDevice
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.air_to_air_heat_recovery.air_to_air_heat_recovery import AirToAirHeatRecovery
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil import Coil
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.damper.damper import Damper
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.valve.valve import Valve
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.fan.fan import Fan
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater import SpaceHeater
from twin4build.utils.configReader import configReader, fiwareConfig
import requests


class fiwareReader:

    def __init__(self):
        self.system_dict = {"ventilation": {},
                            "heating": {},
                            "cooling": {},
                            }
        self.component_base_dict = {}
        self.config = configReader().read_config()

    def get_fiware_access_token(self):
        data = dict(
            client_id=self.config.tokenClientId,
            client_secret=self.config.tokenSecret,
            grant_type='client_credentials',
            scope=self.config.scope
        )

        resp = requests.post(url=self.config.tokenUrl, data=data)
        data = resp.json()
        return data['access_token']

    def load_objects_from_fiware(self, accessToken: str, params: dict):

        headers = {'Authorization': 'Bearer ' + accessToken,
                   'Link': self.config.fiwareContextLink + ';rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
                   'Content-Type': 'application/json',
                   'NGSILD-Tenant': self.config.tenant,
                   'Accept': '*/*'}
        resp = requests.get(url=self.config.fiwareBaseUrl +
                            '/ngsi-ld/v1/entities', params=params, headers=headers)
        data = resp.json()
        return data

    def load_objects_of_type_from_fiware(self, accessToken: str, objectType: str):

        params = dict(
            type=objectType,
            options='keyValues'
        )

        return self.load_objects_from_fiware(accessToken, params)

    def get_device_systems(self, device, idMapping):

        result = []

        if isinstance(device["isSubSystemOf"], collections.abc.Sequence) and (not isinstance(device["isSubSystemOf"], str)):
            for row in device["isSubSystemOf"]:
                result.append(idMapping[row])
        else:
            result.append(idMapping[device["isSubSystemOf"]])

        return result

    def read_config_from_fiware(self):
        """
        Reads configuration from fiware and instantiates a base SAREF4BLDG object for each entry in the fiware.  
        """

        access_token = self.get_fiware_access_token()

        df_Systems = self.load_objects_of_type_from_fiware(
            access_token, "System")
        df_System_id_mapping = dict()
        # BECAUSE WHOLE BUILDING HAS BEEN IMPORTED NEEDS TO SELECT SPACE BY ID(NAME) TO AVOID TONS OF SPACES
        # THIS WILL CHANGE IN THE FUTURE
        df_Space = self.load_objects_from_fiware(access_token, dict(
            type="BuildingSpace", options='keyValues', q='name=="Ø20-601b-2"'))
        df_Space_id_mapping = dict()
        df_Damper = self.load_objects_of_type_from_fiware(
            access_token, "Damper")
        df_SpaceHeater = self.load_objects_of_type_from_fiware(
            access_token, "SpaceHeater")
        df_Valve = self.load_objects_of_type_from_fiware(access_token, "Valve")
        df_Coil = self.load_objects_of_type_from_fiware(access_token, "Coil")
        df_AirToAirHeatRecovery = self.load_objects_of_type_from_fiware(
            access_token, "AirToAirHeatRecovery")
        df_Fan = self.load_objects_of_type_from_fiware(access_token, "Fan")
        df_Controller = self.load_objects_of_type_from_fiware(
            access_token, "Controller")
        df_Sensor = self.load_objects_of_type_from_fiware(
            access_token, "Sensor")
        df_Meter = self.load_objects_of_type_from_fiware(access_token, "Meter")

        for fiwareSystem in df_Systems:

            df_System_id_mapping[fiwareSystem["id"]] = fiwareSystem["name"]
            if fiwareSystem["purpose"] == "ventilation":
                ventilation_system = DistributionDevice(
                    subSystemOf=[], hasSubSystem=[], id=fiwareSystem["name"])
                self.system_dict["ventilation"][fiwareSystem["name"]
                                                ] = ventilation_system

            elif fiwareSystem["purpose"] == "heating":
                heating_system = DistributionDevice(
                    subSystemOf=[], hasSubSystem=[], id=fiwareSystem["name"])
                self.system_dict["heating"][fiwareSystem["name"]
                                            ] = heating_system

            elif fiwareSystem["purpose"] == "cooling":
                cooling_system = DistributionDevice(
                    subSystemOf=[], hasSubSystem=[], id=fiwareSystem["name"])
                self.system_dict["cooling"][fiwareSystem["name"]
                                            ] = cooling_system

        for row in df_Space:
            space_name = row["name"]
            try:
                space = BuildingSpace(
                    airVolume=row["airVolume"],
                    contains=[],
                    connectedThrough=[],
                    connectsAt=[],
                    id=space_name)
                self.component_base_dict[space_name] = space

                df_Space_id_mapping[row["id"]] = space_name
            except NoSpaceModelException:
                print("No fitting space model for space " +
                      "\"" + space_name + "\"")
                print("Continuing...")

        for row in df_Damper:
            damper_name = row["name"]
            # Check that an appropriate space object exists
            if row["isContainedInBuildingSpace"] not in df_Space_id_mapping:
                warnings.warn(
                    "Cannot find a matching mathing BuildingSpace object for damper \"" + damper_name + "\"")
            else:
                systems = self.get_device_systems(row, df_System_id_mapping)
                systems = [system for system_dict in self.system_dict.values(
                ) for system in system_dict.values() if system.id in systems]
                damper = Damper(
                    subSystemOf=systems,
                    isContainedIn=self.component_base_dict[df_Space_id_mapping[row["isContainedInBuildingSpace"]]],
                    operationMode=row["operationMode"],
                    nominalAirFlowRate=Measurement(
                        hasValue=row["nominalAirFlowRate"]),
                    connectedThrough=[],
                    connectsAt=[],
                    id=damper_name)
                self.component_base_dict[damper_name] = damper

        for row in df_SpaceHeater:
            space_heater_name = row["name"]
            # Check that an appropriate space object exists
            if row["isContainedInBuildingSpace"] not in df_Space_id_mapping:
                warnings.warn(
                    "Cannot find a matching mathing BuildingSpace object for space heater \"" + space_heater_name + "\"")
            else:
                systems = self.get_device_systems(row, df_System_id_mapping)
                systems = [system for system_dict in self.system_dict.values(
                ) for system in system_dict.values() if system.id in systems]
                space_heater = SpaceHeater(
                    subSystemOf=systems,
                    isContainedIn=self.component_base_dict[df_Space_id_mapping[row["isContainedInBuildingSpace"]]],
                    outputCapacity=Measurement(hasValue=row["outputCapacity"]),
                    temperatureClassification=row["temperatureClassification"],
                    thermalMassHeatCapacity=Measurement(
                        hasValue=row["thermalMassHeatCapacity"]),
                    connectedThrough=[],
                    connectsAt=[],
                    id=space_heater_name)
                self.component_base_dict[space_heater_name] = space_heater

        for row in df_Valve:
            valve_name = row["name"]
            # Check that an appropriate space object exists
            if row["isContainedInBuildingSpace"] not in df_Space_id_mapping:
                warnings.warn(
                    "Cannot find a matching mathing BuildingSpace object for valve \"" + valve_name + "\"")
            else:
                systems = self.get_device_systems(row, df_System_id_mapping)
                systems = [system for system_dict in self.system_dict.values(
                ) for system in system_dict.values() if system.id in systems]
                valve = Valve(
                    subSystemOf=systems,
                    isContainedIn=self.component_base_dict[df_Space_id_mapping[row["isContainedInBuildingSpace"]]],
                    flowCoefficient=Measurement(
                        hasValue=row["flowCoefficient"]),
                    testPressure=Measurement(hasValue=row["testPressure"]),
                    connectedThrough=[],
                    connectsAt=[],
                    id=valve_name)
                self.component_base_dict[valve_name] = valve

        for row in df_Coil:
            coil_name = row["name"]
            systems = self.get_device_systems(row, df_System_id_mapping)
            systems = [system for system_dict in self.system_dict.values(
            ) for system in system_dict.values() if system.id in systems]
            coil = Coil(
                subSystemOf=systems,
                operationMode=row["operationMode"],
                connectedThrough=[],
                connectsAt=[],
                id=coil_name)
            self.component_base_dict[coil_name] = coil

        for row in df_AirToAirHeatRecovery:
            air_to_air_heat_recovery_name = row["name"]
            systems = self.get_device_systems(row, df_System_id_mapping)
            systems = [system for system_dict in self.system_dict.values(
            ) for system in system_dict.values() if system.id in systems]
            air_to_air_heat_recovery = AirToAirHeatRecovery(
                subSystemOf=systems,
                primaryAirFlowRateMax=Measurement(
                    hasValue=row["primaryAirFlowRateMax"]),
                secondaryAirFlowRateMax=Measurement(
                    hasValue=row["secondaryAirFlowRateMax"]),
                connectedThrough=[],
                connectsAt=[],
                id=air_to_air_heat_recovery_name)
            self.component_base_dict[air_to_air_heat_recovery_name] = air_to_air_heat_recovery

        for row in df_Fan:
            fan_name = row["name"]
            systems = self.get_device_systems(row, df_System_id_mapping)
            systems = [system for system_dict in self.system_dict.values(
            ) for system in system_dict.values() if system.id in systems]
            fan = Fan(
                subSystemOf=systems,
                operationMode=row["operationMode"],
                nominalAirFlowRate=Measurement(
                    hasValue=row["nominalAirFlowRate"]),
                nominalPowerRate=Measurement(hasValue=row["nominalPowerRate"]),
                connectedThrough=[],
                connectsAt=[],
                id=fan_name)
            self.component_base_dict[fan_name] = fan

        for row in df_Controller:
            controller_name = row["name"]
            if row["isContainedInBuildingSpace"] not in df_Space_id_mapping:
                warnings.warn(
                    "Cannot find a matching mathing BuildingSpace object for controller \"" + controller_name + "\"")
            else:
                systems = self.get_device_systems(row, df_System_id_mapping)
                systems = [system for system_dict in self.system_dict.values(
                ) for system in system_dict.values() if system.id in systems]
                controller = Controller(
                    subSystemOf=systems,
                    isContainedIn=self.component_base_dict[df_Space_id_mapping[row["isContainedInBuildingSpace"]]],
                    controllingProperty=row["controllingProperty"],
                    connectedThrough=[],
                    connectsAt=[],
                    id=controller_name)
                self.component_base_dict[controller_name] = controller

        for row in df_Sensor:
            sensor_name = row["name"]
            if row["isContainedInBuildingSpace"] not in df_Space_id_mapping:
                warnings.warn(
                    "Cannot find a matching mathing BuildingSpace object for sensor \"" + sensor_name + "\"")
            else:
                sensor = Sensor(
                    hasFunction=SensingFunction(
                        hasSensingRange=[
                            Measurement(hasValue=row["hasSensingRange"])
                        ]
                    ),
                    hasSensorType=row["hasSensorType"],
                    id=sensor_name
                )
                self.component_base_dict[sensor_name] = sensor

        for row in df_Meter:
            meter_name = row["name"]
            if row["isContainedInBuildingSpace"] not in df_Space_id_mapping:
                warnings.warn(
                    "Cannot find a matching mathing BuildingSpace object for meter \"" + meter_name + "\"")
            else:
                meter = Meter(
                    hasFunction=MeteringFunction(
                        hasMeterReading=[
                            Measurement(hasValue=row["hasMeterReading"])
                        ]
                    ),
                    hasMeterReadingType=row["hasMeterReadingType"],
                    id=meter_name
                )
                self.component_base_dict[meter_name] = meter
