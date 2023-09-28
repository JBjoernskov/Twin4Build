
import unittest
import os , sys 
from datetime import datetime
import requests

# Only for testing before distributing package
if __name__ == '__main__':
    # Define a function to move up in the directory hierarchy
    def uppath(_path, n): return os.sep.join(_path.split(os.sep)[:-n])
    # Calculate the file path using the uppath function
    file_path = uppath(os.path.abspath(__file__), 4)
    # Append the calculated file path to the system path
    sys.path.append(file_path)


from twin4build.api.codes.database.db_data_handler import db_connector
from twin4build.config.Config import ConfigReader
from twin4build.api.codes.ml_layer.input_data import input_data
from twin4build.api.codes.ml_layer.validator import Validator
from twin4build.api.codes.ml_layer.request_to_api import request_class

class Test(unittest.TestCase):

    # Configuration function get read data from config.ini file
    def get_configuration(self):
        try:
            conf = ConfigReader()
            config_path = os.path.join(os.path.abspath(
                uppath(os.path.abspath(__file__), 4)), "config", "conf.ini")
            config = conf.read_config_section(config_path)
            
            return config
        except Exception as e:
            return None

    def setUp(self):
        self.config_path = os.path.join(os.path.abspath(
                  uppath(os.path.abspath(__file__), 3)), "config", "conf.ini")
        
        self.logfile_path = os.path.join(os.path.abspath(
            uppath(os.path.abspath(__file__),3)),"logger","logs")
        
        self.connector = db_connector()
        self.config = self.get_configuration()
        self.input_data = input_data()
        self.request_obj = request_class()
        self.validator = Validator()

    def test_python_version(self):
        self.assertEqual(sys.version_info.major, 3, "Python version is not 3")
        
    def test_config_file(self):
        self.assertTrue(os.path.isfile(self.config_path), "Config file not found at %s" %self.config_path)
    '''    
    # We need to check if we require this function or not 
    def test_logfiles_generation(self):
        self.assertTrue(os.path.exists(self.logfile_path),"Logs Folder not created")
        self.assertTrue(os.path.isdir(self.logfile_path),"Logs Directory Not Found")

        log_files = [f for f in os.listdir(self.logfile_path) if f.endswith(".log")]
        self.assertTrue(log_files, "No log files found in the 'logs' directory.")
    '''

    def test_database_connect(self):
        self.connector.connect()
        self.assertIsNotNone(self.connector.engine)
        
    @classmethod
    def setUpClass(self):
        self.connector = db_connector()
        self.connector.connect()

    def test_get_all_inputs(self):
        table_name = 'ml_inputs'
        
        try:
            self.connector.connect() 
            queried_data = self.connector.get_all_inputs(table_name)
            
            self.assertIsNotNone(queried_data)
            self.assertTrue(isinstance(queried_data, list))
            self.assertGreater(len(queried_data), 0)
        except Exception as e:
            self.fail(f"Failed to retrieve data from the database: {e}")
        finally:
            self.connector.disconnect() 

    def test_get_latest_inputs(self):
        table_name = 'ml_inputs'
        roomname = 'O20-601b-2'
        
        try:
            self.connector.connect() 
            queried_data = self.connector.get_latest_values(table_name,roomname)
            
            self.assertIsNotNone(queried_data)
        except Exception as e:
            self.fail(f"Failed to retrieve data from the database: {e}")
        finally:
            self.connector.disconnect() 

    @classmethod
    def tearDownClass(self):
        self.connector.disconnect()
    
    def test_get_filter_columns(self):
        table_name = 'ml_inputs'
        columns = self.input_data.get_filter_columns(table_name)
        expected_columns = ['opcuats', 'co2concentration', 'damper', 'shadingposition', 'temperature']
        self.assertEqual(columns, expected_columns)

    def test_input_data_for_simulation(self):
        start_time = datetime(2023,8,17,8,50,0)
        end_time = datetime(2023,8,22,10,40,0)

        query_data = self.input_data.input_data_for_simulation(start_time,end_time)
        
        self.assertIsNotNone(query_data)
        self.assertIsInstance(query_data,dict)
        self.assertIn("metadata",query_data)
        self.assertIn("inputs_sensor",query_data)
        self.assertIn("input_schedules",query_data)    

    def test_transform_list(self):
        # Test the transform_list method
        formatted_response_list_data = [
            {
                'time': '2023-08-17 08:50:00',
                'Outdoorenvironment_outdoorTemperature': 25.0,
                'Outdoorenvironment_globalIrradiation': 1000.0,
                'OE20-601b-2_indoorTemperature': 22.5,
                'OE20-601b-2_indoorCo2Concentration': 500.0,
                'Supplydamper_airFlowRate': 50.0,
                'Supplydamper_damperPosition': 0.7,
                'Exhaustdamper_airFlowRate': 30.0,
                'Exhaustdamper_damperPosition': 0.5,
                'Spaceheater_outletWaterTemperature': 40.0,
                'Spaceheater_Power': 2000.0,
                'Spaceheater_Energy': 100.0,
                'Valve_waterFlowRate': 25.0,
                'Valve_valvePosition': 0.6,
                'Temperaturecontroller_inputSignal': 22.0,
                'CO2controller_inputSignal': 450.0,
                'temperaturesensor_indoorTemperature': 22.5,
                'Valvepositionsensor_valvePosition': 0.6,
                'Damperpositionsensor_damperPosition': 0.5,
                'CO2sensor_indoorCo2Concentration': 500.0,
                'Heatingmeter_Energy': 100.0,
                'Occupancyschedule_scheduleValue': 1,
                'Temperaturesetpointschedule_scheduleValue': 20.0,
                'Supplywatertemperatureschedule_supplyWaterTemperatureSetpoint': 60.0,
                'Supplyairtemperatureschedule_scheduleValue': 23.0,
            },
            # Add more data points as needed
        ]

        start_time = datetime(2023,8,17,8,50,0)
        end_time = datetime(2023,8,22,10,40,0)

        query_data = self.input_data.input_data_for_simulation(start_time,end_time)
        
        transformed_data = self.input_data.transform_list(formatted_response_list_data)

        # Assert that the transformed data has the expected structure
        self.assertIsNotNone(transformed_data)
        self.assertTrue(isinstance(transformed_data, list))
        self.assertGreater(len(transformed_data), 0)

        # Check that the transformed data has the expected keys
        expected_keys = [
            'simulation_time',
            'outdoorenvironment_outdoortemperature',
            'outdoorenvironment_globalirradiation',
            'indoortemperature',
            'indoorco2concentration',
            'supplydamper_airflowrate',
            'supplydamper_damperposition',
            'exhaustdamper_airflowrate',
            'exhaustdamper_damperposition',
            'spaceheater_outletwatertemperature',
            'spaceheater_power',
            'spaceheater_energy',
            'valve_waterflowrate',
            'valve_valveposition',
            'temperaturecontroller_inputsignal',
            'co2controller_inputsignal',
            'temperaturesensor_indoortemperature',
            'valvepositionsensor_valveposition',
            'damperpositionsensor_damperposition',
            'co2sensor_indoorco2concentration',
            'heatingmeter_energy',
            'occupancyschedule_schedulevalue',
            'temperaturesetpointschedule_schedulevalue',
            'supplywatertemperatureschedule_supplywatertemperaturesetpoint',
            'ventilationsystem_supplyairtemperatureschedule_schedulevaluet',
            'input_start_datetime',
            'input_end_datetime',
            'spacename',
        ]

        for data_point in transformed_data:
            self.assertEqual(sorted(data_point.keys()), sorted(expected_keys))

        # Add more specific assertions as needed

    def test_simulation_api_connect(self):
        url = 'http://localhost:8005/simulate'

        start_time = str(datetime(2023,8,17,8,50,0))
        end_time = str(datetime(2023,8,22,10,40,0))

        query_data = self.input_data.input_data_for_simulation(start_time,end_time)
        
        response = requests.post(url,json=query_data)
        response_json = response.json()

        expected_response = {
                'time': ['2023-08-17 08:50:00'],
                'Outdoorenvironment_outdoorTemperature': [25.0],
                'Outdoorenvironment_globalIrradiation': [1000.0],
                'OE20-601b-2_indoorTemperature': [22.5],
                'OE20-601b-2_indoorCo2Concentration': [500.0],
                'Supplydamper_airFlowRate': [50.0],
                'Supplydamper_damperPosition': [0.7],
                'Exhaustdamper_airFlowRate': [30.0],
                'Exhaustdamper_damperPosition': [0.5],
                'Spaceheater_outletWaterTemperature': [40.0],
                'Spaceheater_Power': [2000.0],
                'Spaceheater_Energy': [100.0],
                'Valve_waterFlowRate': [25.0],
                'Valve_valvePosition': [0.6],
                'Temperaturecontroller_inputSignal': [22.0],
                'CO2controller_inputSignal': [450.0],
                'temperaturesensor_indoorTemperature': [22.5],
                'Valvepositionsensor_valvePosition': [0.6],
                'Damperpositionsensor_damperPosition': [0.5],
                'CO2sensor_indoorCo2Concentration': [500.0],
                'Heatingmeter_Energy': [100.0],
                'Occupancyschedule_scheduleValue': [1],
                'Temperaturesetpointschedule_scheduleValue': [20.0],
                'Supplywatertemperatureschedule_supplyWaterTemperatureSetpoint': [60.0],
                'Supplyairtemperatureschedule_scheduleValue': [23.0],
            }
        

        self.assertEqual(response.status_code,200)    
        self.assertEqual(sorted(response_json.keys()), sorted(expected_response.keys()))







if __name__ == '__main__':
    unittest.main()
    #print("All the test cases have been passed , proceed further ...")
