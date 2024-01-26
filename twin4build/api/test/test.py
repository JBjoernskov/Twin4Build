
import unittest
import os , sys 
from datetime import datetime , timedelta
import requests
import pytz
import json
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
from twin4build.api.codes.ml_layer.request_timer import RequestTimer


class Test(unittest.TestCase):

    # Configuration function get read data from config.ini file
    def get_configuration(self):
        try:
            conf = ConfigReader()
            config_path = os.path.join(os.path.abspath(
                uppath(os.path.abspath(__file__), 3)), "config", "conf.ini")
            config = conf.read_config_section(config_path)
            
            return config
        except Exception as e:
            return None

    def setUp(self):
        self.config_path = os.path.join(os.path.abspath(
                  uppath(os.path.abspath(__file__), 3)), "config", "conf.ini")
        
        self.logfile_path = os.path.join(os.path.abspath(
            uppath(os.path.abspath(__file__),3)),"logger","logs")
        
        self.denmark_timezone = pytz.timezone('Europe/Copenhagen')
        self.current_time_denmark = datetime.now(self.denmark_timezone)
        self.time_format = '%Y-%m-%d %H:%M:%S%z'
        
        self.connector = db_connector()
        self.config = self.get_configuration()
        self.input_data = input_data()
        self.request_obj = request_class()
        self.validator = Validator()
        #self.request_timer = RequestTimer(self.request_obj)

        self.simulation_duration = int(self.config["simulation_variables"]["simulation_duration"])
        self.warmup_time = int(self.config["simulation_variables"]["warmup_time"])
        
        self.forecast_simulation_duration = int(self.config["forecast_simulation_variables"]["forecast_simulation_duration"])
        self.forecast_warmup_time = int(self.config["forecast_simulation_variables"]["warmup_time"])

        self.start_time = self.current_time_denmark - timedelta(hours=3)
        # start time now + 24 hours = 3 , 30
        self.end_time = self.start_time + timedelta(hours=self.forecast_simulation_duration) # simulation_duration = 24
        # start time - 12 hours = warm up  3 28
        self.total_start_time = self.start_time - timedelta(hours=self.forecast_warmup_time)

        #  # formatted start , end  time to '%Y-%m-%d %H:%M:%S' format 
        self.forecast_endtime = self.end_time.strftime(self.time_format)
        #forecast_startime = start_time.strftime(self.time_format)
        self.forecast_total_start_time = self.total_start_time.strftime(self.time_format)


        self.history_end_time = self.current_time_denmark - timedelta(hours=3)

        #start time = end time - simulation time ( 1 hour ) 
        self.history_start_time = self.history_end_time - timedelta(hours=self.simulation_duration)

        # total time (warmp up time start - 12 which will be the start for the input dict) = start time - warmup time
        self.history_total_start_time = self.history_start_time - timedelta(hours=self.warmup_time)

        # formatted total start (warmup) time to '%Y-%m-%d %H:%M:%S' format
        self.history_formatted_total_start_time = self.history_total_start_time.strftime(self.time_format)
        ## formatted end time to '%Y-%m-%d %H:%M:%S' format
        self.history_formatted_endtime = self.history_end_time.strftime(self.time_format)
        # formatted start time to '%Y-%m-%d %H:%M:%S' format
        self.history_formatted_startime = self.history_start_time.strftime(self.time_format)

    def test_python_version(self):
        self.assertEqual(sys.version_info.major, 3, "Python version is not 3")
        self.assertTrue(sys.version_info.minor >= 7, "Python version is not Greater than 7")

        
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
        table_name = ['ml_inputs','ml_inputs_dmi','ml_forecast_inputs_dmi']
        
        try:
            for table in table_name:
                self.connector.connect() 
                queried_data = self.connector.get_all_inputs(table)
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

    def __test_transform_list(self):
        # Test the transform_list method
        formatted_response_list_data =[
            {
               "time": "2023-12-12 03:13:52+0100",
                "outdoor_environment_outdoorTemperature": 25.0,
                "outdoor_environment_globalIrradiation": 1000.0,
                "OE20-601b-2_indoorTemperature": 22.5,
                "OE20-601b-2_indoorCo2Concentration": 500.0,
                "Supplydamper_airFlowRate": 50.0,
                "Supplydamper_damperPosition": 0.7,
                "Exhaustdamper_airFlowRate": 30.0,
                "Exhaustdamper_damperPosition": 0.5,
                "Spaceheater_outletWaterTemperature": 40.0,
                "Spaceheater_Power": 2000.0,
                "Spaceheater_Energy": 100.0,
                "Valve_waterFlowRate": 25.0,
                "Valve_valvePosition": 0.6,
                "Temperaturecontroller_inputSignal": 22.0,
                "CO2controller_inputSignal": 450.0,
                "temperaturesensor_indoorTemperature": 22.5,
                "Valvepositionsensor_valvePosition": 0.6,
                "Damperpositionsensor_damperPosition": 0.5,
                "CO2sensor_indoorCo2Concentration": 500.0,
                "Heatingmeter_Energy": 100.0,
                "Occupancyschedule_scheduleValue": 1,
                "Temperaturesetpointschedule_scheduleValue": 20.0,
                "Supplywatertemperatureschedule_scheduleValue": 60.0,
                "Supplyairtemperatureschedule_scheduleValue": 23.0
            }
        ]

        query_data = self.input_data.input_data_for_simulation(self.history_total_start_time,self.history_formatted_endtime,forecast=False)
   
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

    def _test_data_insertion(self):
        
        url = 'http://127.0.0.1:8070/simulate'
        start_time = str(datetime(2023,11,1,21,14,7))
        end_time = str(datetime(2023,11,2,4,14,7))

        query_data = self.input_data.input_data_for_simulation(start_time,end_time,forecast=False)
        response = requests.post(url,json=query_data)

        model_output_data = response.json()
        model_output_data = self.request_obj.extract_actual_simulation(model_output_data,start_time,end_time)
        formatted_response_list_data = self.request_obj.convert_response_to_list(response_dict=model_output_data)
        input_list_data = self.input_data.transform_list(formatted_response_list_data)
        self.connector.add_data('ml_simulation_results',inputs=input_list_data)
        
    
    def test_simulation_api_connect(self):
        url = 'http://127.0.0.1:8070/simulate'

        #2023-11-01 21:14:07 2023-11-02 04:14:07


        query_data = self.input_data.input_data_for_simulation(self.history_formatted_total_start_time,self.history_formatted_endtime,forecast=False)
        response = requests.post(url,json=query_data)

        model_output_data = response.json()
        self.assertEqual(self.validator.validate_input_data(query_data,forecast=False),True)   
        self.assertEqual(self.validator.validate_response_data(model_output_data),True)
        self.assertEqual(response.status_code,200) 
        #self.assertEqual(sorted(response_json.keys()), sorted(expected_response.keys()))


    # test for forecast 
        
    def test_input_data_for_simulation(self):

        query_data = self.input_data.input_data_for_simulation(self.forecast_total_start_time,self.forecast_endtime,forecast=True)
        
        # check for ml_forecast_inputs_dmi key in inputs_sensor
        self.assertIn("ml_forecast_inputs_dmi",query_data['inputs_sensor'])

        self.assertIsNotNone(query_data)
        self.assertIsInstance(query_data,dict)
        self.assertIn("metadata",query_data)
        self.assertIn("inputs_sensor",query_data)
        self.assertIn("input_schedules",query_data)    

    # database read values 
    def test_get_forecast_inputs(self):

        try:
            self.connector.connect() 
            queried_data = self.connector.get_filtered_forecast_inputs(
                table_name="ml_forecast_inputs_dmi",
                start_time=self.forecast_total_start_time,
                end_time=self.forecast_endtime
            )
            self.assertGreater(len(queried_data), 0)
        except Exception as e:
            self.fail(f"Failed to retrieve data from the database: {e}")
        finally:
            self.connector.disconnect() 

    

    

        










if __name__ == '__main__':
    unittest.main()
    #print("All the test cases have been passed , proceed further ...")


'''
test for forecast

# database read values 
# keys 
# forecast values 

# insertion 
# reading forecast 

# dmi values 
# dmi files 
# dmi files delete 




'''