"""
Fetching data from Quantum Leap api 
Using keycloak to get the token 
"""


from keycloak_token import KeycloakAuthenticator
import requests
import sys
import os 
from datetime import datetime,  timezone

###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)

from twin4build.config.Config import ConfigReader
from twin4build.logger.Logging import Logging

logger = Logging.get_logger("API_logfile")

class APIClient:
    def __init__(self) -> None:
        logger.info("APICLient Initialise Function")
        self.get_configuration()

        # Object for keycloak token class
        self.keycloak_token = KeycloakAuthenticator(server_url=self.server_url,client_id=self.client_id,realm_name=self.realm_name)
        #authenticating with the keycloack
        self.keycloak_token.authenticate(user=self.user,password=self.password)

    def create_url(self,room,sensor):
        return f"https://quantumleap.dev.twin4build.kmd.dk/v2/entities/{room}/attrs/{sensor}"


    def get_configuration(self):
        '''
            using ConfigReader to read configuration from config.ini file 
        '''
        try: 
            conf=ConfigReader()
            config_path = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "config", "conf.ini")
            self.config=conf.read_config_section(config_path)

            self.server_url = self.config['keycloak_cred']['server_url']
            self.realm_name = self.config['keycloak_cred']['realm_name']
            self.client_id = self.config['keycloak_cred']['client_id']
        
            self.user = self.config['keycloak_cred']['user']
            self.password = self.config['keycloak_cred']['password']

            logger.info("APICLient : Data has been read from the configuration file")

        except Exception as e :
            logger.error("Error reading the configuration file Exception : ".e)

    # creating headers for quantum leap API
    def create_ql_headers(self):
        logger.info("Creating headers for QL")
        headers = {
            "Accept": "*/*",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "NGSILD-Tenant": "sdu",
            "NGSILD-Path": "/GetSensorData",
        }
        return headers
    
    # Checking for token expiration , getting refresh token and fetching QL API
    def get_ql_response(self,api_url):
        """
            Using request to get response from the QL API
        """
        logger.info("Getting QL response, refresh token check")
        if self.keycloak_token.is_access_token_expired():
            if self.keycloak_token.is_refresh_token_expired():
                self.keycloak_token.authenticate(user=self.user, password=self.password)
            else:
                self.keycloak_token.refresh_token()

        headers = self.create_ql_headers()
        headers["Authorization"] = "Bearer " + self.keycloak_token.get_access_token()
        response = requests.get(api_url, headers=headers,verify=False)
        return response
    
    # getting the response  for QL API
    def get_ql_sensor_data(self,api_url):
        """
            Fetching all the sensor data from QL API
        """
        try:
            response = self.get_ql_response(api_url=api_url)
            response.raise_for_status()

            if response.status_code == 401:
                response = self.get_ql_response(api_url=api_url)

            if response.status_code == 200:
                return response.json()
            
            else:
                logger.info(f"Request failed with status code: {response.status_code}")
                print(f"Request failed with status code: {response.status_code}")
                return None
            
        except requests.exceptions.RequestException as e:
            logger.error("Error Got On Fetching Data:", e)
            print("Error Got On Fetching Data:", e)
            return None
        
    def get_all_data(self,room,sensor):
        """
            Using this code we are creating the url of QL and then we are going to call get_ql_sensor_data
        """
        api_url = self.create_url(room=room,sensor=sensor)
        return self.get_ql_sensor_data(api_url)
    

    def convert_to_custom_format(self,input_date_time):
        try:
            # Parse the input date/time string
            input_datetime = datetime.strptime(input_date_time, '%Y-%m-%d %H:%M:%S')
            
            # Convert to UTC timezone
            input_datetime_utc = input_datetime.replace(tzinfo=timezone.utc)
            
            # Format the datetime object in the desired format
            formatted_datetime = input_datetime_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
            return formatted_datetime
        except ValueError:
            return "Invalid input format. Please provide a date/time in the format 'YYYY-MM-DD HH:MM:SS'."
    

    def get_data_using_time_filter(self,room,sensor,start_time,end_time):
        """
            Getting all the data from QL based on time filters 
        """
        start_time = self.convert_to_custom_format(start_time)
        end_time = self.convert_to_custom_format(end_time)

        api_url = self.create_url(room=room,sensor=sensor) + f"?fromDate=2023-07-27T07:13:20.000Z&toDate=2023-07-28T07:13:20.000Z"
        return self.get_ql_sensor_data(api_url)
    
    def get_latest_n_data(self,room,sensor, n_value):

        """
            Using this code we are creating the url to fetch n latest values 
            of QL and then we are going to call get_ql_sensor_data            
        """
        api_url = self.create_url(room=room,sensor=sensor)

        n_data_api_url = api_url + f"/value?lastN={n_value}"

        return self.get_ql_sensor_data(n_data_api_url)

def main():
    room = "urn:ngsi-ld:Sensor:O20-601b-2"
    sensors_names = ["damper"]
    sensors_data = {}

    api_client = APIClient()
    
    for sensor in sensors_names:
    #   sensors_data[sensor] = api_client.get_all_data(room=room,sensor=sensor)
    #   print(api_client.get_latest_n_data(room=room,sensor=sensor,n_value=10))
    # 
    # #2023-07-27T07:13:20.000Z   2023-07-28T07:13:20.000Z' \
  
        start_time = "2023-07-27 07:13:20"
        end_time = "2023-07-28 07:13:20"

        data = api_client.get_data_using_time_filter(room=room,sensor=sensor,start_time=start_time,end_time=end_time)


        #print(api_client.get_data_using_time_filter(room=room,sensor=sensor,start_time=start_time,end_time=end_time)) 
           
    #print(all_sensors_data)

""""if __name__ == "__main__":
    main()"""
    