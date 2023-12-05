import datetime
import os
from dmi_open_data import DMIOpenDataClient
from dmi_open_data.client import _construct_datetime_argument
from typing import List, Dict, Optional, Any, Union, Tuple
import requests
import pygrib
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import sys
import pandas as pd
import pytz
import time 
import schedule
import xarray as xr

if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)

from twin4build.utils.uppath import uppath
from twin4build.api.codes.database.db_data_handler import db_connector


class DMIOpenDataClient(DMIOpenDataClient):
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)

    def __init__(self, api_key: str, version: str = "v2"):
        self.api_key = api_key
        self.version = version

    def base_url(self, api: str):
        if api not in ("climateData", "metObs", "forecastdata"): #Extended to "forecastdata"
            raise NotImplementedError(f"Following api is not supported yet: {api}")
        return self._base_url.format(version=self.version, api=api)
    
    def _query(self, api: str, service: str, params: Dict[str, Any], **kwargs):
        res = requests.get(
            url=f"{self.base_url(api=api)}/{service}",
            params={
                "api-key": self.api_key,
                **params,
            },
            **kwargs,
        )
        data = res.json()
        http_status_code = data.get("http_status_code", 200)
        if http_status_code != 200:
            message = data.get("message")
            raise ValueError(
                f"Failed HTTP request with HTTP status code {http_status_code} and message: {message}"
            )
        return res.json()
    
    def get_distance(self, point1, point2):
        R = 6371
        dlat = radians(point2[0]-point1[0])
        dlon = radians(point2[1]-point1[1])
        a = sin(dlat / 2)**2 + cos(point1[0]) * cos(point2[0]) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance
    
    
    def get_forecast(self,
                    modelRun: Optional[datetime.datetime] = None,
                    from_time: Optional[datetime.datetime] = None,
                    to_time: Optional[datetime.datetime] = None,
                    reference_coordinate: Optional[Tuple] = None,
                    parameters: Optional[dict] = None,
                    keep_grib_file: Optional[bool] = True
                ) -> List[Dict[str, Any]]:
        """Get DMI forecast.
        """
        allowable_hours = [0, 3, 6, 9, 12, 15, 18, 21]
        dmi_model_horizon = 54 #hours
        dmi_max_kept_hours = 48
        assert modelRun <= from_time, "The argument \"from_time\" must be later or equal to \"modelRun\""
        assert modelRun.hour in [0, 3, 6, 9, 12, 15, 18, 21] and modelRun.minute==0 and modelRun.second==0 and modelRun.microsecond==0, f"The modelRun argument must be a datetime object with one of the following whole hours {', '.join(str(x) for x in allowable_hours)}"
        assert to_time <= modelRun + datetime.timedelta(hours=dmi_model_horizon), f"The argument \"to_time\" must be before or equal to {dmi_model_horizon} hours after \"modelRun\""
        assert modelRun>=datetime.datetime.now()-datetime.timedelta(hours=dmi_max_kept_hours), f"The argument \"modelRun\" must be later or equal to {dmi_max_kept_hours} hours before current time"
        #Get the download urls for the DMI model forecasts
        res = self._query(
                api="forecastdata",
                service="collections/harmonie_nea_sf/items",
                params={
                    # "parameterId": None if parameter is None else parameter.value,
                    "modelRun": _construct_datetime_argument(from_time=modelRun),
                    "datetime": _construct_datetime_argument(
                        from_time=from_time, to_time=to_time
                    ),
                },
            )
        
        n_timesteps = len(res["features"])
        folder_path = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "grib_files")
        output_dict = {
            "time": []
        }
        for parameter in parameters:
            output_dict[parameter] = []

        tol = 1e-8
        saved_coordinate = None
        for i in range(n_timesteps):
            res_filename = res["features"][i]["id"]
            res_url = res["features"][i]["asset"]["data"]["href"]
            weather_file_name = os.path.join(folder_path, res_filename)
            timestep_string = res["features"][i]["properties"]["datetime"]
            timestep = datetime.datetime.strptime(timestep_string, '%Y-%m-%dT%H:%M:%SZ')
            output_dict["time"].append(timestep)
            print("=========================================================")
            print(f"Getting forecast for timestep \"{timestep}\"")
            print("=========================================================")

            #Download the DMI model forecasts from the res_url and save in weather_file_name
            self.download(res_url, weather_file_name)
            grbs = pygrib.open(weather_file_name)
            for grb in grbs:
                
                if grb.parameterName in parameters and grb.level in parameters[grb.parameterName] and grb.levelType=="sfc":
                    print(f"Extracting forecast data for parameter \"{grb.parameterName}\" level \"{grb.level}\"")
                    values = np.array(grb.values)
                    values = values.reshape((values.shape[0]*values.shape[1]))
                    lats, lons = grb.latlons()
                    lats = np.array(lats).reshape((lats.shape[0]*lats.shape[1]))
                    lons = np.array(lons).reshape((lons.shape[0]*lons.shape[1]))
                    idx = self.find_closest(lats, lons, reference_coordinate)
                    closest_coordinate = (lats[idx], lons[idx])
                    output_dict[grb.parameterName].append(values[idx])
                    if saved_coordinate is None or (closest_coordinate[0]-saved_coordinate[0]>tol and closest_coordinate[1]-saved_coordinate[1]>tol):
                        print(f"Distance between reference coordinate and closest coordinate is: {self.get_distance(reference_coordinate, closest_coordinate):.2f} KM")
                        saved_coordinate = closest_coordinate
                        print("saved coridnate" , saved_coordinate)
            if keep_grib_file==False: #  The process cannot access the file because it is being used by another process - error
                os.remove(weather_file_name)
        return output_dict , saved_coordinate

    def download(self, url, file_name):

        print(url,file_name)
        # open in binary mode
        if os.path.isfile(file_name)==False or os.stat(file_name).st_size==0:
            print(f"Downloading...")
            with open(file_name, "wb") as file:
                # get request
                response = requests.get(url)
                # write to file
                file.write(response.content)

    def find_closest(self, lats, lons, coordinate):
        coordinate = np.array([coordinate[0], coordinate[1]]).reshape((1,2))
        coordinates = np.zeros((lats.shape[0],2))
        coordinates[:,0] = lats
        coordinates[:,1] = lons
        dist = np.linalg.norm(coordinates-coordinate, axis=1)
        idx = np.argmin(dist)
        return idx

def read_grib_files():
    
    folder_path = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "grib_files")

    filename = "HARMONIE_NEA_SF_2023-11-22T090000Z_2023-11-22T120000Z.grib"

    grib_file = os.path.join(folder_path, filename)

    grb = pygrib.open(grib_file)

    for data in grb:
        values = np.array(data.values)

        lats, lons = data.latlons()
        lats = np.array(lats).reshape((lats.shape[0]*lats.shape[1]))
        lons = np.array(lons).reshape((lons.shape[0]*lons.shape[1]))

        df= pd.DataFrame(values)

        df.to_csv("grib_file_data.csv")

def insert_to_db(inputs):
    table_name = 'ml_forecast_inputs_dmi'

    connector = db_connector()
    connector.connect()
    # if the time extist then update else add .. 

    for entry in inputs:
        forecast_time = entry["forecast_time"]

        existing_data = connector.get_data_using_forecast(forecast_time)

        if existing_data:

            updated_values_dict = {
                'latitude': entry['latitude'],
                'longitude':  entry['longitude'],
                'radia_glob':  entry['radia_glob'],
                'temp_dry':  entry['temp_dry'],
                'stationid': entry['stationid'],
            } 

            connector.update_forecast_data(forecast_time,updated_values_dict)

        else:
            connector.add_data(table_name,inputs=inputs)




def test():
    api_key = "369bf1d5-6d8f-49aa-b3bb-368857de207a"
    client = DMIOpenDataClient(api_key=api_key, version="v1")
    reference_coordinate = (55.365306, 10.421584) #Coordinate at SDU, Odense, Denmark

    parameters = {"11": [2],
                  "117": [0]}
    
    def get_forecast():
        
        denmark_timezone = pytz.timezone('Europe/Copenhagen')

        #current_time = datetime.datetime.now(tz=denmark_timezone)
        
        #print(current_time)

        #just for testing 
        current_time  = datetime.datetime(2023, 11, 23, 6)

        formatted_current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # Convert the formatted current time back to a datetime object
        formatted_current_time_datetime = datetime.datetime.strptime(formatted_current_time, '%Y-%m-%d %H:%M:%S')

        near_hours = formatted_current_time_datetime.hour

        near_hours = near_hours-(near_hours%3)

        print("Nearest time is :",near_hours)

        model_run_datetime = formatted_current_time_datetime.replace(minute=0, second=0, microsecond=0,hour=near_hours)

        print(model_run_datetime)

        #print(formatted_current_time,'::',model_run_datetime,'::',datetime.datetime(2023, 11, 6, 6))
        # 2023-11-22 08:03:03 :: 2023-11-22 08:00:00 :: 2023-11-06 06:00:00

        # Get forecast from DMI weather model in given time period
        forecast, saved_coordinate = client.get_forecast(modelRun=model_run_datetime,
                                            from_time=formatted_current_time_datetime,
                                            to_time=formatted_current_time_datetime+datetime.timedelta(hours=48),#54
                                            reference_coordinate=reference_coordinate,
                                            parameters=parameters,
                                            keep_grib_file=True) # If the grib file is kept, subsequent runs will be much faster
        
        # 0, 3, 6, 9, 12, 15, 18, 21

        df = pd.DataFrame(forecast).set_index("time")
        df["Temperature"] = df["11"]-273.15 #Convert from Kelvin to Celcius
        df["globalIrradiation"] = -df["117"].diff(periods=-1)/3600 #Convert from h*J/m2 to W/m2

        data_list = df.apply(lambda row: {
            "forecast_time": row.name.strftime('%Y-%m-%d %H:%M:%S'),
            "latitude": saved_coordinate[0],  # Replace with the actual latitude value
            "longitude": saved_coordinate[1],  # Replace with the actual longitude value
            "radia_glob": row["globalIrradiation"],
            "temp_dry": row["Temperature"],
          
            "stationid": 0  # Replace with the actual stationid value
        }, axis=1)

        if not data_list.empty:
            insert_to_db(data_list)

    duration = 3 

    get_forecast()
    # Schedule subsequent function calls at 3-hour intervals
    sleep_interval = duration * 60 * 60  # 3 hours in seconds

    # Create a schedule job that runs the request_simulator function every 2 hours
    job = schedule.every(sleep_interval).seconds.do(get_forecast)

    while True:
        try:
            schedule.run_pending()
            print("Function called at:", time.strftime("%Y-%m-%d %H:%M:%S"))
            time.sleep(sleep_interval)

        except Exception as schedule_error:
            print(schedule_error)
            schedule.cancel_job(job)
            break

    #import matplotlib.pyplot as plt
    #df.plot(subplots=True, sharex=True)
    #plt.show()

    #read_grib_files()


if __name__=="__main__":
    test()