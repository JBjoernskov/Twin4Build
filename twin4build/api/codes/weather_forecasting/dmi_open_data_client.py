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
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)
from twin4build.utils.uppath import uppath


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
        print(res)
        n_timesteps = len(res["features"])
        folder_path = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "grib_files")
        output_dict = {"time": []}
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
            if keep_grib_file==False:
                os.remove(weather_file_name)
        return output_dict

    def download(self, url, file_name):
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



def test():
    api_key = "369bf1d5-6d8f-49aa-b3bb-368857de207a"
    client = DMIOpenDataClient(api_key=api_key, version="v1")
    reference_coordinate = (55.365306, 10.421584) #Coordinate at SDU, Odense, Denmark

    parameters = {"11": [2],
                  "117": [0]}
    
    # Get forecast from DMI weather model in given time period
    forecast = client.get_forecast(modelRun=datetime.datetime(2023, 11, 21, 6),
                                        from_time=datetime.datetime(2023, 11, 21, 6),
                                        to_time=datetime.datetime(2023, 11, 21, 6)+datetime.timedelta(hours=54),#54
                                        reference_coordinate=reference_coordinate,
                                        parameters=parameters,
                                        keep_grib_file=True) # If the grib file is kept, subsequent runs will be much faster
    
    # 0, 3, 6, 9, 12, 15, 18, 21

    df = pd.DataFrame(forecast).set_index("time")
    df["Temperature"] = df["11"]-273.15 #Convert from Kelvin to Celcius
    df["globalIrradiation"] = -df["117"].diff(periods=-1)/3600 #Convert from h*J/m2 to W/m2
    df.drop(columns=["11", "117"], inplace=True)

    df.to_csv("DMI_forecast_sample.csv")

    print(df)

    import matplotlib.pyplot as plt
    df.plot(subplots=True, sharex=True)
    plt.show()
    
if __name__=="__main__":
    test()