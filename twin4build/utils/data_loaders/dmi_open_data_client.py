from datetime import datetime
import os
from dmi_open_data import DMIOpenDataClient, Parameter, ClimateDataParameter
from dmi_open_data.client import _construct_datetime_argument
from typing import List, Dict, Optional, Any, Union
import requests

def get_api_key():
    """
    Requires that the system variable DMI_API_KEY is set.
    """
    return "369bf1d5-6d8f-49aa-b3bb-368857de207a"#os.getenv('DMI_API_KEY')

class DMIOpenDataClient(DMIOpenDataClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        print(f"{self.base_url(api=api)}/{service}")
        print()
        print(res)
        data = res.json()
        http_status_code = data.get("http_status_code", 200)
        if http_status_code != 200:
            message = data.get("message")
            raise ValueError(
                f"Failed HTTP request with HTTP status code {http_status_code} and message: {message}"
            )
        return res.json()
    
    def get_forecast(self,
                    limit: Optional[int] = 10000,
                    offset: Optional[int] = 0,
                    modelRun: Optional[datetime] = None,
                    from_time: Optional[datetime] = None,
                    to_time: Optional[datetime] = None,
                    south_west_point: Optional[str] = None,
                    north_east_point: Optional[str] = None,
                ) -> List[Dict[str, Any]]:
        """Get DMI forecast.

        Args:
            parameter_id (Optional[Parameter], optional): Returns observations for a specific parameter.
                Defaults to None.
            station_id (Optional[int], optional): Search for a specific station using the stationID.
                Defaults to None.
            from_time (Optional[datetime], optional): Returns only objects with a "timeObserved" equal
                to or after a given timestamp. Defaults to None.
            to_time (Optional[datetime], optional): Returns only objects with a "timeObserved" before
                (not including) a given timestamp. Defaults to None.
            limit (Optional[int], optional): Specify a maximum number of observations
                you want to be returned. Defaults to 10000.
            offset (Optional[int], optional): Specify the number of observations that should be skipped
                before returning matching objects. Defaults to 0.

        Returns:
            List[Dict[str, Any]]: List of raw DMI observations.
        """
        res = self._query(
                api="forecastdata",
                service="collections/harmonie_nea_sf/items",
                params={
                    # "parameterId": None if parameter is None else parameter.value,
                    "limit": limit,
                    "offset": offset,
                    "modelRun": _construct_datetime_argument(from_time=modelRun),
                    "datetime": _construct_datetime_argument(
                        from_time=from_time, to_time=to_time
                    ),
                    "bbox": f"{south_west_point},{north_east_point}",

                },
            )
        # print(res)
        return res.get("features", [])


def test():
    # Get 10 stations
    client = DMIOpenDataClient(api_key=get_api_key(), version="v2")
    client.version = "v1"
    # stations = client.get_stations(limit=10)

    south_west_point = "10.421584,55.365306"
    north_east_point = "10.433023,55.372196"


    # Get temperature observations from DMI station in given time period
    observations = client.get_forecast(modelRun=datetime(2023, 10, 31, 0),
                                        from_time=datetime(2023, 10, 31, 0),
                                        to_time=datetime(2023, 10, 31, 16),
                                        south_west_point=south_west_point,
                                        north_east_point=north_east_point)
    import json
    print(json.dumps(observations, indent=2))
    print(observations)
if __name__=="__main__":
    test()