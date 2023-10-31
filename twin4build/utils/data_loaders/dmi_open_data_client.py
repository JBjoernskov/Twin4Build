from datetime import datetime
import os
from dmi_open_data import DMIOpenDataClient, Parameter, ClimateDataParameter


class DMIOpenDataClient(DMIOpenDataClient):
    def __init__(self):
        super(self).__init__()
    
    def get_forecast(self):
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
                service="collections/observation/items",
                params={
                    "parameterId": None if parameter is None else parameter.value,
                    "stationId": station_id,
                    "datetime": _construct_datetime_argument(
                        from_time=from_time, to_time=to_time
                    ),
                    "limit": limit,
                    "offset": offset,
                },
            )
