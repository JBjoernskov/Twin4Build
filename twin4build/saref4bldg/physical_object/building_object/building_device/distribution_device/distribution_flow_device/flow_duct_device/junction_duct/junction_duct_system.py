import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_duct_device.junction_duct.junction_duct as junction_duct

class JunctionDuctSystem(junction_duct.JunctionDuct):
    def __init__(self,
                airFlowRateBias = None,
                **kwargs):
        super().__init__(**kwargs)
        if airFlowRateBias is not None:
            self.airFlowRateBias = airFlowRateBias
        else:
            self.airFlowRateBias = 0
        self._config = {"parameters": ["airFlowRateBias"]}

    @property
    def config(self):
        return self._config

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None,
                    model=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output["totalAirFlowRate"] = sum(v for k, v in self.input.items() if "airFlowRate" in k) + self.airFlowRateBias

