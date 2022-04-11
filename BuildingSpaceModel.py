import Saref4Syst


class BuildingSpaceModel(Saref4Syst.System):
    def __init__(self, 
                **kwargs):
        super().__init__(**kwargs)

    def update_output(self):
        self.output["indoorTemperature"] = 23
        self.output["indoorCo2Concentration"] = self.output["indoorCo2Concentration"] + (self.input["outdoorCo2Concentration"]*self.input["supplyAirFlowRate"] - self.output["indoorCo2Concentration"]*self.input["returnAirFlowRate"] + self.input["numberOfPeople"]*self.input["generationCo2Concentration"])*self.timeStep/self.airMass