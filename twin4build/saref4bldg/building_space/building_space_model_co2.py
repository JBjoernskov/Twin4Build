import twin4build.saref4bldg.building_space.building_space as building_space
class BuildingSpaceModelCo2(building_space.BuildingSpace):
    
    def __init__(self,
                densityAir=None,
                airVolume=None,
                stepSize=None,
                **kwargs):
        super().__init__(**kwargs)

        self.densityAir = densityAir ###
        self.airVolume = airVolume ###
        self.airMass = self.airVolume*self.densityAir ###
        self.stepSize = stepSize ###

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        pass

    def do_step(self):
        M_air = 28.9647 #g/mol
        M_CO2 = 44.01 #g/mol
        K_conversion = M_CO2/M_air*1e-6
        self.output["indoorCo2Concentration"] = (self.airMass*self.output["indoorCo2Concentration"] + 
                                                self.input["outdoorCo2Concentration"]*(self.input["supplyAirFlowRate"] + self.input["infiltration"])*self.stepSize + 
                                                self.input["generationCo2Concentration"]*self.input["numberOfPeople"]*self.stepSize/K_conversion)/(self.airMass + (self.input["returnAirFlowRate"]+self.input["infiltration"])*self.stepSize)