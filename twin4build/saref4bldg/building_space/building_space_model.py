from .building_space import BuildingSpace
import os
import torch
import pickle
import datetime
from twin4build.utils.space_data_collection import SpaceDataCollection

class BuildingSpaceModel(BuildingSpace):
    def __init__(self,
                densityAir = None,
                airVolume = None,
                startPeriod = None,
                timeStep = None, 
                **kwargs):
        super().__init__(**kwargs)

        self.densityAir = densityAir ###
        self.airVolume = airVolume ###
        self.airMass = self.airVolume*self.densityAir ###
        self.time = startPeriod ###
        self.timeStep = timeStep ###
        

        # saved_space_list = ["Ø20-603-0", "Ø22-605a-2", "Ø22-603b-2"] #classroom 
        # saved_space_alias_list = ["Classroom", "Office 1", "Office 2"]
        space_name = "Ø20-603-0"
        self.model = self.get_model(space_name)


        
        save_folder = "C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python"
        save_filename = save_folder + "/saved_building_data_collection_dict_notime" + ".pickle"
        filehandler = open(save_filename, 'rb') 
        building_data_collection_dict = pickle.load(filehandler)


        space_data_collection = building_data_collection_dict[space_name]
        self.sw_radiation_idx = list(space_data_collection.clean_data_dict.keys()).index("sw_radiation")
        self.lw_radiation_idx = list(space_data_collection.clean_data_dict.keys()).index("lw_radiation")
        self.OAT_idx = list(space_data_collection.clean_data_dict.keys()).index("OAT")
        self.temperature_idx = list(space_data_collection.clean_data_dict.keys()).index("temperature")
        self.CO2_idx = list(space_data_collection.clean_data_dict.keys()).index("CO2")
        self.r_valve_idx = list(space_data_collection.clean_data_dict.keys()).index("r_valve")
        self.v_valve_idx = list(space_data_collection.clean_data_dict.keys()).index("v_valve")
        self.shades_idx = list(space_data_collection.clean_data_dict.keys()).index("shades")

        # self.day_of_year_cos_idx = list(space_data_collection.clean_data_dict.keys()).index("day_of_year_cos")
        # self.day_of_year_sin_idx = list(space_data_collection.clean_data_dict.keys()).index("day_of_year_sin")
        # self.hour_of_day_cos_idx = list(space_data_collection.clean_data_dict.keys()).index("hour_of_day_cos")
        # self.hour_of_day_sin_idx = list(space_data_collection.clean_data_dict.keys()).index("hour_of_day_sin")


        self.sw_radiation_min = space_data_collection.data_min_vec[self.sw_radiation_idx]
        self.sw_radiation_max = space_data_collection.data_max_vec[self.sw_radiation_idx]
        self.lw_radiation_min = space_data_collection.data_min_vec[self.lw_radiation_idx]
        self.lw_radiation_max = space_data_collection.data_max_vec[self.lw_radiation_idx]
        self.OAT_min = space_data_collection.data_min_vec[self.OAT_idx]
        self.OAT_max = space_data_collection.data_max_vec[self.OAT_idx]
        self.temperature_min = space_data_collection.data_min_vec[self.temperature_idx]
        self.temperature_max = space_data_collection.data_max_vec[self.temperature_idx]
        self.CO2_min = space_data_collection.data_min_vec[self.CO2_idx]
        self.CO2_max = space_data_collection.data_max_vec[self.CO2_idx]
        self.r_valve_min = space_data_collection.data_min_vec[self.r_valve_idx]
        self.r_valve_max = space_data_collection.data_max_vec[self.r_valve_idx]
        self.v_valve_min = space_data_collection.data_min_vec[self.v_valve_idx]
        self.v_valve_max = space_data_collection.data_max_vec[self.v_valve_idx]
        self.shades_min = space_data_collection.data_min_vec[self.shades_idx]
        self.shades_max = space_data_collection.data_max_vec[self.shades_idx]


        self.n_input = space_data_collection.data_matrix.shape[1]


        self.first_time_step = True

        n_layers = 1
        n_neurons = 20
        h_0_input = torch.zeros((n_layers,1,n_neurons)).cpu()
        c_0_input = torch.zeros((n_layers,1,n_neurons)).cpu()

        h_0_output = torch.zeros((1,1,1)).cpu()
        c_0_output = torch.zeros((1,1,1)).cpu()


        self.hidden_state = ((h_0_input,c_0_input), (h_0_output,c_0_output))


    def rescale(self,y,y_min,y_max,low,high):
        y = (y-low)/(high-low)*(y_max-y_min) + y_min
        return y

    def min_max_norm(self,y,y_min,y_max,low,high):
        y = (y-y_min)/(y_max-y_min)*(high-low) + low
        return y


    def get_model(self, space_name):
        search_path = "C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/OU44_space_models/rooms_no_time_600k_20n_test_all"
        directory = os.fsencode(search_path)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.find(space_name.replace("Ø", "OE") + "_Network") != -1:
                break

        full_path = search_path + "/" + filename
        model = torch.jit.load(full_path)
        return model

    def get_temperature(self):
        NN_input = torch.zeros((1,1,self.n_input))

        NN_input[0,0,self.sw_radiation_idx] = self.min_max_norm(self.input["directRadiation"], self.sw_radiation_min, self.sw_radiation_max, -1, 1)
        NN_input[0,0,self.lw_radiation_idx] = self.min_max_norm(self.input["diffuseRadiation"], self.lw_radiation_min, self.lw_radiation_max, -1, 1)
        NN_input[0,0,self.OAT_idx] = self.min_max_norm(self.input["outdoorTemperature"], self.OAT_min, self.OAT_max, -1, 1)
        NN_input[0,0,self.temperature_idx] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, -1, 1)
        NN_input[0,0,self.CO2_idx] = self.min_max_norm(self.output["indoorCo2Concentration"], self.CO2_min, self.CO2_max, -1, 1)
        NN_input[0,0,self.r_valve_idx] = self.min_max_norm(self.input["valveSignal"], self.r_valve_min, self.r_valve_max, -1, 1)
        NN_input[0,0,self.v_valve_idx] = self.min_max_norm(self.input["supplyDamperSignal"], self.v_valve_min, self.v_valve_max, -1, 1)
        NN_input[0,0,self.shades_idx] = self.min_max_norm(self.input["shadesSignal"], self.shades_min, self.shades_max, -1, 1)

        # NN_input[0,0,self.day_of_year_cos_idx] = math.cos(2*math.pi*self.time.timetuple().tm_yday/366)
        # NN_input[0,0,self.day_of_year_sin_idx] = math.sin(2*math.pi*self.time.timetuple().tm_yday/366)
        # NN_input[0,0,self.hour_of_day_cos_idx] = math.cos(2*math.pi*self.time.hour/23)
        # NN_input[0,0,self.hour_of_day_sin_idx] = math.sin(2*math.pi*self.time.hour/23)

        # print("-----")
        # print(self.input["outdoorTemperature"])
        # print(NN_input[0,0,self.OAT_idx])
        # print(self.OAT_min, self.OAT_max)

        NN_input = NN_input.float()
        with torch.no_grad():    
            NN_output_temp,self.hidden_state = self.model(NN_input,self.hidden_state, training_mode=self.first_time_step)
            NN_output_temp = NN_output_temp.detach().numpy()[0][0][0]

        y_min = -1 ########
        y_max = 1 #######
        dT = self.rescale(NN_output_temp, y_min, y_max, -1, 1)

        T = self.output["indoorTemperature"] + dT
        return T

    def update_output(self):
        self.output["indoorTemperature"] = self.get_temperature()
        self.output["indoorCo2Concentration"] = self.output["indoorCo2Concentration"] + (self.input["outdoorCo2Concentration"]*self.input["supplyAirFlowRate"] - self.output["indoorCo2Concentration"]*self.input["returnAirFlowRate"] + self.input["numberOfPeople"]*self.input["generationCo2Concentration"])*self.timeStep/self.airMass


        if self.first_time_step == True:
            self.first_time_step = False

        self.time += datetime.timedelta(seconds = self.timeStep)