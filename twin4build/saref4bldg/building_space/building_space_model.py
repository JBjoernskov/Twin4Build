# from .building_space import BuildingSpace
import twin4build.saref4bldg.building_space.building_space as building_space
import os
import torch
import datetime
import numpy as np
from twin4build.utils.uppath import uppath
from typing import List, Tuple
from torch import Tensor
import copy


class LSTM_split_all(torch.nn.Module):

    def __init__(self, 
                 n_input=None, 
                 n_lstm_hidden=None, 
                 n_lstm_layers=None, 
                 n_output=None, 
                 dropout=None):

        self.kwargs = {"n_input": n_input,
                        "n_lstm_hidden": n_lstm_hidden,
                        "n_lstm_layers": n_lstm_layers,
                        "n_output": n_output,
                        "dropout": dropout}
        
        self.n_input = n_input
    
        (self.n_lstm_hidden_OUTDOORTEMPERATURE,
        self.n_lstm_hidden_RADIATION,
        self.n_lstm_hidden_SPACEHEATER,
        self.n_lstm_hidden_VENTILATION) = n_lstm_hidden
        

        (self.n_lstm_layers_OUTDOORTEMPERATURE,
        self.n_lstm_layers_RADIATION,
        self.n_lstm_layers_SPACEHEATER,
        self.n_lstm_layers_VENTILATION) = n_lstm_layers

        self.n_output = n_output

        self.dropout = dropout

        super(LSTM_split_all, self).__init__()

        # self.lstm_input_OUTDOORTEMPERATURE = torch.nn.LSTM(2, self.n_lstm_hidden_OUTDOORTEMPERATURE, self.n_lstm_layers_OUTDOORTEMPERATURE, batch_first=True, dropout=self.dropout, bias=False)
        # self.lstm_output_OUTDOORTEMPERATURE = torch.nn.LSTM(self.n_lstm_hidden_OUTDOORTEMPERATURE, self.n_output, 1, batch_first=True, bias=False)

        # self.lstm_hidden1_OUTDOORTEMPERATURE = torch.nn.LSTM(self.n_lstm_hidden_OUTDOORTEMPERATURE, self.n_lstm_hidden_OUTDOORTEMPERATURE, self.n_lstm_layers_OUTDOORTEMPERATURE, batch_first=True, dropout=self.dropout, bias=False)
        # self.lstm_hidden2_OUTDOORTEMPERATURE = torch.nn.LSTM(self.n_lstm_hidden_OUTDOORTEMPERATURE, self.n_lstm_hidden_OUTDOORTEMPERATURE, self.n_lstm_layers_OUTDOORTEMPERATURE, batch_first=True, dropout=self.dropout, bias=False)
        # self.lstm_hidden3_OUTDOORTEMPERATURE = torch.nn.LSTM(self.n_lstm_hidden_OUTDOORTEMPERATURE, self.n_lstm_hidden_OUTDOORTEMPERATURE, self.n_lstm_layers_OUTDOORTEMPERATURE, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_input_OUTDOORTEMPERATURE = torch.nn.LSTM(2, self.n_lstm_hidden_OUTDOORTEMPERATURE, self.n_lstm_layers_OUTDOORTEMPERATURE, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_OUTDOORTEMPERATURE = torch.nn.LSTM(self.n_lstm_hidden_OUTDOORTEMPERATURE, self.n_output, 1, batch_first=True, bias=False)


        # self.T_w_in__cp = torch.nn.Parameter(torch.randn(1))
        # self.m_w_max = torch.nn.Parameter(torch.randn(1))
        # self.UA = torch.nn.Parameter(torch.randn(1))

        # self.lstm_hidden1_RADIATION = torch.nn.LSTM(self.n_lstm_hidden_RADIATION, self.n_lstm_hidden_RADIATION, self.n_lstm_layers_RADIATION, batch_first=True, dropout=self.dropout, bias=False)
        # self.lstm_hidden2_RADIATION = torch.nn.LSTM(self.n_lstm_hidden_RADIATION, self.n_lstm_hidden_RADIATION, self.n_lstm_layers_RADIATION, batch_first=True, dropout=self.dropout, bias=False)
        # self.lstm_hidden3_RADIATION = torch.nn.LSTM(self.n_lstm_hidden_RADIATION, self.n_lstm_hidden_RADIATION, self.n_lstm_layers_RADIATION, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_input_RADIATION = torch.nn.LSTM(2, self.n_lstm_hidden_RADIATION, self.n_lstm_layers_RADIATION, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_RADIATION = torch.nn.LSTM(self.n_lstm_hidden_RADIATION, self.n_output, 1, batch_first=True, bias=False)


        # self.T_w_in__cp = torch.nn.Parameter(torch.randn(1))
        # self.m_w_max = torch.nn.Parameter(torch.randn(1))
        # self.UA = torch.nn.Parameter(torch.randn(1))

        self.lstm_input_RADIATION = torch.nn.LSTM(2, self.n_lstm_hidden_RADIATION, self.n_lstm_layers_RADIATION, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_RADIATION = torch.nn.LSTM(self.n_lstm_hidden_RADIATION, self.n_output, 1, batch_first=True, bias=False)

        self.lstm_input_SPACEHEATER = torch.nn.LSTM(2, self.n_lstm_hidden_SPACEHEATER, self.n_lstm_layers_SPACEHEATER, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_SPACEHEATER = torch.nn.LSTM(self.n_lstm_hidden_SPACEHEATER, self.n_output, 1, batch_first=True, bias=False)

        self.lstm_input_VENTILATION = torch.nn.LSTM(3, self.n_lstm_hidden_VENTILATION, self.n_lstm_layers_VENTILATION, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_VENTILATION = torch.nn.LSTM(self.n_lstm_hidden_VENTILATION, self.n_output, 1, batch_first=True, bias=False)

        self.dropout = torch.nn.Dropout(p=self.dropout, inplace=False)


        # self.linear_OUTDOORTEMPERATURE = torch.nn.Linear(self.n_lstm_hidden_OUTDOORTEMPERATURE,1)
        # self.linear_RADIATION = torch.nn.Linear(self.n_lstm_hidden_RADIATION,1)
        # self.linear_SPACEHEATER = torch.nn.Linear(self.n_lstm_hidden_SPACEHEATER,1)
        # self.linear_VENTILATION = torch.nn.Linear(self.n_lstm_hidden_VENTILATION,1)

        self.linear_u = torch.nn.Linear(1,1, bias=False)
        self.linear_T_o_hid = torch.nn.Linear(1,self.n_lstm_hidden_VENTILATION)
        self.linear_T_o_out = torch.nn.Linear(self.n_lstm_hidden_VENTILATION,1)
        self.linear_T_z = torch.nn.Linear(1,1)
        self.T_set = torch.nn.Parameter(torch.Tensor([1]))

        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

        
    # @jit.script_method
    def forward(self, 
                input: Tuple[Tensor, Tensor, Tensor, Tensor], 
                hidden_state: Tuple[Tuple[Tensor, Tensor], 
                                    Tuple[Tensor, Tensor], 
                                    Tuple[Tensor, Tensor], 
                                    Tuple[Tensor, Tensor],
                                    Tuple[Tensor, Tensor], 
                                    Tuple[Tensor, Tensor], 
                                    Tuple[Tensor, Tensor], 
                                    Tuple[Tensor, Tensor]]):
        
        
        (x_OUTDOORTEMPERATURE,
        x_RADIATION,
        x_SPACEHEATER,
        x_VENTILATION) = input

        (hidden_state_input_OUTDOORTEMPERATURE,
        hidden_state_output_OUTDOORTEMPERATURE,
        hidden_state_input_RADIATION,
        hidden_state_output_RADIATION,
        hidden_state_input_SPACEHEATER,
        hidden_state_output_SPACEHEATER,
        hidden_state_input_VENTILATION,
        hidden_state_output_VENTILATION) = hidden_state

        x_OUTDOORTEMPERATURE,hidden_state_input_OUTDOORTEMPERATURE = self.lstm_input_OUTDOORTEMPERATURE(x_OUTDOORTEMPERATURE,hidden_state_input_OUTDOORTEMPERATURE)
        # x_OUTDOORTEMPERATURE = self.dropout(x_OUTDOORTEMPERATURE)
        # x_OUTDOORTEMPERATURE = x_OUTDOORTEMPERATURE + x_OUTDOORTEMPERATURE_
        x_OUTDOORTEMPERATURE,hidden_state_output_OUTDOORTEMPERATURE = self.lstm_output_OUTDOORTEMPERATURE(x_OUTDOORTEMPERATURE,hidden_state_output_OUTDOORTEMPERATURE)

        x_RADIATION,hidden_state_input_RADIATION = self.lstm_input_RADIATION(x_RADIATION,hidden_state_input_RADIATION)
        # x_RADIATION = self.dropout(x_RADIATION)
        # x_RADIATION = x_RADIATION + x_RADIATION_
        x_RADIATION,hidden_state_output_RADIATION = self.lstm_output_RADIATION(x_RADIATION,hidden_state_output_RADIATION)

        x_SPACEHEATER,hidden_state_input_SPACEHEATER = self.lstm_input_SPACEHEATER(x_SPACEHEATER,hidden_state_input_SPACEHEATER)
        # x_SPACEHEATER = self.dropout(x_SPACEHEATER)
        # x_SPACEHEATER = x_SPACEHEATER + x_SPACEHEATER_
        x_SPACEHEATER,hidden_state_output_SPACEHEATER = self.lstm_output_SPACEHEATER(x_SPACEHEATER,hidden_state_output_SPACEHEATER)


        # x_VENTILATION,hidden_state_input_VENTILATION = self.lstm_input_VENTILATION(x_VENTILATION,hidden_state_input_VENTILATION)
        # x_VENTILATION = self.dropout(x_VENTILATION)
        # x_VENTILATION = x_VENTILATION + x_VENTILATION_
        # x_VENTILATION,hidden_state_output_VENTILATION = self.lstm_output_VENTILATION(x_VENTILATION,hidden_state_output_VENTILATION)



        d1,d2,d3 = x_VENTILATION.size()
        x_VENTILATION = x_VENTILATION.reshape(d1*d2,d3)
        # x_VENTILATION = self.tanh(self.linear_VENTILATION(x_VENTILATION))
        
        x1 = torch.unsqueeze(x_VENTILATION[:,0],1)
        x2 = torch.unsqueeze(x_VENTILATION[:,2],1)
        x3 = torch.unsqueeze(x_VENTILATION[:,1],1)
        # x_VENTILATION = (self.linear_T_z(x1)-self.relu(self.linear_T_o(x2)))*self.linear_u(x3)
        # x_VENTILATION = x_VENTILATION.reshape(d1,d2,1)
        

        T_a_hid = self.sigmoid(self.linear_T_o_hid(x2))
        T_a = self.linear_T_o_out(T_a_hid)
        # print("---")
        # print(self.linear_T_o_out.weight.data)
        # print(T_a)
        x_VENTILATION = (T_a-self.linear_T_z(x1))*self.linear_u(x3)
        x_VENTILATION = x_VENTILATION.reshape(d1,d2,1)


        x = (x_OUTDOORTEMPERATURE, x_RADIATION, x_SPACEHEATER, x_VENTILATION)
        y = x_OUTDOORTEMPERATURE + x_RADIATION + x_SPACEHEATER + x_VENTILATION




        hidden_state = (hidden_state_input_OUTDOORTEMPERATURE,
                        hidden_state_output_OUTDOORTEMPERATURE,
                        hidden_state_input_RADIATION,
                        hidden_state_output_RADIATION,
                        hidden_state_input_SPACEHEATER,
                        hidden_state_output_SPACEHEATER,
                        hidden_state_input_VENTILATION,
                        hidden_state_output_VENTILATION)

        return y,hidden_state,x

class LSTM_split(torch.nn.Module):

    def __init__(self, n_input, n_lstm_hidden, n_lstm_hidden_SH, n_lstm_layers, n_output):
        self.n_input = n_input
        self.n_lstm_hidden = n_lstm_hidden
        self.n_lstm_hidden_SH = n_lstm_hidden_SH
        self.n_lstm_layers = n_lstm_layers
        self.n_output = n_output

        super(LSTM_split, self).__init__()

        self.lstm_input_SH = torch.nn.LSTM(2, self.n_lstm_hidden_SH, self.n_lstm_layers, batch_first=True)
        self.lstm_output_SH = torch.nn.LSTM(self.n_lstm_hidden_SH, self.n_output, 1, batch_first=True)

        self.lstm_input = torch.nn.LSTM(self.n_input-1, self.n_lstm_hidden, self.n_lstm_layers, batch_first=True)
        self.lstm_output = torch.nn.LSTM(self.n_lstm_hidden, self.n_output, 1, batch_first=True)

        
    # @jit.script_method
    def forward(self, input: Tuple[Tensor, Tensor], hidden_state: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]):
        
        
        x_SH, x = input
        hidden_state_input_SH,hidden_state_output_SH,hidden_state_input,hidden_state_output = hidden_state

        x_SH,hidden_state_input_SH = self.lstm_input_SH(x_SH,hidden_state_input_SH)
        x_SH,hidden_state_output_SH = self.lstm_output_SH(x_SH,hidden_state_output_SH)

        x,hidden_state_input = self.lstm_input(x,hidden_state_input)
        x,hidden_state_output = self.lstm_output(x,hidden_state_output)

        y = x_SH+x


        hidden_state = (hidden_state_input_SH,hidden_state_output_SH,hidden_state_input,hidden_state_output)

        if self.training:
            return y,hidden_state,x_SH
        else:
            return y,hidden_state,x_SH

class LSTM(torch.nn.Module):

    def __init__(self, n_input, n_lstm_hidden, n_lstm_layers, n_output):
        self.n_input = n_input
        self.n_lstm_hidden = n_lstm_hidden
        self.n_lstm_layers = n_lstm_layers
        self.n_output = n_output
        super(LSTM, self).__init__()
        self.lstm_input = torch.nn.LSTM(self.n_input, self.n_lstm_hidden, self.n_lstm_layers, batch_first=True)
        self.lstm_output = torch.nn.LSTM(self.n_lstm_hidden, self.n_output, 1, batch_first=True)

        
    # @jit.script_method
    def forward(self, flat_sequence_input: Tensor, hidden_state: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]):
        hidden_state_input,hidden_state_output = hidden_state
        x,hidden_state_input = self.lstm_input(flat_sequence_input,hidden_state_input)
        x,hidden_state_output = self.lstm_output(x,hidden_state_output)
        hidden_state = (hidden_state_input,hidden_state_output)

        return x,hidden_state

class NoSpaceModelException(Exception): 
    def __init__(self, message="No fitting space model"):
        self.message = message
        super().__init__(self.message)


class BuildingSpaceModel(building_space.BuildingSpace):
    
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

        self.x_list = []
        self.input_OUTDOORTEMPERATURE = []
        self.input_RADIATION = []
        self.input_SPACEHEATER = []
        self.input_VENTILATION = []

        try:
            building_data_collection_dict
        except:
            import twin4build.utils.building_data_collection_dict as building_data_collection_dict
        space_data_collection = building_data_collection_dict.building_data_collection_dict[self.id]
        if space_data_collection.has_sufficient_data==False:
            raise NoSpaceModelException
        self.sw_radiation_idx = list(space_data_collection.clean_data_dict.keys()).index("sw_radiation")
        # self.lw_radiation_idx = list(space_data_collection.clean_data_dict.keys()).index("lw_radiation")  ################
        self.OAT_idx = list(space_data_collection.clean_data_dict.keys()).index("OAT")
        self.temperature_idx = list(space_data_collection.clean_data_dict.keys()).index("temperature")
        # self.CO2_idx = list(space_data_collection.clean_data_dict.keys()).index("CO2") ################
        self.r_valve_idx = list(space_data_collection.clean_data_dict.keys()).index("r_valve")
        self.v_valve_idx = list(space_data_collection.clean_data_dict.keys()).index("v_valve")
        self.shades_idx = list(space_data_collection.clean_data_dict.keys()).index("shades")

        # self.day_of_year_cos_idx = list(space_data_collection.clean_data_dict.keys()).index("day_of_year_cos") ##################
        # self.day_of_year_sin_idx = list(space_data_collection.clean_data_dict.keys()).index("day_of_year_sin") ##################
        # self.hour_of_day_cos_idx = list(space_data_collection.clean_data_dict.keys()).index("hour_of_day_cos") ###############
        # self.hour_of_day_sin_idx = list(space_data_collection.clean_data_dict.keys()).index("hour_of_day_sin") ###################


        self.sw_radiation_min = space_data_collection.data_min_vec[self.sw_radiation_idx]
        self.sw_radiation_max = space_data_collection.data_max_vec[self.sw_radiation_idx]
        # self.lw_radiation_min = space_data_collection.data_min_vec[self.lw_radiation_idx] ################
        # self.lw_radiation_max = space_data_collection.data_max_vec[self.lw_radiation_idx] ################
        self.OAT_min = space_data_collection.data_min_vec[self.OAT_idx]
        self.OAT_max = space_data_collection.data_max_vec[self.OAT_idx]
        self.temperature_min = space_data_collection.data_min_vec[self.temperature_idx]
        self.temperature_max = space_data_collection.data_max_vec[self.temperature_idx]
        # self.CO2_min = space_data_collection.data_min_vec[self.CO2_idx] ################
        # self.CO2_max = space_data_collection.data_max_vec[self.CO2_idx] ################
        self.r_valve_min = space_data_collection.data_min_vec[self.r_valve_idx]
        self.r_valve_max = space_data_collection.data_max_vec[self.r_valve_idx]
        self.v_valve_min = space_data_collection.data_min_vec[self.v_valve_idx]
        self.v_valve_max = space_data_collection.data_max_vec[self.v_valve_idx]
        self.shades_min = space_data_collection.data_min_vec[self.shades_idx]
        self.shades_max = space_data_collection.data_max_vec[self.shades_idx]


        self.adjacent_min = space_data_collection.data_min_vec[4:8]
        self.adjacent_max = space_data_collection.data_max_vec[4:8]


        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")
        print("Using device: "+ str(self.device))
        self.first_time_step = True
        self.n_input = space_data_collection.data_matrix.shape[1] ########################
        # self.n_lstm_hidden = 20 #############################
        # self.n_lstm_hidden_SH = 20 #############################
        # self.n_lstm_hidden = (3,3,2,3) #20
        # self.n_lstm_layers = 1
        # self.n_output = 1

        self.get_model()

        # h_0_input = torch.zeros((self.n_lstm_layers,1,self.n_lstm_hidden)).to(self.device)
        # c_0_input = torch.zeros((self.n_lstm_layers,1,self.n_lstm_hidden)).to(self.device)
        # h_0_output = torch.zeros((1,1,1)).to(self.device)
        # c_0_output = torch.zeros((1,1,1)).to(self.device)
        # self.hidden_state = ((h_0_input,c_0_input), (h_0_output,c_0_output))



        # h_0_input_layer_SH = torch.zeros((self.n_lstm_layers,1,self.n_lstm_hidden_SH))
        # c_0_input_layer_SH = torch.zeros((self.n_lstm_layers,1,self.n_lstm_hidden_SH))
        # h_0_output_layer_SH = torch.zeros((1,1,self.n_output))
        # c_0_output_layer_SH = torch.zeros((1,1,self.n_output))


        # h_0_input_layer = torch.zeros((self.n_lstm_layers,1,self.n_lstm_hidden))
        # c_0_input_layer = torch.zeros((self.n_lstm_layers,1,self.n_lstm_hidden))
        # h_0_output_layer = torch.zeros((1,1,self.n_output))
        # c_0_output_layer = torch.zeros((1,1,self.n_output))


        # hidden_state_input_SH = (h_0_input_layer_SH,c_0_input_layer_SH)
        # hidden_state_output_SH = (h_0_output_layer_SH,c_0_output_layer_SH)

        # hidden_state_input = (h_0_input_layer,c_0_input_layer)
        # hidden_state_output = (h_0_output_layer,c_0_output_layer)


        # self.hidden_state = (hidden_state_input_SH,hidden_state_output_SH,hidden_state_input,hidden_state_output) ###############



        

        
        



    def rescale(self,y,y_min,y_max,low,high):
        y = (y-low)/(high-low)*(y_max-y_min) + y_min
        return y

    def min_max_norm(self,y,y_min,y_max,low,high):
        y = (y-y_min)/(y_max-y_min)*(high-low) + low
        return y

    def unpack_dict(self, dict_):
        dict_

    def get_model(self):
        search_path = os.path.join(uppath(os.path.abspath(__file__), 3), "test", "data", "space_models", "test")
        # search_path = "C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/OU44_space_models/rooms_no_time_600k_20n_test_all"
        # search_path = os.path.join(uppath(os.path.abspath(__file__), 3), "test", "data", "space_models", "rooms_notime_noCO2_nolw_500k_20n_test_all_split")
        # search_path = os.path.join(uppath(os.path.abspath(__file__), 3), "test", "data", "space_models", "rooms_notime_noCO2_nolw_1000k_20n_test_all")
        directory = os.fsencode(search_path)
        found_file = False
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.find(self.id.replace("Ø", "OE") + "_Network") != -1:
                found_file = True
                break

        if found_file==False:
            raise NoSpaceModelException
        full_path = search_path + "/" + filename


        # model = torch.jit.load(full_path).to(self.device)
        # model = LSTM_split(self.n_input, self.n_lstm_hidden, self.n_lstm_hidden_SH, self.n_lstm_layers, self.n_output)
        
        # model = LSTM(self.n_input, self.n_lstm_hidden, self.n_lstm_layers, self.n_output)
        self.kwargs, state_dict = torch.load(full_path)
        self.model = LSTM_split_all(**self.kwargs)
        self.model.load_state_dict(state_dict)#.to(self.device)
        self.model.eval()

        h_0_input_layer_OUTDOORTEMPERATURE = torch.zeros((self.kwargs["n_lstm_layers"][0],1,self.kwargs["n_lstm_hidden"][0]))
        c_0_input_layer_OUTDOORTEMPERATURE = torch.zeros((self.kwargs["n_lstm_layers"][0],1,self.kwargs["n_lstm_hidden"][0]))
        h_0_output_layer_OUTDOORTEMPERATURE = torch.zeros((1,1,self.kwargs["n_output"]))
        c_0_output_layer_OUTDOORTEMPERATURE = torch.zeros((1,1,self.kwargs["n_output"]))
        hidden_state_input_OUTDOORTEMPERATURE = (h_0_input_layer_OUTDOORTEMPERATURE,c_0_input_layer_OUTDOORTEMPERATURE)
        hidden_state_output_OUTDOORTEMPERATURE = (h_0_output_layer_OUTDOORTEMPERATURE,c_0_output_layer_OUTDOORTEMPERATURE)

        h_0_input_layer_RADIATION = torch.zeros((self.kwargs["n_lstm_layers"][1],1,self.kwargs["n_lstm_hidden"][1]))
        c_0_input_layer_RADIATION = torch.zeros((self.kwargs["n_lstm_layers"][1],1,self.kwargs["n_lstm_hidden"][1]))
        h_0_output_layer_RADIATION = torch.zeros((1,1,self.kwargs["n_output"]))
        c_0_output_layer_RADIATION = torch.zeros((1,1,self.kwargs["n_output"]))
        hidden_state_input_RADIATION = (h_0_input_layer_RADIATION,c_0_input_layer_RADIATION)
        hidden_state_output_RADIATION = (h_0_output_layer_RADIATION,c_0_output_layer_RADIATION)

        h_0_input_layer_SPACEHEATER = torch.zeros((self.kwargs["n_lstm_layers"][2],1,self.kwargs["n_lstm_hidden"][2]))
        c_0_input_layer_SPACEHEATER = torch.zeros((self.kwargs["n_lstm_layers"][2],1,self.kwargs["n_lstm_hidden"][2]))
        h_0_output_layer_SPACEHEATER = torch.zeros((1,1,self.kwargs["n_output"]))
        c_0_output_layer_SPACEHEATER = torch.zeros((1,1,self.kwargs["n_output"]))
        hidden_state_input_SPACEHEATER = (h_0_input_layer_SPACEHEATER,c_0_input_layer_SPACEHEATER)
        hidden_state_output_SPACEHEATER = (h_0_output_layer_SPACEHEATER,c_0_output_layer_SPACEHEATER)

        h_0_input_layer_VENTILATION = torch.zeros((self.kwargs["n_lstm_layers"][3],1,self.kwargs["n_lstm_hidden"][3]))
        c_0_input_layer_VENTILATION = torch.zeros((self.kwargs["n_lstm_layers"][3],1,self.kwargs["n_lstm_hidden"][3]))
        h_0_output_layer_VENTILATION = torch.zeros((1,1,self.kwargs["n_output"]))
        c_0_output_layer_VENTILATION = torch.zeros((1,1,self.kwargs["n_output"]))
        hidden_state_input_VENTILATION = (h_0_input_layer_VENTILATION,c_0_input_layer_VENTILATION)
        hidden_state_output_VENTILATION = (h_0_output_layer_VENTILATION,c_0_output_layer_VENTILATION)



        self.hidden_state = (hidden_state_input_OUTDOORTEMPERATURE,
                            hidden_state_output_OUTDOORTEMPERATURE,
                            hidden_state_input_RADIATION,
                            hidden_state_output_RADIATION,
                            hidden_state_input_SPACEHEATER,
                            hidden_state_output_SPACEHEATER,
                            hidden_state_input_VENTILATION,
                            hidden_state_output_VENTILATION)
    
    def get_temperature(self):
        
        x_OUTDOORTEMPERATURE = torch.zeros((1, 1, 2))
        x_RADIATION = torch.zeros((1, 1, 2))
        x_SPACEHEATER = torch.zeros((1, 1, 2))
        x_VENTILATION = torch.zeros((1, 1, 3))

        # x_OUTDOORTEMPERATURE[:,:,0] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, -1, 1) #indoor
        # x_OUTDOORTEMPERATURE[:,:,1] = self.min_max_norm(self.input["outdoorTemperature"], self.OAT_min, self.OAT_max, -1, 1) #outdoor
        # x_RADIATION[:,:,0] = self.min_max_norm(self.input["shadePosition"], self.shades_min, self.shades_max, -1, 1) #shades
        # x_RADIATION[:,:,1] = self.min_max_norm(self.input["shortwaveRadiation"], self.sw_radiation_min, self.sw_radiation_max, -1, 1) #SW
        # x_SPACEHEATER[:,:,0] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, -1, 1) #indoor
        # x_SPACEHEATER[:,:,1] = self.min_max_norm(self.input["valvePosition"], self.r_valve_min, self.r_valve_max, -1, 1) #valve
        # x_VENTILATION[:,:,0] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, -1, 1) #indoor
        # x_VENTILATION[:,:,1] = self.min_max_norm(self.input["supplyDamperPosition"], self.v_valve_min, self.v_valve_max, -1, 1) #damper


        y_low = 0
        y_high = 1


        x_OUTDOORTEMPERATURE[:,:,0] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, y_low, y_high) #indoor
        x_OUTDOORTEMPERATURE[:,:,1] = self.min_max_norm(self.input["outdoorTemperature"], self.OAT_min, self.OAT_max, y_low, y_high) #outdoor
        # x_OUTDOORTEMPERATURE[:,:,2:6] = torch.from_numpy(self.min_max_norm(23, self.adjacent_min, self.adjacent_max, -1, 1)) #adjacent temperatures



        x_RADIATION[:,:,0] = self.min_max_norm(self.input["shadePosition"], self.shades_min, self.shades_max, y_low, y_high) #shades
        x_RADIATION[:,:,1] = self.min_max_norm(self.input["shortwaveRadiation"], self.sw_radiation_min, self.sw_radiation_max, y_low, y_high) #SW
        # x_RADIATION[:,:,2] = np.cos(2*np.pi*self.time.timetuple().tm_yday/365)
        # x_RADIATION[:,:,3] = np.sin(2*np.pi*self.time.timetuple().tm_yday/365)
        # x_RADIATION[:,:,4] = np.cos(2*np.pi*self.time.hour/23)
        # x_RADIATION[:,:,5] = np.sin(2*np.pi*self.time.hour/23)
        x_SPACEHEATER[:,:,0] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, y_low, y_high) #indoor
        x_SPACEHEATER[:,:,1] = self.min_max_norm(self.input["valvePosition"], self.r_valve_min, self.r_valve_max, y_low, y_high) #valve
        x_VENTILATION[:,:,0] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, y_low, y_high) #indoor
        x_VENTILATION[:,:,1] = self.min_max_norm(self.input["supplyDamperPosition"], self.v_valve_min, self.v_valve_max, y_low, y_high) #damper
        x_VENTILATION[:,:,2] = self.min_max_norm(self.input["outdoorTemperature"], self.OAT_min, self.OAT_max, y_low, y_high) #outdoor

        input = (x_OUTDOORTEMPERATURE,
                x_RADIATION,
                x_SPACEHEATER,
                x_VENTILATION)




        with torch.no_grad():    
            NN_output_temp,self.hidden_state,x = self.model(input,self.hidden_state)
            NN_output_temp = NN_output_temp.detach().cpu().numpy()[0][0][0]

            self.input_OUTDOORTEMPERATURE.append(input[0][0,0,:].tolist()) 
            self.input_RADIATION.append(input[1][0,0,:].tolist()) 
            self.input_SPACEHEATER.append(input[2][0,0,:].tolist()) 
            self.input_VENTILATION.append(input[3][0,0,:].tolist()) 
            self.x_list.append([x[0][0,0,0], x[1][0,0,0], x[2][0,0,0], x[3][0,0,0]])

            # print("---")
            # print(input)
            # print(x)


        y_min = -1 
        y_max = 1 
        dT = self.rescale(NN_output_temp, y_min, y_max, -1, 1)
        T = self.output["indoorTemperature"] + dT
        return T

    def get_temperature_old(self):
        

        x_SH = torch.zeros((1,1,2)).to(self.device)
        x_SH[0,0,0] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, -1, 1)
        x_SH[0,0,1] = self.min_max_norm(self.input["valvePosition"], self.r_valve_min, self.r_valve_max, -1, 1)
        

        x = torch.zeros((1,1,self.n_input-1)).to(self.device)
        x[0,0,0] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, -1, 1)
        x[0,0,1] = self.min_max_norm(self.input["supplyDamperPosition"], self.v_valve_min, self.v_valve_max, -1, 1)
        x[0,0,2] = self.min_max_norm(self.input["shadePosition"], self.shades_min, self.shades_max, -1, 1)
        x[0,0,3] = self.min_max_norm(self.input["shortwaveRadiation"], self.sw_radiation_min, self.sw_radiation_max, -1, 1)
        x[0,0,4] = self.min_max_norm(self.input["outdoorTemperature"], self.OAT_min, self.OAT_max, -1, 1)

        input = (x_SH, x)



        x_OUTDOORTEMPERATURE = torch.zeros((1, 1, 2))
        x_RADIATION = torch.zeros((1, 1, 2))
        x_SPACEHEATER = torch.zeros((1, 1, 2))
        x_VENTILATION = torch.zeros((1, 1, 2))

        x_OUTDOORTEMPERATURE[:,:,0] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, -1, 1) #indoor
        x_OUTDOORTEMPERATURE[:,:,1] = self.min_max_norm(self.input["outdoorTemperature"], self.OAT_min, self.OAT_max, -1, 1) #outdoor
        x_RADIATION[:,:,0] = self.min_max_norm(self.input["shadePosition"], self.shades_min, self.shades_max, -1, 1) #shades
        x_RADIATION[:,:,1] = self.min_max_norm(self.input["shortwaveRadiation"], self.sw_radiation_min, self.sw_radiation_max, -1, 1) #SW
        x_SPACEHEATER[:,:,0] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, -1, 1) #indoor
        x_SPACEHEATER[:,:,1] = self.min_max_norm(self.input["valvePosition"], self.r_valve_min, self.r_valve_max, -1, 1) #valve
        x_VENTILATION[:,:,0] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, -1, 1) #indoor
        x_VENTILATION[:,:,1] = self.min_max_norm(self.input["supplyDamperPosition"], self.v_valve_min, self.v_valve_max, -1, 1) #damper

        input = (x_OUTDOORTEMPERATURE,
                x_RADIATION,
                x_SPACEHEATER,
                x_VENTILATION)


        NN_input = torch.zeros((1,1,self.n_input)).to(self.device)

        NN_input[0,0,self.sw_radiation_idx] = self.min_max_norm(self.input["shortwaveRadiation"], self.sw_radiation_min, self.sw_radiation_max, -1, 1)
        # NN_input[0,0,self.lw_radiation_idx] = self.min_max_norm(self.input["longwaveRadiation"], self.lw_radiation_min, self.lw_radiation_max, -1, 1) ################
        NN_input[0,0,self.OAT_idx] = self.min_max_norm(self.input["outdoorTemperature"], self.OAT_min, self.OAT_max, -1, 1)
        NN_input[0,0,self.temperature_idx] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, -1, 1)
        # NN_input[0,0,self.CO2_idx] = self.min_max_norm(self.output["indoorCo2Concentration"], self.CO2_min, self.CO2_max, -1, 1) ################
        NN_input[0,0,self.r_valve_idx] = self.min_max_norm(self.input["valvePosition"], self.r_valve_min, self.r_valve_max, -1, 1)
        NN_input[0,0,self.v_valve_idx] = self.min_max_norm(self.input["supplyDamperPosition"], self.v_valve_min, self.v_valve_max, -1, 1)
        NN_input[0,0,self.shades_idx] = self.min_max_norm(self.input["shadePosition"], self.shades_min, self.shades_max, -1, 1)

        NN_input = NN_input.float()
        # input = NN_input ########################################################################
        with torch.no_grad():    
            # NN_output_temp,self.hidden_state,nonnegative = self.model(input,self.hidden_state)
            NN_output_temp,self.hidden_state,x = self.model(input,self.hidden_state)
            
            # NN_output_temp,self.hidden_state = self.model(input,self.hidden_state)
            # NN_output_temp,self.hidden_state = self.model(input,self.hidden_state, training_mode=False)
            NN_output_temp = NN_output_temp.detach().cpu().numpy()[0][0][0]
            print("----")
            print(x)


        y_min = -1 ########
        y_max = 1 #######
        dT = self.rescale(NN_output_temp, y_min, y_max, -1, 1)
        T = self.output["indoorTemperature"] + dT
        return T

    def update_output(self):

        M_air = 28.9647 #g/mol
        M_CO2 = 44.01 #g/mol
        K_conversion = M_CO2/M_air*1e-6

        self.output["indoorTemperature"] = self.get_temperature()
        # self.output["indoorCo2Concentration"] = self.output["indoorCo2Concentration"] + (self.input["outdoorCo2Concentration"]*self.input["supplyAirFlowRate"] - self.output["indoorCo2Concentration"]*self.input["returnAirFlowRate"] + self.input["numberOfPeople"]*self.input["generationCo2Concentration"])*self.timeStep/self.airMass
        self.output["indoorCo2Concentration"] = (self.airMass*self.output["indoorCo2Concentration"] + self.input["outdoorCo2Concentration"]*self.input["supplyAirFlowRate"]*self.timeStep + self.input["generationCo2Concentration"]*self.input["numberOfPeople"]*self.timeStep/K_conversion)/(self.airMass + self.input["returnAirFlowRate"]*self.timeStep)

        if self.first_time_step == True:
            self.first_time_step = False

        self.time += datetime.timedelta(seconds = self.timeStep)