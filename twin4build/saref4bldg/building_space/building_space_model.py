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
import onnxruntime


class LSTMColapsed(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        x_OUTDOORTEMPERATURE: torch.Tensor,
        x_RADIATION: torch.Tensor,
        x_SPACEHEATER: torch.Tensor,
        x_VENTILATION: torch.Tensor,
        h_0_input_layer_OUTDOORTEMPERATURE: torch.Tensor,
        c_0_input_layer_OUTDOORTEMPERATURE: torch.Tensor,
        h_0_output_layer_OUTDOORTEMPERATURE: torch.Tensor,
        c_0_output_layer_OUTDOORTEMPERATURE: torch.Tensor,
        h_0_input_layer_RADIATION: torch.Tensor,
        c_0_input_layer_RADIATION: torch.Tensor,
        h_0_output_layer_RADIATION: torch.Tensor,
        c_0_output_layer_RADIATION: torch.Tensor,
        h_0_input_layer_SPACEHEATER: torch.Tensor,
        c_0_input_layer_SPACEHEATER: torch.Tensor,
        h_0_output_layer_SPACEHEATER: torch.Tensor,
        c_0_output_layer_SPACEHEATER: torch.Tensor,
        h_0_input_layer_VENTILATION: torch.Tensor,
        c_0_input_layer_VENTILATION: torch.Tensor,
        h_0_output_layer_VENTILATION: torch.Tensor,
        c_0_output_layer_VENTILATION: torch.Tensor):

        input = (x_OUTDOORTEMPERATURE,
                x_RADIATION,
                x_SPACEHEATER,
                x_VENTILATION)

        hidden_state_input_OUTDOORTEMPERATURE = (h_0_input_layer_OUTDOORTEMPERATURE,c_0_input_layer_OUTDOORTEMPERATURE)
        hidden_state_output_OUTDOORTEMPERATURE = (h_0_output_layer_OUTDOORTEMPERATURE,c_0_output_layer_OUTDOORTEMPERATURE)
        hidden_state_input_RADIATION = (h_0_input_layer_RADIATION,c_0_input_layer_RADIATION)
        hidden_state_output_RADIATION = (h_0_output_layer_RADIATION,c_0_output_layer_RADIATION)
        hidden_state_input_SPACEHEATER = (h_0_input_layer_SPACEHEATER,c_0_input_layer_SPACEHEATER)
        hidden_state_output_SPACEHEATER = (h_0_output_layer_SPACEHEATER,c_0_output_layer_SPACEHEATER)
        hidden_state_input_VENTILATION = (h_0_input_layer_VENTILATION,c_0_input_layer_VENTILATION)
        hidden_state_output_VENTILATION = (h_0_output_layer_VENTILATION,c_0_output_layer_VENTILATION)
        hidden_state = (hidden_state_input_OUTDOORTEMPERATURE,
                            hidden_state_output_OUTDOORTEMPERATURE,
                            hidden_state_input_RADIATION,
                            hidden_state_output_RADIATION,
                            hidden_state_input_SPACEHEATER,
                            hidden_state_output_SPACEHEATER,
                            hidden_state_input_VENTILATION,
                            hidden_state_output_VENTILATION)


        output, hidden_state, x = self.model(input, hidden_state)
        return output, hidden_state, x

class LSTM(torch.nn.Module):
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

        super(LSTM, self).__init__()

        self.lstm_input_OUTDOORTEMPERATURE = torch.nn.LSTM(2, self.n_lstm_hidden_OUTDOORTEMPERATURE, self.n_lstm_layers_OUTDOORTEMPERATURE, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_OUTDOORTEMPERATURE = torch.nn.LSTM(self.n_lstm_hidden_OUTDOORTEMPERATURE, self.n_output, 1, batch_first=True, bias=False)
        self.lstm_input_RADIATION = torch.nn.LSTM(2, self.n_lstm_hidden_RADIATION, self.n_lstm_layers_RADIATION, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_RADIATION = torch.nn.LSTM(self.n_lstm_hidden_RADIATION, self.n_output, 1, batch_first=True, bias=False)

        self.lstm_input_RADIATION = torch.nn.LSTM(2, self.n_lstm_hidden_RADIATION, self.n_lstm_layers_RADIATION, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_RADIATION = torch.nn.LSTM(self.n_lstm_hidden_RADIATION, self.n_output, 1, batch_first=True, bias=False)

        self.lstm_input_SPACEHEATER = torch.nn.LSTM(2, self.n_lstm_hidden_SPACEHEATER, self.n_lstm_layers_SPACEHEATER, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_SPACEHEATER = torch.nn.LSTM(self.n_lstm_hidden_SPACEHEATER, self.n_output, 1, batch_first=True, bias=False)

        self.lstm_input_VENTILATION = torch.nn.LSTM(3, self.n_lstm_hidden_VENTILATION, self.n_lstm_layers_VENTILATION, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_VENTILATION = torch.nn.LSTM(self.n_lstm_hidden_VENTILATION, self.n_output, 1, batch_first=True, bias=False)

        self.dropout = torch.nn.Dropout(p=self.dropout, inplace=False)

        self.linear_u = torch.nn.Linear(1,1, bias=False)
        self.linear_T_o_hid = torch.nn.Linear(1,self.n_lstm_hidden_VENTILATION)
        self.linear_T_o_out = torch.nn.Linear(self.n_lstm_hidden_VENTILATION,1)
        self.linear_T_z = torch.nn.Linear(1,1)
        self.T_set = torch.nn.Parameter(torch.Tensor([1]))

        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

        self.T_set = torch.nn.Parameter(torch.Tensor([0.7]))

        
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
        x_OUTDOORTEMPERATURE,hidden_state_output_OUTDOORTEMPERATURE = self.lstm_output_OUTDOORTEMPERATURE(x_OUTDOORTEMPERATURE,hidden_state_output_OUTDOORTEMPERATURE)

        x_RADIATION,hidden_state_input_RADIATION = self.lstm_input_RADIATION(x_RADIATION,hidden_state_input_RADIATION)
        x_RADIATION,hidden_state_output_RADIATION = self.lstm_output_RADIATION(x_RADIATION,hidden_state_output_RADIATION)

        x_SPACEHEATER,hidden_state_input_SPACEHEATER = self.lstm_input_SPACEHEATER(x_SPACEHEATER,hidden_state_input_SPACEHEATER)
        x_SPACEHEATER,hidden_state_output_SPACEHEATER = self.lstm_output_SPACEHEATER(x_SPACEHEATER,hidden_state_output_SPACEHEATER)



        d1,d2,d3 = x_VENTILATION.size()
        x_VENTILATION = x_VENTILATION.reshape(d1*d2,d3)
        x_indoor = torch.unsqueeze(x_VENTILATION[:,0],1)
        x_outdoor = torch.unsqueeze(x_VENTILATION[:,2],1)
        x_damper = torch.unsqueeze(x_VENTILATION[:,1],1)
        # T_a_hid = self.relu(self.linear_T_o_hid(x2))
        # T_a = self.linear_T_o_out(T_a_hid)
        T_a = self.relu(x_outdoor-self.T_set)+self.T_set
        x_VENTILATION = (T_a-self.linear_T_z(x_indoor))*self.linear_u(x_damper)
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
        self.OAT_idx = list(space_data_collection.clean_data_dict.keys()).index("OAT")
        self.temperature_idx = list(space_data_collection.clean_data_dict.keys()).index("temperature")
        self.r_valve_idx = list(space_data_collection.clean_data_dict.keys()).index("r_valve")
        self.v_valve_idx = list(space_data_collection.clean_data_dict.keys()).index("v_valve")
        self.shades_idx = list(space_data_collection.clean_data_dict.keys()).index("shades")
        self.sw_radiation_min = space_data_collection.data_min_vec[self.sw_radiation_idx]
        self.sw_radiation_max = space_data_collection.data_max_vec[self.sw_radiation_idx]
        self.OAT_min = space_data_collection.data_min_vec[self.OAT_idx]
        self.OAT_max = space_data_collection.data_max_vec[self.OAT_idx]
        self.temperature_min = space_data_collection.data_min_vec[self.temperature_idx]
        self.temperature_max = space_data_collection.data_max_vec[self.temperature_idx]
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

        self.use_onnx = True

    def rescale(self,y,y_min,y_max,low,high):
        y = (y-low)/(high-low)*(y_max-y_min) + y_min
        return y

    def min_max_norm(self,y,y_min,y_max,low,high):
        y = (y-y_min)/(y_max-y_min)*(high-low) + low
        return y

    def unpack_dict(self, dict_):
        dict_

    def unpack(self, input, hidden_state):
        unpacked = [tensor for tensor in input]
        unpacked.extend([i for tuple in hidden_state for i in tuple])
        return tuple(unpacked)

    def get_input_dict(self, input, hidden_state):
        unpacked = self.unpack(input, hidden_state)
        input_dict = {obj.name: tensor for obj, tensor in zip(self.onnx_model.get_inputs(), unpacked)}
        return input_dict

    def pack(self, list_):
        output = list_[0]
        hidden_state_flat = list_[1:-4]
        hidden_state = [(i,j) for i,j in zip(hidden_state_flat[0::2], hidden_state_flat[1::2])]
        x = list_[-4:]
        return output, hidden_state, x

    def init_torch_hidden_state(self):

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



        hidden_state = (hidden_state_input_OUTDOORTEMPERATURE,
                            hidden_state_output_OUTDOORTEMPERATURE,
                            hidden_state_input_RADIATION,
                            hidden_state_output_RADIATION,
                            hidden_state_input_SPACEHEATER,
                            hidden_state_output_SPACEHEATER,
                            hidden_state_input_VENTILATION,
                            hidden_state_output_VENTILATION)

        return hidden_state


    def init_numpy_hidden_state(self):
        h_0_input_layer_OUTDOORTEMPERATURE = np.zeros((self.kwargs["n_lstm_layers"][0],1,self.kwargs["n_lstm_hidden"][0]), dtype=np.float32)
        c_0_input_layer_OUTDOORTEMPERATURE = np.zeros((self.kwargs["n_lstm_layers"][0],1,self.kwargs["n_lstm_hidden"][0]), dtype=np.float32)
        h_0_output_layer_OUTDOORTEMPERATURE = np.zeros((1,1,self.kwargs["n_output"]), dtype=np.float32)
        c_0_output_layer_OUTDOORTEMPERATURE = np.zeros((1,1,self.kwargs["n_output"]), dtype=np.float32)
        hidden_state_input_OUTDOORTEMPERATURE = (h_0_input_layer_OUTDOORTEMPERATURE,c_0_input_layer_OUTDOORTEMPERATURE)
        hidden_state_output_OUTDOORTEMPERATURE = (h_0_output_layer_OUTDOORTEMPERATURE,c_0_output_layer_OUTDOORTEMPERATURE)

        h_0_input_layer_RADIATION = np.zeros((self.kwargs["n_lstm_layers"][1],1,self.kwargs["n_lstm_hidden"][1]), dtype=np.float32)
        c_0_input_layer_RADIATION = np.zeros((self.kwargs["n_lstm_layers"][1],1,self.kwargs["n_lstm_hidden"][1]), dtype=np.float32)
        h_0_output_layer_RADIATION = np.zeros((1,1,self.kwargs["n_output"]), dtype=np.float32)
        c_0_output_layer_RADIATION = np.zeros((1,1,self.kwargs["n_output"]), dtype=np.float32)
        hidden_state_input_RADIATION = (h_0_input_layer_RADIATION,c_0_input_layer_RADIATION)
        hidden_state_output_RADIATION = (h_0_output_layer_RADIATION,c_0_output_layer_RADIATION)

        h_0_input_layer_SPACEHEATER = np.zeros((self.kwargs["n_lstm_layers"][2],1,self.kwargs["n_lstm_hidden"][2]), dtype=np.float32)
        c_0_input_layer_SPACEHEATER = np.zeros((self.kwargs["n_lstm_layers"][2],1,self.kwargs["n_lstm_hidden"][2]), dtype=np.float32)
        h_0_output_layer_SPACEHEATER = np.zeros((1,1,self.kwargs["n_output"]), dtype=np.float32)
        c_0_output_layer_SPACEHEATER = np.zeros((1,1,self.kwargs["n_output"]), dtype=np.float32)
        hidden_state_input_SPACEHEATER = (h_0_input_layer_SPACEHEATER,c_0_input_layer_SPACEHEATER)
        hidden_state_output_SPACEHEATER = (h_0_output_layer_SPACEHEATER,c_0_output_layer_SPACEHEATER)

        h_0_input_layer_VENTILATION = np.zeros((self.kwargs["n_lstm_layers"][3],1,self.kwargs["n_lstm_hidden"][3]), dtype=np.float32)
        c_0_input_layer_VENTILATION = np.zeros((self.kwargs["n_lstm_layers"][3],1,self.kwargs["n_lstm_hidden"][3]), dtype=np.float32)
        h_0_output_layer_VENTILATION = np.zeros((1,1,self.kwargs["n_output"]), dtype=np.float32)
        c_0_output_layer_VENTILATION = np.zeros((1,1,self.kwargs["n_output"]), dtype=np.float32)
        hidden_state_input_VENTILATION = (h_0_input_layer_VENTILATION,c_0_input_layer_VENTILATION)
        hidden_state_output_VENTILATION = (h_0_output_layer_VENTILATION,c_0_output_layer_VENTILATION)



        hidden_state = (hidden_state_input_OUTDOORTEMPERATURE,
                            hidden_state_output_OUTDOORTEMPERATURE,
                            hidden_state_input_RADIATION,
                            hidden_state_output_RADIATION,
                            hidden_state_input_SPACEHEATER,
                            hidden_state_output_SPACEHEATER,
                            hidden_state_input_VENTILATION,
                            hidden_state_output_VENTILATION)

        return hidden_state

    def get_model(self):
        search_path = os.path.join(uppath(os.path.abspath(__file__), 3), "test", "data", "space_models", "test")
        # search_path = "C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/OU44_space_models/rooms_no_time_600k_20n_test_all"
        # search_path = os.path.join(uppath(os.path.abspath(__file__), 3), "test", "data", "space_models", "rooms_notime_noCO2_nolw_500k_20n_test_all_split")
        # search_path = os.path.join(uppath(os.path.abspath(__file__), 3), "test", "data", "space_models", "rooms_notime_noCO2_nolw_1000k_20n_test_all")
        directory = os.fsencode(search_path)
        found_file = False
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.find(self.id.replace("Ø", "OE") + "_Network") != -1 and filename.find(".pt") != -1:
                found_file = True
                break

        if found_file==False:
            raise NoSpaceModelException
        full_path = search_path + "/" + filename
        self.kwargs, state_dict = torch.load(full_path)
        self.model = LSTM(**self.kwargs)
        self.model.load_state_dict(state_dict)#.to(self.device)
        self.model.eval()

        # print(self.model.T_set)
        # print(self.rescale(self.model.T_set, self.OAT_min, self.OAT_max, 0, 1))
        # aa

        
        if self.use_onnx:
            x_OUTDOORTEMPERATURE = torch.zeros((1, 1, 2))
            x_RADIATION = torch.zeros((1, 1, 2))
            x_SPACEHEATER = torch.zeros((1, 1, 2))
            x_VENTILATION = torch.zeros((1, 1, 3))

            input = (x_OUTDOORTEMPERATURE,
                    x_RADIATION,
                    x_SPACEHEATER,
                    x_VENTILATION)
            hidden_state_torch = self.init_torch_hidden_state()
            torch.onnx.export(LSTMColapsed(self.model), self.unpack(input, hidden_state_torch), full_path.replace(".pt", ".onnx"))
            self.onnx_model = onnxruntime.InferenceSession(full_path.replace(".pt", ".onnx"))
            self.hidden_state = self.init_numpy_hidden_state()
        else:
            self.hidden_state = self.init_torch_hidden_state()




    def input_to_numpy(self, input):
        return tuple([tensor.numpy() for tensor in input])

    def hidden_state_to_numpy(self, hidden_state):
        return tuple([(tuple[0],tuple[1]) for tuple in hidden_state])

    def get_model_input(self):
        if self.use_onnx:
            x_OUTDOORTEMPERATURE = np.zeros((1, 1, 2), dtype=np.float32)
            x_RADIATION = np.zeros((1, 1, 2), dtype=np.float32)
            x_SPACEHEATER = np.zeros((1, 1, 2), dtype=np.float32)
            x_VENTILATION = np.zeros((1, 1, 3), dtype=np.float32)
        else:
            x_OUTDOORTEMPERATURE = torch.zeros((1, 1, 2))
            x_RADIATION = torch.zeros((1, 1, 2))
            x_SPACEHEATER = torch.zeros((1, 1, 2))
            x_VENTILATION = torch.zeros((1, 1, 3))

        y_low = 0
        y_high = 1


        x_OUTDOORTEMPERATURE[:,:,0] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, y_low, y_high) #indoor
        x_OUTDOORTEMPERATURE[:,:,1] = self.min_max_norm(self.input["outdoorTemperature"], self.OAT_min, self.OAT_max, y_low, y_high) #outdoor
        x_RADIATION[:,:,0] = self.min_max_norm(self.input["shadePosition"], self.shades_min, self.shades_max, y_low, y_high) #shades
        x_RADIATION[:,:,1] = self.min_max_norm(self.input["shortwaveRadiation"], self.sw_radiation_min, self.sw_radiation_max, y_low, y_high) #SW
        x_SPACEHEATER[:,:,0] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, y_low, y_high) #indoor
        x_SPACEHEATER[:,:,1] = self.min_max_norm(self.input["valvePosition"], self.r_valve_min, self.r_valve_max, y_low, y_high) #valve
        x_VENTILATION[:,:,0] = self.min_max_norm(self.output["indoorTemperature"], self.temperature_min, self.temperature_max, y_low, y_high) #indoor
        x_VENTILATION[:,:,1] = self.min_max_norm(self.input["supplyDamperPosition"], self.v_valve_min, self.v_valve_max, y_low, y_high) #damper
        x_VENTILATION[:,:,2] = self.min_max_norm(self.input["outdoorTemperature"], self.OAT_min, self.OAT_max, y_low, y_high) #outdoor

        input = (x_OUTDOORTEMPERATURE,
                x_RADIATION,
                x_SPACEHEATER,
                x_VENTILATION)

        return input


    def get_temperature(self):
        input = self.get_model_input()
        

        with torch.no_grad():
            if self.use_onnx:
                onnx_output = self.onnx_model.run(None, self.get_input_dict(input, self.hidden_state))
                output, self.hidden_state, x = self.pack(onnx_output)
                output = output[0][0][0]
            else:
                output,self.hidden_state,x = self.model(input,self.hidden_state)
                output = output.detach().cpu().numpy()[0][0][0]

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
        dT = self.rescale(output, y_min, y_max, -1, 1)
        T = self.output["indoorTemperature"] + dT

        return T
    

    def update_output(self):
        M_air = 28.9647 #g/mol
        M_CO2 = 44.01 #g/mol
        K_conversion = M_CO2/M_air*1e-6

        self.output["indoorTemperature"] = self.get_temperature()
        # self.output["indoorCo2Concentration"] = self.output["indoorCo2Concentration"] + (self.input["outdoorCo2Concentration"]*self.input["supplyAirFlowRate"] - self.output["indoorCo2Concentration"]*self.input["returnAirFlowRate"] + self.input["numberOfPeople"]*self.input["generationCo2Concentration"])*self.timeStep/self.airMass
        self.output["indoorCo2Concentration"] = (self.airMass*self.output["indoorCo2Concentration"] + 
                                                self.input["outdoorCo2Concentration"]*(self.input["supplyAirFlowRate"] + self.input["infiltration"])*self.timeStep + 
                                                self.input["generationCo2Concentration"]*self.input["numberOfPeople"]*self.timeStep/K_conversion)/(self.airMass + (self.input["returnAirFlowRate"]+self.input["infiltration"])*self.timeStep)

        if self.first_time_step == True:
            self.first_time_step = False

        self.time += datetime.timedelta(seconds = self.timeStep)