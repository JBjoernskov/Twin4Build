# from .building_space import BuildingSpace
import twin4build.saref4bldg.building_space.building_space as building_space
from twin4build.utils.constants import Constants
import os
import torch
import datetime
import numpy as np
from twin4build.utils.uppath import uppath
from typing import List, Tuple
from torch import Tensor
import copy
import onnxruntime
onnxruntime.set_default_logger_severity(3)
onnxruntime.set_default_logger_severity(3)

class LSTMColapsed(torch.nn.Module):
    """
    onnx models only takes flat inputs, i.e. no nested tuples, lists etc.
    LSTMColapsed therefore acts as a wrapper of the LSTM model.
    """
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

class LSTM_old(torch.nn.Module):
    def __init__(self, 
                 n_input=None, 
                 n_lstm_hidden=None, 
                 n_lstm_layers=None, 
                 n_output=None, 
                 dropout=None,
                 scaling_value_dict=None):

        self.kwargs = {"n_input": n_input,
                        "n_lstm_hidden": n_lstm_hidden,
                        "n_lstm_layers": n_lstm_layers,
                        "n_output": n_output,
                        "dropout": dropout,
                        "scaling_value_dict": scaling_value_dict}
        
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

        self.lstm_input_SPACEHEATER = torch.nn.LSTM(3, self.n_lstm_hidden_SPACEHEATER, self.n_lstm_layers_SPACEHEATER, batch_first=True, dropout=self.dropout, bias=False)
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

        # x_damper = torch.unsqueeze(x_VENTILATION[:,:,1],2)
        # x_VENTILATION,hidden_state_input_VENTILATION = self.lstm_input_VENTILATION(x_VENTILATION,hidden_state_input_VENTILATION)
        # x_VENTILATION,hidden_state_output_VENTILATION = self.lstm_output_VENTILATION(x_VENTILATION,hidden_state_output_VENTILATION)
        # x_VENTILATION = x_VENTILATION*x_damper

        d1,d2,d3 = x_VENTILATION.size()
        x_VENTILATION = x_VENTILATION.reshape(d1*d2,d3)
        x_indoor = torch.unsqueeze(x_VENTILATION[:,0],1)
        x_outdoor = torch.unsqueeze(x_VENTILATION[:,2],1)
        x_damper = torch.unsqueeze(x_VENTILATION[:,1],1)
        # T_a_hid = self.relu(self.linear_T_o_hid(x_outdoor))
        # T_a = self.linear_T_o_out(T_a_hid)
        T_a = self.relu(x_outdoor-self.T_set)+self.T_set
        x_VENTILATION = (T_a-self.linear_T_z(x_indoor))*self.linear_u(x_damper)
        x_VENTILATION = x_VENTILATION.reshape(d1,d2,1)

        # # PLOT SUPPLY AIR TEMPERTAURE AS FUNCTION OF OUTDOOR TEMPERATURE
        # t_out_min = torch.Tensor([-6.87]) 
        # t_out_max = torch.Tensor([31.14])
        # low = torch.Tensor([0]) 
        # high = torch.Tensor([1])
        # x = torch.unsqueeze(torch.linspace(start=t_out_min[0], end=t_out_max[0], steps=50),1)
        # x_outdoor = (x-t_out_min)/(t_out_max-t_out_min)*(high-low) + low
        # # T_a_hid = self.relu(self.linear_T_o_hid(x_outdoor))
        # # T_a = self.linear_T_o_out(T_a_hid)
        # T_a = self.relu(x_outdoor-self.T_set)+self.T_set
        # print((self.T_set-low)/(high-low)*(t_out_max-t_out_min) + t_out_min)
        # print(self.linear_T_z.bias.data)
        # print(self.linear_T_z.weight.data)
        # y = (T_a-self.linear_T_z.bias.data)/self.linear_T_z.weight.data
        # t_min = torch.Tensor([18.299999237060547])
        # t_max = torch.Tensor([31.5])
        # y_rescaled =  (y-low)/(high-low)*(t_max-t_min) + t_min
        # import matplotlib.pyplot as plt
        # plt.plot(x,y_rescaled)
        # plt.show()
        

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

class LSTM(torch.nn.Module):

    def __init__(self, 
                 n_input=None, 
                 n_lstm_hidden=None, 
                 n_lstm_layers=None, 
                 n_output=None, 
                 dropout=None,
                 scaling_value_dict=None):

        self.kwargs = {"n_input": n_input,
                        "n_lstm_hidden": n_lstm_hidden,
                        "n_lstm_layers": n_lstm_layers,
                        "n_output": n_output,
                        "dropout": dropout,
                        "scaling_value_dict": scaling_value_dict}
        
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
        self.lstm_input_RADIATION = torch.nn.LSTM(1, self.n_lstm_hidden_RADIATION, self.n_lstm_layers_RADIATION, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_RADIATION = torch.nn.LSTM(self.n_lstm_hidden_RADIATION, self.n_output, 1, batch_first=True, bias=False)

        self.lstm_input_SPACEHEATER = torch.nn.LSTM(3, self.n_lstm_hidden_SPACEHEATER, self.n_lstm_layers_SPACEHEATER, batch_first=True, dropout=self.dropout, bias=False)
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
        x_OUTDOORTEMPERATURE,hidden_state_output_OUTDOORTEMPERATURE = self.lstm_output_OUTDOORTEMPERATURE(x_OUTDOORTEMPERATURE,hidden_state_output_OUTDOORTEMPERATURE)

        x_RADIATION,hidden_state_input_RADIATION = self.lstm_input_RADIATION(x_RADIATION,hidden_state_input_RADIATION)
        x_RADIATION,hidden_state_output_RADIATION = self.lstm_output_RADIATION(x_RADIATION,hidden_state_output_RADIATION)

        x_SPACEHEATER,hidden_state_input_SPACEHEATER = self.lstm_input_SPACEHEATER(x_SPACEHEATER,hidden_state_input_SPACEHEATER)
        x_SPACEHEATER,hidden_state_output_SPACEHEATER = self.lstm_output_SPACEHEATER(x_SPACEHEATER,hidden_state_output_SPACEHEATER)

        x_VENTILATION,hidden_state_input_VENTILATION = self.lstm_input_VENTILATION(x_VENTILATION,hidden_state_input_VENTILATION)
        x_VENTILATION,hidden_state_output_VENTILATION = self.lstm_output_VENTILATION(x_VENTILATION,hidden_state_output_VENTILATION)


        # d1,d2,d3 = x_VENTILATION.size()
        # x_VENTILATION = x_VENTILATION.reshape(d1*d2,d3)
        # x_indoor = torch.unsqueeze(x_VENTILATION[:,0],1)
        # x_damper = torch.unsqueeze(x_VENTILATION[:,1],1)
        # x_supply_air_temperature = torch.unsqueeze(x_VENTILATION[:,2],1)
        # x_VENTILATION = (x_supply_air_temperature-self.linear_T_z(x_indoor))*self.linear_u(x_damper)
        # x_VENTILATION = x_VENTILATION.reshape(d1,d2,1)

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
                airVolume=None,
                **kwargs):
        super().__init__(**kwargs)

        self.densityAir = Constants.density["air"] ###
        self.airVolume = airVolume ###
        self.airMass = self.airVolume*self.densityAir ###

        self.x_list = []
        self.input_OUTDOORTEMPERATURE = []
        self.input_RADIATION = []
        self.input_SPACEHEATER = []
        self.input_VENTILATION = []

        self.input = {'supplyAirFlowRate': None, 
                    'supplyDamperPosition': None, 
                    'returnAirFlowRate': None, 
                    'exhaustDamperPosition': None, 
                    'valvePosition': None, 
                    'shadePosition': None, 
                    'supplyAirTemperature': None, 
                    'supplyWaterTemperature': None, 
                    'globalIrradiation': None, 
                    'outdoorTemperature': None, 
                    'numberOfPeople': None}
        self.output = {"indoorTemperature": None, "indoorCo2Concentration": None}


        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")
        print("Using device: "+ str(self.device))
        self.use_onnx = True

    def _rescale(self,y,y_min,y_max,low,high):
        y = (y-low)/(high-low)*(y_max-y_min) + y_min
        return y

    def _min_max_norm(self,y,y_min,y_max,low,high):
        y = (y-y_min)/(y_max-y_min)*(high-low) + low
        return y

    def _unpack_dict(self, dict_):
        dict_

    def _unpack(self, input, hidden_state):
        unpacked = [tensor for tensor in input]
        unpacked.extend([i for tuple in hidden_state for i in tuple])
        return tuple(unpacked)

    def _get_input_dict(self, input, hidden_state):
        unpacked = self._unpack(input, hidden_state)
        input_dict = {obj.name: tensor for obj, tensor in zip(self.onnx_model.get_inputs(), unpacked)}
        return input_dict

    def _pack(self, list_):
        output = list_[0]
        hidden_state_flat = list_[1:-4]
        hidden_state = [(i,j) for i,j in zip(hidden_state_flat[0::2], hidden_state_flat[1::2])]
        x = list_[-4:]
        return output, hidden_state, x

    def _init_torch_hidden_state(self):

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


    def _init_numpy_hidden_state(self):
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


    def _get_model(self):
        search_path = os.path.join(uppath(os.path.abspath(__file__), 3), "test", "data", "space_models", "BMS_data")
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

        
        if self.use_onnx:
            # x_OUTDOORTEMPERATURE = torch.zeros((1, 1, 2))
            # x_RADIATION = torch.zeros((1, 1, 2))
            # x_SPACEHEATER = torch.zeros((1, 1, 2))
            # x_VENTILATION = torch.zeros((1, 1, 3))
            x_OUTDOORTEMPERATURE = torch.zeros((1, 1, 2))
            x_RADIATION = torch.zeros((1, 1, 1))
            x_SPACEHEATER = torch.zeros((1, 1, 3))
            x_VENTILATION = torch.zeros((1, 1, 3))

            input = (x_OUTDOORTEMPERATURE,
                    x_RADIATION,
                    x_SPACEHEATER,
                    x_VENTILATION)
            hidden_state_torch = self._init_torch_hidden_state()
            torch.onnx.export(LSTMColapsed(self.model), self._unpack(input, hidden_state_torch), full_path.replace(".pt", ".onnx"))
            self.onnx_model = onnxruntime.InferenceSession(full_path.replace(".pt", ".onnx"))
            self.hidden_state = self._init_numpy_hidden_state()
        else:
            self.hidden_state = self._init_torch_hidden_state()




    def _input_to_numpy(self, input):
        return tuple([tensor.numpy() for tensor in input])

    def _hidden_state_to_numpy(self, hidden_state):
        return tuple([(tuple[0],tuple[1]) for tuple in hidden_state])

    def _get_model_input(self):
        if self.use_onnx:
            x_OUTDOORTEMPERATURE = np.zeros((1, 1, 2), dtype=np.float32)
            x_RADIATION = np.zeros((1, 1, 1), dtype=np.float32)
            x_SPACEHEATER = np.zeros((1, 1, 3), dtype=np.float32)
            x_VENTILATION = np.zeros((1, 1, 3), dtype=np.float32)
        else:
            x_OUTDOORTEMPERATURE = torch.zeros((1, 1, 2))
            x_RADIATION = torch.zeros((1, 1, 1))
            x_SPACEHEATER = torch.zeros((1, 1, 3))
            x_VENTILATION = torch.zeros((1, 1, 3))

        y_low = 0
        y_high = 1


        # x_OUTDOORTEMPERATURE[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        # x_OUTDOORTEMPERATURE[:,:,1] = self._min_max_norm(self.input["outdoorTemperature"], self.model.kwargs["scaling_value_dict"]["outdoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["outdoorTemperature"]["max"], y_low, y_high) #outdoor
        # x_RADIATION[:,:,0] = self._min_max_norm(self.input["shadePosition"], self.model.kwargs["scaling_value_dict"]["shadePosition"]["min"], self.model.kwargs["scaling_value_dict"]["shadePosition"]["max"], y_low, y_high) #shades
        # x_RADIATION[:,:,1] = self._min_max_norm(self.input["globalIrradiation"], self.model.kwargs["scaling_value_dict"]["globalIrradiation"]["min"], self.model.kwargs["scaling_value_dict"]["globalIrradiation"]["max"], y_low, y_high) #SW
        # x_SPACEHEATER[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        # x_SPACEHEATER[:,:,1] = self._min_max_norm(self.input["valvePosition"], self.model.kwargs["scaling_value_dict"]["valvePosition"]["min"], self.model.kwargs["scaling_value_dict"]["valvePosition"]["max"], y_low, y_high) #valve
        # x_VENTILATION[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        # x_VENTILATION[:,:,1] = self._min_max_norm(self.input["damperPosition"], self.model.kwargs["scaling_value_dict"]["damperPosition"]["min"], self.model.kwargs["scaling_value_dict"]["damperPosition"]["max"], y_low, y_high) #damper
        # x_VENTILATION[:,:,2] = self._min_max_norm(self.input["outdoorTemperature"], self.model.kwargs["scaling_value_dict"]["outdoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["outdoorTemperature"]["max"], y_low, y_high) #outdoor

        x_OUTDOORTEMPERATURE[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        x_OUTDOORTEMPERATURE[:,:,1] = self._min_max_norm(self.input["outdoorTemperature"], self.model.kwargs["scaling_value_dict"]["outdoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["outdoorTemperature"]["max"], y_low, y_high) #outdoor
        x_RADIATION[:,:,0] = self._min_max_norm(self.input["globalIrradiation"], self.model.kwargs["scaling_value_dict"]["globalIrradiation"]["min"], self.model.kwargs["scaling_value_dict"]["globalIrradiation"]["max"], y_low, y_high) #shades
        x_SPACEHEATER[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        x_SPACEHEATER[:,:,1] = self.input["valvePosition"]#self._min_max_norm(self.input["valvePosition"], self.model.kwargs["scaling_value_dict"]["radiatorValvePosition"]["min"], self.model.kwargs["scaling_value_dict"]["radiatorValvePosition"]["max"], y_low, y_high) #valve
        x_SPACEHEATER[:,:,2] = self._min_max_norm(self.input["supplyWaterTemperature"], self.model.kwargs["scaling_value_dict"]["supplyWaterTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["supplyWaterTemperature"]["max"], y_low, y_high) #valve
        # x_SPACEHEATER[:,:,2] = self._min_max_norm(70, self.model.kwargs["scaling_value_dict"]["supplyWaterTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["supplyWaterTemperature"]["max"], y_low, y_high) #valve
        x_VENTILATION[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        x_VENTILATION[:,:,1] = self.input["supplyDamperPosition"]#self._min_max_norm(self.input["damperPosition"], self.model.kwargs["scaling_value_dict"]["damperPosition"]["min"], self.model.kwargs["scaling_value_dict"]["damperPosition"]["max"], y_low, y_high) #damper
        # x_VENTILATION[:,:,2] = self._min_max_norm(self.input["supplyAirTemperature"], self.model.kwargs["scaling_value_dict"]["supplyAirTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["supplyAirTemperature"]["max"], y_low, y_high) #outdoor
        x_VENTILATION[:,:,2] = self._min_max_norm(21, self.model.kwargs["scaling_value_dict"]["supplyAirTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["supplyAirTemperature"]["max"], y_low, y_high) #outdoor

        # x_OUTDOORTEMPERATURE[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        # x_OUTDOORTEMPERATURE[:,:,1] = self._min_max_norm(self.input["outdoorTemperature"], self.model.kwargs["scaling_value_dict"]["outdoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["outdoorTemperature"]["max"], y_low, y_high) #outdoor
        # x_RADIATION[:,:,0] = self._min_max_norm(self.input["globalIrradiation"], self.model.kwargs["scaling_value_dict"]["globalIrradiation"]["min"], self.model.kwargs["scaling_value_dict"]["globalIrradiation"]["max"], y_low, y_high) #shades
        # x_SPACEHEATER[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        # x_SPACEHEATER[:,:,1] = self.input["valvePosition"]#self._min_max_norm(self.input["valvePosition"], self.model.kwargs["scaling_value_dict"]["radiatorValvePosition"]["min"], self.model.kwargs["scaling_value_dict"]["radiatorValvePosition"]["max"], y_low, y_high) #valve
        # x_SPACEHEATER[:,:,2] = self._min_max_norm(40, self.model.kwargs["scaling_value_dict"]["supplyWaterTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["supplyWaterTemperature"]["max"], y_low, y_high) #valve
        # x_SPACEHEATER[:,:,3] = x_SPACEHEATER[:,:,1]*x_SPACEHEATER[:,:,2] #energy
        # x_VENTILATION[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        # x_VENTILATION[:,:,1] = self.input["damperPosition"]#self._min_max_norm(self.input["damperPosition"], self.model.kwargs["scaling_value_dict"]["damperPosition"]["min"], self.model.kwargs["scaling_value_dict"]["damperPosition"]["max"], y_low, y_high) #damper
        # x_VENTILATION[:,:,2] = self._min_max_norm(20, self.model.kwargs["scaling_value_dict"]["supplyAirTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["supplyAirTemperature"]["max"], y_low, y_high) #outdoor
        # x_VENTILATION[:,:,3] = x_VENTILATION[:,:,1]*x_VENTILATION[:,:,2] #energy in 
        # x_VENTILATION[:,:,4] = x_VENTILATION[:,:,1]*x_OUTDOORTEMPERATURE[:,:,0] #energy out


        input = (x_OUTDOORTEMPERATURE,
                x_RADIATION,
                x_SPACEHEATER,
                x_VENTILATION)

        return input


    def _get_temperature(self):
        input = self._get_model_input()
        
        with torch.no_grad():
            if self.use_onnx:
                onnx_output = self.onnx_model.run(None, self._get_input_dict(input, self.hidden_state))
                output, self.hidden_state, x = self._pack(onnx_output)
                output = output[0][0][0]
            else:
                output,self.hidden_state,x = self.model(input,self.hidden_state)
                output = output.detach().cpu().numpy()[0][0][0]

            self.input_OUTDOORTEMPERATURE.append(input[0][0,0,:].tolist()) 
            self.input_RADIATION.append(input[1][0,0,:].tolist()) 
            self.input_SPACEHEATER.append(input[2][0,0,:].tolist()) 
            self.input_VENTILATION.append(input[3][0,0,:].tolist()) 
            self.x_list.append([x[0][0,0,0], x[1][0,0,0], x[2][0,0,0], x[3][0,0,0]])

        y_min = -1 
        y_max = 1 
        dT = self._rescale(output, y_min, y_max, -1, 1)
        T = self.output["indoorTemperature"] + dT

        return T
    
    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        self._get_model()

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        M_air = 28.9647 #g/mol
        M_CO2 = 44.01 #g/mol
        K_conversion = M_CO2/M_air*1e-6
        outdoorCo2Concentration = 400
        infiltration = 0.07
        generationCo2Concentration = 0.000008316

        self.output["indoorTemperature"] = self._get_temperature()
        # self.output["indoorCo2Concentration"] = (self.airMass*self.output["indoorCo2Concentration"] + 
        #                                         outdoorCo2Concentration*(self.input["supplyAirFlowRate"] + infiltration)*stepSize + 
        #                                         generationCo2Concentration*self.input["numberOfPeople"]*stepSize/K_conversion)/(self.airMass + (self.input["returnAirFlowRate"]+infiltration)*stepSize)


