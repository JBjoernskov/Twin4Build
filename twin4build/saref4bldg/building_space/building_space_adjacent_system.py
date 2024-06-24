
import sys
import os

uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 4)
sys.path.append(file_path)

import twin4build.saref4bldg.building_space.building_space as building_space
from twin4build.utils.constants import Constants
import torch
# from nalu.layers import NaluLayer
# from nalu.core import NaluCell, NacCell
import datetime
import numpy as np
from twin4build.utils.uppath import uppath
from typing import List, Tuple
from torch import Tensor
import copy
import onnxruntime
from numpy import NaN
# onnxruntime.set_default_logger_severity(3)
# onnxruntime.set_default_logger_severity(3)
from twin4build.logger.Logging import Logging
logger = Logging.get_logger("ai_logfile")

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

        logger.info("[LSTM Colapsed] : Entered in Forward Function")

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

        logger.info("[LSTM Colapsed] : Exited from Forward Function")

        return output, hidden_state, x

# class NaluCell(NaluCell):
#     def __init__(self, in_shape, out_shape):
#         super().__init__(in_shape, out_shape)
#         self.nac = NacCell(in_shape, out_shape)

class LSTM_NALU(torch.nn.Module):
    '''
    The LSTM class is a PyTorch module for a recurrent neural network that uses LSTM (Long Short-Term Memory) cells. 
    Tt takes as input four sequences of features: OUTDOORTEMPERATURE, RADIATION, SPACEHEATER, and VENTILATION    
    '''

    def __init__(self, 
                 n_input=None, 
                 n_hidden=None, 
                 n_layers=None, 
                 n_output=None, 
                 dropout=None,
                 scaling_value_dict=None):
        
        logger.info("[LSTM Colapsed] : Entered in Initialise Function")

        self.kwargs = {"n_input": n_input,
                        "n_hidden": n_hidden,
                        "n_layers": n_layers,
                        "n_output": n_output,
                        "dropout": dropout,
                        "scaling_value_dict": scaling_value_dict}

        (self.n_input_OUTDOORTEMPERATURE,
        self.n_input_RADIATION,
        self.n_input_SPACEHEATER,
        self.n_input_VENTILATION) = n_input
    
        (self.n_hidden_OUTDOORTEMPERATURE,
        self.n_hidden_RADIATION,
        self.n_hidden_SPACEHEATER,
        self.n_hidden_VENTILATION) = n_hidden
        
        (self.n_layers_OUTDOORTEMPERATURE,
        self.n_layers_RADIATION,
        self.n_layers_SPACEHEATER,
        self.n_layers_VENTILATION) = n_layers

        self.n_output = n_output

        self.dropout = dropout

        super(LSTM_NALU, self).__init__()

        self.tanh = torch.nn.Tanh()

        # self.linear_input_OUTDOORTEMPERATURE = torch.nn.Linear(self.n_input_OUTDOORTEMPERATURE, self.n_input_OUTDOORTEMPERATURE)
        # self.nalu_input_OUTDOORTEMPERATURE = NaluCell(in_shape=(self.n_input_OUTDOORTEMPERATURE), out_shape=(self.n_input_OUTDOORTEMPERATURE))
        self.lstm_input_OUTDOORTEMPERATURE = torch.nn.LSTM(self.n_input_OUTDOORTEMPERATURE, self.n_hidden_OUTDOORTEMPERATURE, self.n_layers_OUTDOORTEMPERATURE, batch_first=True, dropout=self.dropout, bias=False)
        # self.nalu_hidden_OUTDOORTEMPERATURE = NaluCell(in_shape=(self.n_hidden_OUTDOORTEMPERATURE), out_shape=(self.n_hidden_OUTDOORTEMPERATURE))
        # self.lstm_hidden_OUTDOORTEMPERATURE = torch.nn.LSTM(self.n_hidden_OUTDOORTEMPERATURE, self.n_hidden_OUTDOORTEMPERATURE, self.n_layers_OUTDOORTEMPERATURE, batch_first=True, dropout=self.dropout, bias=False)
        # self.linear_hidden_OUTDOORTEMPERATURE = torch.nn.Linear(self.n_hidden_OUTDOORTEMPERATURE, self.n_output, bias=False)
        # self.nalu_output_OUTDOORTEMPERATURE = NaluCell(in_shape=(self.n_hidden_OUTDOORTEMPERATURE), out_shape=(self.n_output))
        self.lstm_output_OUTDOORTEMPERATURE = torch.nn.LSTM(self.n_hidden_OUTDOORTEMPERATURE, self.n_output, 1, batch_first=True, bias=False)

        # self.linear_input_RADIATION = torch.nn.Linear(self.n_input_RADIATION, self.n_input_RADIATION)
        # self.nalu_input_RADIATION = NaluCell(in_shape=(self.n_input_RADIATION), out_shape=(self.n_input_RADIATION))
        self.lstm_input_RADIATION = torch.nn.LSTM(self.n_input_RADIATION, self.n_hidden_RADIATION, self.n_layers_RADIATION, batch_first=True, dropout=self.dropout, bias=False)
        # self.nalu_hidden_RADIATION = NaluCell(in_shape=(self.n_hidden_RADIATION), out_shape=(self.n_hidden_RADIATION))
        # self.lstm_hidden_RADIATION = torch.nn.LSTM(self.n_hidden_RADIATION, self.n_hidden_RADIATION, self.n_layers_RADIATION, batch_first=True, dropout=self.dropout, bias=False)
        # self.linear_hidden_RADIATION = torch.nn.Linear(self.n_hidden_RADIATION, self.n_output, bias=False)
        # self.nalu_output_RADIATION = NaluCell(in_shape=(self.n_hidden_RADIATION), out_shape=(self.n_output))
        self.lstm_output_RADIATION = torch.nn.LSTM(self.n_hidden_RADIATION, self.n_output, 1, batch_first=True, bias=False)
        
        # self.linear_input_SPACEHEATER = torch.nn.Linear(self.n_input_SPACEHEATER, self.n_input_SPACEHEATER)
        # self.nalu_input_SPACEHEATER = NaluCell(in_shape=(self.n_input_SPACEHEATER), out_shape=(self.n_input_SPACEHEATER))
        self.lstm_input_SPACEHEATER = torch.nn.LSTM(self.n_input_SPACEHEATER, self.n_hidden_SPACEHEATER, self.n_layers_SPACEHEATER, batch_first=True, dropout=self.dropout, bias=False)
        # self.nalu_hidden_SPACEHEATER = NaluCell(in_shape=(self.n_hidden_SPACEHEATER), out_shape=(self.n_hidden_SPACEHEATER))
        # self.lstm_hidden_SPACEHEATER = torch.nn.LSTM(self.n_hidden_SPACEHEATER, self.n_hidden_SPACEHEATER, self.n_layers_SPACEHEATER, batch_first=True, dropout=self.dropout, bias=False)
        # self.linear_hidden_SPACEHEATER = torch.nn.Linear(self.n_hidden_SPACEHEATER, self.n_output, bias=False)
        # self.nalu_output_SPACEHEATER = NaluCell(in_shape=(self.n_hidden_SPACEHEATER), out_shape=(self.n_output))
        self.lstm_output_SPACEHEATER = torch.nn.LSTM(self.n_hidden_SPACEHEATER, self.n_output, 1, batch_first=True, bias=False)

        # self.linear_input_VENTILATION = torch.nn.Linear(self.n_input_VENTILATION, self.n_input_VENTILATION)
        # self.nalu_input_VENTILATION = NaluCell(in_shape=(self.n_input_VENTILATION), out_shape=(self.n_input_VENTILATION))
        self.lstm_input_VENTILATION = torch.nn.LSTM(self.n_input_VENTILATION, self.n_hidden_VENTILATION, self.n_layers_VENTILATION, batch_first=True, dropout=self.dropout, bias=False)
        # self.nalu_hidden_VENTILATION = NaluCell(in_shape=(self.n_hidden_VENTILATION), out_shape=(self.n_hidden_VENTILATION))
        # self.lstm_hidden_VENTILATION = torch.nn.LSTM(self.n_hidden_VENTILATION, self.n_hidden_VENTILATION, self.n_layers_VENTILATION, batch_first=True, dropout=self.dropout, bias=False)
        # self.linear_hidden_VENTILATION = torch.nn.Linear(self.n_hidden_VENTILATION, self.n_output, bias=False)
        # self.nalu_output_VENTILATION = NaluCell(in_shape=(self.n_hidden_VENTILATION), out_shape=(self.n_output))
        self.lstm_output_VENTILATION = torch.nn.LSTM(self.n_hidden_VENTILATION, self.n_output, 1, batch_first=True, bias=False)

        logger.info("[LSTM Colapsed] : Exited from Initialise Function")


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
        
        '''
        The function "forward" takes two arguments:
            "input": a tuple of four tensors (x_OUTDOORTEMPERATURE, x_RADIATION, x_SPACEHEATER, x_VENTILATION).
            "hidden_state": a tuple of eight tuples, where each tuple contains two tensors representing the input and output hidden state of the corresponding LSTM layer.
        '''

        logger.info("[LSTM Colapsed] : Entered in Forward Function")



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

        # (hidden_state_input_OUTDOORTEMPERATURE,
        # hidden_state_hidden_OUTDOORTEMPERATURE,
        # hidden_state_output_OUTDOORTEMPERATURE,
        # hidden_state_input_RADIATION,
        # hidden_state_hidden_RADIATION,
        # hidden_state_output_RADIATION,
        # hidden_state_input_SPACEHEATER,
        # hidden_state_hidden_SPACEHEATER,
        # hidden_state_output_SPACEHEATER,
        # hidden_state_input_VENTILATION,
        # hidden_state_hidden_VENTILATION,
        # hidden_state_output_VENTILATION) = hidden_state


        # x_OUTDOORTEMPERATURE = self.linear_input_OUTDOORTEMPERATURE(x_OUTDOORTEMPERATURE)
        # x_OUTDOORTEMPERATURE = self.nalu_input_OUTDOORTEMPERATURE(x_OUTDOORTEMPERATURE)
        x_OUTDOORTEMPERATURE,hidden_state_input_OUTDOORTEMPERATURE = self.lstm_input_OUTDOORTEMPERATURE(x_OUTDOORTEMPERATURE,hidden_state_input_OUTDOORTEMPERATURE)
        # x_OUTDOORTEMPERATURE = self.nalu_hidden_OUTDOORTEMPERATURE(x_OUTDOORTEMPERATURE)
        # x_OUTDOORTEMPERATURE,hidden_state_hidden_OUTDOORTEMPERATURE = self.lstm_hidden_OUTDOORTEMPERATURE(x_OUTDOORTEMPERATURE, hidden_state_hidden_OUTDOORTEMPERATURE)
        # x_OUTDOORTEMPERATURE = self.tanh(self.linear_hidden_OUTDOORTEMPERATURE(x_OUTDOORTEMPERATURE))
        # x_OUTDOORTEMPERATURE = self.tanh(self.nalu_output_OUTDOORTEMPERATURE(x_OUTDOORTEMPERATURE))
        x_OUTDOORTEMPERATURE,hidden_state_output_OUTDOORTEMPERATURE = self.lstm_output_OUTDOORTEMPERATURE(x_OUTDOORTEMPERATURE,hidden_state_output_OUTDOORTEMPERATURE)
        
        # x_RADIATION = self.linear_input_RADIATION(x_RADIATION)
        # x_RADIATION = self.nalu_input_RADIATION(x_RADIATION)
        x_RADIATION,hidden_state_input_RADIATION = self.lstm_input_RADIATION(x_RADIATION,hidden_state_input_RADIATION)
        # x_RADIATION = self.nalu_hidden_RADIATION(x_RADIATION)
        # x_RADIATION,hidden_state_hidden_RADIATION = self.lstm_hidden_RADIATION(x_RADIATION, hidden_state_hidden_RADIATION)
        # x_RADIATION = self.tanh(self.linear_hidden_RADIATION(x_RADIATION))
        # x_RADIATION = self.tanh(self.nalu_output_RADIATION(x_RADIATION))
        x_RADIATION,hidden_state_output_RADIATION = self.lstm_output_RADIATION(x_RADIATION,hidden_state_output_RADIATION)

        # x_SPACEHEATER = self.linear_input_SPACEHEATER(x_SPACEHEATER)
        # x_SPACEHEATER = self.nalu_input_SPACEHEATER(x_SPACEHEATER)
        x_SPACEHEATER,hidden_state_input_SPACEHEATER = self.lstm_input_SPACEHEATER(x_SPACEHEATER,hidden_state_input_SPACEHEATER)
        # x_SPACEHEATER = self.nalu_hidden_SPACEHEATER(x_SPACEHEATER)
        # x_SPACEHEATER,hidden_state_hidden_SPACEHEATER = self.lstm_hidden_SPACEHEATER(x_SPACEHEATER, hidden_state_hidden_SPACEHEATER)
        # x_SPACEHEATER = self.tanh(self.linear_hidden_SPACEHEATER(x_SPACEHEATER))
        # x_SPACEHEATER = self.tanh(self.nalu_output_SPACEHEATER(x_SPACEHEATER))
        x_SPACEHEATER,hidden_state_output_SPACEHEATER = self.lstm_output_SPACEHEATER(x_SPACEHEATER,hidden_state_output_SPACEHEATER)

        # x_VENTILATION = self.linear_input_VENTILATION(x_VENTILATION)
        # x_VENTILATION = self.nalu_input_VENTILATION(x_VENTILATION)
        x_VENTILATION,hidden_state_input_VENTILATION = self.lstm_input_VENTILATION(x_VENTILATION,hidden_state_input_VENTILATION)
        # x_VENTILATION = self.nalu_hidden_VENTILATION(x_VENTILATION)
        # x_VENTILATION,hidden_state_hidden_VENTILATION = self.lstm_hidden_VENTILATION(x_VENTILATION, hidden_state_hidden_VENTILATION)
        # x_VENTILATION = self.tanh(self.linear_hidden_VENTILATION(x_VENTILATION))
        # x_VENTILATION = self.tanh(self.nalu_output_VENTILATION(x_VENTILATION))
        x_VENTILATION,hidden_state_output_VENTILATION = self.lstm_output_VENTILATION(x_VENTILATION,hidden_state_output_VENTILATION)

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
        # hidden_state = (hidden_state_input_OUTDOORTEMPERATURE,
        #                 hidden_state_hidden_OUTDOORTEMPERATURE,
        #                 hidden_state_output_OUTDOORTEMPERATURE,
        #                 hidden_state_input_RADIATION,
        #                 hidden_state_hidden_RADIATION,
        #                 hidden_state_output_RADIATION,
        #                 hidden_state_input_SPACEHEATER,
        #                 hidden_state_hidden_SPACEHEATER,
        #                 hidden_state_output_SPACEHEATER,
        #                 hidden_state_input_VENTILATION,
        #                 hidden_state_hidden_VENTILATION,
        #                 hidden_state_output_VENTILATION)
        
        logger.info("[LSTM Colapsed] : Exited from Initialise Function")


        return y,hidden_state,x

class LSTM(torch.nn.Module):

    '''
    The LSTM class is a PyTorch module for a recurrent neural network that uses LSTM (Long Short-Term Memory) cells. 
    Tt takes as input four sequences of features: OUTDOORTEMPERATURE, RADIATION, SPACEHEATER, and VENTILATION    
    '''

    def __init__(self, 
                 n_input=None, 
                 n_hidden=None, 
                 n_layers=None, 
                 n_output=None, 
                 dropout=None,
                 scaling_value_dict=None):
        
        logger.info("[LSTM Colapsed] : Entered in Initialise Function")

        self.kwargs = {"n_input": n_input,
                        "n_hidden": n_hidden,
                        "n_layers": n_layers,
                        "n_output": n_output,
                        "dropout": dropout,
                        "scaling_value_dict": scaling_value_dict}

        (self.n_input_OUTDOORTEMPERATURE,
        self.n_input_RADIATION,
        self.n_input_SPACEHEATER,
        self.n_input_VENTILATION) = n_input
    
        (self.n_hidden_OUTDOORTEMPERATURE,
        self.n_hidden_RADIATION,
        self.n_hidden_SPACEHEATER,
        self.n_hidden_VENTILATION) = n_hidden
        
        (self.n_layers_OUTDOORTEMPERATURE,
        self.n_layers_RADIATION,
        self.n_layers_SPACEHEATER,
        self.n_layers_VENTILATION) = n_layers

        self.n_output = n_output

        self.dropout = dropout

        super(LSTM, self).__init__()
        self.lstm_input_OUTDOORTEMPERATURE = torch.nn.LSTM(self.n_input_OUTDOORTEMPERATURE, self.n_hidden_OUTDOORTEMPERATURE, self.n_layers_OUTDOORTEMPERATURE, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_OUTDOORTEMPERATURE = torch.nn.LSTM(self.n_hidden_OUTDOORTEMPERATURE, self.n_output, 1, batch_first=True, bias=False)

        self.lstm_input_RADIATION = torch.nn.LSTM(self.n_input_RADIATION, self.n_hidden_RADIATION, self.n_layers_RADIATION, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_RADIATION = torch.nn.LSTM(self.n_hidden_RADIATION, self.n_output, 1, batch_first=True, bias=False)

        self.lstm_input_SPACEHEATER = torch.nn.LSTM(self.n_input_SPACEHEATER, self.n_hidden_SPACEHEATER, self.n_layers_SPACEHEATER, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_SPACEHEATER = torch.nn.LSTM(self.n_hidden_SPACEHEATER, self.n_output, 1, batch_first=True, bias=False)

        self.lstm_input_VENTILATION = torch.nn.LSTM(self.n_input_VENTILATION, self.n_hidden_VENTILATION, self.n_layers_VENTILATION, batch_first=True, dropout=self.dropout, bias=False)
        self.lstm_output_VENTILATION = torch.nn.LSTM(self.n_hidden_VENTILATION, self.n_output, 1, batch_first=True, bias=False)

        logger.info("[LSTM Colapsed] : Exited from Initialise Function")


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
        
        '''
        The function "forward" takes two arguments:
            "input": a tuple of four tensors (x_OUTDOORTEMPERATURE, x_RADIATION, x_SPACEHEATER, x_VENTILATION).
            "hidden_state": a tuple of eight tuples, where each tuple contains two tensors representing the input and output hidden state of the corresponding LSTM layer.
                    
        '''

        logger.info("[LSTM Colapsed] : Entered in Forward Function")



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
        
        logger.info("[LSTM Colapsed] : Exited from Initialise Function")


        return y,hidden_state,x



class NoSpaceModelException(Exception):
    def __init__(self, message="No fitting space model"):
        self.message = message
        super().__init__(self.message)


class BuildingSpaceSystem(building_space.BuildingSpace):

    '''
        This is a Python class named BuildingSpaceSystem that inherits from another class called BuildingSpace. 
        The constructor of this class initializes some variables, creates empty lists, and sets some dictionaries
        for input and output. It also determines the device to be used for running the code, which is the CPU in this case. 
        Additionally, it sets a boolean variable use_onnx to True.
    '''


    def __init__(self,
                airVolume=None,
                outdoorCo2Concentration=500,
                infiltration=0.005,
                generationCo2Concentration=0.0042/1000*1.225,
                search_folder=None,
                **kwargs):
        super().__init__(**kwargs)
        logger.info("[BuildingSpaceSystem] : Entered in Initialise Function")
        self.densityAir = Constants.density["air"] ###
        self.airVolume = airVolume ###
        self.airMass = self.airVolume*self.densityAir ###
        self.outdoorCo2Concentration = outdoorCo2Concentration
        self.infiltration = infiltration
        self.generationCo2Concentration = generationCo2Concentration #kgCO2/s/person

        if search_folder is None:
            self.search_folder = os.path.join(uppath(os.path.abspath(__file__), 3), "saref4bldg", "building_space")
        else:
            self.search_folder = search_folder

        self.densityAir = Constants.density["air"] ###
        self.airVolume = airVolume ###
        self.airMass = self.airVolume*self.densityAir ###

        self.x_list = []
        self.input_OUTDOORTEMPERATURE = []
        self.input_RADIATION = []
        self.input_SPACEHEATER = []
        self.input_VENTILATION = []

        self.input = {
            'supplyAirFlowRate': 0,
                    'supplyDamperPosition': 0,
                    'returnAirFlowRate': 0,
                    'returnDamperPosition': 0,
                    'valvePosition': None,
                    # 'shadePosition': None,
                    'supplyAirTemperature': None,
                    'supplyWaterTemperature': None,
                    'globalIrradiation': None,
                    'outdoorTemperature': None,
                    'numberOfPeople': None,}
                    # "adjacentIndoorTemperature_OE20-601b-1": None,
                    # "adjacentIndoorTemperature_OE20-603-1": None,
                    # "adjacentIndoorTemperature_OE20-603c-2": None}
        self.output = {"indoorTemperature": None, "indoorCo2Concentration": None}


        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")
        print("Using device: "+ str(self.device))
        self.use_onnx = True

        logger.info("[BuildingSpaceSystem] : Exited from Initialise Function")
        self._config = {"parameters": ["airMass",
                                        "outdoorCo2Concentration",
                                        "infiltration",
                                        "generationCo2Concentration"]}

    @property
    def config(self):
        return self._config

    def _rescale(self,y,y_min,y_max,low,high):
        '''
        Rescales a given value y from the range [low, high] to the range [y_min, y_max]        
        '''
        logger.info("[BuildingSpaceSystem] : Entered in Rescale Function")
        y = (y-low)/(high-low)*(y_max-y_min) + y_min
        logger.info("[BuildingSpaceSystem] : Exited from Rescale Function")
        return y

    def _min_max_norm(self,y,y_min,y_max,low,high):
        '''
        Performs min-max normalization on a given value y
        '''
        logger.info("[BuildingSpaceSystem] : Entered in Min Max Norm Function")
        y = (y-y_min)/(y_max-y_min)*(high-low) + low
        logger.info("[BuildingSpaceSystem] : Exited from Min Max Norm Function")
        return y

    def _unpack_dict(self, dict_):
        dict_

    def _unpack(self, input, hidden_state):
        logger.info("[BuildingSpaceSystem] : Entered in Unpack Function")
        unpacked = [tensor for tensor in input]
        unpacked.extend([i for tuple in hidden_state for i in tuple])
        logger.info("[BuildingSpaceSystem] : Exited from Unpack Function")

        return tuple(unpacked)

    def _get_input_dict(self, input, hidden_state):
        '''
         Returns a dictionary of input tensors and their corresponding names required by the ONNX model.
        '''
        logger.info("[BuildingSpaceSystem] : Entered in Get input Dict Function")
        unpacked = self._unpack(input, hidden_state)
        input_dict = {obj.name: tensor for obj, tensor in zip(self.onnx_model.get_inputs(), unpacked)}
        logger.info("[BuildingSpaceSystem] : Exxited from Get input Dict Function")
        return input_dict

    def _pack(self, list_):
        '''
        Packs the output tensor, hidden state tensor, and the last four input tensors into a list. 
        '''
        logger.info("[BuildingSpaceSystem] : Entered in Pack Function")
        output = list_[0]
        hidden_state_flat = list_[1:-4]
        hidden_state = [(i,j) for i,j in zip(hidden_state_flat[0::2], hidden_state_flat[1::2])]
        x = list_[-4:]
        logger.info("[BuildingSpaceSystem] : Exited from Pack Function")
        return output, hidden_state, x

    def _init_torch_hidden_state(self):

        '''
            The _init_torch_hidden_state function initializes the hidden state for a LSTM neural network for four input 
            features: outdoor temperature, radiation, space heater, and ventilation. It creates the initial values for the cell
            and hidden states for both input and output layers for each feature, and returns a tuple containing all the hidden states.
        '''
        logger.info("[BuildingSpaceSystem] : Entered in Init torch Hidden State Function")
        h_0_input_layer_OUTDOORTEMPERATURE = torch.zeros((self.kwargs["n_layers"][0],1,self.kwargs["n_hidden"][0]))
        c_0_input_layer_OUTDOORTEMPERATURE = torch.zeros((self.kwargs["n_layers"][0],1,self.kwargs["n_hidden"][0]))
        h_0_output_layer_OUTDOORTEMPERATURE = torch.zeros((1,1,self.kwargs["n_output"]))
        c_0_output_layer_OUTDOORTEMPERATURE = torch.zeros((1,1,self.kwargs["n_output"]))
        hidden_state_input_OUTDOORTEMPERATURE = (h_0_input_layer_OUTDOORTEMPERATURE,c_0_input_layer_OUTDOORTEMPERATURE)
        hidden_state_output_OUTDOORTEMPERATURE = (h_0_output_layer_OUTDOORTEMPERATURE,c_0_output_layer_OUTDOORTEMPERATURE)

        h_0_input_layer_RADIATION = torch.zeros((self.kwargs["n_layers"][1],1,self.kwargs["n_hidden"][1]))
        c_0_input_layer_RADIATION = torch.zeros((self.kwargs["n_layers"][1],1,self.kwargs["n_hidden"][1]))
        h_0_output_layer_RADIATION = torch.zeros((1,1,self.kwargs["n_output"]))
        c_0_output_layer_RADIATION = torch.zeros((1,1,self.kwargs["n_output"]))
        hidden_state_input_RADIATION = (h_0_input_layer_RADIATION,c_0_input_layer_RADIATION)
        hidden_state_output_RADIATION = (h_0_output_layer_RADIATION,c_0_output_layer_RADIATION)

        h_0_input_layer_SPACEHEATER = torch.zeros((self.kwargs["n_layers"][2],1,self.kwargs["n_hidden"][2]))
        c_0_input_layer_SPACEHEATER = torch.zeros((self.kwargs["n_layers"][2],1,self.kwargs["n_hidden"][2]))
        h_0_output_layer_SPACEHEATER = torch.zeros((1,1,self.kwargs["n_output"]))
        c_0_output_layer_SPACEHEATER = torch.zeros((1,1,self.kwargs["n_output"]))
        hidden_state_input_SPACEHEATER = (h_0_input_layer_SPACEHEATER,c_0_input_layer_SPACEHEATER)
        hidden_state_output_SPACEHEATER = (h_0_output_layer_SPACEHEATER,c_0_output_layer_SPACEHEATER)

        h_0_input_layer_VENTILATION = torch.zeros((self.kwargs["n_layers"][3],1,self.kwargs["n_hidden"][3]))
        c_0_input_layer_VENTILATION = torch.zeros((self.kwargs["n_layers"][3],1,self.kwargs["n_hidden"][3]))
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

        logger.info("[BuildingSpaceSystem] : Exited from Init torch Hidden State Function")
        return hidden_state


    def _init_numpy_hidden_state(self):
        '''
            The function _init_numpy_hidden_state takes in keyword arguments 
            that specify the number of layers and number of hidden units for four different input features. 
            It initializes numpy arrays for the hidden states and returns them as a tuple of tuples. 
            The outer tuple contains four inner tuples, each representing the hidden state for one input feature. 
            The inner tuples contain two numpy arrays each, representing the hidden state and cell state for the input 
            and output layers of an LSTM model.            
        '''

        logger.info("[BuildingSpaceSystem] : Entered in Numpy Hidden State Init Function")

        h_0_input_layer_OUTDOORTEMPERATURE = np.zeros((self.kwargs["n_layers"][0],1,self.kwargs["n_hidden"][0]), dtype=np.float32)
        c_0_input_layer_OUTDOORTEMPERATURE = np.zeros((self.kwargs["n_layers"][0],1,self.kwargs["n_hidden"][0]), dtype=np.float32)
        h_0_output_layer_OUTDOORTEMPERATURE = np.zeros((1,1,self.kwargs["n_output"]), dtype=np.float32)
        c_0_output_layer_OUTDOORTEMPERATURE = np.zeros((1,1,self.kwargs["n_output"]), dtype=np.float32)
        hidden_state_input_OUTDOORTEMPERATURE = (h_0_input_layer_OUTDOORTEMPERATURE,c_0_input_layer_OUTDOORTEMPERATURE)
        hidden_state_output_OUTDOORTEMPERATURE = (h_0_output_layer_OUTDOORTEMPERATURE,c_0_output_layer_OUTDOORTEMPERATURE)

        h_0_input_layer_RADIATION = np.zeros((self.kwargs["n_layers"][1],1,self.kwargs["n_hidden"][1]), dtype=np.float32)
        c_0_input_layer_RADIATION = np.zeros((self.kwargs["n_layers"][1],1,self.kwargs["n_hidden"][1]), dtype=np.float32)
        h_0_output_layer_RADIATION = np.zeros((1,1,self.kwargs["n_output"]), dtype=np.float32)
        c_0_output_layer_RADIATION = np.zeros((1,1,self.kwargs["n_output"]), dtype=np.float32)
        hidden_state_input_RADIATION = (h_0_input_layer_RADIATION,c_0_input_layer_RADIATION)
        hidden_state_output_RADIATION = (h_0_output_layer_RADIATION,c_0_output_layer_RADIATION)

        h_0_input_layer_SPACEHEATER = np.zeros((self.kwargs["n_layers"][2],1,self.kwargs["n_hidden"][2]), dtype=np.float32)
        c_0_input_layer_SPACEHEATER = np.zeros((self.kwargs["n_layers"][2],1,self.kwargs["n_hidden"][2]), dtype=np.float32)
        h_0_output_layer_SPACEHEATER = np.zeros((1,1,self.kwargs["n_output"]), dtype=np.float32)
        c_0_output_layer_SPACEHEATER = np.zeros((1,1,self.kwargs["n_output"]), dtype=np.float32)
        hidden_state_input_SPACEHEATER = (h_0_input_layer_SPACEHEATER,c_0_input_layer_SPACEHEATER)
        hidden_state_output_SPACEHEATER = (h_0_output_layer_SPACEHEATER,c_0_output_layer_SPACEHEATER)

        h_0_input_layer_VENTILATION = np.zeros((self.kwargs["n_layers"][3],1,self.kwargs["n_hidden"][3]), dtype=np.float32)
        c_0_input_layer_VENTILATION = np.zeros((self.kwargs["n_layers"][3],1,self.kwargs["n_hidden"][3]), dtype=np.float32)
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
        
        logger.info("[BuildingSpaceSystem] : Exited from Numpy Hidden State Init Function")
        return hidden_state


    def _get_model(self):
        '''
        It searches for a specific file in a directory, loads the file as a PyTorch model, 
        and sets it as an attribute of the class. The method also initializes the hidden state of the model and can export it to an ONNX format if requested.        
        '''

        logger.info("[BuildingSpaceSystem] : Entered in Get Model Function")

        directory = os.fsencode(self.search_folder)
        found_file = False
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            # if filename.find(self.id.replace("Ø", "OE") + "_Network") != -1 and filename.find(".pt") != -1:
            if filename.find("OE20-601b-2_Network") != -1 and filename.find(".pt") != -1:
                found_file = True
                break

        if found_file==False:
            raise NoSpaceModelException
                
        full_path = os.path.join(self.search_folder, filename)
        self.kwargs, state_dict = torch.load(full_path)

        # old_keys = ["n_lstm_hidden", "n_lstm_layers"]
        # new_keys = ['n_hidden','n_layers']
        # for old_key,new_key in zip(old_keys, new_keys):
        #     self.kwargs[new_key] = self.kwargs.pop(old_key)
        # self.kwargs["n_input"] = (2, 1, 2, 2)
        # layers_to_remove = ["linear_u.weight", "linear_T_o_hid.weight", "linear_T_o_hid.bias", "linear_T_o_out.weight", "linear_T_o_out.bias", "linear_T_z.weight", "linear_T_z.bias"]
        # for key in layers_to_remove:
        #     del state_dict[key]

        self.model = LSTM_NALU(**self.kwargs)
        self.model.load_state_dict(state_dict)#.to(self.device)
        self.model.eval()


        # aa
        # torch.save((self.kwargs, self.model.state_dict()), filename)
        
        if self.use_onnx:
            # x_OUTDOORTEMPERATURE = torch.zeros((1, 1, 2))
            # x_RADIATION = torch.zeros((1, 1, 2))
            # x_SPACEHEATER = torch.zeros((1, 1, 2))
            # x_VENTILATION = torch.zeros((1, 1, 3))
            x_OUTDOORTEMPERATURE = torch.zeros((1, 1, self.model.n_input_OUTDOORTEMPERATURE))
            x_RADIATION = torch.zeros((1, 1, self.model.n_input_RADIATION))
            x_SPACEHEATER = torch.zeros((1, 1, self.model.n_input_SPACEHEATER))
            x_VENTILATION = torch.zeros((1, 1, self.model.n_input_VENTILATION))

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

        
        logger.info("[BuildingSpaceSystem] : Exited from Get Model Function")



    def _input_to_numpy(self, input):
        return tuple([tensor.numpy() for tensor in input])

    def _hidden_state_to_numpy(self, hidden_state):
        return tuple([(tuple[0],tuple[1]) for tuple in hidden_state])

    def _get_model_input(self, dateTime):

        '''
        The code is a method that prepares input data for a machine learning model. It takes in a dateTime argument and generates several arrays of input values for the model. The arrays are created using a combination of the input and output data and various normalization techniques.        
        '''

        logger.info("[BuildingSpaceSystem] : Entered in Get Model Input Function")

                
        if self.use_onnx:
            x_OUTDOORTEMPERATURE = np.zeros((1, 1, self.model.n_input_OUTDOORTEMPERATURE), dtype=np.float32)
            x_RADIATION = np.zeros((1, 1, self.model.n_input_RADIATION), dtype=np.float32)
            x_SPACEHEATER = np.zeros((1, 1, self.model.n_input_SPACEHEATER), dtype=np.float32)
            x_VENTILATION = np.zeros((1, 1, self.model.n_input_VENTILATION), dtype=np.float32)
        else:
            x_OUTDOORTEMPERATURE = torch.zeros((1, 1, self.model.n_input_OUTDOORTEMPERATURE))
            x_RADIATION = torch.zeros((1, 1, self.model.n_input_RADIATION))
            x_SPACEHEATER = torch.zeros((1, 1, self.model.n_input_SPACEHEATER))
            x_VENTILATION = torch.zeros((1, 1, self.model.n_input_VENTILATION))

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

        # x_OUTDOORTEMPERATURE[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        # x_OUTDOORTEMPERATURE[:,:,1] = self._min_max_norm(self.input["outdoorTemperature"], self.model.kwargs["scaling_value_dict"]["outdoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["outdoorTemperature"]["max"], y_low, y_high) #outdoor
        # x_OUTDOORTEMPERATURE[:,:,2] = self._min_max_norm(self.input["adjacentIndoorTemperature_OE20-601b-1"], self.model.kwargs["scaling_value_dict"]["adjacentIndoorTemperature_OE20-601b-1"]["min"], self.model.kwargs["scaling_value_dict"]["adjacentIndoorTemperature_OE20-601b-1"]["max"], y_low, y_high) #outdoor
        # x_OUTDOORTEMPERATURE[:,:,3] = self._min_max_norm(self.input["adjacentIndoorTemperature_OE20-603-1"], self.model.kwargs["scaling_value_dict"]["adjacentIndoorTemperature_OE20-603-1"]["min"], self.model.kwargs["scaling_value_dict"]["adjacentIndoorTemperature_OE20-603-1"]["max"], y_low, y_high) #outdoor
        # x_OUTDOORTEMPERATURE[:,:,4] = self._min_max_norm(self.input["adjacentIndoorTemperature_OE20-603c-2"], self.model.kwargs["scaling_value_dict"]["adjacentIndoorTemperature_OE20-603c-2"]["min"], self.model.kwargs["scaling_value_dict"]["adjacentIndoorTemperature_OE20-603c-2"]["max"], y_low, y_high) #outdoor
        # x_RADIATION[:,:,0] = self._min_max_norm(self.input["globalIrradiation"], self.model.kwargs["scaling_value_dict"]["globalIrradiation"]["min"], self.model.kwargs["scaling_value_dict"]["globalIrradiation"]["max"], y_low, y_high) #shades
        # x_SPACEHEATER[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        # x_SPACEHEATER[:,:,1] = self.input["valvePosition"]#self._min_max_norm(self.input["valvePosition"], self.model.kwargs["scaling_value_dict"]["radiatorValvePosition"]["min"], self.model.kwargs["scaling_value_dict"]["radiatorValvePosition"]["max"], y_low, y_high) #valve
        # x_SPACEHEATER[:,:,2] = self._min_max_norm(self.input["supplyWaterTemperature"], self.model.kwargs["scaling_value_dict"]["supplyWaterTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["supplyWaterTemperature"]["max"], y_low, y_high) #valve
        # # x_SPACEHEATER[:,:,2] = self._min_max_norm(70, self.model.kwargs["scaling_value_dict"]["supplyWaterTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["supplyWaterTemperature"]["max"], y_low, y_high) #valve
        # x_VENTILATION[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        # x_VENTILATION[:,:,1] = self.input["supplyDamperPosition"]#self._min_max_norm(self.input["damperPosition"], self.model.kwargs["scaling_value_dict"]["damperPosition"]["min"], self.model.kwargs["scaling_value_dict"]["damperPosition"]["max"], y_low, y_high) #damper
        # # x_VENTILATION[:,:,2] = self._min_max_norm(self.input["supplyAirTemperature"], self.model.kwargs["scaling_value_dict"]["supplyAirTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["supplyAirTemperature"]["max"], y_low, y_high) #outdoor
        # x_VENTILATION[:,:,2] = self._min_max_norm(25, self.model.kwargs["scaling_value_dict"]["supplyAirTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["supplyAirTemperature"]["max"], y_low, y_high) #outdoor
        
        time_of_day = (dateTime.hour*60+dateTime.minute)/(23*60+50)
        time_of_year = ((dateTime.timetuple().tm_yday-1)*24*60+dateTime.hour*60+dateTime.minute)/(364*24*60 + 23*60 + 50)


        #### STANDARD ####
        # x_OUTDOORTEMPERATURE[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        # x_OUTDOORTEMPERATURE[:,:,1] = self._min_max_norm(self.input["outdoorTemperature"], self.model.kwargs["scaling_value_dict"]["outdoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["outdoorTemperature"]["max"], y_low, y_high) #outdoor
        # x_RADIATION[:,:,0] = self._min_max_norm(self.input["globalIrradiation"], self.model.kwargs["scaling_value_dict"]["globalIrradiation"]["min"], self.model.kwargs["scaling_value_dict"]["globalIrradiation"]["max"], y_low, y_high) #shades
        # x_SPACEHEATER[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        # x_SPACEHEATER[:,:,1] = self._min_max_norm(self.input["valvePosition"]*self.input["supplyWaterTemperature"]*100, self.model.kwargs["scaling_value_dict"]["spaceHeaterAddedEnergy"]["min"], self.model.kwargs["scaling_value_dict"]["spaceHeaterAddedEnergy"]["max"], y_low, y_high) #valve
        # x_VENTILATION[:,:,0] = self._min_max_norm(self.input["supplyDamperPosition"]*self.input["supplyAirTemperature"]*100, self.model.kwargs["scaling_value_dict"]["ventilationAddedEnergy"]["min"], self.model.kwargs["scaling_value_dict"]["ventilationAddedEnergy"]["max"], y_low, y_high) #valve
        # x_VENTILATION[:,:,1] = self._min_max_norm(self.input["supplyDamperPosition"]*self.output["indoorTemperature"]*100, self.model.kwargs["scaling_value_dict"]["ventilationRemovedEnergy"]["min"], self.model.kwargs["scaling_value_dict"]["ventilationRemovedEnergy"]["max"], y_low, y_high) #valve


        #### NALU USING INDIVIDUAL POSTITION SIGNALS ####
        x_OUTDOORTEMPERATURE[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        x_OUTDOORTEMPERATURE[:,:,1] = self._min_max_norm(self.input["outdoorTemperature"], self.model.kwargs["scaling_value_dict"]["outdoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["outdoorTemperature"]["max"], y_low, y_high) #outdoor
        x_RADIATION[:,:,0] = self._min_max_norm(self.input["globalIrradiation"], self.model.kwargs["scaling_value_dict"]["globalIrradiation"]["min"], self.model.kwargs["scaling_value_dict"]["globalIrradiation"]["max"], y_low, y_high) #shades
        x_RADIATION[:,:,1] = self._min_max_norm(np.cos(2*np.pi*time_of_day), self.model.kwargs["scaling_value_dict"]["time_of_day_cos"]["min"], self.model.kwargs["scaling_value_dict"]["time_of_day_cos"]["max"], y_low, y_high) #shades ##################
        x_RADIATION[:,:,2] = self._min_max_norm(np.sin(2*np.pi*time_of_day), self.model.kwargs["scaling_value_dict"]["time_of_day_sin"]["min"], self.model.kwargs["scaling_value_dict"]["time_of_day_sin"]["max"], y_low, y_high) #shades ##################
        x_RADIATION[:,:,3] = self._min_max_norm(np.cos(2*np.pi*time_of_year), self.model.kwargs["scaling_value_dict"]["time_of_year_cos"]["min"], self.model.kwargs["scaling_value_dict"]["time_of_year_cos"]["max"], y_low, y_high) #shades ##################
        x_RADIATION[:,:,4] = self._min_max_norm(np.sin(2*np.pi*time_of_year), self.model.kwargs["scaling_value_dict"]["time_of_year_sin"]["min"], self.model.kwargs["scaling_value_dict"]["time_of_year_sin"]["max"], y_low, y_high) #shades ##################
        x_SPACEHEATER[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        x_SPACEHEATER[:,:,1] = self.input["valvePosition"]
        x_SPACEHEATER[:,:,2] = self._min_max_norm(self.input["supplyWaterTemperature"], self.model.kwargs["scaling_value_dict"]["supplyWaterTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["supplyWaterTemperature"]["max"], y_low, y_high) #valve
        x_VENTILATION[:,:,0] = self._min_max_norm(self.output["indoorTemperature"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["indoorTemperature"]["max"], y_low, y_high) #indoor
        x_VENTILATION[:,:,1] = self.input["supplyDamperPosition"]
        x_VENTILATION[:,:,2] = self._min_max_norm(self.input["supplyAirTemperature"], self.model.kwargs["scaling_value_dict"]["supplyAirTemperature"]["min"], self.model.kwargs["scaling_value_dict"]["supplyAirTemperature"]["max"], y_low, y_high) #outdoor


        input = (x_OUTDOORTEMPERATURE,
                x_RADIATION,
                x_SPACEHEATER,
                x_VENTILATION)

        for arr in input:
            arr[np.isnan(arr)] = 0


        logger.info("[BuildingSpaceSystem] : Exited from Get Model Input Function")

        return input


    def _get_temperature(self, dateTime):

        logger.info("[BuildingSpaceSystem] : Entered in Get Temperature Function")

        input = self._get_model_input(dateTime)
        
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

        logger.info("[BuildingSpaceSystem] : Exited from Get Temperature Function")
        return T
    
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
        self._get_model()

        self.input = {
            'supplyAirFlowRate': 0,
            'supplyDamperPosition': 0,
            'returnAirFlowRate': 0,
            'returnDamperPosition': 0,}

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output["indoorTemperature"] = self._get_temperature(dateTime)
        self.output["indoorCo2Concentration"] = (self.airMass*self.output["indoorCo2Concentration"] + 
                                                self.outdoorCo2Concentration*(self.input["supplyAirFlowRate"] + self.infiltration)*stepSize + 
                                                self.generationCo2Concentration*1e+6*self.input["numberOfPeople"]*stepSize)/(self.airMass + (self.input["returnAirFlowRate"]+self.infiltration)*stepSize)
                                                # self.generationCo2Concentration*self.input["numberOfPeople"]*stepSize/self.K_conversion

