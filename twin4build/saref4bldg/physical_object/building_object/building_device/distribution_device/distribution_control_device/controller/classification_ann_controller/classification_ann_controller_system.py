
"""
ANN-based controller for predicting damper position in ventilated rooms
Inputs:
- Room identifier (0-19)
- CO2 concentration
- Time embeddings (time of the day, time of the year, day of the week)
Output:
- Damper position (0-1)

The controller uses a feedforward neural network with 3 layers and ReLU activation functions.
Input and output sizes are fixed to 12 and 20, respectively.
Network architecture is fixed to 12-50-100-20.
The model weights are loaded from a file named "room_controller_classification_net_room{room_identifier}.pth" in the "saved_networks" folder.
The CO2 data is normalized using the mean and standard deviation of the CO2 data for each room. The mean and standard deviation are hardcoded in the "normalize_co2_data" function.
The time embeddings are extracted from the simulation timestamp and include the time of the day, time of the year, and day of the week.

A road map to extend the idea of an ANN-based controller for building energy systems is expected to be developed in the future.
"""
from twin4build.base import ClassificationAnnController
import sys
import os
import numpy as np
import torch.nn as nn
import torch
import datetime
import calendar
from pathlib import Path

uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 9)
sys.path.append(file_path)

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

logger.info("Controller Model ann classificator File")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'Using device: {device}')


class room_controller_net(nn.Module):
    def __init__(self, input_size, output_size):
        super(room_controller_net, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, output_size)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def load_model(filename, input_size, output_size, device):
    model_state_dict = torch.load(filename, map_location=torch.device('cpu'))
    model = room_controller_net(input_size, output_size).to(device)
    model.load_state_dict(model_state_dict)
    return model

class ClassificationAnnControllerSystem(ClassificationAnnController):
    def __init__(self, 
                room_identifier = None,
                **kwargs):
        super().__init__(**kwargs)
        logger.info("[Controller Model Classification Ann] : Entered in Initialise Funtion")
        self.input = {"actualValue": None}
        self.output = {"inputSignal": None}

        

        self.room_identifier = room_identifier

        self.current_file_path = Path(__file__)
        self.models_path = self.current_file_path.parent / 'saved_networks'
        self.model_filename = os.path.join(self.models_path, f"room_controller_classification_net_room{self.room_identifier}.pth")
        self.input_size =  12
        self.output_size =  20
        self.device =  device
        self.model = load_model(self.model_filename, self.input_size, self.output_size, self.device)

        self._config = {"parameters": ["input_size", "output_size"]}


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
                    stepSize=None):
        pass

    def normalize_co2_data(self, room, co2_concentration):
        """
        room 0: co2_mean: 487.44831648462457 co2_std: 30.339372824816042
        room 1: co2_mean: 517.6011272124435 co2_std: 62.44622430595211
        room 2: co2_mean: 503.275009561795 co2_std: 79.25997188443297
        room 3: co2_mean: 532.9186420552495 co2_std: 77.48523587317794
        room 4: co2_mean: 366.82634861682925 co2_std: 66.57726832988382
        room 5: co2_mean: 502.24242795058024 co2_std: 70.67566502968417
        room 6: co2_mean: 420.70573520926814 co2_std: 56.10775742008361
        room 7: co2_mean: 545.5623315863077 co2_std: 102.70715619859574
        room 8: co2_mean: 457.87479920901734 co2_std: 75.1616908362888
        room 9: co2_mean: 428.0474030227456 co2_std: 15.646712694277939
        room 10: co2_mean: 470.83303275491016 co2_std: 49.51922150786335
        room 11: co2_mean: 491.0285326339487 co2_std: 48.6202179703961
        room 12: co2_mean: 496.9776380926451 co2_std: 53.60251104644996
        room 13: co2_mean: 501.99778467480616 co2_std: 58.151236658135886
        room 14: co2_mean: 485.41123619760845 co2_std: 43.1011251325834
        room 15: co2_mean: 501.5831298984028 co2_std: 51.30480696887379
        room 16: co2_mean: 410.66069431936035 co2_std: 53.173559427276956
        room 17: co2_mean: 532.7565586771808 co2_std: 43.53018793206768
        room 18: co2_mean: 474.61733262387884 co2_std: 27.786654824704172
        room 19: co2_mean: 442.9465092733341 co2_std: 17.03559498220718
        """
        #Make a dictionary with the mean and standard deviation of the CO2 data for each room
        co2_mean_std = {
            0: (487.44831648462457, 30.339372824816042),
            1: (517.6011272124435, 62.44622430595211),
            2: (503.275009561795, 79.25997188443297),
            3: (532.9186420552495, 77.48523587317794),
            4: (366.82634861682925, 66.57726832988382),
            5: (502.24242795058024, 70.67566502968417),
            6: (420.70573520926814, 56.10775742008361),
            7: (545.5623315863077, 102.70715619859574),
            8: (457.87479920901734, 75.1616908362888),
            9: (428.0474030227456, 15.646712694277939),
            10: (470.83303275491016, 49.51922150786335),
            11: (491.0285326339487, 48.6202179703961),
            12: (496.9776380926451, 53.60251104644996),
            13: (501.99778467480616, 58.151236658135886),
            14: (485.41123619760845, 43.1011251325834),
            15: (501.5831298984028, 51.30480696887379),
            16: (410.66069431936035, 53.173559427276956),
            17: (532.7565586771808, 43.53018793206768),
            18: (474.61733262387884, 27.786654824704172),
            19: (442.9465092733341, 17.03559498220718)
        }


        # Find the mean and standard deviation of the CO2 data
        co2_mean = co2_mean_std[room][0]
        co2_std = co2_mean_std[room][1]

        # Normalize the CO2 data
        co2_concentration = (co2_concentration - co2_mean) / (co2_std *  4)

        # if co2_concentration is an array, unpack the co2_concentration
        if isinstance(co2_concentration, np.ndarray):
            co2_concentration = co2_concentration[0]
        
        return co2_concentration

    def time_embedding(self, timestamp: datetime.datetime):
        # Check if it's a leap year
        year_days = 366 if calendar.isleap(timestamp.year) else 365

        # Extract the time of the year, adjusting for leap year
        time_of_year = timestamp.timetuple().tm_yday / year_days
        # Extract the time of the day
        time_of_day = timestamp.hour + timestamp.minute / 60
        # Normalize the time of the day
        time_of_day = time_of_day / 24

        # Create sine and cosine embeddings for the time of the year
        time_of_year_sin = np.sin(2 * np.pi * time_of_year)
        time_of_year_cos = np.cos(2 * np.pi * time_of_year)
        # Create sine and cosine embeddings for the time of the day
        time_of_day_sin = np.sin(2 * np.pi * time_of_day)
        time_of_day_cos = np.cos(2 * np.pi * time_of_day)

        # Create a vector with one-hot encoding for the day of the week
        day_of_week_vector = np.zeros(7)
        day_of_week_vector[timestamp.weekday()] = 1

        # Create a numpy array with the time embeddings
        time_embeddings = np.array([
            time_of_day_sin, time_of_day_cos,
            time_of_year_sin, time_of_year_cos,
            *day_of_week_vector
        ])

        #make sure the time embeddings are an array
        time_embeddings = np.array(time_embeddings)

        return time_embeddings



    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        
        logger.info("[Controller Model Classification Ann] : Entered in Do Step Funtion")
        #The input of the model is a data vector of 5 elements: Month, day, hour, minute, CO2. Extract the time-related elements from the simulation timestamp
        co2_concentration = self.normalize_co2_data(self.room_identifier, self.input["actualValue"])
        time_embeddings = self.time_embedding(dateTime)

        #create a torch tensor with co2_concentration and time_embeddings
        inputs = torch.tensor([co2_concentration, *time_embeddings]).float().to(self.device)


        self.model.eval()  # Set model to evaluation mode
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs, 0)
        #Make the output signal to be in the range of 0-1
        predicted = predicted / 20

        self.output["inputSignal"] = predicted.item()
        


