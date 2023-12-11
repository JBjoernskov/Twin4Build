import datetime
from dateutil.tz import tzutc
import numpy as np
import copy
import pickle
import pandas as pd

import os
import sys

uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 4)
sys.path.append(file_path)

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class DataCollection:
    def __init__(self, name, df, nan_interpolation_gap_limit=None, n_sequence=72):
        
        logger.info("[Data Collection Class] : Entered in Initialise Function")

        self.id=None
        self.has_sufficient_data=None
        self.lower_limit = {"globalIrradiation": 0, 
                            "outdoorTemperature": -100, 
                            "indoorTemperature": 0, 
                            "CO2": 200, 
                            "occupancy": 0,
                            "humidity": -9999999,
                            "radiatorValvePosition": 0, 
                            "damperPosition": 0,
                            "shadePosition": 0}
        self.upper_limit = {"globalIrradiation": 5000, 
                            "outdoorTemperature": 50, 
                            "indoorTemperature": 50, 
                            "CO2": 4000, 
                            "occupancy": 999,
                            "humidity": 9999999,
                            "radiatorValvePosition": 100, 
                            "damperPosition": 100,
                            "shadePosition": 100}
        self.name = name
        self.time = df.index.to_numpy()

        # self.time = np.vectorize(lambda data:pd.to_datetime(data)) (self.time)
        self.raw_data_dict = df.to_dict("list")

        for key in self.raw_data_dict.keys():
            self.raw_data_dict[key] = np.array(self.raw_data_dict[key])

        self.n_sequence = n_sequence
        self.nan_interpolation_gap_limit = nan_interpolation_gap_limit
        self.n_data_sequence_min = 1
        self.clean_data_dict = {}
        self.n_data_points=None
        self.n_data_sequence=None
        self.has_sequence_vec=None
        self.property_no_data_list = []
        self.data_matrix=None
        self.adjacent_space_data_frac=None
        self.data_min_vec=None
        self.data_max_vec=None
        self.required_property_key_list = []

        #print("Raw Dict ",self.raw_data_dict,'\n')        

        for property_key in self.raw_data_dict:
            if property_key in self.required_property_key_list:
                if self.raw_data_dict[property_key] is None:
                    self.has_sufficient_data = False
                    break
            else:
                #print("the test data is , " , self.raw_data_dict[property_key] , "\n key" , property_key , self.required_property_key_list) 
                try:
                    if self.raw_data_dict[property_key] is None or np.all(np.isnan(self.raw_data_dict[property_key])):
                        self.property_no_data_list.append(property_key)

                except Exception as e:
                    print("Np NAN " , e)

        for property_key in self.property_no_data_list:
            self.raw_data_dict.pop(property_key)
            
        if self.has_sufficient_data is None and len(self.raw_data_dict)!=0:
            self.has_sufficient_data = True

        self.clean_data_dict = copy.deepcopy(self.raw_data_dict)

        logger.info("[Data Collection Class] : Exited from Initialise Function")

    def get_dataframe(self):
        df = pd.DataFrame(self.clean_data_dict)
        df.insert(0, "time", self.time)
        return df


    def filter_by_limit(self):

        for property_key in self.clean_data_dict:
            data_vec = self.clean_data_dict[property_key]
            before = np.sum(np.isnan(data_vec))

            if property_key in self.lower_limit:
                data_vec[data_vec<self.lower_limit[property_key]] = np.nan
                data_vec[data_vec>self.upper_limit[property_key]] = np.nan
        
            if property_key=="indoorTemperature":
                N = 8
                dT = self.clean_data_dict["indoorTemperature"][1:]-self.clean_data_dict["indoorTemperature"][:-1]
                bool_vec_lower = dT<=-100
                idx_vec_lower = np.where(bool_vec_lower)[0]
                bool_vec_higher = dT>=100
                idx_vec_higher = np.where(bool_vec_higher)[0]
                data_vec = self.clean_data_dict["indoorTemperature"][1:]
                #Remove N time steps before and after index, N/2 before and N/2 after
                
                for i in range(N):
                    data_vec[idx_vec_lower+i-int(N/2)] = np.nan
                    data_vec[idx_vec_higher+i-int(N/2)] = np.nan
            after = np.sum(np.isnan(data_vec))
            logger.info(f"filter_by_limit() for property {property_key} has removed {after-before}")

    def nan_helper(self,y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    def interpolate_1D_array(self,y):
        nans, x = self.nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        return y


    def interpolate_nans(self):

        for property_key in self.clean_data_dict:
            data_vec = self.clean_data_dict[property_key]
            is_not_nan_vec = np.isnan(data_vec)==False

            nan_start_bool_vec = np.zeros((is_not_nan_vec.shape[0]),dtype=bool)
            nan_end_bool_vec = np.zeros((is_not_nan_vec.shape[0]),dtype=bool)


            nan_start_bool_vec[1:] = np.logical_and(is_not_nan_vec[:-1],is_not_nan_vec[1:]==False)
            nan_start_bool_vec[0] = is_not_nan_vec[0]==False
            nan_end_bool_vec[1:] = np.logical_and(is_not_nan_vec[:-1]==False,is_not_nan_vec[1:])
            nan_end_bool_vec[-1] = is_not_nan_vec[-1]==False

            nan_start_idx_vec = np.where(nan_start_bool_vec)[0]
            nan_end_idx_vec = np.where(nan_end_bool_vec)[0]

            n_nan_group_vec = np.zeros((data_vec.shape[0]),dtype=int)
            for start_idx,end_idx in zip(nan_start_idx_vec,nan_end_idx_vec):
                # print(start_idx,end_idx)
                n_nan_group_vec[start_idx:end_idx] = end_idx-start_idx

            #Mark indices where the timegap is too large
            violated_gap_bool_vec = n_nan_group_vec>self.nan_interpolation_gap_limit

            #Interpolate all nan values in data
            data_vec = self.interpolate_1D_array(data_vec)

            #Set violated timegaps to nan values again
            data_vec[violated_gap_bool_vec] = np.nan

            self.clean_data_dict[property_key] = data_vec



    def filter_by_repeat_values(self):

        property_key_list = ["indoorTemperature", "CO2", "radiatorValvePosition", "damperPosition", "shadePosition", "occupancy"]
        only_if_larger_than_0 = [False, False, True, True, True, True]
        n_sequence_repeat_list = [144, 144, 144, 144, 144, 144]
        tol = 0.01

        for property_key, cond, n_sequence_repeat in zip(property_key_list, only_if_larger_than_0, n_sequence_repeat_list):
            if property_key in self.clean_data_dict.keys():
                data_vec = self.clean_data_dict[property_key]

                is_repeat_vec_acc = np.ones((data_vec.shape[0]-n_sequence_repeat),dtype=bool)
                for i in range(n_sequence_repeat):
                    
                    if i+1 == n_sequence_repeat:
                        is_repeat_vec = np.isclose(data_vec[i:-1], data_vec[i+1:], rtol=1e-05, atol=1e-08, equal_nan=True)
                        if cond:
                            is_repeat_vec = np.logical_and(is_repeat_vec, data_vec[i:-1]>tol)
                    else:
                        is_repeat_vec = np.isclose(data_vec[i:-n_sequence_repeat+i], data_vec[i+1:-n_sequence_repeat+i+1], rtol=1e-05, atol=1e-08, equal_nan=True)
                        if cond:
                            is_repeat_vec = np.logical_and(is_repeat_vec, data_vec[i:-n_sequence_repeat+i]>tol)
                    
                    is_repeat_vec_acc = np.logical_and(is_repeat_vec_acc,is_repeat_vec)

                before = np.sum(np.isnan(data_vec))
                is_repeat_vec_acc_idx = np.where(is_repeat_vec_acc)[0]
                for i in range(n_sequence_repeat):
                    data_vec[is_repeat_vec_acc_idx] = np.nan
                    is_repeat_vec_acc_idx += 1
                after = np.sum(np.isnan(data_vec))

                logger.info(f"filter_by_repeat_values() for property {property_key} has removed {after-before}")
                self.clean_data_dict[property_key] = data_vec
  
    def clean_data(self):
        self.interpolate_nans()
        self.filter_by_repeat_values()
        self.filter_by_limit()


    def filter_for_short_sequences(self, required_property_key_list):
        if self.has_sufficient_data == True:
            # print("---")
            # print(self.name)
            adjacent_spaces_no_data_list = []
            self.clean_data()
            is_not_nan_vec_acc = np.ones((self.time.shape[0]),dtype=bool)
            for property_key in self.clean_data_dict:
                data_vec = self.clean_data_dict[property_key]
            
                is_not_nan_vec = np.isnan(data_vec)==False
                
                if property_key not in required_property_key_list:#################################################################################################
                    if (np.sum(is_not_nan_vec_acc)-np.sum(np.logical_and(is_not_nan_vec_acc,is_not_nan_vec)))/np.sum(is_not_nan_vec_acc)>1:#0.05: #0.05
                        adjacent_spaces_no_data_list.append(property_key)
                    else:
                        is_not_nan_vec_acc = np.logical_and(is_not_nan_vec_acc,is_not_nan_vec)
                else:
                    is_not_nan_vec_acc = np.logical_and(is_not_nan_vec_acc,is_not_nan_vec)

            for property_key in adjacent_spaces_no_data_list:
                self.clean_data_dict.pop(property_key)
            self.property_no_data_list.extend(adjacent_spaces_no_data_list)

            is_not_followed_by_nan_vec = is_not_nan_vec_acc[0:-self.n_sequence]
            for i in range(self.n_sequence):
                if i+1 == self.n_sequence:
                    is_not_followed_by_nan_vec = np.logical_and(is_not_followed_by_nan_vec,is_not_nan_vec_acc[i+1:])
                else:
                    is_not_followed_by_nan_vec = np.logical_and(is_not_followed_by_nan_vec,is_not_nan_vec_acc[i+1:-self.n_sequence+i+1])

            self.has_sequence_vec = is_not_followed_by_nan_vec
            self.n_data_sequence = np.sum(self.has_sequence_vec)
            is_not_nan_vec_acc_idx = np.where(self.has_sequence_vec)[0]
            data_vec = np.zeros(self.time.shape)
            for property_key in self.clean_data_dict:
                # self.clean_data_dict[property_key][:] = np.nan
                
                data_vec[:] = np.nan
                for i in range(self.n_sequence):
                    data_vec[is_not_nan_vec_acc_idx+i] = self.clean_data_dict[property_key][is_not_nan_vec_acc_idx+i]
                self.clean_data_dict[property_key] = data_vec.copy()


            n_nan_points = np.sum(np.isnan(data_vec))
            self.n_data_points = data_vec.shape[0]-n_nan_points

            if self.n_data_sequence <= self.n_data_sequence_min:
                self.has_sufficient_data = False

    def construct_clean_data_matrix(self):
        if self.has_sufficient_data == True:
            self.data_matrix = []
            for property_key in self.clean_data_dict:
                data_vec = self.clean_data_dict[property_key]
                self.data_matrix.append(data_vec)
            self.data_matrix = np.array(self.data_matrix).transpose()

            self.data_min_vec = np.nanmin(self.data_matrix, axis=0)
            self.data_max_vec = np.nanmax(self.data_matrix, axis=0)

            self.data_min_vec[-4:] = -1
            self.data_max_vec[-4:] = 1

            logger.info(self.data_min_vec)
            logger.info(self.data_max_vec)
            logger.info(self.clean_data_dict.keys())

            self.data_min_vec
            self.data_max_vec


            low_y = 0
            high_y = 1

        
            for i,(y_min,y_max) in enumerate(zip(self.data_min_vec,self.data_max_vec)):
                self.data_matrix[:,i] = min_max_norm(self.data_matrix[:,i],y_min,y_max,low_y,high_y)

    def create_data_statistics(self):
        if self.has_sufficient_data == True:
            time = self.time[:-self.n_sequence]
            month_vec = np.vectorize(lambda x: x.month)(time[self.has_sequence_vec])

            self.sequence_distribution_list = []
            for month in range(1,13):
                avg = np.sum(month_vec==month)#/month_vec.shape[0]
                self.sequence_distribution_list.append(avg)


            self.sequence_distribution_by_season_vec = np.zeros((4))
            season_month_list = [[12,1,2,],[3,4,5],[6,7,8],[9,10,11]]
            for i,season in enumerate(season_month_list):
                for month in season:
                    avg = np.sum(month_vec==month)#/month_vec.shape[0]
                    self.sequence_distribution_by_season_vec[i] += avg

    def prepare_for_data_batches(self):
        self.filter_for_short_sequences(self.required_property_key_list)
        self.construct_clean_data_matrix()
        self.create_data_statistics()
        self.adjacent_space_data_frac = 1-len(self.property_no_data_list)/len(self.clean_data_dict.keys())

    def create_data_batches(self, save_folder):
        logger.info("[Data Collection Class] : Entered in Data Batches Creation Function")

        if self.has_sufficient_data == True:
            logger.info("Space \"%s\" has %d sequences -> Creating batches..." % (self.name, self.n_data_sequence))
            n_row = self.time.shape[0]-self.n_sequence
            row_vec = np.arange(n_row)
            np.random.shuffle(row_vec)
            days_vec = np.arange(1,32,1)
            training_days_list = list(days_vec[0:21]) #0, 5, 10
            validation_days_list = list(days_vec[21:27]) #+21_26 -- 0_26 -- 10
            testing_days_list = list(days_vec[27:31]) #+26_21 -- 26_0 --
            data_type_list = ["training", "validation", "test"]
            days_list = [training_days_list,validation_days_list,testing_days_list]
            for i,data_type in enumerate(data_type_list):
                NN_input_flat_lookup_dict = {}
                NN_input_flat = []
                NN_output = []
                sample_counter = 0
                
                for row in row_vec:
                    if self.time[row].day in days_list[i] and self.has_sequence_vec[row]:
                        NN_input_flat_sequence = []
                        NN_output_sequence = []
                        for row_sequence in range(row,row+self.n_sequence):
                            if row_sequence not in NN_input_flat_lookup_dict:
                                NN_input_flat_lookup_dict[row_sequence] = list(self.data_matrix[row_sequence])
                                    
                            NN_input_flat_sequence.append(NN_input_flat_lookup_dict[row_sequence])
                            NN_output_sequence.append([self.clean_data_dict["indoorTemperature"][row_sequence]]) #####################

                        NN_input_flat.append(NN_input_flat_sequence)
                        NN_output.append(NN_output_sequence)
                        sample_counter += 1


                save_filename = save_folder + "/" + self.name.replace("Ø","OE") + "_" + data_type + ".npz"
                NN_input_flat = np.array(NN_input_flat)
                NN_output = np.array(NN_output)
                np.savez_compressed(save_filename,NN_input_flat,NN_output)


                if np.sum(np.isnan(NN_output))>0:
                    logger.error(("Generated output batch contains NaN values."))
                    raise Exception("Generated output batch contains NaN values.")
                if np.sum(np.isnan(NN_input_flat))>0:
                    logger.error(("Generated input batch contains NaN values."))
                    raise Exception("Generated input batch contains NaN values.")


                logger.info(self.name.replace("Ø","OE") + "_" + data_type + "_batch_" + str(sample_counter) + ".npz")
                logger.info(NN_input_flat.shape)
                logger.info(NN_output.shape)

                NN_input_flat_lookup_dict = {}
                NN_input_flat = []
                NN_output = []

            save_filename = save_folder + "/" + self.name.replace("Ø","OE") + "_scaling_value_dict" + ".pickle"
            scaling_value_dict = {}
            for key in self.clean_data_dict.keys():
                scaling_value_dict[key] = {"min": None, "max": None}
                idx = list(self.clean_data_dict.keys()).index(key)
                scaling_value_dict[key]["min"] = self.data_min_vec[idx]
                scaling_value_dict[key]["max"] = self.data_max_vec[idx]
            

            filehandler = open(save_filename, 'wb')
            pickle.dump(scaling_value_dict, filehandler)
            filehandler.close()
            
        else:
            logger.info("Space \"%s\" does not have sufficient data -> Skipping..." % self.name)

    def save_building_data_collection_dict(self, save_folder):
        building_data_collection_dict = {self.name: self}
        save_building_data_collection_dict = True
        if save_building_data_collection_dict:
            print("Saving Building Data Collection Dictionary...")
            save_filename = save_folder + "/building_data_collection_dict" + ".pickle"
            filehandler = open(save_filename, 'wb')
            pickle.dump(building_data_collection_dict, filehandler)
            filehandler.close()

    
def min_max_norm(y,y_min,y_max,low,high):
    y = (y-y_min)/(y_max-y_min)*(high-low) + low
    return y 



