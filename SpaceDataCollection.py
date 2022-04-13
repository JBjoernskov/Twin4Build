
class SpaceDataCollection:
    def __init__(self):
        self.space_name = None
        self.has_sufficient_data = None
        self.lower_limit = {"sw_radiation": 0, 
                            "lw_radiation": 0, 
                            "OAT": -100, 
                            "temperature": 0, 
                            "CO2": 200, 
                            "r_valve": 0, 
                            "v_valve": 0,
                            "shades": 0}
        self.upper_limit = {"sw_radiation": 5000, 
                            "lw_radiation": 5000, 
                            "OAT": 50, 
                            "temperature": 50, 
                            "CO2": 4000, 
                            "r_valve": 1, 
                            "v_valve": 1,
                            "shades": 1}
        self.time = None
        self.raw_data_dict = {}

        self.n_sequence = 144 #72
        self.nan_interpolation_gap_limit = 12
        self.n_sequence_repeat = 32
        self.n_data_sequence_min = 1
        

        self.clean_data_dict = {}
        self.n_data_points = None
        self.n_data_sequence = None
        self.has_sequence_vec = None

        self.adjacent_spaces_no_data_list = []

        self.data_matrix = None
        self.adjacent_space_data_frac = None


        self.data_min_vec = None
        self.data_max_vec = None


    def filter_by_limit(self):

        for property_key in self.raw_data_dict:
            space_data_vec = self.raw_data_dict[property_key]
            property_key_limit = property_key.split("-")[0]
            space_data_vec[space_data_vec<self.lower_limit[property_key_limit]] = np.nan
            space_data_vec[space_data_vec>self.upper_limit[property_key_limit]] = np.nan
        

        N = 8
        dT = self.raw_data_dict["temperature"][1:]-self.raw_data_dict["temperature"][:-1]
        bool_vec_lower = dT<=-1
        idx_vec_lower = np.where(bool_vec_lower)[0]
        bool_vec_higher = dT>=1
        idx_vec_higher = np.where(bool_vec_higher)[0]
        space_data_vec = self.raw_data_dict["temperature"][1:]
        #Remove N time steps before and after index, N/2 before and N/2 after
        
        for i in range(N):
            space_data_vec[idx_vec_lower+i-int(N/2)] = np.nan
            space_data_vec[idx_vec_higher+i-int(N/2)] = np.nan


        



    def nan_helper(self,y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    def interpolate_1D_array(self,y):
        nans, x = self.nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        return y


    def interpolate_nans(self):

        for property_key in self.raw_data_dict:
            space_data_vec = self.raw_data_dict[property_key]

            is_not_nan_vec = np.isnan(space_data_vec)==False

            nan_start_bool_vec = np.zeros((is_not_nan_vec.shape[0]),dtype=np.bool)
            nan_end_bool_vec = np.zeros((is_not_nan_vec.shape[0]),dtype=np.bool)


            nan_start_bool_vec[1:] = np.logical_and(is_not_nan_vec[:-1],is_not_nan_vec[1:]==False)
            nan_start_bool_vec[0] = is_not_nan_vec[0]==False
            nan_end_bool_vec[1:] = np.logical_and(is_not_nan_vec[:-1]==False,is_not_nan_vec[1:])
            nan_end_bool_vec[-1] = is_not_nan_vec[-1]==False

            nan_start_idx_vec = np.where(nan_start_bool_vec)[0]
            nan_end_idx_vec = np.where(nan_end_bool_vec)[0]

            n_nan_group_vec = np.zeros((space_data_vec.shape[0]),dtype=np.int)
            for start_idx,end_idx in zip(nan_start_idx_vec,nan_end_idx_vec):
                # print(start_idx,end_idx)
                n_nan_group_vec[start_idx:end_idx] = end_idx-start_idx

            #Mark indices where the timegap is too large
            violated_gap_bool_vec = n_nan_group_vec>self.nan_interpolation_gap_limit      

            #Interpolate all nan values in data
            space_data_vec = self.interpolate_1D_array(space_data_vec)

            #Set violated timegaps to nan values again
            space_data_vec[violated_gap_bool_vec] = np.nan

            self.raw_data_dict[property_key] = space_data_vec



    def filter_by_repeat_values(self):

        property_key_list = ["temperature", "CO2"]

        for property_key in self.raw_data_dict:
            if property_key in property_key_list:
                space_data_vec = self.raw_data_dict[property_key]

                is_repeat_vec_acc = np.ones((space_data_vec.shape[0]-self.n_sequence_repeat),dtype=np.bool)
                for i in range(self.n_sequence_repeat):
                    
                    if i+1 == self.n_sequence_repeat:
                        is_repeat_vec = np.isclose(space_data_vec[i:-1], space_data_vec[i+1:], rtol=1e-05, atol=1e-08, equal_nan=True)
                        is_repeat_vec_acc = np.logical_and(is_repeat_vec_acc,is_repeat_vec)
                    else:
                        is_repeat_vec = np.isclose(space_data_vec[i:-self.n_sequence_repeat+i], space_data_vec[i+1:-self.n_sequence_repeat+i+1], rtol=1e-05, atol=1e-08, equal_nan=True)
                        is_repeat_vec_acc = np.logical_and(is_repeat_vec_acc,is_repeat_vec)


                is_repeat_vec_acc_idx = np.where(is_repeat_vec_acc)[0]
                for i in range(self.n_sequence_repeat):
                    space_data_vec[is_repeat_vec_acc_idx] = np.nan
                    is_repeat_vec_acc_idx += 1

                self.raw_data_dict[property_key] = space_data_vec
  
    def clean_data(self):


        self.interpolate_nans()



        # fig,ax = plt.subplots()
        # fig.suptitle('before filter repeat', fontsize=16)
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        # ax.plot(self.raw_data_dict["temperature"])


        # fig,ax = plt.subplots()
        # fig.suptitle('before filter repeat CO2', fontsize=16)
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        # ax.plot(self.raw_data_dict["CO2"])

        self.filter_by_repeat_values()


        # fig,ax = plt.subplots()
        # fig.suptitle('after filter repeat', fontsize=16)
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        # ax.plot(self.raw_data_dict["temperature"])


        # fig,ax = plt.subplots()
        # fig.suptitle('after filter repeat CO2', fontsize=16)
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        # ax.plot(self.raw_data_dict["CO2"])

        # plt.show()

        self.filter_by_limit()


    def construct_clean_data_dict(self):
        if self.has_sufficient_data == True:
            # print("---")
            # print(self.space_name)
            self.clean_data()
            is_not_nan_vec_acc = np.ones((self.time.shape[0]),dtype=np.bool)
            for property_key in self.raw_data_dict:
                space_data_vec = self.raw_data_dict[property_key]
            
                is_not_nan_vec = np.isnan(space_data_vec)==False
                is_not_nan_vec_acc = np.logical_and(is_not_nan_vec_acc,is_not_nan_vec)

            is_not_followed_by_nan_vec = is_not_nan_vec_acc[0:-self.n_sequence]
            for i in range(self.n_sequence):
                if i+1 == self.n_sequence:
                    is_not_followed_by_nan_vec = np.logical_and(is_not_followed_by_nan_vec,is_not_nan_vec_acc[i+1:])
                else:
                    is_not_followed_by_nan_vec = np.logical_and(is_not_followed_by_nan_vec,is_not_nan_vec_acc[i+1:-self.n_sequence+i+1])

            self.has_sequence_vec = is_not_followed_by_nan_vec
            self.n_data_sequence = np.sum(self.has_sequence_vec)
            is_not_nan_vec_acc_idx = np.where(self.has_sequence_vec)[0]
            self.clean_data_dict = copy.deepcopy(self.raw_data_dict)
            for property_key in self.clean_data_dict:
                self.clean_data_dict[property_key][:] = np.nan
                space_data_vec = self.clean_data_dict[property_key]
                for i in range(self.n_sequence):
                    space_data_vec[is_not_nan_vec_acc_idx+i] = self.raw_data_dict[property_key][is_not_nan_vec_acc_idx+i]
                self.clean_data_dict[property_key] = space_data_vec


            n_nan_points = np.sum(np.isnan(space_data_vec))
            self.n_data_points = space_data_vec.shape[0]-n_nan_points

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
        
            for i,(y_min,y_max) in enumerate(zip(self.data_min_vec,self.data_max_vec)):
                self.data_matrix[:,i] = min_max_norm(self.data_matrix[:,i],y_min,y_max,-1,1)



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


    def create_data_batches(self):
        file_batch = int(2**11)
        n_batch = 0
        if self.has_sufficient_data == True and self.n_data_sequence>=file_batch:
            save_folder = "C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/OU44_LSTM_data_10min_144seq"
            # true_counter = 0
            late_start_row = -1 #n-1
            skip_row_start = np.array([np.inf])
            skip_row_end = skip_row_start + int(144*5)


            print("Space \"%s\" has %d sequences -> Creating batches..." % (self.space_name, self.n_data_sequence))
            
            n_row = self.time.shape[0]-self.n_sequence
            row_vec = np.arange(n_row)
            np.random.shuffle(row_vec)
            
            days_vec = np.arange(1,32,1)
            # np.random.shuffle(days_vec) ###### Dont mix up sequences - otherwise training data will contain much of validation data
            training_days_list = list(days_vec[0:21]) #0, 5, 10
            validation_days_list = list(days_vec[21:27]) #+21_26 -- 0_26 -- 10
            testing_days_list = list(days_vec[27:31]) #+26_21 -- 26_0 --
            data_type_list = ["training", "validation", "test"]
            days_list = [training_days_list,validation_days_list,testing_days_list]
            print(days_list)
            total_counter = 0
            for i,data_type in enumerate(data_type_list):
                NN_input_flat_lookup_dict = {}
                NN_input_flat = []
                h_0_input = []
                NN_output = []
                true_counter = 0
                
                for row in row_vec:
                    if self.time[row].day in days_list[i]:
                        cond1 = skip_row_start<=row
                        cond2 = skip_row_end>=row

                        cond = np.logical_and(cond1,cond2)

                        if np.any(cond)==False and self.has_sequence_vec[row]:
                        # if self.has_sequence_vec[row]:

                            if true_counter>late_start_row:
                                NN_input_flat_sequence = []
                                NN_output_sequence = []
                                
                                for row_sequence in range(row,row+self.n_sequence):
                                    if row_sequence not in NN_input_flat_lookup_dict:
                                        NN_input_flat_lookup_dict[row_sequence] = list(self.data_matrix[row_sequence])
                                            
                                    NN_input_flat_sequence.append(NN_input_flat_lookup_dict[row_sequence])
                                    NN_output_sequence.append([self.clean_data_dict["temperature"][row_sequence]]) #####################

                                NN_input_flat.append(NN_input_flat_sequence)
                                h_0_input.append([self.clean_data_dict["temperature"][row]])
                                NN_output.append(NN_output_sequence)
                            
                            total_counter += 1
                            true_counter += 1
                            if (true_counter % file_batch == 0 and true_counter>late_start_row+1):

                                save_filename = save_folder + "/" + self.space_name.replace("Ø","OE") + "_" + data_type + "_batch_" + str(true_counter) + ".npz"
                                h_0_input = np.array(h_0_input, dtype=np.float16)
                                NN_output = np.array(NN_output, dtype=np.float16)
                                NN_input_flat = np.array(NN_input_flat, dtype=np.float16)
                                np.savez_compressed(save_filename,NN_input_flat,h_0_input,NN_output)


                                if np.sum(np.isnan(h_0_input))>0:
                                    aaa
                                if np.sum(np.isnan(NN_output))>0:
                                    bbb
                                if np.sum(np.isnan(NN_input_flat))>0:
                                    ccc


                                # print(self.space_name.replace("Ø","OE") + "_" + data_type + "_batch_" + str(true_counter) + ".npz")
                                # print(NN_input_flat.shape)
                                # print(h_0_input.shape)
                                # print(NN_output.shape)

                                NN_input_flat_lookup_dict = {}
                                NN_input_flat = []
                                h_0_input = []
                                NN_output = []

                                n_batch += 1
                    
                    # print("---g-g-")
                    # print(true_counter)
                    # print(n_total)            
                    progressbar.progressbar(total_counter,0,self.n_data_sequence)
            print("")


            idx = list(self.clean_data_dict.keys()).index("temperature")
            temperature_min = self.data_min_vec[idx]
            temperature_max = self.data_max_vec[idx]
            save_filename = save_folder + "/" + self.space_name.replace("Ø","OE") + "_saved_min_max_values" + ".npz"
            np.savez_compressed(save_filename, temperature_min, temperature_max)



            # n_training = 
            # n_validation = 
            # n_test = 
            # for i in range(n_batch):
            #     name_idx = file_batch*i + file_batch
            #     save_filename = save_folder + "/" + self.space_name.replace("Ø","OE") + "_batch_" + str(name_idx) + ".npz"
            
        else:
            print("Space \"%s\" does not have sufficient data -> Skipping..." % self.space_name)