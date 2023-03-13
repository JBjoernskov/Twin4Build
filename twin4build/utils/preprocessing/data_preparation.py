import numpy as np
import datetime
def _find_last(A,B):
    sorted_idx_left = np.searchsorted(B,A)
    sorted_idx_right = sorted_idx_left-1
    sorted_idx_right[sorted_idx_right==-1] = 0
    final_idx = sorted_idx_right
    dt = A-B[final_idx]
    return final_idx,dt

def _validate_data_quality(vec):
    frac_limit = 0.99999
    bool_vec = np.isnan(vec)
    # bool_vec = np.isclose(vec[:-1], vec[1:], rtol=1e-05, atol=1e-08, equal_nan=True)
    if np.mean(bool_vec)>frac_limit:
        print("Bad data quality. Most of data contains NaN values.")
        got_data = False
    else:
        got_data = True

    return got_data

def sample_data(data, stepSize, start_time, end_time, dt_limit):
    """
    Arguments
    data: numpy array with size (n_measurements, 2). Dimension [:,0] are the epoch timestamps while [:,1] are the measurements/values. 
    stepSize: step size in seconds
    start_time: start date for the output sampled lists 
    end_time: end date for the output sampled lists 

    Returns
    constructed_time_list: list with datetime objects with the specified stepSize frequency and interval
    constructed_value_list: list with corresponding sampled values
    got_data: True or False
    """
    
    constructed_value_list=None
    constructed_time_list=None
    got_data = False

    constructed_time_list = np.array([start_time + datetime.timedelta(seconds=dt) for dt in range(0, int((end_time-start_time).total_seconds()),stepSize)])
    constructed_time_list_timestamp = np.array([el.timestamp() for el in constructed_time_list])
    constructed_value_list = np.zeros(constructed_time_list.shape)
    constructed_value_list[:] = np.nan
    
    #Make sure time stamps are sorted
    sorted_idx = np.argsort(data[:,0])
    data = data[sorted_idx,:]


    #Remove nan entries
    nan_indices = np.isnan(data[:,1])
    data = data[nan_indices==False,:]

        
    idx_vec,dt_vec = _find_last(constructed_time_list_timestamp,data[:,0]) ###
    # isnan_vec = np.isnan(constructed_value_list)
    constructed_value_list = data[idx_vec,1]

    limit_indices = np.abs(dt_vec)>dt_limit
    constructed_value_list[limit_indices] = np.nan

    remove_before = constructed_time_list_timestamp<data[0,0]
    remove_after = constructed_time_list_timestamp>data[-1,0]
    nan_indices = np.logical_or(remove_before,remove_after)
    constructed_value_list[nan_indices] = np.nan


    

    got_data = _validate_data_quality(constructed_value_list)

    return constructed_time_list,constructed_value_list,got_data


