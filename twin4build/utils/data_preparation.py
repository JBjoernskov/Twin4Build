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
    bool_vec = np.isclose(vec[:-1], vec[1:], rtol=1e-05, atol=1e-08, equal_nan=True)

    if np.mean(bool_vec)>frac_limit:
        print("Bad data quality. Most of data contains NaN values.")

def sample_data(data, stepSize, start_time, end_time):
    """
    data: numpy array with size (n_measurements, 2). Dimension [:,0] are the epoch timestamps while [:,1] are the measurements/values. 
    stepSize: step size in seconds
    start_time: start date for the output sampled lists 
    start_time: end date for the output sampled lists 
    """
    
    constructed_value_list = None
    constructed_time_list = None
    got_data = False

    constructed_time_list = np.array([start_time + datetime.timedelta(seconds=dt) for dt in range(0, int((end_time-start_time).total_seconds()),stepSize)])
    constructed_time_list_timestamp = np.array([el.timestamp() for el in constructed_time_list])
    constructed_value_list = np.zeros(constructed_time_list.shape)
    constructed_value_list[:] = np.nan
        
    print(constructed_time_list_timestamp)
    print(data[:,0])
    idx_vec,dt_vec = _find_last(constructed_time_list_timestamp,data[:,0]) ###
    temp_value_vec = data[idx_vec,1] ###
    # isnan_vec = np.isnan(constructed_value_list)
    constructed_value_list = temp_value_vec

    print(constructed_value_list.shape)
    print(constructed_value_list)
    got_data = _validate_data_quality(constructed_value_list)

    return constructed_time_list,constructed_value_list,got_data

