import pandas as pd
import os
import sys
import pickle
# if __name__ == '__main__':
#     uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
#     file_path = uppath(os.path.abspath(__file__), 4)
#     sys.path.append(file_path)
#     print(file_path)
# from twin4build.utils.preprocessing.data_sampler import data_sampler
import pandas as pd
import matplotlib.pyplot as plt
# from twin4build.utils.uppath import uppath
import datetime
import dateutil
import numpy as np
import os
import copy
# filepath = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "VE02.xlsx")
# df_VE02 = pd.read_excel(filepath)
# filepath = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "OE20-601b-2.xlsx")
# df_OE20_601b_2 = pd.read_excel(filepath)
# filename = "VE02.pickle"
# filehandler = open(filename, 'wb')
# pickle.dump((df_VE02), filehandler)
# filename = "OE20_601b_2.pickle"
# filehandler = open(filename, 'wb')
# pickle.dump((df_OE20_601b_2), filehandler)
from dateutil import parser

# from twin4build.logger.Logging import Logging
# logger = Logging.get_logger("ai_logfile")


def split_file(filename):
    filehandler = open(filename, 'rb')
    name, file_extension = os.path.splitext(filename)

    #Check if file is cached
    # startPeriod_str = start_time.strftime('%d-%m-%Y %H-%M-%S')
    # endPeriod_str = end_time.strftime('%d-%m-%Y %H-%M-%S')
    # cached_filename = f"name({os.path.basename(name)})_stepSize({str(stepSize)})_startPeriod({startPeriod_str})_endPeriod({endPeriod_str})_cached.pickle"
    # cached_filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "test", "data", "time_series_data", "cached_data", cached_filename)

    if file_extension==".csv":
        df = pd.read_csv(filehandler, low_memory=False, encoding_errors="ignore", delimiter=";")

    elif file_extension==".xlsx":
        df = pd.read_excel(filehandler)

    else:
        # logger.error((f"Invalid file extension: {file_extension}"))
        raise Exception(f"Invalid file extension: {file_extension}")

    # for column in df.columns.to_list()[1:]:
    #     df[column] = pd.to_numeric(df[column], errors='coerce') #Remove string entries

    df_tag_name = df["TagName"]
    df_tag_name = df_tag_name.drop_duplicates()
    df_dict = {}
    for _, val in df_tag_name.items():
        df_ = df[df['TagName'] == val]
        df_dict[val] = df_
    # aa
    # n = df.shape[0]
    # data = np.zeros((n,2))
    # time = np.vectorize(lambda data:datetime.datetime.strptime(data, format)) (df.iloc[:, 0])
    # epoch_timestamp = np.vectorize(lambda data:datetime.datetime.timestamp(data)) (time)
    # data[:,0] = epoch_timestamp
    # df_sample = pd.DataFrame()
    # for column in df.columns.to_list()[1:]:
    #     data[:,1] = df[column].to_numpy()
    #     if np.isnan(data[:,1]).all():
    #         print(f"Dropping column: {column}")
    #     else:
    #         constructed_time_list,constructed_value_list,got_data = data_sampler(data=data, stepSize=stepSize, start_time=start_time, end_time=end_time, dt_limit=dt_limit)
    #         if got_data==True:
    #             df_sample[column] = constructed_value_list[:,0]
    #         else:
    #             print(f"Bad data quality. Most of data contains NaN values in file: \n\"{filename}\"")
    #             print(f"Dropping column: {column}")
    # df_sample.insert(0, df.columns.values[0], constructed_time_list)
    # df_sample.to_pickle(cached_filename)
    return df_dict

def merge_files(filenames):
    df_dict_total = {}
    for filename in filenames:
        df_dict = split_file(filename)
        for key,value in df_dict.items():
            if key in df_dict_total:
                merged = pd.concat([df_dict_total[key], value.copy()], ignore_index=True, sort=False)
                df_dict_total[key] = merged
            else:
                df_dict_total[key] = value.copy()
    return df_dict_total

def create_raw_data_folders(df_dict_rooms, save_folder=None):
    save_folder = r"C:\Users\jabj\Downloads\new_OUH_time_series_data\Rooms"
    tag_names = list(df_dict_rooms.keys())
    room_names = list(set([el[0:13] for el in tag_names]))
    for name in room_names:
        path = os.path.join(save_folder, name)
        os.makedirs(path)

        for key, value in df_dict_rooms.items():
            if name in key:
                filename = os.path.join(path, key + ".csv")
                value.to_csv(filename)

def clean_df_dict_rooms(df_dict_total, date_format="%m/%d/%Y %I:%M:%S %p"):
    df_dict_total_clean = {}
    tag_names = list(df_dict_total.keys())
    room_names = list(set([el[0:13] for el in tag_names]))
    for name in room_names:
        for key, value in df_dict_total.items():
            if name in key:
                if name in df_dict_total_clean:
                    x = pd.to_numeric(value["vValue"], errors='coerce').values #Remove string entries
                    df_dict_total_clean[name].insert(0, key, x)
                else:
                    time = np.vectorize(lambda data:dateutil.parser.parse(data))(value["DateTime"])
                    # time = pd.to_datetime(value["DateTime"])
                    # time = np.vectorize(lambda data:datetime.datetime.strptime(data, date_format)) (value["DateTime"])
                    df_dict_total_clean[name] = pd.DataFrame()
                    df_dict_total_clean[name].insert(0, "DateTime", time)
                    x = pd.to_numeric(value["vValue"], errors='coerce').values #Remove string entries
                    df_dict_total_clean[name].insert(0, key, x)
        df_dict_total_clean[name] = df_dict_total_clean[name].set_index("DateTime")
    return df_dict_total_clean

def test():
    filenames = [r"C:\Users\jabj\Downloads\OD095_01_Rumdata_juni.csv",
                r"C:\Users\jabj\Downloads\OD095_01_Rumdata_juli.csv",
                r"C:\Users\jabj\Downloads\OD095_01_Rumdata_august.csv",
                r"C:\Users\jabj\Downloads\OD095_01_Rumdata_september.csv"]
    df_dict_total_rooms = merge_files(filenames=filenames)
    df_dict_total_clean = clean_df_dict_rooms(df_dict_total_rooms, date_format="%Y/%m/%d %H:%M:%S.%fZ")
    for key,value in df_dict_total_clean.items():
        axes = value.plot(subplots=True, sharex=True)
        fig = axes[0].get_figure()
        fig.suptitle(key, fontsize=20)
        plt.show()
        # axes[0].set_title(key)

        fig, ax = plt.subplots()
        ax.set_title("QNB10")
        ax.scatter(value.iloc[:,1], value.iloc[:,0])
        fig, ax = plt.subplots()
        ax.set_title("QNB09")
        ax.scatter(value.iloc[:,3], value.iloc[:,2])
        plt.show()

    # filenames = [r"C:\Users\jabj\Downloads\OD095_01_HF04_juni.csv",
    #             r"C:\Users\jabj\Downloads\OD095_01_HF04_juli.csv",
    #             r"C:\Users\jabj\Downloads\OD095_01_HF04_august.csv",
    #             r"C:\Users\jabj\Downloads\OD095_01_HF04_september.csv"]
    # df_ventilation = merge_files(filenames=filenames)


if __name__=="__main__":
    test()