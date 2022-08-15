from twin4build.utils.uppath import uppath
from twin4build.utils.custom_unpickler import CustomUnpickler
import os

filehandler = open(os.path.join(uppath(__file__, 2), "test", "data", "saved_building_data_collection_dict_notime.pickle"), 'rb')
building_data_collection_dict = CustomUnpickler(filehandler).load()
