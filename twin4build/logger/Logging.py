import logging
import  sys
import os, os.path
from datetime import datetime
from logging.handlers import RotatingFileHandler


## Temp solution Might have to change
uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 3)
sys.path.append(file_path)

from twin4build.config.Config import ConfigReader

class Logging():
    
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def initialize_logs_files(cls, dir_path, file_name):
        """Set path for intialize log file"""
        try:
            log_file_name=datetime.now().strftime(dir_path+file_name+'_%d_%m_%Y.log')
            
            return log_file_name
        except Exception as exce:
            return False

    @classmethod
    def get_logger(cls, log_file_name):
        """Get  active instance of logger"""
        try:
            conf=ConfigReader()
            config=conf.read_config_section('twin4build\config\conf.ini')

            dir_path=config['logs']['directory']
            maxBytes=config['logs']['maxBytes']
            backupCount=config['logs']['backupCount']
            log_level=config['logs']['log_level']
            levels = {
                "info" : logging.INFO
                , "error" : logging.ERROR
                , "warning": logging.WARNING
                , "debug" : logging.DEBUG
            }

            if log_level.lower() not in levels.keys():
                log_level=logging.INFO
            else:
                log_level = levels[log_level.lower()]
                
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            else:
                log_file_name=cls.initialize_logs_files(dir_path,log_file_name)
                #print("log file name:{}".format(log_file_name))
                logging.basicConfig(format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)5s() ] - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_name, mode='a'),
                        RotatingFileHandler(
                            log_file_name,
                            maxBytes=float(maxBytes), 
                            backupCount=int(backupCount)
                            )])
                logger = logging.getLogger()
                # Setting the threshold of logger to DEBUG
                logger.setLevel(log_level)
                return logger
        except Exception as exce:
            print("exception",exce)
            