import logging
import  sys
import os, os.path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from twin4build.config.Config import ConfigReader
from twin4build.utils.uppath import uppath
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
        """Get active instance of logger"""
        try:
            config_path = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "config", "conf.ini")
            conf=ConfigReader()
            config=conf.read_config_section(config_path)
            dir_path = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), *config['logs']['directory'].split("/"))
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
                        #logging.FileHandler(log_file_name, mode='a'),
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
            