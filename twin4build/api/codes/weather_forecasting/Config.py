import configparser


class ConfigReader:
    
    def __init__(self) -> None:
        pass
                
    def read_config_section(self,config_file):
        parser=configparser.ConfigParser()
        parser.read(config_file)
        return parser

    