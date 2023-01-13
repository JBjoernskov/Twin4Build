import configparser

class fiwareConfig:
    def __init__(self, fiwareBaseUrl, fiwareContextLink, tokenUrl, tokenClientId, tokenSecret, tenant, scope):
        self.fiwareBaseUrl = fiwareBaseUrl
        self.fiwareContextLink = fiwareContextLink
        self.tokenUrl = tokenUrl
        self.tokenClientId = tokenClientId
        self.tokenSecret = tokenSecret
        self.tenant = tenant
        self.scope = scope

class configReader():    
    def read_config(self):
        config = configparser.ConfigParser()
        config.read(['config.ini', 'secrets.ini'])

        return fiwareConfig(config['DEFAULT']["FiwareBaseUrl"], 
                            config['DEFAULT']["FiwareContextLink"], 
                            config['DEFAULT']["TokenUrl"], 
                            config['DEFAULT']["TokenClientId"], 
                            config['DEFAULT']["TokenSecret"], 
                            config['DEFAULT']["Tenant"],
                            config['DEFAULT']["Scope"])

