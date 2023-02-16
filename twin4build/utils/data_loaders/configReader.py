import configparser

class fiwareConfig:
    def __init__(self, scorpioBaseUrl, quantumLeapBaseUrl, fiwareContextLink, tokenUrl, tokenClientId, tokenSecret, tenant, scope):
        self.scorpioBaseUrl = scorpioBaseUrl
        self.quantumLeapBaseUrl = quantumLeapBaseUrl
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

        return fiwareConfig(config['DEFAULT']["ScorpioBaseUrl"], 
                            config['DEFAULT']["QuantumLeapBaseUrl"],
                            config['DEFAULT']["FiwareContextLink"], 
                            config['DEFAULT']["TokenUrl"], 
                            config['DEFAULT']["TokenClientId"], 
                            config['DEFAULT']["TokenSecret"], 
                            config['DEFAULT']["Tenant"],
                            config['DEFAULT']["Scope"])

