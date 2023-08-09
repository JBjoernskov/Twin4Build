'''
Using keycloak to authenticate user and get token and refresh token
'''


from keycloak import KeycloakOpenID
from datetime import datetime , timedelta
import keycloak.exceptions 
import os
import sys


###Only for testing before distributing package
uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 5)
sys.path.append(file_path)

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")


class KeycloakAuthenticator:
    def __init__(self, server_url, realm_name, client_id) -> None:
        logger.info("KeycloakAuthenticator : Initialise Function")
        self.server_url = server_url
        self.realm_name = realm_name
        self.client_id = client_id
        self.keycloak_openid = KeycloakOpenID(server_url=self.server_url, 
                                         client_id=self.client_id, 
                                         realm_name=self.realm_name)
        
        self.token = None
        self.access_token_expiration = None
        self.refresh_token_expiration = None

    #authenticate to keycloack with user and password
    def authenticate(self, user, password):
        logger.info("Authenticating KeyCloak")
        try : 
            self.token = self.keycloak_openid.token(user, password)

            self.access_token_expiration = datetime.now() + timedelta(seconds=self.get_access_expiration_time())
            self.refresh_token_expiration = datetime.now() + timedelta(seconds=self.get_refresh_expiration_time())

            logger.info("Authentication made with keycloak")
        except Exception as e:
            logger.error("Authentication to keycloak failed , Exception :" ,e)
    #get the refresh token
    def get_refresh_token(self) -> str:
        try : 
            logger.info("Getting Refresh Token")
            return self.token['refresh_token']
        
        except keycloak.exceptions.KeycloakAuthenticationError as e:    
            logger.error("Error on getting refresh token: ", e)
            print("Error on getting refresh token: ", e)
            exit(1)

    # get the access token
    def get_access_token(self) -> str:
        try : 
            logger.info("Getting Access Token")
            return self.token['access_token']
        
        except keycloak.exceptions.KeycloakAuthenticationError as e:   
            logger.error("Error On getting access token: ", e)
            print("Error On getting access token: ", e)
            exit(1)


    #get the expiration time of  refresh token
    def get_refresh_expiration_time(self):
        return self.token['refresh_expires_in']
    
    # get the expiration time of acess token
    def get_access_expiration_time(self):
        return self.token['expires_in']
    

    # checking is the access token expired ?
    def is_access_token_expired(self):
        return datetime.now() >= self.access_token_expiration
    
    # checking is the refresh token expired ?
    def is_refresh_token_expired(self):
        return datetime.now() >= self.refresh_token_expiration
    
    # creating new refresh token from existing one 
    def refresh_token(self):
        try: 
            logger.info("Getting new Token from KeyCloack for QL")
            refresh_token = self.get_refresh_token()
            new_token = self.keycloak_openid.refresh_token(refresh_token)
            self.token = new_token
            self.access_token_expiration = datetime.now() + timedelta(seconds=self.get_access_expiration_time())
            self.refresh_token_expiration = datetime.now() + timedelta(seconds=self.get_refresh_expiration_time())
        except Exception as e :
            logger.error("Error creating refresh token , Exception : ",e)

    
