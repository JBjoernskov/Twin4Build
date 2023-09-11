# Import required modules
from keycloak import KeycloakOpenID
from datetime import datetime, timedelta
import keycloak.exceptions
import os
import sys

# Calculate the path to the root directory and add it to the sys path
uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 5)
sys.path.append(file_path)

# Import the custom logger module
from twin4build.logger.Logging import Logging

# Initialize the logger
logger = Logging.get_logger("ai_logfile")

# Class to handle Keycloak authentication and token management
class KeycloakAuthenticator:
    def __init__(self, server_url, realm_name, client_id):
        logger.info("KeycloakAuthenticator: Initialize Function")
        self.server_url = server_url
        self.realm_name = realm_name
        self.client_id = client_id

        # Create a KeycloakOpenID instance
        self.keycloak_openid = KeycloakOpenID(
            server_url=self.server_url,
            client_id=self.client_id,
            realm_name=self.realm_name
        )

        # Initialize token and expiration times
        self.token = None
        self.access_token_expiration = None
        self.refresh_token_expiration = None

    # Authenticate to Keycloak using user credentials
    def authenticate(self, user, password):
        logger.info("Authenticating with Keycloak")
        try:
            # Get the token using user credentials
            self.token = self.keycloak_openid.token(user, password)

            # Calculate token expiration times
            self.access_token_expiration = datetime.now() + timedelta(seconds=self.get_access_expiration_time())
            self.refresh_token_expiration = datetime.now() + timedelta(seconds=self.get_refresh_expiration_time())

            logger.info("Authentication successful with Keycloak")
        except Exception as e:
            logger.error("Authentication to Keycloak failed. Exception:", e)

    # Get the refresh token
    def get_refresh_token(self) -> str:
        try:
            logger.info("Getting Refresh Token")
            return self.token['refresh_token']
        except keycloak.exceptions.KeycloakAuthenticationError as e:
            logger.error("Error getting refresh token:", e)
            print("Error getting refresh token:", e)
            exit(1)

    # Get the access token
    def get_access_token(self) -> str:
        try:
            logger.info("Getting Access Token")
            return self.token['access_token']
        except keycloak.exceptions.KeycloakAuthenticationError as e:
            logger.error("Error getting access token:", e)
            print("Error getting access token:", e)
            exit(1)

    # Get the expiration time of the refresh token
    def get_refresh_expiration_time(self):
        return self.token['refresh_expires_in']

    # Get the expiration time of the access token
    def get_access_expiration_time(self):
        return self.token['expires_in']

    # Check if the access token has expired
    def is_access_token_expired(self):
        return datetime.now() >= self.access_token_expiration

    # Check if the refresh token has expired
    def is_refresh_token_expired(self):
        return datetime.now() >= self.refresh_token_expiration

    # Refresh the access token using the refresh token
    def refresh_token(self):
        try:
            logger.info("Refreshing Token from Keycloak")
            refresh_token = self.get_refresh_token()
            new_token = self.keycloak_openid.refresh_token(refresh_token)
            self.token = new_token

            # Update token expiration times
            self.access_token_expiration = datetime.now() + timedelta(seconds=self.get_access_expiration_time())
            self.refresh_token_expiration = datetime.now() + timedelta(seconds=self.get_refresh_expiration_time())
        except Exception as e:
            logger.error("Error refreshing token. Exception:", e)

# Example usage:
"""if __name__ == "__main__":
    # Initialize the KeycloakAuthenticator with server URL, realm name, and client ID
    authenticator = KeycloakAuthenticator(
        server_url="https://keycloak.example.com/auth",
        realm_name="your-realm-name",
        client_id="your-client-id"
    )
    
    # Authenticate with user credentials
    authenticator.authenticate(user="your-username", password="your-password")
    
    # Check if the access token has expired and refresh it if needed
    if authenticator.is_access_token_expired():
        if not authenticator.is_refresh_token_expired():
            authenticator.refresh_token()
    
    # Get the refreshed access token
    access_token = authenticator.get_access_token()
    
    # Print the access token
    print("Access Token:", access_token)"""
