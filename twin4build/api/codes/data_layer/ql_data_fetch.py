
'''
A request call to fetch the data from API provided
'''

import requests

class ApiClient:
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers

    def get_response(self,verify=False):
        return requests.get(url=self.url, headers=self.headers, verify=verify)

    def fetch_data(self):
        try:
            response = self.get_response()
            response.raise_for_status()

            if response.status_code == 401:
                token = self.get_token()  # Call create_token function to get the token

                if token:
                    self.headers["Authorization"] = "Bearer " + token
                    response =self.get_response()  # Retry the request with updated token
                else:
                    print("Error creating token.")
                    return None

            if response.status_code == 200:
                return response.json()   # If the response is JSON, return the parsed data
            else:
                print(f"Request failed with status code: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print("Error Got On Fetching Data:", e)
            return None

    def get_token(self):
        # For this example, we are just returning a string as the token.
        # implement actual token retrieval logic. / API calls
        return "token"

    def update(self, url, headers):
        self.url = url
        self.headers = headers

def create_headers():
    headers = {
        "Accept": "*/*",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "NGSILD-Tenant": "sdu",
        "NGSILD-Path": "/GetSensorData",
    }
    return headers

def main():
    url_1 = "https://quantumleap.dev.twin4build.kmd.dk/v2/entities/urn:ngsi-ld:Sensor:O20-601b-2/attrs/damper"
    headers_1 = create_headers()
    headers_1["Authorization"] = "Bearer "
    client = ApiClient(url_1, headers_1)
    data_1 = client.fetch_data()

    url_2= "https://quantumleap.dev.twin4build.kmd.dk/v2/entities"
    headers_2 = create_headers()
    headers_2["Authorization"] = "Bearer "
    client.update(url_2,headers_2)
    data_2 = client.fetch_data()
    
    if data_1:
        print("Data from url_1:")
        print(data_1)
    else:
        print("Error fetching data from url_1")

    if data_2:
        print("Data from url_2:")
        print(data_2)
    else:
        print("Error fetching data from url_2")

if __name__ == "__main__":
    main()