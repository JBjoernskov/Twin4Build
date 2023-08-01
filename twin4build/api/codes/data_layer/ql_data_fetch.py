
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
    #token_1 = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJwOUREeGpCTS1ZX0xKQzRUUDJaTnI3bGZTTmpuQ0poM3pvTHY0MlgxYzlrIn0.eyJleHAiOjE2OTA4NTkyOTIsImlhdCI6MTY5MDg1ODM5MiwiYXV0aF90aW1lIjoxNjkwODU4Mzg5LCJqdGkiOiJhMzZlNDc3NC1lY2Q5LTRlOGItYTk1Yi00MjM4OTI3NzY1MTEiLCJpc3MiOiJodHRwczovL2tleWNsb2FrLmRldi50d2luNGJ1aWxkLmttZC5kay9yZWFsbXMvb3MyaW90IiwiYXVkIjpbInNjb3JwaW8iLCJhY2NvdW50Il0sInN1YiI6IjE4ZTcwNjBhLWUwZWMtNGRiNC1iZGFmLTM4ZmUxNjVlNzhlZSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImNidCIsIm5vbmNlIjoiZGJlYzQzNjktZDliYy00MTBhLThlNjMtMDNmNmUwZWM0NWYwIiwic2Vzc2lvbl9zdGF0ZSI6IjhhYWFjN2Q3LWU0NDEtNDg1ZS1hZDY0LTQ2YTA2ZTEyNTFlMiIsImFjciI6IjEiLCJhbGxvd2VkLW9yaWdpbnMiOlsiKiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsidGVuYW50Omd0bV9qYWtvYiIsInRlbmFudDpuZWNpX2pha29iX2NvcHkiLCJvZmZsaW5lX2FjY2VzcyIsInRlbmFudDpMZW5vdm8iLCJjYnQtYWRtaW4iLCJkZWZhdWx0LXJvbGVzLW9zMmlvdCIsInVtYV9hdXRob3JpemF0aW9uIiwidGVuYW50OlR1dG9yaWFsIiwidGVuYW50OnNkdSJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImNidCI6eyJyb2xlcyI6WyJBZG1pbiJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJvcGVuaWQgZW1haWwgcHJvZmlsZSBjb250ZXh0X2Jyb2tlciIsInNpZCI6IjhhYWFjN2Q3LWU0NDEtNDg1ZS1hZDY0LTQ2YTA2ZTEyNTFlMiIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJjYnQtYWRtaW4iLCJnaXZlbl9uYW1lIjoiIiwiZmFtaWx5X25hbWUiOiIiLCJlbWFpbCI6ImNidC1hZG1pbkBrbWQuZGsifQ.KnqwtrDwIHtk7YmgLIG2bfCZJ7CY-9b1WWt2dKL2chIE1xp3m4i1rAa9y5oijclW6j9URruAu24Bwf7ESDbYc-6UDjjcVDsF7ucsjcZFT0DmLSYKnhg9g0eISg0j6lsro6gJE6U_6sy2J4Ero-WO_0-0oElnIQgqmALQ2-bEHMYyMbPOm9XWn5IBrJLFOqVLbKfl903i-vIL-wNcvinGzO5f5JEY9rBMe9GoDA1_kgSxBAXDVCSOI_rEB5GgywaklNi7PR76MTgH0Kkb_cJrGNGN2zaP5ZjUyhOoHaKAGh4rr2PZYzvf76m4ykoP0tWGhyz1YToM1isODKFAPAQEcQ",
    url_1 = "https://quantumleap.dev.twin4build.kmd.dk/v2/entities/urn:ngsi-ld:Sensor:O20-601b-2/attrs/damper"
    headers_1 = create_headers()
    headers_1["Authorization"] = ""
    client = ApiClient(url_1, headers_1)
    data_1 = client.fetch_data()

    #token_2 = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJwOUREeGpCTS1ZX0xKQzRUUDJaTnI3bGZTTmpuQ0poM3pvTHY0MlgxYzlrIn0.eyJleHAiOjE2OTA4NjAwNzIsImlhdCI6MTY5MDg1OTE3MiwiYXV0aF90aW1lIjoxNjkwODU5MTcxLCJqdGkiOiIwZDE2ZjVjOS1hNWJiLTRiOGItYTY4Ny05YzYyOTk5ZTNkNzciLCJpc3MiOiJodHRwczovL2tleWNsb2FrLmRldi50d2luNGJ1aWxkLmttZC5kay9yZWFsbXMvb3MyaW90IiwiYXVkIjpbInNjb3JwaW8iLCJhY2NvdW50Il0sInN1YiI6IjE4ZTcwNjBhLWUwZWMtNGRiNC1iZGFmLTM4ZmUxNjVlNzhlZSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImNidCIsIm5vbmNlIjoiNmVmODA2NTktNWMxOC00MmZiLWFiMTItMGIzN2RhMDU1YzI0Iiwic2Vzc2lvbl9zdGF0ZSI6IjVjMGVjNGE5LTI3MTMtNGJkNC05ZTFkLTY3NDVkN2M1MzVhZiIsImFjciI6IjEiLCJhbGxvd2VkLW9yaWdpbnMiOlsiKiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsidGVuYW50Omd0bV9qYWtvYiIsInRlbmFudDpuZWNpX2pha29iX2NvcHkiLCJvZmZsaW5lX2FjY2VzcyIsInRlbmFudDpMZW5vdm8iLCJjYnQtYWRtaW4iLCJkZWZhdWx0LXJvbGVzLW9zMmlvdCIsInVtYV9hdXRob3JpemF0aW9uIiwidGVuYW50OlR1dG9yaWFsIiwidGVuYW50OnNkdSJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImNidCI6eyJyb2xlcyI6WyJBZG1pbiJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJvcGVuaWQgZW1haWwgcHJvZmlsZSBjb250ZXh0X2Jyb2tlciIsInNpZCI6IjVjMGVjNGE5LTI3MTMtNGJkNC05ZTFkLTY3NDVkN2M1MzVhZiIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJjYnQtYWRtaW4iLCJnaXZlbl9uYW1lIjoiIiwiZmFtaWx5X25hbWUiOiIiLCJlbWFpbCI6ImNidC1hZG1pbkBrbWQuZGsifQ.ZKU8vuX6RGS6pwSogEeywisSnT43gikSUPJVHwZTv7Y58BHar2Rli4nPPQrVDVNDbbWC6siFV5kLB0ax3wccK4eH75OO1smmeoEPxjP1HC20DFZg_pQlwGuG0rvSFB4p3K9VUnB-hEBEoQMyQae9UA32PGnAqy2hQBqNIhkHVaNgNixhpaUMJQ8Q8UkVHpmuFLrEsU6bfRFmRMnif7us8XsH1Z0FIKKOcsM3No6VJPMrHMpKuTeHb8dVETr-Q3YRjWME6AbmsxFricI3j22qrWRGW3oAoRVCqMFkoA5RcFRPZH1Yb59E6Cc3YIVWYgg-yDJzhF-It3GwUXFySOc1jA"   
    url_2= "https://quantumleap.dev.twin4build.kmd.dk/v2/entities"
    headers_2 = create_headers()
    headers_2["Authorization"] = ""
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