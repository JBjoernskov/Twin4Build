Twin4build API Documentation
====================================================================================================

This is a work-in-progress library and the functionality is therefore
updated regularly. More information on the use of the framework and code
examples are coming in the near future!

Running the API locally
------------------------

•	Installing the required module using: pip install -r requirements.txt
•	Adjust conf.ini as parameters as per you system variables.
•	To run the API, use the simulation_api.py file.
•	To query this API run request_to_api.py file.
Data fetching Method 1:
•	Getting the access token and refresh token use keycloak_token.py
•	Ql_data_fetch.py is used to fetch the data from QL API using the access token from the keycloak
Data fetching Method 2:
•	Db_data_handler.py is used to store and retrieve data from the PostgreSQL database
•	All the variables are defined in the config.ini file, we’re retrieving the data using Config.py file in config folder
•	Input_data.py is used to format the input and output data as required 
•	Run the request__to_api.py file using python which will retrieve the information from the config.ini file use input_data.py to format the data as required pass the data as POST request to the simulation_api, FastAPI server running on port: 8005
•	The response is formatted and is stored to the database using db_data_handler.py 


Dockerizing the API 
--------------------
After having a model ready to ship in a Docker container, the next step is to create a Docker image that contains the model and its dependencies. 
This is done by creating a Dockerfile. The Dockerfile is a text file that contains all the commands a user could call on the command line to assemble an image. 
Using docker build, users can create an automated build that executes several command-line instructions in succession.

An example dockerfile is provided `Here <https://github.com/SebsCubs/Twin4Build/blob/twin4build_api_updates/twin4build/api/dockerization/Dockerfile>`__

This file is used to create a Docker image, which is a lightweight, standalone, executable package that includes everything needed to run a digital twin model, including the code, a runtime, libraries, environment variables, and config files. This file sets up an environment with Ubuntu, Python, and several dependencies, clones a development branch of this GitHub repository, checks out a specific branch, installs further dependencies from a requirements file, and sets up a script to run when the Docker container starts.

The script starts a uvicorn server and exposes the port 8070 for requests to the FastAPI implementation.


Reference input
~~~~~~~~


       

