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


API Usage
--------------------
The Twin4Build API uses `REST (Representational State Transfer)`. JSON is returned by all API responses including errors and HTTP response status codes are to designate success and failure.

The main use-case of the API is to perform model simulations over a time-range. This is done with the library's simulator class. The `entrypoint <https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/api/codes/ml_layer/simulator_api.py>`__ script creates a uvicorn webserver that exposes a FastAPI in the port 8070.

This API exposes the endpoint: 

.. code-block:: 

       POST
       /simulate/

**Example request**:

**Example response**:


Reference input
~~~~~~~~

.. code-block:: json

   {
    "metadata": {
        "location": "SDU",
        "building_id": "001",
        "floor_number": "003",
        "room_id": "601b",
        "start_time": "2023-09-05 08:34:05",
        "end_time": "2023-09-05 10:34:05",
        "roomname": "O20-601b-2",
        "stepSize": 600
    },
    "inputs_sensor": {
        "ml_inputs_dmi": {
            "radia_glob": [
                "539.0",
                "558.0",
                "653.0"
            ],
            "temp_dry": [
                "22.1",
                "22.2",
                "23.9"
            ],
            "observed": [
                "2023-09-05 08:40:00+00:00",
                "2023-09-05 08:50:00+00:00",
                "2023-09-05 10:30:00+00:00"
            ]
        },
        "ml_inputs": {
            "damper": [
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0",
                "100.0"
            ],
            "opcuats": [
                "2023-09-05 08:38:51+00:00",
                "2023-09-05 08:43:51+00:00",
                "2023-09-05 08:48:51+00:00",
                "2023-09-05 08:53:51+00:00",
                "2023-09-05 08:58:51+00:00",
                "2023-09-05 09:03:51+00:00",
                "2023-09-05 09:08:51+00:00",
                "2023-09-05 09:13:51+00:00",
                "2023-09-05 09:18:51+00:00",
                "2023-09-05 09:23:51+00:00",
                "2023-09-05 09:28:51+00:00",
                "2023-09-05 09:33:51+00:00",
                "2023-09-05 09:38:51+00:00",
                "2023-09-05 09:43:51+00:00",
                "2023-09-05 09:48:51+00:00",
                "2023-09-05 09:53:51+00:00",
                "2023-09-05 09:58:51+00:00",
                "2023-09-05 10:03:51+00:00",
                "2023-09-05 10:08:51+00:00",
                "2023-09-05 10:13:51+00:00",
                "2023-09-05 10:18:51+00:00",
                "2023-09-05 10:23:51+00:00",
                "2023-09-05 10:28:51+00:00",
                "2023-09-05 10:33:51+00:00"
            ],
            "shadingposition": [
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None"
            ],
            "co2concentration": [
                "990.72",
                "963.84",
                "931.84",
                "926.72",
                "925.44",
                "857.6",
                "814.72",
                "805.76",
                "913.92",
                "1063.68",
                "1145.6",
                "1072.0",
                "1000.96",
                "950.4",
                "890.88",
                "849.92",
                "826.88",
                "816.64",
                "810.88",
                "803.84",
                "778.88",
                "737.92",
                "716.8",
                "728.96"
            ],
            "temperature": [
                "23.6",
                "23.6",
                "23.8",
                "24.0",
                "24.0",
                "24.1",
                "24.1",
                "24.1",
                "24.1",
                "24.2",
                "24.2",
                "24.4",
                "24.4",
                "24.4",
                "24.4",
                "24.4",
                "24.4",
                "24.2",
                "24.0",
                "24.0",
                "24.0",
                "24.2",
                "24.4",
                "24.4"
            ]
        }
    },
    "input_schedules": {
        "temperature_setpoint_schedule": {
            "weekDayRulesetDict": {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [
                    0,
                    0
                ],
                "ruleset_end_minute": [
                    0,
                    0
                ],
                "ruleset_start_hour": [
                    0,
                    7
                ],
                "ruleset_end_hour": [
                    7,
                    18
                ],
                "ruleset_value": [
                    19,
                    21
                ]
            }
        },
        "shade_schedule": {
            "weekDayRulesetDict": {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [
                    30
                ],
                "ruleset_end_minute": [
                    0
                ],
                "ruleset_start_hour": [
                    11
                ],
                "ruleset_end_hour": [
                    18
                ],
                "ruleset_value": [
                    0
                ]
            },
            "fridayRulesetDict": {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [
                    30
                ],
                "ruleset_end_minute": [
                    0
                ],
                "ruleset_start_hour": [
                    8
                ],
                "ruleset_end_hour": [
                    18
                ],
                "ruleset_value": [
                    0
                ]
            },
            "weekendRulesetDict": {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [],
                "ruleset_end_minute": [],
                "ruleset_start_hour": [],
                "ruleset_end_hour": [],
                "ruleset_value": []
            }
        },
        "occupancy_schedule": {
            "weekDayRulesetDict": {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                "ruleset_end_minute": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                "ruleset_start_hour": [
                    6,
                    7,
                    8,
                    12,
                    14,
                    16,
                    18
                ],
                "ruleset_end_hour": [
                    7,
                    8,
                    12,
                    14,
                    16,
                    18,
                    22
                ],
                "ruleset_value": [
                    3,
                    5,
                    20,
                    25,
                    27,
                    7,
                    3
                ]
            },
            "weekendRulesetDict": {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                "ruleset_end_minute": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                "ruleset_start_hour": [
                    6,
                    7,
                    8,
                    12,
                    14,
                    16,
                    18
                ],
                "ruleset_end_hour": [
                    7,
                    8,
                    12,
                    14,
                    16,
                    18,
                    22
                ],
                "ruleset_value": [
                    3,
                    5,
                    20,
                    25,
                    27,
                    7,
                    3
                ]
            }
        },
        "supply_water_temperature_schedule_pwlf": {
            "weekDayRulesetDict": {
                "ruleset_default_value": {
                    "X": [
                        -5,
                        5,
                        7
                    ],
                    "Y": [
                        58,
                        65,
                        60.5
                    ]
                },
                "ruleset_start_minute": [
                    0
                ],
                "ruleset_end_minute": [
                    0
                ],
                "ruleset_start_hour": [
                    5
                ],
                "ruleset_end_hour": [
                    7
                ],
                "ruleset_value": [
                    {
                        "X": [
                            -7,
                            5,
                            9
                        ],
                        "Y": [
                            72,
                            55,
                            50
                        ]
                    }
                ]
            }
        }
    }
 }
       
Reference output
~~~~~~~~~~~~~~~~

.. code-block:: json

   {
    "time": [
        "2023-09-05T08:34:05",
        "2023-09-05T08:44:05",
        "2023-09-05T08:54:05",
        "2023-09-05T09:04:05",
        "2023-09-05T09:14:05",
        "2023-09-05T09:24:05",
        "2023-09-05T09:34:05",
        "2023-09-05T09:44:05",
        "2023-09-05T09:54:05",
        "2023-09-05T10:04:05",
        "2023-09-05T10:14:05",
        "2023-09-05T10:24:05"
    ],
    "OE20-601b-2_indoorTemperature": [
        21.10229484261945,
        21.109176606312396,
        21.122742475382985,
        21.144295174069704,
        21.173813625611366,
        21.210104930587114,
        21.251490611024202,
        21.296413897909225,
        21.343684527464212,
        21.392474769987167,
        21.442233584262432,
        21.492600112594666
    ],
    "OE20-601b-2_indoorCo2Concentration": [
        600.2051518286698,
        591.3579279697896,
        585.9854699188653,
        582.7230560849518,
        580.7419621568422,
        579.5389470423296,
        578.8084186482888,
        578.3648068199237,
        578.0954244716527,
        577.9318426081404,
        577.8325078661209,
        577.772187053286
    ],
    "Supplydamper_airFlowRate": [
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607
    ],
    "Supplydamper_damperPosition": [
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45
    ],
    "Exhaustdamper_airFlowRate": [
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607,
        0.5460645649316607
    ],
    "Exhaustdamper_damperPosition": [
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45
    ],
    "Spaceheater_outletWaterTemperature": [
        [
            21.36038446356353,
            21.36038446356353,
            21.36038446356353,
            21.36038446356353,
            21.36038446356353,
            21.36038446356353,
            21.36038446356353,
            21.36038446356353,
            21.36038446356353,
            21.36038446356353
        ],
        [
            20.541153317038546,
            20.541153317038546,
            20.541153317038546,
            20.541153317038546,
            20.541153317038546,
            20.541153317038546,
            20.541153317038546,
            20.541153317038546,
            20.541153317038546,
            20.541153317038546
        ],
        [
            20.70705975728773,
            20.70705975728773,
            20.70705975728773,
            20.70705975728773,
            20.70705975728773,
            20.70705975728773,
            20.70705975728773,
            20.70705975728773,
            20.70705975728773,
            20.70705975728773
        ],
        [
            20.83178727414114,
            20.83178727414114,
            20.83178727414114,
            20.83178727414114,
            20.83178727414114,
            20.83178727414114,
            20.83178727414114,
            20.83178727414114,
            20.83178727414114,
            20.83178727414114
        ],
        [
            20.92935507372903,
            20.92935507372903,
            20.92935507372903,
            20.92935507372903,
            20.92935507372903,
            20.92935507372903,
            20.92935507372903,
            20.92935507372903,
            20.92935507372903,
            20.92935507372903
        ],
        [
            21.009442896350038,
            21.009442896350038,
            21.009442896350038,
            21.009442896350038,
            21.009442896350038,
            21.009442896350038,
            21.009442896350038,
            21.009442896350038,
            21.009442896350038,
            21.009442896350038
        ],
        [
            21.07849039170974,
            21.07849039170974,
            21.07849039170974,
            21.07849039170974,
            21.07849039170974,
            21.07849039170974,
            21.07849039170974,
            21.07849039170974,
            21.07849039170974,
            21.07849039170974
        ],
        [
            21.140656119253727,
            21.140656119253727,
            21.140656119253727,
            21.140656119253727,
            21.140656119253727,
            21.140656119253727,
            21.140656119253727,
            21.140656119253727,
            21.140656119253727,
            21.140656119253727
        ],
        [
            21.198572811806155,
            21.198572811806155,
            21.198572811806155,
            21.198572811806155,
            21.198572811806155,
            21.198572811806155,
            21.198572811806155,
            21.198572811806155,
            21.198572811806155,
            21.198572811806155
        ],
        [
            21.253886056865145,
            21.253886056865145,
            21.253886056865145,
            21.253886056865145,
            21.253886056865145,
            21.253886056865145,
            21.253886056865145,
            21.253886056865145,
            21.253886056865145,
            21.253886056865145
        ],
        [
            21.30761482287098,
            21.30761482287098,
            21.30761482287098,
            21.30761482287098,
            21.30761482287098,
            21.30761482287098,
            21.30761482287098,
            21.30761482287098,
            21.30761482287098,
            21.30761482287098
        ],
        [
            21.36038446356353,
            21.36038446356353,
            21.36038446356353,
            21.36038446356353,
            21.36038446356353,
            21.36038446356353,
            21.36038446356353,
            21.36038446356353,
            21.36038446356353,
            21.36038446356353
        ]
    ],
    "Spaceheater_Power": [
        -65.50937877858713,
        -47.23089560444362,
        -34.56384171857969,
        -25.984899344459865,
        -20.326624914144322,
        -16.684963046044206,
        -14.384894866604634,
        -12.95119323833079,
        -12.06597761508885,
        -11.523592720622917,
        -11.193492917882436,
        -10.993675144282127
    ],
    "Spaceheater_Energy": [
        -0.01091822979643119,
        -0.018790045730505127,
        -0.024550686016935074,
        -0.02888150257434505,
        -0.0322692733933691,
        -0.035050100567709803,
        -0.03744758304547724,
        -0.039606115251865706,
        -0.04161711152104718,
        -0.04353771030781767,
        -0.04540329246079807,
        -0.04723557165151176
    ],
    "Valve_waterFlowRate": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ],
    "Valve_valvePosition": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ],
    "Temperaturecontroller_inputSignal": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ],
    "CO2controller_inputSignal": [
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45
    ],
    "temperaturesensor_indoorTemperature": [
        21.10229484261945,
        21.109176606312396,
        21.122742475382985,
        21.144295174069704,
        21.173813625611366,
        21.210104930587114,
        21.251490611024202,
        21.296413897909225,
        21.343684527464212,
        21.392474769987167,
        21.442233584262432,
        21.492600112594666
    ],
    "Valvepositionsensor_valvePosition": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ],
    "Damperpositionsensor_damperPosition": [
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45,
        0.45
    ],
    "CO2sensor_indoorCo2Concentration": [
        600.2051518286698,
        591.3579279697896,
        585.9854699188653,
        582.7230560849518,
        580.7419621568422,
        579.5389470423296,
        578.8084186482888,
        578.3648068199237,
        578.0954244716527,
        577.9318426081404,
        577.8325078661209,
        577.772187053286
    ],
    "Heatingmeter_Energy": [
        -0.01091822979643119,
        -0.018790045730505127,
        -0.024550686016935074,
        -0.02888150257434505,
        -0.0322692733933691,
        -0.035050100567709803,
        -0.03744758304547724,
        -0.039606115251865706,
        -0.04161711152104718,
        -0.04353771030781767,
        -0.04540329246079807,
        -0.04723557165151176
    ],
    "Outdoorenvironment_outdoorTemperature": [
        22.2,
        22.2,
        23.9,
        23.9,
        23.9,
        23.9,
        23.9,
        23.9,
        23.9,
        23.9,
        23.9,
        23.9
    ],
    "Outdoorenvironment_globalIrradiation": [
        558.0,
        558.0,
        653.0,
        653.0,
        653.0,
        653.0,
        653.0,
        653.0,
        653.0,
        653.0,
        653.0,
        653.0
    ],
    "Occupancyschedule_scheduleValue": [
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        20
    ],
    "Temperaturesetpointschedule_scheduleValue": [
        21,
        21,
        21,
        21,
        21,
        21,
        21,
        21,
        21,
        21,
        21,
        21
    ],
    "Supplywatertemperatureschedule_supplyWaterTemperatureSetpoint": [
        60.5,
        60.5,
        60.5,
        60.5,
        60.5,
        60.5,
        60.5,
        60.5,
        60.5,
        60.5,
        60.5,
        60.5
    ],
    "Supplyairtemperatureschedule_scheduleValue": [
        21,
        21,
        21,
        21,
        21,
        21,
        21,
        21,
        21,
        21,
        21,
        21
    ]
 }
