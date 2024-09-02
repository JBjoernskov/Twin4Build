from datetime import datetime

time_format = '%Y-%m-%d %H:%M:%S%z'


def ventilation_output_formating(output_data):
    # Convert data into desired format
    records = []

    common_data = output_data['common_data']
    rooms_data = output_data['rooms']

    for room_name, room_info in rooms_data.items():
        air_flow_rates = room_info['Supply_damper_{}_airFlowRate'.format(room_name)]
        damper_positions = room_info['Supply_damper_{}_damperPosition'.format(room_name)]
        
        for i in range(len(common_data['Simulation_time'])):
            record = {
                'room_name': room_name,
                'ventilation_system_name':'ve01',
                'simulation_time': common_data['Simulation_time'][i],
                'total_air_flow_rate': common_data['Sum_of_damper_air_flow_rates'][i],
                'damper_position': damper_positions[i],
                'air_flow_rate': air_flow_rates[i],    
            }
            records.append(record)\
            
    return records

def convert_response_to_list(response_dict):
    try:
        # Extract the keys from the response dictionary
        keys = response_dict.keys()
        # Initialize an empty list to store the result
        result = []
        # Iterate over the data and create dictionaries
        for i in range(len(response_dict["time"])):
            data_dict = {}
            for key in keys:
                data_dict[key] = response_dict[key][i]
            result.append(data_dict)

        #temp file finally we will comment it out
        #logger.info("[request_class]:Converted the response dict to list")
        
        return result
    
    except Exception as converion_error:
        #logger.error('An error has occured %s',str(converion_error))
        print("Responce dictonary has some problem please look into that")
        return None

def space_output_formating(formatted_response_list_data,start_time,end_time):
            '''
            This function transforms the input list data got from space model response into desirable format
            '''
            if len(formatted_response_list_data) < 1:
                  #logger.error("[input_data.py] : Empty formatted_response_list_data got for transforming ")
                  return []

            input_data_list = []
            #logger.info("[request_class]: Enterd Into transform_dict method")

            for original_dict in formatted_response_list_data:
                  # format = '%Y-%m-%d %H:%M:%S%z'
                  #  "time": "2023-12-12 03:13:52+0100",
                  time_str = original_dict['time']
                  datetime_obj = datetime.strptime(time_str, time_format)
                  formatted_time = datetime_obj.strftime(time_format)
                  transformed_dict = {
                        'simulation_time': formatted_time,  
                        'outdoorenvironment_outdoortemperature': original_dict['outdoor_environment_outdoorTemperature'],
                        'outdoorenvironment_globalirradiation': original_dict['outdoor_environment_globalIrradiation'],
                        'indoortemperature': original_dict['OE20-601b-2_indoorTemperature'], 
                        'indoorco2concentration': original_dict['OE20-601b-2_indoorCo2Concentration'],  
                        'supplydamper_airflowrate': original_dict['Supplydamper_airFlowRate'], 
                        'supplydamper_damperposition': original_dict['Supplydamper_damperPosition'], 
                        'exhaustdamper_airflowrate': original_dict['Exhaustdamper_airFlowRate'],  
                        'exhaustdamper_damperposition': original_dict['Exhaustdamper_damperPosition'],  
                        'spaceheater_outletwatertemperature': original_dict['Spaceheater_outletWaterTemperature'],  
                        'spaceheater_power': original_dict['Spaceheater_Power'],   
                        'spaceheater_energy': original_dict['Spaceheater_Energy'],  
                        'valve_waterflowrate': original_dict['Valve_waterFlowRate'], 
                        'valve_valveposition': original_dict['Valve_valvePosition'], 
                        'temperaturecontroller_inputsignal': original_dict['Temperaturecontroller_inputSignal'],  
                        'co2controller_inputsignal': original_dict['CO2controller_inputSignal'], 
                        #change OE20-601b-2 this value with room name when we are scaling the t4b
                        'temperaturesensor_indoortemperature': original_dict['OE20-601b-2temperaturesensor_indoorTemperature'],   
                        'valvepositionsensor_valveposition': original_dict['OE20-601b-2Valvepositionsensor_valvePosition'],   
                        'damperpositionsensor_damperposition': original_dict['OE20-601b-2Damperpositionsensor_damperPosition'],   
                        'co2sensor_indoorco2concentration': original_dict['OE20-601b-2CO2sensor_indoorCo2Concentration'], 
                        'heatingmeter_energy': original_dict['OE20-601b-2Heatingmeter_Energy'],  
                        'occupancyschedule_schedulevalue': original_dict['OE20-601b-2_occupancy_schedule_scheduleValue'],  
                        'temperaturesetpointschedule_schedulevalue': original_dict['OE20-601b-2_temperature_setpoint_schedule_scheduleValue'],  
                        'supplywatertemperatureschedule_supplywatertemperaturesetpoint': original_dict['Heatingsystem_supply_water_temperature_schedule_scheduleValue'], 
                        'ventilationsystem_supplyairtemperatureschedule_schedulevaluet': original_dict['Ventilationsystem_supply_air_temperature_schedule_scheduleValue'], 
                  }

                  transformed_dict['input_start_datetime'] = start_time
                  transformed_dict['input_end_datetime'] = end_time
                  #Hardcoding the room name since we are working with single room
                  transformed_dict['spacename'] = 'OE20-601b-2'

                  input_data_list.append(transformed_dict)            
            #logger.info("[request_class]: Exited from transform_dict method")
            return input_data_list