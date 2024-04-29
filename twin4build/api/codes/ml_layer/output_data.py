
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