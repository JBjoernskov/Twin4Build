import json
import os

def split_json_by_room(input_file):
    # Load the JSON data from the input file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Extract the rooms_sensor_data dictionary
    rooms_sensor_data = data.get("rooms_sensor_data", {})

    # Iterate over each room in the rooms_sensor_data
    for room_name, room_data in rooms_sensor_data.items():
        # Extract only the occupancy schedule for the room
        occupancy_schedule = room_data.get("occupancy_schedule", {})

        # Create a new JSON structure for the room's occupancy schedule
        room_json = {
            "occupancy_schedule": occupancy_schedule
        }

        # Define the output file name based on the room name
        output_file = f"{room_name}_schedules.json"

        # Write the room's data to the output file
        with open(output_file, 'w') as outfile:
            json.dump(room_json, outfile, indent=4, separators=(',', ': '), default=str)

# Example usage
input_file = 'ventilation_input_data_schedules_all_rooms.json'
split_json_by_room(input_file)
