import os
import sys
import pandas as pd

# Only for testing before distributing package
if __name__ == '__main__':
    def uppath(_path, n): return os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)

from twin4build.api.codes.database.db_data_handler import db_connector

# Function to read CSV files, drop selected columns by index, rename columns, and convert to list of dictionaries
def process_csv_files(directory, selected_column_indices, column_names):
    data_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and filename.startswith("O"):
            print("++++++++ Processing File:",filename)
            file_path = os.path.join(directory, filename)
            #df = pd.read_csv(file_path) #reads every row 

            # Read every 5th row
            df = pd.read_csv(file_path, skiprows=lambda x: x % 5 != 0) 

            df = df.assign(room_name=filename.split(".")[0])  # Add filename column
            df = df.assign(ventilation_system_name="VE01")
            df.drop(df.columns[selected_column_indices], axis=1, inplace=True)

            df.rename(columns=dict(zip(df.columns, column_names)), inplace=True)
            df = df.iloc[:, [5, 6, 0, 1, 2, 3, 4]]

            # Filter DataFrame to remove rows where float columns has strings
            float_col = ['co2concentration', 'temperature', 'air_damper_position', 'radiator_valve_position']
            for col_name in float_col:
                df = df[pd.to_numeric(df[col_name], errors='coerce').notnull()]

            # Remove rows with null or empty values
            df.dropna(inplace=True)
            df.dropna(how='all', inplace=True)

            # Convert DataFrame rows to dictionaries and store in a list
            for _, row in df.iterrows():
                data_list.append(row.to_dict())
    print("--------------- Data Processing has been complete")
                       
    return data_list


if __name__ == "__main__":
    connector = db_connector()
    connector.connect()

    tablename = "ml_ventilation_dummy_inputs"
    directory = 'C:/Temp/DummyData/'  # Replace with the path to your CSV files directory
    selected_column_indices = [2]  # Replace with the indices of columns you want to drop
    column_names = ['simulation_time', 'co2concentration', 'temperature', 'air_damper_position', 'radiator_valve_position']  # Replace with the new column names
    data_list = process_csv_files(directory, selected_column_indices, column_names)

    # Insert data into database using ORM
    connector.add_large_data(tablename, data_list)

    connector.disconnect()
