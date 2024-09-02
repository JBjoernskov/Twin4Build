import os
import time  # Import the time module

filename = "dmi_api_data.py"

while True:
    try:
        os.system('cmd /k C:/ProgramData/miniconda3/python.exe %s' % filename)
        print("Code is running")
    except Exception as e:
        print(e)
    
    # Add a delay before the next iteration
    time.sleep(60)  # Adjust the delay time as needed