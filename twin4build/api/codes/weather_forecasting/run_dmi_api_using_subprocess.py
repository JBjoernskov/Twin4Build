"This code is created to easly convert this code into EXE "

import subprocess
import time

filename = "dmi_api_forecast_data_handler.py"
process = None

# Schedule subsequent function calls at 3-hour intervals
duration = 1
sleep_interval = duration * 60 * 60  # 3 hours in seconds

count = 0

while True:
    try:
        # If a process is already running, terminate it
        if process:
            process.terminate()
            process.wait()  # Wait for the process to terminate before continuing
        
        # Start a new process
        process = subprocess.Popen(['C:/ProgramData/miniconda3/python.exe', filename])
        count = count+1
        print("Code is running at this count",count)
        
    except Exception as e:
        print("An Exception Occured at exe code and error is %s"%e)
    
    # Add a delay before the next iteration
    time.sleep(sleep_interval)  # Adjust the delay time as needed
