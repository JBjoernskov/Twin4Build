import os
import sys
import subprocess
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import time

class MyService(win32serviceutil.ServiceFramework):
    _svc_name_ = 'dmi_open_data_client'
    _svc_display_name_ = 'dmi_open_data_client Service'

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        self.is_alive = True

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.is_alive = False

    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                              servicemanager.PYS_SERVICE_STARTED,
                              (self._svc_name_, ''))
        self.main()

    def main(self):
        # Set the path to your Anaconda environment activation script
        anaconda_activate_script = r'C:\ProgramData\miniconda3\Scripts\activate.bat'

        # Set the path to your Python script
        python_script_path  = r'D:\Data Science\Twin4Build\twin4build\api\codes\weather_forecasting\dmi_open_data_client.py'

        while self.is_alive:
            # Activate the Anaconda environment
            activation_command = f'call "{anaconda_activate_script}"'
            subprocess.call(activation_command, shell=True)

            # Run your Python script
            subprocess.call(['python', python_script_path])

            # Sleep for some time before running the script again
            time.sleep(60)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(MyService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(MyService)

# C:\ProgramData\miniconda3\python.exe