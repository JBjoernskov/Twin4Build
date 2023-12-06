import sched
import time

# Initialize the scheduler
scheduler = sched.scheduler(time.time, time.sleep)

# Define the function to be called
def function_to_call():
    print("Function called at:", time.strftime("%Y-%m-%d %H:%M:%S"))

# ScheduleSystem the first function call
scheduler.enter(0, 1, function_to_call, ())

# ScheduleSystem subsequent function calls at 2-hour intervals
interval = 2 * 60 * 60  # 2 hours in seconds

while True:
    scheduler.run()
    time.sleep(interval)

# This loop will keep the scheduler running indefinitely, calling the function every 2 hours.

"""Keep in mind that if you want to stop the scheduler at some point, 
you'll need to add a condition to exit the loop. This is a basic example, 
and in a real-world application, you might want to handle exceptions, manage the scheduler's state, 
and possibly use a more robust scheduling library depending on your requirements."""
