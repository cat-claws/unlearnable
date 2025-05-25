import time
from datetime import datetime, timedelta

# Set the runtime duration: 5 days
run_duration = timedelta(days=5)
start_time = datetime.now()
end_time = start_time + run_duration

# Optional: log start
print(f"Start time: {start_time.isoformat()}")

# Loop until the target time, sleeping most of the time
while datetime.now() < end_time:
    time.sleep(60)  # Sleep 60 seconds at a time
    # Optional: log heartbeat every hour, etc.
    # print(f"Still running at {datetime.now().isoformat()}")
    
# Optional: log end
print(f"End time: {datetime.now().isoformat()}")
