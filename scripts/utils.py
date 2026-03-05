import pandas as pd
def read_events(file_path):
    events = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if "-" in line and ";" in line:
                parts = line.split(";")

                time_part = parts[0].strip()
                duration = float(parts[1].strip())
                breathing_label = parts[2].strip()
                sleep_stage = parts[3].strip()

                start_str, end_str = time_part.split("-")

                start_time = pd.to_datetime(
                    start_str,
                    format="%d.%m.%Y %H:%M:%S,%f"
                )

                # add date to end time
                end_time = pd.to_datetime(
                    start_str[:11] + end_str,
                    format="%d.%m.%Y %H:%M:%S,%f"
                )

                events.append({
                    "start": start_time,
                    "end": end_time,
                    "duration": duration,
                    "breathing_label": breathing_label,
                    "sleep_stage": sleep_stage
                })

    return pd.DataFrame(events)
def read_signal(file_path):
    data_started = False
    timestamps = []
    values = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Check when actual data starts
            if line == "Data:":
                data_started = True
                continue

            if data_started and line:
                parts = line.split(';')
                time_str = parts[0].strip()
                value = parts[1].strip()

                timestamps.append(time_str)
                values.append(float(value))

    # Create dataframe
    df = pd.DataFrame({
        "timestamp": timestamps,
        "value": values
    })

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d.%m.%Y %H:%M:%S,%f"
    )

    df.set_index("timestamp", inplace=True)

    return df

def read_sleep_profile(file_path):
    timestamps = []
    stages = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Detect actual data lines
            if ";" in line and "." in line:
                parts = line.split(";")

                time_str = parts[0].strip()
                stage = parts[1].strip()

                timestamp = pd.to_datetime(
                    time_str,
                    format="%d.%m.%Y %H:%M:%S,%f"
                )

                timestamps.append(timestamp)
                stages.append(stage)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "sleep_stage": stages
    })

    df.set_index("timestamp", inplace=True)

    return df

