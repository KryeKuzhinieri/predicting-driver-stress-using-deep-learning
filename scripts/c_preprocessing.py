from pathlib import Path

import pandas as pd


# Dataset comes from Akbas(2011)
CSV_PATH = "/media/kryekuzhinieri/Thesis/Datasets 2/Stress Recognition In Automobile Drivers/csv_files"
MARKER_DATA_CSV = "Marker Data/marker_info.csv"


def calculate_starting_times(data):
    """
    Calculates all starting times for each driving period.
    For example, Rest1: 15.13, City1: 16.00 -> Rest1: 0, City: 15.13
    input: data (marker data)
    """
    formated_data = data.copy()
    # Starting time is 0.
    formated_data['Rest1'] = 0
    # Exclude columns such as drive number and total.
    numeric_cols = data.iloc[:, 1:8].columns
    for index, column in enumerate(numeric_cols):
        if index == len(numeric_cols) - 1:
            pass
        else:
            next_column = numeric_cols[index + 1]
            # Calculate the time between the current col and next one.
            formated_data[next_column] = formated_data[column] + data[column]
    print("Data with Starting Times in Minutes\n", formated_data)
    return formated_data


def get_starting_indices(processed_markers, drive):
    """
    Gets the drive minutes for each driving periods and converts them to slices
    for the dataframe.
    inputs:
        processed_markers: marker dataframe with starting times.
        drive: drive name (str) - i.e: Drive05
    """
    signals = processed_markers[processed_markers["Driver"] == drive].iloc[:, 1:8]
    # Signals are given in 15.5Hz frequency. Marker data is in Minutes.
    signals = signals * 15.5 * 60
    return signals.astype(int).values[0]


def label_data(starting_indices, row_index):
    """
    Gets the row number and coverts it to a label.
    inputs:
        starting_indices (list) - list of ints for the slices,
        row_index (int)
    """
    relaxed = (row_index >= starting_indices[0] and row_index < starting_indices[1]) \
        or (row_index > starting_indices[6])

    medium = (row_index >= starting_indices[2] and row_index < starting_indices[3]) \
        or (row_index >= starting_indices[4] and row_index < starting_indices[5])

    stressed = (row_index >= starting_indices[1] and row_index < starting_indices[2]) \
        or (row_index >= starting_indices[3] and row_index < starting_indices[4]) \
        or (row_index >= starting_indices[5] and row_index < starting_indices[6])

    if relaxed:
        return 1.0
    elif medium:
        return 3.0
    elif stressed:
        return 5.0


def process_data(marker_data_path="./csv_files/Marker Data", csv_files_path="./csv_files"):
    # Create a new folder called preprocessed_data if it does not exist.
    Path(csv_files_path).joinpath("preprocessed_data").mkdir(parents=True, exist_ok=True)
    # Grabs all the csv files from the path.
    csv_files = list(Path(csv_files_path).glob("*.csv"))
    all_drives = []
    # Keep only filenames from each path.
    csv_files_names = [Path(f).stem for f in csv_files]
    # Read Marker Data
    marker_data = pd.read_csv(Path(marker_data_path).joinpath("marker_info.csv"))
    # Each column represents the time the drive was spent in that state.
    # For example, Drive05 spent the first 15.13 minutes resting, then
    # 16 minutes driving in the city, then 7.74 minutes driving in Highway etc.
    processed_markers = calculate_starting_times(data=marker_data)
    # Select only the drives that have full data.
    selected_drives = processed_markers["Driver"]
    for drive in selected_drives:
        if drive.lower() in csv_files_names:
            print("\n", f"Processing {drive}.")
            idx = csv_files_names.index(drive.lower())
            data = pd.read_csv(csv_files[idx])
            # Marker column is not needed.
            data = data.drop(['marker-mV'], axis=1)
            # Get the starting time for each category (rest, highway and city)
            starting_indices = get_starting_indices(processed_markers, drive)
            # Label the data according the driving times.
            data['Stress'] = data.apply(
                lambda row: label_data(starting_indices, row.name), axis=1
            )
            # Save data for each drive.
            data.to_csv(f"{csv_files_path}/preprocessed_data/{drive}.csv", index=False)
            data['Drive'] = drive
            all_drives.append(data)
    # Join all the driver data into one big dataframe.
    all_drives_data = pd.concat(all_drives, ignore_index=True)
    # Save the dataframe.
    all_drives_data.to_csv(f"{csv_files_path}/preprocessed_data/all_drives.csv", index=False)
    print(f"Data saved in the {csv_files_path}/preprocessed_data directory.")


# Reference
# 1. Akbas A., “Evaluation of the physiological data indicating the dynamic stress level of drivers”,
# Scientific Research and Essays, Vol. 6(2), pp. 430-439, 18 Jan. 2011.
# 2. corilei (2019)
# https://www.kaggle.com/corilei/stress-recognition-in-automobile-driver-1/notebook
