from pathlib import Path

import numpy as np
import pandas as pd
from a_convert_to_csv import read_data
from scipy import signal


def create_resp_features(resp, band):
    """
    Calculates Welch Average Modified Periodogram with Hanning Window.
    inputs:
        resp: (array) respiration signal,
        band: (string) Band type. Options: ulf, vlf, lf, hf.
    Example:
        create_resp_features(resp=x, band="ulf")
    """
    # Welch Average Modified Periodogram with Hanning window.
    frequencies, powers = signal.welch(
        x=resp,
        fs=15.5,
        window="hann",
        nperseg=int(len(resp)),
        noverlap=int(len(resp)/2)
    )

    # Frequency bands to extract the sum for each interval.
    fbands = {'ulf': (0.0, 0.1), 'vlf': (0.1, 0.2), 'lf': (0.2, 0.3), 'hf': (0.3, 0.4)}

    return np.sum(
        powers[np.where(
            (frequencies >= fbands[band][0]) &
            (frequencies <= fbands[band][1])
        )]
    )


def create_gsr_features(gsr, return_type):
    """
    Calculates Peaks, Magnitude, Duration, and Area of gsr signal.
    inputs:
        gsr: (array) gsr signal,
        return_type: (string) Input to return. Options: frequency, magnitude, duration, area.
    Example:
        create_gsr_features(gsr=x, return_type="frequency")
    """
    # Find peaks for the signal
    peaks, _ = signal.find_peaks(gsr)
    # Find widths, widths heights
    widths, width_heights, left_lps, right_lps = signal.peak_widths(
        gsr,
        peaks,
        rel_height=1
    )
    # Fgsr Frequency - Frequency of Peaks
    if return_type == "frequency":
        return len(peaks)
    # Fgsr Magnitude - Sum of the magnitude (peak - initial height)
    elif return_type == "magnitude":
        return sum(gsr[peaks] - width_heights)
    # Fgsr Duration - Sum of the duration of peak (peak position - onset)
    elif return_type == "duration":
        return sum(peaks - left_lps)
    # Fgsr Area - sum(1/2 * Magnitude * Duration)
    elif return_type == "area":
        return sum(0.5 * (gsr[peaks] - width_heights) * (peaks - left_lps))


def create_hrv_feature(hr):
    """
    Calculates Lomscargles Periodogram for hr signal.
    inputs:
        hr: (array) hr signal,
    Example:
        create_hrv_feature(hr)
    """
    # Creating an evenly spaced array. (0.01, 0.02, ..., 0.5)
    periods = np.linspace(0.01, 0.5, 50)
    # Convert the period to Angular Frequency - (2*pi) / T
    angular_frequencies = (2 * np.pi) / periods
    # Create a timestamp for the algorithm
    timestamp = np.linspace(1/15.5, len(hr) * (1/15.5), num = len(hr))
    try:
        # Compute Lomb Scargle
        lomb = signal.lombscargle(timestamp, hr, angular_frequencies, normalize=True)
        # Get the ratio of low frequency (0-0.08 Hz) and high frequency (0.15-0.5 Hz)
        ratio = sum(lomb[0:8]) / sum(lomb[14:])
    except ZeroDivisionError:
        print("Failed to calculate Lomb, returning mean instead.")
        ratio = np.mean(hr)
    return ratio


def filter_minutes(x, end_minute, start_minute):
    """
    Filters the data according to given time interval.
    end_minute - start_minute has to be larger or equal to 3.
    inputs:
        x: (series) cumulative sum in group by clause.
        end_minute: (int) ending time interval.
        start_minute: (int) starting time interval.
    Example:
        filter_minutes(x, 9, 3)
    """
    # Get cumulative count for group by clause.
    cumulative_count = pd.Series(np.arange(len(x)), x.index).reset_index(drop=True)
    maximum = max(cumulative_count)
    minimum = min(cumulative_count)
    result = []
    for row in cumulative_count:
        if (maximum - minimum < end_minute - start_minute) or (row >= start_minute and row <= end_minute):
            result.append(True)
        else:
            result.append(False)
    return pd.Series(result)


# Dictionary of functions to apply to downsampled data for the pandas group by method.
functions_to_apply = {
    "EMG": np.mean,
    "footGSR": [
        np.mean,
        np.std,
        lambda x: create_gsr_features(gsr=x, return_type="frequency"),
        lambda x: create_gsr_features(gsr=x, return_type="magnitude"),
        lambda x: create_gsr_features(gsr=x, return_type="duration"),
        lambda x: create_gsr_features(gsr=x, return_type="area")
    ],
    "handGSR": [
        np.mean,
        np.std,
        lambda x: create_gsr_features(gsr=x, return_type="frequency"),
        lambda x: create_gsr_features(gsr=x, return_type="magnitude"),
        lambda x: create_gsr_features(gsr=x, return_type="duration"),
        lambda x: create_gsr_features(gsr=x, return_type="area")
    ],
    "HR": [
        np.mean,
        np.std,
        create_hrv_feature
    ],
    "RESP": [
        np.mean,
        np.std,
        lambda x: create_resp_features(resp=x, band="ulf"),
        lambda x: create_resp_features(resp=x, band="vlf"),
        lambda x: create_resp_features(resp=x, band="lf"),
        lambda x: create_resp_features(resp=x, band="hf")
    ],
    "Stress": np.mean,
}


# Final column names for the final dataset.
column_names = [
    "time",
    # EMG
    "EMG_mean",
    # Foot GSR
    "footGSR_mean", "footGSR_std", "footGSR_frequency",
    "footGSR_magnitude", "footGSR_duration", "footGSR_area",
    # Hand GSR
    "handGSR_mean", "handGSR_std", "handGSR_frequency",
    "handGSR_magnitude", "handGSR_duration", "handGSR_area",
    # HR
    "HR_mean", "HR_std", "HRV_ratio",
    # RESP
    "RESP_mean", "RESP_std", "RESP_ulf", "RESP_vlf",
    "RESP_lf", "RESP_hf",
    # Stress
    "Stress_mean"
]


def calculate_features(functions_to_apply, column_names, downsample_frequency="10S", time_range=(3, 9), path="./csv_files/preprocessed_data", folder_name="final_data"):
    all_drives = []
    for file in read_data(path=path, extension="*.csv"):
        # Create a new folder called final_data if it does not exist.
        Path(path).joinpath(folder_name).mkdir(parents=True, exist_ok=True)
        file_name, data = file
        if file_name == "all_drives":
            continue
        print(f"\n\nCalculating Features For {file_name}")
        # Sampling frequency
        fs = 15.5
        start_minute = int(fs * time_range[0] * 60)
        end_minute = int(fs * time_range[1] * 60)
        # Remove units and empty spaces from column names.
        data.columns = [column.split("-")[0].replace(" ", "") for column in data.columns]
        # Remove NaN values in stress column.
        data = data.dropna().reset_index(drop=True)
        # Get the indices where Stress is changing.
        # It returns the place in the dataframe where we are moving from one driving mode into another.
        stress_change_index = data.ne(data.shift()).apply(lambda x: x.index[x].tolist())["Stress"]
        # To create proper ranges, we add the last position in the data.
        stress_change_index.append(len(data))
        labels = ['Rest1', 'City1', 'Highway1', 'City2', 'Highway2', 'City3', 'Rest2']
        # Create ranges from the indices of stress change.
        # right = false ignores the right side selection, i.e: (x, y) not (x, y]
        ranges = pd.cut(data.index, bins=stress_change_index, labels=labels, right=False)
        # Filter data according to the time interval.
        selected_minutes = data.groupby(ranges, as_index=False).apply(
            lambda x: filter_minutes(x, end_minute, start_minute)
        ).reset_index(drop=True)
        selected_minutes = selected_minutes[selected_minutes == False].index
        data = data.drop(selected_minutes).reset_index(drop=True)
        stress_change_index = data.ne(data.shift()).apply(lambda x: x.index[x].tolist())["Stress"]
        stress_change_index.append(len(data))
        ranges = pd.cut(data.index, bins=stress_change_index, labels=labels, right=False)
        # Group by each driving mode and get their means and standard deviations.
        means = data.groupby(ranges).agg({'std', 'mean'})
        # Normalize the data.
        columns_to_normalize = ["EMG", "RESP", "HR", "footGSR", "handGSR"]
        for col in columns_to_normalize:
            # MinMax Normalization
            if col in ["footGSR", "handGSR"]:
                data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
            else:
                # Normalizing by substracting the mean of the first rest period.
                data[col] = data[col] - means.loc["Rest1", col]["mean"]
        # Create a time column and set it as the data index.
        # f"{round(1/fs, 4)}S" is the sampling rate.
        data["time"] = pd.date_range(start=0, periods=len(data), freq=f"{round(1/fs, 4)}S")
        data.set_index("time", inplace=True)
        # group by each driving mode (rest1, hightway1,....) and downsample to n seconds.
        # Aggregate each group into functions to apply (i.e: means, std,...)
        data = data.groupby(ranges).resample(downsample_frequency).agg(functions_to_apply)
        # Convert multIndex into single index where colnames are changed to "EMG_mean", "HR_std", ...
        data.columns = ["_".join(col) for col in data.columns]
        # Remove indices and keep time as a column.
        data = data.reset_index(level=0, drop=True).reset_index()
        data.columns = column_names
        # Show only time in datetime.
        data['time'] = data['time'].dt.time
        print(data['Stress_mean'].value_counts())
        print(f"{len(data.columns) - 2} Columns")
        data.to_csv(f"{path}/{folder_name}/{file_name}.csv", index=False)
        data['Drive'] = file_name
        all_drives.append(data)
    all_drives = pd.concat(all_drives, ignore_index=True)
    all_drives.to_csv(f"{path}/{folder_name}/all_drives.csv", index=False)
    print("\n\nSaved the all drives file too.")
    print(f"Data Saved in the {path}/{folder_name} directory.")


# calculate_features(
    # functions_to_apply,
    # column_names,
    # downsample_frequency="10S",
    # folder_name = "final_data"
# )
