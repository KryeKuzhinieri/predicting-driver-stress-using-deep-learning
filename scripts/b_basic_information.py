from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from a_convert_to_csv import read_data


DIR = "/media/kryekuzhinieri/Data/Thesis/Datasets 2/Stress Recognition In Automobile Drivers/csv_files"


def get_all_csv_info(path="./csv_files"):
    """
    Code to create subplots for all the columns in each csv file in the
        provided working directory.
    Path - location of the csv files (str)
    """
    print("Reading csv files...")
    Path(path).joinpath("images").mkdir(parents=True, exist_ok=True)
    driver_columns = {}
    for file in read_data(path=path, extension="*.csv"):
        file_name, data = file
        print(f"Creating Subplots For {file_name}")
        create_subplots(path, data, file_name)
        column_information(data, file_name, driver_columns)
    # print(driver_columns)
    print("Successfully created subplots.")


def create_subplots(file_path, data, file_name):
    """
    Function to create subplots for each column in the dataset.
    file_path - path where to save the image.
    Data - pandas dataframe with column names.
    file_name - name of the drive's.
    """
    fig, axs = plt.subplots(len(data.columns), 1, figsize=(15, 10))
    fig.suptitle(f"Physiological Signals For Driver {file_name}")
    for idx, col in enumerate(data.columns):
        axs[idx].plot(data.index, data[col])
        axs[idx].set_ylabel(col, fontsize=13)
        axs[idx].set_xlabel("Time", fontsize=13)
    fig.tight_layout(h_pad=2)
    image_path = Path(file_path).joinpath("images") / f"{file_name}_plots.png"
    plt.savefig(image_path)


def column_information(data, file_name, driver_columns):
    """
    Create dictionary with driver data.
    data - pandas data frame,
    file_name - name of the file,
    driver_columns - dictonary where to store the data.
    """
    driver_columns[file_name] = list(data.columns)

