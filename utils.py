import scipy
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
from scipy.signal import butter, freqz

from os import listdir, getcwd
from os.path import isdir
from os.path import join

from pathlib import Path  
from glob import glob
from natsort import natsorted


def find_best_order_aic(orders, avg_cutoff_frequency):
    """Find the best filter order using the Akaike Information Criterion (AIC).

    Args:
        orders (array-like): Array of filter orders to search.
        avg_cutoff_frequency (float): Average cutoff frequency.

    Returns:
        int: Best filter order based on AIC criterion.

    """

    def compute_aic(h):
        """Compute the Akaike Information Criterion (AIC) based on the frequency response."""
        return np.sum(np.log(np.abs(h)**2))

    def objective_function(order):
        """Objective function to evaluate the AIC for a given filter order."""
        b, a = butter(N=order, Wn=avg_cutoff_frequency, btype='low', analog=False)
        w, h = freqz(b, a)
        return compute_aic(h)

    best_order = None
    best_aic = np.inf

    # Optimization loop
    for order in orders:
        aic = objective_function(order)

        if aic < best_aic:
            best_aic = aic
            best_order = order

    return best_order


def calculate_avg_cutoff_frequency(frequency_domain_data):
    """
    Calculate the average cutoff frequency from a list of frequency domain data.

    Args:
        frequency_domain_data (list): List of frequency domain data arrays.

    Returns:
        float: Average cutoff frequency rounded to three decimal places.
    """
    cutoff_frequencies = []

    for freq_dom_inst in frequency_domain_data:
        frequencies = np.fft.fftfreq(freq_dom_inst.shape[0])
        cutoff_frequency = frequencies[np.argmax(freq_dom_inst)]
        cutoff_frequencies.append(cutoff_frequency)

    avg_cutoff_frequency = np.mean(cutoff_frequencies).round(3)
    
    return avg_cutoff_frequency



def transform_to_frequency_domain(data):
    """Applies fourier transform to each instance in data

    Args:
        data: List of dataframes (each df is a windowed instance)

    Returns:
        A list of one-dimensional DFT of a sequence of input data. 
        Each item in list, contains the magnitudes of each frequency component in the input signal.
    """
    transformed_data = [np.abs(fft(df.values.flatten())) for df in data]
    return transformed_data


def format_wrist_data(data_path, dest_path=None):
    """
        - accelaration-gyroscope measurements are joined with "elapsed (s)" foreign key
        - length not equal, why?
    """
    classes = [cls_f for cls_f in listdir(data_path) if isdir(join(data_path, cls_f))]
    
    new_data_path = join(getcwd() if not dest_path else dest_path, 'data')
    
    for cls in classes:
        print(30 * '=' + cls + 30 * '=')

        # get folder class path
        class_path = join(data_path, cls)

        # seperate gyr/acc measurements and sort files alphanumerically
        acc_files = natsorted(glob(class_path + '/*acc*.csv'))
        gyr_files = natsorted(glob(class_path + '/*gyr*.csv'))

        print(f"Class '{cls}' contains {len(acc_files) + len(gyr_files)} files")

        for i, (acc_path, gyr_path) in enumerate(zip(acc_files, gyr_files)):

            # read pairwise csv's
            acc = pd.read_csv(acc_path)
            gyr = pd.read_csv(gyr_path)

            # merge measurements
            merged_acc_gyr = pd.merge(left=acc, right=gyr.loc[:, 'elapsed (s)': 'z-axis (deg/s)'],
                                    left_on='elapsed (s)', right_on='elapsed (s)')

            # create path and root directories if neeeded
            filepath = Path(join(new_data_path, cls, f'data_{cls[-1]}_{i}.csv'))  
            filepath.parent.mkdir(parents=True, exist_ok=True)  

            # save df as csv
            merged_acc_gyr.to_csv(filepath, index=False)
        print(f"{len(acc_files)} total sessions after merging the axes")
        
        
def sliding_window_pd(df, ws=500, overlap=250, w_type="hann", w_center=True, print_stats=False):
    """Applies the sliding window algorithm to the DataFrame rows.

    Args:
        df: The DataFrame with all the values that will be inserted to the sliding window algorithm.
        ws: The window size in number of samples.
        overlap: The hop length in number of samples.
        w_type: The windowing function.
        w_center: If False, set the window labels as the right edge of the window index. If True, set the window
                labels as the center of the window index.
        print_stats: Print statistical inferences from the process (Default: False).

    Returns:
        A list of DataFrames each one corresponding to a produced window.
    """
    counter = 0
    windows_list = list()
    # min_periods: Minimum number of observations in window required to have a value;
    # For a window that is specified by an integer, min_periods will default to the size of the window.
    for window in df.rolling(window=ws, step=overlap, min_periods=ws, win_type=w_type, center=w_center):
        if window[window.columns[0]].count() >= ws:
            if print_stats:
                print("Print Window:", counter)
                print("Number of samples:", window[window.columns[0]].count())
            windows_list.append(window)
        counter += 1
    if print_stats:
        print("List number of window instances:", len(windows_list))

    return windows_list


def apply_filter(arr, order=5, wn=0.1, filter_type="lowpass"):
    """Apply filter to the multi-axis signal.

    Args:
        arr: The initial NumPy signal array values.
        order: The order of the filter.
        wn: The critical frequency or frequencies.
        filter_type: The type of filter. {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}

    Returns:
        NumPy Array with the filtered signal.
    """
    fbd_filter = scipy.signal.butter(N=order, Wn=wn, btype=filter_type, output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos=fbd_filter, x=arr, padlen=0)

    return filtered_signal


def filter_instances(instances_list, order, wn, ftype):
    """Apply filter to a list of windows (each window is a DataFrame).

    Args:
        instances_list: List of DataFrames.
        order: The order of the filter.
        wn: The critical frequency or frequencies.
        ftype: The type of filter. {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}

    Returns:

    """
    filtered_instances_list = list()
    for item in instances_list:
        filtered_instance = item.apply(apply_filter, args=(order, wn, ftype))
        filtered_instances_list.append(filtered_instance)
    print("Number of filtered instances in the list:", len(filtered_instances_list))

    return filtered_instances_list


def flatten_instances_df(instances_list):
    """Flatten each instance and create a DataFrame with the whole flattened instances.

    Args:
        instances_list: The list of DataFrames to be flattened

    Returns:
        A DataFrame that includes the whole flattened DataFrames
    """
    flattened_instances_list = list()
    for item in instances_list:
        instance = item.to_numpy().flatten()
        flattened_instances_list.append(instance)
    df = pd.DataFrame(flattened_instances_list)

    return df


def df_rebase(df, order_list, rename_dict):
    """Changes the order and name of DataFrame columns to the project's needs for readability.

    Args:
        df: The pandas DataFrame.
        order_list: List object that contains the proper order of the default sensor column names.
        rename_dict: Dictionary object that contains the renaming list based on the project needs.

    Returns:
        A DataFrame with the new columns order and names.
    """
    df = df[order_list]  # keep and re-order only the necessary columns of the initial DataFrame
    df = df.rename(columns=rename_dict)  # rename the columns

    return df


def rename_df_column_values(np_array, y, columns_names=("acc_x", "acc_y", "acc_z")):
    """Creates a DataFrame with a "y" label column and replaces the values of the y with the index
    of the unique values of y.

    Args:
        np_array: 2D NumPy array.
        y: List with the y labels
        columns_names: List with the DF columns names.

    Returns:
        DataFrame with the multi-axes values and the target labels column.
    """
    arr_y = np.array(y)  # list to numpy array
    unique_values_list = np.unique(arr_y)  # unique list of values

    df = pd.DataFrame(np_array, columns=columns_names)
    df["y"] = y

    # replace the row item value in the y column of the df, with its index in the unique list
    for idx, x in enumerate(unique_values_list):
        df["y"] = np.where(df["y"] == x, idx, df["y"])

    return df


def encode_labels(instances_list):
    """Encodes target labels.

    Args:
        instances_list: List of instances to be encoded.

    Returns:
        The encoded array.
    """
    le = preprocessing.LabelEncoder()
    le.fit(instances_list)
    instances_arr = le.transform(instances_list)

    return instances_arr
