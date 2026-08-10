# BV_processing.py
"""Behavioural/psychophysics data processing and plotting."""

import os
import re
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.integrate import quad

from plot_utils import check_dataframe, save_or_show


# Function to process each file and extract ID
def process_file(f, id_column='ID'):
    """
    Process a single file to extract its content and add an ID.

    Parameters:
        f (str): The file path.
        id_column (str): The name of the column to store the extracted ID (default is 'ID').

    Returns:
        pd.DataFrame: DataFrame containing the data from the file with an added ID column.
    """
    try:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"The file '{f}' does not exist or is not a valid file. Please provide a valid file path.")

        try:
            df = pd.read_excel(f)  # Read the Excel file
            if df.empty:
                raise ValueError(f"The file '{f}' is empty. Please provide a non-empty Excel file.")
        except Exception as e:
            print(f"Error: Unable to read the Excel file '{f}'. Ensure it is a valid Excel file (.xls or .xlsx format).", e)
            return None

        # Check if the ID can be extracted
        id_matches = re.findall(r'\d+', f)
        if len(id_matches) < 3:
            raise ValueError(f"Unable to extract ID from file name '{f}'. Expected a name format with three numeric parts (e.g., '999_[date]_[test]_[eye]_[amb]_[test]_[stage].xlsx').")

        # `f` is a full path, not just a filename, so earlier path segments (e.g. the
        # subject folder matched by `folder_head`) also contribute numeric matches;
        # the participant ID is specifically the third \d+ match, not the first.
        ID = int(id_matches[2])  # Extract the ID number from the file name
        df[id_column] = ID  # Add the ID to the DataFrame

        return df

    except Exception as e:
        print("Error in process_file:", e)
        return None



# Function to process all files in a directory based on a pattern
def process_files_total(base_path, pattern_file='cutoff', refined=False, remove_dirs=None, 
                        eye_column='Eye', stage_column='stage', amb_column='amb', 
                        position_column='Position', date_column='Date_time', amb_eye_column='AMB_eye', folder_head='sub_*'):
    """
    Process all files in the specified directory and return concatenated results.

    Parameters:
        base_path (str): The base directory containing the subdirectories.
        pattern_file (str): Pattern to filter files (default is 'cutoff').
        refined (bool): Flag to refine the selection of files (default is False).
        remove_dirs (list of str): Characters to filter out directories containing them (default is None).
        eye_column (str): Name of the 'Eye' column to annotate (default is 'Eye').
        stage_column (str): Name of the 'stage' column (default is 'stage').
        amb_column (str): Name of the 'amb' column (default is 'amb').
        position_column (str): Name of the 'Position' column (default is 'Position').
        date_column (str): Name of the 'Date_time' column (default is 'Date_time').
        amb_eye_column (str): Name of the 'AMB_eye' column for threshold calculations (default is 'AMB_eye').

    Returns:
        tuple: DataFrames containing processed data and thresholds.
    """
    try:
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"The base path '{base_path}' does not exist or is not a directory. Please provide a valid directory path.")

        # Get a list of all directories
        all_dirs = [d for d in glob(os.path.join(base_path, folder_head)) if os.path.isdir(d)] 

        # Filter out directories if specified
        if remove_dirs:
            all_dirs = [d for d in all_dirs if not any(char in os.path.basename(d) for char in remove_dirs)]

        all_cutoff_files = []

        # Loop through each subdirectory and list the files with the cutoff pattern in their names
        for dir in all_dirs:
            files = glob(os.path.join(dir, '**', f'*{pattern_file}*.xls'), recursive=True)
            if pattern_file == 'snellen':
                files = [f for f in files if not re.search("crowded", os.path.basename(f))]
            elif pattern_file == 'crowded':
                files = [f for f in files if re.search("crowded", os.path.basename(f))]

            if refined:
                files = [f for f in files if 'familiarisation' not in f]

            all_cutoff_files.extend(files)

        if not all_cutoff_files:
            print(f"Warning: No files found matching pattern '{pattern_file}' in the specified directories.")
            return None, None

        # Apply the process function to all files
        result = []
        for f in all_cutoff_files:
            processed_file = process_file(f, id_column='ID')
            if processed_file is not None:
                result.append(processed_file)

        # Annotate the result list
        for i, f in enumerate(all_cutoff_files):
            if 'LEFT' in f:
                result[i][eye_column] = 'Left'
            elif 'RIGHT' in f:
                result[i][eye_column] = 'Right'
            elif 'BOTH' in f:
                result[i][eye_column] = 'Both'
            else:
                print(f"Warning: Eye type not recognized in file '{f}'. Expected values are 'LEFT', 'RIGHT', or 'BOTH' in the file name.")

            if 'familiarisation' in f:
                result[i][stage_column] = 'familiarisation'
            elif 'pretraining' in f:
                result[i][stage_column] = 'pretraining'
            elif 'posttraining' in f:
                result[i][stage_column] = 'posttraining'
            else:
                print(f"Warning: Testing stage not recognized in file '{f}'. Expected values are 'familiarisation', 'pretraining', or 'posttraining' in the file name.")

            result[i][amb_column] = 'Strong'
            if 'AMB' in f:
                result[i][amb_column] = 'Amb'
            elif 'BOTH' in f:
                result[i][amb_column] = 'Both'

        # Bind all the results into a single DataFrame
        df = pd.concat(result, ignore_index=True)

        # Handle NA values in the Position column
        if pattern_file != 'asteroid' and position_column in df.columns:
            df[position_column] = df[position_column].fillna(0)
        else:
            print(f"Warning: '{position_column}' column is missing or not applicable for 'asteroid' pattern. The DataFrame should contain a '{position_column}' column if the pattern_file is not 'asteroid'.")

        df = df.dropna()

        df[eye_column] = pd.Categorical(df[eye_column], categories=["Left", "Right", "Both"], ordered=True)
        df[stage_column] = pd.Categorical(df[stage_column], categories=["familiarisation", "pretraining", "posttraining"], ordered=True)

        # Get the thresholds
        if pattern_file == 'asteroid':
            df_thresholds = []
        else:
            if amb_eye_column not in df.columns:
                print(f"Warning: '{amb_eye_column}' column is missing in the DataFrame. Ensure that your data includes an '{amb_eye_column}' column for threshold calculations.")
                return df, None
            df_thresholds = df[df[amb_eye_column] == 'MEAN'].dropna()

        if pattern_file != 'asteroid':
            if date_column in df_thresholds.columns:
                df_thresholds[date_column] = pd.Categorical(df_thresholds[date_column].astype(str).str[:10])
                df[date_column] = pd.Categorical(df[date_column].astype(str).str[:10])
            else:
                print(f"Warning: '{date_column}' column is missing in one or more DataFrames. Ensure '{date_column}' is present and in string format.")

        return df, df_thresholds

    except Exception as e:
        print("Error in process_files_total:", e)
        return None, None

################################################################################


def log_bp(x):
    """
    Convert an interocular contrast ratio to a log Balance Point (logBP) score:
    the absolute log10 distance from equal contrast in both eyes (x=1, logBP=0).
    Larger values mean a bigger imbalance between the two eyes, in either direction.

    Parameters:
        x (float or np.ndarray): Contrast ratio between the two eyes.

    Returns:
        float or np.ndarray: logBP value(s).
    """
    return abs(np.log10(x))


def log_parabola(f, a, b, f_peak):
    """
    Log-parabola CSF model in log10(sensitivity) space.

    Parameters:
        f (float or np.ndarray): Spatial frequency, same linear units as f_peak (cpd).
        a (float): Peak log10(sensitivity).
        b (float): Curvature (bandwidth) of the parabola.
        f_peak (float): Peak spatial frequency.

    Returns:
        float or np.ndarray: log10(sensitivity) predicted at `f`.
    """
    return a - b * (np.log10(f) - np.log10(f_peak)) ** 2


################################################################################



def calculate_visual_metrics(
        df_se_thresholds, 
        screen_width_mm, 
        screen_height_mm, 
        screen_resolution_width, 
        screen_resolution_height, 
        distance_mm,
        opto_size_pixel_column='opto_size_pixel'):
    """
    Calculate visual metrics based on screen dimensions, resolution, and viewing distance.

    Parameters:
        df_se_thresholds (pd.DataFrame): DataFrame containing threshold data, expected to include a column 
            for optical size in pixels. Default column name is 'opto_size_pixel'.
        screen_width_mm (float): Width of the screen in millimeters.
        screen_height_mm (float): Height of the screen in millimeters.
        screen_resolution_width (int): Horizontal resolution of the screen in pixels.
        screen_resolution_height (int): Vertical resolution of the screen in pixels.
        distance_mm (float): Distance from the viewer to the screen in millimeters.
        opto_size_pixel_column (str): Name of the column in `df_se_thresholds` that contains the optical 
            size in pixels. Default is 'opto_size_pixel'.

    Returns:
        pd.DataFrame: A DataFrame with calculated visual metrics including:
            - 'pixel_width_mm': Pixel width in mm.
            - 'pixel_height_mm': Pixel height in mm.
            - 'average_pixel_size_mm': Average pixel size in mm.
            - 'letter_height_mm': Letter height in mm.
            - 'visual_angle_rad': Visual angle in radians.
            - 'visual_angle_arcmin': Visual angle in arcminutes.
            - 'snellen_denominator': Snellen denominator.
            - 'logMAR': Logarithm of the minimum angle of resolution.

    Raises:
        ValueError: If the required column is missing or if the input dimensions are not positive.
        TypeError: If a required parameter is not of the expected type.
    """
    # Check if df_se_thresholds is a DataFrame
    if not isinstance(df_se_thresholds, pd.DataFrame):
        raise TypeError("`df_se_thresholds` must be a pandas DataFrame.")

    # Check if necessary column exists in the DataFrame
    if opto_size_pixel_column not in df_se_thresholds.columns:
        raise ValueError(
            f"The specified column '{opto_size_pixel_column}' is missing from the DataFrame. "
            "This column should contain optical size in pixels."
        )

    # Validate numerical parameters
    for param_name, param_value in {
        "screen_width_mm": screen_width_mm, 
        "screen_height_mm": screen_height_mm, 
        "screen_resolution_width": screen_resolution_width,
        "screen_resolution_height": screen_resolution_height, 
        "distance_mm": distance_mm
    }.items():
        if not isinstance(param_value, (int, float)):
            raise TypeError(f"`{param_name}` must be a number.")
        if param_value <= 0:
            raise ValueError(f"`{param_name}` must be a positive value.")

    try:
        # Calculate pixel size in mm
        df_se_thresholds['pixel_width_mm'] = screen_width_mm / screen_resolution_width
        df_se_thresholds['pixel_height_mm'] = screen_height_mm / screen_resolution_height
        df_se_thresholds['average_pixel_size_mm'] = (
            df_se_thresholds['pixel_width_mm'] + df_se_thresholds['pixel_height_mm']
        ) / 2

        # Convert pixels to physical size in mm. Snellen optotypes are drawn on a 5x5
        # stroke-width grid, so overall letter height is 5x the stroke width in `opto_size_pixel`.
        df_se_thresholds['letter_height_mm'] = (
            df_se_thresholds[opto_size_pixel_column] * 5 * df_se_thresholds['average_pixel_size_mm']
        )

        # Calculate visual angle in radians
        df_se_thresholds['visual_angle_rad'] = 2 * np.arctan(
            df_se_thresholds['letter_height_mm'] / (2 * distance_mm)
        )

        # Convert visual angle to arcminutes
        df_se_thresholds['visual_angle_arcmin'] = (
            df_se_thresholds['visual_angle_rad'] * (180 / np.pi) * 60
        )

        # Convert visual angle to Snellen denominator. A normal-acuity (20/20) letter
        # subtends 5 arcmin, so dividing by 5 and scaling by the 20 ft numerator gives
        # the denominator of the equivalent 20/X Snellen fraction.
        df_se_thresholds['snellen_denominator'] = (
            20 * df_se_thresholds['visual_angle_arcmin'] / 5
        )

        # logMAR = log10(Minimum Angle of Resolution) = log10(Snellen denominator / 20)
        df_se_thresholds['logMAR'] = np.log10(
            df_se_thresholds['snellen_denominator'] / 20
        )

    except Exception as e:
        raise RuntimeError("An error occurred during the calculation of visual metrics.") from e

    return df_se_thresholds


################################################################################



def process_threshold_data(
        df_co_thresholds=None, 
        df_cs_thresholds=None, 
        df_bi_thresholds=None, 
        df_di_thresholds=None, 
        df_as_thresholds=None, 
        df_se_thresholds=None, 
        df_sc_thresholds=None,
        co_freq_col='Frequency_cpd', 
        cs_contrast_col='Contrast', 
        cs_staircase_col='Staircase', 
        cs_sensitivity_col='Sensitivity',
        cs_freq_col='Frequency_dva',
        bi_contrast_col='Contrast_FE', 
        bi_staircase_col='Staircase', 
        bi_logBP_col='logBP', 
        bi_freq_col='Frequency_dva',
        di_freq_col='Frequency_cpp', 
        di_noise_col='Noise100',
        as_disparity_arcsec_col='disparitydisparity_arcsec', 
        as_disparity_log_col='disparitylog',
        se_snellen_col='snellen_denominator', 
        se_logMAR_col='logMAR',
        sc_snellen_col='snellen_denominator', 
        sc_logMAR_col='logMAR',
        merge_output=True):
    """
    Processes and aggregates various threshold data DataFrames, selectively based on provided DataFrames. 

    Parameters:
        Each threshold DataFrame is optional; only the ones provided are processed.
            df_co_thresholds (pd.DataFrame): Cut-off frequency threshold data.
            df_cs_thresholds (pd.DataFrame): Contrast sensitivity threshold data.
            df_bi_thresholds (pd.DataFrame): Interocular imbalance threshold data.
            df_di_thresholds (pd.DataFrame): Dichoptic threshold data.
            df_as_thresholds (pd.DataFrame): Stereoacuity (asteroid) threshold data.
            df_se_thresholds (pd.DataFrame): Snellen equivalent acuity data.
            df_sc_thresholds (pd.DataFrame): Snellen crowded acuity data.
        The `*_col` arguments let you remap each DataFrame's source column names
        (e.g. `co_freq_col`, `cs_contrast_col`, ...) to this function's expected names;
        see the defaults above for which columns are read from each DataFrame.
        merge_output (bool): If True, merges the processed DataFrames if more than one is processed.

    Returns:
        pd.DataFrame or dict: Merged DataFrame or dictionary of processed DataFrames.

    Raises:
        ValueError: If required columns are missing in any provided DataFrame.
        TypeError: If provided data isn't a pandas DataFrame.
    """
    # Define functions for individual DataFrame processing
    def process_co(df):
        check_columns(df, ['ID', 'stage', 'amb', co_freq_col], 'df_co_thresholds')
        processed = df[['ID', 'stage', 'amb', co_freq_col]].copy()
        processed.rename(columns={co_freq_col: 'cutoff_frequency_cpd'}, inplace=True)
        return processed

    def process_cs(df):
        check_columns(df, ['ID', 'stage', 'amb', cs_contrast_col, cs_staircase_col, cs_sensitivity_col, cs_freq_col], 'df_cs_thresholds')
        processed = df.groupby(['ID', 'stage', 'amb']).agg({
            cs_contrast_col: lambda x: x.tolist(),
            cs_staircase_col: lambda x: x.tolist(),
            cs_sensitivity_col: lambda x: x.tolist(),
            cs_freq_col: lambda x: x.tolist()
        }).reset_index()
        processed.rename(columns={
            cs_contrast_col: 'CS_contrasts',
            cs_staircase_col: 'CS_staircases',
            cs_sensitivity_col: 'CS_sensitivities',
            cs_freq_col: 'CS_frequencies_cpd'
        }, inplace=True)
        return processed

    def process_bi(df):
        check_columns(df, ['ID', 'stage', 'amb', bi_contrast_col, bi_staircase_col, bi_logBP_col, bi_freq_col], 'df_bi_thresholds')
        processed = df.groupby(['ID', 'stage', 'amb']).agg({
            bi_contrast_col: lambda x: x.tolist(),
            bi_staircase_col: lambda x: x.tolist(),
            bi_logBP_col: lambda x: x.tolist(),
            bi_freq_col: lambda x: x.tolist()
        }).reset_index()
        processed.rename(columns={
            bi_contrast_col: 'IntImb_Contrast_FE',
            bi_staircase_col: 'IntImb_staircases',
            bi_logBP_col: 'IntImb_logBP',
            bi_freq_col: 'IntImb_frequencies_cpd'
        }, inplace=True)
        return processed

    def process_di(df):
        check_columns(df, ['ID', 'stage', 'amb', di_freq_col, di_noise_col], 'df_di_thresholds')
        processed = df[['ID', 'stage', 'amb', di_freq_col, di_noise_col]].copy()
        processed.rename(columns={di_freq_col: 'Dichop_frequency_cpd', di_noise_col: 'Dichop_Noise'}, inplace=True)
        return processed

    def process_as(df):
        check_columns(df, ['ID', 'stage', 'amb', as_disparity_arcsec_col, as_disparity_log_col], 'df_as_thresholds')
        processed = df[['ID', 'stage', 'amb', as_disparity_arcsec_col, as_disparity_log_col]].copy()
        processed.rename(columns={as_disparity_arcsec_col: 'Ast_disparity_arcsec', as_disparity_log_col: 'Ast_disparity_log_arcsec'}, inplace=True)
        return processed

    def process_se(df):
        check_columns(df, ['ID', 'stage', 'amb', se_snellen_col, se_logMAR_col], 'df_se_thresholds')
        processed = df[['ID', 'stage', 'amb', se_snellen_col, se_logMAR_col]].copy()
        processed.rename(columns={se_snellen_col: 'SE_denominator', se_logMAR_col: 'SE_logMAR'}, inplace=True)
        return processed

    def process_sc(df):
        check_columns(df, ['ID', 'stage', 'amb', sc_snellen_col, sc_logMAR_col], 'df_sc_thresholds')
        processed = df[['ID', 'stage', 'amb', sc_snellen_col, sc_logMAR_col]].copy()
        processed.rename(columns={sc_snellen_col: 'SC_denominator', sc_logMAR_col: 'SC_logMAR'}, inplace=True)
        return processed

    # Column validation delegated to the shared helper
    check_columns = check_dataframe

    # Dictionary of DataFrames and their processing functions
    df_functions = {
        'df_co_thresholds': (df_co_thresholds, process_co),
        'df_cs_thresholds': (df_cs_thresholds, process_cs),
        'df_bi_thresholds': (df_bi_thresholds, process_bi),
        'df_di_thresholds': (df_di_thresholds, process_di),
        'df_as_thresholds': (df_as_thresholds, process_as),
        'df_se_thresholds': (df_se_thresholds, process_se),
        'df_sc_thresholds': (df_sc_thresholds, process_sc)
    }

    # Process each provided DataFrame and collect results
    processed_dfs = {}
    for df_name, (df, func) in df_functions.items():
        if df is not None:
            try:
                processed_dfs[df_name] = func(df)
                print(f"Processed: {df_name}")
            except (ValueError, TypeError) as e:
                print(f"Error processing {df_name}: {e}")

    # Merge option
    if merge_output and len(processed_dfs) > 1:
        final_df = None
        for df in processed_dfs.values():
            final_df = df if final_df is None else pd.merge(final_df, df, on=['ID', 'stage', 'amb'], how='outer')
        return final_df
    elif merge_output:
        print("Only one DataFrame was processed; merge not performed.")

    return processed_dfs if not merge_output else list(processed_dfs.values())[0]


################################################################################


def calculate_training_differences(final_df, 
                                    id_col='ID', 
                                    amb_col='amb', 
                                    stage_col='stage', 
                                    pre_stage='pretraining', 
                                    post_stage='posttraining',
                                    measures=None):
    """
    Separates the final DataFrame into pretraining and posttraining DataFrames,
    merges them, and calculates the differences for specified measures.

    Parameters:
        final_df (pd.DataFrame): The DataFrame containing combined training data.
        id_col (str): The column name for the ID field (default: 'ID').
        amb_col (str): The column name for the amblyopia field (default: 'amb').
        stage_col (str): The column name for the stage field (default: 'stage').
        pre_stage (str): The value representing pretraining in the stage column (default: 'pretraining').
        post_stage (str): The value representing posttraining in the stage column (default: 'posttraining').
        measures (list of str): List of measures for which to calculate differences. 
                                 If None, defaults to a predefined list.

    Returns:
        pd.DataFrame: A DataFrame containing the differences between pretraining 
                      and posttraining for the specified measures.

    Raises:
        ValueError: If required columns are missing from the DataFrame.
        TypeError: If the provided data is not a pandas DataFrame.
    """

    # Check the final DataFrame
    check_dataframe(final_df, [id_col, amb_col, stage_col])

    # Set default measures if not provided
    if measures is None:
        measures = [
            'cutoff_frequency_cpd', 'CS_contrasts', 'CS_sensitivities', 
            'CS_frequencies_cpd', 'IntImb_Contrast_FE', 'IntImb_logBP', 
            'Dichop_Noise', 'Ast_disparity_arcsec', 
            'Ast_disparity_log_arcsec', 'SE_logMAR', 'SC_logMAR'
        ]

    # Separate the data into pretraining and posttraining DataFrames
    pretraining_df = final_df[final_df[stage_col] == pre_stage].copy()
    posttraining_df = final_df[final_df[stage_col] == post_stage].copy()

    # Merge the pretraining and posttraining DataFrames
    merged_df = pd.merge(
        pretraining_df, posttraining_df,
        on=[id_col, amb_col],  # Merge on relevant columns
        suffixes=('_pre', '_post')
    )

    def subtract_lists(list1, list2):
        """Subtracts two lists element-wise or a list and a single value."""
        if isinstance(list1, list) and isinstance(list2, list):
            return [a - b for a, b in zip(list1, list2)]
        elif isinstance(list1, list):
            return [a - list2 for a in list1]
        elif isinstance(list2, list):
            return [list1 - b for b in list2]
        else:
            return list1 - list2

    # Apply the subtraction to each row in the DataFrame
    for measure in measures:
        if f'{measure}_post' in merged_df.columns and f'{measure}_pre' in merged_df.columns:
            merged_df[f'{measure}_training_diff'] = merged_df.apply(
                lambda row: subtract_lists(row[f'{measure}_post'], row[f'{measure}_pre']), axis=1
            )

    # Drop unnecessary columns (e.g., individual pre and post columns)
    columns_to_drop = [f'{measure}_pre' for measure in measures] + [f'{measure}_post' for measure in measures]
    final_training_diff_df = merged_df.drop(columns=columns_to_drop)

    return final_training_diff_df


################################################################################


def pre_post_plot(df, 
                  x_col='posttraining', 
                  y_col='pretraining', 
                  hue_col='ID', 
                  facet_col=None,  # Default to None
                  height=5, 
                  aspect=1, 
                  xlim=(1, 30), 
                  ylim=(1, 30), 
                  title="Frequency threshold: Pre-training vs Post-training", 
                  line_coords=([-1, 30], [-1, 30]),  # Default diagonal line coordinates
                  line_style='--', 
                  line_color='k', 
                  line_width=2,
                  x_label=None, 
                  y_label=None, 
                  save_path=None):
    """
    Creates a scatter plot comparing pre-training and post-training frequency thresholds.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data to plot.
        x_col (str): Column name for the x-axis (default: 'posttraining').
        y_col (str): Column name for the y-axis (default: 'pretraining').
        hue_col (str): Column name for coloring points (default: 'ID').
        facet_col (str or None): Column name for creating facets. If None, no facets will be created (default: None).
        height (int): Height of each facet (default: 5).
        aspect (float): Aspect ratio of each facet (default: 1).
        xlim (tuple): Limits for the x-axis (default: (1, 30)).
        ylim (tuple): Limits for the y-axis (default: (1, 30)).
        title (str): Title for the overall plot (default: "Frequency threshold: Pre-training vs Post-training").
        line_coords (tuple): Coordinates for the diagonal reference line (default: ([-1, 30], [-1, 30])).
        line_style (str): Line style for the diagonal reference line (default: '--').
        line_color (str): Color for the diagonal reference line (default: 'k').
        line_width (int): Width for the diagonal reference line (default: 2).
        x_label (str): Custom label for the x-axis (default: None).
        y_label (str): Custom label for the y-axis (default: None).

    Raises:
        ValueError: If required columns are missing from the DataFrame.
        TypeError: If the provided data is not a pandas DataFrame.
    """


    # Check for required columns based on whether facet_col is provided or not
    required_columns = [x_col, y_col, hue_col]
    if facet_col:
        required_columns.append(facet_col)
    check_dataframe(df, required_columns)

    # Create a FacetGrid if facet_col is provided, otherwise create a single plot
    if facet_col:
        g = sns.FacetGrid(df, col=facet_col, sharex=True, sharey=True, height=height, aspect=aspect)
        g.map_dataframe(sns.scatterplot, x=x_col, y=y_col, hue=hue_col, s=100)

        # Add the diagonal reference line to each plot
        for ax in g.axes.flatten():
            ax.plot(line_coords[0], line_coords[1], linestyle=line_style, color=line_color, lw=line_width)

        # Set the axis labels
        g.set_axis_labels(
            x_label if x_label else f"Post-training {x_col} (cpd)", 
            y_label if y_label else f"Pre-training {y_col} (cpd)"
        )
        g.set(xlim=xlim, ylim=ylim)

        # Add the overall title
        g.fig.suptitle(title, y=1.05)

        # Adjust the legend
        g.add_legend(title=hue_col)
    else:
        # Create a scatter plot without faceting
        plt.figure(figsize=(height * aspect, height))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, s=100)

        # Add the diagonal reference line
        plt.plot(line_coords[0], line_coords[1], linestyle=line_style, color=line_color, lw=line_width)

        # Set the axis labels
        plt.xlabel(x_label if x_label else f"Post-training {x_col} (cpd)")
        plt.ylabel(y_label if y_label else f"Pre-training {y_col} (cpd)")

        # Set the plot limits
        plt.xlim(xlim)
        plt.ylim(ylim)

        # Add the title
        plt.title(title)

        # Adjust the legend
        plt.legend(title=hue_col)

    # Save or show the plot based on save_path
    save_or_show(save_path)

def plot_cutoff_frequency(df, 
                          x_col='stage', 
                          y_col='Frequency_cpp', 
                          hue_col='amb', 
                          id_col='ID', 
                          figsize=(7, 7), 
                          col_wrap=2, 
                          ylim=(1, 30), 
                          title="Cut-Off Frequency", 
                          x_label=None, 
                          y_label=None, 
                          save_path=None):
    """
    Creates a scatter and line plot of the cut-off frequency over stages of the study.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data to plot.
        x_col (str): Column name for the x-axis (default: 'stage').
        y_col (str): Column name for the y-axis (default: 'Frequency_cpp').
        hue_col (str): Column name for coloring points (default: 'amb').
        id_col (str): Column name for creating facets (default: 'ID'). If None, no faceting will be applied.
        figsize (tuple): Size of the figure (default: (7, 7)).
        col_wrap (int): Number of columns to wrap the facets (default: 2).
        ylim (tuple): Limits for the y-axis (default: (1, 30)).
        title (str): Title for the overall plot (default: "Cut-Off Frequency").
        x_label (str): Custom label for the x-axis (default: None).
        y_label (str): Custom label for the y-axis (default: None).
        save_path (str): File path to save the figure. If None, the plot will be displayed (default: None).

    Raises:
        ValueError: If required columns are missing from the DataFrame.
        TypeError: If the provided data is not a pandas DataFrame.
    """


    # Check the DataFrame for required columns
    required_columns = [x_col, y_col, hue_col] + ([id_col] if id_col else [])
    check_dataframe(df, required_columns)

    # Scale the frequency column
    df['Frequency_cpp_scaled'] = df[y_col] * 72

    if id_col:
        # Initialize the FacetGrid with faceting by 'id_col'
        g = sns.FacetGrid(df, col=id_col, col_wrap=col_wrap, sharex=False, sharey=False, height=figsize[1], hue=hue_col)

        # Map the scatterplot and lineplot to each facet
        g.map_dataframe(sns.scatterplot, x=x_col, y='Frequency_cpp_scaled', s=100)
        g.map_dataframe(sns.lineplot, x=x_col, y='Frequency_cpp_scaled', linewidth=2)

        # Set axis labels with customizable options
        g.set_axis_labels(
            x_label if x_label else "Stage of the Study", 
            y_label if y_label else "Frequency threshold (cycles per dva)"
        )
        g.set(ylim=ylim)

        # Add the overall title
        g.fig.suptitle(title, y=1.02)

        # Adjust the legend
        g.add_legend(title=hue_col)

    else:
        # Plot without faceting
        plt.figure(figsize=figsize)
        sns.scatterplot(data=df, x=x_col, y='Frequency_cpp_scaled', hue=hue_col, s=100)
        sns.lineplot(data=df, x=x_col, y='Frequency_cpp_scaled', hue=hue_col, linewidth=2)
        plt.ylim(ylim)

        # Set axis labels and title
        plt.xlabel(x_label if x_label else "Stage of the Study")
        plt.ylabel(y_label if y_label else "Frequency threshold (cycles per dva)")
        plt.title(title)

        # Show the legend
        plt.legend(title=hue_col)

    # Save or show the plot based on save_path
    save_or_show(save_path)


def plot_contrast_sensitivity(df, 
                               x_col='Frequency_dva', 
                               y_col='Sensitivity', 
                               row_col=None, 
                               col_col='amb', 
                               hue_col='stage', 
                               figsize=(7, 7), 
                               title="Contrast Sensitivity", 
                               x_label=None, 
                               y_label=None, 
                               save_path=None):
    """
    Creates a linear model plot of contrast sensitivity thresholds against frequency.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data to plot.
        x_col (str): Column name for the x-axis (default: 'Frequency_dva').
        y_col (str): Column name for the y-axis (default: 'Sensitivity').
        row_col (str): Column name for the rows in the facet grid (default: None).
        col_col (str): Column name for the columns in the facet grid (default: 'amb').
        hue_col (str): Column name for coloring points (default: 'stage').
        figsize (tuple): Size of the figure (default: (7, 7)).
        title (str): Title for the overall plot (default: "Contrast Sensitivity").
        x_label (str): Custom label for the x-axis (default: None).
        y_label (str): Custom label for the y-axis (default: None).

    Raises:
        ValueError: If required columns are missing from the DataFrame.
        TypeError: If the provided data is not a pandas DataFrame.
    """


    # Check the DataFrame for required columns
    check_dataframe(df, [x_col, y_col, col_col, hue_col] + ([row_col] if row_col else []))

    # Initialize the lmplot with row_col if specified
    g = sns.lmplot(x=x_col, y=y_col, row=row_col, col=col_col, 
                   hue=hue_col, data=df, ci=None, robust=True, 
                   height=figsize[1]/len(df[row_col].unique()) if row_col else figsize[1])

    # Set axis labels with customizable options
    g.set_axis_labels(
        x_label if x_label else "Frequency", 
        y_label if y_label else "Sensitivity Thresholds"
    )

    # Add legend and titles
    g.add_legend(title="Stage of testing")
    g.set_titles(col_template="{col_name}", row_template="{row_name}" if row_col else "")

    # Set y-axis to logarithmic scale
    g.set(yscale='log')

    # Set y-axis bounds
    for ax in g.axes.flat:  # Iterate over all axes in the grid
        ax.set_ylim(1, 100)  # Adjust the bounds to your desired range


    # Add the overall title
    g.fig.suptitle(title, y=1.02)

   # Save or show the plot based on save_path
    save_or_show(save_path)

################################################################################

def calculate_auc_difference(df, 
                             x_col, 
                             y_col, 
                             id_col='ID', 
                             amb_col='amb', 
                             stage_col='stage', 
                             stages_to_compare=('pretraining', 'posttraining')):
    """
    Calculates the area under the curve (AUC) difference between two stages for each combination of ID and amb.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data to analyze.
        x_col (str): Column name for the x values (e.g., 'Staircase').
        y_col (str): Column name for the y values to integrate (e.g., 'logBP').
        id_col (str): Column name for participant ID (default: 'ID').
        amb_col (str): Column name for amblyopia condition (default: 'amb').
        stage_col (str): Column name for the stage (default: 'stage').
        stages_to_compare (tuple): Tuple of two stages to compare (default: ('pretraining', 'posttraining')).

    Returns:
        pd.DataFrame: DataFrame containing the AUC differences with columns for ID, amb, and ABC.

    Raises:
        ValueError: If required columns are missing from the DataFrame.
        TypeError: If the provided data is not a pandas DataFrame.
        ValueError: If the specified stages are not found in the data.
    """


    # Check the DataFrame for required columns
    check_dataframe(df, [x_col, y_col, id_col, amb_col, stage_col])

    # Unpack stages to compare
    stage1, stage2 = stages_to_compare

    auc_data = {}

    # Group the data by ID and amb condition
    for (id_val, amb_val), group in df.groupby([id_col, amb_col]):
        stages = group[stage_col].unique()

        # Ensure the desired stages are present
        if stage1 not in stages or stage2 not in stages:
            continue

        # Filter data for the two selected stages
        stage1_data = group[group[stage_col] == stage1]
        stage2_data = group[group[stage_col] == stage2]

        # Sort data based on the x column for consistent integration
        stage1_data = stage1_data.sort_values(by=x_col)
        stage2_data = stage2_data.sort_values(by=x_col)

        # Perform numerical integration for each curve
        slope1, intercept1, _, _, _ = linregress(stage1_data[x_col], stage1_data[y_col])
        slope2, intercept2, _, _, _ = linregress(stage2_data[x_col], stage2_data[y_col])

        # Define linear functions
        f_stage1 = lambda x: slope1 * x + intercept1
        f_stage2 = lambda x: slope2 * x + intercept2

        # Integrate the linear functions
        # Define the range of x for integration
        x_min1, x_max1 = 0.001, 50
        x_min2, x_max2 = 0.001, 50

        auc_stage1 = quad(f_stage1, x_min1, x_max1)[0]
        auc_stage2 = quad(f_stage2, x_min2, x_max2)[0]

        # Store the difference in AUC (ABC) for each comparison
        auc_data[(id_val, amb_val)] = auc_stage2 - auc_stage1

    # Convert the ABC data into a DataFrame for plotting
    abc_df = pd.DataFrame(list(auc_data.items()), columns=['Condition', 'ABC'])
    abc_df[[id_col, amb_col]] = pd.DataFrame(abc_df['Condition'].tolist(), index=abc_df.index)
    abc_df = abc_df.drop(columns='Condition')

    return abc_df



def plot_abc_values(auc_data, 
                    x_col='ID', 
                    y_col='ABC', 
                    title='ABC Values for Contrast Sensitivity', 
                    figsize=(7, 7), 
                    save_path=None):
    """
    Converts AUC data into a DataFrame and plots the Area Between Curves (ABC) values.

    Parameters:
        auc_data (dict): Dictionary with AUC data where keys are IDs and values are ABC values.
        x_col (str): Column name for the x-axis (default: 'ID').
        y_col (str): Column name for the y-axis (default: 'ABC').
        title (str): Title of the plot (default: 'ABC Values for Contrast Sensitivity').
        figsize (tuple): Size of the figure (default: (7, 7)).

    Raises:
        ValueError: If the input data is empty or not a dictionary.
        TypeError: If required columns are missing after processing.
    """

    # Check input data type
    if not isinstance(auc_data, dict):
        raise TypeError("Input must be a dictionary.")

    if not auc_data:
        raise ValueError("Input data cannot be empty.")

    # Convert the ABC data into a DataFrame
    abc_df = pd.DataFrame(list(auc_data.items()), columns=[x_col, y_col])

    # Check for required columns
    if x_col not in abc_df.columns or y_col not in abc_df.columns:
        missing_columns = [col for col in [x_col, y_col] if col not in abc_df.columns]
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

    # Plot the ABC values
    plt.figure(figsize=figsize)
    sns.barplot(x=x_col, y=y_col, data=abc_df)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)

    # Save or show the plot based on save_path
    save_or_show(save_path)



def plot_multiple_difference(df, 
                             x_col='Staircase', 
                             y_col='Difference', 
                             col_col='amb', 
                             hue_col=None, 
                             figsize=(7, 7), 
                             title="Contrast Sensitivity Differences", 
                             save_path=None):
    """
    Creates a linear model plot of differences against staircase values.

    Parameters:
        df (DataFrame): DataFrame containing the data to plot.
        x_col (str): Column name for the x-axis (default: 'Staircase').
        y_col (str): Column name for the y-axis (default: 'Difference').
        col_col (str): Column name for creating facets (default: 'amb').
        hue_col (str, optional): Optional column name for coloring points (default: None).
        figsize (tuple): Size of the figure (default: (7, 7)).
        title (str): Title for the overall plot (default: "Contrast Sensitivity Differences").

    Raises:
        ValueError: If the input DataFrame is empty or missing required columns.
        TypeError: If the input df is not a DataFrame.
    """

    check_dataframe(df, [x_col, y_col, col_col], allow_empty=False)

    # Initialize the lmplot
    g = sns.lmplot(
        x=x_col, 
        y=y_col, 
        hue=hue_col, 
        col=col_col, 
        data=df, 
        ci=None, 
        facet_kws={'sharey': False, 'sharex': True}
    )

    # Set axis labels
    g.set_axis_labels(x_col, y_col)

    # Add overall title
    g.fig.suptitle(title, y=1.02)

    # Save or show the plot based on save_path
    save_or_show(save_path)


def plot_interocular_imbalance(df, 
                               x_col='Staircase', 
                               y_col='logBP', 
                               hue_col='stage', 
                               col_col='ID', 
                               figsize=(15, 10), 
                               title="Interocular Imbalance",
                               unique_hues=['familiarisation', 'pretraining', 'posttraining'],
                               save_path=None):
    """
    Creates a linear model plot for interocular imbalance, arranged in a grid with up to 4 columns per row.
    """

    check_dataframe(df, [x_col, y_col, col_col, hue_col], allow_empty=False)

    # Define a consistent palette for the 'stage' column
    palette = sns.color_palette("husl", len(unique_hues))
    stage_palette = dict(zip(unique_hues, palette))

    # Set up the FacetGrid
    g = sns.FacetGrid(
        df,
        col=col_col,
        col_wrap=4,
        height=figsize[1] / 4,
        aspect=figsize[0] / figsize[1]
    )

    # Add regression lines
    def plot_regression(data, **kwargs):
        kwargs.pop('color', None)  # Remove 'color' from kwargs to avoid conflicts
        for stage in unique_hues:
            stage_data = data[data[hue_col] == stage]
            if not stage_data.empty:
                sns.regplot(
                    x=x_col,
                    y=y_col,
                    data=stage_data,
                    ci=None,
                    scatter=False,
                    color=stage_palette[stage],
                    **kwargs
                )

    g.map_dataframe(plot_regression)

    # Add scatter points
    g.map_dataframe(
        sns.scatterplot,
        x=x_col,
        y=y_col,
        hue=hue_col,
        style=hue_col,
        palette=stage_palette,
        legend=False
    )

    # Add a legend
    handles = [plt.Line2D([], [], color=stage_palette[stage], marker='o', linestyle='', label=stage) for stage in unique_hues]
    g.fig.legend(
        handles=handles,
        title="Stage of Testing",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(unique_hues)
    )

    # Set axis properties
    for ax in g.axes.flat:
        ax.set_xticks([1, 2, 3, 4])
    g.set_axis_labels("Frequency", "log(BP) - Balance Point")
    g.set_titles("{col_name}")
    g.fig.suptitle(title, y=1.05)

    # Adjust layout and save/show plot
    g.tight_layout(w_pad=1)
    save_or_show(save_path)

def calculate_auc_difference_updated(
    df,
    x_col,
    y_col,
    id_col='ID',
    amb_col='amb',
    stage_col='stage',
    stages_to_compare=('pretraining', 'posttraining'),
    fit_type='weibull',         # 'weibull' | 'linear' | 'logparabola' | 'barten' | 'daly'
    rescale=True,
    range_mode='union',        # 'union' | 'each' | 'fixed'
    fixed_range=None,          # tuple (xmin, xmax) when range_mode == 'fixed'
    integrate_mode='union',    # 'union' | 'intersection' | 'each'
    weibull_maxfev=10000,
    weibull_retries=3, 
    plot=None
):
    """
    Calculates AUC difference between two stages per ID/amb, with multiple curve-fitting options.
    New fit_type options: 'logparabola', 'barten', 'daly' (plus the original 'weibull' and 'linear').

    For each participant/amb group, fits one curve to the pre-stage points and one to the
    post-stage points (using `fit_type`), integrates each curve numerically, and takes the
    difference in area (ABC = "area between curves") as a plasticity/training-effect measure.

    `range_mode` controls what x/y range the two curves are normalized against before fitting
    (only used when `rescale=True`); `integrate_mode` separately controls what x range the AUCs
    are actually integrated over. They can differ, e.g. fit on the union range but integrate
    each curve only over its own data range with `integrate_mode='each'`.

    Input data expectation:
      - x_col: spatial frequency in dva (must be > 0)
      - y_col: sensitivity values (linear, not log). If you have thresholds, convert outside.

    Returns:
      pd.DataFrame with columns: [id_col, amb_col, 'ABC' (auc2-auc1), 'pretraining', 'posttraining',
                                 'x_integration_min', 'x_integration_max']
    """
    # ********** model definitions **********
    # Four alternative Contrast Sensitivity Function (CSF) shapes to fit per curve;
    # `fit_type` picks which one `fit_curve` below uses.
    def weibull_4p(x, alpha, beta, gamma, lamb):
        x = np.array(x, dtype=float)
        with np.errstate(over='ignore', invalid='ignore'):
            return gamma + (1.0 - gamma - lamb) * (1.0 - np.exp(- (x / alpha) ** beta))

    def csf_log_parabola(f, a, b, c):
        # f: frequency in dva (>0). returns sensitivity (linear)
        f = np.array(f, dtype=float)
        # guard against non-positive f; for non-positive, return tiny values (not used typically)
        out = np.full_like(f, np.nan, dtype=float)
        pos = f > 0
        out[pos] = 10 ** (a - b * (np.log10(f[pos]) - c) ** 2)
        return out

    def csf_barten_like(f, k, alpha, f0, beta):
        # flexible Barten-like form: k * f^alpha * exp(-(f/f0)^beta)
        f = np.array(f, dtype=float)
        out = np.zeros_like(f, dtype=float)
        pos = f > 0
        out[pos] = k * (f[pos] ** alpha) * np.exp(- (f[pos] / f0) ** beta)
        return out

    def csf_daly_like(f, A, mu, sigma, C):
        # simple log-normal (gaussian on log10 axis) + baseline C
        f = np.array(f, dtype=float)
        out = np.full_like(f, np.nan, dtype=float)
        pos = f > 0
        # using log10 for stability (mu is log10 of peak freq)
        out[pos] = A * np.exp(- (np.log10(f[pos]) - mu) ** 2 / (2.0 * (sigma ** 2))) + C
        return out

    # ********** fitting helper **********
    def fit_curve(x, y, fit_type='weibull'):
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)

        # require at least 3 points for more flexible models; allow 2 for simple linear/weibull fallback
        if len(x_arr) < 2:
            return None

        # Weibull fits as before
        if fit_type == 'weibull':
            bounds = ([1e-6, 0.1, 0.0, 0.0], [np.inf, 10.0, 0.9, 0.9])
            p0_candidates = [
                [max(np.median(x_arr), 1e-3), 2.0, 0.01, 0.01],
                [max(np.mean(x_arr), 1e-3), 1.5, 0.0, 0.0],
                [max(np.max(x_arr)*0.5, 1e-3), 2.5, 0.01, 0.01]
            ]
            for p0 in p0_candidates[:weibull_retries]:
                try:
                    popt, _ = curve_fit(weibull_4p, x_arr, y_arr, p0=p0, bounds=bounds, maxfev=weibull_maxfev)
                    return lambda x_: weibull_4p(x_, *popt)
                except Exception:
                    continue
            return None

        elif fit_type == 'linear':
            if len(x_arr) < 2 or np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
                intercept = float(np.mean(y_arr))
                return lambda x_: np.full_like(np.array(x_), intercept, dtype=float) if np.ndim(x_) else intercept
            slope, intercept, *_ = linregress(x_arr, y_arr)
            return lambda x_: slope * np.array(x_) + intercept

        elif fit_type == 'logparabola':
            # fit csf_log_parabola to sensitivity y
            # only use positive frequencies
            mask = x_arr > 0
            x_pos = x_arr[mask]
            y_pos = y_arr[mask]
            if len(x_pos) < 3:
                return None
            # initial guesses
            a0 = np.log10(np.nanmax(y_pos))
            idx_max = int(np.nanargmax(y_pos))
            c0 = np.log10(x_pos[idx_max])
            b0 = 0.5
            p0 = [a0, b0, c0]
            lower = [-6.0, 0.01, np.log10(np.min(x_pos)) - 1.0]
            upper = [6.0, 10.0, np.log10(np.max(x_pos)) + 1.0]
            try:
                popt, _ = curve_fit(csf_log_parabola, x_pos, y_pos, p0=p0, bounds=(lower, upper), maxfev=5000)
                return lambda x_: csf_log_parabola(x_, *popt)
            except Exception:
                return None

        elif fit_type == 'barten':
            # flexible parametric model: k * f^alpha * exp(-(f/f0)^beta)
            mask = x_arr > 0
            x_pos = x_arr[mask]
            y_pos = y_arr[mask]
            if len(x_pos) < 3:
                return None
            # initial guesses
            k0 = max(np.nanmax(y_pos), 1e-3)
            alpha0 = 1.0
            f0_0 = np.median(x_pos)
            beta0 = 1.5
            p0_candidates = [
                [k0, alpha0, f0_0, beta0],
                [k0, 0.5, f0_0/2.0 if f0_0>0 else 0.1, 1.0],
                [k0*0.5, 1.5, f0_0*1.5, 2.0]
            ]
            lower = [1e-8, -5.0, 1e-6, 0.1]
            upper = [np.inf, 5.0, np.max(x_pos)*10.0 + 1e-6, 10.0]
            for p0 in p0_candidates:
                try:
                    popt, _ = curve_fit(csf_barten_like, x_pos, y_pos, p0=p0, bounds=(lower, upper), maxfev=8000)
                    return lambda x_: csf_barten_like(x_, *popt)
                except Exception:
                    continue
            return None

        elif fit_type == 'daly':
            # log-normal (on log10 axis) + baseline: A * exp(- (log10 f - mu)^2 / (2 sigma^2)) + C
            mask = x_arr > 0
            x_pos = x_arr[mask]
            y_pos = y_arr[mask]
            if len(x_pos) < 3:
                return None
            A0 = max(np.nanmax(y_pos) - np.nanmin(y_pos), 1e-3)
            mu0 = np.log10(x_pos[int(np.nanargmax(y_pos))])
            sigma0 = 0.4
            C0 = np.nanmin(y_pos) * 0.1
            p0_candidates = [
                [A0, mu0, sigma0, C0],
                [max(1e-3, A0*0.8), mu0, 0.6, 0.0],
                [A0*1.2, mu0, 0.3, max(0.0, np.nanmin(y_pos)*0.5)]
            ]
            lower = [1e-8, np.log10(np.min(x_pos)) - 2.0, 1e-3, 0.0]
            upper = [np.inf, np.log10(np.max(x_pos)) + 2.0, 5.0, np.max(y_pos)]
            for p0 in p0_candidates:
                try:
                    popt, _ = curve_fit(csf_daly_like, x_pos, y_pos, p0=p0, bounds=(lower, upper), maxfev=8000)
                    return lambda x_: csf_daly_like(x_, *popt)
                except Exception:
                    continue
            return None

        else:
            raise ValueError("Unsupported fit_type. Choose 'weibull','linear','logparabola','barten','daly'.")

    # ********** basic checks & setup **********
    check_dataframe(df, [x_col, y_col, id_col, amb_col, stage_col])
    stage1, stage2 = stages_to_compare

    # prepare plotting directory if requested
    make_plots = False
    plot_dir = None
    if isinstance(plot, str) and plot.strip() != "":
        plot_dir = os.path.abspath(plot)
        try:
            os.makedirs(plot_dir, exist_ok=True)
            make_plots = True
        except Exception:
            make_plots = False
            plot_dir = None

    def _safe_name(s):
        s = str(s)
        s = re.sub(r'\s+', '_', s)
        s = re.sub(r'[^\w\-_\.]', '', s)
        return s

    results = []
    for (id_val, amb_val), group in df.groupby([id_col, amb_col]):
        # must have both stages present
        if not all(stage in group[stage_col].values for stage in [stage1, stage2]):
            continue

        s1 = group[group[stage_col] == stage1].sort_values(by=x_col)
        s2 = group[group[stage_col] == stage2].sort_values(by=x_col)

        x1 = s1[x_col].astype(float).values
        y1 = s1[y_col].astype(float).values
        x2 = s2[x_col].astype(float).values
        y2 = s2[y_col].astype(float).values

        # drop any NaNs
        mask1 = ~np.isnan(x1) & ~np.isnan(y1)
        mask2 = ~np.isnan(x2) & ~np.isnan(y2)
        x1, y1 = x1[mask1], y1[mask1]
        x2, y2 = x2[mask2], y2[mask2]

        if len(x1) < 2 or len(x2) < 2:
            # too few points to fit reliably
            continue

        # integration ranges based on modes
        x1_min, x1_max = float(np.min(x1)), float(np.max(x1))
        x2_min, x2_max = float(np.min(x2)), float(np.max(x2))

        if range_mode == 'fixed':
            if fixed_range is None or not (isinstance(fixed_range, (list, tuple)) and len(fixed_range) == 2):
                raise ValueError("fixed_range must be provided as (xmin, xmax) when range_mode=='fixed'")
            global_xmin, global_xmax = float(fixed_range[0]), float(fixed_range[1])
            if global_xmax <= global_xmin:
                raise ValueError("fixed_range must be (xmin, xmax) with xmax > xmin")
        else:
            global_xmin = min(x1_min, x2_min)
            global_xmax = max(x1_max, x2_max)

        if integrate_mode == 'union':
            int_min = global_xmin
            int_max = global_xmax
        elif integrate_mode == 'intersection':
            int_min = max(x1_min, x2_min)
            int_max = min(x1_max, x2_max)
            if int_max <= int_min:
                continue
        elif integrate_mode == 'each':
            int_min = None
            int_max = None
        else:
            raise ValueError("integrate_mode must be one of 'union','intersection','each'")

        # scaling helper
        def safe_scale(x_arr, src_min, src_max):
            denom = (src_max - src_min)
            if denom == 0.0:
                return np.zeros_like(x_arr, dtype=float)
            return (x_arr - src_min) / denom

        # prepare fit functions
        # `rescale` fits each curve in normalized [0, 1] x/y space rather than raw units:
        # curve_fit's initial guesses and bounds above are tuned for that range, so fitting
        # on raw data (which varies a lot in scale between participants/measures) would make
        # the optimizer far less reliable. f1/f2 un-normalize the fitted curve back to the
        # original x/y units so callers can query them at any real x.
        f1 = None
        f2 = None
        if rescale:
            if range_mode in ('union', 'fixed'):
                y_min_global = min(np.min(y1), np.min(y2))
                y_max_global = max(np.max(y1), np.max(y2))
                y_span = y_max_global - y_min_global
                if y_span == 0:
                    f1 = lambda x_: np.full_like(np.array(x_), float(y_min_global), dtype=float) if np.ndim(x_) else float(y_min_global)
                    f2 = lambda x_: np.full_like(np.array(x_), float(y_min_global), dtype=float) if np.ndim(x_) else float(y_min_global)
                else:
                    x_min_global = global_xmin
                    x_max_global = global_xmax
                    x1s = safe_scale(x1, x_min_global, x_max_global)
                    x2s = safe_scale(x2, x_min_global, x_max_global)
                    y1s = (y1 - y_min_global) / y_span
                    y2s = (y2 - y_min_global) / y_span
                    f1_scaled = fit_curve(x1s, y1s, fit_type)
                    f2_scaled = fit_curve(x2s, y2s, fit_type)
                    if f1_scaled is None or f2_scaled is None:
                        continue

                    def f1(xq):
                        xq = np.array(xq, dtype=float)
                        xq_s = safe_scale(xq, x_min_global, x_max_global)
                        return f1_scaled(xq_s) * y_span + y_min_global

                    def f2(xq):
                        xq = np.array(xq, dtype=float)
                        xq_s = safe_scale(xq, x_min_global, x_max_global)
                        return f2_scaled(xq_s) * y_span + y_min_global

            elif range_mode == 'each':
                y1_min, y1_max = float(np.min(y1)), float(np.max(y1))
                y2_min, y2_max = float(np.min(y2)), float(np.max(y2))
                y1_span = y1_max - y1_min
                y2_span = y2_max - y2_min

                x1s = safe_scale(x1, x1_min, x1_max)
                x2s = safe_scale(x2, x2_min, x2_max)

                y1s = (y1 - y1_min) / (y1_span if y1_span != 0 else 1.0)
                y2s = (y2 - y2_min) / (y2_span if y2_span != 0 else 1.0)

                f1_scaled = fit_curve(x1s, y1s, fit_type)
                f2_scaled = fit_curve(x2s, y2s, fit_type)
                if f1_scaled is None or f2_scaled is None:
                    continue

                def f1(xq):
                    xq = np.array(xq, dtype=float)
                    xq_s = safe_scale(xq, x1_min, x1_max)
                    if y1_span == 0:
                        return np.full_like(xq, y1_min, dtype=float)
                    return f1_scaled(xq_s) * y1_span + y1_min

                def f2(xq):
                    xq = np.array(xq, dtype=float)
                    xq_s = safe_scale(xq, x2_min, x2_max)
                    if y2_span == 0:
                        return np.full_like(xq, y2_min, dtype=float)
                    return f2_scaled(xq_s) * y2_span + y2_min

        else:
            f1 = fit_curve(x1, y1, fit_type)
            f2 = fit_curve(x2, y2, fit_type)
            if f1 is None or f2 is None:
                continue

        # compute AUCs
        try:
            if integrate_mode == 'each':
                auc1 = float(quad(lambda v: np.atleast_1d(f1(v)).item(), x1_min, x1_max, limit=200)[0])
                auc2 = float(quad(lambda v: np.atleast_1d(f2(v)).item(), x2_min, x2_max, limit=200)[0])
                integration_min, integration_max = (x1_min, x1_max), (x2_min, x2_max)
            else:
                auc1 = float(quad(lambda v: np.atleast_1d(f1(v)).item(), int_min, int_max, limit=200)[0])
                auc2 = float(quad(lambda v: np.atleast_1d(f2(v)).item(), int_min, int_max, limit=200)[0])
                integration_min, integration_max = int_min, int_max
        except Exception:
            continue

        auc_diff = auc2 - auc1

        # plotting & peak annotation
        if make_plots and plot_dir is not None:
            try:
                if integrate_mode == 'each':
                    plot_xmin = min(x1_min, x2_min)
                    plot_xmax = max(x1_max, x2_max)
                else:
                    plot_xmin = int_min
                    plot_xmax = int_max

                if plot_xmax <= plot_xmin:
                    plot_xmin = min(x1_min, x2_min)
                    plot_xmax = max(x1_max, x2_max)

                xs = np.linspace(plot_xmin, plot_xmax, 500)

                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                ax.scatter(x1, y1, label=f'{stage1} (data)', marker='o', alpha=0.85, s=30)
                ax.scatter(x2, y2, label=f'{stage2} (data)', marker='s', alpha=0.85, s=30)

                # compute fitted ys and peaks
                peak_x1 = peak_y1 = None
                peak_x2 = peak_y2 = None
                ys1 = None
                ys2 = None

                try:
                    ys1 = np.atleast_1d(f1(xs))
                    ax.plot(xs, ys1, label=f'{stage1} (fit)', linewidth=2)
                    if np.all(np.isfinite(ys1)):
                        idx1 = int(np.nanargmax(ys1))
                        peak_x1 = xs[idx1]
                        peak_y1 = ys1[idx1]
                        # mark peak
                        ax.plot(peak_x1, peak_y1, marker='D', markersize=6)
                        ax.axvline(peak_x1, linestyle=':', linewidth=1, alpha=0.5)
                except Exception:
                    pass

                try:
                    ys2 = np.atleast_1d(f2(xs))
                    ax.plot(xs, ys2, label=f'{stage2} (fit)', linewidth=2, linestyle='--')
                    if np.all(np.isfinite(ys2)):
                        idx2 = int(np.nanargmax(ys2))
                        peak_x2 = xs[idx2]
                        peak_y2 = ys2[idx2]
                        ax.plot(peak_x2, peak_y2, marker='X', markersize=6)
                        ax.axvline(peak_x2, linestyle='--', linewidth=1, alpha=0.5)
                except Exception:
                    pass

                # annotation text
                if integrate_mode == 'each':
                    base_txt = (f"AUC {stage1}: {auc1:.4g}\n"
                                f"AUC {stage2}: {auc2:.4g}\n"
                                f"Δ (ABC): {auc_diff:.4g}\n"
                                f"{stage1} range: [{x1_min:.3g}, {x1_max:.3g}]\n"
                                f"{stage2} range: [{x2_min:.3g}, {x2_max:.3g}]\n")
                else:
                    base_txt = (f"AUC {stage1}: {auc1:.4g}\n"
                                f"AUC {stage2}: {auc2:.4g}\n"
                                f"Δ (ABC): {auc_diff:.4g}\n"
                                f"Integration: [{integration_min:.3g}, {integration_max:.3g}]\n")

                peak_txt = ""
                if peak_x1 is not None:
                    peak_txt += f"peak x ({stage1}): {peak_x1:.4g}\n"
                if peak_x2 is not None:
                    peak_txt += f"peak x ({stage2}): {peak_x2:.4g}\n"

                txt = base_txt + peak_txt

                ax.set_title(f"{id_col}={id_val}  {amb_col}={amb_val}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.legend(loc='best', fontsize='small', framealpha=0.9)
                ax.grid(True, alpha=0.3)
                ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", alpha=0.14))

                fname = f"{_safe_name(id_val)}_{_safe_name(amb_val)}.png"
                fullpath = os.path.join(plot_dir, fname)
                fig.tight_layout()
                fig.savefig(fullpath, dpi=150)
                plt.close(fig)
            except Exception:
                try:
                    plt.close('all')
                except Exception:
                    pass

        results.append({
            id_col: id_val,
            amb_col: amb_val,
            'ABC': auc_diff,
            'pretraining': auc1,
            'posttraining': auc2,
            'x_integration_min': integration_min,
            'x_integration_max': integration_max
        })

    if not results:
        return pd.DataFrame(columns=[id_col, amb_col, 'ABC', 'pretraining', 'posttraining', 'x_integration_min', 'x_integration_max'])

    return pd.DataFrame(results)

