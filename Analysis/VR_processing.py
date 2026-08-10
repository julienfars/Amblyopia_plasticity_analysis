# VR_processing.py
"""VR training session data processing and plotting."""

import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import save_or_show

# VR functions
def process_all_data(base_dir, pattern='_training_regime.csv', folder='3', start='sub_'):
    processed_data_dict = {}  # Dictionary to store processed DataFrames by subject ID

    # Iterate over all subdirectories in the base directory
    for sub_dir in os.listdir(base_dir):
        if sub_dir.startswith(start):
            # Construct the file search pattern with wildcard
            file_pattern = os.path.join(base_dir, sub_dir, folder, f"*{pattern}")
            # Find all matching files
            matching_files = glob(file_pattern)
            for file_path in matching_files:
                if os.path.exists(file_path):
                    try:
                        # Process the data file
                        df = process_data(file_path)  # Call to process_data function
                        cleaned_data = clean_data(df['result_table'])  # Clean the result_table
                        
                        # Store cleaned DataFrame in the dictionary
                        processed_data_dict[sub_dir] = cleaned_data
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                else:
                    print(f"File not found: {file_path}")

    return processed_data_dict

def process_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    # Read CSV file
    df = pd.read_csv(file_path, sep=";")
    
    # Data processing
    df['gabor_orientation'] = df['gabor_orientation'].replace("'-45", "-45")
    
    # Process dates: some rows have a time-of-day suffix appended to the date
    # (making the string longer than 13 chars); truncate those back down to
    # just the first 10 characters (the MM/DD/YYYY portion) so all rows are comparable.
    processed_dates = []
    for date_str in df['date_inserted']:
        if len(date_str) > 13:
            processed_date = date_str[:-(len(date_str) - 10)]
            processed_dates.append(processed_date)
        else:
            processed_dates.append(date_str)
    df['date_inserted'] = pd.Categorical(processed_dates)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Staircase threshold estimate: average the noise_contrast at each "reversal"
    # (a point where the staircase direction flips), following the standard
    # adaptive-staircase convention of estimating threshold from the reversal points.
    def calculate_last_6_mean(group):
        group = group.copy()
        group['last_6_mean'] = np.nan

        # A reversal point is a row where the running reversal count just increased by 1
        condition = group['reversal'] - group['reversal'].shift(1, fill_value=group['reversal'].iloc[0]) == 1

        # Check if the reversal values are not all zeros
        has_valid_reversal = not (group['reversal'] == 0).all()

        if has_valid_reversal and condition.any():
            group.loc[condition, 'last_6_mean'] = group.loc[condition, 'noise_contrast']

        # Carry each reversal's value forward until the next reversal, so every
        # row in between has a running mean available
        group['last_6_mean'] = group['last_6_mean'].ffill()

        return group

    # Group and apply the function
    df = df.groupby(['date_inserted', 'run_number'], group_keys=False).apply(calculate_last_6_mean)

    # Keep only reversal rows (plus each group's first row, where the diff is NaN) -
    # i.e. drop the non-reversal trials now that their contribution is already folded
    # into the forward-filled 'last_6_mean' above.
    df = df[(df['reversal'] - df['reversal'].shift(1, fill_value=df['reversal'].iloc[0]) == 1) |
            (df['reversal'] - df['reversal'].shift(1, fill_value=df['reversal'].iloc[0]).isna())]

    # Only the first 6 reversals of each run should count towards the threshold mean;
    # freeze 'last_6_mean' at its 6th-reversal value for any reversals after that
    # (per the "average of the last 6 reversals" staircase convention).
    df['row_number'] = df.groupby(['date_inserted', 'run_number']).cumcount()
    df['last_6_mean'] = np.where(df['row_number'] > 6, df['last_6_mean'], np.nan)
    df['last_6_mean'] = df.groupby(['date_inserted', 'run_number'])['last_6_mean'].ffill()

    # Drop temporary columns
    df = df.drop(columns=['row_number'])
    
    # Sort DataFrame
    df = df.sort_values(by=['date_inserted', 'run_number'])
    
    # Compute means
    table_means = {}
    for date_value in df['date_inserted'].unique():
        for run_value in df[df['date_inserted'] == date_value]['run_number'].unique():
            subset = df[(df['date_inserted'] == date_value) & (df['run_number'] == run_value)]
            mean_value = subset['last_6_mean'].mean()
            table_means[f"{date_value}_run_{run_value}"] = mean_value
    
    # Prepare the last rows table: one row per training day (its final reversal),
    # used by plot_last_rows to show the end-of-day threshold.
    last_rows_table = df.dropna(subset=['last_6_mean']).groupby('date_inserted').tail(1)
    # Map each calendar date to a sequential day-of-training index (0, 1, 2, ...)
    # so training days plot evenly spaced on the x-axis, regardless of gaps between sessions.
    last_rows_table['dates'] = pd.to_datetime(last_rows_table['date_inserted'], format="%m/%d/%Y")
    unique_dates = last_rows_table['dates'].unique()
    last_rows_table['date_to_number'] = last_rows_table['dates'].apply(lambda x: list(unique_dates).index(x))

    df['dates'] = pd.to_datetime(df['date_inserted'], format="%m/%d/%Y")
    unique_dates = df['dates'].unique()
    df['date_to_number'] = df['dates'].apply(lambda x: list(unique_dates).index(x))

    # Absolute contrast difference between the two Gabor stimuli for this trial;
    # plot_last_rows annotates this value at the first/last trial of each day.
    df['delta'] = (df['gabor_contrast1'] - df['gabor_contrast2']).abs()
    df['delta_change'] = df['delta'].diff().fillna(0).ne(0).astype(int)
    
    return {
        'result_table': df,
        'table_means': table_means,
        'last_rows_table': last_rows_table
    }

def clean_data(df):
    """Replace inf values with NaN."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def plot_last_rows(sub_id, processed_data, output_file=None):
    """
    Plot the last rows of data for a specific subject.

    Parameters:
    - sub_id: Subject ID string (e.g., 'sub_005').
    - processed_data: Dictionary containing processed data for each subject.
    - output_file: Optional path to save the plot.
    """
    # Access the processed DataFrame for the given subject
    df_sub = processed_data.get(sub_id, None)  # Get the DataFrame dynamically
    if df_sub is None:
        print(f"No data found for {sub_id}.")
        return

    # Check if the necessary columns exist in the DataFrame
    required_columns = ['last_6_mean', 'date_to_number', 'id_participant', 'run_number', 'delta']
    if not all(col in df_sub.columns for col in required_columns):
        print(f"Data structure for {sub_id} is incorrect or missing tables.")
        print(f"Current structure: {df_sub.columns}")
        return

    # Set the style of seaborn
    sns.set(style="whitegrid")
    
    # Convert date_to_number to numeric if it is categorical
    df_sub['date_to_number'] = pd.to_numeric(df_sub['date_to_number'], errors='coerce')
    
    # Create the plot
    plt.figure(figsize=(10, 6))  # Adjust size as needed
    
    # Plot lines connecting points for id_participant
    if 'id_participant' in df_sub.columns:
        sns.lineplot(data=df_sub, x='date_to_number', y='last_6_mean', hue='id_participant', 
                     palette='Reds', linewidth=3, legend=False)
    
    # Plot points for result table with dodge
    sns.scatterplot(data=df_sub, x='date_to_number', y='last_6_mean', hue='run_number', 
                    palette='husl', size='run_number', legend='full', style='run_number')
    
    # Highlight the first and last delta values on each day of training:
    # green label just above the mid-line (y=55) for the day's first trial,
    # red label just below it (y=50) for the day's last trial. The fixed y
    # positions are chart-specific, chosen to sit near the middle of the
    # noise-contrast y-axis (0-100) without overlapping the plotted points.
    grouped_df = df_sub.groupby('date_to_number')
    first_delta_each_day = grouped_df.first().reset_index()
    last_delta_each_day = grouped_df.last().reset_index()

    for _, row in first_delta_each_day.iterrows():
        delta_value = round(row['delta']) if not float(row['delta']).is_integer() else int(row['delta'])
        plt.text(row['date_to_number'], 55, str(delta_value),
                 color='green', fontsize=12, weight='bold', ha='center', va='bottom')

    for _, row in last_delta_each_day.iterrows():
        delta_value = round(row['delta']) if not float(row['delta']).is_integer() else int(row['delta'])
        plt.text(row['date_to_number'], 50, str(delta_value),
                 color='red', fontsize=12, weight='bold', ha='center', va='bottom')
    
    # Set labels and title
    plt.xlabel("Day of training")
    plt.ylabel("Noise contrast (% contrast)")
    plt.title(f"VR training - {df_sub['id_participant'].iloc[0]}")
    
    # Adjust x and y axis scales and ticks
    plt.xticks(ticks=range(df_sub['date_to_number'].max() + 1))  # Adjust range as needed
    plt.yticks(ticks=[0, 25, 50, 75, 100])
    
    # Show the legend
    plt.legend(title='Legend', loc='upper right', bbox_to_anchor=(1.3, 1))
    
    # Save the plot if output file is specified, then show it either way
    save_or_show(output_file, dpi=300, close_after_save=False)