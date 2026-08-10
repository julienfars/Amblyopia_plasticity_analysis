# MRS_processing.py
"""MR spectroscopy (MRS) data processing and plotting."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from plot_utils import check_dataframe, save_or_show


def plot_mrs_fwhm_across_sessions(mrs_df, 
                                   id_vars, 
                                   value_vars, 
                                   tissue_type_col='Tissue Type', 
                                   value_name='Tissue Fraction', 
                                   row_col='location', 
                                   col_col='Metab', 
                                   x_col='session', 
                                   y_col='FWHM', 
                                   hue_col='participant_number', 
                                   figsize=(10, 6), 
                                   palette_box="vlag", 
                                   palette_line="husl", 
                                   save_path=None):
    """
    Creates a FacetGrid plot to visualize the FWHM across sessions for different metabolites and locations.
    
    Parameters:
        mrs_df (DataFrame): DataFrame containing the MRS tissue fraction and FWHM data.
        id_vars (list): Columns to keep fixed during melting (e.g., 'participant_number', 'session', 'location').
        value_vars (list): Columns to melt into rows (e.g., ['CSF', 'GM', 'WM']).
        tissue_type_col (str): Name for the new column that represents tissue types (default: 'Tissue Type').
        value_name (str): Name for the new column that holds the values (default: 'Tissue Fraction').
        row_col (str): Column to use for facet rows (default: 'location').
        col_col (str): Column to use for facet columns (default: 'Metab').
        x_col (str): Column to use for the x-axis (default: 'session').
        y_col (str): Column to use for the y-axis (default: 'FWHM').
        hue_col (str): Column to use for differentiating participants in the line plot (default: 'participant_number').
        figsize (tuple): Size of the figure (default: (10, 6)).
        palette_box (str): Color palette for the boxplot (default: "vlag").
        palette_line (str): Color palette for the line plot (default: "husl").

    Raises:
        ValueError: If the input DataFrame is empty or missing required columns.
        TypeError: If the input mrs_df is not a DataFrame.
    """
    
    check_dataframe(mrs_df, id_vars + value_vars + [x_col, y_col, hue_col, row_col, col_col], allow_empty=False)

    # Melt the dataframe to convert tissue fractions into rows
    melted_df = pd.melt(mrs_df, id_vars=id_vars, value_vars=value_vars, 
                        var_name=tissue_type_col, value_name=value_name)
    
    # Sort by participant and session for better readability
    melted_df = melted_df.sort_values(by=id_vars)
    
    # Set the style of the plot
    sns.set(style="whitegrid")
    
    # Initialize the FacetGrid
    plt.figure(figsize=figsize)
    g = sns.FacetGrid(melted_df, row=row_col, col=col_col, margin_titles=True, height=4, aspect=1.2, sharey=False)
    
    # Map a boxplot for each facet
    g.map_dataframe(sns.boxplot, x=x_col, y=y_col, whis=[0, 100], width=0.4, palette=palette_box)
    
    # Map a line plot to show individual participants
    g.map_dataframe(sns.lineplot, x=x_col, y=y_col, hue=hue_col, style=hue_col, 
                    markers=True, dashes=False, ci=None, palette=palette_line, alpha=1)
    
    # Adjust the legends and titles
    g.add_legend(title='Participant Number', bbox_to_anchor=(1.05, 1), loc='upper left')
    g.set_axis_labels('Session', 'FWHM')
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    
    # Adjust the layout to avoid overlap
    g.tight_layout()

    save_or_show(save_path)
    
    
def plot_mrs_fwhm_by_metab(mrs_qc_df, 
                             metabs, 
                             row_col='location', 
                             col_col='Metab', 
                             x_col='session', 
                             y_col='FWHM', 
                             hue_col='participant_number', 
                             figsize=(10, 6), 
                             palette_box="vlag", 
                             palette_line="husl", 
                             session_order=['pre', 'post'], 
                             save_path=None):
    """
    Creates a FacetGrid plot to visualize the FWHM for different metabolites across sessions and participants.
    
    Parameters:
        mrs_qc_df (DataFrame): DataFrame containing the MRS FWHM data.
        metabs (list): List of metabolites to include in the plot.
        row_col (str): Column to use for facet rows (default: 'location').
        col_col (str): Column to use for facet columns (default: 'Metab').
        x_col (str): Column to use for the x-axis (default: 'session').
        y_col (str): Column to use for the y-axis (default: 'FWHM').
        hue_col (str): Column to use for differentiating participants in the line plot (default: 'participant_number').
        figsize (tuple): Size of the figure (default: (10, 6)).
        palette_box (str): Color palette for the boxplot (default: "vlag").
        palette_line (str): Color palette for the line plot (default: "husl").
        session_order (list): Order of sessions to display on the x-axis (default: ['pre', 'post']).

    Raises:
        ValueError: If the input DataFrame is empty or missing required columns.
        TypeError: If the input mrs_qc_df is not a DataFrame.
    """
    
    check_dataframe(mrs_qc_df, [x_col, y_col, hue_col, row_col, col_col], allow_empty=False)

    # Ensure 'session' column is ordered as specified
    mrs_qc_df[x_col] = pd.Categorical(mrs_qc_df[x_col], categories=session_order, ordered=True)
    
    # Filter the DataFrame for only the relevant Metab values
    mrs_qc_df_filtered = mrs_qc_df[mrs_qc_df[col_col].isin(metabs)]
    
    # Set the style of the plot
    sns.set(style="whitegrid")
    
    # Initialize the FacetGrid with location as rows and Metab as columns
    plt.figure(figsize=figsize)
    g = sns.FacetGrid(mrs_qc_df_filtered, row=row_col, col=col_col, margin_titles=True, height=4, aspect=1.2, sharey=False)
    
    # Map a boxplot onto the grid
    g.map_dataframe(sns.boxplot, x=x_col, y=y_col, whis=[0, 100], width=0.4, palette=palette_box)
    
    # Map a line plot onto the grid for participant data
    g.map_dataframe(sns.lineplot, x=x_col, y=y_col, hue=hue_col, style=hue_col, 
                    markers=True, dashes=False, ci=None, palette=palette_line, alpha=1)
    
    # Adjust the legends and titles
    g.add_legend(title='Participant Number', bbox_to_anchor=(1.05, 1), loc='upper left')
    g.set_axis_labels('Session', 'FWHM')
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    
    # Adjust the layout to avoid overlap
    g.tight_layout()

    save_or_show(save_path)
    
    
def plot_tissue_fraction_by_type(mrs_tf_df, 
                                   tissue_types, 
                                   row_col='location', 
                                   col_col='Tissue Type',
                                   x_col='session', 
                                   y_col='Tissue Fraction', 
                                   hue_col='participant_number',
                                   figsize=(10, 6), 
                                   palette_box="vlag", 
                                   palette_line="husl", 
                                   session_order=['pre', 'post'], 
                                   save_path=None):
    """
    Creates a FacetGrid plot to visualize tissue fractions (CSF, GM, WM) across sessions and participants.
    
    Parameters:
        mrs_tf_df (DataFrame): DataFrame containing the MRS tissue fraction data.
        tissue_types (list): List of tissue types to include in the plot (e.g., ['CSF', 'GM', 'WM']).
        row_col (str): Column to use for facet rows (default: 'location').
        col_col (str): Column to use for facet columns (default: 'Tissue Type').
        x_col (str): Column to use for the x-axis (default: 'session').
        y_col (str): Column to use for the y-axis (default: 'Tissue Fraction').
        hue_col (str): Column to use for differentiating participants in the line plot (default: 'participant_number').
        figsize (tuple): Size of the figure (default: (10, 6)).
        palette_box (str): Color palette for the boxplot (default: "vlag").
        palette_line (str): Color palette for the line plot (default: "husl").
        session_order (list): Order of sessions to display on the x-axis (default: ['pre', 'post']).

    Raises:
        ValueError: If the input DataFrame is empty or missing required columns.
        TypeError: If the input mrs_tf_df is not a DataFrame.
    """
    
    required_columns = ['participant_number', 'session', row_col]
    check_dataframe(mrs_tf_df, required_columns, allow_empty=False)

    # Melt the dataframe to make tissue types into rows
    mrs_tf_melted = pd.melt(mrs_tf_df, 
                             id_vars=required_columns, 
                             value_vars=tissue_types, 
                             var_name=col_col, 
                             value_name=y_col)

    # Ensure 'session' column is ordered
    mrs_tf_melted[x_col] = pd.Categorical(mrs_tf_melted[x_col], categories=session_order, ordered=True)

    # Sort the DataFrame for readability
    mrs_tf_melted = mrs_tf_melted.sort_values(by=['participant_number', x_col, row_col])

    # Set the style of the plot
    sns.set(style="whitegrid")

    # Initialize the FacetGrid with location as rows and Tissue Type as columns
    plt.figure(figsize=figsize)
    g = sns.FacetGrid(mrs_tf_melted, 
                       row=row_col, 
                       col=col_col, 
                       margin_titles=True, 
                       height=4, 
                       aspect=1.2, 
                       sharey=False)

    # Map a boxplot onto the grid
    g.map_dataframe(sns.boxplot, x=x_col, y=y_col, whis=[0, 100], width=0.4, palette=palette_box)

    # Map a line plot onto the grid for participant data
    g.map_dataframe(sns.lineplot, x=x_col, y=y_col, hue=hue_col, style=hue_col,
                    markers=True, dashes=False, ci=None, palette=palette_line, alpha=1)

    # Adjust the legends and titles
    g.add_legend(title='Participant Number', bbox_to_anchor=(1.05, 1), loc='upper left')
    g.set_axis_labels('Session', 'Tissue Fraction')
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    # Adjust the layout to avoid overlap
    g.tight_layout()

    save_or_show(save_path)
    
    

def plot_metabolite_crlb_by_session(mrs_diff_df, 
                                      metabs, 
                                      row_col='location', 
                                      col_col='Metab',
                                      x_col='session', 
                                      y_col='mM CRLB', 
                                      hue_col='participant_number',
                                      figsize=(10, 6), 
                                      palette_box="vlag", 
                                      palette_line="husl", 
                                      session_order=['pre', 'post'], 
                                      save_path=None):
    """
    Creates a FacetGrid plot to visualize CRLB (Concentration Ratio Limit Bound) values for metabolites across sessions.
    
    Parameters:
        mrs_diff_df (DataFrame): DataFrame containing the MRS CRLB data.
        metabs (list): List of metabolite values to include in the plot (e.g., ['GABA', 'sysMM', 'GABA+sysMM']).
        row_col (str): Column to use for facet rows (default: 'location').
        col_col (str): Column to use for facet columns (default: 'Metab').
        x_col (str): Column to use for the x-axis (default: 'session').
        y_col (str): Column to use for the y-axis (default: 'mM CRLB').
        hue_col (str): Column to use for differentiating participants in the line plot (default: 'participant_number').
        figsize (tuple): Size of the figure (default: (10, 6)).
        palette_box (str): Color palette for the boxplot (default: "vlag").
        palette_line (str): Color palette for the line plot (default: "husl").
        session_order (list): Order of sessions to display on the x-axis (default: ['pre', 'post']).

    Raises:
        ValueError: If the input DataFrame is empty or missing required columns.
        TypeError: If the input mrs_diff_df is not a DataFrame.
    """
    
    required_columns = ['participant_number', 'session', row_col]
    check_dataframe(mrs_diff_df, required_columns, allow_empty=False)

    # Ensure 'session' column is ordered
    mrs_diff_df[x_col] = pd.Categorical(mrs_diff_df[x_col], categories=session_order, ordered=True)
    
    # Filter the DataFrame for only the relevant Metab values
    df_filtered = mrs_diff_df[mrs_diff_df[col_col].isin(metabs)]
    
    # Set the style of the plot
    sns.set(style="whitegrid")
    
    # Initialize the FacetGrid with location as rows and Metab as columns
    plt.figure(figsize=figsize)
    g = sns.FacetGrid(df_filtered, 
                       row=row_col, 
                       col=col_col, 
                       margin_titles=True, 
                       height=4, 
                       aspect=1.2, 
                       sharey=False)
    
    # Map a boxplot onto the grid
    g.map_dataframe(sns.boxplot, x=x_col, y=y_col, whis=[0, 100], width=0.4, palette=palette_box)
    
    # Map a line plot onto the grid for participant data
    g.map_dataframe(sns.lineplot, x=x_col, y=y_col, hue=hue_col, style=hue_col,
                    markers=True, dashes=False, ci=None, palette=palette_line, alpha=1)
    
    # Adjust the legends and titles
    g.add_legend(title='Participant Number', bbox_to_anchor=(1.05, 1), loc='upper left')
    g.set_axis_labels('Session', y_col)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    
    # Adjust the layout to avoid overlap
    g.tight_layout()
    
    save_or_show(save_path)
    
    
def process_mrs_diff_data(mrs_diff_df, 
                           metabs, 
                           locations, 
                           session_order=['pre', 'post']):
    """
    Processes MRS data by creating separate DataFrames for each combination of metabolite and location.
    
    Parameters:
        mrs_diff_df (DataFrame): DataFrame containing MRS data with 'Metab', 'location', and 'session' columns.
        metabs (list): List of metabolites to filter (e.g., ['GABA', 'GSH', 'Glu']).
        locations (list): List of locations to filter (e.g., ['M1', 'V1']).
        session_order (list): Ordered list of sessions (default: ['pre', 'post']).
        
    Returns:
        tuple: A filtered DataFrame with only the relevant metabolites and a dictionary where the keys are dynamic DataFrame names,
               and the values are the corresponding filtered DataFrames.

    Raises:
        ValueError: If the input DataFrame is empty or missing required columns.
        TypeError: If the input mrs_diff_df is not a DataFrame.
    """
    
    check_dataframe(mrs_diff_df, ['Metab', 'location', 'session'], allow_empty=False)

    # Ensure 'session' column is ordered correctly
    mrs_diff_df['session'] = pd.Categorical(mrs_diff_df['session'], categories=session_order, ordered=True)
    
    # Dictionary to store the dynamically created DataFrames
    df_dict = {}
    
    # Loop through each Metab value and each location
    for metab in metabs:
        for location in locations:
            # Filter the DataFrame for the current Metab and location
            df_metab_location = mrs_diff_df[(mrs_diff_df['Metab'] == metab) & (mrs_diff_df['location'] == location)]
            # Dynamically create a name for each DataFrame
            df_name = f"df_{metab.replace('+', '_').replace(' ', '_')}_{location}_diff"
            
            # Assign the DataFrame to the dictionary
            df_dict[df_name] = df_metab_location
    
    # Filter the DataFrame for only the relevant Metab values
    diff_filtered = mrs_diff_df[mrs_diff_df['Metab'].isin(metabs)]
    
    return diff_filtered, df_dict


def plot_mrs_global_effect(df_dict, output_path=None):
    """
    Plots MRS data from a dictionary of DataFrames for different metabolite and location combinations.
    
    Parameters:
        df_dict (dict): Dictionary of DataFrames containing filtered MRS data 
                        (key format: 'df_{metab}_{location}_diff').
        output_path (str, optional): Path to save the plots. If None, the plots are displayed but not saved.

    Raises:
        TypeError: If df_dict is not a dictionary.
    """
    
    # Check if df_dict is a dictionary
    if not isinstance(df_dict, dict):
        raise TypeError("df_dict must be a dictionary.")
    
    # Loop through each DataFrame in the dictionary and plot
    for df_name, df in df_dict.items():
        required_columns = ['session', 'mMol/kg', 'location', 'Metab', 'participant_number']
        check_dataframe(df, required_columns, df_name=f"DataFrame '{df_name}'")

        # Initialize the FacetGrid with location as rows and Metab as columns, and independent y-axes
        g = sns.FacetGrid(df, row="location", col="Metab", margin_titles=True, height=4, aspect=1.2, sharey=False)

        # Map a boxplot onto the grid
        g.map_dataframe(sns.boxplot, x='session', y='mMol/kg', whis=[0, 100], width=0.4, palette="vlag")

        # Map a line plot onto the grid for participant data
        g.map_dataframe(sns.lineplot, x='session', y='mMol/kg', hue='participant_number', style='participant_number', 
                        markers=True, dashes=False, ci=None, palette="husl", alpha=1)

        # Adjust the legends and titles
        g.add_legend(title='Participant Number', bbox_to_anchor=(1.05, 1), loc='upper left')
        g.set_axis_labels('Session', 'mMol/kg')
        g.set_titles(col_template="{col_name}", row_template="{row_name}")

        # Adjust the layout to avoid overlap
        g.tight_layout()

        plot_path = f"{output_path}/MRS_global_effect_{df_name}.png" if output_path else None
        save_or_show(plot_path)


        
        
def plot_correlations_with_conditions(df_dict, Cut_off_diff=None, abc_df=None, abc_df_bi=None, Ast_diff=None, 
                                      conditions=None, 
                                      cutoff_col='Diff', abc_col='ABC', asteroid_col='Diff', 
                                      mrs_col='diff', session_col='session', 
                                      cutoff_freq_col=None, abc_bi_col=None, 
                                      save_path=None):
    """
    Plots correlations between metabolite concentrations and cutoff, ABC, ABC_bi, and asteroid values for different conditions.

    For each metabolite/location DataFrame in `df_dict`, produces up to three figures:
    one row of cutoff-frequency-vs-metabolite scatter/regression plots (one subplot per
    condition), one equivalent row for ABC, and a final two-panel figure for asteroid
    (stereoacuity) and ABC_bi, each annotated with its Pearson r and p-value. Any of
    Cut_off_diff/abc_df/abc_df_bi/Ast_diff can be omitted to skip its plots.

    Parameters:
    - df_dict: Dictionary of DataFrames containing metabolite concentration differences.
    - Cut_off_diff: DataFrame containing cutoff values (default None).
    - abc_df: DataFrame containing ABC values (default None).
    - abc_df_bi: DataFrame containing ABC_bi values (default None).
    - Ast_diff: DataFrame containing asteroid values (default None).
    - conditions: List of conditions (e.g., ['Amb', 'Both', 'Strong']) to filter and plot. Default is ['Amb', 'Both', 'Strong'].
    - cutoff_col: Column name for cutoff values in Cut_off_diff (default 'Diff').
    - abc_col: Column name for ABC values in abc_df (default 'ABC').
    - asteroid_col: Column name for asteroid values in Ast_diff (default 'Diff').
    - mrs_col: Column name for MRS concentration differences (default 'diff').
    - session_col: Column name for session (default 'session').
    - cutoff_freq_col: Column name for frequency in Cut_off_diff (default None).
    - abc_bi_col: Column name for ABC_bi in abc_df_bi (default None).
    """

    if conditions is None:
        conditions = ['Amb', 'Both', 'Strong']

    # Process DataFrames if provided
    if Cut_off_diff is not None:
        # Ensure it has the required columns
        if cutoff_col not in Cut_off_diff.columns:
            raise ValueError(f"{cutoff_col} not found in Cut_off_diff DataFrame.")
        
    if abc_df is not None:
        # Ensure it has the required columns
        if abc_col not in abc_df.columns:
            raise ValueError(f"{abc_col} not found in abc_df DataFrame.")
    
    if abc_df_bi is not None:
        # Ensure it has the required columns
        if abc_bi_col is not None and abc_bi_col not in abc_df_bi.columns:
            raise ValueError(f"{abc_bi_col} not found in abc_df_bi DataFrame.")
    
    if Ast_diff is not None:
        # Ensure it has the required columns
        if asteroid_col not in Ast_diff.columns:
            raise ValueError(f"{asteroid_col} not found in Ast_diff DataFrame.")

    # Loop through each DataFrame and its corresponding name
    for df_name, df in df_dict.items():
        # Process the current DataFrame: pivot and calculate the difference (pre - post)
        df_processed = df.pivot(index=['participant_number'], columns=session_col, values='mMol/kg').reset_index()
        df_processed[mrs_col] = df_processed['pre'] - df_processed['post']

        # Merge with Cut-off_diff and abc_df for correlations if provided
        if Cut_off_diff is not None:
            df_merged_cutoff = pd.merge(Cut_off_diff, df_processed, left_on='ID', right_on='participant_number')
        else:
            df_merged_cutoff = df_processed

        if abc_df is not None:
            df_merged_abc = pd.merge(abc_df, df_processed, left_on='ID', right_on='participant_number')
        else:
            df_merged_abc = df_processed

        # Create subplots for Cut-off and ABC correlations
        fig_cutoff, axes_cutoff = plt.subplots(1, len(conditions), figsize=(18, 5), sharey=True)
        fig_abc, axes_abc = plt.subplots(1, len(conditions), figsize=(18, 5), sharey=True)

        for i, condition in enumerate(conditions):
            # Filter the data for the current condition
            df_condition_cutoff = df_merged_cutoff[df_merged_cutoff['amb'] == condition]
            df_condition_abc = df_merged_abc[df_merged_abc['amb'] == condition]

            # Pearson correlation for Cut-off
            if not df_condition_cutoff.empty:
                valid_cutoff = df_condition_cutoff[[cutoff_col, mrs_col]].dropna()
                if not valid_cutoff.empty:
                    r_value_cutoff, p_value_cutoff = stats.pearsonr(valid_cutoff[cutoff_col], valid_cutoff[mrs_col])
                else:
                    r_value_cutoff, p_value_cutoff = float('nan'), float('nan')  # Handle empty data
                
                # Create the plot for Cut-off
                sns.regplot(x=cutoff_col, y=mrs_col, data=df_condition_cutoff, scatter_kws={'s': 50}, line_kws={'color': 'red'}, ax=axes_cutoff[i])
                axes_cutoff[i].set_title(f'{condition} eye\nr = {r_value_cutoff:.2f}, p = {p_value_cutoff:.2e}')
                axes_cutoff[i].set_xlabel(f'Diff ({condition})')
                if i == 0:
                    axes_cutoff[i].set_ylabel(f'{df_name} concentration (mMol/kg)')

            # Pearson correlation for ABC
            if not df_condition_abc.empty:                
                valid_abc = df_condition_abc[[abc_col, mrs_col]].dropna()
                if not valid_abc.empty:
                    r_value_abc, p_value_abc = stats.pearsonr(valid_abc[abc_col], valid_abc[mrs_col])
                else:
                    r_value_abc, p_value_abc = float('nan'), float('nan')  # Handle empty data

                # Create the plot for ABC
                sns.regplot(x=abc_col, y=mrs_col, data=df_condition_abc, scatter_kws={'s': 50}, line_kws={'color': 'blue'}, ax=axes_abc[i])
                axes_abc[i].set_title(f'ABC {condition} eye\nr = {r_value_abc:.2f}, p = {p_value_abc:.2e}')
                axes_abc[i].set_xlabel(f'ABC ({condition})')
                if i == 0:
                    axes_abc[i].set_ylabel(f'{df_name} concentration (mMol/kg)')

        # Adjust layout and show the Cut-off figure
        fig_cutoff.suptitle(f'Correlations between Cutoff values and {df_name} concentration', y=1.02)
        plt.tight_layout()
        plt.show()

        # Adjust layout and show the ABC figure
        fig_abc.suptitle(f'Correlations between ABC values and {df_name} concentration', y=1.02)
        plt.tight_layout()
        plt.show()

        # Merge with Asteroid values and ABC_bi if provided
        if Ast_diff is not None:
            df_merged_ast = pd.merge(Ast_diff, df_processed, left_on='ID', right_on='participant_number')
        else:
            df_merged_ast = df_processed

        if abc_df_bi is not None:
            df_merged_abc_bi = pd.merge(abc_df_bi, df_processed, left_on='ID', right_on='participant_number')
        else:
            df_merged_abc_bi = df_processed

        # Create subplots for Asteroid and ABC_bi correlations
        fig_ast, ax_ast = plt.subplots(1, 2, figsize=(12, 5))

        # Pearson correlation for Asteroid
        if not df_merged_ast.empty:
            valid_ast = df_merged_ast[[asteroid_col, mrs_col]].dropna()
            if not valid_ast.empty:
                r_value_ast, p_value_ast = stats.pearsonr(valid_ast[asteroid_col], valid_ast[mrs_col])
            else:
                r_value_ast, p_value_ast = float('nan'), float('nan')  # Handle empty data
            
            
            
            sns.regplot(x=asteroid_col, y=mrs_col, data=df_merged_ast, scatter_kws={'s': 50}, line_kws={'color': 'red'}, ax=ax_ast[0])
            ax_ast[0].set_title(f'Asteroid values\nr = {r_value_ast:.2f}, p = {p_value_ast:.2e}')
            ax_ast[0].set_ylabel(f'{df_name} concentration (mMol/kg)')
            ax_ast[0].set_xlabel('Asteroid values (log Arcsec)')

        # Pearson correlation for ABC_bi
        if abc_df_bi is not None and not df_merged_abc_bi.empty:            
            valid_bi = df_merged_abc_bi[[abc_bi_col, mrs_col]].dropna()
            if not valid_bi.empty:
                r_value_abc_bi, p_value_abc_bi = stats.pearsonr(valid_bi[abc_bi_col], valid_bi[mrs_col])
            else:
                r_value_abc_bi, p_value_abc_bi = float('nan'), float('nan')  # Handle empty data
            
            
            sns.regplot(x=abc_bi_col, y=mrs_col, data=df_merged_abc_bi, scatter_kws={'s': 50}, line_kws={'color': 'blue'}, ax=ax_ast[1])
            ax_ast[1].set_title(f'ABC_bi values\nr = {r_value_abc_bi:.2f}, p = {p_value_abc_bi:.2e}')
            ax_ast[1].set_ylabel(f'{df_name} concentration (mMol/kg)')
            ax_ast[1].set_xlabel('ABC_bi values')

        # Adjust layout and show
        plt.tight_layout()
        save_or_show(save_path)


def plot_paired_sessions(df, requested_label, y_col, y_label,
                          location_col='location', session_col='session', id_col='participant_number',
                          location_name_map=None, session_order=('pre', 'post'),
                          ylim=None, sep=0.25, figsize=(2, 6), show_diag=False):
    """
    Plot per-subject values at two sessions (e.g. pre/post) for one anatomical location,
    with faint per-subject connecting lines and a connected median marker per session.

    `requested_label` accepts either the DataFrame's own location code (e.g. 'V1') or,
    if `location_name_map` is given (e.g. {'V1': 'EVC', 'M1': 'SMC'}), the friendly name.

    Returns the Axes so callers can add custom annotations before saving/showing.
    """
    location_name_map = location_name_map or {}
    friendly_to_code = {friendly: code for code, friendly in location_name_map.items()}

    # Resolve `requested_label` to the DataFrame's own location code: try an exact
    # match against the friendly names, then the codes themselves, then fall back to
    # a case-insensitive substring match against either. If nothing matches, use the
    # request as-is (the row lookup just below will raise if that code isn't present).
    req = str(requested_label).strip()
    if req in friendly_to_code:
        code_label, friendly_label = friendly_to_code[req], req
    elif req in location_name_map:
        code_label, friendly_label = req, location_name_map[req]
    else:
        all_codes = list(location_name_map.keys())
        all_friendly = list(location_name_map.values())
        matches_code = [c for c in all_codes if req.lower() in c.lower()]
        matches_friendly = [f for f in all_friendly if req.lower() in f.lower()]
        if matches_code:
            code_label = matches_code[0]
            friendly_label = location_name_map.get(code_label, code_label)
        elif matches_friendly:
            friendly_label = matches_friendly[0]
            code_label = friendly_to_code[friendly_label]
        else:
            code_label, friendly_label = req, req

    if show_diag:
        print(f"All distinct {location_col} values: {sorted(df[location_col].astype(str).unique())}")
        print(f"Resolved request {requested_label!r} -> code '{code_label}' (friendly '{friendly_label}').")

    df_loc = df[df[location_col].astype(str) == code_label].copy()
    if df_loc.empty:
        df_loc = df[df[location_col].astype(str).str.lower() == code_label.lower()].copy()
    if df_loc.empty:
        raise ValueError(f"No rows found for location '{code_label}'.")

    df_loc[session_col] = df_loc[session_col].astype(str).str.lower().str.strip()
    session_ord = [s for s in session_order if s in df_loc[session_col].unique()]

    if len(session_ord) == 2:
        x_positions = [0.5 - sep / 2.0, 0.5 + sep / 2.0]
    else:
        x_positions = [0.5]
    x_map = dict(zip(session_ord, x_positions))

    plt.figure(figsize=figsize)
    ax = plt.gca()

    for sess in session_ord:
        rows = df_loc[df_loc[session_col] == sess]
        ax.scatter(np.repeat(x_map[sess], rows.shape[0]), rows[y_col].values,
                   s=40, marker='o', facecolor='black', edgecolor='black',
                   linewidth=0.6, alpha=0.2, zorder=5)

    n_lines = 0
    for _, grp in df_loc.groupby(id_col):
        vals = {row[session_col]: row[y_col] for _, row in grp.iterrows()}
        if 'pre' in vals and 'post' in vals:
            ax.plot([x_map['pre'], x_map['post']], [vals['pre'], vals['post']],
                    color='black', alpha=0.2, linewidth=1, zorder=3)
            n_lines += 1

    if session_ord:
        medians = df_loc.groupby(session_col)[y_col].median().reindex(session_ord)
        for sess in session_ord:
            ax.scatter([x_map[sess]], [medians.loc[sess]], s=140, marker='o',
                       facecolor='blue', edgecolor='black', linewidth=1.5, zorder=20)
        if len(session_ord) >= 2:
            ax.plot([x_map[s] for s in session_ord], [medians.loc[s] for s in session_ord],
                    color='blue', linewidth=2, zorder=15)

    if show_diag:
        print(f"Filtered rows: {df_loc.shape[0]}; plotted {n_lines} subject lines.")

    ax.set_xlim(0.25, 0.75)
    ax.set_xticks([x_map[s] for s in session_ord])
    ax.set_xticklabels([s.capitalize() for s in session_ord], fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_xlabel("")
    ax.set_title(friendly_label, fontsize=16)
    ax.tick_params(axis='both', which='major', width=2, length=6, colors='black')
    ax.tick_params(axis='both', which='minor', width=1, length=3, colors='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.xticks([x_map[s] for s in session_ord],
               [s.capitalize() + '-Training' for s in session_ord], fontsize=13)

    return ax