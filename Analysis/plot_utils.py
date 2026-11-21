# plot_utils.py
"""Shared validation and save/show helpers used by BV_processing, MRS_processing, and VR_processing."""

import os

import pandas as pd
import matplotlib.pyplot as plt


def check_dataframe(df, required_columns, df_name="Input", allow_empty=True):
    """Raise if `df` isn't a DataFrame with the given columns (and, optionally, isn't empty)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{df_name} must be a pandas DataFrame.")
    if not allow_empty and df.empty:
        raise ValueError(f"{df_name} cannot be empty.")
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def save_or_show(save_path=None, dpi=None, close_after_save=True):
    """Save the current matplotlib figure to `save_path`, or show it if no path is given."""
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        kwargs = {"bbox_inches": "tight"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        plt.savefig(save_path, **kwargs)
        if close_after_save:
            plt.close()
            return
    plt.show()