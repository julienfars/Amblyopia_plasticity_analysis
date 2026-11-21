# my_imports.py

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openpyxl
import pingouin as pg

from glob import glob
from datetime import datetime
from scipy.optimize import least_squares, curve_fit
from scipy.integrate import simpson, quad
from scipy import stats
from scipy.stats import linregress, pearsonr, spearmanr
from numpy.polynomial import Polynomial
from sklearn.linear_model import LinearRegression
from matplotlib.legend_handler import HandlerTuple

import BV_processing as bv
import MRS_processing as mrs
import VR_processing as vr

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Create the figure output folders the notebooks save into (Data/Plots is not
# tracked in git, so it doesn't exist on a fresh clone).
_PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Plots")
for _sub in ["", "new_new", "Narrative_plots", "Narrative_plots/v3", "Narrative_plots/fam"]:
    os.makedirs(os.path.join(_PLOTS_DIR, _sub), exist_ok=True)
