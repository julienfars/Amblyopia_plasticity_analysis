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
from scipy.stats import linregress, pearsonr, spearmanr, stats
from numpy.polynomial import Polynomial
from sklearn.linear_model import LinearRegression
from matplotlib.legend_handler import HandlerTuple

import BV_processing as bv
import MRS_processing as mrs
import VR_processing as vr

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
