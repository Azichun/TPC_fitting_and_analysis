import os, logging, warnings, winsound, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from nlsfunc import *
from tqdm import tqdm
import nlsfunc

matplotlib.use("svg")

logging.basicConfig(level=logging.DEBUG, format="")
warnings.simplefilter("ignore")

######################
##### User input #####
######################

# The folder where all individual heart rate traces and only these traces are located (in csv format)
# First column: body temperature
# Second column: Heart rate
df_path = "F:\\mphil\\Github\\TPC_fitting_and_analysis_for_publish\\test_data"
save_path = "F:\\mphil\\Github\\TPC_fitting_and_analysis_for_publish\\test_data\\para"

# Choose which function to fit to the raw data
func = ["quaqua"]

# May modify the function "randomize" in module "nlsfunc" if necessary

###############
##### Run #####
###############
if not os.path.exists(save_path):
    os.makedirs(save_path)  # create directory for output if not exist

dfs = {csv.replace(".csv", "").replace(".", "_"): pd.read_csv(f"{df_path}\\{csv}").dropna().to_numpy(dtype=np.float64) for
       csv in os.listdir(df_path) if csv[-3:] == "csv"}  # import heart rate traces where key is name and value is data
out = TPC_fit(dfs, func, audio=True)  # first curve fitting run
for k, v in out.items():
    v.to_csv(f"{save_path}\\{k}_param.csv")  # export finalized parameters

for k, v in out.items():
    for i in range(v.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(dfs[v.iloc[i][0]][:, 0], dfs[v.iloc[i][0]][:, 1])
        ax.plot(dfs[v.iloc[i][0]][:, 0], globals()[k](dfs[v.iloc[i][0]][:, 0], list(v.iloc[i][1:])))
        plt.title(f"{k}_{v.iloc[i][0]}")
        plt.savefig(f"{save_path}\\{k}_{v.iloc[i][0]}.jpg")  # visualization

##################
##### Re-run #####
##################
out = TPC_fit({k: v for k, v in dfs.items() if k in ["SO20S2D_L", "SO20S2H_S"]},
              func, audio=True, patience=100)  # re-run unsatisfactory curves
for k, v in out.items():
    v.to_csv(f"{save_path}\\{k}_param_rerun.csv")  # export finalized parameters # export finalized parameters in a separate csv

for k, v in out.items():
    for i in range(v.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(dfs[v.iloc[i][0]][:, 0], dfs[v.iloc[i][0]][:, 1])
        ax.plot(dfs[v.iloc[i][0]][:, 0], globals()[k](dfs[v.iloc[i][0]][:, 0], list(v.iloc[i][1:])))
        plt.title(f"{k}_{v.iloc[i][0]}")
        plt.savefig(f"{save_path}\\{k}_{v.iloc[i][0]}_rerun.jpg")  # visualization
