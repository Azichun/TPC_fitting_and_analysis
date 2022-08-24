import setup
import nlsfunc
import os, logging, warnings, winsound
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from nlsfunc import *
from tqdm import tqdm

######################
##### User input #####
######################

# The folder where all individual heart rate traces and only these traces are located (in csv format)
# First column: body temperature
# Second column: Heart rate
path = "F:\\mphil\\Github\\Curve-Fitting\\test_data"

###################
##### Backend #####
###################
logging.basicConfig(level=logging.DEBUG, format="")
warnings.simplefilter("ignore")

os.chdir(path) #set the folder containing traces as directory
df = {csv.replace(".csv", ""):pd.read_csv(csv).dropna().to_numpy(dtype=np.float64)
      for csv in os.listdir()} #store all individual heart rate traces into one dictionary


##########################################################################################
for csv in [csv for csv in os.listdir() if csv[-3:] == "csv"]:
    df_name = csv.replace(".csv", "").replace(".", "_")
    globals()[df_name] = pd.read_csv(csv).dropna().to_numpy(dtype=np.float64)

dfs_name = [obj_name for obj_name in dir() if season in obj_name]
dfs = [globals()[obj_name] for obj_name in dir() if season in obj_name]

error_threshold = np.inf
no_of_groups = 100
ind_per_group = 200
perm = no_of_groups * ind_per_group

##############################
##### beta-beta function #####
##############################

# count = 0
# results = []
# while True:
#     df = dfs[count]
#     df_name = dfs_name[count]
#     xdata = df[:, 0]
#     ydata = df[:, 1]
#
#     k1 = np.random.exponential(1, perm)
#     min1 = np.random.uniform(-20, 20, perm)
#     max1 = np.random.uniform(43, 60, perm)
#     m1 = np.random.exponential(1, perm)
#     n1 = np.random.exponential(1, perm)
#     k2 = np.random.exponential(1, perm)
#     min2 = np.random.uniform(-20, 20, perm)
#     max2 = np.random.uniform(43, 60, perm)
#     m2 = np.random.exponential(1, perm)
#     n2 = np.random.exponential(1, perm)
#     a1 = np.random.exponential(1, perm)
#     bp1 = np.random.uniform(30, 50, perm)
#
#     param = np.array([k1, min1, max1, m1, n1, k2, min2, max2, m2, n2, a1, bp1]).transpose()
#
#     rmse_final = np.full(no_of_groups, np.inf)
#     param_final = np.full((no_of_groups, param.shape[1]), np.inf)
#
#     print(f"Fitting curve for individual {df_name} ...")
#     for group in tqdm(range(no_of_groups)):
#         rmse_group = np.full(ind_per_group, np.inf)
#         for ind in range(ind_per_group):
#             yrandom = bibet(xdata, param[group * ind_per_group + ind])
#             rmse_group[ind] = ((yrandom - ydata) ** 2).mean()
#         try:
#             best_group = np.where(rmse_group == min(rmse_group))[0][0]
#         except(IndexError):
#             best_group = 0
#         try:
#             param_final[group], _ = curve(bibet, xdata, ydata, p0=param[best_group])
#             ybest_group = bibet(xdata, param_final[group])
#             rmse_final[group] = ((ybest_group - ydata) ** 2).mean()
#         except(RuntimeError):
#             param_final[group] = np.zeros(param.shape[1])
#             rmse_final[group] = np.inf
#     if min(rmse_final) > error_threshold ** 2:
#         print(f"RMSE larger than {error_threshold ** 2}... Re-run...")
#         continue
#     try:
#         best_final = np.where(rmse_final == min(rmse_final))[0][0]
#         ybest_final = bibet(xdata, param_final[best_final])
#         results.append([df_name] + list(param_final[best_final]))
#         count += 1
#     except(IndexError):
#         print("Failed to converge... Re-run...")
#         continue
#     if count == len(dfs):
#         print("All curve-fittings are completed!")
#         break
#
# results_pd = pd.DataFrame(results)
# results_pd.columns = ["Individual", "k1", "min1", "max1", "m1", "n1", "k2", "min2", "max2", "m2", "n2", "a1", "bp1"]
# results_pd.to_csv("param_SO19W_bibet.csv")

#############################
##### beta-log function #####
#############################

# count = 0
# results = []
# while True:
#     df = dfs[count]
#     df_name = dfs_name[count]
#     xdata = df[:, 0]
#     ydata = df[:, 1]
#
#     k1 = np.random.exponential(1, perm)
#     min1 = np.random.uniform(-20, 20, perm)
#     max1 = np.random.uniform(43, 60, perm)
#     m1 = np.random.exponential(1, perm)
#     n1 = np.random.exponential(1, perm)
#     k2 = np.random.exponential(1, perm)
#     b2 = np.random.uniform(38, 55, perm)
#     c2 = np.random.uniform(-10, 10, perm)
#     a1 = np.random.exponential(1, perm)
#     bp1 = np.random.uniform(30, 50, perm)
#
#     param = np.array([k1, min1, max1, m1, n1, k2, b2, c2, a1, bp1]).transpose()
#
#     rmse_final = np.full(no_of_groups, np.inf)
#     param_final = np.full((no_of_groups, param.shape[1]), np.inf)
#
#     print(f"Fitting curve for individual {df_name} ...")
#     for group in tqdm(range(no_of_groups)):
#         rmse_group = np.full(ind_per_group, np.inf)
#         for ind in range(ind_per_group):
#             yrandom = betlog(xdata, param[group * ind_per_group + ind])
#             rmse_group[ind] = ((yrandom - ydata) ** 2).mean()
#         try:
#             best_group = np.where(rmse_group == min(rmse_group))[0][0]
#         except(IndexError):
#             best_group = 0
#         try:
#             param_final[group], _ = curve(betlog, xdata, ydata, p0=param[best_group])
#             ybest_group = betlog(xdata, param_final[group])
#             rmse_final[group] = ((ybest_group - ydata) ** 2).mean()
#         except(RuntimeError):
#             param_final[group] = np.zeros(param.shape[1])
#             rmse_final[group] = np.inf
#     if min(rmse_final) > error_threshold ** 2:
#         print(f"RMSE larger than {error_threshold ** 2}... Re-run...")
#         continue
#     try:
#         best_final = np.where(rmse_final == min(rmse_final))[0][0]
#         ybest_final = betlog(xdata, param_final[best_final])
#         results.append([df_name] + list(param_final[best_final]))
#         count += 1
#     except(IndexError):
#         print("Failed to converge... Re-run...")
#         continue
#     if count == len(dfs):
#         print("All curve-fittings are completed!")
#         break
#
# results_pd = pd.DataFrame(results)
# results_pd.columns = ["Individual", "k1", "min1", "max1", "m1", "n1", "k2", "b2", "c2", "a1", "bp1"]
# results_pd.to_csv("param_SO20S_betlog.csv")

##################################
##### gaussian-beta function #####
##################################

# count = 0
# results = []
# while True:
#     df = dfs[count]
#     df_name = dfs_name[count]
#     xdata = df[:, 0]
#     ydata = df[:, 1]
#
#     k1 = np.random.uniform(0, 20, perm)
#     u1 = np.random.uniform(25, 45, perm)
#     sig1 = np.random.exponential(1, perm)
#     k2 = np.random.uniform(0, 20, perm)
#     min2 = np.random.uniform(-20, 20, perm)
#     max2 = np.random.uniform(43, 60, perm)
#     m2 = np.random.exponential(1, perm)
#     n2 = np.random.exponential(1, perm)
#     a1 = np.random.exponential(1, perm)
#     bp1 = np.random.uniform(30, 50, perm)
#
#     param = np.array([k1, u1, sig1, k2, min2, max2, m2, n2, a1, bp1]).transpose()
#
#     rmse_final = np.full(no_of_groups, np.inf)
#     param_final = np.full((no_of_groups, param.shape[1]), np.inf)
#
#     print(f"Fitting curve for individual {df_name} ...")
#     for group in tqdm(range(no_of_groups)):
#         rmse_group = np.full(ind_per_group, np.inf)
#         for ind in range(ind_per_group):
#             yrandom = gaubet(xdata, param[group * ind_per_group + ind])
#             rmse_group[ind] = ((yrandom - ydata) ** 2).mean()
#         try:
#             best_group = np.where(rmse_group == min(rmse_group))[0][0]
#         except(IndexError):
#             best_group = 0
#         try:
#             param_final[group], _ = curve(gaubet, xdata, ydata, p0=param[best_group])
#             ybest_group = gaubet(xdata, param_final[group])
#             rmse_final[group] = ((ybest_group - ydata) ** 2).mean()
#         except(RuntimeError):
#             param_final[group] = np.zeros(param.shape[1])
#             rmse_final[group] = np.inf
#     if min(rmse_final) > error_threshold ** 2:
#         print(f"RMSE larger than {error_threshold ** 2}... Re-run...")
#         continue
#     try:
#         best_final = np.where(rmse_final == min(rmse_final))[0][0]
#         ybest_final = gaubet(xdata, param_final[best_final])
#         results.append([df_name] + list(param_final[best_final]))
#         count += 1
#     except(IndexError):
#         print("Failed to converge... Re-run...")
#         continue
#     if count == len(dfs):
#         print("All curve-fittings are completed!")
#         break
#
# results_pd = pd.DataFrame(results)
# results_pd.columns = ["Individual", "k1", "u1", "sig1", "k2", "min2", "max2", "m2", "n2", "a1", "bp1"]
# results_pd.to_csv(f"param_{season}_gaubet.csv")

######################################
##### gaussian-gaussian function #####
######################################

# count = 0
# results = []
# while True:
#     df = dfs[count]
#     df_name = dfs_name[count]
#     xdata = df[:, 0]
#     ydata = df[:, 1]
#
#     k1 = np.random.uniform(0, 20, perm)
#     u1 = np.random.uniform(25, 45, perm)
#     sig1 = np.random.exponential(1, perm)
#     k2 = np.random.uniform(0, 20, perm)
#     u2 = np.random.uniform(30, 55, perm)
#     sig2 = np.random.exponential(1, perm)
#     a1 = np.random.exponential(1, perm)
#     bp1 = np.random.uniform(30, 50, perm)
#
#     param = np.array([k1, u1, sig1, k2, u2, sig2, a1, bp1]).transpose()
#
#     rmse_final = np.full(no_of_groups, np.inf)
#     param_final = np.full((no_of_groups, param.shape[1]), np.inf)
#
#     print(f"Fitting curve for individual {df_name} ...")
#     for group in tqdm(range(no_of_groups)):
#         rmse_group = np.full(ind_per_group, np.inf)
#         for ind in range(ind_per_group):
#             yrandom = bigau(xdata, param[group * ind_per_group + ind])
#             rmse_group[ind] = ((yrandom - ydata) ** 2).mean()
#         try:
#             best_group = np.where(rmse_group == min(rmse_group))[0][0]
#         except(IndexError):
#             best_group = 0
#         try:
#             param_final[group], _ = curve(bigau, xdata, ydata, p0=param[best_group])
#             ybest_group = bigau(xdata, param_final[group])
#             rmse_final[group] = ((ybest_group - ydata) ** 2).mean()
#         except(RuntimeError):
#             param_final[group] = np.zeros(param.shape[1])
#             rmse_final[group] = np.inf
#     if min(rmse_final) > error_threshold ** 2:
#         print(f"RMSE larger than {error_threshold ** 2}... Re-run...")
#         continue
#     try:
#         best_final = np.where(rmse_final == min(rmse_final))[0][0]
#         ybest_final = bigau(xdata, param_final[best_final])
#         results.append([df_name] + list(param_final[best_final]))
#         count += 1
#     except(IndexError):
#         print("Failed to converge... Re-run...")
#         continue
#     if count == len(dfs):
#         print("All curve-fittings are completed!")
#         break
#
# results_pd = pd.DataFrame(results)
# results_pd.columns = ["Individual", "k1", "u1", "sig1", "k2", "u2", "sig2", "a1", "bp1"]
# results_pd.to_csv(f"param_{season}_bigau.csv")

########################################
##### quadratic-quadratic function #####
########################################

# count = 0
# results = []
# while True:
#     df = dfs[count]
#     df_name = dfs_name[count]
#     xdata = df[:, 0]
#     ydata = df[:, 1]
#
#     k1 = -np.random.exponential(1, perm)
#     qa1 = np.random.uniform(-20, 0, perm)
#     qb1 = np.random.uniform(0, 50, perm)
#     qc1 = np.random.uniform(100, 200, perm)
#     k2 = np.random.uniform(0, 20, perm)
#     qa2 = -np.random.exponential(1, perm)
#     qb2 = np.random.uniform(0, 50, perm)
#     qc2 = np.random.uniform(100, 200, perm)
#     a1 = np.random.exponential(1, perm)
#     bp1 = np.random.uniform(30, 50, perm)
#
#     param = np.array([k1, qa1, qb1, qc1, k2, qa2, qb2, qc2, a1, bp1]).transpose()
#
#     rmse_final = np.full(no_of_groups, np.inf)
#     param_final = np.full((no_of_groups, param.shape[1]), np.inf)
#
#     print(f"Fitting curve for individual {df_name} ...")
#     for group in tqdm(range(no_of_groups)):
#         rmse_group = np.full(ind_per_group, np.inf)
#         for ind in range(ind_per_group):
#             yrandom = biqua(xdata, param[group * ind_per_group + ind])
#             rmse_group[ind] = ((yrandom - ydata) ** 2).mean()
#         try:
#             best_group = np.where(rmse_group == min(rmse_group))[0][0]
#         except(IndexError):
#             best_group = 0
#         try:
#             param_final[group], _ = curve_fit(biqua, xdata, ydata, p0=param[best_group])
#             ybest_group = biqua(xdata, param_final[group])
#             rmse_final[group] = ((ybest_group - ydata) ** 2).mean()
#         except(RuntimeError):
#             param_final[group] = np.zeros(param.shape[1])
#             rmse_final[group] = np.inf
#     if min(rmse_final) > error_threshold ** 2:
#         print(f"RMSE larger than {error_threshold ** 2}... Re-run...")
#         continue
#     try:
#         best_final = np.where(rmse_final == min(rmse_final))[0][0]
#         ybest_final = biqua(xdata, param_final[best_final])
#         results.append([df_name] + list(param_final[best_final]))
#         count += 1
#     except(IndexError):
#         print("Failed to converge... Re-run...")
#         continue
#     if count == len(dfs):
#         print("All curve-fittings are completed!")
#         break
#
# results_pd = pd.DataFrame(results)
# results_pd.columns = ["Individual", "k1", "qa1", "qb1", "qc1", "k2", "qa2", "qb2", "qc2", "a1", "bp1"]
# results_pd.to_csv(f"param_{season}_biqua.csv")

########################################
##### quadratic-quadratic function #####
########################################

# count = 0
# results = []
# while True:
#     df = dfs[count]
#     df_name = dfs_name[count]
#     xdata = df[:, 0]
#     ydata = df[:, 1]
#
#     k1 = -np.random.exponential(1, perm)
#     qa1 = np.random.uniform(-20, 0, perm)
#     qb1 = np.random.uniform(0, 50, perm)
#     qc1 = np.random.uniform(100, 200, perm)
#     k2 = np.random.uniform(0, 20, perm)
#     u2 = np.random.uniform(30, 55, perm)
#     sig2 = np.random.exponential(1, perm)
#     a1 = np.random.exponential(1, perm)
#     bp1 = np.random.uniform(30, 50, perm)
#
#     param = np.array([k1, qa1, qb1, qc1, k2, u2, sig2, a1, bp1]).transpose()
#
#     rmse_final = np.full(no_of_groups, np.inf)
#     param_final = np.full((no_of_groups, param.shape[1]), np.inf)
#
#     print(f"Fitting curve for individual {df_name} ...")
#     for group in tqdm(range(no_of_groups)):
#         rmse_group = np.full(ind_per_group, np.inf)
#         for ind in range(ind_per_group):
#             yrandom = quagau(xdata, param[group * ind_per_group + ind])
#             rmse_group[ind] = ((yrandom - ydata) ** 2).mean()
#         try:
#             best_group = np.where(rmse_group == min(rmse_group))[0][0]
#         except(IndexError):
#             best_group = 0
#         try:
#             param_final[group], _ = curve_fit(quagau, xdata, ydata, p0=param[best_group])
#             ybest_group = quagau(xdata, param_final[group])
#             rmse_final[group] = ((ybest_group - ydata) ** 2).mean()
#         except(RuntimeError):
#             param_final[group] = np.zeros(param.shape[1])
#             rmse_final[group] = np.inf
#     if min(rmse_final) > error_threshold ** 2:
#         print(f"RMSE larger than {error_threshold ** 2}... Re-run...")
#         continue
#     try:
#         best_final = np.where(rmse_final == min(rmse_final))[0][0]
#         ybest_final = quagau(xdata, param_final[best_final])
#         results.append([df_name] + list(param_final[best_final]))
#         count += 1
#     except(IndexError):
#         print("Failed to converge... Re-run...")
#         continue
#     if count == len(dfs):
#         print("All curve-fittings are completed!")
#         break
#
# results_pd = pd.DataFrame(results)
# results_pd.columns = ["Individual", "k1", "qa1", "qb1", "qc1", "k2", "u2", "sig2", "a1", "bp1"]
# results_pd.to_csv(f"param_{season}_quagau.csv")

###################################
##### quadratic-beta function #####
###################################

# count = 0
# results = []
# while True:
#     df = dfs[count]
#     df_name = dfs_name[count]
#     xdata = df[:, 0]
#     ydata = df[:, 1]
#
#     k1 = -np.random.exponential(1, perm)
#     qa1 = np.random.uniform(-20, 0, perm)
#     qb1 = np.random.uniform(0, 50, perm)
#     qc1 = np.random.uniform(100, 200, perm)
#     k2 = np.random.uniform(0, 20, perm)
#     min2 = np.random.uniform(-20, 20, perm)
#     max2 = np.random.uniform(43, 60, perm)
#     m2 = np.random.exponential(1, perm)
#     n2 = np.random.exponential(1, perm)
#     a1 = np.random.exponential(1, perm)
#     bp1 = np.random.uniform(30, 50, perm)
#
#     param = np.array([k1, qa1, qb1, qc1, k2, min2, max2, m2, n2, a1, bp1]).transpose()
#
#     rmse_final = np.full(no_of_groups, np.inf)
#     param_final = np.full((no_of_groups, param.shape[1]), np.inf)
#
#     print(f"Fitting curve for individual {df_name} ...")
#     for group in tqdm(range(no_of_groups)):
#         rmse_group = np.full(ind_per_group, np.inf)
#         for ind in range(ind_per_group):
#             yrandom = quabet(xdata, param[group * ind_per_group + ind])
#             rmse_group[ind] = ((yrandom - ydata) ** 2).mean()
#         try:
#             best_group = np.where(rmse_group == min(rmse_group))[0][0]
#         except(IndexError):
#             best_group = 0
#         try:
#             param_final[group], _ = curve_fit(quabet, xdata, ydata, p0=param[best_group])
#             ybest_group = quabet(xdata, param_final[group])
#             rmse_final[group] = ((ybest_group - ydata) ** 2).mean()
#         except(RuntimeError):
#             param_final[group] = np.zeros(param.shape[1])
#             rmse_final[group] = np.inf
#     if min(rmse_final) > error_threshold ** 2:
#         print(f"RMSE larger than {error_threshold ** 2}... Re-run...")
#         continue
#     try:
#         best_final = np.where(rmse_final == min(rmse_final))[0][0]
#         ybest_final = quabet(xdata, param_final[best_final])
#         results.append([df_name] + list(param_final[best_final]))
#         count += 1
#     except(IndexError):
#         print("Failed to converge... Re-run...")
#         continue
#     if count == len(dfs):
#         print("All curve-fittings are completed!")
#         break
#
# results_pd = pd.DataFrame(results)
# results_pd.columns = ["Individual", "k1", "qa1", "qb1", "qc1", "k2", "min2", "max2", "m2", "n2", "a1", "bp1"]
# results_pd.to_csv(f"param_{season}_quabet.csv")

##############################
##### beta-beta function #####
##############################

# count = 0
# results = []
# while True:
#     df = dfs[count]
#     df_name = dfs_name[count]
#     xdata = df[:, 0]
#     ydata = df[:, 1]
#
#     Tmin1 = np.random.uniform(-20, 20, perm)
#     Tmax1 = np.random.uniform(43, 60, perm)
#     Tm1 = np.random.uniform(25, 50, perm)
#     Hmax1 = np.random.uniform(40, 100, perm)
#     Tmin2 = np.random.uniform(-20, 20, perm)
#     Tmax2 = np.random.uniform(43, 60, perm)
#     Tm2 = np.random.uniform(25, 50, perm)
#     Hmax2 = np.random.uniform(40, 100, perm)
#     a1 = np.random.exponential(1, perm)
#     bp1 = np.random.uniform(30, 50, perm)
#
#     param = np.array([Tmin1, Tmax1, Tm1, Hmax1, Tmin2, Tmax2, Tm2, Hmax2, a1, bp1]).transpose()
#
#     rmse_final = np.full(no_of_groups, np.inf)
#     param_final = np.full((no_of_groups, param.shape[1]), np.inf)
#
#     print(f"Fitting curve for individual {df_name} ...")
#     for group in tqdm(range(no_of_groups)):
#         rmse_group = np.full(ind_per_group, np.inf)
#         for ind in range(ind_per_group):
#             yrandom = betbet(xdata, param[group * ind_per_group + ind])
#             rmse_group[ind] = ((yrandom - ydata) ** 2).mean()
#         try:
#             best_group = np.where(rmse_group == min(rmse_group))[0][0]
#         except(IndexError):
#             best_group = 0
#         try:
#             param_final[group], _ = curve_fit(betbet, xdata, ydata, p0=param[best_group])
#             ybest_group = betbet(xdata, param_final[group])
#             rmse_final[group] = ((ybest_group - ydata) ** 2).mean()
#         except(RuntimeError):
#             param_final[group] = np.zeros(param.shape[1])
#             rmse_final[group] = np.inf
#     if min(rmse_final) > error_threshold ** 2:
#         print(f"RMSE larger than {error_threshold ** 2}... Re-run...")
#         continue
#     try:
#         best_final = np.where(rmse_final == min(rmse_final))[0][0]
#         ybest_final = betbet(xdata, param_final[best_final])
#         results.append([df_name] + list(param_final[best_final]))
#         count += 1
#     except(IndexError):
#         print("Failed to converge... Re-run...")
#         continue
#     if count == len(dfs):
#         print("All curve-fittings are completed!")
#         break
#
# results_pd = pd.DataFrame(results)
# results_pd.columns = ["Individual", "Tmin1", "Tmax1", "Tm1", "Hmax1", "Tmin2", "Tmax2", "Tm2", "Hmax2", "a1", "bp1"]
# results_pd.to_csv(f"param_{season}_betbet.csv")

##############################
##### beta-beta function #####
##############################
#
# count = 0
# results = []
# while True:
#     df = dfs[count]
#     df_name = dfs_name[count]
#     xdata = df[:, 0]
#     ydata = df[:, 1]
#
#     Tmin1 = np.random.uniform(-20, 20, perm)
#     Tmax1 = np.random.uniform(43, 60, perm)
#     Tm1 = np.random.uniform(25, 50, perm)
#     Hmax1 = np.random.uniform(40, 100, perm)
#     Tmin2 = np.random.uniform(-20, 20, perm)
#     Tmax2 = np.random.uniform(43, 60, perm)
#     Tm2 = np.random.uniform(25, 50, perm)
#     Hmax2 = np.random.uniform(40, 100, perm)
#     a1 = np.random.exponential(1, perm)
#     bp1 = np.random.uniform(30, 50, perm)
#
#     param = np.array([Tmin1, Tmax1, Tm1, Hmax1, Tmin2, Tmax2, Tm2, Hmax2, a1, bp1]).transpose()
#
#     rmse_final = np.full(no_of_groups, np.inf)
#     param_final = np.full((no_of_groups, param.shape[1]), np.inf)
#
#     print(f"Fitting curve for individual {df_name} ...")
#     for group in tqdm(range(no_of_groups)):
#         rmse_group = np.full(ind_per_group, np.inf)
#         for ind in range(ind_per_group):
#             yrandom = betbet(xdata, param[group * ind_per_group + ind])
#             rmse_group[ind] = ((yrandom - ydata) ** 2).mean()
#         try:
#             best_group = np.where(rmse_group == min(rmse_group))[0][0]
#         except(IndexError):
#             best_group = 0
#         try:
#             param_final[group], _ = curve_fit(betbet, xdata, ydata, p0=param[best_group])
#             ybest_group = betbet(xdata, param_final[group])
#             rmse_final[group] = ((ybest_group - ydata) ** 2).mean()
#         except(RuntimeError):
#             param_final[group] = np.zeros(param.shape[1])
#             rmse_final[group] = np.inf
#     if min(rmse_final) > error_threshold ** 2:
#         print(f"RMSE larger than {error_threshold ** 2}... Re-run...")
#         continue
#     try:
#         best_final = np.where(rmse_final == min(rmse_final))[0][0]
#         ybest_final = betbet(xdata, param_final[best_final])
#         results.append([df_name] + list(param_final[best_final]))
#         count += 1
#     except(IndexError):
#         print("Failed to converge... Re-run...")
#         continue
#     if count == len(dfs):
#         print("All curve-fittings are completed!")
#         break
#
# results_pd = pd.DataFrame(results)
# results_pd.columns = ["Individual", "Tmin1", "Tmax1", "Tm1", "Hmax1", "Tmin2", "Tmax2", "Tm2", "Hmax2", "a1", "bp1"]
# results_pd.to_csv(f"param_{season}_betbet.csv")

##################################
##### gaussian-beta function #####
##################################

# count = 0
# results = []
# while True:
#     df = dfs[count]
#     df_name = dfs_name[count]
#     xdata = df[:, 0]
#     ydata = df[:, 1]
#
#     k1 = np.random.exponential(1, perm)
#     u1 = np.random.uniform(25, 50, perm)
#     sig1 = np.random.exponential(1, perm)
#     Tmin2 = np.random.uniform(-20, 20, perm)
#     Tmax2 = np.random.uniform(43, 60, perm)
#     Tm2 = np.random.uniform(25, 50, perm)
#     Hmax2 = np.random.uniform(40, 100, perm)
#     a1 = np.random.exponential(1, perm)
#     bp1 = np.random.uniform(30, 50, perm)
#
#     param = np.array([k1, u1, sig1, Tmin2, Tmax2, Tm2, Hmax2, a1, bp1]).transpose()
#
#     rmse_final = np.full(no_of_groups, np.inf)
#     param_final = np.full((no_of_groups, param.shape[1]), np.inf)
#
#     print(f"Fitting curve for individual {df_name} ...")
#     for group in tqdm(range(no_of_groups)):
#         rmse_group = np.full(ind_per_group, np.inf)
#         for ind in range(ind_per_group):
#             yrandom = gaubet(xdata, param[group * ind_per_group + ind])
#             rmse_group[ind] = ((yrandom - ydata) ** 2).mean()
#         try:
#             best_group = np.where(rmse_group == min(rmse_group))[0][0]
#         except(IndexError):
#             best_group = 0
#         try:
#             param_final[group], _ = curve_fit(gaubet, xdata, ydata, p0=param[best_group])
#             ybest_group = gaubet(xdata, param_final[group])
#             rmse_final[group] = ((ybest_group - ydata) ** 2).mean()
#         except(RuntimeError):
#             param_final[group] = np.zeros(param.shape[1])
#             rmse_final[group] = np.inf
#     if min(rmse_final) > error_threshold ** 2:
#         print(f"RMSE larger than {error_threshold ** 2}... Re-run...")
#         continue
#     try:
#         best_final = np.where(rmse_final == min(rmse_final))[0][0]
#         ybest_final = gaubet(xdata, param_final[best_final])
#         results.append([df_name] + list(param_final[best_final]))
#         count += 1
#     except(IndexError):
#         print("Failed to converge... Re-run...")
#         continue
#     if count == len(dfs):
#         print("All curve-fittings are completed!")
#         break
#
# results_pd = pd.DataFrame(results)
# results_pd.columns = ["Individual", "k1", "u1", "sig1", "Tmin2", "Tmax2", "Tm2", "Hmax2", "a1", "bp1"]
# results_pd.to_csv(f"param_{season}_gaubet.csv")

##################################
##### quadratic-beta function #####
##################################

count = 0
results = []
while True:
    df = dfs[count]
    df_name = dfs_name[count]
    xdata = df[:, 0]
    ydata = df[:, 1]

    k1 = -np.random.exponential(1, perm)
    qa1 = np.random.uniform(-20, 0, perm)
    qb1 = np.random.uniform(0, 50, perm)
    qc1 = np.random.uniform(100, 200, perm)
    Tmin2 = np.random.uniform(-20, 20, perm)
    Tmax2 = np.random.uniform(43, 60, perm)
    Tm2 = np.random.uniform(25, 50, perm)
    Hmax2 = np.random.uniform(40, 100, perm)
    a1 = np.random.exponential(1, perm)
    bp1 = np.random.uniform(30, 50, perm)

    param = np.array([k1, qa1, qb1, qc1, Tmin2, Tmax2, Tm2, Hmax2, a1, bp1]).transpose()

    rmse_final = np.full(no_of_groups, np.inf)
    param_final = np.full((no_of_groups, param.shape[1]), np.inf)

    print(f"Fitting curve for individual {df_name} ...")
    for group in tqdm(range(no_of_groups)):
        rmse_group = np.full(ind_per_group, np.inf)
        for ind in range(ind_per_group):
            yrandom = quabet(xdata, param[group * ind_per_group + ind])
            rmse_group[ind] = ((yrandom - ydata) ** 2).mean()
        try:
            best_group = np.where(rmse_group == min(rmse_group))[0][0]
        except(IndexError):
            best_group = 0
        try:
            param_final[group], _ = curve_fit(quabet, xdata, ydata, p0=param[best_group])
            ybest_group = quabet(xdata, param_final[group])
            rmse_final[group] = ((ybest_group - ydata) ** 2).mean()
        except(RuntimeError):
            param_final[group] = np.zeros(param.shape[1])
            rmse_final[group] = np.inf
    if min(rmse_final) > error_threshold ** 2:
        print(f"RMSE larger than {error_threshold ** 2}... Re-run...")
        continue
    try:
        best_final = np.where(rmse_final == min(rmse_final))[0][0]
        ybest_final = quabet(xdata, param_final[best_final])
        results.append([df_name] + list(param_final[best_final]))
        count += 1
    except(IndexError):
        print("Failed to converge... Re-run...")
        continue
    if count == len(dfs):
        print("All curve-fittings are completed!")
        break

results_pd = pd.DataFrame(results)
results_pd.columns = ["Individual", "k1", "qa1", "qb1", "qc1", "Tmin2", "Tmax2", "Tm2", "Hmax2", "a1", "bp1"]
results_pd.to_csv(f"param_{season}_quabet.csv")

for _ in range(10):
    winsound.Beep(5000, 100)