import numpy as np
import pandas as pd
import matplotlib, PyQt5
import matplotlib.pyplot as plt
import os, csaps, copy, TPC, warnings, types
from nlsfunc import *
from importlib import reload
import pickle
import statsmodels.api as sm
from statsmodels.formula.api import ols

matplotlib.use("Qt5Agg")
warnings.simplefilter("ignore", RuntimeWarning)

#########################################
##### import saved TPCgroup objects #####
#########################################

# os.chdir("D:\\Mirror\\mphil\\Data\\Heart rate\\lab")
# for pickle_name in [file for file in os.listdir() if "pkl" in file]:
#     name = pickle_name.replace(".pkl", "")
#     with open(pickle_name, "rb") as pickle_file:
#         globals()[name] = pickle.load(pickle_file)

#############################
##### import parameters #####
#############################

funcs = [betbet, gaubet]  # functions use to fit the curves
func_names = [func.__name__ for func in funcs]  # store the names of functions
param_dfs = {} # initialize a dictionary

os.chdir("F:\\mphil\\Github\\TPC_fitting_and_analysis_for_publish\\test_data\\para")  # set directory to the folder coataining parameters
for func_name in func_names:  # for each function
    param_dfs[func_name] = pd.read_csv(f"{func_name}_param.csv", index_col=0)  # import parameters

###################################
##### create TPCgroup objects #####
###################################

width = 0.1  # set the resolution of temperature
x = np.arange(25, 55, width)  # create an array of temperatures

# Initialize a new TPC object
# First argument: the array of temperature
# Second argument: the parameters of one function
# Third argument: the function
Lottia = TPC.TPCgroup(x, param_dfs["betbet"], betbet)

for func_name, para in {n:p for n, p in param_dfs.items() if n != "betbet"}.items():  # for the remaining functions
    Lottia.import_fit(para, globals()[func_name])  # import parameters to the TPC object

############################
##### import raw data ######
############################

raw_path = "F:\\mphil\\Github\\TPC_fitting_and_analysis_for_publish\\test_data"  # set directory to the folder coataining raw data
raw = {csv.replace(".csv", "").replace(".", "_"): pd.read_csv(f"{raw_path}\\{csv}").dropna().to_numpy(dtype=np.float64) for
       csv in os.listdir(raw_path) if csv[-3:] == "csv"}  # import heart rate traces where key is name and value is data
Lottia.import_raw(raw)  # import the raw the data to the TPC object

Lottia.calc_accuracy()  # calculate the goodness-of-fit (i.e. rss, AIC and AICc) of different fitted functions
Lottia.confirm_fit("rss")  # select the best fit curve for each individual by which goodness-of-fit metric?
Lottia.calc_curvature(standardize=True)  # calculate the curvatures of the curves
Lottia.filter_breakpoint()  # remove break breakpoints matching the 2 criteria:
# (1) upward concaving  bp at the start; (2) less curved breakpoints in the same signed section

Lottia.plot(func_names=["betbet", "gaubet"])
Lottia.plot_curvature()
Lottia.plot_flatline()

#############################
##### fit cubic splines #####
#############################

obj_list = [summer_small, summer_large, winter_small, winter_large]
perm = 20000

for obj in obj_list:
    obj.calc_accuracy()
    obj.confirm_fit("rss")
    obj.calc_curvature(standardize=True)
    obj.filter_breakpoint()

# for obj in obj_list:
#     obj.plot_curvature()

for obj in obj_list:
    obj.fit_flatline(line, [[0, 50], [0, 50]], perm)
    obj.fit_flatline(expo, [[0, 20], [1.1, 2], [45, 3], [0, 50]], perm)
    obj.fit_flatline(loga, [[0, 50], [43, 4], [45, 2]], perm)
    obj.fit_flatline(qua, [[0, 50], [90, 20], [43, 4], [45, 2]], perm)

for obj in obj_list:
    obj.compare_flatline()
    obj.confirm_flatline_fit("aicc")

for obj in obj_list:
    obj.plot_flatline("all")

for obj in obj_list:
    print(obj.flt)


os.chdir("D:\\Mirror\\mphil\\Data\\Heart rate\\lab")
summer_small.save("Summer 10 - 18 mm", display_breakpoint=True, columns=3, plot=False)
summer_large.save("Summer 20 - 28 mm", display_breakpoint=True, columns=3, plot=False)
winter_small.save("Winter 10 - 18 mm", display_breakpoint=True, columns=3, plot=False)
winter_large.save("Winter 20 - 28 mm", display_breakpoint=True, columns=3, plot=False)

# summer_small.del_breakpoint([7,4])
# summer_large.del_breakpoint([1,1], [6,1])
# winter_small.del_breakpoint([4,3])
# winter_large.del_breakpoint([6,1])

fig, ax1 = plt.subplots()
fig.subplots_adjust(right=0.8)
ax2, ax3 = (ax1.twinx() for _ in range(2))
ax3.spines["right"].set_position(("axes", 1.1))
ax3.spines["right"].set_visible(True)
# ax2 = ax1.twinx()
ax1.plot(summer_small.temp, summer_small.hr[0], "k", label="Heart rate")
ax2.plot(summer_small.temp, summer_small.d1[0], "b", label="First derivative of heart rate")
ax3.plot(summer_small.temp, summer_small.d2[0], "r", label="Second derivative of heart rate")
ax1.set_ylabel("Heart rate (beats per min")
ax2.set_ylabel("First derivative of heart rate")
ax3.set_ylabel("Second derivative of heart rate")
ax1.set_xlabel("Body temperature (°C)")
fig.legend()
plt.show()

fig, ax1 = plt.subplots()
fig.subplots_adjust(right=0.8)
ax2, ax3 = (ax1.twinx() for _ in range(2))
ax3.spines["right"].set_position(("axes", 1.1))
ax3.spines["right"].set_visible(True)
# ax2 = ax1.twinx()
ax1.plot(summer_small.temp, summer_small.hr[0], "k", label="Heart rate")
ax2.plot(summer_small.temp, summer_small.angle[0], "b", label="Angle of slope")
ax3.plot(summer_small.temp, summer_small.d_rate[0], "r", label="Rate of angle change")
ax1.set_ylabel("Heart rate (beats per min")
ax2.set_ylabel("Angle of slope (°)")
ax3.set_ylabel("Rate of angle change (° per 0.1 °C)")
ax1.set_xlabel("Body temperature (°C)")
fig.legend()
plt.show()

################################
##### save model rss & aic #####
################################

obj_list = [summer_small, summer_large, winter_small, winter_large]

for obj in obj_list:
    if obj == obj_list[0]:
        dummy = obj.rss
    else:
        dummy = dummy.append(obj.rss)
os.chdir("D:\\Mirror\\mphil\\Manuscripts\\Lab heart rate\\Figures")
dummy.to_csv("rss.csv")

for obj in obj_list:
    if obj == obj_list[0]:
        dummy = obj.flt_aic
    else:
        dummy = dummy.append(obj.flt_aicc)
os.chdir("D:\\Mirror\\mphil\\Manuscripts\\Lab heart rate\\Figures")
dummy.to_csv("flt_aicc.csv")
##########################################################
##### Bootstrap confidence interval (time-consuming) #####
##########################################################

# obj_list = [summer_s, summer_l, winter_s, winter_l]
#
# for obj in obj_list:
#     obj.bootstrap_sd(50000, balanced=True)

#################################
##### save TPCgroup objects #####
#################################

# os.chdir("D:\\Mirror\\mphil\\Data\\Heart rate\\lab")
# obj_list = [summer_small, summer_large, winter_small, winter_large]
# obj_list_name = ["summer_small", "summer_large", "winter_small", "winter_large"]
# for name, obj in zip(obj_list_name, obj_list):
#     pickle_name = name + ".pkl"
#     with open(pickle_name, "wb") as pickle_file:
#         pickle.dump(obj, pickle_file)

######################
##### export raw #####
######################

start_index = np.max([np.where(pd.Series(hr).notnull())[0][0] for obj in obj_list for hr in obj.hr])

season, size, replicate, temp, hr = ([] for _ in range(5))
for obj in obj_list:
    for ind in range(obj.size):
        season.extend([obj.season] * len(obj.temp[start_index:]))
        size.extend([obj.size_class] * len(obj.temp[start_index:]))
        replicate.extend([str(ind)] * len(obj.temp[start_index:]))
        temp.extend(obj.temp[start_index:].tolist())
        hr.extend(obj.hr[ind][start_index:].tolist())
df_hr = pd.DataFrame({"Season": season, "Size": size, "Ind": replicate, "Temp": temp, "HR": hr})
os.chdir("D:\\Mirror\\mphil\\Manuscripts\\Lab heart rate\\Figures")
df_hr.to_csv("df_hr.csv")