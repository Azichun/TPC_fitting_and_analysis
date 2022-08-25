import os, winsound
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats, optimize
from scipy.optimize import curve_fit
import warnings
from typing import Union
from pandas.core.common import SettingWithCopyWarning
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

warnings.simplefilter('always', UserWarning)


def arg_processing(para_names, args, kwargs):
    if len(kwargs) > 0:  # if keyword arguments were inputted
        if len(args) > 0:  # if both unnamed and keyword arguments were inputted
            warnings.warn(
                "Only keyword arguments were used and unnamed arguments were ignored"
            )  # tell the user that only the keyword arguments were used
    elif len(args) > 0:  # if only unnamed arguments were inputted
        if len(args) == 1:  # if only 1 unnamed argument was inputted
            if isinstance(args[0], dict):
                kwargs = args[0]
            elif isinstance(args[0], (np.ndarray, list, tuple)):
                kwargs = {p: a for p, a in zip(para_names, args[0], strict=True)}
            else:
                raise TypeError("Incorrect type of argument inputted")  # raise error
        elif len(args) == len(para_names):  # if unnamed arguments were separately inputted
            kwargs = {p: a for p, a in zip(para_names, args, strict=True)}
    else:  # if no parameters were inputted
        raise ValueError("No parameters were inputted")  # raise error
    if set(kwargs.keys()) != set(para_names):  # if the keys are not equals to parameter names
        raise ValueError("Incorrect number of parameters/"
                         "incorrectly named parameters were inputted")  # raise error
    return kwargs


def line(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
         *args, **kwargs):
    para_names = ["m", "c"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return out["m"] * Temp + out["c"]  # calculate and return


def loga(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
         *args, **kwargs):
    para_names = ["k", "b", "c"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return out["k"] * np.log(-Temp + out["b"]) + out["c"]  # calculate and return


def expo(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
         *args, **kwargs):
    para_names = ["k", "a", "b", "c"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return out["k"] * (out["a"] ** (Temp - out["b"])) + out["c"]  # calculate and return


def logi(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
         *args, **kwargs):
    para_names = ["a", "bp"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return 1 / (1 + np.exp(-out["a"] * (out["bp"] - Temp)))  # calculate and return


####################
##### Unimodal #####
####################
def bet(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
        *args, **kwargs):
    para_names = ["Tmin", "Tmax", "Tm", "Hmax"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return out["Hmax"] * ((out["Tmax"] - Temp) / (out["Tmax"] - out["Tm"])) * \
           ((Temp - out["Tmin"]) / (out["Tm"] - out["Tmin"])) ** \
           ((out["Tm"] - out["Tmin"]) / (out["Tmax"] - out["Tm"]))  # calculate and return


def gau(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
        *args, **kwargs):
    para_names = ["k", "u", "sig"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return out["k"] * stats.norm.pdf(Temp, loc=out["u"], scale=out["sig"])  # calculate and return


def qua(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
        *args, **kwargs):
    para_names = ["qa", "qb", "qc"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return out["qa"] * Temp ** 2 + out["qb"] * Temp + out["qc"]  # calculate and return


###################
##### Bimodal #####
###################
def betbet(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
           *args, **kwargs):
    para_names = ["Tmin", "Tmax", "Tm", "Hmax", "Tmin2", "Tmax2", "Tm2", "Hmax2", "a",
                  "bp"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return (bet(Temp, [out[e] for e in ["Tmin", "Tmax", "Tm", "Hmax"]]) ** logi(Temp, out["a"], out["bp"])) * \
           (bet(Temp, [out[e] for e in ["Tmin2", "Tmax2", "Tm2", "Hmax2"]]) ** logi(Temp, -out["a"],
                                                                                    out["bp"]))  # calculate and return


def gaugau(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
           *args, **kwargs):
    para_names = ["k", "u", "sig", "k2", "u2", "sig2", "a", "bp"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return (gau(Temp, [out[e] for e in ["k", "u", "sig"]]) ** logi(Temp, out["a"], out["bp"])) * \
           (gau(Temp, [out[e] for e in ["k2", "u2", "sig2"]]) ** logi(Temp, -out["a"],
                                                                      out["bp"]))  # calculate and return


def quaqua(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
           *args, **kwargs):
    para_names = ["qa", "qb", "qc", "qa2", "qb2", "qc2", "a", "bp"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return (qua(Temp, [out[e] for e in ["qa", "qb", "qc"]]) ** logi(Temp, out["a"], out["bp"])) * \
           (qua(Temp, [out[e] for e in ["qa2", "qb2", "qc2"]]) ** logi(Temp, -out["a"],
                                                                       out["bp"]))  # calculate and return


def gaubet(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
           *args, **kwargs):
    para_names = ["k", "u", "sig", "Tmin", "Tmax", "Tm", "Hmax", "a", "bp"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return (gau(Temp, [out[e] for e in ["k", "u", "sig"]]) ** logi(Temp, out["a"], out["bp"])) * \
           (bet(Temp, [out[e] for e in ["Tmin", "Tmax", "Tm", "Hmax"]]) ** logi(Temp, -out["a"],
                                                                                out["bp"]))  # calculate and return


def quabet(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
           *args, **kwargs):
    para_names = ["qa", "qb", "qc", "Tmin", "Tmax", "Tm", "Hmax", "a", "bp"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return (qua(Temp, [out[e] for e in ["qa", "qb", "qc"]]) ** logi(Temp, out["a"], out["bp"])) * \
           (bet(Temp, [out[e] for e in ["Tmin", "Tmax", "Tm", "Hmax"]]) ** logi(Temp, -out["a"],
                                                                                out["bp"]))  # calculate and return


def quagau(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
           *args, **kwargs):
    para_names = ["qa", "qb", "qc", "k", "u", "sig", "a", "bp"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return (qua(Temp, [out[e] for e in ["qa", "qb", "qc"]]) ** logi(Temp, out["a"], out["bp"])) * \
           (gau(Temp, [out[e] for e in ["k", "u", "sig"]]) ** logi(Temp, -out["a"], out["bp"]))  # calculate and return


def randomize(para_name: list, perm: int):
    out = []
    for p in para_name:
        if p in ["Tmin", "Tmin2"]:
            out.append(np.random.uniform(-200, 100, perm))  # Beta: CTmin
        elif p in ["Tmax", "Tmax2"]:
            out.append(np.random.uniform(-100, 200, perm))  # Beta: CTmax
        elif p in ["Tm", "Tm2"]:
            out.append(np.random.uniform(0, 60, perm))  # Beta: Temperature when heart rate is maximized
        elif p in ["Hmax", "Hmax2"]:
            out.append(np.random.uniform(0, 300, perm))  # Beta: Maximum heart rate
        elif p in ["k", "k2"]:
            out.append(np.random.uniform(0, 10000, perm))  # Gaussian: Scale coefficient
        elif p in ["u", "u2"]:
            out.append(np.random.uniform(0, 60, perm))  # Gaussian: Mean
        elif p in ["sig", "sig2"]:
            out.append(np.random.exponential(1, perm))  # Gaussian: SD
        elif p in ["qa", "qa2"]:
            out.append(np.random.uniform(-10, 0, perm))  # Quadratic: x2
        elif p in ["qb", "qb2"]:
            out.append(np.random.uniform(0, 60, perm))  # Quadratic: x1
        elif p in ["qc", "qc2"]:
            out.append(np.random.uniform(-200, 200, perm))  # Quadratic: x0
        elif p == "a":
            out.append(np.random.exponential(1, perm))  # Bimodal: rate of change
        elif p == "bp":
            out.append(np.random.uniform(0, 60, perm))  # Bimodal: temperature when dominance shift
    return np.array(out).transpose()  # organize starting parameters


def TPC_fit(dfs: dict, func: Union[str, list], no_of_groups: int = 100, ind_per_group: int = 200, patience: int = 10,
            audio: bool = True):
    perm = no_of_groups * ind_per_group  # calculate how many sets of random starting parameters need to be generated

    para_names = {
        "bet": ["Tmin", "Tmax", "Tm", "Hmax"], "gau": ["k", "u", "sig"], "qua": ["qa", "qb", "qc"],
        "betbet": ["Tmin", "Tmax", "Tm", "Hmax", "Tmin2", "Tmax2", "Tm2", "Hmax2", "a", "bp"],
        "gaugau": ["k", "u", "sig", "k2", "u2", "sig2", "a", "bp"],
        "quaqua": ["qa", "qb", "qc", "qa2", "qb2", "qc2", "a", "bp"],
        "gaubet": ["k", "u", "sig", "Tmin", "Tmax", "Tm", "Hmax", "a", "bp"],
        "quabet": ["qa", "qb", "qc", "Tmin", "Tmax", "Tm", "Hmax", "a", "bp"],
        "quagau": ["qa", "qb", "qc", "k", "u", "sig", "a", "bp"]
    }  # dictionary for storing the parameters required for each function

    if isinstance(func, str):
        func = [func]  # convert argument func from str type to list type

    out = {}  # initialize list for storage
    for f_name in func:  # for each function
        f = globals()[f_name]  # get function
        count = 0  # set counter
        results = []  # initialize list for storage
        for df_name, df in dfs.items():  # for each individual
            count += 1  # counter increment
            print(f"Fitting {f_name} curve for individual {df_name} (#{count}) ...")  # notify user which individual is being processed
            patience_count = 0  # set patience counter
            while True:  # loop until at least one model converged
                xdata = df[:, 0]  # extract temperature column
                ydata = df[:, 1]  # extract heart rate column

                param = randomize(para_names[f_name], perm)  # generate random sets of starting parameters

                rmse_final = np.full(no_of_groups, np.inf)  # initialize rmse list
                param_final = np.full((no_of_groups, param.shape[1]), np.inf)  # initialize finalized parameter list

                for group in tqdm(range(no_of_groups)):  # for each batch for processing (i.e. group)
                    rmse_group = np.full(ind_per_group, np.inf)  # initialize rmse list for group
                    for ind in range(ind_per_group):  # for each model
                        yrandom = f(xdata, param[group * ind_per_group + ind])  # fit curve with starting parameters
                        rmse_group[ind] = ((yrandom - ydata) ** 2).mean()  # calculate rmse
                    try:
                        best_group = np.where(rmse_group == min(rmse_group))[0][0]  # choose the best set of parameters
                    except(IndexError):  # if no good set of parameters by chance
                        best_group = 0  # choose the first group
                    try:
                        param_final[group], _ = curve_fit(f, xdata, ydata, p0=param[
                            best_group])  # nls regression with the best set of parameters
                        ybest_group = f(xdata, param_final[group])  # get fitted values
                        rmse_final[group] = ((ybest_group - ydata) ** 2).mean()  # calculate rmse
                    except(RuntimeError):  # if none converges
                        param_final[group] = np.zeros(param.shape[1])  # store null value
                        rmse_final[group] = np.inf  # store null value
                try:
                    best_final = np.where(rmse_final == min(rmse_final))[0][0]  # choose the best set of parameters among different groups
                    results.append([df_name] + list(param_final[best_final]))  # add finalized parameters to list
                    break  # if a model converged, do not re-run
                except(IndexError):  # if none converges
                    patience_count += 1  # pateince counter increment
                    if patience_count >= patience:  # if reach patience limit
                        results.append([df_name] + [np.nan] * len(para_names[f_name]))  # add finalized parameters to list
                        print("Ran out of patience...Proceed to next individual...")  # notify user
                        break  # give up
                    print(f"Failed to converge... Re-run (patience: "
                          f"{(patience - patience_count) / patience * 100}%)...")  # notify user
                    continue  # re-run
        results_pd = pd.DataFrame(results)  # list to data frame
        results_pd.columns = ["Individual"] + para_names[f_name]  # rename
        out[f_name] = results_pd

    print("All curve-fittings are completed!")  # notify the user curve-fitting is completed
    if audio:
        for _ in range(10):
            winsound.Beep(5000, 100)  # notify the user in audio

    return out
