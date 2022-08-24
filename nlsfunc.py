import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats, optimize
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


def bet(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
        *args, **kwargs):
    para_names = ["Tmin", "Tmax", "Tm", "Hmax"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return out["Hmax"] * ((out["Tmax"] - Temp) / (out["Tmax"] - out["Tm"])) * \
           ((Temp - out["Tmin"]) / (out["Tm"] - out["Tmin"])) ** \
           ((out["Tm"] - out["Tmin"]) / (out["Tmax"] - out["Tm"]))  # calculate and return


def betbet(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
           *args, **kwargs):
    para_names = ["Tmin", "Tmax", "Tm", "Hmax", "Tmin2", "Tmax2", "Tm2", "Hmax2", "a", "bp"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return (bet(Temp, [out[e] for e in ["Tmin", "Tmax", "Tm", "Hmax"]]) ** logi(Temp, out["a"], out["bp"])) * \
           (bet(Temp, [out[e] for e in ["Tmin2", "Tmax2", "Tm2", "Hmax2"]]) ** logi(Temp, -out["a"], out["bp"]))  # calculate and return


def gaubet(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
           *args, **kwargs):
    para_names = ["k", "u", "sig", "Tmin", "Tmin", "Tmax", "Tm", "Hmax", "a", "bp"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return (out["k"] * stats.norm.pdf(Temp, loc=out["u"], scale=out["sig"]) ** logi(Temp, out["a"], out["bp"])) * \
           (bet(Temp, [out[e] for e in ["Tmin", "Tmax", "Tm", "Hmax"]]) ** logi(Temp, -out["a"], out["bp"]))  # calculate and return


def gaugau(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
           *args, **kwargs):
    para_names = ["k", "u", "sig", "k2", "u2", "sig2", "a", "bp"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return (out["k"] * stats.norm.pdf(Temp, loc=out["u"], scale=out["sig"]) ** logi(Temp, out["a"], out["bp"])) * \
           (out["k2"] * stats.norm.pdf(Temp, loc=out["u2"], scale=out["sig2"]) ** logi(Temp, -out["a"], out["bp"]))  # calculate and return


def qua(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
        *args, **kwargs):
    para_names = ["k", "qa", "qb", "qc"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return out["k"] * (out["qa"] * Temp ** 2 + out["qb"] * Temp + out["qc"])  # calculate and return


def quaqua(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
           *args, **kwargs):
    para_names = ["k", "qa", "qb", "qc", "k2", "qa2", "qb2", "qc2", "a", "bp"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return (qua(Temp, [out[e] for e in ["k", "qa", "qb", "qc"]]) ** logi(Temp, out["a"], out["bp"])) * \
           (qua(Temp, [out[e] for e in ["k2", "qa2", "qb2", "qc2"]]) ** logi(Temp, out["a"], out["bp"]))  # calculate and return


def quagau(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
           *args, **kwargs):
    para_names = ["k", "qa", "qb", "qc", "k", "u", "sig", "a", "bp"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return (qua(Temp, [out[e] for e in ["k", "qa", "qb", "qc"]]) ** logi(Temp, out["a"], out["bp"])) * \
           (out["k"] * stats.norm.pdf(Temp, loc=out["u"], scale=out["sig"])  ** logi(Temp, out["a"], out["bp"]))  # calculate and return


def quabet(Temp: Union[int, float, np.ndarray, list, tuple],  # constrain data type
           *args, **kwargs):
    para_names = ["k", "qa", "qb", "qc", "Tmin", "Tmax", "Tm", "Hmax", "a", "bp"]  # names of parameters to be used
    out = arg_processing(para_names, args, kwargs)  # format parameters
    return (qua(Temp, [out[e] for e in ["k", "qa", "qb", "qc"]]) ** logi(Temp, out["a"], out["bp"])) * \
           (bet(Temp, [out[e] for e in ["Tmin", "Tmax", "Tm", "Hmax"]]) ** logi(Temp, out["a"], out["bp"]))  # calculate and return


def bimodal_fitting(dfs, function, *args):
    count = 0
    results = []
    while True:
        df = dfs[count]
        df_name = dfs_name[count]
        xdata = df[:, 0]
        ydata = df[:, 1]

        k1 = np.random.exponential(1, perm)
        min1 = np.random.uniform(-20, 20, perm)
        max1 = np.random.uniform(43, 60, perm)
        m1 = np.random.exponential(1, perm)
        n1 = np.random.exponential(1, perm)
        k2 = np.random.exponential(1, perm)
        min2 = np.random.uniform(-20, 20, perm)
        max2 = np.random.uniform(43, 60, perm)
        m2 = np.random.exponential(1, perm)
        n2 = np.random.exponential(1, perm)
        a1 = np.random.exponential(1, perm)
        bp1 = np.random.uniform(30, 50, perm)

        param = np.array([k1, min1, max1, m1, n1, k2, min2, max2, m2, n2, a1, bp1]).transpose()

        rmse_final = np.full(no_of_groups, np.inf)
        param_final = np.full((no_of_groups, param.shape[1]), np.inf)

        print(f"Fitting curve for individual {df_name} ...")
        for group in tqdm(range(no_of_groups)):
            rmse_group = np.full(ind_per_group, np.inf)
            for ind in range(ind_per_group):
                yrandom = bibet(xdata, param[group * ind_per_group + ind])
                rmse_group[ind] = ((yrandom - ydata) ** 2).mean()
            try:
                best_group = np.where(rmse_group == min(rmse_group))[0][0]
            except(IndexError):
                best_group = 0
            try:
                param_final[group], _ = curve(bibet, xdata, ydata, p0=param[best_group])
                ybest_group = bibet(xdata, param_final[group])
                rmse_final[group] = ((ybest_group - ydata) ** 2).mean()
            except(RuntimeError):
                param_final[group] = np.zeros(param.shape[1])
                rmse_final[group] = np.inf
        if min(rmse_final) > error_threshold ** 2:
            print(f"RMSE larger than {error_threshold ** 2}... Re-run...")
            continue
        try:
            best_final = np.where(rmse_final == min(rmse_final))[0][0]
            ybest_final = bibet(xdata, param_final[best_final])
            results.append([df_name] + list(param_final[best_final]))
            count += 1
        except(IndexError):
            print("Failed to converge... Re-run...")
            continue
        if count == len(dfs):
            print("All curve-fittings are completed!")
            break


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def significance_bar(axis, start, end, height, displaystring, linewidth=1.2, markersize=8, boxpad=0.3, color='k'):
    # draw a line with downticks at the ends
    axis.plot([start, end], [height] * 2, '-', color=color, lw=linewidth, marker=TICKDOWN, markeredgewidth=linewidth,
              markersize=markersize)
    # draw the text with a bounding box covering up the line
    axis.text(0.5 * (start + end), height, displaystring, ha='center', va='center',
              bbox=dict(facecolor='1.', edgecolor='none', boxstyle='Square,pad=' + str(boxpad)))


def Welch_F(groups, alpha: float = 0.05):
    # symbols used in Parra-Frutos, 2013
    k = len(groups)
    n = [len(gp) for gp in groups]
    w = [ni / np.std(gp, ddof=1) for ni, gp in zip(n, groups)]
    W = np.sum(w)
    Z_asterisk = np.sum([wi * np.mean(gp) for wi, gp in zip(w, groups)]) / W
    Q_num = np.sum([wi * (np.mean(gp) - Z_asterisk) ** 2 for wi, gp in zip(w, groups)]) / (k - 1)
    Q_denom = 1 + np.sum([(1 - wi / W) ** 2 / (ni - 1) for wi, ni in zip(w, n)]) * 2 * (k - 2) / (k ** 2 - 1)
    Q = Q_num / Q_denom

    df_num = k - 1
    df_denom = (k ** 2 - 1) / 3 / np.sum([(1 - wi / W) ** 2 / (ni - 1) for wi, ni in zip(w, n)])

    Q_crit = stats.f.ppf(1 - alpha, df_num, df_denom)
    p_value = stats.f.sf(Q, df_num, df_denom)

    return {"df": (df_num, df_denom), "Welch's F": Q, "critical Welch's F": Q_crit, "p value": p_value}
