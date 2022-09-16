import numpy as np
import pandas as pd
import matplotlib, os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats, optimize, interpolate
from nlsfunc import *
import copy, csaps, types, warnings
from pandas.core.common import SettingWithCopyWarning

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.sans-serif"] = "Arial"
matplotlib.rcParams["font.family"] = "sans-serif"

class TPCgroup:
    def __init__(self, temp, param, func, verbose=True, **kwargs):
        warnings.simplefilter("ignore", SettingWithCopyWarning)
        param.sort_values("Individual", ascending=True, inplace=True)
        self.temp = temp
        self.x_len = len(self.temp)
        self.x_resolution = (self.temp[-1] - self.temp[0]) / self.x_len
        self.funcs = [func]
        self.func_names = [func.__name__]
        self.ind = param.reset_index()["Individual"].tolist()
        self.size = len(self.ind)
        setattr(self, f"{func.__name__}_param", param.reset_index(drop=True).drop("Individual", axis=1))
        setattr(self, f"{func.__name__}_hr", np.array([func(self.temp,
                                                            getattr(self, f"{func.__name__}_param").loc[ind].tolist())
                                                       for ind in range(self.size)]))
        attr_final = ["temp", "x", "x_len", "x_resolution", "funcs", "func_names", "ind", "size",
                      f"{func.__name__}_param", f"{func.__name__}_hr"]
        for extra_name, extra in kwargs.items():
            if any([extra_name == attr_name for attr_name in dir(self)]):
                raise ValueError(f'Attribute name "{extra_name}" crashes with TPCgroup default attributes.')
            else:
                setattr(self, extra_name, extra)
                attr_final.append(extra_name)
        if verbose: print(f"Attributes {', '.join(attr_final)} are created.")

    def import_fit(self, params, funcs, verbose=True):
        if not isinstance(params, list):
            params = [params]
            funcs = [funcs]
        attr_names = []
        for param, func in zip(params, funcs):
            param.sort_values("Individual", ascending=True, inplace=True)
            if param.reset_index()["Individual"].tolist() != self.ind:
                raise ValueError("Individual names cannot be matched!")
            self.funcs.append(func)
            self.func_names.append(func.__name__)
            setattr(self, f"{func.__name__}_param", param.reset_index(drop=True).drop("Individual", axis=1))
            setattr(self, f"{func.__name__}_hr", np.array([func(self.temp,
                                                                getattr(self, f"{func.__name__}_param").loc[
                                                                    ind].tolist()) for ind in range(self.size)]))
            attr_names.append(f"{func.__name__}_param")
            attr_names.append(f"{func.__name__}_hr")
        if verbose: print(f"Attributes {', '.join(attr_names)} are created.")

    def import_raw(self, raw, verbose=True):
        if isinstance(raw, (list, np.ndarray)):
            self.raw = raw
        elif isinstance(raw, dict):
            self.raw = [raw[ind] for ind in self.ind]
        self.n = np.array([self.raw[ind].shape[0] for ind in range(self.size)])
        if verbose: print(f"Attributes raw and n are created.")

    def fit_spline(self, smooth=0.5, verbose=True):
        if not hasattr(self, "raw"):
            raise ValueError("Please import raw data!")
        self.spline, spline_hr = [], []
        for ind in range(self.size):
            raw_mean = copy.deepcopy(self.raw[ind])
            temp, count = np.unique(raw_mean[:, 0], return_counts=True)
            dups = temp[count > 1]
            for dup in dups:
                dup_index = np.where(raw_mean[:, 0] == dup)
                raw_mean[dup_index[0][0]] = raw_mean[dup_index].mean(axis=0)
                raw_mean = np.delete(raw_mean, dup_index[0][1:], axis=0)
            raw_mean = raw_mean[raw_mean[:, 0].argsort()]
            self.spline.append(csaps.csaps(raw_mean[:, 0], raw_mean[:, 1], smooth=smooth))
            spline_hr.append(self.spline[ind](self.temp))
        self.spline_hr = np.array(spline_hr)
        if verbose: print(f"Attributes spline and spline_hr are created.")

    def calc_accuracy(self, bp_weighed=False, weight=(1, 0.1), verbose=True):
        weightings = [""]
        col_names = ["Individual"] + self.func_names
        rss_final, aic_final, aicc_final = ([self.ind] for _ in range(3))
        if bp_weighed:
            rss_final_bp, aic_final_bp, aicc_final_bp = ([self.ind] for _ in range(3))
            weightings.append("_bp")
            dummy_self = copy.deepcopy(self)
        for weighting in weightings:
            for func, func_name in zip(self.funcs, self.func_names):
                rss = []
                if bp_weighed:
                    rss_bp = []
                    dummy_self.confirm_fit(func_name, verbose=False)
                    dummy_self.calc_curvature(verbose=False)
                    concavity = dummy_self.breakpoint_concavity
                    bp = dummy_self.breakpoint
                for ind in range(self.size):
                    fitted = func(self.raw[ind][:, 0], getattr(self, f"{func_name}_param").loc[ind].tolist())
                    resid = np.array(fitted - self.raw[ind][:, 1])
                    rss.append((resid ** 2).sum())
                    if bp_weighed:
                        fbp = bp[ind][np.where(-1 == np.array(concavity[ind]))[0][-1]]
                        resid_bp = resid * np.array([weight[0] if temp <= fbp else weight[1]
                                             for temp in self.raw[ind][:, 0]])
                        rss_bp.append((resid_bp ** 2).sum())
                k = getattr(self, f"{func_name}_param").shape[1]
                rss = np.array(locals()[f"rss{weighting}"])
                aic = 2 * k - 2 * 0.5 * (-self.n * (np.log(2 * np.pi) + 1 + np.log(self.n) + np.log(rss)))
                locals()[f"rss_final{weighting}"].append(rss)
                locals()[f"aic_final{weighting}"].append(aic)
                locals()[f"aicc_final{weighting}"].append(aic + ((2 * k ** 2) + 2 * k) / (self.n - k - 1))
        for metric in ["rss", "aic", "aicc"]:
            for weighting in weightings:
                setattr(self, f"{metric}{weighting}", pd.DataFrame(locals()[f"{metric}_final{weighting}"]).transpose())
                getattr(self, f"{metric}{weighting}").columns = col_names
                for ind in range(self.size):
                    row = getattr(self, f"{metric}{weighting}").loc[ind][1:]
                    winner = np.where(row == row.min())[0][0] + 1
                    getattr(self, f"{metric}{weighting}").loc[ind, "best performing"] =\
                        getattr(self, f"{metric}{weighting}").columns[winner]
                dummy = getattr(self, f"{metric}{weighting}").drop(["Individual", "best performing"], axis=1)
                min = dummy.min(axis=1)
                d_result = pd.concat([getattr(self, f"{metric}{weighting}")["Individual"],
                                      dummy.subtract(list(min), axis=0),
                                      getattr(self, f"{metric}{weighting}")["best performing"]], axis=1)
                setattr(self, f"d_{metric}{weighting}", d_result)
        if verbose: print(f"Attributes rss, aic, aicc, d_rss, d_aic, d_aicc are created.")

    def trunc_hr(self, verbose=True):
        for ind in range(self.size):
            min_temp = self.raw[ind][:, 0][0]
            max_temp = self.raw[ind][:, 0][-1]
            min_trunc = np.where(self.temp < min_temp)[0]
            max_trunc = np.where(self.temp > max_temp)[0]
            trunc = np.concatenate([min_trunc, max_trunc])
            self.hr[ind] = np.array([hr if index not in trunc else np.nan for index, hr in enumerate(self.hr[ind])])
        if verbose: print(f"Attribute hr is truncated.")

    def confirm_fit(self, fit, truncate=True, verbose=True):
        if any(fit == metric for metric in ["rss", "aic", "aicc"]):
            fit = list(getattr(self, fit)["best performing"])
        if not isinstance(fit, (list, tuple)):
            fit = [fit]
        if isinstance(fit, (list, tuple)) and len(fit) == 1:
            fit = fit * self.size
        hr_result = []
        fit_result = []
        for ind, func_name in enumerate(fit):
            if isinstance(func_name, types.FunctionType):
                func_name = func_name.__name__
            hr_result.append(getattr(self, f"{func_name}_hr")[ind])
            fit_result.append(func_name)
        self.hr = np.array(hr_result)
        self.fit = fit_result
        if truncate:
            self.trunc_hr(verbose)
        self.mean = np.mean(self.hr, axis=0)
        self.sd = np.std(self.hr, ddof=1, axis=0)
        self.maxhr = np.nanmax(self.hr, axis=1)
        if verbose: print(f"Attributes fit, hr, mean, sd, maxhr are created.")

    def calc_d1(self, verbose=True):
        yd1 = []
        for ind in range(self.size):
            yd1.append([np.nan] + [(self.hr[ind, i + 1] - self.hr[ind, i - 1]) / (2 * self.x_resolution)
                                   for i in range(1, len(self.hr[ind]) - 1)] + [np.nan])
        self.d1 = np.array(yd1)
        if verbose: print(f"Attribute d1 is created.")

    def calc_d2(self, verbose=True):
        if not hasattr(self, "d1"):
            self.calc_d1()
        yd2 = []
        for ind in range(self.d1.shape[0]):
            yd2.append([np.nan] * 2 + [(self.d1[ind, i + 1] - self.d1[ind, i - 1]) / (2 * self.x_resolution)
                                       for i in range(2, len(self.d1[ind]) - 2)] + [np.nan] * 2)
        self.d2 = np.array(yd2)
        if verbose: print(f"Attribute d2 is created.")

    def bootstrap_mean(self, boot_sample_size=1000, alpha=0.05, balanced=False, verbose=True):
        boot_mean = []
        for temp in tqdm(range(len(self.temp))):
            boot_sample = np.array([np.random.choice(self.hr[:, temp], size=self.size, replace=True)
                                    for rep in range(boot_sample_size)])
            boot_mean.append(np.mean(boot_sample, axis=1))
        if balanced:
            correction = np.array([mean_x - np.mean(boot_mean_x) for mean_x, boot_mean_x in zip(self.mean, boot_mean)])
        else:
            correction = np.zeros(self.mean.shape)
        self.mean_loci = np.array([np.percentile(boot_mean_x, alpha / 2 * 100) for boot_mean_x in boot_mean]) \
                         + correction
        self.mean_hici = np.array([np.percentile(boot_mean_x, 100 - alpha / 2 * 100) for boot_mean_x in boot_mean]) \
                         + correction
        if verbose: print(f"Attributes mean_loci and mean_hici are created.")

    def bootstrap_sd(self, boot_sample_size=1000, alpha=0.05, balanced=False, verbose=True):
        boot_sd = []
        for temp in tqdm(range(len(self.temp))):
            boot_sample = np.array([np.random.choice(self.hr[:, temp], size=self.size, replace=True)
                                    for rep in range(boot_sample_size)])
            boot_sd.append(np.std(boot_sample, ddof=1, axis=1))
        if balanced:
            correction = np.array([sd_x - np.mean(boot_sd_x) for sd_x, boot_sd_x in zip(self.sd, boot_sd)])
        else:
            correction = np.zeros(self.sd.shape)
        self.sd_loci = np.array([np.percentile(boot_sd_x, alpha / 2 * 100) for boot_sd_x in boot_sd]) \
                       + correction
        self.sd_hici = np.array([np.percentile(boot_sd_x, 100 - alpha / 2 * 100) for boot_sd_x in boot_sd]) \
                       + correction
        if verbose: print(f"Attributes sd_loci and sd_hici are created.")

    def calc_curvature(self, standardize=True, verbose=True):
        kappa, kappa_x, breakpoint, concavity = ([] for _ in range(4))
        for ind in range(self.size):
            x_result = np.array([x for x in self.temp if min(self.raw[ind][:, 0]) <= x <= max(self.raw[ind][:, 0])])
            y_result = globals()[self.fit[ind]](x_result, np.array(getattr(self, f"{self.fit[ind]}_param").loc[ind].tolist()))
            if standardize:
                x = (x_result - x_result[0]) / (x_result[-1] - x_result[0])
                y = y_result / y_result.max()
                d1 = np.array([np.nan] + [(y[i + 1] - y[i - 1]) / (2 * x[1])
                                          for i in range(1, len(y) - 1)] + [np.nan])
                d2 = np.array([np.nan] + [(d1[i + 1] - d1[i - 1]) / (2 * x[1])
                                          for i in range(1, len(y) - 1)] + [np.nan])
            else:
                x = x_result
                y = y_result
                d1 = np.array([np.nan] + [(y[i + 1] - y[i - 1]) / (2 * self.x_resolution)
                                       for i in range(1, len(y) - 1)] + [np.nan])
                d2 = np.array([np.nan] + [(d1[i + 1] - d1[i - 1]) / (2 * self.x_resolution)
                                       for i in range(1, len(y) - 1)] + [np.nan])
            kappa_ind = d2 / (1 + (d1 ** 2)) ** (3 / 2)
            kappa_ind_d1 = np.array([np.nan] + [(kappa_ind[i + 1] - kappa_ind[i - 1]) / (2 * self.x_resolution)
                                                for i in range(1, len(kappa_ind) - 1)] + [np.nan])
            bp_index_dummy = [temp if kappa_ind_d1[temp] * kappa_ind_d1[temp + 1] <= 0 and
                         abs(kappa_ind_d1[temp]) <= abs(kappa_ind_d1[temp + 1])
                         else temp + 1 if kappa_ind_d1[temp] * kappa_ind_d1[temp + 1] <= 0 and
                         abs(kappa_ind_d1[temp]) > abs(kappa_ind_d1[temp + 1]) else None
                         for temp in range(len(kappa_ind_d1) - 1)]
            bp_index = np.array([temp for temp in bp_index_dummy if temp is not None])
            breakpoint.append(np.array([x_result[index] for index in bp_index]))
            concavity.append(np.array([1 if kappa_ind[index] > 0 else -1 for index in bp_index]))
            kappa.append(kappa_ind)
            kappa_x.append(x_result)
        self.kappa_x = kappa_x
        self.kappa = np.array(kappa)
        self.breakpoint = breakpoint
        self.breakpoint_concavity = concavity
        if verbose: print(f"Attributes kappa_x, kappa, breakpoint, breakpoint_concavity are created.")

    def del_breakpoint(self, verbose=True, *args):
        if not all([isinstance(arg, (list, np.array, tuple)) for arg in args]):
            raise TypeError("Please input arguments in the format of list/tuple/ndarray!")
        elif not all([len(arg) == 2 for arg in args]):
            raise ValueError("Each argument input should be an array of two elements (individual, breakpoint)!")
        unwanted_bps = np.array(args)
        unwanted_bps[:, 0] = unwanted_bps[:, 0]
        unwanted_bps[unwanted_bps[:, 1].argsort()]
        for ind, unwanted_bp in unwanted_bps:
            self.breakpoint[ind] = np.delete(self.breakpoint[ind], unwanted_bp - 1)
        if verbose: print("Selected breakpoints are deleted.")

    def filter_breakpoint(self, start_concave_down=True, choose_curviest=True, verbose=True):
        for ind in range(self.size):
            if start_concave_down:
                while self.breakpoint_concavity[ind][0] != -1:
                    self.breakpoint[ind] = self.breakpoint[ind][1:]
                    self.breakpoint_concavity[ind] = self.breakpoint_concavity[ind][1:]
            if choose_curviest:
                self.breakpoint[ind] = list(self.breakpoint[ind])
                self.breakpoint_concavity[ind] = list(self.breakpoint_concavity[ind])
                kappa_index = [np.where(self.kappa_x[ind] == bp)[0][0] for bp in self.breakpoint[ind]]
                kappa_bp = [self.kappa[ind][i] for i in kappa_index]
                counter, pos = [0, 0]
                while counter < len(kappa_bp):
                    counter += 1
                    if np.array(kappa_bp[pos:pos + 2]).prod() > 0:
                        if abs(kappa_bp[pos]) > abs(kappa_bp[pos + 1]):
                            del self.breakpoint[ind][pos + 1]
                            del self.breakpoint_concavity[ind][pos + 1]
                            del kappa_bp[pos + 1]
                        elif abs(kappa_bp[pos]) < abs(kappa_bp[pos + 1]):
                            del self.breakpoint[ind][pos]
                            del self.breakpoint_concavity[ind][pos]
                            del kappa_bp[pos]
                    else:
                        pos += 1
                        continue
        self.peaks = [ind.count(-1) for ind in self.breakpoint_concavity]
        if verbose: print("Breakpoints that fit the criteria are filtered.")

    def fit_flatline(self, func, random_gau_args, perm=20000, no_of_groups=100, error_threshold=np.inf,
                     min_data_points=10, verbose=True):
        if not isinstance(func, types.FunctionType):
            raise ValueError('Please provide a function for the argument "func"!')
        if not hasattr(self, "raw"):
            raise ValueError("Please import raw data!")
        if not hasattr(self, "breakpoint"):
            raise ValueError('Please calculate and pre-process breakpoints '
                             '(methods "calc_curvature" and "filter_breakpoint")!')
        if not hasattr(self, "flt_funcs"): self.flt_funcs, self.flt_func_names = [], []
        if not hasattr(self, "flt_raw"):
            index_cut, temp_cut, raw_for_flt, flt_x = ([] for _ in range(4))
            for ind in range(self.size):
                index_cut.append([index for index, conc in enumerate(self.breakpoint_concavity[ind]) if conc == -1][-1])
                temp_cut.append(self.breakpoint[ind][index_cut[ind]])
                raw_check = np.array([[temp, hr] for temp, hr in self.raw[ind] if temp > temp_cut[ind]])
                if raw_check.shape[0] < min_data_points:
                    raw_for_flt.append(self.raw[ind][-min_data_points:])
                    flt_x.append(np.array([temp for temp in self.temp if temp >= self.raw[ind][-min_data_points][0]]))
                else:
                    raw_for_flt.append(raw_check)
                    flt_x.append(np.array([temp for temp in self.temp if temp > temp_cut[ind]]))
            self.flt_raw = raw_for_flt
            self.flt_temp_cut = temp_cut
            self.flt_x = flt_x
        if func not in self.flt_funcs:
            self.flt_funcs.append(func)
            self.flt_func_names.append(func.__name__)
        ind_per_group = perm // no_of_groups
        count = 0
        hr, results = [], []
        while True:
            df = self.flt_raw[count]
            xdata = df[:, 0]
            ydata = df[:, 1]

            param = []
            for random_arg in random_gau_args:
                param.append(np.random.normal(random_arg[0], random_arg[1], perm))
            param = np.array(param).transpose()

            rmse_final = np.full(no_of_groups, np.inf)
            param_final = np.full((no_of_groups, param.shape[1]), np.inf)
            print(f"Fitting curve for individual {self.ind[count]} ...")
            for group in tqdm(range(no_of_groups)):
                rmse_group = np.full(ind_per_group, np.inf)
                for ind in range(ind_per_group):
                    yrandom = func(xdata, param[group * ind_per_group + ind])
                    rmse_group[ind] = ((yrandom - ydata) ** 2).mean()
                try:
                    best_group = np.where(rmse_group == min(rmse_group))[0][0]
                except(IndexError):
                    best_group = 0
                try:
                    param_final[group], _ = optimize.curve_fit(func, xdata, ydata, p0=param[best_group])
                    ybest_group = func(xdata, param_final[group])
                    rmse_final[group] = ((ybest_group - ydata) ** 2).mean()
                except(RuntimeError):
                    param_final[group] = np.zeros(param.shape[1])
                    rmse_final[group] = np.inf
            if min(rmse_final) > error_threshold ** 2:
                print(f"RMSE larger than {error_threshold ** 2}... Re-run...")
                continue
            try:
                best_final = np.where(rmse_final == min(rmse_final))[0][0]
                ybest_final = func(xdata, param_final[best_final])
                results.append(list(param_final[best_final]))
                hr.append(func(self.flt_x[count], param_final[best_final]))
                count += 1
            except (KeyboardInterrupt):
                break
            except(IndexError):
                print("Failed to converge... Re-run...")
                continue
            if count == self.size:
                print("All curve-fittings are completed!")
                break
        setattr(self, f"flt_{func.__name__}_param", np.array(results))
        setattr(self, f"flt_{func.__name__}_hr", hr)

    def refit_flatline(self, inds, func, random_gau_args, perm=50000, no_of_groups=1000, error_threshold=np.inf, verbose=True):
        func = globals()[func] if isinstance(func, str) else func
        inds = [inds] if not isinstance(inds, (np.ndarray, list, tuple)) else inds
        if not any([isinstance(ind, int) for ind in inds]):
            inds = [np.where(self.ind == ind)[0][0] for ind in np.array(inds)]
        if not hasattr(self, f"flt_{func.__name__}_param"):
            raise ValueError(f'You do not have any fitted curves using the {func.__name__} function.')
        ind_per_group = perm // no_of_groups
        count = 0
        while True:
            df = self.flt_raw[inds[count]]
            xdata = df[:, 0]
            ydata = df[:, 1]

            param = []
            for random_arg in random_gau_args:
                param.append(np.random.normal(random_arg[0], random_arg[1], perm))
            param = np.array(param).transpose()

            rmse_final = np.full(no_of_groups, np.inf)
            param_final = np.full((no_of_groups, param.shape[1]), np.inf)
            print(f"Fitting curve for individual {self.ind[inds[count]]} ...")
            for group in tqdm(range(no_of_groups)):
                rmse_group = np.full(ind_per_group, np.inf)
                for ind in range(ind_per_group):
                    yrandom = func(xdata, param[group * ind_per_group + ind])
                    rmse_group[ind] = ((yrandom - ydata) ** 2).mean()
                try:
                    best_group = np.where(rmse_group == min(rmse_group))[0][0]
                except(IndexError):
                    best_group = 0
                try:
                    param_final[group], _ = optimize.curve_fit(func, xdata, ydata, p0=param[best_group])
                    ybest_group = func(xdata, param_final[group])
                    rmse_final[group] = ((ybest_group - ydata) ** 2).mean()
                except(RuntimeError):
                    param_final[group] = np.zeros(param.shape[1])
                    rmse_final[group] = np.inf
            if min(rmse_final) > error_threshold ** 2:
                print(f"RMSE larger than {error_threshold ** 2}... Re-run...")
                continue
            try:
                best_final = np.where(rmse_final == min(rmse_final))[0][0]
                ybest_final = func(xdata, param_final[best_final])
                getattr(self, f"flt_{func.__name__}_param")[inds[count]] = np.array((param_final[best_final]))
                getattr(self, f"flt_{func.__name__}_hr")[inds[count]] = func(self.flt_x[inds[count]], param_final[best_final])
                count += 1
            except (KeyboardInterrupt):
                break
            except(IndexError):
                print("Failed to converge... Re-run...")
                continue
            if count == len(inds):
                print("All curve-fittings are completed!")
                break

    def compare_flatline(self, verbose=True):
        if not hasattr(self, "flt_funcs"):
            raise ValueError("Please fit curves after the final breakpoint using the method 'fit_flatline'!")
        col_names = ["Individual"] + self.flt_func_names
        rss_final, aic_final, aicc_final = ([self.ind] for _ in range(3))
        for func_name, func in zip(self.flt_func_names, self.flt_funcs):
            rss = []
            for ind in range(self.size):
                fitted = np.array([(temp, hr) for temp, hr in zip(self.flt_x[ind], getattr(self, f"flt_{func_name}_hr")[ind])
                                  if self.flt_raw[ind][:, 0].min() <= temp <= self.flt_raw[ind][:, 0].max()])
                fitted = func(self.flt_raw[ind][:, 0], getattr(self, f"flt_{func_name}_param")[ind])
                resid = np.array(fitted - self.flt_raw[ind][:, 1])
                rss.append((resid ** 2).sum())
            k = getattr(self, f"flt_{func_name}_param").shape[1]
            rss = np.array(rss)
            aic = 2 * k - 2 * 0.5 * (-self.flt_raw[ind].shape[0] * (np.log(2 * np.pi) + 1 +
                                                                    np.log(self.flt_raw[ind].shape[0]) + np.log(rss)))
            rss_final.append(rss)
            aic_final.append(aic)
            aicc_final.append(aic + ((2 * k ** 2) + 2 * k) / (self.flt_raw[ind].shape[0] - k - 1))
        for metric in ["rss", "aic", "aicc"]:
            setattr(self, f"flt_{metric}", pd.DataFrame(locals()[f"{metric}_final"]).transpose())
            getattr(self, f"flt_{metric}").columns = col_names
            for ind in range(self.size):
                row = getattr(self, f"flt_{metric}").loc[ind][1:]
                winner = np.where(row == row.min())[0][0] + 1
                getattr(self, f"flt_{metric}").loc[ind, "best performing"] = \
                    getattr(self, f"flt_{metric}").columns[winner]
            dummy = getattr(self, f"flt_{metric}").drop(["Individual", "best performing"], axis=1)
            min = dummy.min(axis=1)
            d_result = pd.concat([getattr(self, f"flt_{metric}")["Individual"],
                                  dummy.subtract(list(min), axis=0),
                                  getattr(self, f"flt_{metric}")["best performing"]], axis=1)
            setattr(self, f"flt_d_{metric}", d_result)
        if verbose: print(f"Attributes flt_rss, flt_aic, flt_aicc, flt_d_rss, flt_d_aic, flt_d_aicc are created.")

    def confirm_flatline_fit(self, flt_fit, verbose=True):
        if any(flt_fit == metric for metric in ["rss", "aic", "aicc"]):
            flt_fit = list(getattr(self, f"flt_{flt_fit}")["best performing"])
        if not isinstance(flt_fit, (list, tuple)):
            flt_fit = [flt_fit]
        if isinstance(flt_fit, (list, tuple)) and len(flt_fit) == 1:
            flt_fit = flt_fit * self.size
        hr_result, fit_result, flt_result = ([] for _ in range(3))
        for ind, func_name in enumerate(flt_fit):
            if isinstance(func_name, types.FunctionType):
                func_name = func_name.__name__
            hr = getattr(self, f"flt_{func_name}_hr")[ind]
            hr_result.append(hr)
            fit_result.append(func_name)
            flt_result.append(self.flt_x[ind][np.where(abs(hr) == min(abs(hr)))[0][0]])
        self.flt_hr = hr_result
        self.flt_fit = fit_result
        self.flt = np.array(flt_result)
        if verbose: print(f"Attributes flt_hr, flt_fit, flt are created.")

    def plot(self, func_names=None, raw=True, accuracy="rss", verbose=True):
        if all([accuracy != metric for metric in [None, "rss", "aic", "aicc"]]):
            raise ValueError("Please input a valid type if accuracy metric.")
        elif accuracy is not None and not hasattr(self, accuracy):
            self.calc_accuracy(verbose)
        metric = getattr(self, accuracy) if accuracy is not None else None
        if func_names == "all":
            func_names = self.func_names + ["spline"] if hasattr(self, "spline_hr") else self.func_names
        elif func_names is not None and func_names != "fit":
            func_names = [func_names] if not isinstance(func_names, (list, tuple)) else func_names
        for ind in range(self.size):
            fig, ax1 = plt.subplots()
            if raw:
                ax1.plot(self.raw[ind][:, 0], self.raw[ind][:, 1], "o")
            if func_names == "fit":
                curve = np.array([(temp, hr) for temp, hr in zip(self.temp, getattr(self, f"{self.fit[ind]}_hr")[ind])
                                  if self.raw[ind][0, 0] <= temp <= self.raw[ind][-1, 0]])
                label = f"{self.fit[ind]}" if accuracy is None \
                    else f"{self.fit[ind]}\n{accuracy}: {round(metric.loc[ind, self.fit[ind]], 2)}"
                ax1.plot(curve[:, 0], curve[:, 1], "-", label=label)
            else:
                for func_name in func_names:
                    func_name = func_name.__name__ if isinstance(func_name, types.FunctionType) else func_name
                    curve = np.array([(temp, hr) for temp, hr in zip(self.temp, getattr(self, f"{func_name}_hr")[ind])
                                      if self.raw[ind][:, 0].min() <= temp <= self.raw[ind][:, 0].max()])
                    label = f"{self.fit[ind]}" if accuracy is None \
                        else f"{self.fit[ind]}\n{accuracy}: {round(metric.loc[ind, func_name], 2)}"
                    ax1.plot(curve[:, 0], curve[:, 1], "-", label=label)
            ax1.set_title(f"{self.ind[ind]} (Individual: {ind})")
            ax1.set_xlabel("Temperature")
            ax1.set_ylabel("Heart rate")
            ax1.legend()
            plt.show()
            if ind < self.size - 1:
                if plt.waitforbuttonpress(): continue

    def plot_curvature(self, aspect_equal=False, verbose=True):
        if not hasattr(self, "kappa") or not hasattr(self, "breakpoint"):
            self.calc_curvature(standardize=True, verbose=verbose)
        for ind in range(self.size):
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            if aspect_equal:
                ax1.set_aspect("equal")
                ax2.set_aspect("equal")
            ax1.plot(self.temp, self.hr[ind], "k-")
            ax2.plot(self.kappa_x[ind], self.kappa[ind], "b--")
            for i in self.breakpoint[ind]:
                ax2.axvline(x=i, color="r", linestyle=":")
            ax1.set_title(f"{self.ind[ind]} (Individual: {ind})")
            ax1.set_xlabel("Temperature")
            ax1.set_ylabel("Heart rate")
            ax2.set_ylabel("Curvature (kappa)")
            plt.show()
            if ind < self.size - 1:
                if plt.waitforbuttonpress():
                    print(plt.waitforbuttonpress())
                    continue

    def plot_flatline(self, flt_func_names=None, raw=True, verbose=True):
        # if all([accuracy != metric for metric in [None, "rss", "aic", "aicc"]]):
        #     raise ValueError("Please input a valid type if accuracy metric.")
        # elif accuracy is not None and not hasattr(self, accuracy):
        #     self.calc_accuracy(verbose)
        # metric = getattr(self, accuracy) if accuracy is not None else None
        if flt_func_names == "all":
            flt_func_names = self.flt_func_names
        elif flt_func_names is not None and flt_func_names != "fit":
            flt_func_names = [flt_func_names] if not isinstance(flt_func_names, (list, tuple)) else flt_func_names
        for ind in range(self.size):
            fig, ax1 = plt.subplots()
            if raw:
                ax1.plot(self.flt_raw[ind][:, 0], self.flt_raw[ind][:, 1], "o")
            for func_name in flt_func_names:
                func_name = func_name.__name__ if isinstance(func_name, types.FunctionType) else func_name
                curve = np.array([(temp, hr) for temp, hr in zip(self.flt_x[ind], getattr(self, f"flt_{func_name}_hr")[ind])
                                  if self.flt_raw[ind][:, 0].min() <= temp <= self.flt_raw[ind][:, 0].max()])
                label = f"{func_name}"
                ax1.plot(curve[:, 0], curve[:, 1], "-", label=label)
            ax1.set_title(f"{self.ind[ind]} (Individual: {ind})")
            ax1.set_xlabel("Temperature")
            ax1.set_ylabel("Heart rate")
            ax1.legend()
            plt.show()
            if ind < self.size - 1:
                if plt.waitforbuttonpress(): continue

    def save(self, figsize=(2.3622, 1.77165), xlim=(24, 50), ylim=(0, 180), dpi=200,
             breakpoint=True, labels_separate=False):
        try:
            os.mkdir(f"{self.name}_TPCs")
        except FileExistsError:
            pass
        os.chdir(f".\\{self.name}_TPCs")
        for ind in range(self.size):
            fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
            ax1.scatter(self.raw[ind][:, 0], self.raw[ind][:, 1], color="dimgrey", marker=".", s=8)
            curve = np.array([(temp, hr) for temp, hr in zip(self.temp, getattr(self, f"{self.fit[ind]}_hr")[ind])
                              if self.raw[ind][0, 0] <= temp <= self.raw[ind][-1, 0]])
            ax1.plot(curve[:, 0], curve[:, 1], "b-")
            if breakpoint:
                for x in self.breakpoint[ind]: ax1.axvline(x, color="red", linestyle="--")
            if labels_separate:
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
            else:
                ax1.set_xlabel("Temperature (°C)")
                ax1.set_ylabel("Heart rate (beats per minute)")
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            plt.tight_layout()
            plt.savefig(f"{self.ind[ind]}.pdf")
        if labels_separate:
            fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
            ax1.scatter(self.raw[0][:, 0], self.raw[0][:, 1], color="dimgrey", marker=".", s=8)
            curve = np.array([(temp, hr) for temp, hr in zip(self.temp, getattr(self, f"{self.fit[0]}_hr")[0])
                              if self.raw[0][0, 0] <= temp <= self.raw[0][-1, 0]])
            ax1.plot(curve[:, 0], curve[:, 1], "b-")
            if breakpoint:
                for x in self.breakpoint[0]: ax1.axvline(x, color="red", linestyle="--")
            ax1.set_xlabel("Temperature (°C)")
            ax1.set_ylabel("Heart rate (beats per minute)")
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            plt.tight_layout()
            plt.savefig(f"{self.name}_axes.pdf")
        os.chdir("..\\")