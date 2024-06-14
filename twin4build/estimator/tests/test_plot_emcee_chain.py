import matplotlib.pyplot as plt
import matplotlib
import pickle
import math
import numpy as np
import os
import datetime
import sys
import corner
import seaborn as sns
import copy
from dateutil import tz
from io import StringIO
from matplotlib.colors import LinearSegmentedColormap
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
from twin4build.utils.uppath import uppath
from twin4build.simulator.simulator import Simulator
from twin4build.model.model import Model
from twin4build.model.tests.test_LBNL_bypass_coil_model import fcn
import twin4build.utils.plot.plot as plot


def get_attr_list(model: Model):
    x = model.chain_log["chain.x"].shape[3]
    number_list = list(range(1, x + 1))
    flat_attr_list_ = []
    
    for number in number_list:
        flat_attr_list_.append(str(number))

    return flat_attr_list_

def plot_preprocessing(model: Model):

    flat_attr_list_ = [r"$\overline{\dot{m}}_{c,w}$", r"$\overline{\dot{m}}_{c,a}$", r"$\tau_w$", r"$\tau_a$",
                      r"$\tau_m$", r"$\overline{UA}$", r"$\overline{\dot{m}}_{v,w}$", r"$\overline{\dot{m}}_{cv,w}$",
                      r"$K_{cv}$", r"$\Delta P_{s,res}$", r"$\overline{\Delta P}_{c}$", r"$\Delta P_{p}$",
                      r"$\Delta P_{s}$", r"$c_1$", r"$c_2$", r"$c_3$", r"$c_4$", r"$f_{tot}$", r"$K_P$", r"$T_I$",
                      r"$T_D$"]

    x = model.chain_log["chain.x"].shape[3]
    number_list = list(range(1,x+1))
    flat_attr_list_ = []

    for number in number_list:
        flat_attr_list_.append(str(number))

    flat_attr_list__ = [flat_attr_list_]

    result_list = model.chain_log["chain.x"]

    for ii, (flat_attr_list_, result_) in enumerate(zip(flat_attr_list__, result_list)):
        nparam = len(flat_attr_list_)
        ncols = 3
        nrows = math.ceil(nparam / ncols)

        ndim = model.chain_log["chain.x"].shape[3]
        ntemps = model.chain_log["chain.x"].shape[1]
        nwalkers = model.chain_log["chain.x"].shape[2]  # Round up to nearest even number and multiply by 2

        cm_sb = sns.diverging_palette(210, 0, s=50, l=50, n=ntemps, center="dark")  # vlag_r
        cm_sb_rev = list(reversed(cm_sb))
        cm_mpl = LinearSegmentedColormap.from_list("seaborn", cm_sb)  # , N=ntemps)
        cm_mpl_rev = LinearSegmentedColormap.from_list("seaborn_rev", cm_sb_rev, N=ntemps)

        fig_trace_beta, axes_trace = plt.subplots(nrows=nrows, ncols=ncols, layout='compressed')
        fig_trace_beta.set_size_inches((17, 12))

    return result_ , fig_trace_beta , axes_trace , flat_attr_list_ , ncols , ntemps, cm_mpl_rev, nwalkers, cm_sb

def plot_jump_plot(model: Model):

    ntemps = model.chain_log["chain.x"].shape[1]

    cm_sb = sns.diverging_palette(210, 0, s=50, l=50, n=ntemps, center="dark")

    fig_jump, ax_jump = plt.subplots(layout='compressed')
    fig_jump.set_size_inches((17, 12))
    fig_jump.suptitle("Jumps", fontsize=20)

    n_it = model.chain_log["chain.jump_acceptance"].shape[0]
    for i in range(ntemps):
        if i == 0:
            ax_jump.plot(range(n_it), model.chain_log["chain.jump_acceptance"][:, i], color=cm_sb[i])

    plt.show()

def plot_logl_plot(model: Model):
    ntemps = model.chain_log["chain.x"].shape[1]
    nwalkers = model.chain_log["chain.x"].shape[2]

    cm_sb = sns.diverging_palette(210, 0, s=50, l=50, n=ntemps, center="dark")

    fig_logl, ax_logl = plt.subplots(layout='compressed')
    fig_logl.set_size_inches((17 / 4, 12 / 4))
    fig_logl.suptitle("Log-likelihood", fontsize=20)
    logl = model.chain_log["chain.logl"]
    logl[np.abs(logl) > 1e+9] = np.nan

    indices = np.where(logl[:, 0, :] == np.nanmax(logl[:, 0, :]))
    #print(logl[:, 0, :].max())
    s0 = indices[0][0]
    s1 = indices[1][0]
    #print("logl_max: ", logl[s0, 0, s1])

    n_it = model.chain_log["chain.logl"].shape[0]
    for i_walker in range(nwalkers):
        for i in range(ntemps):
            if i_walker == 0:  #######################################################################
                ax_logl.plot(range(n_it), logl[:, i, i_walker], color=cm_sb[i])

    plt.show()

def plot_iac_plot(model: Model):
    colors = sns.color_palette("deep")
    red = colors[3]
    plot.load_params()

    result_, fig_trace_beta, axes_trace, flat_attr_list_, ncols, ntemps, cm_mpl_rev, nwalkers, cm_sb = iac_plot_preprocessing(model)


    axes_iac = copy.deepcopy(axes_trace)
    for j, attr in enumerate(flat_attr_list_):
        row = math.floor(j / ncols)
        col = int(j - ncols * row)
        axes_iac[row, col] = axes_trace[row, col].twinx()

    print(result_)

    iac = result_["integratedAutoCorrelatedTime"][:-1]
    print(iac)

    n_it = iac.shape[0]
    for i in range(ntemps):
        for j, attr in enumerate(flat_attr_list_):
            row = math.floor(j / ncols)

    heuristic_line = np.arange(n_it) / 20
    for j, attr in enumerate(flat_attr_list_):
        row = math.floor(j / ncols)
        col = int(j - ncols * row)
        axes_iac[row, col].plot(range(n_it), heuristic_line, color="black", linewidth=1, linestyle="dashed",
                                alpha=1, label=r"$\tau=N/50$")
        axes_iac[row, col].set_ylim([0 - 0.05 * iac.max(), iac.max() + 0.05 * iac.max()])

    plt.show()

def plot_swap_plot(model: Model):
    result_, fig_trace_beta, axes_trace, flat_attr_list_, ncols, ntemps, cm_mpl_rev, nwalkers, cm_sb = iac_plot_preprocessing(
        model)

    fig_swap, ax_swap = plt.subplots(layout='compressed')
    fig_swap.set_size_inches((17, 12))
    fig_swap.suptitle("Swaps", fontsize=20)
    n = ntemps-1
    for i in range(n):
        if i==0: #######################################################################
            ax_swap.plot(range(result_["chain.swaps_accepted"][:,i].shape[0]), result_["chain.swaps_accepted"][:,i]/result_["chain.swaps_proposed"][:,i], color=cm_sb[i])

    plt.show()

def plot_trace_plot(model: Model, n_subplots: int = 20, one_plot = False, burnin: int = 2000, max_cols = 4, save_plot: bool = True, file_name: str = 'TracePlot'):
    flat_attr_list_ = get_attr_list(model)

    ntemps = model.chain_log["chain.x"].shape[1]
    nwalkers = model.chain_log["chain.x"].shape[2]

    cm_sb = sns.diverging_palette(210, 0, s=50, l=50, n=ntemps, center="dark")
    cm_sb_rev = list(reversed(cm_sb))
    cm_mpl_rev = LinearSegmentedColormap.from_list("seaborn_rev", cm_sb_rev, N=ntemps)

    vmin = np.min(model.chain_log["chain.betas"])
    vmax = np.max(model.chain_log["chain.betas"])
    burnin = burnin

    chain_logl = model.chain_log["chain.logl"]
    bool_ = chain_logl < -5e+9
    chain_logl[bool_] = np.nan
    chain_logl[np.isnan(chain_logl)] = np.nanmin(chain_logl)

    num_attributes = len(flat_attr_list_)
    max_cols = max_cols
    
    if one_plot:
        n_subplots = len(flat_attr_list_)

    for start in range(0, num_attributes, n_subplots):
        end = min(start + n_subplots, num_attributes)
        current_attrs = flat_attr_list_[start:end]
        num_current_attrs = len(current_attrs)

        num_cols = max_cols
        num_rows = math.ceil(num_current_attrs / num_cols)

        fig, axes_trace = plt.subplots(num_rows, num_cols)
        fig.set_size_inches(17, 12)

        if num_rows == 1:
            axes_trace = np.expand_dims(axes_trace, axis=0)
        if num_cols == 1:
            axes_trace = np.expand_dims(axes_trace, axis=1)

        axes_trace = axes_trace.flatten()

        for nt in reversed(range(ntemps)):
            for nw in range(nwalkers):
                x = model.chain_log["chain.x"][:, nt, nw, :]
                beta = model.chain_log["chain.betas"][:, nt]

                for j, attr in enumerate(current_attrs):
                    ax = axes_trace[j]
                    if ntemps > 1:
                        sc = ax.scatter(range(x[:, start + j].shape[0]), x[:, start + j], c=beta, vmin=vmin, vmax=vmax,
                                        s=0.3, cmap=cm_mpl_rev, alpha=0.1)
                    else:
                        sc = ax.scatter(range(x[:, start + j].shape[0]), x[:, start + j], s=0.3, color=cm_sb[0],
                                        alpha=0.1)
                    ax.axvline(burnin, color="black", linewidth=1, alpha=0.8)

        x_left = 0.1
        x_mid_left = 0.515
        x_right = 0.9
        x_mid_right = 0.58
        dx_left = x_mid_left - x_left
        dx_right = x_right - x_mid_right

        fontsize = 12
        for j, attr in enumerate(current_attrs):
            ax = axes_trace[j]
            ax.axvline(burnin, color="black", linestyle=":", linewidth=1.5, alpha=0.5)
            y = np.array([-np.inf, np.inf])
            x1 = -burnin
            ax.fill_betweenx(y, x1, x2=0)
            ax.text(x_left + dx_left / 2, 0.44, 'Burn-in', ha='center', va='center',
                    rotation='horizontal', fontsize=fontsize, transform=ax.transAxes)

            ax.text(x_mid_right + dx_right / 2, 0.44, 'Posterior', ha='center', va='center',
                    rotation='horizontal', fontsize=fontsize, transform=ax.transAxes)

            ax.set_ylabel(attr, fontsize=20)
            ax.ticklabel_format(style='plain', useOffset=False)

        if ntemps > 1:
            cb = fig.colorbar(sc, ax=axes_trace.ravel().tolist())
            cb.set_label(label=r"$T$", size=30)
            cb.solids.set(alpha=1)
            dist = (vmax - vmin) / (ntemps) / 2
            tick_start = vmin + dist
            tick_end = vmax - dist
            tick_locs = np.linspace(tick_start, tick_end, ntemps)[::-1]
            cb.set_ticks(tick_locs)
            labels = list(model.chain_log["chain.T"][0, :])
            inf_label = r"$\infty$"
            labels[-1] = inf_label
            ticklabels = [str(round(float(label), 1)) if not isinstance(label, str) else label for label in labels]
            cb.set_ticklabels(ticklabels, size=12)

            for tick in cb.ax.get_yticklabels():
                tick.set_fontsize(12)
                txt = tick.get_text()
                if txt == inf_label:
                    tick.set_fontsize(20)
                    
        if ntemps == 1:
            plt.tight_layout()
        plt.show()

        if save_plot == True:
            fig.savefig(file_name + str(start + 1) + ".png")


def plot_corner_plot(model: Model, subsample_factor=10, burnin: int = 2000, save_plot: bool = False, file_name = "CornerPlot"):
    burnin = burnin

    flat_attr_list_ = get_attr_list(model)

    ntemps = model.chain_log["chain.x"].shape[1]

    cm_sb = sns.diverging_palette(210, 0, s=50, l=50, n=ntemps, center="dark")

    parameter_chain = model.chain_log["chain.x"][burnin:, 0, :, :]
    parameter_chain = parameter_chain.reshape(parameter_chain.shape[0] * parameter_chain.shape[1],
                                              parameter_chain.shape[2])

    subsampled_chain = parameter_chain[::subsample_factor]

    fig_corner = corner.corner(subsampled_chain, fig=None, labels=flat_attr_list_, labelpad=-0.2, show_titles=True,
                               color=cm_sb[0], plot_contours=True, bins=15, hist_bin_factor=5, max_n_ticks=3,
                               quantiles=[0.16, 0.5, 0.84],
                               title_kwargs={"fontsize": 10, "ha": "left", "position": (0.03, 1.01)})
    fig_corner.set_size_inches((12, 12))
    pad = 0.025
    fig_corner.subplots_adjust(left=pad, bottom=pad, right=1 - pad, top=1 - pad, wspace=0.08, hspace=0.08)
    axes = fig_corner.get_axes()
    for ax in axes:
        ax.set_xticks([], minor=True)
        ax.set_xticks([])
        ax.set_yticks([], minor=True)
        ax.set_yticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    median = np.median(parameter_chain, axis=0)
    corner.overplot_lines(fig_corner, median, color='red', linewidth=0.5)
    corner.overplot_points(fig_corner, median.reshape(1, median.shape[0]), marker="s", color='red')

    plt.show()

    if save_plot == True:
        plt.savefig(file_name + ".png")

def plot_corner_plot_multiple_splits(model, subsample_factor=10, burnin: int = 2000, save_plot: bool = False, file_name = "CornerPlot"):
    flat_attr_list_ = get_attr_list(model)

    ntemps = model.chain_log["chain.x"].shape[1]

    cm_sb = sns.diverging_palette(210, 0, s=50, l=50, n=ntemps, center="dark")

    burnin = burnin

    parameter_chain = model.chain_log["chain.x"][burnin:, 0, :, :]
    parameter_chain = parameter_chain.reshape(parameter_chain.shape[0] * parameter_chain.shape[1],
                                              parameter_chain.shape[2])

    parameter_chain = parameter_chain[::subsample_factor]

    num_params = parameter_chain.shape[1]
    midpoint = num_params // 2

    chain_subset1 = parameter_chain[:, :midpoint]
    chain_subset2 = parameter_chain[:, midpoint:]

    subsampled_chain1 = chain_subset1
    subsampled_chain2 = chain_subset2

    fig_corner1 = plt.figure(figsize=(12, 12))
    corner.corner(subsampled_chain1, fig=fig_corner1, labels=flat_attr_list_[:midpoint], labelpad=-0.2,
                  show_titles=True, color=cm_sb[0], plot_contours=True, bins=15, hist_bin_factor=5, max_n_ticks=3,
                  quantiles=[0.16, 0.5, 0.84],
                  title_kwargs={"fontsize": 10, "ha": "left", "position": (0.03, 1.01)},
                  range=[(np.min(parameter_chain), np.max(parameter_chain))] * midpoint)
    fig_corner1.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    axes1 = fig_corner1.get_axes()

    for ax in axes1:
        ax.set_xticks([], minor=True)
        ax.set_xticks([])
        ax.set_yticks([], minor=True)
        ax.set_yticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    median1 = np.median(subsampled_chain1, axis=0)
    corner.overplot_lines(fig_corner1, median1, color='red', linewidth=0.5)
    corner.overplot_points(fig_corner1, median1.reshape(1, median1.shape[0]), marker="s", color='red')

    if save_plot == True:
        plt.savefig(file_name + "_split1")

    plt.show()

    fig_corner2 = plt.figure(figsize=(12, 12))
    corner.corner(subsampled_chain2, fig=fig_corner2, labels=flat_attr_list_[midpoint:], labelpad=-0.2,
                  show_titles=True, color=cm_sb[0], plot_contours=True, bins=15, hist_bin_factor=5, max_n_ticks=3,
                  quantiles=[0.16, 0.5, 0.84],
                  title_kwargs={"fontsize": 10, "ha": "left", "position": (0.03, 1.01)},
                  range=[(np.min(parameter_chain), np.max(parameter_chain))] * (num_params - midpoint))
    fig_corner2.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    axes2 = fig_corner2.get_axes()

    for ax in axes2:
        ax.set_xticks([], minor=True)
        ax.set_xticks([])
        ax.set_yticks([], minor=True)
        ax.set_yticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    median2 = np.median(subsampled_chain2, axis=0)
    corner.overplot_lines(fig_corner2, median2, color='red', linewidth=0.5)
    corner.overplot_points(fig_corner2, median2.reshape(1, median2.shape[0]), marker="s", color='red')

    plt.show()

    if save_plot == True:
        plt.savefig(file_name + "_split2")