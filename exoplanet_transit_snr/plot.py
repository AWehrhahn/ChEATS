# -*- coding: utf-8 -*-
from os import makedirs
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

from .stats import gauss


def plot_cc_overview(
    cc_data, vp_idx=None, show_line=False, title="", folder=None, show=False
):
    if show_line and vp_idx is None:
        raise ValueError("vp_idx is required if line is set to True")

    fig = plt.figure(figsize=(12, 8))
    plt.clf()

    for i in range(cc_data.shape[0]):
        plt.subplot(6, 3, i + 1)
        plt.imshow(cc_data[i], aspect="auto", origin="lower", interpolation="none")
        if show_line:
            plt.plot(vp_idx, np.arange(len(vp_idx)), "r-.", alpha=0.5)
    plt.suptitle(title)

    if folder is not None:
        if not show_line:
            plot_fname = f"{folder}/ccresult.png"
        else:
            plot_fname = f"{folder}/ccresult_line.png"

        makedirs(dirname(plot_fname), exist_ok=True)
        plt.savefig(plot_fname)
    if show:
        plt.show()
    plt.close(fig)


def plot_cc_overview_details(
    cc_data, vp_idx=None, show_line=False, title="", folder="", show=False
):
    if show_line and vp_idx is None:
        raise ValueError("vp_idx is required if line is set to True")

    fig = plt.figure(figsize=(12, 8))
    n_ord = cc_data.shape[0]
    for i in range(n_ord):
        plt.clf()
        plt.imshow(cc_data[i], aspect="auto", origin="lower", interpolation="none")
        if show_line:
            plt.plot(vp_idx, np.arange(len(vp_idx)), "r-.", alpha=0.5)
        plt.suptitle(f"{title}\nSegment {i}")

        if folder is not None:
            if not show_line:
                plot_fname = f"{folder}/ccresult_segment_{i}.png"
            else:
                plot_fname = f"{folder}/ccresult_segment_{i}_line.png"

            makedirs(dirname(plot_fname), exist_ok=True)
            plt.savefig(plot_fname)
        if show:
            plt.show()
        plt.close(fig)


def plot_combined(
    rv_array,
    combined,
    in_transit=None,
    vp_idx=None,
    show_line=False,
    title="",
    folder=None,
    suffix="",
    show=False,
):
    if show_line and vp_idx is None:
        raise ValueError("vp_idx is required if line is set to True")

    fig = plt.figure(figsize=(12, 8))
    plt.clf()

    vmin, vmax = np.nanpercentile(combined.ravel(), [1, 99])
    plt.imshow(
        combined,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        # interpolation="none",
    )
    plt.xlabel("rv [km/s]")
    xticks = plt.xticks()[0][1:-1]
    rv_ticks = np.linspace(xticks[0], xticks[-1], len(rv_array))
    xticks_labels = np.interp(xticks, rv_ticks, rv_array)
    xticks_labels = [f"{x:.3g}" for x in xticks_labels]
    plt.xticks(xticks, labels=xticks_labels)

    if show_line:
        plt.plot(vp_idx, np.arange(len(vp_idx)), "r-.", alpha=0.5)

    if in_transit is not None and np.any(in_transit):
        n_obs, n_points = combined.shape
        plt.hlines(
            np.arange(n_obs)[in_transit][0] - 0.5,
            -0.5,
            n_points + 0.5,
            "k",
            "--",
        )
        plt.hlines(
            np.arange(n_obs)[in_transit][-1] + 0.5,
            -0.5,
            n_points + 0.5,
            "k",
            "--",
        )
        plt.xlim(-0.5, n_points + 0.5)

    plt.suptitle(f"{title}\nCombined")

    if folder is not None:
        if not show_line:
            plot_fname = f"{folder}/ccresult_combined{suffix}.png"
        else:
            plot_fname = f"{folder}/ccresult_combined_line{suffix}.png"

        makedirs(dirname(plot_fname), exist_ok=True)
        plt.savefig(plot_fname)
    if show:
        plt.show()
    plt.close(fig)


def plot_combined_hist(combined, title="", folder=None, show=False):
    fig = plt.figure(figsize=(12, 8))
    plt.clf()

    plt.hist(combined.ravel(), bins="auto")
    plt.suptitle(title)

    if folder is not None:
        plot_fname = f"{folder}/ccresult_combined_hist.png"
        makedirs(dirname(plot_fname), exist_ok=True)
        plt.savefig(plot_fname)
    if show:
        plt.show()
    plt.close(fig)


def plot_vsys_kp(
    res,
    title="",
    folder=None,
    plot_sums=True,
    plot_rectangle=True,
    plot_lines=True,
    show=False,
):
    plt.clf()

    if plot_sums:
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(3, 3, figure=fig, wspace=0.1, hspace=0.1)
        ax = fig.add_subplot(gs[1:, :2])  # Main plot
        ax2 = fig.add_subplot(gs[0, :2])  # plot over columns, i.e. vsys
        ax3 = fig.add_subplot(gs[1:, 2])  # Plot over rows, i.e. kp
    else:
        fig, ax = plt.subplots(figsize=(6, 8))

    # data = np.log(1 + np.abs(res["combined"]))
    # data *= np.sign(res["combined"])
    vmax = np.nanmax(np.abs(res["combined"]))
    vmin = -vmax
    ax.imshow(
        res["combined"],
        aspect="auto",
        origin="lower",
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
    )
    if plot_rectangle:
        ax.add_patch(
            plt.Rectangle(
                (
                    res["vsys_peak"] - res["vsys_width"],
                    res["kp_peak"] - res["kp_width"],
                ),
                2 * res["vsys_width"],
                2 * res["kp_width"],
                fill=False,
                color="red",
            )
        )
    if plot_lines:
        gap = 0.05
        n, m = res["combined"].shape
        x = np.interp(res["vsys_expected"], res["vsys"], np.arange(m)) + 0.5
        y = np.interp(res["kp_expected"], res["kp"], np.arange(n)) + 0.5
        ax.axvline(x, 0, y / n - gap, color="k", linestyle="dashed")
        ax.axvline(x, y / n + gap, 1, color="k", linestyle="dashed")
        ax.axhline(y, 0, x / m - gap, color="k", linestyle="dashed")
        ax.axhline(y, x / m + gap, 1, color="k", linestyle="dashed")

        if res["vsys_popt"] is not None and res["kp_popt"] is not None:
            x = np.interp(res["vsys_popt"][1], res["vsys"], np.arange(m)) + 0.5
            y = np.interp(res["kp_popt"][1], res["kp"], np.arange(n)) + 0.5
            ax.axvline(x, 0, y / n - gap, color="r", linestyle="dashdot")
            ax.axvline(x, y / n + gap, 1, color="r", linestyle="dashdot")
            ax.axhline(y, 0, x / m - gap, color="r", linestyle="dashdot")
            ax.axhline(y, x / m + gap, 1, color="r", linestyle="dashdot")

    ax.set_xlabel("vsys [km/s]", fontsize="x-large")
    xticks = np.unique(res["vsys"] // 5) * 20
    xticks_labels = np.array([f"{x:.3g}" for x in xticks])
    npoints = res["combined"].shape[1]
    xticks = interp1d(res["vsys"], np.arange(npoints), fill_value="extrapolate")(xticks)
    xticks_labels = xticks_labels[(xticks >= 0) & (xticks <= npoints)]
    xticks = xticks[(xticks >= 0) & (xticks <= npoints)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels, fontsize="large")

    ax.set_ylabel("Kp [km/s]", fontsize="x-large")
    yticks = np.unique(res["kp"] // 50) * 50
    yticks_labels = np.array([f"{x:.3g}" for x in yticks])
    npoints = res["combined"].shape[0]
    yticks = interp1d(res["kp"], np.arange(npoints), fill_value="extrapolate")(yticks)
    yticks_labels = yticks_labels[(yticks >= 0) & (yticks <= npoints)]
    yticks = yticks[(yticks >= 0) & (yticks <= npoints)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_labels, fontsize="large")

    if plot_sums:
        # vsys plot
        ax2.plot(res["vsys"], res["vsys_mean"])

        if plot_lines:
            ax2.axvline(
                res["vsys_expected"],
                color="k",
                linestyle="dashed",
            )
        if res["vsys_popt"] is not None:
            ax2.plot(res["vsys"], gauss(res["vsys"], *res["vsys_popt"]), "r-.")
            ax2.axvline(
                res["vsys_popt"][1],
                color="r",
                linestyle="dashdot",
            )
        # ax2.set_xlabel("vsys [km/s]")
        vmin, vmax = res["vsys"][0], res["vsys"][-1]
        ax2.set_xlim(vmin - 0.5, vmax + 0.5)
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position("top")

        # Kp plot
        ax3.plot(res["kp_mean"], res["kp"])
        if plot_lines:
            ax3.axhline(
                res["kp_expected"],
                color="k",
                linestyle="dashed",
            )
        if res["kp_popt"] is not None:
            ax3.plot(gauss(res["kp"], *res["kp_popt"]), res["kp"], "r-.")
            ax3.axhline(
                res["kp_popt"][1],
                color="r",
                linestyle="dashdot",
            )
        # ax3.set_ylabel("Kp [km/s]")
        vmin, vmax = res["kp"][0], res["kp"][-1]
        ax3.set_ylim(vmin - 0.5, vmax + 0.5)
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")

    plt.suptitle(title)
    if folder is not None:
        if plot_sums:
            plot_fname = f"{folder}/ccresults_vsys_kp_plot.png"
        else:
            plot_fname = f"{folder}/ccresults_vsys_kp_plot_detail.png"
        makedirs(dirname(plot_fname), exist_ok=True)
        plt.savefig(plot_fname)
    if show:
        plt.show()
    plt.close(fig)


def plot_cohend_distribution(res, title="", folder=None, show=False):
    fig = plt.figure(figsize=(12, 8))
    plt.clf()

    plt.hist(
        res["in_trail"].ravel(),
        bins=res["bins"],
        density=True,
        histtype="step",
        label="in-trail",
    )
    plt.hist(
        res["out_of_trail"].ravel(),
        bins=res["bins"],
        density=True,
        histtype="step",
        label="out-of-trail",
    )
    plt.legend()
    plt.suptitle(f"{title}\nWelch t: {res['t']}\nCohen d: {res['d']}")
    if folder is not None:
        plot_fname = f"{folder}/ccresult_cohend.png"
        makedirs(dirname(plot_fname), exist_ok=True)
        plt.savefig(plot_fname)
    if show:
        plt.show()
    plt.close(fig)


def plot_results(rv_array, cc_data, combined, res, title="", folder=None, show=False):
    # Plot the cross correlation
    in_transit = res["in_transit"]
    vp_idx = np.interp(res["vp"], rv_array, np.arange(rv_array.size))
    vp_new = res["vsys"][res["vsys_peak"]] + res["kp"][res["kp_peak"]] * np.sin(
        2 * np.pi * res["phi"]
    )
    if res["rv_bary"] is not None:
        vp_new += res["rv_bary"]
    vp_idx_new = np.interp(vp_new, rv_array, np.arange(rv_array.size))

    # Plot the results

    # CC functions for each order
    plot_cc_overview(cc_data, vp_idx, False, title=title, folder=folder, show=show)
    plot_cc_overview(cc_data, vp_idx, True, title=title, folder=folder, show=show)
    # Plot the cc function for each order alone, for more details
    plot_cc_overview_details(
        cc_data, vp_idx, False, title=title, folder=folder, show=False
    )
    plot_cc_overview_details(
        cc_data, vp_idx, True, title=title, folder=folder, show=False
    )
    # Plot the combined data
    plot_combined(
        rv_array,
        combined,
        in_transit,
        vp_idx,
        False,
        title=title,
        folder=folder,
        show=show,
    )
    plot_combined(
        rv_array,
        combined,
        in_transit,
        vp_idx,
        True,
        title=title,
        folder=folder,
        show=show,
    )
    # combined with best fit line
    suffix = "_best_fit"
    plot_combined(
        rv_array,
        combined,
        in_transit,
        vp_idx_new,
        True,
        title=title,
        folder=folder,
        show=show,
        suffix=suffix,
    )
    # histogram of combined
    plot_combined_hist(combined, title=title, folder=folder, show=show)
    # Vsys - Kp Plot
    plot_vsys_kp(res, title=title, folder=folder, plot_rectangle=False, show=show)
    plot_vsys_kp(
        res,
        title=title,
        folder=folder,
        plot_rectangle=False,
        plot_sums=False,
        show=show,
    )
    # Cohen d distribution
    plot_cohend_distribution(res, title=title, folder=folder, show=show)

    pass
