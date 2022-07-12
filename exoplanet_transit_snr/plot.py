# -*- coding: utf-8 -*-
from os import makedirs
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from .stats import gauss


def plot_cc_overview(
    cc_data, vp_idx=None, show_line=False, title="", folder="", show=False
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
    folder="",
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
        interpolation="none",
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
            np.arange(n_obs)[in_transit][0],
            -0.5,
            n_points + 0.5,
            "k",
            "--",
        )
        plt.hlines(
            np.arange(n_obs)[in_transit][-1],
            -0.5,
            n_points + 0.5,
            "k",
            "--",
        )
        plt.xlim(-0.5, n_points + 0.5)

    plt.suptitle(f"{title}\nCombined")

    if not show_line:
        plot_fname = f"{folder}/ccresult_combined{suffix}.png"
    else:
        plot_fname = f"{folder}/ccresult_combined_line{suffix}.png"

    makedirs(dirname(plot_fname), exist_ok=True)
    plt.savefig(plot_fname)
    if show:
        plt.show()
    plt.close(fig)


def plot_combined_hist(combined, title="", folder="", show=False):
    fig = plt.figure(figsize=(12, 8))
    plt.clf()

    plt.hist(combined.ravel(), bins="auto")
    plt.suptitle(title)

    plot_fname = f"{folder}/ccresult_combined_hist.png"

    makedirs(dirname(plot_fname), exist_ok=True)
    plt.savefig(plot_fname)
    if show:
        plt.show()
    plt.close(fig)


def plot_vsys_kp(
    res,
    title="",
    folder="",
    plot_sums=True,
    plot_rectangle=True,
    plot_lines=True,
    show=False,
):
    plt.clf()

    if plot_sums:
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(121)
    else:
        fig, ax = plt.subplots(figsize=(6, 8))

    # data = np.log(1 + np.abs(res["combined"]))
    # data *= np.sign(res["combined"])
    vmax = np.nanmax(np.abs(res["combined"]))
    vmin = -vmax
    plt.imshow(
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
        n = res["combined"].shape[0]
        x = np.interp(res["kp_expected"], res["kp"], np.arange(n)) + 0.5
        n = res["combined"].shape[1]
        plt.hlines(x, 0.5, n - 0.5, "k", "dashed")
        n = res["combined"].shape[1]
        x = np.interp(res["vsys_expected"], res["vsys"], np.arange(n)) + 0.5
        n = res["combined"].shape[0]
        plt.vlines(x, 0.5, n - 0.5, "k", "dashed")

    plt.xlabel("vsys [km/s]", fontsize="x-large")
    xticks = np.unique(res["vsys"] // 10) * 10
    xticks_labels = np.array([f"{x:.3g}" for x in xticks])
    npoints = res["combined"].shape[1]
    xticks = interp1d(res["vsys"], np.arange(npoints), fill_value="extrapolate")(xticks)
    xticks_labels = xticks_labels[(xticks >= 0) & (xticks <= npoints)]
    xticks = xticks[(xticks >= 0) & (xticks <= npoints)]
    plt.xticks(xticks, labels=xticks_labels, fontsize="large")

    plt.ylabel("Kp [km/s]", fontsize="x-large")
    yticks = np.unique(res["kp"] // 50) * 50
    yticks_labels = np.array([f"{x:.3g}" for x in yticks])
    npoints = res["combined"].shape[0]
    yticks = interp1d(res["kp"], np.arange(npoints), fill_value="extrapolate")(yticks)
    yticks_labels = yticks_labels[(yticks >= 0) & (yticks <= npoints)]
    yticks = yticks[(yticks >= 0) & (yticks <= npoints)]
    plt.yticks(yticks, labels=yticks_labels, fontsize="large")

    if plot_sums:
        plt.subplot(222)
        plt.plot(res["vsys"], res["vsys_mean"])

        if plot_lines:
            n = np.digitize(res["vsys_expected"], res["vsys"])
            plt.vlines(
                res["vsys_expected"],
                np.min(res["vsys_mean"]),
                res["vsys_mean"][n],
                "k",
                "--",
            )
        if res["vsys_popt"] is not None:
            plt.plot(res["vsys"], gauss(res["vsys"], *res["vsys_popt"]), "r--")
        plt.xlabel("vsys [km/s]")

        plt.subplot(224)
        plt.plot(res["kp"], res["kp_mean"])
        if plot_lines:
            n = np.digitize(res["kp_expected"], res["kp"])
            plt.vlines(
                res["kp_expected"],
                np.min(res["kp_mean"]),
                res["kp_mean"][n],
                "k",
                "--",
            )
        if res["kp_popt"] is not None:
            plt.plot(res["kp"], gauss(res["kp"], *res["kp_popt"]), "r--")
        plt.xlabel("Kp [km/s]")

    plt.suptitle(title)
    if plot_sums:
        plot_fname = f"{folder}/ccresults_vsys_kp_plot.png"
    else:
        plot_fname = f"{folder}/ccresults_vsys_kp_plot_detail.png"

    makedirs(dirname(plot_fname), exist_ok=True)
    plt.savefig(plot_fname)
    if show:
        plt.show()
    plt.close(fig)


def plot_cohend_distribution(res, title="", folder="", show=False):
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
    plot_fname = f"{folder}/ccresult_cohend.png"
    makedirs(dirname(plot_fname), exist_ok=True)
    plt.savefig(plot_fname)
    if show:
        plt.show()
    plt.close(fig)


def plot_results(rv_array, cc_data, combined, res, title="", folder="", show=False):
    # Plot the cross correlation
    in_transit = res["in_transit"]
    vp_idx = np.interp(res["vp"], rv_array, np.arange(rv_array.size))
    vp_new = res["vsys"][res["vsys_peak"]] + res["kp"][res["kp_peak"]] * np.sin(
        2 * np.pi * res["phi"]
    )
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
    plot_vsys_kp(res, title=title, folder=folder, show=show)
    plot_vsys_kp(res, title=title, folder=folder, plot_sums=False, show=show)
    # Cohen d distribution
    plot_cohend_distribution(res, title=title, folder=folder, show=show)

    pass
