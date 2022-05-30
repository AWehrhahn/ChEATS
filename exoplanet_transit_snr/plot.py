# -*- coding: utf-8 -*-
from os import makedirs
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np

from .stats import gauss


def plot_cc_overview(
    cc_data, vp_idx=None, show_line=False, title="", folder="", show=False
):
    if show_line and vp_idx is None:
        raise ValueError("vp_idx is required if line is set to True")

    plt.figure(figsize=(12, 8))
    plt.clf()

    for i in range(cc_data.shape[0]):
        plt.subplot(6, 3, i + 1)
        plt.imshow(cc_data[i], aspect="auto", origin="lower")
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


def plot_cc_overview_details(
    cc_data, vp_idx=None, show_line=False, title="", folder="", show=False
):
    if show_line and vp_idx is None:
        raise ValueError("vp_idx is required if line is set to True")

    plt.figure(figsize=(12, 8))
    n_ord = cc_data.shape[0]
    for i in range(n_ord):
        plt.clf()
        plt.imshow(cc_data[i], aspect="auto", origin="lower")
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

    plt.figure(figsize=(12, 8))
    plt.clf()

    vmin, vmax = np.nanpercentile(combined.ravel(), [5, 95])
    plt.imshow(combined, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    plt.xlabel("rv [km/s]")
    xticks = plt.xticks()[0][1:-1]
    rv_ticks = np.linspace(xticks[0], xticks[-1], len(rv_array))
    xticks_labels = np.interp(xticks, rv_ticks, rv_array)
    xticks_labels = [f"{x:.3g}" for x in xticks_labels]
    plt.xticks(xticks, labels=xticks_labels)

    if show_line:
        plt.plot(vp_idx, np.arange(len(vp_idx)), "r-.", alpha=0.5)

    if in_transit is not None:
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


def plot_vsys_kp(res, title="", folder="", show=False):
    plt.figure(figsize=(12, 8))
    plt.clf()

    ax = plt.subplot(121)
    plt.imshow(res["combined"], aspect="auto", origin="lower")
    ax.add_patch(
        plt.Rectangle(
            (res["vsys_peak"] - res["vsys_width"], res["kp_peak"] - res["kp_width"]),
            2 * res["vsys_width"],
            2 * res["kp_width"],
            fill=False,
            color="red",
        )
    )

    plt.xlabel("vsys [km/s]")
    xticks = plt.xticks()[0][1:-1]
    rv_ticks = np.linspace(xticks[0], xticks[-1], len(res["vsys"]))
    xticks_labels = np.interp(xticks, rv_ticks, res["vsys"])
    xticks_labels = [f"{x:.3g}" for x in xticks_labels]
    plt.xticks(xticks, labels=xticks_labels)

    plt.ylabel("Kp [km/s]")
    yticks = plt.yticks()[0][1:-1]
    yticks_labels = np.interp(yticks, np.arange(len(res["kp"])), res["kp"])
    yticks_labels = [f"{y:.3g}" for y in yticks_labels]
    plt.yticks(yticks, labels=yticks_labels)

    plt.subplot(222)
    plt.plot(res["vsys"], res["vsys_mean"])
    plt.vlines(
        res["vsys"][res["vsys_peak"]],
        np.min(res["vsys_mean"]),
        res["vsys_mean"][res["vsys_peak"]],
        "k",
        "--",
    )
    if res["vsys_popt"] is not None:
        plt.plot(res["vsys"], gauss(res["vsys"], *res["vsys_popt"]), "r--")
    plt.xlabel("vsys [km/s]")

    plt.subplot(224)
    plt.plot(res["kp"], res["kp_mean"])
    plt.vlines(
        res["kp"][res["kp_peak"]],
        np.min(res["kp_mean"]),
        res["kp_mean"][res["kp_peak"]],
        "k",
        "--",
    )
    if res["kp_popt"] is not None:
        plt.plot(res["kp"], gauss(res["kp"], *res["kp_popt"]), "r--")
    plt.xlabel("Kp [km/s]")

    plt.suptitle(title)
    plot_fname = f"{folder}/ccresults_vsys_kp_plot.png"
    makedirs(dirname(plot_fname), exist_ok=True)
    plt.savefig(plot_fname)
    if show:
        plt.show()


def plot_cohend_distribution(res, title="", folder="", show=False):
    plt.figure(figsize=(12, 8))
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
    plt.suptitle(f"{title}\nCohen d: {res['d']}")
    plot_fname = f"{folder}/ccresult_cohend.png"
    makedirs(dirname(plot_fname), exist_ok=True)
    plt.savefig(plot_fname)
    if show:
        plt.show()


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
    # Vsys - Kp Plot
    plot_vsys_kp(res, title=title, folder=folder, show=show)
    # Cohen d distribution
    plot_cohend_distribution(res, title=title, folder=folder, show=show)

    pass