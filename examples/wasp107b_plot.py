# -*- coding: utf-8 -*-
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from tqdm import tqdm


def plot_vsys_kp(
    ax,
    res,
    title="",
    plot_rectangle=False,
    plot_lines=True,
    xlabels=False,
    ylabels=False,
    vmin=None,
    vmax=None,
):
    ax.set_title(title)

    # vmax = np.nanmax(np.abs(res["combined"]))
    # vmin = -vmax
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
        ax.axvline(x, 0, y / n - gap, color="k", linestyle="dashed", alpha=0.5)
        ax.axvline(x, y / n + gap, 1, color="k", linestyle="dashed", alpha=0.5)
        ax.axhline(y, 0, x / m - gap, color="k", linestyle="dashed", alpha=0.5)
        ax.axhline(y, x / m + gap, 1, color="k", linestyle="dashed", alpha=0.5)

        if res["vsys_popt"] is not None and res["kp_popt"] is not None:
            x = np.interp(res["vsys_popt"][1], res["vsys"], np.arange(m)) + 0.5
            y = np.interp(res["kp_popt"][1], res["kp"], np.arange(n)) + 0.5
            ax.axvline(x, 0, y / n - gap, color="w", linestyle="dashdot", alpha=0.5)
            ax.axvline(x, y / n + gap, 1, color="w", linestyle="dashdot", alpha=0.5)
            ax.axhline(y, 0, x / m - gap, color="w", linestyle="dashdot", alpha=0.5)
            ax.axhline(y, x / m + gap, 1, color="w", linestyle="dashdot", alpha=0.5)

    xticks = np.unique(res["vsys"] // 50) * 50
    xticks_labels = np.array([f"{x:.3g}" for x in xticks])
    npoints = res["combined"].shape[1]
    xticks = interp1d(res["vsys"], np.arange(npoints), fill_value="extrapolate")(xticks)
    xticks_labels = xticks_labels[(xticks >= 0) & (xticks <= npoints)]
    xticks = xticks[(xticks >= 0) & (xticks <= npoints)]
    ax.set_xticks(xticks)
    if xlabels:
        ax.set_xticklabels(xticks_labels, fontsize="large")
        ax.set_xlabel("vsys [km/s]", fontsize="x-large")
    else:
        ax.set_xticklabels([])

    yticks = np.unique(res["kp"] // 50) * 50
    yticks_labels = np.array([f"{x:.3g}" for x in yticks])
    npoints = res["combined"].shape[0]
    yticks = interp1d(res["kp"], np.arange(npoints), fill_value="extrapolate")(yticks)
    yticks_labels = yticks_labels[(yticks >= 0) & (yticks <= npoints)]
    yticks = yticks[(yticks >= 0) & (yticks <= npoints)]
    ax.set_yticks(yticks)
    if ylabels:
        ax.set_yticklabels(yticks_labels, fontsize="large")
        ax.set_ylabel("Kp [km/s]", fontsize="x-large")
    else:
        ax.set_yticklabels([])


star = "WASP-107"
planet = "b"
n1 = 0
elem = ("H2O",)
elem_str = "_".join(elem)

ns = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# ns = [1, 2, 3, 4, 8]
vmin = -4
vmax = 4

fig = plt.figure(figsize=(16, 9))
gs = GridSpec(3, len(ns), figure=fig, wspace=0.1, hspace=0.1)

# this is one column
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[1, 0])
# ax3 = fig.add_subplot(gs[2, 0])
# ax4 = fig.add_subplot(gs[3, 0])


for i, n2 in tqdm(enumerate(ns), total=len(ns)):
    n_sec = 1
    for j, (k, selection) in tqdm(
        enumerate([(1, "IT"), (1, "OOT")]), total=2, leave=False
    ):
        folder = join(
            dirname(__file__),
            (
                f"../plots/{star}_{planet}_{n1}_{n2}_{elem_str}_real_{k}_{n_sec}_{selection}"
            ),
        )
        try:
            data = np.load(join(folder, "data.npz"), allow_pickle=True)
            res = data["res"][()]
            selection = data["selection"][()]
            rv_array = data["rv_array"]
            cc_data = data["cc_data"]
            combined = data["combined"]

            if j == 0:
                title = f"{n2}"
            else:
                title = ""

            ax = fig.add_subplot(gs[j, i])
            plot_vsys_kp(
                ax,
                res,
                title=title,
                plot_lines=True,
                xlabels=j == 1,
                ylabels=i == 0,
                vmin=vmin,
                vmax=vmax,
            )

            if j == 0:
                in_transit = res["in_transit"]
                vp_idx = np.interp(res["vp"], rv_array, np.arange(rv_array.size))
                vp_new = res["vsys"][res["vsys_peak"]] + res["kp"][
                    res["kp_peak"]
                ] * np.sin(2 * np.pi * res["phi"])
                vp_idx_new = np.interp(vp_new, rv_array, np.arange(rv_array.size))

                n_obs, n_points = combined.shape

                ax = fig.add_subplot(gs[-1, i])
                ax.imshow(combined, aspect="auto", origin="lower")
                ax.axhline(np.arange(n_obs)[in_transit][0] - 0.5, c="k", ls="--")
                ax.axhline(np.arange(n_obs)[in_transit][-1] + 0.5, c="k", ls="--")

                vp_y = np.where(in_transit, np.nan, np.arange(len(vp_idx)))
                ax.plot(vp_idx, vp_y, "k--", alpha=0.5)

                vp_y = np.where(in_transit, np.nan, np.arange(len(vp_idx_new)))
                ax.plot(vp_idx_new, vp_y, "w--", alpha=0.5)

        except FileNotFoundError:
            print(f"File not found: {folder}")
            pass

plt.suptitle(elem_str)
plt.show()
pass
