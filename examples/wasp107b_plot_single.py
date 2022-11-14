# -*- coding: utf-8 -*-
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from tqdm import tqdm

from exoplanet_transit_snr.plot import plot_results
from exoplanet_transit_snr.stats import gauss


def plot_vsys_kp(
    res,
    combined,
    title="",
    plot_sums=True,
    plot_rectangle=True,
    plot_lines=True,
    plot_lines_fit=True,
):
    plt.clf()
    lw = 1

    if plot_sums:
        fig = plt.figure(figsize=(6, 7))
        gs = GridSpec(2, 30, figure=fig, wspace=0.2, hspace=0.2)
        ax = fig.add_subplot(gs[0, :-1])  # Main plot
        ax2 = fig.add_subplot(gs[1, :-1])  # plot over columns, i.e. vsys
        ax3 = fig.add_subplot(gs[0, -1])  # Plot over rows, i.e. kp
    else:
        fig, ax = plt.subplots(figsize=(6, 8))

    # in_transit = res["in_transit"]
    # vsys = res["vsys"][res["vsys_peak"]]
    # kp = res["kp"][res["kp_peak"]]
    # vp_new = vsys + kp * np.sin(2 * np.pi * res["phi"]) + res["rv_bary"]

    # combined_shift = np.copy(combined)
    # for i in range(combined.shape[0]):
    #     combined_shift[i] = np.interp(rv_array, rv_array - vp_new[i], combined[i])

    # total = np.mean(combined_shift[in_transit], axis=0)
    # res["vsys_popt"][1] = rv_array[np.argmax(total)] + vsys

    idx = np.unravel_index(np.argmax(res["combined"]), res["combined"].shape)
    total = res["combined"][idx[0]]

    if res["vsys_popt"] is not None:
        res["vsys_popt"][1] = res["vsys"][idx[1]]
    else:
        res["vsys_popt"] = [None, res["vsys"][idx[1]]]

    if res["kp_popt"] is not None:
        res["kp_popt"][1] = res["kp"][idx[0]]
    else:
        res["kp_popt"] = [None, res["kp"][idx[0]]]

    # data = np.log(1 + np.abs(res["combined"]))
    # data *= np.sign(res["combined"])
    # vmax = np.nanmax(np.abs(res["combined"]))
    # vmin = -vmax
    img = ax.imshow(
        res["combined"],
        aspect="auto",
        origin="lower",
        # interpolation="none",
        # vmin=vmin,
        # vmax=vmax,
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
        gap = 0.1
        n, m = res["combined"].shape
        x = np.interp(res["vsys_expected"], res["vsys"], np.arange(m)) + 0.5
        y = np.interp(res["kp_expected"], res["kp"], np.arange(n)) + 0.5
        ax.axvline(x, 0, y / n - gap, color="k", linestyle="dashed", linewidth=lw)
        ax.axvline(x, y / n + gap, 1, color="k", linestyle="dashed", linewidth=lw)
        ax.axhline(y, 0, x / m - gap, color="k", linestyle="dashed", linewidth=lw)
        ax.axhline(y, x / m + gap, 1, color="k", linestyle="dashed", linewidth=lw)

        if (
            res["vsys_popt"] is not None
            and res["kp_popt"] is not None
            and plot_lines_fit
        ):
            x = np.interp(res["vsys_popt"][1], res["vsys"], np.arange(m)) + 0.5
            y = np.interp(res["kp_popt"][1], res["kp"], np.arange(n)) + 0.5
            ax.axvline(x, 0, y / n - gap, color="w", linestyle="dashdot", linewidth=lw)
            ax.axvline(x, y / n + gap, 1, color="w", linestyle="dashdot", linewidth=lw)
            ax.axhline(y, 0, x / m - gap, color="w", linestyle="dashdot", linewidth=lw)
            ax.axhline(y, x / m + gap, 1, color="w", linestyle="dashdot", linewidth=lw)

    # ax.set_xlabel("vsys [km/s]", fontsize="x-large")
    xticks = np.unique(res["vsys"] // 50) * 50
    xticks_labels = np.array([f"{x:.3g}" for x in xticks])
    npoints = res["combined"].shape[1]
    xticks = interp1d(res["vsys"], np.arange(npoints), fill_value="extrapolate")(xticks)
    xticks_labels = xticks_labels[(xticks >= 0) & (xticks <= npoints)]
    xticks = xticks[(xticks >= 0) & (xticks <= npoints)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels, fontsize="large")

    ax.set_ylabel("$K_p$ [km/s]", fontsize="x-large")
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

        ax2.plot(res["vsys"], total)
        # ax2.plot(res["vsys"], res["vsys_mean"])

        if plot_lines:
            ax2.axvline(
                res["vsys_expected"], color="k", linestyle="dashed", linewidth=lw
            )
        if res["vsys_popt"] is not None and plot_lines_fit:
            # ax2.plot(res["vsys"], gauss(res["vsys"], *res["vsys_popt"]), "-.", color="tab:orange", linewidth=lw)
            ax2.axvline(
                res["vsys_popt"][1],
                color="tab:orange",
                linestyle="dashdot",
                linewidth=lw,
            )
        # ax2.set_xlabel("vsys [km/s]")
        vmin, vmax = res["vsys"][0], res["vsys"][-1]
        ax2.set_xlim(vmin - 0.5, vmax + 0.5)
        ax2.set_ylabel("SNR", fontsize="x-large")
        ax2.set_xlabel("$v_{sys}$ [km/s]", fontsize="x-large")
        # ax2.set_xticks(xticks)
        # ax2.set_xticklabels(xticks_labels, fontsize="large")

        # ax2.xaxis.tick_top()
        # ax2.xaxis.set_label_position("top")

        # Kp plot
        # ax3.plot(res["kp_mean"], res["kp"])
        # if plot_lines:
        #     ax3.axhline(
        #         res["kp_expected"], color="k", linestyle="dashed",
        #     )
        # if res["kp_popt"] is not None:
        #     ax3.plot(gauss(res["kp"], *res["kp_popt"]), res["kp"], "r-.")
        #     ax3.axhline(
        #         res["kp_popt"][1], color="r", linestyle="dashdot",
        #     )
        # # ax3.set_ylabel("Kp [km/s]")
        # vmin, vmax = res["kp"][0], res["kp"][-1]
        # ax3.set_ylim(vmin - 0.5, vmax + 0.5)
        # ax3.yaxis.tick_right()
        # ax3.yaxis.set_label_position("right")

    fig.colorbar(img, cax=ax3)
    plt.suptitle(title)
    return fig


star = "WASP-107"
planet = "b"
n1 = 0
# n2 = 9
elem = ("CH4_hargreaves",)
elem_str = "_".join(elem)

n2 = {"H2O": 11, "CO": 13, "CH4": 14, "CH4_hargreaves": 8, "CO2": 10}[elem[0]]


folder = join(
    dirname(__file__), f"../plots/{star}_{planet}_{n1}_{n2}_{elem_str}_real_1_1_IT"
)
# folder = join(dirname(__file__), f"../plots/{star}_{planet}_{n1}_{n2}_ccf")


data = np.load(join(folder, "data.npz"), allow_pickle=True)
res = data["res"][()]
selection = data["selection"][()]
rv_array = data["rv_array"]
cc_data = data["cc_data"]
combined = data["combined"]

# plot_results(rv_array, cc_data, combined, res, show=False, folder=folder)

plot_lines_fit = elem_str in ["H2O", "CO"]
fig = plot_vsys_kp(
    res,
    combined,
    plot_rectangle=False,
    plot_lines=True,
    title=elem_str,
    plot_lines_fit=plot_lines_fit,
)
fig.savefig(f"{elem_str}.png")
plt.show()
pass
