# -*- coding: utf-8 -*-
from os.path import dirname, exists, join

import matplotlib.pyplot as plt
import numpy as np

from exoplanet_transit_snr.snr_estimate import (
    calculate_cohen_d_for_dataset,
    load_data,
    run_cross_correlation,
)
from exoplanet_transit_snr.stellardb import StellarDb

star, planet = "WASP-107", "b"
datasets = {50: "WASP-107b_SNR50", 100: "WASP-107b_SNR100", 200: "WASP-107b_SNR200"}

sdb = StellarDb()
star = sdb.get(star)
planet = star.planets[planet]

rv_range = 200
rv_step = 0.25

for snr in [200]:
    data_dir = join(dirname(__file__), "../datasets", datasets[snr], "Spectrum_00")
    data = load_data(data_dir, load=True)
    cc_data = run_cross_correlation(
        data, rv_range=rv_range, rv_step=rv_step, load=True, data_dir=data_dir
    )
    d = calculate_cohen_d_for_dataset(
        data,
        cc_data,
        star,
        planet,
        rv_range=rv_range,
        rv_step=rv_step,
        sysrem="7",
        plot=False,
        title=f"WASP-107 b SNR{snr}",
    )

# filename = "cohends.npz"
# if not exists(filename):
#     ds = {}
#     for sysrem in "3 4 5 6 7 8 9".split():
#         for snr, dataset in datasets.items():
#             ds[f"{sysrem}_{snr}"] = calculate_cohen_d_for_dataset(
#                 star, planet, dataset, sysrem=str(sysrem), plot=False
#             )
#     np.savez("cohends.npz", **ds)
# else:
#     ds = np.load("cohends.npz")

# # Plot the results
# sysrem_snr = [s.split("_") for s in list(ds.keys())]
# sysrem = {s[0] for s in sysrem_snr}
# snr = {int(s[1]) for s in sysrem_snr}

# x = np.array(sorted(list(snr)))[:-1]
# xd = np.linspace(x.min(), x.max(), 100)
# for sysrem in ["4", "5", "6", "7", "8", "9"]:
#     y = [ds[f"{sysrem}_{snr}"] for snr in x]
#     line = plt.plot(x, y, "+", label=sysrem)
#     color = line[0].get_color()
#     yf = np.polyval(np.polyfit(x, y, 2), xd)
#     plt.plot(xd, yf, color=color)
#     maxpos = xd[np.argmax(yf)]
#     plt.vlines(maxpos, yf.min(), yf.max(), color=color)
# plt.xlabel("SNR")
# plt.ylabel("Cohen d")
# plt.legend()
# plt.show()
# pass
