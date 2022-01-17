# -*- coding: utf-8 -*-
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np

from exoplanet_transit_snr.snr_estimate import (
    calculate_cohen_d_for_dataset,
    init_cats,
    run_cross_correlation,
)

star, planet = "L 98-59", "c"
datasets = {
    50: "L98-59c_Earth_SNR50",
    100: "L98-59c_Earth_SNR100",
    200: "L98-59c_Earth_SNR200",
}

for snr in [50, 100, 200]:
    runner = init_cats(star, planet, datasets[snr], raw_dir="Spectrum_00")
    runner.configuration["planet_reference_spectrum"]["method"] = "petitRADTRANS"
    data = run_cross_correlation(runner, load=False)
    d = calculate_cohen_d_for_dataset(
        runner, sysrem="5", plot=False, title=f"{star} {planet} SNR{snr}"
    )
pass
# filename = "cohends.npz"
# if not exists(filename):
#     ds = {}
#     for sysrem in ["4.1"]: #"3 4 5 6 7 8 9".split():
#         for snr, dataset in datasets.items():
#             ds[f"{sysrem}_{snr}"] = calculate_cohen_d_for_dataset(
#                 runner, sysrem=str(sysrem), plot=True
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
