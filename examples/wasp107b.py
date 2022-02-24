# -*- coding: utf-8 -*-
from os.path import dirname, exists, join

import matplotlib.pyplot as plt
import numpy as np
from exoorbit.orbit import Orbit

from exoplanet_transit_snr.snr_estimate import (
    calculate_cohen_d_for_dataset,
    coadd_cross_correlation,
    load_data,
    run_cross_correlation,
)
from exoplanet_transit_snr.stats import gaussfit
from exoplanet_transit_snr.stellardb import StellarDb

# define the names of the star and planet
# as well as the datasets within the datasets folder
star, planet = "WASP-107", "b"
datasets = {50: "WASP-107b_SNR50", 100: "WASP-107b_SNR100", 200: "WASP-107b_SNR200"}

# Load the nominal data for this star and planet from simbad/nasa exoplanet archive
sdb = StellarDb()
star = sdb.get(star)
planet = star.planets[planet]
orbit = Orbit(star, planet)

# Define the +- range of the radial velocity points,
# and the density of the sampling
rv_range = 200
rv_step = 0.25

for snr in [200]:
    # Where to find the data, might need to be adjusted
    data_dir = join(dirname(__file__), "../datasets", datasets[snr], "Spectrum_00")
    # load the data from the fits files, returns several objects
    data = load_data(data_dir, load=True)
    wave, flux, uncs, times, segments = data

    # Run the cross correlation to the next neighbour
    cc_data, rv_array = run_cross_correlation(
        data,
        nsysrem=10,
        rv_range=rv_range,
        rv_step=rv_step,
        load=True,
        data_dir=data_dir,
    )

    # Coadd all the ccfs together
    rv = orbit.radial_velocity_planet(times)
    cc_data = cc_data["10"]
    cc_data_coadd, cc_it, cc_oot = coadd_cross_correlation(
        cc_data, rv, rv_array, times, planet, data_dir=data_dir, load=True
    )

    # Fit a gaussian
    p0 = [np.max(cc_data_coadd) - np.min(cc_data_coadd), 0, 10, np.min(cc_data_coadd)]
    gauss, pval = gaussfit(rv_array, cc_data_coadd, p0=p0)

    # Plot the results
    plt.plot(rv_array, cc_data_coadd, label="all observations")
    plt.plot(rv_array, gauss, label="best fit gaussian")
    plt.plot(rv_array, cc_it, label="in-transit")
    plt.plot(rv_array, cc_oot, label="out-of-transit")
    plt.legend()
    plt.title(f"{star.name} {planet.name} SNR{snr}")
    plt.xlabel(r"$\Delta$RV [km/s]")
    plt.ylabel("CCF")
    plt.show()
