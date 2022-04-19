# -*- coding: utf-8 -*-
from glob import glob
from os.path import basename, dirname, exists, join, realpath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from exoorbit.orbit import Orbit
from molecfit_wrapper.molecfit import Molecfit
from tqdm import tqdm

from exoplanet_transit_snr.snr_estimate import (
    calculate_cohen_d_for_dataset,
    coadd_cross_correlation,
    run_cross_correlation,
)
from exoplanet_transit_snr.stats import gaussfit
from exoplanet_transit_snr.stellardb import StellarDb


def load_data(data_dir, load=False):
    savefilename = realpath(join(data_dir, "../medium/spectra.npz"))
    if load and exists(savefilename):
        data = np.load(savefilename, allow_pickle=True)
        fluxlist = data["flux"]
        wavelist = data["wave"]
        uncslist = data["uncs"]
        times = Time(data["time"])
        segments = data["segments"]
        return wavelist, fluxlist, uncslist, times, segments

    files_fname = join(data_dir, "cr2res_obs_nodding_extracted_combined.fits")
    files = glob(files_fname)

    fluxlist, wavelist, uncslist, times = [], [], [], []
    for f in tqdm(files):
        hdu = fits.open(f)
        header = hdu[0].header
        # The header of the datafile is unchanged during the extraction
        # i.e. the header info is ONLY the info from the first observation
        # in the sequence. Thus we add the expsoure time to get the center
        # of the four observations
        time = Time(header["MJD-OBS"], format="mjd")
        exptime = header["ESO DET SEQ1 EXPTIME"] << u.s
        time += 2 * exptime

        fluxes, waves, uncses = [], [], []

        for i in [1, 2, 3]:
            chip = f"CHIP{i}.INT1"
            data = hdu[chip].data
            for order in range(1, 10):
                spec_name = f"{order:02}_01_SPEC"
                wave_name = f"{order:02}_01_WL"
                uncs_name = f"{order:02}_01_ERR"
                try:
                    waves += [data[wave_name]]
                    fluxes += [data[spec_name]]
                    uncses += [data[uncs_name]]
                except KeyError:
                    pass

        nseg = len(fluxes)
        npoints = len(fluxes[0])
        segments = np.arange(0, (nseg + 1) * npoints, npoints)

        flux = np.concatenate(fluxes)
        wave = np.concatenate(waves)
        uncs = np.concatenate(uncses)
        fluxlist += [flux]
        wavelist += [wave]
        uncslist += [uncs]
        times += [time]
        hdu.close()

    fluxlist = np.stack(fluxlist)
    wavelist = np.stack(wavelist)
    uncslist = np.stack(uncslist)

    times = Time(times)
    sort = np.argsort(times)

    fluxlist = fluxlist[sort]
    wavelist = wavelist[sort]
    uncslist = uncslist[sort]
    times = times[sort]

    np.savez(
        savefilename,
        flux=fluxlist,
        wave=wavelist,
        uncs=uncslist,
        time=times,
        segments=segments,
    )
    return wavelist, fluxlist, uncslist, times, segments, header


# define the names of the star and planet
# as well as the datasets within the datasets folder
star, planet = "LTT1445A", "b"
datasets = "211028_LTTS1445Ab"

# Load the nominal data for this star and planet from simbad/nasa exoplanet archive
sdb = StellarDb()
star = sdb.get(star)
planet = star.planets[planet]
orbit = Orbit(star, planet)


# Define the +- range of the radial velocity points,
# and the density of the sampling
rv_range = 200
rv_step = 0.25

# Where to find the data, might need to be adjusted
data_dir = "/DATA/ESO/CRIRES+/GTO/211028_LTTS1445Ab/??"
# load the data from the fits files, returns several objects
data = load_data(data_dir, load=False)
wave, flux, uncs, times, segments, header = data

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
plt.title(f"{star.name} {planet.name}")
plt.xlabel(r"$\Delta$RV [km/s]")
plt.ylabel("CCF")
plt.show()
