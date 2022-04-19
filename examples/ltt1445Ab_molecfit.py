# -*- coding: utf-8 -*-
import os
import re
import shutil
from glob import glob
from os.path import basename, dirname, exists, join, realpath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants
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

c_light = constants.c


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

# Setup Molecfit

mf = Molecfit(
    esorex_exec="/home/ansgar/ESO/molecfit/bin/esorex",
    cpl_dir="/home/ansgar/ESO/molecfit",
    recipe_dir="/home/ansgar/ESO/molecfit/lib/esopipes-plugins",
    output_dir=realpath(join(dirname(__file__), "data/LTT1445A")),
    column_lambda="WAVE",
    column_flux="FLUX",
    column_dflux="ERR",
    column_mask="MASK",
    wlg_to_micron=u.nm.to(u.um),
    fit_wlc=0,
    wlc_n=1,
    wlc_const=0,
    fit_continuum=1,
    continuum_n=3,
    continuum_const=1,
    # varkern=True,
    # kernfac=1,
    # kernmode=True,
)

# Where to find the data, might need to be adjusted
data_dir = "/DATA/ESO/CRIRES+/GTO/211028_LTTS1445Ab/??"
# load the data from the fits files, returns several objects

flat_fname = realpath(
    join(data_dir, "../cr2res_util_calib_calibrated_collapsed_extr1D.fits")
)
flat = fits.open(flat_fname)

files_fname = join(data_dir, "cr2res_obs_nodding_extracted_combined.fits")
files = glob(files_fname)
files = np.sort(files)

for f in tqdm(files):
    # Step 0: load data from input files
    hdu = fits.open(f)
    header = hdu[0].header
    # The header of the datafile is unchanged during the extraction
    # i.e. the header info is ONLY the info from the first observation
    # in the sequence. Thus we add the expsoure time to get the center
    # of the four observations
    time = Time(header["MJD-OBS"], format="mjd")
    exptime = header["ESO DET SEQ1 EXPTIME"] << u.s
    # time += 2 * exptime
    header["ESO DET SEQ1 EXPTIME"] *= 4

    fluxes, waves, uncses = [], [], []
    names = []

    for i in [1, 2, 3]:
        chip = f"CHIP{i}.INT1"
        data = hdu[chip].data
        for order in range(1, 10):
            spec_name = f"{order:02}_01_SPEC"
            wave_name = f"{order:02}_01_WL"
            uncs_name = f"{order:02}_01_ERR"
            try:
                blaze = flat[chip].data[spec_name]
                waves += [data[wave_name]]
                fluxes += [data[spec_name] / blaze]
                uncses += [data[uncs_name]]
                names += [f"CHIP{i}_ORDER{order}"]
            except KeyError:
                pass

    nseg = len(fluxes)
    npoints = len(fluxes[0])
    segments = np.arange(0, (nseg + 1) * npoints, npoints)

    # quick normalization
    uncs = [np.nan_to_num(u / np.nanpercentile(f, 99)) for u, f in zip(uncses, fluxes)]
    flux = [np.nan_to_num(f / np.nanpercentile(f, 99)) for f in fluxes]
    wave = [np.nan_to_num(w) for w in waves]

    # Sort wavelength
    wmean = [np.nanmean(w) for w in wave]
    sort = np.argsort(wmean)
    wave = [wave[i] for i in sort]
    flux = [flux[i] for i in sort]
    uncs = [uncs[i] for i in sort]
    if names is not None:
        names = [names[i] for i in sort]

    hdu.close()

    # Run Molecfit to get the telluric spectrum

    # Step 1:
    # Since we modifed the flux and wavelength we need to write the data to a new datafile
    i_obs = basename(dirname(f))
    mf.output_dir = realpath(join(dirname(__file__), f"data/LTT1445A/{i_obs}"))

    input_file = mf.prepare_fits(header, wave, flux, err=uncs, names=names)

    output_model = mf.molecfit_model(input_file)
    products = output_model["products"]
    atm_parameters = products["atm_parameters"]
    model_molecules = products["model_molecules"]
    best_fit_parameters = products["best_fit_parameters"]

    # atm_parameters = join(mf.output_dir, "ATM_PARAMETERS.fits")
    # model_molecules = join(mf.output_dir, "MODEL_MOLECULES.fits")
    # best_fit_parameters = join(mf.output_dir, "BEST_FIT_PARAMETERS.fits")

    # Step 2:
    output_calctrans = mf.molecfit_calctrans(
        input_file, atm_parameters, model_molecules, best_fit_parameters
    )
    telluric_data = output_calctrans["products"]["telluric_data"]
    # telluric_data = join(mf.output_dir, "TELLURIC_DATA.fits")

    # Step 3: Save it back into the CRIRES+ format
    hdu = fits.open(telluric_data)
    data = hdu[1].data

    columns = [[], [], []]
    for i in range(nseg):
        name = names[i]
        match = re.match(r"CHIP(\d+)_ORDER(\d+)", name)
        chip, order = int(match.group(1)), int(match.group(2))
        idx = data["chip"] == i + 1
        wave = data[idx]["mlambda"] * u.um.to(u.nm)
        flux = data[idx]["mtrans"]
        spec_name = f"{order:02}_01_SPEC"
        wave_name = f"{order:02}_01_WL"
        col_wave = fits.Column(wave_name, format="1D", array=wave)
        col_spec = fits.Column(spec_name, format="1D", array=flux)
        columns[chip - 1] += [col_wave, col_spec]

    primary = fits.PrimaryHDU(header=header)
    tbhdulist = [primary]
    for i in range(3):
        cols = fits.ColDefs(columns[i])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        tbhdu.header["EXTNAME"] = f"CHIP{i+1}.INT1"
        tbhdulist += [tbhdu]
    tbhdulist = fits.HDUList(tbhdulist)

    filename = join(
        dirname(__file__), f"data/LTT1445A/results/molecfit_model_{i_obs}.fits"
    )
    os.makedirs(dirname(filename), exist_ok=True)
    tbhdulist.writeto(filename, overwrite=True)
