# -*- coding: utf-8 -*-
import json
import os
import sys
from glob import glob
from os.path import basename, dirname, exists, join, realpath

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation
from astropy.io import fits
from astropy.time import Time
from cache_decorator import Cache as cache
from exoorbit.orbit import Orbit
from scipy.constants import speed_of_light
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from tqdm import tqdm

from exoplanet_transit_snr.petitradtrans import petitRADTRANS
from exoplanet_transit_snr.plot import plot_results
from exoplanet_transit_snr.snr_estimate import (
    calculate_cohen_d_for_dataset,
    cross_correlation_reference,
    run_cross_correlation_ptr,
)
from exoplanet_transit_snr.stellardb import StellarDb
from exoplanet_transit_snr.sysrem import Sysrem

c_light = speed_of_light * 1e-3
rp = realpath(join(dirname(__file__), ".."))


def clear_cache(func, args=None, kwargs=None):
    args = args if args is not None else ()
    kwargs = kwargs if kwargs is not None else {}
    cache_path = func.__cacher_instance._get_formatted_path(args, kwargs)
    try:
        os.remove(cache_path)
    except FileNotFoundError:
        pass


def load_data(data_dir, load=False):
    savefilename = realpath(join(data_dir, "../medium/spectra.npz"))
    if load and exists(savefilename):
        data = np.load(savefilename, allow_pickle=True)
        fluxlist = data["flux"]
        wavelist = data["wave"]
        uncslist = data["uncs"]
        times = Time(data["time"])
        segments = data["segments"]
        header = data["header"]
        return wavelist, fluxlist, uncslist, times, segments, header

    files_fname = join(data_dir, "*.fits")
    files = glob(files_fname)

    add_fname = join(data_dir, "*.csv")
    add_fname = glob(add_fname)[0]
    add = pd.read_csv(add_fname)

    # flat_data_dir = "/DATA/ESO/CRIRES+/GTO/211028_LTTS1445Ab/??"
    # flat_fname = realpath(
    #     join(flat_data_dir, "../cr2res_util_calib_calibrated_collapsed_extr1D.fits")
    # )
    # flat = fits.open(flat_fname)

    fluxlist, wavelist, uncslist, times = [], [], [], []
    for f in tqdm(files):
        hdu = fits.open(f)
        header = hdu[0].header
        # The header of the datafile is unchanged during the extraction
        # i.e. the header info is ONLY the info from the first observation
        # in the sequence. Thus we add the expsoure time to get the center
        # of the four observations
        i_obs = int(basename(f).split("_")[1].split(".")[0])
        time = Time(add["time"][i_obs], format="jd")

        wave_data = hdu[1].data
        flux_data = hdu[2].data

        wave = wave_data.ravel()
        flux = flux_data.ravel()

        nseg, npoints = np.prod(flux_data.shape[:2]), flux_data.shape[-1]
        segments = np.arange(0, (nseg + 1) * npoints, npoints)

        fluxlist += [flux]
        wavelist += [wave]
        times += [time]
        hdu.close()

    fluxlist = np.stack(fluxlist)
    wavelist = np.stack(wavelist)

    times = Time(times)
    sort = np.argsort(times)

    fluxlist = fluxlist[sort]
    wavelist = wavelist[sort]
    times = times[sort]

    os.makedirs(dirname(savefilename), exist_ok=True)

    np.savez(
        savefilename,
        flux=fluxlist,
        wave=wavelist,
        time=times,
        segments=segments,
        header=header,
    )
    return wavelist, fluxlist, None, times, segments, header


def correct_data(data):
    wave, flux, uncs, times, segments, header = data

    # Remove excess flux points
    # flux[np.abs(flux) > 15000] = np.nan
    uncs = np.ones_like(flux)
    # Fit uncertainty estimate of the observations
    # https://arxiv.org/pdf/2201.04025.pdf
    # 3 components as determined by elbow plot
    # as they contribute most of the variance
    pca = PCA(3, svd_solver="full")
    for low, upp in zip(segments[:-1], segments[1:]):
        while True:
            # Use PCA model of the observations
            param = pca.fit_transform(np.nan_to_num(flux[:, low:upp], nan=0))
            model = pca.inverse_transform(param)
            resid = flux[:, low:upp] - model

            # Clean up outliers
            std = np.nanstd(resid)
            idx = np.abs(resid) > 5 * std
            resid[idx] = np.nan
            flux[:, low:upp][idx] = model[idx]

            if not np.any(idx):
                break

        # Make a new model
        param = pca.fit_transform(np.nan_to_num(flux[:, low:upp], nan=0))
        model = pca.inverse_transform(param)
        resid = flux[:, low:upp] - model

        # Fit the expected uncertainties
        def func(x):
            a, b = x[0], x[1]
            sigma = np.sqrt(a * np.abs(flux[:, low:upp]) + b)
            logL = -0.5 * np.nansum((resid / sigma) ** 2) - np.nansum(np.log(sigma))
            return -logL

        res = minimize(
            func, x0=[1, 1], bounds=[(0, None), (1e-16, None)], method="Nelder-Mead"
        )
        a, b = res.x

        uncs[:, low:upp] = np.sqrt(a * np.abs(flux[:, low:upp]) + b)

    # Correct for the large scale variations
    for low, upp in zip(segments[:-1], segments[1:]):
        spec = np.nanmedian(flux[:, low:upp], axis=0)
        mod = flux[:, low:upp] / spec
        mod = np.nan_to_num(mod, nan=1)
        for i in range(flux.shape[0]):
            mod[i] = median_filter(mod[i], 501, mode="constant", cval=1)
            mod[i] = gaussian_filter1d(mod[i], 100)
        flux[:, low:upp] /= mod
        uncs[:, low:upp] /= mod

    # area = orbit.stellar_surface_covered_by_planet(times).to_value(1)
    for low, upp in zip(segments[:-1], segments[1:]):
        flux[:, low : low + 20] = np.nan
        flux[:, upp - 20 : upp] = np.nan
        uncs[:, low : low + 20] = np.nan
        uncs[:, upp - 20 : upp] = np.nan
        spec = np.nanpercentile(flux[:, low:upp], 99, axis=1)[:, None]
        flux[:, low:upp] /= spec
        uncs[:, low:upp] /= spec
        # flux[:, low:upp] *= (1 - area)[:, None]
        # spec = np.nanmedian(flux[:, low:upp], axis=0)
        # ratio = np.nanmedian(flux[:, low:upp] / spec, axis=1)
        # model = spec[None, :] * ratio[:, None]
        # flux[:, low:upp] /= np.nanpercentile(spec, 95)
        # flux[:, low:upp] /= np.nanpercentile(flux[:, low:upp], 95, axis=1)[:, None]

    # Find outliers by comparing with the median observation
    # correct for the scaling between observations with the factor r
    for low, upp in zip(segments[:-1], segments[1:]):
        spec = np.nanmedian(flux[:, low:upp], axis=0)
        ratio = np.nanmedian(flux[:, low:upp] / spec, axis=1)
        diff = flux[:, low:upp] - spec * ratio[:, None]
        std = np.nanmedian(np.abs(np.nanmedian(diff, axis=0) - diff), axis=0)
        std = np.clip(std, 0.01, 0.1, out=std)
        idx = np.abs(diff) > 10 * std
        flux[:, low:upp][idx] = np.nan  # (ratio[:, None] * spec)[idx]
        uncs[:, low:upp][idx] = np.nan  # 1

    # flux = np.nan_to_num(flux, nan=1, posinf=1, neginf=1, copy=False)
    # uncs = np.nan_to_num(uncs, nan=1, posinf=1, neginf=1, copy=False)
    uncs = np.clip(uncs, 0, None)

    # divide by the median spectrum
    spec = np.nanmedian(flux, axis=0)
    flux /= spec[None, :]

    # Correct for airmass
    # mean = np.nanmean(flux, axis=1)
    # c0 = np.polyfit(airmass, mean, 1)
    # # # res = least_squares(lambda c: np.polyval(c, airmass) - mean, c0, loss="soft_l1")
    # ratio = np.polyval(c0, airmass)
    # flux /= (mean * ratio)[:, None]

    data = wave, flux, uncs, times, segments, header
    return data


# Setup the parameters for the star and planet
# --------------------------------------------
# define the names of the star and planet
# as well as the datasets within the datasets folder
star, planet = "WASP-107", "b"
datasets = "WASP-107b_SNR100"

# Load the nominal data for this star and planet from simbad/nasa exoplanet archive
sdb = StellarDb()
# sdb.refresh(star)
star = sdb.get(star)
planet = star.planets[planet]
# Temporary Fix, while the units in the NASA Exoplanet Archive are in the wrong units
planet.transit_duration = planet.transit_duration.to_value(u.day) * u.hour

orbit = Orbit(star, planet)
telescope = EarthLocation.of_site("Paranal")

# Define the +- range of the radial velocity points,
# and the density of the sampling
rv_range = 200
rv_step = 0.25

if len(sys.argv) > 1:
    n1 = int(sys.argv[1])
    n2 = int(sys.argv[2])
else:
    n1, n2 = 0, 7

# Load the observation data, and correct it
# ------------------------------------------
# Where to find the data, might need to be adjusted
data_dir = join(dirname(__file__), "../datasets", datasets, "Spectrum_00")
# load the data from the fits files, returns several objects
data = load_data(data_dir, load=False)
# Correct outliers, basic continuum normalization
data = correct_data(data)
wave, flux, uncs, times, segments, header = data

# Calculate airmass
altaz = star.coordinates.transform_to(AltAz(obstime=times, location=telescope))
airmass = altaz.secz.value

# Barycentric correction
rv_bary = -star.coordinates.radial_velocity_correction(
    obstime=times, location=telescope
)
rv_bary -= np.mean(rv_bary)
rv_bary = -rv_bary.to_value("km/s")


# Determine telluric lines
# and remove the strongest ones
fname = join(dirname(__file__), "../psg_trn.txt")
df = pd.read_table(
    fname,
    sep=r"\s+",
    comment="#",
    header=None,
    names=[
        "wave",
        "total",
        "H2O",
        "CO2",
        "O3",
        "N2O",
        "CO",
        "CH4",
        "O2",
        "N2",
        "Rayleigh",
        "CIA",
    ],
)
mwave = df["wave"] * u.nm.to(u.AA)
mflux = df["total"]
mflux = np.interp(wave[wave.shape[0] // 2], mwave, mflux, left=1, right=1)
idx = mflux < 0.9
flux[:, idx] = np.nan

# Get the expected planet spectrum reference spectrum from petitRADTRANS
# ----------------------------------------------------------------------


@cache(cache_path=f"{rp}/{star.name}_{planet.name}_petitRADTRANS.npz")
def ptr_spec(wave, star, planet, rv_range):
    wmin, wmax = wave.min() << u.AA, wave.max() << u.AA
    wmin *= 1 - rv_range / c_light
    wmax *= 1 + rv_range / c_light
    ptr = petitRADTRANS(
        wmin,
        wmax,
        # line_species=("H2O",),
        # mass_fractions={"H2": 0.74, "He": 0.24, "H2O": 0.02},
        rayleigh_species=(),
        continuum_species=(),
        line_species=("CO", "CO2", "H2O"),
        mass_fractions={"H2": 0.9, "He": 0.1, "CO": 1e-3, "CO2": 1e-3, "H2O": 1e-3},
    )
    ptr.init_temp_press_profile(star, planet)
    ptr_wave, ptr_flux = ptr.run()
    return ptr_wave, ptr_flux


# Don't clear the cache unless you have petitRadtrans installed and working
# clear_cache(ptr_spec, (wave, star, planet, rv_range))
ptr_wave, ptr_flux = ptr_spec(wave, star, planet, rv_range)
if hasattr(ptr_wave, "unit"):
    # ptr_wave gets saved without the quantity information by the cache
    ptr_wave = ptr_wave.to_value(u.um)
ptr_wave *= u.um.to(u.AA)


@cache(cache_path=f"/tmp/ccfref_{star.name}_{planet.name}.npz")
def ptr_ref(wave, ptr_wave, ptr_flux, rv_range, rv_step):
    ref = cross_correlation_reference(
        wave, ptr_wave, ptr_flux, rv_range=rv_range, rv_step=rv_step
    )
    return ref


n = wave.shape[0] // 2
clear_cache(ptr_ref, (wave[n], ptr_wave, ptr_flux, rv_range, rv_step))
ref = ptr_ref(wave[n], ptr_wave, ptr_flux, rv_range, rv_step)
# Normalize each segment of the cc reference
for low, upp in zip(segments[:-1], segments[1:]):
    ref[:, low:upp] -= np.nanmin(ref[:, low:upp], axis=1)[:, None]
    ref[:, low:upp] /= np.nanmax(ref[:, low:upp], axis=1)[:, None]

# Run SYSREM on the observations
# ------------------------------


@cache(cache_path=f"{rp}/{star.name}_{planet.name}_sysrem_{{n1}}_{{n2}}.npz")
def run_sysrem(flux, uncs, segments, n1, n2, rv_bary, airmass):
    corrected_flux = np.zeros_like(flux)
    # n = wave.shape[0] // 2
    for low, upp in zip(segments[:-1], segments[1:]):
        # sysrem = SysremWithProjection(
        #     wave[n, low:upp], flux[:, low:upp], rv_bary, airmass, None
        # )
        # corrected_flux[:, low:upp], *_ = sysrem.run(n1)
        sysrem = Sysrem(flux[:, low:upp], errors=uncs[:, low:upp])
        corrected_flux[:, low:upp], *_ = sysrem.run(n2)
    return corrected_flux


# the first number is the number of sysrem iterations accounting for the barycentric velocity
# the second is for regular sysrem iterations
clear_cache(run_sysrem, (flux, uncs, segments, n1, n2, rv_bary, airmass))
corrected_flux = run_sysrem(flux, uncs, segments, n1, n2, rv_bary, airmass)

# Normalize by the standard deviation in this wavelength column
corrected_flux -= np.nanmean(corrected_flux, axis=0)
std = np.nanstd(corrected_flux, axis=0)
std[std == 0] = 1
corrected_flux /= std

# Run the cross correlation between the sysrem residuals and the expected planet spectrum
# ---------------------------------------------------------------------------------------
cache_suffix = f"_{n1}_{n2}_{star.name}_{planet.name}".lower().replace(" ", "_")
cc_data, rv_array = run_cross_correlation_ptr(
    corrected_flux,
    ref,
    segments,
    rv_range=rv_range,
    rv_step=rv_step,
    load=False,
    data_dir=data_dir,
    cache_suffix=cache_suffix,
)

# Normalize the cross correlation of each segment
for i in range(len(cc_data)):
    cc_data[i] -= np.nanmean(cc_data[i], axis=1)[:, None]
    cc_data[i] /= np.nanstd(cc_data[i], axis=1)[:, None]

# Combine all data, and determine the cohen d value
# -------------------------------------------------
combined = (
    np.nansum(cc_data[1:5])
    + np.nansum(cc_data[6:11], axis=0)
    + np.nansum(cc_data[12:17], axis=0)
)
res = calculate_cohen_d_for_dataset(
    combined, times, star, planet, rv_range, rv_step, fix_kp_vsys=True
)


# Save the cohen d value
fname = f"{rp}/results/cohen_d_{star.name}_{planet.name}_{n1}_{n2}.json"
cohend = {"cohen_d": res["d"], "sysrem_n": n2, "myrem_n": n1}
os.makedirs(dirname(fname), exist_ok=True)
with open(fname, "w") as f:
    json.dump(cohend, f)

# Plot the cross correlation results
# ----------------------------------
# by default the plots will be saved to disk and actively displayed
# set show to False if you do not want to see all the plots
title = f"{star.name}_{planet.name}_{n1}_{n2}"
folder = f"{rp}/plots/{star.name}_{planet.name}_{n1}_{n2}_ccf"
plot_results(rv_array, cc_data, combined, res, title, folder, show=False)

pass
