# -*- coding: utf-8 -*-
import json
import os
import sys
from glob import glob
from os.path import dirname, exists, join, realpath

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

    files_fname = join(data_dir, "cr2res_obs_nodding_extracted_combined.fits")
    files = glob(files_fname)

    flat_data_dir = "/DATA/ESO/CRIRES+/GTO/211028_LTTS1445Ab/??"
    flat_fname = realpath(
        join(flat_data_dir, "../cr2res_util_calib_calibrated_collapsed_extr1D.fits")
    )
    flat = fits.open(flat_fname)

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
                    blaze = flat[chip].data[spec_name]
                    waves += [data[wave_name]]
                    fluxes += [data[spec_name] / blaze]
                    uncses += [data[uncs_name] / blaze]
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

    # Sort observations in time
    times = Time(times)
    sort = np.argsort(times)

    fluxlist = fluxlist[sort]
    wavelist = wavelist[sort]
    uncslist = uncslist[sort]
    times = times[sort]

    # Sort the segments by wavelength
    for i in range(len(wavelist)):
        sort = np.argsort(wavelist[i])
        wavelist[i] = wavelist[i][sort]
        fluxlist[i] = fluxlist[i][sort]
        uncslist[i] = uncslist[i][sort]

    os.makedirs(dirname(savefilename), exist_ok=True)

    np.savez(
        savefilename,
        flux=fluxlist,
        wave=wavelist,
        uncs=uncslist,
        time=times,
        segments=segments,
        header=header,
    )
    return wavelist, fluxlist, uncslist, times, segments, header


def correct_data(data):
    wave, flux, uncs, times, segments, header = data

    # Remove excess flux points
    flux[np.abs(flux) > 15000] = np.nan
    # Fit uncertainty estimate of the observations
    # https://arxiv.org/pdf/2201.04025.pdf
    # 3 components as determined by elbow plot
    # as they contribute most of the variance
    pca = PCA(3)
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


def remove_tellurics(wave, flux):
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
    n = wave.shape[0] // 2
    mflux = np.interp(wave[n], mwave, mflux, left=1, right=1)
    idx = mflux < 0.90
    flux[:, idx] = np.nan
    return flux


# define the names of the star and planet
# as well as the datasets within the datasets folder
star, planet = "WASP-107", "b"
datasets = "220310_WASP107"

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
rv_step = 1

if len(sys.argv) > 1:
    n1 = int(sys.argv[1])
    n2 = int(sys.argv[2])
else:
    n1, n2 = 0, 7

# Where to find the data, might need to be adjusted
data_dir = "/DATA/ESO/CRIRES+/GTO/220310_WASP107/1xAB_??"
# load the data from the fits files, returns several objects
data = load_data(data_dir, load=False)
wave, flux, uncs, times, segments, header = data
wave *= u.nm.to(u.AA)

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
rv_bary = rv_bary.to_value("km/s")

# Determine telluric lines
# and remove the strongest ones
flux = remove_tellurics(wave, flux)


@cache(cache_path=f"/tmp/{star.name}_{planet.name}.npz")
def ptr_spec(wave, star, planet, rv_range):
    wmin, wmax = wave.min() << u.AA, wave.max() << u.AA
    wmin *= 1 - rv_range / c_light
    wmax *= 1 + rv_range / c_light
    ptr = petitRADTRANS(
        wmin,
        wmax,
        # The line species is more important than the exact composition of the atmosphere
        # Earth-like
        # rayleigh_species=("N2", "O2"),
        # continuum_species=("N2", "O2"),
        # line_species=("H2O", "CO2"),
        # mass_fractions={"N2": 0.78, "O2": 0.2, "Ar": 0.01, "CO2": 4e-4, "H2O": 1e-3},
        # Jupiter-like
        # line_species=("H2O",),
        # mass_fractions={"H2": 0.9, "He": 0.1, "H2O": 1e-3},
        # CO2 lines
        # line_species=("CO2", ),
        # mass_fractions={"H2": 0.9, "He": 0.1, "CO2": 1e-3}
        # CO-CO2 lines
        rayleigh_species=(),
        continuum_species=(),
        # line_species=("CO", "CO2", "H2O"),
        line_species=("H2O", "CO", "CO2"),
        mass_fractions={"H2": 0.9, "He": 0.1, "CO": 1e-3, "CO2": 1e-3, "H2O": 1e-3},
    )
    ptr.init_temp_press_profile(star, planet)
    ptr_wave, ptr_flux = ptr.run()
    return ptr_wave, ptr_flux


clear_cache(ptr_spec, (wave, star, planet, rv_range))
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
clear_cache(ptr_ref, (wave[16], ptr_wave, ptr_flux, rv_range, rv_step))
ref = ptr_ref(wave[n], ptr_wave, ptr_flux, rv_range, rv_step)
ref -= np.nanmin(ref, axis=1)[:, None]
ref /= np.nanmax(ref, axis=1)[:, None]

# Run SYSREM
rp = realpath(join(dirname(__file__), "../cache"))


@cache(cache_path=f"{rp}/sysrem_{{n1}}_{{n2}}_{star.name}_{planet.name}.npz")
def run_sysrem(flux, uncs, segments, n1, n2, rv_bary, airmass):
    corrected_flux = np.zeros_like(flux)
    # n = wave.shape[0] // 2
    for low, upp in zip(segments[:-1], segments[1:]):
        # sysrem = SysremWithProjection(
        #     wave[n, low:upp], flux[:, low:upp], rv_bary, airmass, uncs
        # )
        # corrected_flux[:, low:upp], *_ = sysrem.run(n1)
        # sysrem = Sysrem(corrected_flux[:, low:upp])
        # corrected_flux[:, low:upp], *_ = sysrem.run(n2)
        sysrem = Sysrem(flux[:, low:upp], errors=uncs[:, low:upp])
        corrected_flux[:, low:upp], *_ = sysrem.run(n2)
        # resid, model = sysrem.run(n2)
        # model = model[0] + np.nansum(model[1:], axis=0)
        # corrected_flux[:, low:upp] = flux[:, low:upp] / model
    return corrected_flux


# Run Sysrem
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

combined = np.nansum(cc_data, axis=0)
res = calculate_cohen_d_for_dataset(
    combined,
    times,
    star,
    planet,
    rv_range,
    rv_step,
    kp_range=(-100, 600),
    vsys_range=(-30, 20),
)

# Save the cohen d value
# fname = f"{rp}/results/cohen_d_{star.name}_{planet.name}_{n1}_{n2}.json"
# cohend = {"cohen_d": res["d"], "sysrem_n": n2, "myrem_n": n1}
# os.makedirs(dirname(fname), exist_ok=True)
# with open(fname, "w") as f:
#     json.dump(cohend, f)

# Plot all the results
title = f"{star.name}_{planet.name}_{n1}_{n2}"
folder = f"plots/{star.name}_{planet.name}_{n1}_{n2}_real"

plot_results(rv_array, cc_data, combined, res, title=title, folder=folder)
pass
