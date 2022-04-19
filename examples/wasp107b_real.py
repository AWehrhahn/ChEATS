# -*- coding: utf-8 -*-
import os
from glob import glob
from os.path import basename, dirname, exists, join, realpath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation
from astropy.io import fits
from astropy.time import Time
from cache_decorator import Cache as cache
from exoorbit.orbit import Orbit
from molecfit_wrapper.molecfit import Molecfit
from scipy.constants import speed_of_light
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from sklearn import gaussian_process
from tqdm import tqdm

from exoplanet_transit_snr.petitradtrans import petitRADTRANS
from exoplanet_transit_snr.snr_estimate import (
    coadd_cross_correlation,
    cross_correlation_reference,
    run_cross_correlation,
    run_cross_correlation_ptr,
)
from exoplanet_transit_snr.stats import gaussfit
from exoplanet_transit_snr.stellardb import StellarDb
from exoplanet_transit_snr.sysrem import Sysrem, SysremWithProjection

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
rv_step = 0.25

# Where to find the data, might need to be adjusted
data_dir = "/DATA/ESO/CRIRES+/GTO/220310_WASP107/1xAB_??"
# load the data from the fits files, returns several objects
data = load_data(data_dir, load=False)
wave, flux, uncs, times, segments, header = data
wave *= u.nm.to(u.AA)

flux = flux[:, :-2048]
wave = wave[:, :-2048]
uncs = uncs[:, :-2048]
segments = segments[:-1]
for low, upp in zip(segments[:-1], segments[1:]):
    flux[:, low:upp] /= np.nanmedian(flux[:, low:upp])
    flux[:, low : low + 20] = np.nan
    flux[:, upp - 20 : upp] = np.nan
    # for i in range(len(flux)):
    #     blurred = median_filter(flux[i, low:upp], 5) - flux[i, low:upp]
    #     flux[i, low:upp][blurred < np.nanstd(blurred) * 5] = np.nan

for i in range(len(wave)):
    sort = np.argsort(wave[i])
    wave[i] = wave[i][sort]
    flux[i] = flux[i][sort]
    uncs[i] = uncs[i][sort]

spec = np.nanmedian(flux, axis=0)
std = np.nanmedian(np.abs(np.nanmedian(flux - spec) - (flux - spec)))
for i in range(len(flux)):
    r = np.nanmedian(flux[i] / spec)
    idx = np.abs(flux[i] - r * spec) > 3 * std
    flux[i, idx] = r * spec[idx]

flux = np.nan_to_num(flux, nan=1, posinf=1, neginf=1, copy=False)
uncs = np.nan_to_num(uncs, nan=1, posinf=1, neginf=1, copy=False)
n = flux.shape[0] // 2
uncs = np.clip(uncs, 1, None)

# Correct for airmass and seeing
altaz = star.coordinates.transform_to(AltAz(obstime=times, location=telescope))
airmass = altaz.secz.value

mean = np.nanmean(flux, axis=1)
c0 = np.polyfit(airmass, mean, 1)
# # res = least_squares(lambda c: np.polyval(c, airmass) - mean, c0, loss="soft_l1")
ratio = np.polyval(c0, airmass)
flux /= (mean * ratio)[:, None]

# estimate tellurics and spectrum contribution
tell, spec = np.polyfit(airmass - 1, flux, 1)
tell += 1

data = wave, flux, uncs, times, segments, header


@cache(cache_path=f"/tmp/{star.name}_{planet.name}.npz")
def ptr_spec(wave, star, planet, rv_range):
    wmin, wmax = wave.min() << u.AA, wave.max() << u.AA
    wmin *= 1 - rv_range / c_light
    wmax *= 1 + rv_range / c_light
    ptr = petitRADTRANS(wmin, wmax)
    ptr.init_temp_press_profile(star, planet)
    ptr_wave, ptr_flux = ptr.run()
    return ptr_wave, ptr_flux


# clear_cache(ptr_spec, (wave, star, planet, rv_range))
ptr_wave, ptr_flux = ptr_spec(wave, star, planet, rv_range)
if hasattr(ptr_wave, "unit"):
    ptr_wave = ptr_wave.to_value(u.um)
ptr_wave *= u.um.to(u.AA)


@cache(cache_path=f"/tmp/ccfref_{star.name}_{planet.name}.npz")
def ptr_ref(wave, ptr_wave, ptr_flux, rv_range, rv_step):
    ref = cross_correlation_reference(
        wave, ptr_wave, ptr_flux, rv_range=rv_range, rv_step=rv_step
    )
    return ref


# clear_cache(ptr_ref, (wave[16], ptr_wave, ptr_flux, rv_range, rv_step))
ref = ptr_ref(wave[16], ptr_wave, ptr_flux, rv_range, rv_step)
ref -= np.nanmin(ref, axis=1)[:, None]
ref /= np.nanmax(ref, axis=1)[:, None]

# Barycentric correction
rv_bary = -star.coordinates.radial_velocity_correction(
    obstime=times, location=telescope
)
rv_bary -= np.mean(rv_bary)
rv_bary = rv_bary.to_value("km/s")


# Run SYSREM
@cache(cache_path=f"/tmp/sysrem_{star.name}_{planet.name}.npz")
def run_sysrem(flux, uncs, segments, n1, n2, rv_bary, airmass):
    corrected_flux = np.zeros_like(flux)
    for low, upp in zip(segments[:-1], segments[1:]):
        sysrem = SysremWithProjection(
            wave[n, low:upp], flux[:, low:upp], rv_bary, airmass, uncs[:, low:upp]
        )
        corrected_flux[:, low:upp], *_ = sysrem.run(n1)
        sysrem = Sysrem(corrected_flux[:, low:upp])
        corrected_flux[:, low:upp], *_ = sysrem.run(n2)
    return corrected_flux


clear_cache(run_sysrem, (flux, uncs, segments, 1, 15, rv_bary, airmass))
corrected_flux = run_sysrem(flux, uncs, segments, 1, 15, rv_bary, airmass)

# Normalize by the standard deviation in this wavelength column
corrected_flux -= np.nanmean(corrected_flux, axis=0)
std = np.nanstd(corrected_flux, axis=0)
std[std == 0] = 1
corrected_flux /= std

# Run the cross correlation to the next neighbour
cache_suffix = f"_{star.name}_{planet.name}".lower().replace(" ", "_")
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

cc_data -= np.nanmean(cc_data, axis=(1, 2))[:, None, None]
cc_data /= np.nanstd(cc_data, axis=(1, 2))[:, None, None]

# Plot the cross correlation
for i in range(17):
    plt.subplot(6, 3, i + 1)
    plt.imshow(cc_data[i], aspect="auto")
plt.show()

# Plot the results
rv_points = int(2 * rv_range / rv_step + 1)
phi = (times - planet.time_of_transit) / planet.period
phi = phi.to_value(1)
phi = phi % 1
vsys = star.radial_velocity.to_value("km/s")
kp = Orbit(star, planet).radial_velocity_semiamplitude_planet().to_value("km/s")
vp = vsys - kp * np.sin(2 * np.pi * phi)
vp_idx = np.interp(vp, rv_array, np.arange(rv_points))

plt.imshow(np.nansum(cc_data, axis=0), aspect="auto")
plt.plot(vp_idx, np.arange(len(vp_idx)), "r-.", alpha=0.5)
plt.show()


# # Coadd all the ccfs together
# rv = orbit.radial_velocity_planet(times)
# cc_data_coadd, cc_it, cc_oot = coadd_cross_correlation(
#     cc_data, rv, rv_array, times, planet, data_dir=data_dir, load=True
# )

# # Fit a gaussian
# p0 = [np.max(cc_data_coadd) - np.min(cc_data_coadd), 0, 10, np.min(cc_data_coadd)]
# gauss, pval = gaussfit(rv_array, cc_data_coadd, p0=p0)

# # Plot the results
# plt.plot(rv_array, cc_data_coadd, label="all observations")
# plt.plot(rv_array, gauss, label="best fit gaussian")
# plt.plot(rv_array, cc_it, label="in-transit")
# plt.plot(rv_array, cc_oot, label="out-of-transit")
# plt.legend()
# plt.title(f"{star.name} {planet.name}")
# plt.xlabel(r"$\Delta$RV [km/s]")
# plt.ylabel("CCF")
# plt.show()
