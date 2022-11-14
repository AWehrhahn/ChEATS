# -*- coding: utf-8 -*-
import os
import sys
from glob import glob
from os.path import basename, dirname, join, realpath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.time import Time
from exoorbit.orbit import Orbit
from pysme.continuum_and_radial_velocity import determine_radial_velocity
from pysme.iliffe_vector import Iliffe_vector
from pysme.sme import SME_Structure
from scipy.constants import speed_of_light
from scipy.ndimage import binary_dilation, gaussian_filter1d, median_filter
from tqdm import tqdm

from exoplanet_transit_snr.petitradtrans import petitRADTRANS
from exoplanet_transit_snr.snr_estimate import (
    cross_correlation_reference,
    kp_vsys_combine,
    run_cross_correlation_ptr,
)
from exoplanet_transit_snr.stats import nanmad
from exoplanet_transit_snr.stellardb import StellarDb
from exoplanet_transit_snr.sysrem import Sysrem

c_light = speed_of_light * 1e-3
path = realpath(join(dirname(__file__), "../plots"))


def load_data(data_dir):
    files_fname = join(data_dir, "cr2res_obs_nodding_extracted[AB].fits")
    files = glob(files_fname)

    flat_data_dir = "/DATA/ESO/CRIRES+/GTO/211028_LTTS1445Ab/??"
    flat_fname = realpath(
        join(flat_data_dir, "../cr2res_util_calib_calibrated_collapsed_extr1D.fits")
    )
    flat = fits.open(flat_fname)

    fluxlist, wavelist, uncslist, times = [], [], [], []
    nodding = []
    for f in tqdm(files):
        hdu = fits.open(f)
        header = hdu[0].header
        # The header of the datafile is unchanged during the extraction
        # i.e. the header info is ONLY the info from the first observation
        # in the sequence. Thus we add the expsoure time to get the center
        # of the four observations

        # Get the actual times of the observations from the original fits files
        sof_fname = join(dirname(f), basename(dirname(f)) + ".sof")
        with open(sof_fname) as sof_file:
            sof = sof_file.readlines()

        sof = [s.split() for s in sof]
        sof = [s[0] for s in sof if s[1] == "OBS_NODDING_OTHER"]
        sof = [join(dirname(data_dir), basename(s)) for s in sof]
        headers = [fits.getheader(s) for s in sof]
        nodpos = [h["ESO SEQ NODPOS"] for h in headers]
        header = [h for i, h in enumerate(headers) if nodpos[i] == f[-6]][0]

        time = Time(header["MJD-OBS"], format="mjd")
        exptime = np.asarray(header["ESO DET SEQ1 EXPTIME"]) << u.s
        time += exptime / 2
        nodding += [f[-6]]
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
    nodding = np.asarray(nodding)
    # Sort observations in time
    times = Time(times)
    sort = np.argsort(times)

    fluxlist = fluxlist[sort]
    wavelist = wavelist[sort]
    uncslist = uncslist[sort]
    times = times[sort]
    nodding = nodding[sort]

    # Sort the segments by wavelength
    for i in range(len(wavelist)):
        sort = np.argsort(wavelist[i])
        wavelist[i] = wavelist[i][sort]
        fluxlist[i] = fluxlist[i][sort]
        uncslist[i] = uncslist[i][sort]

    # Convert to AA
    wavelist *= u.nm.to(u.AA)

    return wavelist, fluxlist, uncslist, times, segments, header, nodding


def determine_rv_offset(wave_A, spec_A, wave_B, spec_B, segments):
    sme = SME_Structure()
    sme.wave = [wave_A[low:upp] for low, upp in enum(segments)]
    sme.spec = [spec_A[low:upp] for low, upp in enum(segments)]
    syn_wave_B = Iliffe_vector([wave_B[low:upp] for low, upp in enum(segments)])
    syn_spec_B = Iliffe_vector([spec_B[low:upp] for low, upp in enum(segments)])
    sme.vrad_flag = "whole"
    vrad = determine_radial_velocity(sme, syn_wave_B, syn_spec_B, range(18), whole=True)
    return vrad


def remove_outliers(flux, unc=None, copy=False):
    if copy:
        flux = np.copy(flux)
        if unc is not None:
            unc = np.copy(unc)

    model = median_filter(flux, (5, 3))
    resid = flux - model
    std = nanmad(resid)
    flux[flux > model + 5 * std] = np.nan
    flux[flux < model - 10 * std] = np.nan

    spec = np.nanmedian(flux, axis=0)
    std = nanmad(flux - spec, axis=0)
    flux[:, std > np.nanmedian(std) * 5] = np.nan

    mask = np.isfinite(flux.ravel())
    idx = np.where(~mask)[0]
    x = np.arange(flux.size)
    y = flux.ravel()
    flux.ravel()[idx] = np.interp(idx, x[mask], y[mask])
    if unc is not None:
        y = unc.ravel()
        unc.ravel()[idx] = np.interp(idx, x[mask], y[mask])
    return flux, unc


def correct_data(data):
    wave, flux, uncs, times, segments, header, nodding = data

    # Correct for the large scale variations
    for low, upp in tqdm(enum(segments), total=len(segments) - 1, leave=False):
        spec = np.nanmedian(flux[:, low:upp], axis=0)
        mod = flux[:, low:upp] / spec
        mod = np.nan_to_num(mod, nan=1)
        for i in range(flux.shape[0]):
            mod[i] = median_filter(mod[i], 501, mode="constant", cval=1)
            mod[i] = gaussian_filter1d(mod[i], 100)
        flux[:, low:upp] /= mod
        uncs[:, low:upp] /= mod

    # Remove the 20 edge pixels as they are bad
    flux[:, 12186:12269] = np.nan
    for low, upp in enum(segments):
        flux[:, low : low + 20] = np.nan
        flux[:, upp - 20 : upp] = np.nan
        uncs[:, low : low + 20] = np.nan
        uncs[:, upp - 20 : upp] = np.nan

    # divide by the median spectrum
    spec = np.nanmedian(flux, axis=0)
    flux /= spec[None, :]
    uncs /= spec[None, :]

    uncs = np.clip(uncs, 0, None)

    data = wave, flux, uncs, times, segments, header, nodding
    return data


def remove_tellurics(wave, flux, rv_bary=0):
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
    mflux = np.interp(wave[n], mwave * (1 - rv_bary / c_light), mflux, left=1, right=1)
    idx = mflux < 0.90
    idx = binary_dilation(idx, iterations=10)
    flux[:, idx] = np.nan
    return flux


def ptr_spec(wave, star, planet, rv_range, elem=("H2O", "CO", "CO2")):
    wmin, wmax = wave.min() << u.AA, wave.max() << u.AA
    wmin *= 1 - rv_range / c_light
    wmax *= 1 + rv_range / c_light

    mass_fractions = {
        "H2": 0.74,
        "He": 0.24,
        "TiO": 1e-3,
    }
    for el in elem:
        if el not in ["H2", "He", "TiO"]:
            mass_fractions[el] = 1e-3

    ptr = petitRADTRANS(
        wmin,
        wmax,
        line_species=("TiO",),
        mass_fractions=mass_fractions,
    )
    ptr.init_temp_press_profile(star, planet)
    cont_wave, cont_flux = ptr.run()

    ptr = petitRADTRANS(
        wmin,
        wmax,
        line_species=elem,
        mass_fractions=mass_fractions,
    )
    ptr.init_temp_press_profile(star, planet)
    ptr_wave, ptr_flux = ptr.run()

    # Assuming that cont_wave == ptr_wave
    ptr_flux /= cont_flux

    return ptr_wave, ptr_flux


def ptr_ref(wave, ptr_wave, ptr_flux, rv_range, rv_step):
    ref = cross_correlation_reference(
        wave, ptr_wave, ptr_flux, rv_range=rv_range, rv_step=rv_step
    )
    return ref


def run_sysrem(flux, uncs, segments, n2):
    corrected_flux = np.zeros_like(flux)
    for low, upp in enum(segments):
        sysrem = Sysrem(flux[:, low:upp], errors=uncs[:, low:upp])
        corrected_flux[:, low:upp], *_ = sysrem.run(n2)
    return corrected_flux


def enum(segments):
    for i in range(len(segments) - 1):
        yield segments[i], segments[i + 1]


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
planet.equilibrium_temperature = lambda x, y: 736 * u.K

# based on alternative in https://iopscience.iop.org/article/10.3847/2041-8213/aabfce/pdf
# planet.intrinsic_temperature = lambda x, y: 500 * u.K

orbit = Orbit(star, planet)
telescope = EarthLocation.of_site("Paranal")

# Define the +- range of the radial velocity points,
# and the density of the sampling
rv_range = 200
rv_step = 0.25

if len(sys.argv) > 1:
    elem = (sys.argv[1],)
    n1 = 0
    n2 = int(sys.argv[2])
else:
    n1, n2 = 0, 15
    elem = ("CO",)
elem_str = "_".join(elem)

# Where to find the data, might need to be adjusted
data_dir = "/DATA/ESO/CRIRES+/GTO/220310_WASP107/1xAB_??"
# load the data from the fits files, returns several objects
data = load_data(data_dir)
wave, flux, uncs, times, segments, header, nodding = data

# Remove obvious excess flux points
flux[flux < 0] = np.nan
flux[flux > np.nanpercentile(flux.ravel(), 99) * 1.5] = np.nan

# Normalize by the 95th percentile of each segment
for low, upp in enum(segments):
    flux[:, low:upp] /= np.nanpercentile(flux[:, low:upp], 95, axis=1)[:, None]

# Split the two nodding positions
wave_A = wave[nodding == "A"][0]
wave_B = wave[nodding == "B"][0]
flux_A = flux[nodding == "A"]
flux_B = flux[nodding == "B"]
uncs_A = uncs[nodding == "A"]
uncs_B = uncs[nodding == "B"]

# Manual masking
flux_A[:, 22956:22960] = np.nan
flux_A[:, 12186:12269] = np.nan
flux_B[:, 12186:12269] = np.nan

# Automatic outlier removal
flux_A, uncs_A = remove_outliers(flux_A, uncs_A)
flux_B, uncs_B = remove_outliers(flux_B, uncs_B)

# Recombine A and B nodding positions
flux[nodding == "A"] = flux_A
flux[nodding == "B"] = flux_B
uncs[nodding == "A"] = uncs_A
uncs[nodding == "B"] = uncs_B

# Determine the radial velocity offset between A and B nodding
spec_A = np.nanmedian(flux_A, axis=0)
spec_B = np.nanmedian(flux_B, axis=0)
vrad = determine_rv_offset(wave_A, spec_A, wave_B, spec_B, segments)

# Barycentric correction
# shift everything into the restframe of the star
rv_bary = -star.coordinates.radial_velocity_correction(
    obstime=times, location=telescope
).to_value("km/s")

# Correct for barycentric and rv offset between nodding positions
# Interpolate to a common wavelength grid
# correct for differences in the wavelength solution between A and B nodding positions

wave_avg = np.nanmean(np.unique(wave, axis=0), axis=0)
for i in range(len(wave)):
    if nodding[i] == "B":
        vel = rv_bary[i] - vrad
    else:
        vel = rv_bary[i]
    vel = 1 - vel / c_light
    flux[i] = np.interp(wave_avg, wave[i] * vel, flux[i])
    wave[i] = wave_avg

# Correct outliers, basic continuum normalization
data = correct_data(data)
wave, flux, uncs, times, segments, header, nodding = data

# Determine telluric lines
# and remove the strongest ones
flux = remove_tellurics(wave, flux, rv_bary=np.mean(rv_bary))

# Calculate airmass
# altaz = star.coordinates.transform_to(AltAz(obstime=times, location=telescope))
# airmass = altaz.secz.value

# Calculate planet transmission spectrum
# with petitradtrans
if elem is None:
    wmin, wmax = wave.min() << u.AA, wave.max() << u.AA
    wmin *= 1 - rv_range / c_light
    wmax *= 1 + rv_range / c_light
    ptr_wave = np.linspace(wmin, wmax, 10_000)
    ptr_flux = np.ones_like(ptr_wave)
else:
    ptr_wave, ptr_flux = ptr_spec(wave, star, planet, rv_range, elem)
    ptr_wave = ptr_wave.to_value(u.AA)

# precompute the reference spectrum at different rv offsets
n = wave.shape[0] // 2
ref = ptr_ref(wave[n], ptr_wave, ptr_flux, rv_range, rv_step)
if elem is not None:
    ref -= 1

# Run Sysrem
# the first number is the number of sysrem iterations accounting for the barycentric velocity
# the second is for regular sysrem iterations
corrected_flux = run_sysrem(flux, uncs, segments, n2)
mad, diff = nanmad(corrected_flux, return_diff=True)
idx = diff > 5 * mad
corrected_flux[idx] = np.nan

# Normalize by the standard deviation in this wavelength column
corrected_flux -= np.nanmedian(corrected_flux, axis=0)
std = nanmad(corrected_flux, axis=0)
std[std == 0] = 1
corrected_flux /= std

# Run the cross correlation between the sysrem residuals and the expected planet spectrum
cc_data, rv_array = run_cross_correlation_ptr(
    corrected_flux,
    ref,
    segments,
    rv_range=rv_range,
    rv_step=rv_step,
)

vsys = star.radial_velocity.to_value("km/s")
kp = Orbit(star, planet).radial_velocity_semiamplitude_planet().to_value("km/s")
kp_range = (-250 - kp, 250 - kp)
vsys_range = (-150 - vsys, 150 - vsys)
for i, (low, upp) in enumerate(enum(segments)):
    selection = "IT"
    kp, vsys, combined = kp_vsys_combine(
        cc_data[i],
        times,
        star,
        planet,
        telescope,
        rv_range,
        rv_step,
        selection=selection,
        vsys_range=vsys_range,
        kp_range=kp_range,
        barycentric_correction=False,
        normalize=False,
    )

    title = ""
    folder = join(
        path,
        f"{star.name}_{planet.name}_{n1}_{n2}_{elem_str}_real_seg{i}_{selection}",
    )

    os.makedirs(folder, exist_ok=True)
    np.savez(
        join(folder, "data.npz"),
        res=combined,
        selection=selection,
        rv_array=rv_array,
        rv_step=rv_step,
        cc_data=cc_data[i],
        corrected_flux=corrected_flux[:, low:upp],
        times=times.mjd,
        kp=kp,
        vsys=vsys,
    )

if len(elem) == 3:
    combined = np.nansum(cc_data[:2], axis=0) + np.nansum(cc_data[3:], axis=0)
elif elem[0] == "H2O":
    combined = np.nansum(cc_data[:2], axis=0) + np.nansum(cc_data[3:], axis=0)
elif elem[0] == "CO2":
    combined = np.nansum(cc_data[:2], axis=0) + np.nansum(cc_data[3:9], axis=0)
elif elem[0] == "CO":
    combined = np.nansum(cc_data[-6:], axis=0)
else:
    combined = np.nansum(cc_data[:2], axis=0) + np.nansum(cc_data[3:], axis=0)

# split into time series?
sections = [0, len(times)]
n_sec = len(sections) - 1

vsys = star.radial_velocity.to_value("km/s")
kp = Orbit(star, planet).radial_velocity_semiamplitude_planet().to_value("km/s")
kp_range = (-250 - kp, 250 - kp)
vsys_range = (-150 - vsys, 150 - vsys)

for i, (low, upp) in enumerate(enum(segments)):
    selection = "IT"
    kp, vsys, kp_vsys = kp_vsys_combine(
        combined[low:upp],
        times[low:upp],
        star,
        planet,
        telescope,
        rv_range,
        rv_step,
        selection=selection,
        vsys_range=vsys_range,
        kp_range=kp_range,
        barycentric_correction=False,
    )

    # Save the results
    folder = join(
        path,
        f"{star.name}_{planet.name}_{n1}_{n2}_{elem_str}_real_{i+1}_{n_sec}_{selection}",
    )

    os.makedirs(folder, exist_ok=True)
    np.savez(
        join(folder, "data.npz"),
        res=kp_vsys,
        kp=kp,
        vsys=vsys,
        selection=selection,
        rv_array=rv_array,
        rv_step=rv_step,
        cc_data=cc_data[:, low:upp],
        combined=combined[low:upp],
        corrected_flux=corrected_flux[low:upp],
        times=times[low:upp].mjd,
    )

    # Run it again, but only for out of transit data
    selection = "OOT"
    kp, vsys, kp_vsys = kp_vsys_combine(
        combined[low:upp],
        times[low:upp],
        star,
        planet,
        telescope,
        rv_range,
        rv_step,
        selection=selection,
        kp_range=kp_range,
        vsys_range=vsys_range,
        barycentric_correction=False,
    )

    # And save to a different directory
    folder = join(
        path,
        f"{star.name}_{planet.name}_{n1}_{n2}_{elem_str}_real_{i+1}_{n_sec}_{selection}",
    )

    os.makedirs(folder, exist_ok=True)
    np.savez(
        join(folder, "data.npz"),
        res=kp_vsys,
        kp=kp,
        vsys=vsys,
        selection=selection,
        rv_array=rv_array,
        rv_step=rv_step,
        cc_data=cc_data[:, low:upp],
        combined=combined[low:upp],
        corrected_flux=corrected_flux[low:upp],
        times=times[low:upp].mjd,
    )

pass
