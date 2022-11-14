# -*- coding: utf-8 -*-
from glob import glob
from itertools import combinations
from os import makedirs
from os.path import basename, dirname, join, realpath
from typing import Tuple

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.constants import c
from astropy.io import fits
from astropy.time import Time
from exoorbit.bodies import Planet, Star
from exoorbit.orbit import Orbit
from genericpath import exists
from scipy.constants import speed_of_light
from scipy.interpolate import interp1d, splev, splrep
from scipy.signal import correlate
from scipy.special import binom
from tqdm import tqdm

from .stats import cohen_d, gauss, gaussfit, nanmad, welch_t
from .sysrem import Sysrem, SysremWithProjection

# Speed of light in km/s
c_light = speed_of_light * 1e-3

# TODO List:
# - automatically mask points before fitting with SME
# - if star and planet steps aren't run manually, we use the initial values
#   instead we should load the data if possible
# - Tests for all the steps
# - Refactoring of the steps, a lot of the code is strewm all over the place
# - Determine Uncertainties for each point


def coadd_cross_correlation(
    cc_data: np.ndarray,
    rv: np.ndarray,
    rv_array: np.ndarray,
    times: Time,
    planet: Planet,
    data_dir: str = None,
    cache_suffix: str = "",
    load: bool = True,
):
    """Sum the cross correlation data along the expected planet trajectory

    Parameters
    ----------
    cc_data : np.ndarray
        cross correlation data, between all combinations of spectra
    rv : np.ndarray
        radial velocity of the planet at each time point
    rv_array : np.ndarray
        radial velocity points of the cc_data
    times : Time
        observation times of the spectra
    planet : Planet
        Planet metadata object
    data_dir : str, optional
        directory to cache data in, by default None
    load : bool, optional
        whether to load data from the cache or not, by default True

    Returns
    -------
    coadd_sum
        sum of all cross correlation data
    coadd_sum_it
        sum of cross correlation data, during the transit
    coadd_sum_oot
        sum of cross correlation data, out of the transit
    """
    if data_dir is not None:
        savefilename = realpath(
            join(data_dir, f"../medium/cross_correlation_coadd{cache_suffix}.npz")
        )
        if load and exists(savefilename):
            data = np.load(savefilename)
            coadd_sum = data["coadd_sum"]
            coadd_sum_it = data["coadd_sum_it"]
            coadd_sum_oot = data["coadd_sum_oot"]
            return coadd_sum, coadd_sum_it, coadd_sum_oot

    cc_data_interp = np.zeros_like(cc_data)
    for i in tqdm(range(len(cc_data)), leave=False):
        for j in tqdm(range(len(cc_data[0])), leave=False):
            # -3 and +4 is the same bc of how python does things
            # cc_data[i, j, mid - offset : mid + offset + 1] = 0
            cc_data_interp[i, j] = np.interp(
                rv_array - (rv[i] - rv[j]).to_value("km/s"), rv_array, cc_data[i, j]
            )
    # Co-add to sort of stack them together
    coadd_sum = np.sum(cc_data_interp, axis=(0, 1))
    phi = (times - planet.time_of_transit) / planet.period
    phi = phi.to_value(1)
    # We only care about the fraction
    phi = phi % 1
    ingress = (-planet.transit_duration / 2 / planet.period).to_value(1) % 1
    egress = (planet.transit_duration / 2 / planet.period).to_value(1) % 1
    in_transit = (phi >= ingress) | (phi <= egress)
    out_transit = ~in_transit
    coadd_sum_it = np.sum(cc_data_interp[in_transit][:, in_transit], axis=(0, 1))
    coadd_sum_oot = np.sum(cc_data_interp[out_transit][:, out_transit], axis=(0, 1))

    if data_dir is not None:
        np.savez(
            savefilename,
            coadd_sum=coadd_sum,
            coadd_sum_it=coadd_sum_it,
            coadd_sum_oot=coadd_sum_oot,
        )

    return coadd_sum, coadd_sum_it, coadd_sum_oot


def run_cross_correlation(
    data: Tuple,
    nsysrem: int,
    rv_range: float = 100,
    rv_step: float = 1,
    skip: Tuple = None,
    load: bool = False,
    data_dir: str = None,
    rv_star: np.ndarray = None,
    rv_planet: np.ndarray = None,
    airmass: np.ndarray = None,
    spec: np.ndarray = None,
    cache_suffix="",
):
    wave, flux, uncs, _, segments, _ = data

    if data_dir is not None:
        savefilename = realpath(
            join(data_dir, f"../medium/cross_correlation{cache_suffix}.npz")
        )
        if load and exists(savefilename):
            data = np.load(savefilename)
            if "rv_array" in data:
                rv_array = data["rv_array"]
            else:
                rv_array = np.arange(-rv_range, rv_range + rv_step, rv_step)
            return data, rv_array

    skip_mask = np.full(flux.shape[1], True)
    if skip is not None:
        for seg in skip:
            skip_mask[segments[seg] : segments[seg + 1]] = False

    # reference = cross_correlation_reference
    if isinstance(wave, u.Quantity):
        wave = wave.to_value(u.AA)
    if isinstance(flux, u.Quantity):
        flux = flux.to_value(1)
    if uncs is None:
        uncs = np.ones_like(flux)
    elif isinstance(uncs, u.Quantity):
        uncs = uncs.to_value(1)
    if rv_planet is not None and isinstance(rv_planet, u.Quantity):
        rv_planet = rv_planet.to_value(u.km / u.s)
    if rv_star is not None and isinstance(rv_star, u.Quantity):
        rv_star = rv_star.to_value(u.km / u.s)

    n = flux.shape[0] // 2
    spec = flux[n]
    uncs = np.clip(uncs, 1, None)
    for low, upp in zip(segments[:-1], segments[1:]):
        # flux[:, low:upp] -= np.nanmin(flux[:, low:upp])
        flux[:, low:upp] /= np.nanmax(flux[:, low:upp])

    # corrected_flux = np.load("corrected_flux_LTT1445A_projected_1.npy")
    sysrem = SysremWithProjection(wave[n], flux, spec, rv_star, airmass, uncs)
    corrected_flux, *_ = sysrem.run(nsysrem)

    # sysrem = Sysrem(corrected_flux)
    # corrected_flux, *_ = sysrem.run(nsysrem-1)
    np.save("corrected_flux_LTT1445A_projected_1.npy", corrected_flux)

    # Normalize by the standard deviation in this wavelength column
    std = nanmad(corrected_flux, axis=0)
    std[std == 0] = 1
    corrected_flux /= std

    rv_array = np.linspace(-10, 10, flux.shape[1])

    spl = [None] * flux.shape[0]
    for i in range(flux.shape[0]):
        spl[i] = splrep(wave[i], corrected_flux[i])

    # Run the cross correlation for all times and radial velocity offsets
    corr = np.zeros((flux.shape[0], flux.shape[0], flux.shape[1]), dtype="f4")
    total = int(binom(flux.shape[0], 2))
    for i, j in tqdm(
        combinations(range(flux.shape[0]), 2), total=total, desc="Combinations"
    ):
        # Zero Normalized Cross Correlation
        a = corrected_flux[i]
        v = corrected_flux[j]

        for k in range(-100, 100, 1):
            pass

        if rv_planet is not None:
            rva = 1 - rv_planet[i] / c_light
            rvv = 1 - rv_planet[j] / c_light
            a = splev(wave[n] * rva, spl[i])
            v = splev(wave[n] * rvv, spl[j])
            # a = interp1d(wave[i] * rva, a, kind="linear", fill_value=1, bounds_error=False)(wave[n])
            # v = interp1d(wave[j] * rvv, v, kind="linear", fill_value=1, bounds_error=False)(wave[n])

        a = (a - np.nanmedian(a)) / (nanmad(a) * len(a))
        v = (v - np.nanmedian(v)) / (nanmad(v))
        corr[i, j] = correlate(a, v, mode="same")

    correlation = {str(nsysrem): corr}
    # correlation[:, :, flux.shape[1] // 2] = np.nan

    if data_dir is not None:
        np.savez(savefilename, **correlation, rv_array=rv_array)
    return correlation, rv_array


def cross_correlation_reference(wave, ptr_wave, ptr_flux, rv_range=100, rv_step=1):
    rv_points = int((2 * rv_range + 1) / rv_step)
    rv = np.linspace(-rv_range, rv_range, num=rv_points)
    try:
        rep = splrep(ptr_wave, ptr_flux)
        interp_func = lambda w: splev(w, rep)
    except ValueError:
        print("Warning: Can not use spline interpolation, using linear instead")
        interp_func = interp1d(ptr_wave, ptr_flux)

    reference = np.zeros((rv_points, wave.size))
    for i in tqdm(range(rv_points)):
        rv_factor = np.sqrt((1 - rv[i] / c_light) / (1 + rv[i] / c_light))
        ref = interp_func(wave * rv_factor)
        reference[i] = np.nan_to_num(ref)

    return reference


def run_cross_correlation_ptr(
    corrected_flux: np.ndarray,
    reference: np.ndarray,
    segments: np.ndarray,
    rv_range: float = 100,
    rv_step: float = 1,
    skip: Tuple = None,
):
    rv_points = int(2 * rv_range / rv_step + 1)

    skip_mask = np.full(corrected_flux.shape[1], True)
    if skip is not None:
        for seg in skip:
            skip_mask[segments[seg] : segments[seg + 1]] = False

    # reference = cross_correlation_reference
    if isinstance(corrected_flux, u.Quantity):
        corrected_flux = corrected_flux.to_value(1)

    rv_array = np.linspace(-rv_range, rv_range, rv_points)
    nseg = len(segments) - 1
    nobs = corrected_flux.shape[0]

    # Run the cross correlation for all times and radial velocity offsets
    corr = np.zeros((nseg, nobs, rv_points))
    for k, (low, upp) in tqdm(
        enumerate(zip(segments[:-1], segments[1:])),
        total=len(segments) - 1,
        desc="Segment",
    ):
        for i in tqdm(range(nobs), total=nobs, desc="Obs", leave=False):
            for j in tqdm(
                range(rv_points), leave=False, desc="rv points", total=rv_points
            ):
                # Zero Normalized Cross Correlation
                a = corrected_flux[i, low:upp]
                v = reference[j, low:upp]

                a = (a - np.nanmedian(a)) / nanmad(a)
                v = (v - np.nanmedian(v)) / nanmad(v)
                corr[k, i, j] += np.nanmean(a * v)

    return corr, rv_array


def kp_vsys_combine(
    data,
    datetime,
    star,
    planet,
    telescope,
    rv_range=100,
    rv_step=1,
    vsys_range=(-70, 70),
    kp_range=(-150, 150),
    selection=None,
    barycentric_correction=True,
    normalize=True,
):
    phi = (datetime - planet.time_of_transit) / planet.period
    phi = phi.to_value(1)
    # We only care about the fraction
    phi = phi % 1
    # Fix the phase so it is continous
    # it is circular anyways
    phi[phi < 0.5] += 1

    vsys = star.radial_velocity.to_value("km/s")
    kp = Orbit(star, planet).radial_velocity_semiamplitude_planet().to_value("km/s")
    vp = vsys + kp * np.sin(2 * np.pi * phi)
    if barycentric_correction:
        rv_bary = -star.coordinates.radial_velocity_correction(
            obstime=datetime, location=telescope
        ).to_value("km/s")
        vp += rv_bary
    else:
        rv_bary = None

    vsys_expected = vsys

    ingress = (-planet.transit_duration / 2 / planet.period).to_value(1) % 1
    egress = 1 + (planet.transit_duration / 2 / planet.period).to_value(1) % 1
    in_transit = (phi >= ingress) & (phi <= egress)
    out_transit = ~in_transit

    rv_points = int(2 * rv_range / rv_step + 1)
    rv = np.linspace(-rv_range, rv_range, rv_points)

    vsys_min, vsys_max = int(vsys + vsys_range[0]), int(vsys + vsys_range[1])
    kp_min, kp_max = int(kp + kp_range[0]), int(kp + kp_range[1])

    vsys = np.linspace(vsys_min, vsys_max, int((vsys_max - vsys_min + 1) // rv_step))
    kp = np.linspace(kp_min, kp_max, int((kp_max - kp_min + 1) // rv_step))
    combined = np.zeros((len(kp), len(vsys)))
    interpolator = interp1d(rv, data, kind="linear", bounds_error=False)
    if selection is None:
        selection = np.full(in_transit.shape, True)
    elif selection == "IT":
        selection = in_transit
    elif selection == "OOT":
        selection = out_transit
    else:
        # keep the points given
        selection = selection

    for i, vs in enumerate(tqdm(vsys)):
        for j, k in enumerate(tqdm(kp, leave=False)):
            vp_loc = vs + k * np.sin(2 * np.pi * phi)
            if barycentric_correction:
                vp_loc += rv_bary
            # shifted = [np.interp(vp[i], rv, data[i], left=np.nan, right=np.nan) for i in range(len(vp))]
            shifted = np.diag(interpolator(vp_loc))
            combined[j, i] = np.nansum(shifted[selection])

    # Normalize to the number of input spectra
    combined /= data.shape[0]

    # Normalize to median 0
    median = np.nanmedian(combined)
    combined -= median

    if normalize:
        mask = (vsys > vsys_expected + vsys_range[1] / 2) | (
            vsys < vsys_expected - vsys_range[0] / 2
        )
        combined /= nanmad(combined[:, mask])

    return kp, vsys, combined


def find_peak(
    combined,
    kp,
    vsys,
    rv_step,
    vsys_width=3,
    kp_width=60,
    fix_kp_vsys=False,
    kp_expected=100,
    vsys_expected=0,
):
    # Find peak
    if fix_kp_vsys:
        kp_peak = np.digitize(kp_expected, kp)
        vsys_peak = np.digitize(vsys_expected, vsys)

        vsys_width = int(vsys_width / rv_step)  # +-3 km/s
        kp_width = int(kp_width / rv_step)  # +-60 km/s

        # (a, mu, sigma, floor)
        A = combined[kp_peak, vsys_peak]
        kp_popt = [A, kp_peak, kp_width, 0]
        vsys_popt = [A, vsys_peak, vsys_width, 0]
    else:
        idx = np.unravel_index(np.argmax(combined), combined.shape)
        kp_peak, vsys_peak = idx
        mean_kp = combined[:, vsys_peak]
        mean_vsys = combined[kp_peak, :]

        kp_width = int(kp_width / rv_step)
        vsys_width = int(vsys_width / rv_step)

        try:
            _, vsys_popt = gaussfit(
                vsys,
                mean_vsys,
                p0=[
                    mean_vsys[vsys_peak] - np.min(mean_vsys),
                    vsys[vsys_peak],
                    1,
                    np.min(mean_vsys),
                ],
            )
            vsys_width = vsys_popt[2] / rv_step
            vsys_width = int(np.ceil(vsys_width))
        except RuntimeError:
            A = combined[kp_peak, vsys_peak]
            vsys_popt = [A, vsys_peak, vsys_width, 0]

        try:
            _, kp_popt = gaussfit(
                kp,
                mean_kp,
                p0=[
                    mean_kp[kp_peak] - np.min(mean_kp),
                    kp[kp_peak],
                    1,
                    np.min(mean_kp),
                ],
            )
            kp_width = kp_popt[2] / rv_step
            kp_width = int(np.ceil(kp_width))
        except RuntimeError:
            A = combined[kp_peak, vsys_peak]
            kp_popt = [A, kp_peak, kp_width, 0]

    return kp_popt, vsys_popt


def calculate_cohen_d(combined, kp_popt, vsys_popt):

    kp_peak, kp_width = kp_popt[1:3]
    vsys_peak, vsys_width = vsys_popt[1:3]

    mask = np.full(combined.shape, False)
    kp_low = max(0, kp_peak - kp_width)
    kp_upp = min(combined.shape[0], kp_peak + kp_width)
    vsys_low = max(0, vsys_peak - vsys_width)
    vsys_upp = min(combined.shape[1], vsys_peak + vsys_width)
    mask[kp_low:kp_upp, vsys_low:vsys_upp] = True

    in_trail = combined[mask].ravel()
    out_trail = combined[~mask].ravel()

    # hrange = (np.min(combined), np.max(combined))
    # bins = 50
    # _, hbins = np.histogram(in_trail, bins=bins, range=hrange, density=True)

    # Calculate the cohen d between the 2 distributions
    d = cohen_d(in_trail, out_trail)
    t = welch_t(in_trail, out_trail)

    return d, t


def load_data(data_dir, load=False):
    savefilename = join(data_dir, "../medium/spectra.npz")
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
    additional_data_fname = join(data_dir, "../*.csv")
    try:
        additional_data = glob(additional_data_fname)[0]
        additional_data = pd.read_csv(additional_data)
    except IndexError:
        additional_data = None

    fluxlist, wavelist, uncslist, times = [], [], [], []
    for f in tqdm(files):
        i = int(basename(f)[-8:-5])
        hdu = fits.open(f)
        header = hdu[0].header
        wave = hdu[1].data << u.AA
        flux = hdu[2].data

        if additional_data is not None:
            add = additional_data.iloc[i]
            time = Time(add["time"], format="jd")

        fluxes, waves, uncses = [], [], []
        orders = list(range(wave.shape[1]))
        for order in orders:
            for det in [1, 2, 3]:
                w = wave[det - 1, order]
                f = flux[det - 1, order]
                if np.all(np.isnan(w)) or np.all(np.isnan(f)):
                    continue

                # We just assume shot noise, no read out noise etc
                unc = np.sqrt(np.abs(f))
                fluxes += [f]
                waves += [w]
                uncses += [unc]

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

    savefilename = join(data_dir, "../medium/spectra.npz")
    makedirs(dirname(savefilename), exist_ok=True)
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
