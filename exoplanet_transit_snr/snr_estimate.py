# -*- coding: utf-8 -*-
from ctypes import c_long
from glob import glob
from itertools import combinations
from os import makedirs
from os.path import basename, dirname, join, realpath
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.constants import c
from astropy.io import fits
from astropy.time import Time
from exoorbit.bodies import Planet, Star

# from cats.extractor.runner import CatsRunner
# from cats.simulator.detector import Crires
# from cats.spectrum import SpectrumArray
from exoorbit.orbit import Orbit
from genericpath import exists
from scipy.constants import speed_of_light
from scipy.interpolate import interp1d, splev, splrep
from scipy.optimize import curve_fit
from scipy.signal import correlate
from scipy.special import binom
from tqdm import tqdm

from .stats import cohen_d, gauss, gaussfit
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
    std = np.nanstd(corrected_flux, axis=0)
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

        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / (np.std(v))
        corr[i, j] = correlate(a, v, mode="same")

    correlation = {str(nsysrem): corr}
    # correlation[:, :, flux.shape[1] // 2] = np.nan

    if data_dir is not None:
        np.savez(savefilename, **correlation, rv_array=rv_array)
    return correlation, rv_array


def cross_correlation_reference(wave, ptr_wave, ptr_flux, rv_range=100, rv_step=1):
    rv_points = int((2 * rv_range + 1) / rv_step)
    rv = np.linspace(-rv_range, rv_range, num=rv_points)
    rep = splrep(ptr_wave, ptr_flux)
    reference = np.zeros((rv_points, wave.size))
    for i in tqdm(range(rv_points)):
        rv_factor = np.sqrt((1 - rv[i] / c_light) / (1 + rv[i] / c_light))
        ref = splev(wave * rv_factor, rep)
        reference[i] = np.nan_to_num(ref)

    return reference


def run_cross_correlation_ptr(
    corrected_flux: np.ndarray,
    reference: np.ndarray,
    segments: np.ndarray,
    rv_range: float = 100,
    rv_step: float = 1,
    skip: Tuple = None,
    load: bool = False,
    data_dir: str = None,
    cache_suffix: str = "",
):
    rv_points = int(2 * rv_range / rv_step + 1)

    if data_dir is not None:
        savefilename = realpath(
            join(data_dir, f"../medium/cross_correlation{cache_suffix}.npz")
        )
        if load and exists(savefilename):
            data = np.load(savefilename)
            rv_array = data["rv_array"]
            data = data["corr"]
            return data, rv_array

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

                corr[k, i, j] += np.nansum(a * v)
                corr[k, i, j] *= corr[k, i, j].size / np.count_nonzero(corr[k, i, j])

                # a = (a - np.nanmean(a)) / np.nanstd(a)
                # v = (v - np.nanmean(v)) / np.nanstd(v)
                # corr[k, i, j] += np.nanmean(a * v)

    if data_dir is not None:
        np.savez(savefilename, corr=corr, rv_array=rv_array)
    return corr, rv_array


def calculate_cohen_d_for_dataset(
    data,
    datetime,
    star,
    planet,
    rv_range=100,
    rv_step=1,
    plot=True,
    title="",
):
    phi = (datetime - planet.time_of_transit) / planet.period
    phi = phi.to_value(1)
    # We only care about the fraction
    phi = phi % 1

    vsys = star.radial_velocity.to_value("km/s")
    kp = Orbit(star, planet).radial_velocity_semiamplitude_planet().to_value("km/s")
    vp = vsys + kp * np.sin(2 * np.pi * phi)

    ingress = (-planet.transit_duration / 2 / planet.period).to_value(1) % 1
    egress = (planet.transit_duration / 2 / planet.period).to_value(1) % 1
    in_transit = (phi >= ingress) | (phi <= egress)

    rv_points = int(2 * rv_range / rv_step + 1)
    rv = np.linspace(-rv_range, rv_range, rv_points)
    # vp_idx = np.interp(vp, rv, np.arange(rv.size))

    if plot:
        vmin, vmax = np.nanpercentile(data.ravel(), [5, 95])
        plt.imshow(data, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        plt.xlabel("rv [km/s]")
        plt.ylabel("#Obs")
        xticks = plt.xticks()[0][1:-1]
        rv_ticks = np.linspace(xticks[0], xticks[-1], len(rv))
        xticks_labels = np.interp(xticks, rv_ticks, rv)
        xticks_labels = [f"{x:.3g}" for x in xticks_labels]
        plt.xticks(xticks, labels=xticks_labels)
        # plt.plot(vp_idx, np.arange(data.shape[0]), "r-.", alpha=0.5)
        plt.hlines(
            np.arange(data.shape[0])[in_transit][0],
            -0.5,
            data.shape[1] + 0.5,
            "k",
            "--",
        )
        plt.hlines(
            np.arange(data.shape[0])[in_transit][-1],
            -0.5,
            data.shape[1] + 0.5,
            "k",
            "--",
        )
        plt.xlim(-0.5, data.shape[1] + 0.5)
        plt.title(title)
        plt.show()

    vsys_min, vsys_max = int(vsys) - 20, int(vsys) + 20
    kp_min, kp_max = int(kp) - 150, int(kp) + 150
    vsys = np.linspace(vsys_min, vsys_max, int((vsys_max - vsys_min + 1) // rv_step))
    kp = np.linspace(kp_min, kp_max, int((kp_max - kp_min + 1) // rv_step))
    combined = np.zeros((len(kp), len(vsys)))
    interpolator = interp1d(rv, data, kind="linear", bounds_error=False)
    for i, vs in enumerate(tqdm(vsys)):
        for j, k in enumerate(tqdm(kp, leave=False)):
            vp = vs + k * np.sin(2 * np.pi * phi)
            # shifted = [np.interp(vp[i], rv, data[i], left=np.nan, right=np.nan) for i in range(len(vp))]
            shifted = np.diag(interpolator(vp))
            combined[j, i] = np.nansum(shifted[in_transit])

    # Normalize to the number of input spectra
    combined /= data.shape[0]
    combined /= combined.std()

    # Normalize to median 0
    median = np.nanmedian(combined)
    combined -= median

    kp_peak = combined.shape[0] // 2
    kp_width = kp_peak

    for i in range(3):
        # Determine the peak position in vsys and kp
        kp_width_int = int(np.ceil(kp_width))
        lower = max(kp_peak - kp_width_int, 0)
        upper = min(kp_peak + kp_width_int + 1, combined.shape[0])
        mean_vsys = np.nanmean(combined[lower:upper, :], axis=0)
        vsys_peak = np.argmax(mean_vsys)

        # And then fit gaussians to determine the width
        try:
            curve, vsys_popt = gaussfit(
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
        except RuntimeError:
            vsys_width = 10
            vsys_popt = None
            break

        # Do the same for the planet velocity
        vsys_width_int = int(np.ceil(vsys_width)) // 4
        lower = max(vsys_peak - vsys_width_int, 0)
        upper = min(vsys_peak + vsys_width_int + 1, combined.shape[1])
        mean_kp = np.nanmean(combined[:, lower:upper], axis=1)
        kp_peak = np.argmax(mean_kp)

        try:
            curve, kp_popt = gaussfit(
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
        except RuntimeError:
            kp_width = 50
            kp_popt = None
            break

    # vsys_width = int(np.ceil(vsys_width))
    # kp_width = int(np.ceil(kp_width))
    vsys_width = int(2 / rv_step)  # +-2 km/s
    kp_width = int(20 / rv_step)  # +-10 km/s

    if plot:
        # vsys_peak_low = vsys[max(vsys_peak - vsys_width, 0)]
        # vsys_peak_upp = vsys[min(vsys_peak + vsys_width, len(vsys) - 1)]
        # kp_peak_low = kp[max(kp_peak - kp_width, 0)]
        # kp_peak_upp = kp[min(kp_peak + kp_width, len(kp) - 1)]

        vp_new = vsys[vsys_peak] + kp[kp_peak] * np.sin(2 * np.pi * phi)
        # vp_low_low = vsys_peak_low + kp_peak_low * np.sin(2 * np.pi * phi)
        # vp_low_upp = vsys_peak_low + kp_peak_upp * np.sin(2 * np.pi * phi)
        # vp_upp_low = vsys_peak_upp + kp_peak_low * np.sin(2 * np.pi * phi)
        # vp_upp_upp = vsys_peak_upp + kp_peak_upp * np.sin(2 * np.pi * phi)
        # vp_low = np.min([vp_low_low, vp_low_upp, vp_upp_low, vp_upp_upp], axis=0)
        # vp_upp = np.max([vp_low_low, vp_low_upp, vp_upp_low, vp_upp_upp], axis=0)

        vp_idx_new = np.interp(vp_new, rv, np.arange(rv.size))
        # vp_idx_low = np.interp(vp_low, rv, np.arange(rv.size))
        # vp_idx_upp = np.interp(vp_upp, rv, np.arange(rv.size))
        vmin, vmax = np.nanpercentile(data.ravel(), [5, 95])

        plt.imshow(data, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        plt.xlabel("rv [km/s]")
        xticks = plt.xticks()[0][1:-1]
        rv_ticks = np.linspace(xticks[0], xticks[-1], len(rv))
        xticks_labels = np.interp(xticks, rv_ticks, rv)
        xticks_labels = [f"{x:.3g}" for x in xticks_labels]
        plt.xticks(xticks, labels=xticks_labels)
        # plt.plot(vp_idx, np.arange(data.shape[0]), "r-.")
        plt.plot(vp_idx_new, np.arange(data.shape[0]), "k-.")
        plt.hlines(
            np.arange(data.shape[0])[in_transit][0],
            -0.5,
            data.shape[1] + 0.5,
            "k",
            "--",
        )
        plt.hlines(
            np.arange(data.shape[0])[in_transit][-1],
            -0.5,
            data.shape[1] + 0.5,
            "k",
            "--",
        )
        # plt.fill_betweenx(
        #     np.arange(data.shape[0]),
        #     vp_idx_low,
        #     vp_idx_upp,
        #     color="tab:orange",
        #     alpha=0.5,
        # )
        plt.xlim(-0.5, data.shape[1] + 0.5)
        plt.title(title)
        plt.show()

        # Plot the results
        ax = plt.subplot(121)
        plt.imshow(combined, aspect="auto", origin="lower")
        ax.add_patch(
            plt.Rectangle(
                (vsys_peak - vsys_width, kp_peak - kp_width),
                2 * vsys_width,
                2 * kp_width,
                fill=False,
                color="red",
            )
        )

        plt.xlabel("vsys [km/s]")
        xticks = plt.xticks()[0][1:-1]
        rv_ticks = np.linspace(xticks[0], xticks[-1], len(vsys))
        xticks_labels = np.interp(xticks, rv_ticks, vsys)
        xticks_labels = [f"{x:.3g}" for x in xticks_labels]
        plt.xticks(xticks, labels=xticks_labels)

        plt.ylabel("Kp [km/s]")
        yticks = plt.yticks()[0][1:-1]
        yticks_labels = np.interp(yticks, np.arange(len(kp)), kp)
        yticks_labels = [f"{y:.3g}" for y in yticks_labels]
        plt.yticks(yticks, labels=yticks_labels)

        plt.subplot(222)
        if vsys_popt is not None:
            plt.plot(vsys, gauss(vsys, *vsys_popt), "r--")
        plt.plot(vsys, mean_vsys)
        plt.vlines(vsys[vsys_peak], np.min(mean_vsys), mean_vsys[vsys_peak], "k", "--")
        plt.xlabel("vsys [km/s]")

        plt.subplot(224)
        plt.plot(kp, mean_kp)
        plt.vlines(kp[kp_peak], np.min(mean_kp), mean_kp[kp_peak], "k", "--")
        if kp_popt is not None:
            plt.plot(kp, gauss(kp, *kp_popt), "r--")
        plt.xlabel("Kp [km/s]")

        plt.suptitle(title)
        plt.show()

    # Have to check that this makes sense
    vsys_width = int(2 / rv_step)  # +-2 km/s
    kp_width = int(20 / rv_step)  # +-10 km/s

    mask = np.full(combined.shape, False)
    kp_low = max(0, kp_peak - kp_width)
    kp_upp = min(kp.size, kp_peak + kp_width)
    vsys_low = max(0, vsys_peak - vsys_width)
    vsys_upp = min(vsys.size, vsys_peak + vsys_width)
    mask[kp_low:kp_upp, vsys_low:vsys_upp] = True

    in_trail = combined[mask].ravel()
    out_trail = combined[~mask].ravel()

    hrange = (np.min(combined), np.max(combined))
    bins = 100
    _, hbins = np.histogram(in_trail, bins=bins, range=hrange, density=True)

    # What does this mean?
    d = cohen_d(in_trail, out_trail)

    if plot:
        plt.hist(
            in_trail.ravel(),
            bins=hbins,
            density=True,
            histtype="step",
            label="in-trail",
        )
        plt.hist(
            out_trail.ravel(),
            bins=hbins,
            density=True,
            histtype="step",
            label="out-of-trail",
        )
        plt.legend()
        plt.title(f"{title}\nCohen d: {d}")
        plt.show()

    return d


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
