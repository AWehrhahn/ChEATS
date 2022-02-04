# -*- coding: utf-8 -*-
from glob import glob
from itertools import combinations
from os import makedirs
from os.path import basename, dirname, join, realpath
from tkinter import N

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.constants import c
from astropy.io import fits
from astropy.time import Time

# from cats.extractor.runner import CatsRunner
# from cats.simulator.detector import Crires
# from cats.spectrum import SpectrumArray
from exoorbit.orbit import Orbit
from genericpath import exists
from scipy.interpolate import interp1d
from scipy.special import binom
from tqdm import tqdm

from .stats import cohen_d, gauss, gaussfit
from .sysrem import sysrem

# TODO List:
# - automatically mask points before fitting with SME
# - if star and planet steps aren't run manually, we use the initial values
#   instead we should load the data if possible
# - Tests for all the steps
# - Refactoring of the steps, a lot of the code is strewm all over the place
# - Determine Uncertainties for each point


# def get_detector(setting="K/2/4", detectors=(1, 2, 3), orders=(7, 6, 5, 4, 3, 2)):
#     detector = Crires(setting, detectors, orders=orders)
#     return detector


# def init_cats(
#     star,
#     planet,
#     dataset,
#     rv_step=0.25,
#     rv_range=200,
#     base_dir=None,
#     raw_dir="Spectrum_00",
#     detector=None,
# ):
#     # Detector
#     if detector is None:
#         detector = get_detector()

#     # Initialize the CATS runner
#     if base_dir is None:
#         dataset_dir = join(dirname(__file__), "../datasets")
#         base_dir = join(dataset_dir, dataset)
#     raw_dir = join(base_dir, raw_dir)
#     medium_dir = join(base_dir, "medium")
#     done_dir = join(base_dir, "done")
#     runner = CatsRunner(
#         detector,
#         star,
#         planet,
#         None,
#         base_dir=base_dir,
#         raw_dir=raw_dir,
#         medium_dir=medium_dir,
#         done_dir=done_dir,
#     )
#     # Set the radial velocity step size
#     rv_points = int((2 * rv_range + 1) / rv_step)
#     runner.configuration["cross_correlation"]["rv_range"] = rv_range
#     runner.configuration["cross_correlation"]["rv_points"] = rv_points
#     runner.configuration["cross_correlation_reference"]["rv_range"] = rv_range
#     runner.configuration["cross_correlation_reference"]["rv_points"] = rv_points

#     # # Override data with known information
#     # star = runner.star
#     # planet = runner.planet
#     return runner


# def create_dataset(star, planet, datasets, plot=True):

#     data = []
#     snrs = []
#     for snr_local, dataset in datasets.items():
#         runner = init_cats(star, planet, dataset)
#         spectra = runner.run_module("spectra", load=True)
#         data += [spectra]
#         snrs += [snr_local]

#     result = []
#     new_snrs = []
#     for i in range(len(data)):
#         # TODO: combine observations from different datasets
#         j = i
#         # snr_i, snr_j = snrs[i], snrs[j]
#         data_i, data_j = data[i], data[j]

#         new_snr = np.sqrt(snrs[i] ** 2 + snrs[j] ** 2)
#         if len(data_i) == len(data_j):
#             n = (len(data_i) // 2) * 2
#             new_data = data_i.flux[:n:2] + data_j.flux[1:n:2]
#             new_wave = data_i.wavelength[:n:2]
#             new_time = (data_i.datetime[:n:2].mjd + data_j.datetime[1:n:2].mjd) / 2
#             new_time = Time(new_time, format="mjd")
#             new_segments = data_i.segments
#         else:
#             # This should not happen (yet)
#             pass

#         arr = SpectrumArray(
#             flux=new_data, spectral_axis=new_wave, segments=new_segments
#         )
#         arr.datetime = new_time

#         new_snrs += [new_snr]
#         result += [arr]

#     datasets = {}
#     for snr, arr in zip(new_snrs, result):
#         dataset = f"WASP-107b_SNR{int(snr)}"
#         base_dir = realpath(join(dirname(__file__), f"../datasets/{dataset}"))
#         raw_dir = join(base_dir, "Spectrum_00")
#         medium_dir = join(base_dir, "medium")
#         done_dir = join(base_dir, "done")
#         makedirs(raw_dir, exist_ok=True)
#         makedirs(medium_dir, exist_ok=True)
#         makedirs(done_dir, exist_ok=True)
#         # arr = SpectrumArray(arr)
#         arr.write(join(medium_dir, "spectra.flex"))
#         datasets[int(snr)] = dataset

#     return datasets


def run_cross_correlation(
    data,
    max_nsysrem=10,
    max_nsysrem_after=3,
    rv_range=100,
    rv_step=1,
    skip=None,
    load=False,
    data_dir=None,
):
    wave, flux, uncs, times, segments = data
    rv_points = int(2 * rv_range / rv_step + 1)

    if data_dir is not None:
        savefilename = join(data_dir, "../medium/cross_correlation.npz")
        if load and exists(savefilename):
            data = np.load(savefilename)
            return data

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

    correlation = {}
    for n in tqdm(range(max_nsysrem), desc="Sysrem N"):
        corrected_flux = sysrem(flux, num_errors=n, errors=uncs)

        # Normalize by the standard deviation in this wavelength column
        std = np.nanstd(corrected_flux, axis=0)
        std[std == 0] = 1
        corrected_flux /= std

        # reference_flux = np.copy(reference.flux.to_value(1))
        # reference_flux -= np.nanmean(reference_flux, axis=1)[:, None]
        # reference_flux /= np.nanstd(reference_flux, axis=1)[:, None]
        wave_noshift = wave[0]
        c_light = c.to_value("km/s")

        # Run the cross correlation for all times and radial velocity offsets
        corr = np.zeros((flux.shape[0], flux.shape[0], int(rv_points)))
        total = binom(flux.shape[0], 2)
        for i, j in tqdm(
            combinations(range(flux.shape[0]), 2), total=total, desc="Combinations"
        ):
            # for i in tqdm(range(flux.shape[0] - 1), leave=False, desc="Observation"):
            for k in tqdm(
                range(rv_points),
                leave=False,
                desc="radial velocity",
            ):
                # Doppler Shift the next spectrum
                rv = -rv_range + k * rv_step
                wave_shift = wave_noshift * (1 + rv / c_light)
                newspectra = np.interp(wave_shift, wave_noshift, corrected_flux[j])

                # Mask bad pixels
                m = np.isfinite(corrected_flux[i])
                m &= np.isfinite(newspectra)
                m &= skip_mask

                # Cross correlate!
                corr[i, j, k] += np.correlate(
                    corrected_flux[i][m],
                    newspectra[m],
                    "valid",
                )
                # Normalize to the number of data points used
                corr[i, j, k] *= m.size / np.count_nonzero(m)

        n_kp = 10
        total_total = np.zeros((n_kp, rv_points))
        for k in range(n_kp):
            total = np.zeros((flux.shape[0], rv_points))
            for i in range(flux.shape[0]):
                for m in range(flux.shape[0] - i):
                    total[i, : rv_points - m * k] += corr[i, i + m, k * m :]
                total[i] = np.roll(total[i], -m * k)
            total_total[k] = np.sum(total, axis=0)
        correlation[str(n)] = total_total

    if data_dir is not None:
        np.savez(savefilename, **correlation)
    return correlation


def calculate_cohen_d_for_dataset(
    spectra,
    data,
    star,
    planet,
    rv_range=100,
    rv_step=1,
    sysrem="7",
    plot=True,
    title="",
):
    wave, flux, uncs, datetime, segments = spectra

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

    data = data[str(sysrem)]
    rv_points = int(2 * rv_range / rv_step + 1)
    rv = np.linspace(-rv_range, rv_range, rv_points)
    vp_idx = np.interp(vp, rv, np.arange(rv.size))

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
        vsys_peak_low = vsys[max(vsys_peak - vsys_width, 0)]
        vsys_peak_upp = vsys[min(vsys_peak + vsys_width, len(vsys) - 1)]
        kp_peak_low = kp[max(kp_peak - kp_width, 0)]
        kp_peak_upp = kp[min(kp_peak + kp_width, len(kp) - 1)]

        vp_new = vsys[vsys_peak] + kp[kp_peak] * np.sin(2 * np.pi * phi)
        vp_low_low = vsys_peak_low + kp_peak_low * np.sin(2 * np.pi * phi)
        vp_low_upp = vsys_peak_low + kp_peak_upp * np.sin(2 * np.pi * phi)
        vp_upp_low = vsys_peak_upp + kp_peak_low * np.sin(2 * np.pi * phi)
        vp_upp_upp = vsys_peak_upp + kp_peak_upp * np.sin(2 * np.pi * phi)
        vp_low = np.min([vp_low_low, vp_low_upp, vp_upp_low, vp_upp_upp], axis=0)
        vp_upp = np.max([vp_low_low, vp_low_upp, vp_upp_low, vp_upp_upp], axis=0)

        vp_idx_new = np.interp(vp_new, rv, np.arange(rv.size))
        vp_idx_low = np.interp(vp_low, rv, np.arange(rv.size))
        vp_idx_upp = np.interp(vp_upp, rv, np.arange(rv.size))

        plt.imshow(data, aspect="auto", origin="lower")
        plt.xlabel("rv [km/s]")
        xticks = plt.xticks()[0][1:-1]
        rv_ticks = np.linspace(xticks[0], xticks[-1], len(rv))
        xticks_labels = np.interp(xticks, rv_ticks, rv)
        xticks_labels = [f"{x:.3g}" for x in xticks_labels]
        plt.xticks(xticks, labels=xticks_labels)
        plt.plot(vp_idx, np.arange(data.shape[0]), "r-.")
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
        plt.fill_betweenx(
            np.arange(data.shape[0]),
            vp_idx_low,
            vp_idx_upp,
            color="tab:orange",
            alpha=0.5,
        )
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
        return wavelist, fluxlist, uncslist, times, segments

    files_fname = join(data_dir, "*.fits")
    files = glob(files_fname)
    additional_data_fname = join(data_dir, "*.csv")
    try:
        additional_data = glob(additional_data_fname)[0]
        additional_data = pd.read_csv(additional_data)
    except IndexError:
        additional_data = None

    fluxlist, wavelist, uncslist, times = [], [], [], []
    for f in tqdm(files):
        i = int(basename(f)[-8:-5])
        hdu = fits.open(f)
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
    np.savez(
        savefilename,
        flux=fluxlist,
        wave=wavelist,
        uncs=uncslist,
        time=times,
        segments=segments,
    )
    return wavelist, fluxlist, uncslist, times, segments
