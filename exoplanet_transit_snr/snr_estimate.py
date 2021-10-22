# -*- coding: utf-8 -*-
from os import makedirs
from os.path import dirname, join, realpath

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from cats.extractor.runner import CatsRunner
from cats.simulator.detector import Crires
from cats.spectrum import SpectrumArray
from exoorbit.orbit import Orbit
from scipy.interpolate import interp1d
from skimage import transform as tf
from tqdm import tqdm

from .stats import cohen_d, gauss, gaussfit

# TODO List:
# - automatically mask points before fitting with SME
# - if star and planet steps aren't run manually, we use the initial values
#   instead we should load the data if possible
# - Tests for all the steps
# - Refactoring of the steps, a lot of the code is strewm all over the place
# - Determine Uncertainties for each point


def shear(x, shear=1, inplace=False):
    afine_tf = tf.AffineTransform(shear=shear)
    modified = tf.warp(x, inverse_map=afine_tf)
    return modified


def init_cats(star, planet, dataset, rv_step=0.25, rv_range=200):
    # Detector
    setting = "K/2/4"
    detectors = [1, 2, 3]
    orders = [7, 6, 5, 4, 3, 2]
    detector = Crires(setting, detectors, orders=orders)

    # Initialize the CATS runner
    dataset_dir = join(dirname(__file__), "../datasets")
    base_dir = join(dataset_dir, dataset)
    raw_dir = join(base_dir, "Spectrum_00")
    medium_dir = join(base_dir, "medium")
    done_dir = join(base_dir, "done")
    runner = CatsRunner(
        detector,
        star,
        planet,
        None,
        base_dir=base_dir,
        raw_dir=raw_dir,
        medium_dir=medium_dir,
        done_dir=done_dir,
    )
    # Set the radial velocity step size
    rv_points = int((2 * rv_range + 1) / rv_step)
    runner.configuration["cross_correlation"]["rv_range"] = rv_range
    runner.configuration["cross_correlation"]["rv_points"] = rv_points
    runner.configuration["cross_correlation_reference"]["rv_range"] = rv_range
    runner.configuration["cross_correlation_reference"]["rv_points"] = rv_points

    # # Override data with known information
    # star = runner.star
    # planet = runner.planet
    return runner


def create_dataset(star, planet, datasets, plot=True):

    data = []
    snrs = []
    for snr_local, dataset in datasets.items():
        runner = init_cats(star, planet, dataset)
        spectra = runner.run_module("spectra", load=True)
        data += [spectra]
        snrs += [snr_local]

    result = []
    new_snrs = []
    for i in range(len(data)):
        # TODO: combine observations from different datasets
        j = i
        # snr_i, snr_j = snrs[i], snrs[j]
        data_i, data_j = data[i], data[j]

        new_snr = np.sqrt(snrs[i] ** 2 + snrs[j] ** 2)
        if len(data_i) == len(data_j):
            n = (len(data_i) // 2) * 2
            new_data = data_i.flux[:n:2] + data_j.flux[1:n:2]
            new_wave = data_i.wavelength[:n:2]
            new_time = (data_i.datetime[:n:2].mjd + data_j.datetime[1:n:2].mjd) / 2
            new_time = Time(new_time, format="mjd")
            new_segments = data_i.segments
        else:
            # This should not happen (yet)
            pass

        arr = SpectrumArray(
            flux=new_data, spectral_axis=new_wave, segments=new_segments
        )
        arr.datetime = new_time

        new_snrs += [new_snr]
        result += [arr]

    datasets = {}
    for snr, arr in zip(new_snrs, result):
        dataset = f"WASP-107b_SNR{int(snr)}"
        base_dir = realpath(join(dirname(__file__), f"../datasets/{dataset}"))
        raw_dir = join(base_dir, "Spectrum_00")
        medium_dir = join(base_dir, "medium")
        done_dir = join(base_dir, "done")
        makedirs(raw_dir, exist_ok=True)
        makedirs(medium_dir, exist_ok=True)
        makedirs(done_dir, exist_ok=True)
        # arr = SpectrumArray(arr)
        arr.write(join(medium_dir, "spectra.flex"))
        datasets[int(snr)] = dataset

    return datasets


def calculate_cohen_d_for_dataset(star, planet, dataset, sysrem="7", plot=True):
    runner = init_cats(star, planet, dataset)
    planet = runner.planet
    star = runner.star

    try:
        runner.run_module("cross_correlation_reference", load=True)
        data = runner.run_module("cross_correlation", load=True)
    except FileNotFoundError:
        runner.run_module("cross_correlation_reference", load=False)
        data = runner.run_module("cross_correlation", load=False)

    spectra = runner.data["spectra"]

    datetime = spectra.datetime
    phi = (datetime - planet.time_of_transit) / planet.period
    phi = phi.to_value(1)
    # We only care about the fraction
    phi = phi % 1

    vsys = star.radial_velocity.to_value("km/s")
    kp = Orbit(star, planet).radial_velocity_semiamplitude_planet().to_value("km/s")
    vp = vsys + kp * np.sin(2 * np.pi * phi)

    data = data[str(sysrem)]
    config = runner.configuration["cross_correlation"]
    rv_range = config["rv_range"]
    rv_points = config["rv_points"]
    rv_step = (2 * rv_range + 1) / rv_points
    rv = np.linspace(-rv_range, rv_range, rv_points)
    vp_idx = np.interp(vp, rv, np.arange(rv.size))

    if plot:
        plt.imshow(data, aspect="auto", origin="lower")
        plt.xlabel("rv [km/s]")
        xticks = plt.xticks()[0][1:-1]
        xticks_labels = np.interp(xticks, np.arange(len(rv)), rv)
        xticks_labels = [f"{x:.3g}" for x in xticks_labels]
        plt.xticks(xticks, labels=xticks_labels)
        plt.plot(vp_idx, np.arange(data.shape[0]), "r-.", alpha=0.5)
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
            combined[j, i] = np.nansum(shifted)

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

    if plot:
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
        xticks_labels = np.interp(xticks, np.arange(len(vsys)), vsys)
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

        plt.suptitle(dataset)
        plt.show()

    # Have to check that this makes sense
    vsys_width = int(np.ceil(vsys_width))
    kp_width = int(np.ceil(kp_width))

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
            label="in transit",
        )
        plt.hist(
            out_trail.ravel(),
            bins=hbins,
            density=True,
            histtype="step",
            label="out of transit",
        )
        plt.legend()
        plt.title(f"{dataset}\nCohen d: {d}")
        plt.show()

    return d


# init_cats("L_98-59", "c", "L98-59c_HotJup_SNR100")
# datasets = {100: "L98-59c_HotJup_SNR100"}
# d = calculate_cohen_d_for_dataset("L 98-59", "c", "L98-59c_HotJup_SNR100", "7")

star, planet = "LTT1445A", "b"
datasets = {
    50: "LTT1445Ab_SNR50_EarthAtmosphere",
    100: "LTT1445Ab_SNR100_EarthAtmosphere",
    200: "LTT1445Ab_SNR200_EarthAtmosphere",
}

# init_cats(star, planet, datasets[50])
for snr in [50, 100, 200]:
    d = calculate_cohen_d_for_dataset(star, planet, datasets[snr], "7.1", plot=True)

# star, planet = "WASP-107", "b"
# datasets = {50: "WASP-107b_SNR50", 100: "WASP-107b_SNR100", 200: "WASP-107b_SNR200"}

# # # init_cats(star, planet, datasets[50])
# d = calculate_cohen_d_for_dataset(star, planet, datasets[100], "7")

print(d)
pass
