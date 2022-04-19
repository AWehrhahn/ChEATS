# -*- coding: utf-8 -*-
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from exoorbit.orbit import Orbit
from pysme.abund import Abund
from pysme.linelist.vald import ValdFile
from pysme.sme import SME_Structure
from pysme.solve import solve
from pysme.synthesize import synthesize_spectrum
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from exoplanet_transit_snr.aaronson_solver import LinearSolver
from exoplanet_transit_snr.petitradtrans import petitRADTRANS
from exoplanet_transit_snr.snr_estimate import load_data
from exoplanet_transit_snr.stellardb import StellarDb
from exoplanet_transit_snr.sysrem import Sysrem

# define the names of the star and planet
# as well as the datasets within the datasets folder
star, planet = "WASP-107", "b"
datasets = {50: "WASP-107b_SNR50", 100: "WASP-107b_SNR100", 200: "WASP-107b_SNR200"}

# Load the nominal data for this star and planet from simbad/nasa exoplanet archive
sdb = StellarDb()
star = sdb.get(star)
planet = star.planets[planet]
orbit = Orbit(star, planet)
telescope = EarthLocation.of_site("Paranal")

# Define the +- range of the radial velocity points,
# and the density of the sampling
rv_range = 200
rv_step = 0.25

# This chooses the dataset we are actually using
snr = 100
nsysrem = 10
regularization = 10000
fname = f"Aaronson_{star.name}_{planet.name}_SNR{snr}_sysrem{nsysrem}_reg{regularization}.npz"

# Where to find the data, might need to be adjusted
data_dir = join(dirname(__file__), "../datasets", datasets[snr], "Spectrum_00")
# load the data from the fits files, returns several objects
data = load_data(data_dir, load=False)
wave, flux, uncs, times, segments = data

# Mask Bad Pixels
flux[flux < 2] = np.nan

# Run sysrem to get a model of Star * Tellurics * TimeVariation
fname_sysrem = f"sysrem_snr{snr}.npy"
try:
    model = np.load(fname_sysrem)
except FileNotFoundError:
    sysrem = Sysrem(flux, iterations=1000)
    residual, syserr = sysrem.run(nsysrem)
    model = syserr[0][:, None] + np.nansum(syserr[1:], axis=0)
    model = model
    model[np.isnan(flux)] = np.nan
    np.save(fname_sysrem, model)

# Use PySME to get the specific intensities for this target
sme_fname = f"sme_intensities_snr{snr}.npz"

sme = SME_Structure()
wmin, wmax = wave[0, 0].to_value(u.AA), wave[-1, -1].to_value(u.AA)
sme.wran = [wmin, wmax]
# We need the specific intensities, not the combined spectrum
sme.vrad_flag = "whole"
sme.cscale_flag = "linear"
sme.cscale_type = "match"
# Set stellar parameters
sme.teff = star.teff.to_value(u.K)
sme.logg = star.logg.to_value(u.one)
sme.abund = Abund(star.monh.to_value(u.one), "asplund2009")
sme.vmic = 1
sme.vmac = 0
sme.vsini = 0
# Define atmosphere
sme.atmo.source = "marcs2012.sav"
sme.atmo.method = "grid"
# Set linelist
valdfile = join(dirname(__file__), "crires_k.lin")
sme.linelist = ValdFile(valdfile)
# Set Mu to the specific positions we need
d = orbit.projected_radius(times)
r = orbit.planet.radius
R = orbit.star.radius
mu = np.sqrt(1 - (d / R) ** 2)
sme.mu = mu[np.isfinite(mu)]
# synthesize
try:
    sme_data = np.load(sme_fname)
    wave_int, flux_int, cont_int = sme_data["wave"], sme_data["flux"], sme_data["cont"]
    wave_sme, flux_sme = sme_data["wave_sme"], sme_data["flux_sme"]
except (FileNotFoundError, KeyError):
    sme = synthesize_spectrum(sme)
    flux_sme = sme.synth.ravel()
    wave_sme = sme.wave.ravel()

    sme.specific_intensities_only = True
    wave_int, flux_int, cont_int = synthesize_spectrum(sme)
    wave_int, flux_int, cont_int = wave_int[0], flux_int[0], cont_int[0]

    np.savez(
        sme_fname,
        wave=wave_int,
        flux=flux_int,
        cont=cont_int,
        wave_sme=wave_sme,
        flux_sme=flux_sme,
    )

# Map the calculated intensities to the correct times
intensities = np.zeros((len(times), wave.shape[1]))
for i in range(len(times)):
    if np.isfinite(mu[i]):
        j = sme.mu == mu[i]
        intensities[i] = np.interp(wave[0].to_value(u.AA), wave_int, flux_int[j][0])

flux_sme = np.interp(wave[0].to_value(u.AA), wave_sme, flux_sme)
intensities /= flux_sme

# Run the aaronson solver
fname_solution = fname
solver = LinearSolver(
    telescope, star, planet, regularization_weight=regularization, normalize=True
)
try:
    solution = np.load(fname_solution)
    solution = [solution[k] for k in solution.keys()]
except FileNotFoundError:
    solution = []
    area = orbit.stellar_surface_covered_by_planet(times).to_value(1)
    for low, upp in tqdm(zip(segments[:-1], segments[1:]), total=len(segments) - 1):
        sol = solver.solve(
            times,
            wave[:, low:upp],
            flux[:, low:upp],
            model[:, low:upp],
            intensities[:, low:upp],
            1,
            area=area,
        )
        solution.append(sol)
    np.savez(fname_solution, *solution)

# Run the planet transmission model
# For comparison only
wmin, wmax = wave.min(), wave.max()
ptr = petitRADTRANS(wmin, wmax)
ptr.init_temp_press_profile(star, planet)
ptr_wave, ptr_flux = ptr.run()

ptr_flux = gaussian_filter1d(ptr_flux, 20)
ptr_flux = np.interp(wave[0], ptr_wave, ptr_flux)

for i, (low, upp) in enumerate(zip(segments[:-1], segments[1:])):
    ptr_flux[low:upp] -= np.min(ptr_flux[low:upp])
    ptr_flux[low:upp] /= np.max(ptr_flux[low:upp])
    plt.plot(wave[0, low:upp], flux[5, low:upp] / np.nanmedian(flux[5, low:upp]))
    plt.plot(wave[0, low:upp], ptr_flux[low:upp])
    plt.plot(solution[i][0], solution[i][1] / np.nanmedian(solution[i][1]))
    plt.show()

pass

# # Run the planet transmission model
# wmin, wmax = wave.min(), wave.max()
# ptr = petitRADTRANS(wmin, wmax)
# ptr.init_temp_press_profile(star, planet)
# ptr_wave, ptr_flux = ptr.run()

# # Starting the MCMC here
# # Setup MCMC
# ndim = 10
# nwalkers = 32
# nsteps = 500_000

# log_prob = LogLikeCalculator(
#     star, planet, times, telescope, wave, flux, model, ptr_wave, ptr_flux
# )
# labels = log_prob.get_labels()
# truths = log_prob.get_truths()
# p0 = truths + np.random.rand(nwalkers, ndim)
# # Prepare the backend
# # backend = emcee.backends.HDFBackend(fname)
# # backend.reset(nwalkers, ndim)
# # Initialize Sampler
# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
# state = sampler.run_mcmc(p0, nsteps, progress=True)
# samples = sampler.get_chain()
# np.save(fname, samples)
