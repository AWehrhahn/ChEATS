# -*- coding: utf-8 -*-
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation
from exoorbit.orbit import Orbit
from pysme.abund import Abund
from pysme.linelist.vald import ValdFile
from pysme.sme import SME_Structure
from pysme.solve import solve
from pysme.synthesize import synthesize_spectrum
from scipy import constants
from scipy.interpolate import splev, splrep
from scipy.io import readsav
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import gaussian
from tqdm import tqdm

from exoplanet_transit_snr.aaronson_solver import LinearSolver
from exoplanet_transit_snr.petitradtrans import petitRADTRANS
from exoplanet_transit_snr.snr_estimate import load_data
from exoplanet_transit_snr.stellardb import StellarDb
from exoplanet_transit_snr.sysrem import Sars, Sysrem, SysremWithProjection

# define the names of the star and planet
# as well as the datasets within the datasets folder
star, planet = "LTT1445A", "b"
datasets = {
    50: "LTT1445Ab_SNR50_EarthAtmosphere",
    100: "LTT1445Ab_SNR100_EarthAtmosphere",
    200: "LTT1445Ab_SNR200_EarthAtmosphere",
}
# Load the nominal data for this star and planet from simbad/nasa exoplanet archive
sdb = StellarDb()
star = sdb.get(star)
planet = star.planets[planet]
orbit = Orbit(star, planet)
telescope = EarthLocation.of_site("Paranal")
c_light = constants.c / 1e3

# Define the +- range of the radial velocity points,
# and the density of the sampling
rv_range = 200
rv_step = 0.25

# This chooses the dataset we are actually using
snr = 100
nsysrem = 6
regularization = 0.001
fname = f"Aaronson_noiseless_{star.name}_{planet.name}_SNR{snr}_sysrem{nsysrem}_reg{regularization}.npz"

# Where to find the data, might need to be adjusted
data_dir = join(dirname(__file__), "../datasets", datasets[snr], "Spectrum_00")
# load the data from the fits files, returns several objects
data = load_data(data_dir, load=False)
_, _, _, times, _ = data

# Read data from savefile
data = readsav(join(dirname(__file__), "test.dat"))
airmass = data["airmass"]
wave = data["wave"] * u.AA
flux = np.copy(data["obs"])
flux /= np.max(flux)
phase = data["phase"]
gamma = np.copy(data["gamma"])
gamma -= np.mean(gamma)
bary_corr = gamma * c_light * (u.km / u.s)
segments = [0, 2048]
low, upp = segments[:2]

# SME Flux
sme_wave = data["sme_wave"]
sme_spec = data["sme_spec"]
spec = np.interp(wave.to_value(u.AA) * (1 - np.mean(gamma)), sme_wave, sme_spec)

# PySME model
sme = SME_Structure()
sme.wave = wave[low:upp].to_value(u.AA)
sme.spec = flux[low:upp]
# We need the specific intensities, not the combined spectrum
sme.vrad_flag = "whole"
sme.vrad = -11
sme.vrad_limit = 20
sme.cscale_flag = "none"
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
valdfile = join(dirname(__file__), "ltt1445A.lin")
sme.linelist = ValdFile(valdfile)
# sme.linelist = sme.linelist.trim(sme.wave[0, 0], sme.wave[0, -1], 1000)
# Set Mu to the specific positions we need
d = orbit.projected_radius(times)
r = orbit.planet.radius
R = orbit.star.radius
mu = np.sqrt(1 - (d / R) ** 2)

sme_fname = f"sme_intensities_{star.name}_{planet.name}_snr{snr}.npz"
try:
    sme_data = np.load(sme_fname)
    wave_int, flux_int, cont_int = sme_data["wave"], sme_data["flux"], sme_data["cont"]
    wave_sme, flux_sme = sme_data["wave_sme"], sme_data["flux_sme"]
    sme.mu = mu[np.isfinite(mu)]
except (FileNotFoundError, KeyError):
    sme = synthesize_spectrum(sme)
    flux_sme = np.copy(sme.synth.ravel())
    wave_sme = np.copy(sme.wave.ravel())

    sme.specific_intensities_only = True
    sme.mu = mu[np.isfinite(mu)]
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

str_wave, str_flux = wave_sme, flux_sme

# Map the calculated intensities to the correct times
intensities = np.zeros((len(times), wave.shape[0]))
for i in range(len(times)):
    if np.isfinite(mu[i]):
        j = sme.mu == mu[i]
        intensities[i] = np.interp(wave[0].to_value(u.AA), wave_int, flux_int[j][0])

flux_sme = np.interp(wave.to_value(u.AA), wave_sme, flux_sme)
intensities /= flux_sme

str_wave = str_wave << u.AA
str_flux = np.interp(wave[0], str_wave, str_flux)

# Put it all together into an observation sequence
area = orbit.stellar_surface_covered_by_planet(times)
area = area.to_value(1)

# Atmosphere estimated size
scale_height = planet.atm_scale_height(star.teff)
area_planet = planet.area / star.area
area_atmosphere = np.pi * (planet.radius + scale_height) ** 2
area_atmosphere /= star.area
area_atmosphere -= area_planet
area_planet = area_planet
area_atmosphere = area_atmosphere
scale = (area_atmosphere / area_planet).to_value(u.one)

# Airmass calculation
altaz = star.coordinates.transform_to(AltAz(obstime=times, location=telescope))
airmass = altaz.secz.value

# Velocities
# bary_corr = -star.coordinates.radial_velocity_correction(
#             obstime=times, location=telescope
#         )
star_velocity = star.radial_velocity
planet_velocity = orbit.radial_velocity_planet(times)

rv_st = star_velocity + bary_corr
rv_pt = star_velocity + bary_corr + planet_velocity

# TODO:
# - good initial guess is important, use PySME
# - good RVs are also important,
#       could use barycentric correction via astropy
#       or could fit rv to the observation, using the initial guess from PySME

# TODO: try this:
#  - use the out of transit spectra to recover the stellar spectrum
#  - then use that to recover the planet spectrum only

fname_sysrem = f"sysrem_with_projection_{nsysrem}.npz"
try:
    sysrem = np.load(fname_sysrem, allow_pickle=True)
    residual = sysrem["residual"]
    syserr = sysrem["syserr"]
    sysrem_spec = sysrem["spec"]
    sysrem_tell = sysrem["tell"]
    sysrem_am = sysrem["am"]
except FileNotFoundError:
    low, upp = segments[:2]
    sysrem = SysremWithProjection(
        wave[low:upp].to_value(u.AA),
        flux[:, low:upp],
        flux_sme,
        bary_corr.to_value("km/s"),
        airmass,
    )
    residual, syserr, sysrem_spec, sysrem_tell, sysrem_am = sysrem.run(1)
    model = syserr[1]
    if nsysrem > 1:
        sysrem = Sysrem(flux[:, low:upp] - model)
        residual, se = sysrem.run(nsysrem - 1)
        syserr[0] = se[0]
        syserr += se[1:]
    np.savez(
        fname_sysrem,
        residual=residual,
        syserr=syserr,
        spec=sysrem_spec,
        tell=sysrem_tell,
        am=sysrem_am,
    )
model = syserr[0] + np.nansum(tuple(syserr[1:]), axis=0)


# Run the aaronson solver
fname_solution = fname
wave = np.array([wave] * flux.shape[0]) << wave.unit
solver = LinearSolver(
    telescope, star, planet, regularization_weight=regularization, normalize=True
)
# try:
#     solution = np.load(fname_solution)
#     solution = [solution[k] for k in solution.keys()]
# except FileNotFoundError:
area = orbit.stellar_surface_covered_by_planet(times)
area = area.to_value(1)[:, None]
solution = []
low, upp = segments[:2]
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
# wmin, wmax = wave.min(), wave.max()
# ptr = petitRADTRANS(wmin, wmax)
# ptr.init_temp_press_profile(star, planet)
# ptr_wave, ptr_flux = ptr.run()

# ptr_flux = gaussian_filter1d(ptr_flux, 20)
# ptr_flux = np.interp(wave[0], ptr_wave, ptr_flux)

ptr_wave, ptr_flux = np.genfromtxt(
    join(dirname(__file__), "Earth-transmisison-CO2-CH4-H2O.txt"),
    comments="#",
    usecols=(0, 1),
    unpack=True,
)
ptr_wave *= 10_000  # convert uu to AA
ptr_flux = np.interp(wave[0, low:upp].to_value(u.AA), ptr_wave, ptr_flux)
ptr_flux /= np.max(ptr_flux)

i = 0
low, upp = segments[:2]
plt.plot(wave[0, low:upp], flux[5, low:upp] / np.nanmedian(flux[5, low:upp]))
plt.plot(wave[0, low:upp], ptr_flux)
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
