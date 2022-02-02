# -*- coding: utf-8 -*-
import time
from os.path import dirname, join

import emcee
import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import EarthLocation
from astropy.time import Time
from matplotlib import backends

from exoplanet_transit_snr.mcmc import LogLikeCalculator
from exoplanet_transit_snr.petitradtrans import petitRADTRANS
from exoplanet_transit_snr.snr_estimate import load_data
from exoplanet_transit_snr.stellardb import StellarDb
from exoplanet_transit_snr.sysrem import Sysrem

# https://iopscience.iop.org/article/10.3847/1538-3881/ac0428
# https://iopscience.iop.org/article/10.3847/1538-3881/aaffd3#ajaaffd3eqn9
# https://watermark.silverchair.com/342-4-1291.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAtYwggLSBgkqhkiG9w0BBwagggLDMIICvwIBADCCArgGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM-DfJHs_yljX27MjBAgEQgIICieXqQl3gDYU0Of0zrhwHNXKAKKuWvomGL9qwwBB9CUo_vKeP8DH5iIx54URSc4cW-RRmDGlS1lyt3Vy_d7AlapVOwRmbQkuBL6NoO7SMhEQNWVOnOWDR0vVM6SQybCveAhXMeo7ebEIZixpBEgA2o2TnTV6MtkYl3eYa_U4TgR_H5UjuGU68PGmwRBv0RYDAVsw8yDeSxvh6VGxxaYh2KMXeEUTBt-_m3VnrvPgYLMFp3YkismE-DEM1J-Q9T3U7a-9xjezZ6FP0Wcqoc60_vJ2dyMfo5dUdWfm6B1oU3o5chUeB1CIStL7fi561-fI5Gud3p_otU2VqRWcnbjVY3zaEMExwTMWNefE-bgdRgoqYI19JFCK95v4i4n8yXKVwuWJEUPUFQvpgYPb2rzbFWoQ9lx0YFnvxt75SKI19M9MoCA8wpWZJcJWD_grkQOGDz3ubK5BK8E-_Eal6ozzorWu1F1tyxejJT7dYoDn8uJeC5rJw3JAjt7DMy1uNd9j9dp6DWP47LjcHhhU-hiC6TEWrPbc9I01ind-VARIsGHkgqGuyBwjdW6dCKEICNdRrIXERvm-9EZucLM3Lpme4AOlcQ2NZydWJZUrO7P5bQDMDBevoDl9aWbg4gbvrx1jiXhTQWp1O7cSKfROGx3uWx6uPXEdX_wRzewy2p2-nCS8gpQNjPiAPU0Ab400WrGEPBepR8Z7GKi5VNRcL_YOTVloiyF_TQDrq7Qlg1i6LXosLfYo-m0hs36J64ntlcJrFR3Vw5t3ibd2lNWza4P7wakPrgSaoO1ar16ayjlY3LzqRqWxos5rEeg0_TvAwQqEvVBbhJLNEMrtcgo8wlrz1cBfheE1AiKdAgUM

# # define the names of the star and planet
# # as well as the datasets within the datasets folder
star, planet = "WASP-107", "b"
datasets = {50: "WASP-107b_SNR50", 100: "WASP-107b_SNR100", 200: "WASP-107b_SNR200"}

# Load the nominal data for this star and planet from simbad/nasa exoplanet archive
sdb = StellarDb()
star = sdb.get(star)
planet = star.planets[planet]
telescope = EarthLocation.of_site("Paranal")

# Define the +- range of the radial velocity points,
# and the density of the sampling
rv_range = 200
rv_step = 0.25

# This chooses the dataset we are actually using
snr = 100
nsysrem = 5
fname = f"MCMC_{star.name}_{planet.name}_SNR{snr}_sysrem{nsysrem}.npy"

# Where to find the data, might need to be adjusted
data_dir = join(dirname(__file__), "../datasets", datasets[snr], "Spectrum_00")
# load the data from the fits files, returns several objects
data = load_data(data_dir, load=False)
wave, flux, uncs, times, segments = data

# Mask Bad Pixels
flux[flux < 2] = np.nan

# Run sysrem to get a model of Star * Tellurics * TimeVariation
sysrem = Sysrem(flux.T)
residual, model = sysrem.run(nsysrem)
model = model[0] + np.nansum(model[1:], axis=0)
model = model.T
model[np.isnan(flux)] = np.nan

# Run the planet transmission model
wmin, wmax = wave.min(), wave.max()
ptr = petitRADTRANS(wmin, wmax)
ptr.init_temp_press_profile(star, planet)
ptr_wave, ptr_flux = ptr.run()

# Starting the MCMC here
# Setup MCMC
ndim = 10
nwalkers = 32
nsteps = 500_000

log_prob = LogLikeCalculator(
    star, planet, times, telescope, wave, flux, model, ptr_wave, ptr_flux
)
labels = log_prob.get_labels()
truths = log_prob.get_truths()
p0 = truths + np.random.rand(nwalkers, ndim)
# Prepare the backend
# backend = emcee.backends.HDFBackend(fname)
# backend.reset(nwalkers, ndim)
# Initialize Sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
state = sampler.run_mcmc(p0, nsteps, progress=True)
samples = sampler.get_chain()
np.save(fname, samples)
