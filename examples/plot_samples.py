# -*- coding: utf-8 -*-
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from exoorbit.orbit import Orbit

from exoplanet_transit_snr.stellardb import StellarDb

star, planet = "WASP-107", "b"
datasets = {50: "WASP-107b_SNR50", 100: "WASP-107b_SNR100", 200: "WASP-107b_SNR200"}

# Load the nominal data for this star and planet from simbad/nasa exoplanet archive
sdb = StellarDb()
star = sdb.get(star)
planet = star.planets[planet]
orbit = Orbit(star, planet)
rv = orbit.radial_velocity_semiamplitude_planet()

snr = 200
nsysrem = 5
fname = f"MCMC_{star.name}_{planet.name}_SNR{snr}_sysrem{nsysrem}.h5"
# fname = "mcmc_samples.npy"

ndim = 10
nwalkers = 32
labels = ["a", "v_sys", "mass", "radius", "sma", "per", "inc", "ecc", "w", "t0"]
truths = np.array(
    [
        1,
        star.radial_velocity.to_value(u.km / u.s),
        planet.mass.to_value(u.M_jup),
        planet.radius.to_value(u.R_jup),
        planet.sma.to_value(u.AU),
        planet.period.to_value(u.day),
        planet.inc.to_value(u.deg),
        planet.ecc.to_value(u.one),
        planet.omega.to_value(u.deg),
        planet.t0.mjd,
    ]
)

sampler = emcee.backends.HDFBackend(fname)
samples = sampler.get_chain()
# samples = np.load(fname)

tau = emcee.autocorr.integrated_time(samples, quiet=True)
# tau = sampler.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))


# Print results
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")
plt.show()

# sampler.get_chain(discard=2000, flat=True)
# ranges=[(1.0, 1.015), (-150, 150), (0, 1), (0, 2), (0, 5), (0, 10), (70, 110), (0, 1), (40, 160), 0.99]
ranges = [0.9] * len(labels)
flat_samples = samples[burnin::thin].reshape((-1, ndim))
fig = corner.corner(flat_samples, labels=labels, truths=truths, range=ranges)
plt.show()

for i in range(ndim):
    low, mid, upp = np.percentile(flat_samples[:, i], [16, 50, 84], axis=0)
    sigma = (upp - low) / 2
    print(f"{labels[i]}: {mid:.5g} + {upp-mid:.5g} - {mid-low:.5g} ; {truths[i]:.5g}")

# a: 1.0065 + 0.00016413 - 0.00014236 ; 1
# v_sys: 12.922 + 31.738 - 30.919 ; 13.74
# mass: 4.1251 + 4.2402 - 3.6097 ; 0.096
# per: 12.947 + 20.586 - 7.4242 ; 5.7215
# inc: 91.563 + 41.107 - 65.425 ; 89.56
# ecc: 0.49631 + 0.32373 - 0.37054 ; 0.06
# w: 96.66 + 29.538 - 38.358 ; 90
# t0: 57578 + 61.426 - 164.84 ; 57584
