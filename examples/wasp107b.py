# -*- coding: utf-8 -*-
from os.path import dirname, exists, join

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, EarthLocation
from exoorbit.orbit import Orbit

from exoplanet_transit_snr.snr_estimate import (
    calculate_cohen_d_for_dataset,
    coadd_cross_correlation,
    load_data,
    run_cross_correlation,
)
from exoplanet_transit_snr.stats import gaussfit
from exoplanet_transit_snr.stellardb import StellarDb

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

for snr in [200]:
    # Where to find the data, might need to be adjusted
    data_dir = join(dirname(__file__), "../datasets", datasets[snr], "Spectrum_00")
    # load the data from the fits files, returns several objects
    data = load_data(data_dir, load=True)
    wave, flux, uncs, times, segments = data

    # Determine orbital parameters
    phi = (times - planet.time_of_transit) / planet.period
    phi = phi.to_value(1)
    phi = phi % 1
    ingress = (-planet.transit_duration / 2 / planet.period).to_value(1) % 1
    egress = (planet.transit_duration / 2 / planet.period).to_value(1) % 1
    in_transit = (phi >= ingress) | (phi <= egress)
    out_transit = ~in_transit

    altaz = star.coordinates.transform_to(AltAz(obstime=times, location=telescope))
    airmass = altaz.secz.value

    rv = orbit.radial_velocity_planet(times)
    rv_star = -star.coordinates.radial_velocity_correction(
        obstime=times, location=telescope
    )

    # TODO: Create initial stellar spectrum guess
    # sme = SME_Structure()
    # sme.wave = wave[low:upp].to_value(u.AA)
    # sme.spec = flux[low:upp]
    # # We need the specific intensities, not the combined spectrum
    # sme.vrad_flag = "whole"
    # sme.vrad = -11
    # sme.vrad_limit = 20
    # sme.cscale_flag = "none"
    # sme.cscale_type = "match"
    # # Set stellar parameters
    # sme.teff = star.teff.to_value(u.K)
    # sme.logg = star.logg.to_value(u.one)
    # sme.abund = Abund(star.monh.to_value(u.one), "asplund2009")
    # sme.vmic = 1
    # sme.vmac = 0
    # sme.vsini = 0
    # # Define atmosphere
    # sme.atmo.source = "marcs2012.sav"
    # sme.atmo.method = "grid"
    # # Set linelist
    # valdfile = join(dirname(__file__), "ltt1445A.lin")
    # sme.linelist = ValdFile(valdfile)
    # # sme.linelist = sme.linelist.trim(sme.wave[0, 0], sme.wave[0, -1], 1000)
    # # Set Mu to the specific positions we need
    # d = orbit.projected_radius(times)
    # r = orbit.planet.radius
    # R = orbit.star.radius
    # mu = np.sqrt(1 - (d / R) ** 2)

    # sme_fname = f"sme_intensities_{star.name}_{planet.name}_snr{snr}.npz"
    # try:
    #     sme_data = np.load(sme_fname)
    #     wave_int, flux_int, cont_int = sme_data["wave"], sme_data["flux"], sme_data["cont"]
    #     wave_sme, flux_sme = sme_data["wave_sme"], sme_data["flux_sme"]
    #     sme.mu = mu[np.isfinite(mu)]
    # except (FileNotFoundError, KeyError):
    #     sme = synthesize_spectrum(sme)
    #     flux_sme = np.copy(sme.synth.ravel())
    #     wave_sme = np.copy(sme.wave.ravel())

    #     sme.specific_intensities_only = True
    #     sme.mu = mu[np.isfinite(mu)]
    #     wave_int, flux_int, cont_int = synthesize_spectrum(sme)
    #     wave_int, flux_int, cont_int = wave_int[0], flux_int[0], cont_int[0]

    #     np.savez(
    #         sme_fname,
    #         wave=wave_int,
    #         flux=flux_int,
    #         cont=cont_int,
    #         wave_sme=wave_sme,
    #         flux_sme=flux_sme,
    #     )

    # str_wave, str_flux = wave_sme, flux_sme

    # Run the cross correlation to the next neighbour
    cc_data, rv_array = run_cross_correlation(
        data,
        nsysrem=1,
        rv_range=rv_range,
        rv_step=rv_step,
        load=False,
        data_dir=data_dir,
        rv_star=rv_star,
        rv_planet=rv,
        airmass=airmass,
    )

    cc_data = cc_data["10"]
    cc_data_coadd, cc_it, cc_oot = coadd_cross_correlation(
        cc_data, rv, rv_array, times, planet, data_dir=data_dir, load=False
    )

    # Normalize by the oot signal
    cc_it /= cc_oot

    # Fit a gaussian
    p0 = [np.max(cc_data_coadd) - np.min(cc_data_coadd), 0, 10, np.min(cc_data_coadd)]
    gauss, pval = gaussfit(rv_array, cc_data_coadd, p0=p0)

    # Plot the results
    plt.plot(rv_array, cc_data_coadd, label="all observations")
    plt.plot(rv_array, gauss, label="best fit gaussian")
    plt.plot(rv_array, cc_it, label="in-transit")
    plt.plot(rv_array, cc_oot, label="out-of-transit")
    plt.legend()
    plt.title(f"{star.name} {planet.name} SNR{snr}")
    plt.xlabel(r"$\Delta$RV [km/s]")
    plt.ylabel("CCF")
    plt.show()
