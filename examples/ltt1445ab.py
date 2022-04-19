# -*- coding: utf-8 -*-
from os.path import dirname, exists, join

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation
from exoorbit.orbit import Orbit

from exoplanet_transit_snr.snr_estimate import (
    coadd_cross_correlation,
    load_data,
    run_cross_correlation,
)
from exoplanet_transit_snr.stats import gaussfit
from exoplanet_transit_snr.stellardb import StellarDb

# define the names of the star and planet
# as well as the datasets within the datasets folder
star, planet = "LTT 1445 A", "b"
datasets = {
    50: "LTT1445Ab_Winters2021/LTT1445Ab_SNR50_EarthAtmosphere_NoNoise",
    100: "LTT1445Ab_Winters2021/LTT1445Ab_SNR100_EarthAtmosphere_NoNoise",
    200: "LTT1445Ab_Winters2021/LTT1445Ab_SNR200_EarthAtmosphere_NoNoise",
}
# Load the nominal data for this star and planet from simbad/nasa exoplanet archive
sdb = StellarDb(regularize=False)
# sdb.refresh(star)
star = sdb.get(star)
planet = star.planets[planet]
planet.transit_duration = planet.transit_duration.to_value(u.day) * u.hour
orbit = Orbit(star, planet)
telescope = EarthLocation.of_site("Paranal")

# Define the +- range of the radial velocity points,
# and the density of the sampling
rv_range = 200
rv_step = 0.25

for snr in [50]:
    # Where to find the data, might need to be adjusted
    data_dir = join(dirname(__file__), "../datasets", datasets[snr], "Spectrum_00")
    # load the data from the fits files, returns several objects
    data = load_data(data_dir, load=True)
    wave, flux, uncs, times, segments, header = data

    segment_i = 4
    low, upp = segments[segment_i : segment_i + 2]
    wave = wave[:, low:upp]
    flux = flux[:, low:upp]
    uncs = uncs[:, low:upp]
    segments = [0, upp - low]
    data = wave, flux, uncs, times, segments, header

    rv_bary = -star.coordinates.radial_velocity_correction(
        obstime=times, location=telescope
    )
    rv_star = star.radial_velocity + rv_bary
    rv_planet = orbit.radial_velocity_planet(times) + rv_bary + rv_star

    altaz = star.coordinates.transform_to(AltAz(obstime=times, location=telescope))
    airmass = altaz.secz.value

    # Run the cross correlation to the next neighbour
    nsysrem = 5
    cache_suffix = f"_{star.name}_{planet.name}_nsysrem{nsysrem}_snr{snr}_seg{segment_i}".lower().replace(
        " ", "_"
    )
    cc_data, rv_array = run_cross_correlation(
        data,
        nsysrem=nsysrem,
        rv_range=rv_range,
        rv_step=rv_step,
        data_dir=data_dir,
        rv_star=rv_bary,
        rv_planet=rv_planet,
        airmass=airmass,
        cache_suffix=cache_suffix,
        load=False,
    )

    phi = (times - planet.time_of_transit) / planet.period
    phi = phi.to_value(1)
    phi = phi % 1
    ingress = (-planet.transit_duration / 2 / planet.period).to_value(1) % 1
    egress = (planet.transit_duration / 2 / planet.period).to_value(1) % 1
    in_transit = (phi >= ingress) | (phi <= egress)
    out_transit = ~in_transit

    # Coadd all the ccfs together
    rv = orbit.radial_velocity_planet(times)
    cc_data = cc_data[str(nsysrem)]
    cc_data_coadd, cc_it, cc_oot = coadd_cross_correlation(
        cc_data,
        rv,
        rv_array,
        times,
        planet,
        data_dir=data_dir,
        cache_suffix=cache_suffix,
        load=False,
    )

    # Fit a gaussian
    p0 = [np.max(cc_data_coadd) - np.min(cc_data_coadd), 0, 10, np.min(cc_data_coadd)]
    try:
        gauss, pval = gaussfit(rv_array, cc_data_coadd, p0=p0)
    except RuntimeError:
        gauss = pval = None

    # Plot the results
    plt.plot(rv_array, cc_data_coadd, label="all observations")
    if gauss is not None:
        plt.plot(rv_array, gauss, label="best fit gaussian")
    plt.plot(rv_array, cc_it, label="in-transit")
    plt.plot(rv_array, cc_oot, label="out-of-transit")
    plt.legend()
    plt.title(f"{star.name} {planet.name} SNR{snr} Segment {segment_i}")
    plt.xlabel(r"$\Delta$RV [km/s]")
    plt.ylabel("CCF")
    plt.show()
    pass
