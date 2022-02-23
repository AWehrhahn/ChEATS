# -*- coding: utf-8 -*-
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from exoorbit.orbit import Orbit

from exoplanet_transit_snr.snr_estimate import (
    calculate_cohen_d_for_dataset,
    load_data,
    run_cross_correlation,
)
from exoplanet_transit_snr.stellardb import StellarDb

def coadd_cross_correlation(cc_data, rv, rv_array):
    cc_data_interp = np.zeros_like(cc_data)
    for i in range(len(cc_data)):
        for j in range(len(cc_data[0])):
            cc_data[i,j,800-3:800+4]=0 #-3 and +4 is the same bc of how python does things
            cc_data_interp[i,j] = np.interp(rv_array-(rv[i]-rv[j]).to_value("km/s"), rv_array, cc_data[i,j])
    # Co-add to sort of stack them together
    coadd_sum = np.sum(cc_data_interp, axis=(0, 1))
    '''
    phi = (times - planet.time_of_transit) / planet.period
    phi = phi.to_value(1)
    # We only care about the fraction
    phi = phi % 1
    ingress = (-planet.transit_duration / 2 / planet.period).to_value(1) % 1
    egress = (planet.transit_duration / 2 / planet.period).to_value(1) % 1
    in_transit = (phi >= ingress) | (phi <= egress)
    out_transit = (phi <= ingress) | (phi >= egress)
    coadd_sum_it = np.sum(cc_data_interp[in_transit,in_transit], axis=(0, 1))
    coadd_sum_oot = np.sum(cc_data_interp[out_transit,out_transit], axis=(0, 1))
    '''
    return coadd_sum
    #,coadd_sum_it, coadd_sum_oot

# define the names of the star and planet
# as well as the datasets within the datasets folder
star, planet = "WASP-107", "b"
datasets = {50: "WASP-107b_SNR50", 100: "WASP-107b_SNR100", 200: "WASP-107b_SNR200"}

# Load the nominal data for this star and planet from simbad/nasa exoplanet archive
sdb = StellarDb()
star = sdb.get(star)
planet = star.planets[planet]
orbit = Orbit(star, planet)

# Define the +- range of the radial velocity points,
# and the density of the sampling
rv_range = 200
rv_step = 0.25

for snr in [50]:
    # Where to find the data, might need to be adjusted
    data_dir = join(dirname(__file__), "../datasets", datasets[snr], "Spectrum_00")
    # load the data from the fits files, returns several objects
    data = load_data(data_dir, load=True)
    wave, flux, uncs, times, segments = data

    # Run the cross correlation to the next neighbour
    cc_data, rv_array = run_cross_correlation(
        data,
        nsysrem=10,
        rv_range=rv_range,
        rv_step=rv_step,
        load=False,
        data_dir=data_dir,
    )

    rv = orbit.radial_velocity_planet(times)
    cc_data = cc_data["10"]
    cc_data_coadd = coadd_cross_correlation(cc_data, rv, rv_array)
    
    #cc_data_coadd,cc_data_coadd_it,cc_data_coadd_oot = coadd_cross_correlation(cc_data, rv, rv_array)
    
    plt.plot(rv_array,cc_data_coadd)
    
    #plt.plot(rv_array,cc_data_coadd_it)
    #plt.plot(rv_array,cc_data_coadd_oot)
    
    plt.show()

    '''
    # Calculate the cohen d value for this cross correlation
    d = calculate_cohen_d_for_dataset(
        data,
        cc_data_coadd,
        star,
        planet,
        rv_range=rv_range,
        rv_step=rv_step,
        sysrem="7",
        plot=True,
        title=f"WASP-107 b SNR{snr}",
    )
    '''
