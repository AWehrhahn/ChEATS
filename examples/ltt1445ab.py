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
            cc_data[i,j,800-3:800+4]=0
            cc_data_interp[i,j] = np.interp(rv_array-(rv[i]-rv[j]).to_value("km/s"), rv_array, cc_data[i,j])
    # Co-add to sort of stack them together
    coadd_sum = np.sum(cc_data_interp, axis=(0, 1))
    return coadd_sum

star, planet = "LTT1445A", "b"
datasets = {
    50: "LTT1445Ab_SNR50_EarthAtmosphere",
    100: "LTT1445Ab_SNR100_EarthAtmosphere",
    200: "LTT1445Ab_SNR200_EarthAtmosphere",
}

sdb = StellarDb()
star = sdb.get(star)
planet = star.planets[planet]
orbit = Orbit(star, planet)

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
# Previous --

for snr in [50, 100, 200]:
    runner = init_cats(star, planet, datasets[snr])
    runner.configuration["planet_reference_spectrum"]["method"] = "petitRADTRANS"
    data = run_cross_correlation(runner, load=True)
    d = calculate_cohen_d_for_dataset(
        runner, sysrem="5", plot=True, title=f"{star} {planet} SNR{snr}"
    )

# ds = {}
# for snr in [50, 100, 200]:
#     d = calculate_cohen_d_for_dataset(star, planet, datasets[snr], "7.1", plot=True)
#     ds[snr] = d

# print(ds)
'''
