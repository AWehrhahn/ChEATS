# -*- coding: utf-8 -*-
from exoplanet_transit_snr.snr_estimate import (
    calculate_cohen_d_for_dataset,
    init_cats,
    run_cross_correlation,
)

star, planet = "LTT1445A", "b"
datasets = {
    50: "LTT1445Ab_SNR50_EarthAtmosphere",
    100: "LTT1445Ab_SNR100_EarthAtmosphere",
    200: "LTT1445Ab_SNR200_EarthAtmosphere",
}


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
