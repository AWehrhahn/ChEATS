# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
from astropy import constants
from astropy import units as u
from astropy.units import Quantity

try:
    # Importing petitRADTRANS takes forever...
    import petitRADTRANS as prt
    from petitRADTRANS import nat_cst as nc

except (ImportError, IOError):
    prt = None
    print(
        "petitRadtrans could not be imported, "
        "please fix that if you want to use it for model spectra"
    )


class petitRADTRANS:
    def __init__(
        self,
        wmin,
        wmax,
        line_species=("H2O", "CO", "CH4", "CO2"),
        rayleigh_species=("H2", "He"),
        continuum_species=("H2-H2", "H2-He"),
        mass_fractions={
            "H2": 0.74,
            "He": 0.24,
            "H2O": 1e-3,
            "CO": 1e-2,
            "CO2": 1e-5,
            "CH4": 1e-6,
        },
        mode="transmission",
    ):
        if prt is None:
            raise RuntimeError("petitRadtrans could not be imported")

        # Initialize atmosphere
        # including the elements in the atmosphere
        wmin = wmin.to_value("um")
        wmax = wmax.to_value("um")
        self.atmosphere = prt.Radtrans(
            line_species=line_species,
            rayleigh_species=rayleigh_species,
            continuum_opacities=continuum_species,
            wlen_bords_micron=[wmin, wmax],
            mode="lbl",
        )

        # Pressure in bar
        # has to be equispaced in log
        self.pressures = np.logspace(-6, 0, 100)
        self.atmosphere.setup_opa_structure(self.pressures)

        # Define mass fractions
        self.mass_fractions = {}
        for elem, frac in mass_fractions.items():
            self.mass_fractions[elem] = frac * np.ones_like(self.pressures)

        # TODO: should this be changed depending on the molecules?
        MMW_H2 = 2.01588
        MMW_He = 4.002602
        mmw = mass_fractions["H2"] * MMW_H2 + mass_fractions["He"] * MMW_He
        self.mmw = mmw * np.ones_like(self.pressures)

        # Parameters for the TP profile
        self.kappa_IR = 0.01  # opacity in the IR
        self.gamma = 0.4  # ratio between the opacity in the optical and the IR

        self.r_pl = None
        self.gravity = None
        self.p0 = None
        self.T_int = None
        self.T_equ = None
        self.T_star = None
        self.temperature = None
        self.mode = mode

    def init_temp_press_profile(self, star, planet, inversion=False):
        # Define star parameters
        self.T_star = star.teff.to_value("K")
        self.R_star = star.radius.to_value("R_sun")
        self.sma = planet.sma.to_value("AU")
        # Define planet parameters
        # Planet radius
        self.r_pl = planet.radius.to_value("cm")
        self.r_star = star.radius.to_value("cm")
        # surface gravity
        self.gravity = planet.surface_gravity.to_value("cm/s**2")
        # reference pressure (for the surface gravity and radius)
        # TODO: ????
        self.p0 = planet.atm_surface_pressure.to_value("bar")

        # Define temperature pressure profile
        self.T_equ = planet.equilibrium_temperature(star.teff, star.radius).to_value(
            "K"
        )

        # Using Eq2 from Thorngren, Gao, Fortney, 2019
        # https://iopscience.iop.org/article/10.3847/2041-8213/ab43d0/pdf
        self.T_int = planet.intrinsic_temperature(star.teff, star.radius).to_value("K")
        # self.T_int = 100.0  # Intrinsic temperature

        self.temperature = nc.guillot_global(
            self.pressures,
            self.kappa_IR,
            self.gamma,
            self.gravity,
            self.T_int,
            self.T_equ,
        )
        # invert
        if inversion:
            self.temperature = self.temperature[::-1]

        # import matplotlib.pyplot as plt
        # plt.plot(self.temperature, self.pressures)
        # plt.yscale("log")
        # plt.gca().invert_yaxis()
        # plt.xlabel("T [K]")
        # plt.ylabel("P [bar]")
        # plt.show()
        # pass

    def run(self) -> Tuple[Quantity, np.ndarray]:
        if self.temperature is None:
            raise RuntimeError("Initialize the Temperature-Pressure Profile first")

        # Calculate transmission spectrum
        if self.mode == "transmission":
            self.atmosphere.calc_transm(
                self.temperature,
                self.mass_fractions,
                self.gravity,
                self.mmw,
                R_pl=self.r_pl,
                P0_bar=self.p0,
            )
            # Normalized transmission spectrum
            # Step 1: transmission radius in units of the stellar radius
            flux = self.atmosphere.transm_rad / self.r_star
            # Step 2: Get the "area" that is covered by the planet
            flux = 1 - flux ** 2
        elif self.mode == "emission":
            self.atmosphere.calc_flux(
                self.temperature,
                self.mass_fractions,
                self.gravity,
                self.mmw,
                Tstar=self.T_star,
                Rstar=self.R_star,
                semimajoraxis=self.sma,
            )
            flux = (
                self.atmosphere.flux / 1e-6
            )  # 1e-6 erg cm**-2 s**-1 Hz**-1 = 1e-9 J/m**2
            # Just normalize it
            flux /= np.max(flux)
            flux = 1 - flux
        else:
            raise ValueError(
                f"Expected mode of ('transmission', 'emission') but received {self.mode}"
            )

        # Wavelength in um
        wave = nc.c / self.atmosphere.freq / 1e-4
        wave = wave << u.um

        return wave, flux
