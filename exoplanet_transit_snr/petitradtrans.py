# -*- coding: utf-8 -*-
from typing import Tuple
from unittest import runner

import numpy as np
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
        self.mmw = 2.33 * np.ones_like(self.pressures)

        # Parameters for the TP profile
        self.kappa_IR = 0.01  # opacity in the IR
        self.gamma = 0.4  # ratio between the opacity in the optical and the IR

        self.r_pl = None
        self.gravity = None
        self.p0 = None
        self.T_int = None
        self.T_equ = None
        self.temperature = None

    def init_temp_press_profile(self, star, planet):
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
        self.T_int = 200.0  # Internal temperature
        self.T_equ = planet.equilibrium_temperature(star.teff, star.radius).to_value(
            "K"
        )
        self.temperature = nc.guillot_global(
            self.pressures,
            self.kappa_IR,
            self.gamma,
            self.gravity,
            self.T_int,
            self.T_equ,
        )

    def run(self) -> Tuple[Quantity, np.ndarray]:
        if self.temperature is None:
            raise RuntimeError("Initialize the Temperature-Pressure Profile first")

        # Calculate transmission spectrum
        self.atmosphere.calc_transm(
            self.temperature,
            self.mass_fractions,
            self.gravity,
            self.mmw,
            R_pl=self.r_pl,
            P0_bar=self.p0,
        )

        # Wavelength in um
        wave = nc.c / self.atmosphere.freq / 1e-4
        wave = wave << u.um

        # Normalized transmission spectrum
        # Step 1: transmission radius in units of the stellar radius
        flux = self.atmosphere.transm_rad / self.r_star
        # flux -= flux.min()
        # flux /= flux.max()
        # Step 2: Get the "area" that is covered by the planet
        flux = 1 - flux ** 2

        return wave, flux
