# -*- coding: utf-8 -*-
from copy import copy

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from matplotlib.pyplot import imshow

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
        star,
        planet,
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
        self.star = star
        self.planet = planet
        self.line_species = line_species
        self.rayleigh_species = rayleigh_species
        self.continuum_species = continuum_species
        # copy the dictionary, so we don't accidentally do stupid mistakes
        self.mass_fractions = copy(mass_fractions)

    def run(self, wmin: Quantity, wmax: Quantity):
        if prt is None:
            raise RuntimeError("petitRadtrans could not be imported")

        # Prepare the input
        wmin = wmin.to_value("um")
        wmax = wmax.to_value("um")
        # Initialize atmosphere
        # including the elements in the atmosphere
        atmosphere = prt.Radtrans(
            line_species=self.line_species,
            rayleigh_species=self.rayleigh_species,
            continuum_opacities=self.continuum_species,
            wlen_bords_micron=[wmin, wmax],
            mode="lbl",
        )

        # Define planet parameters
        # Planet radius
        r_pl = self.planet.radius.to_value("cm")
        # surface gravity
        gravity = self.planet.surface_gravity.to_value("cm/s**2")
        # reference pressure (for the surface gravity and radius)
        # TODO: ????
        p0 = self.planet.atm_surface_pressure.to_value("bar")

        # Pressure in bar
        # has to be equispaced in log
        print("Setup atmosphere pressures")
        pressures = np.logspace(-6, 0, 100)
        atmosphere.setup_opa_structure(pressures)

        # Define temperature pressure profile
        kappa_IR = 0.01  # opacity in the IR
        gamma = 0.4  # ratio between the opacity in the optical and the IR
        T_int = 200.0  # Internal temperature
        # T_equ = 1500.0
        T_equ = self.planet.equilibrium_temperature(
            self.star.teff, self.star.radius
        ).to_value("K")
        temperature = nc.guillot_global(
            pressures, kappa_IR, gamma, gravity, T_int, T_equ
        )

        # Define mass fractions
        mass_fractions = {}
        for elem, frac in self.mass_fractions.items():
            mass_fractions[elem] = frac * np.ones_like(temperature)

        # TODO: should this be changed depending on the molecules?
        mmw = 2.33 * np.ones_like(temperature)

        # Calculate transmission spectrum
        print("Calculate transmission Spectrum")
        atmosphere.calc_transm(
            temperature, mass_fractions, gravity, mmw, R_pl=r_pl, P0_bar=p0
        )
        # Wavelength in um
        wave = nc.c / atmosphere.freq / 1e-4
        wave = wave << u.um

        # Normalized flux spectrum
        flux = atmosphere.transm_rad / self.nc.r_sun
        flux -= flux.min()
        flux /= flux.max()
        flux = 1 - flux ** 2

        return wave, flux
