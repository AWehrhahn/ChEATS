# -*- coding: utf-8 -*-
"""
Get Data from Stellar DB
"""

import logging

import numpy as np

# from scipy import constants as const
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.distances import Distance
from astropy.time import Time
from data_sources.StellarDB import StellarDB as SDB
from exoorbit.bodies import Body, Planet, Star

logger = logging.getLogger(__name__)


class StellarDb:
    def __init__(self, regularize=True):
        self.backend = SDB(regularize=regularize)

    def refresh(self, name):
        self.backend.auto_fill(name)

    def get(self, name):
        """Load the data on the star from the local database, or online
        if not available locally.

        Parameters
        ----------
        name : str
            Name of the star / planet

        Returns
        -------
        star : exoorbit.bodies.Star
            recovered Star
        """

        # self.backend.auto_fill(name)
        data = self.backend.load(name)

        if "distance" in data:
            distance = data["distance"]
        elif "parallax" in data:
            distance = Distance(parallax=data["parallax"])
        else:
            distance = None

        # Convert names
        # Stellar parameters
        star = Star(
            name=name,
            mass=data.get("mass"),
            radius=data.get("radius"),
            teff=data.get("t_eff"),
            logg=data.get("logg"),
            monh=data.get("metallicity"),
            vmac=data.get("velocity_turbulence", 1 * u.km / u.s),
            coordinates=data.get("coordinates"),
            distance=distance,
            radial_velocity=data.get("radial_velocity"),
        )

        planets = {}
        for pname, p in data["planets"].items():
            planet = Planet(
                name=pname,
                radius=p.get("radius"),
                mass=p.get("mass"),
                inc=p.get("inclination", 90 * u.deg),
                sma=p.get("semi_major_axis", 0 * u.AU),
                period=p.get("period"),
                ecc=p.get("eccentricity", 0 * u.one),
                omega=p.get("periastron", 90 * u.deg),
                time_of_transit=p.get("transit_epoch"),
                transit_duration=p.get("transit_duration", 0 * u.day),
            )
            planets[pname] = planet

        star.planets = planets

        return star
