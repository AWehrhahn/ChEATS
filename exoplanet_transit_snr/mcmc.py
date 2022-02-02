# -*- coding: utf-8 -*-
import emcee
import numpy as np

# Some methods have been outsourced to _mcmc for efficiency
# There they are compiled by Cython to C code
import pyximport
from astropy import constants
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from exoorbit import orbit as _orbit
from exoorbit.bodies import Planet, Star

pyximport.install(language_level=3)
from . import _mcmc

c_light = constants.c.to_value(u.km / u.s)


class LogLikeCalculator:
    """
    Calculates the log likelihood
    by creating a forward model of the observation
    """

    def __init__(
        self,
        star: Star,
        planet: Planet,
        times: Time,
        location: EarthLocation,
        wave: np.ndarray,
        flux: np.ndarray,
        model: np.ndarray,
        ptr_wave: np.ndarray,
        ptr_flux: np.ndarray,
    ):
        # Extract only the relevant information in these dictionaries
        # We avoid using astropy.units here for tehe calculations
        # since it only slows us down
        self._star = star
        self._planet = planet
        self.star = {
            "mass": star.mass.to_value(u.M_sun),
            "radius": star.radius.to_value(u.km),
            "rv": star.radial_velocity.to_value(u.km / u.s),
        }
        self.planet = {
            "mass": planet.mass.to_value(u.M_jup),
            "radius": planet.radius.to_value(u.km),
            "sma": planet.sma.to_value(u.km),
            "period": planet.period.to_value(u.day),
            "inc": planet.inc.to_value(u.rad),
            "ecc": planet.ecc.to_value(u.one),
            "omega": planet.omega.to_value(u.rad),
            "t0": planet.time_of_transit.mjd,
        }
        self.times = times.mjd
        self.v_bary = star.coordinates.radial_velocity_correction(
            obstime=times, location=location
        ).to_value(u.km / u.s)
        self.scale = 1

        # Save the observations and planet model in use
        self.wave = wave.to_value(u.AA)
        self.flux = flux
        self.model = np.require(model, requirements="C")
        self.ptr_wave = ptr_wave.to_value(u.AA)
        self.ptr_flux = ptr_flux

        # Conversions for units between the input and the internal calculations
        self.mearth_per_mjup = (u.M_earth / u.M_jup).to(u.one)
        self.rearth_per_km = (u.R_earth / u.km).to(u.one)
        self.rsun_per_km = (u.R_sun / u.km).to(u.one)
        self.deg_per_rad = (u.deg / u.rad).to(u.one)
        self.ms_per_kms = ((u.m / u.s) / (u.km / u.s)).to(u.one)
        self.rjup_per_km = (u.R_jup / u.km).to(u.one)

        # TODO: pre calculate the prior constraints
        # self.tmin = 59672.5 # self.times.mjd.min()
        # self.tmax = 59673.0 # self.times.mjd.max()

        # Pre-calculate fixed values for the loglike calculatoon
        n = self.model.shape[1]
        self.flux = self.flux - np.nanmean(self.flux, axis=1)[:, None]
        self.sd = np.sqrt(np.nansum(self.flux ** 2) / n)
        self.sdsq = self.sd ** 2

    def __call__(self, theta: list):
        return self.log_prob(theta)

    def log_prior(self, theta: list):
        """
        Check that the input values are within reasonable values
        """
        # TODO: put some more sophisticated priors on this
        # maybe gaussian around the expected value?
        # or maybe that is too strong of a prior for an independent analysis
        # maybe:
        #  - mass_planet < mass_star
        #  - period > extremely fast, & < observation time
        #  - inclination depends on the semi major axis and sight line
        a, v_sys, m_planet, r_planet, sma, p_planet, inc, ecc, omega, t0 = theta.T
        prior = np.zeros(theta.shape[0])
        prior[a < 0] = -np.inf
        prior[(m_planet <= 0) | (m_planet > 3000)] = -np.inf
        prior[(r_planet <= 0) | (r_planet > 100)] = -np.inf
        prior[(sma <= 0) | (sma > 100)] = -np.inf
        prior[(p_planet <= 0) | (p_planet > 40)] = -np.inf
        prior[(inc <= 0) | (inc > 180)] = -np.inf
        prior[(ecc <= 0) | (ecc >= 1)] = -np.inf
        prior[(omega < 0) | (omega >= 180)] = -np.inf
        # if t0 < self.tmin or t0 > self.tmax:
        #     return -np.inf
        return prior

    # def update_parameters(self, theta: list):
    #     """ Update the parameters based on the input values """
    #     # TODO: is this slow?
    #     a, v_sys, m_planet, r_planet, sma, p_planet, inc, ecc, omega, t0 = theta

    #     self.planet["mass"] = m_planet * self.mearth_per_mjup
    #     self.planet["radius"] = r_planet * self.rearth_per_km
    #     self.planet["sma"] = sma * self.rsun_per_km
    #     self.planet["period"] = p_planet
    #     self.planet["inc"] = inc * self.deg_per_rad
    #     self.planet["ecc"] = ecc
    #     self.planet["omega"] = omega * self.deg_per_rad
    #     self.planet["t0"] = t0
    #     self.star["rv"] = v_sys
    #     self.scale = a

    @staticmethod
    # @njit(nogil=True, parallel=True)
    def _forward_model(ptr_wave, ptr_flux, wave, model, v_tot, area):
        # Calculate the shifted model spectrum

        ptr_wave_shifted = np.empty((v_tot.shape[0], ptr_wave.shape[0]))
        for i in range(model.shape[0]):
            ptr_wave_shifted[i] = ptr_wave * (1 - v_tot[i] / c_light)

        ptr_flux_new = np.empty_like(model)
        for i in range(model.shape[0]):
            ptr_flux_new[i] = np.interp(wave, ptr_wave_shifted[i], ptr_flux)

        # Use inplace operations to avoid memory usage
        for i in range(ptr_flux_new.shape[0]):
            ptr_flux_new[i] -= np.min(ptr_flux_new[i])
            ptr_flux_new[i] *= area[i]
        ptr_flux_new -= 1
        ptr_flux_new *= -model
        return ptr_flux_new

    def forward_model(self, theta):
        """Create a forward model based on the planet parameters"""

        # Set changed parameters
        a, v_sys, m_planet, r_planet, sma, p_planet, inc, ecc, omega, t0 = theta

        planet = self.planet.copy()
        star = self.star.copy()
        planet["mass"] = m_planet * self.mearth_per_mjup
        planet["radius"] = r_planet * self.rearth_per_km
        planet["sma"] = sma * self.rsun_per_km
        planet["period"] = p_planet
        planet["inc"] = inc * self.deg_per_rad
        planet["ecc"] = ecc
        planet["omega"] = omega * self.deg_per_rad
        planet["t0"] = t0
        star["rv"] = v_sys

        # Determine the offset due to velocity shifts
        v_planet = _orbit._radial_velocity_planet(
            self.times,
            self.planet["t0"],
            self.star["mass"],
            self.planet["mass"],
            self.planet["period"],
            self.planet["ecc"],
            self.planet["inc"],
            self.planet["omega"],
        )
        v_planet *= self.ms_per_kms
        # v_planet = orbit.radial_velocity_planet(self.times)
        v_sys = self.star["rv"]
        v_tot = v_planet + v_sys + self.v_bary

        # Use this to scale the transmission spectrum in time
        # TODO: This is another prior...
        area = _orbit._stellar_surface_covered_by_planet(
            self.times,
            self.planet["t0"],
            self.planet["sma"],
            self.planet["radius"],
            self.star["radius"],
            self.planet["period"],
            self.planet["ecc"],
            self.planet["inc"],
            self.planet["omega"],
        )
        if not np.all(area == 0):
            area /= np.max(area)
        else:
            return -np.inf

        # Calculate the model of the observation,
        # by doppler shifting the model planet
        ptr_flux_new = _mcmc._forward_model(
            self.ptr_wave, self.ptr_flux, self.wave[0], self.model, v_tot, area
        )
        return ptr_flux_new

    @staticmethod
    # @njit(nogil=True, parallel=True)
    def _log_like(model, data, scale, sdsq):
        a = scale
        n = model.shape[1]

        model = np.copy(model)
        for i in range(model.shape[0]):
            model[i] -= np.nanmean(model[i])

        sm = np.sqrt(np.nansum(model ** 2) / n)
        smsq = sm ** 2
        ccf = np.nansum(data * model) / n

        logL = -n / 2 * np.log(sdsq - 2 * a * ccf + a ** 2 * smsq)
        return logL

    def log_like(self, model: np.ndarray, scale: float):
        """Calculate the log likelihood of the forward model"""
        logL = _mcmc._log_like(model, self.flux, scale, self.sdsq)
        return logL

    def log_prob(self, theta: np.ndarray):
        """The main function used to calculate the whole probability"""
        # Check that parameters make physical sense
        if theta.ndim == 1:
            # if not vectorized
            theta = theta[None, :]

        prior = self.log_prior(theta)
        if np.all(prior == -np.inf):
            return prior

        # Determine the offset due to velocity shifts
        logL = np.zeros(theta.shape[0])
        for i in range(theta.shape[0]):
            ptr_flux_new = self.forward_model(theta[i])
            if np.all(ptr_flux_new == -np.inf):
                logL[i] = -np.inf
                continue

            # Calculate the log likelihood
            scale = theta[i, 0]
            logL[i] = self.log_like(ptr_flux_new, scale)

        # Remove unwanted nan values
        logL += prior
        logL = np.nan_to_num(prior, nan=-np.inf, copy=False)
        return logL

    def get_labels(self):
        return ["a", "v_sys", "mass", "radius", "sma", "per", "inc", "ecc", "w", "t0"]

    def get_truths(self):
        truths = np.array(
            [
                self.scale,
                self.star["rv"],
                self.planet["mass"] / self.mearth_per_mjup,
                self.planet["radius"] / self.rearth_per_km,
                self.planet["sma"] / self.rsun_per_km,
                self.planet["period"],
                self.planet["inc"] / self.deg_per_rad,
                self.planet["ecc"],
                self.planet["omega"] / self.deg_per_rad,
                self.times.mean(),
            ]
        )
        return truths
