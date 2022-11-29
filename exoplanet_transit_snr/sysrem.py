# -*- coding: utf-8 -*-
# The following is an algorithm to remove systematic effects in a large set of
# light curves based on a paper by O. Tamuz, T. Mazeh and S. Zucker (2004)
# titled "Correcting systematic effects in a large set of photometric light
# curves".
from __future__ import division, print_function

import warnings

import numpy as np

# import pyximport
from scipy import constants
from scipy.interpolate import interp1d, splev, splrep
from tqdm import tqdm

# pyximport.install(language_level=3)
# from . import _sysrem

c_light = constants.c / 1e3


def generate_matrix(spectra, errors=None):

    # print("generating residuals matrix")
    epoch_dim, stars_dim = spectra.shape
    # Calculate the median along the epoch axis
    median_list = np.nanmedian(spectra, axis=0)
    # Calculate residuals from the ORIGINAL light curve
    residuals = (spectra - median_list).T
    # For the data points with quality flags != 0,
    # set the errors to a large value

    if errors is None:
        errors = np.sqrt(np.abs(spectra)).T
    else:
        errors = errors.T

    return residuals, errors, median_list, spectra


def sysrem(input_star_list, num_errors=5, iterations=100, errors=None, tolerance=1e-6):
    residuals, errors, median_list, star_list = generate_matrix(input_star_list, errors)
    stars_dim, epoch_dim = residuals.shape
    err_squared = errors ** 2

    # Define all the memory we use in the algorithm
    # Below we just operate in place to make it faster
    c = np.zeros(stars_dim)
    a = np.ones(epoch_dim)
    c_loc = np.zeros(stars_dim)
    a_loc = np.zeros(epoch_dim)
    c_numerator = np.zeros(stars_dim)
    c_denominator = np.zeros(stars_dim)
    a_numerator = np.zeros(epoch_dim)
    a_denominator = np.zeros(epoch_dim)

    # This medians.txt file is a 2D list with the first column being the medians
    # of stars' magnitudes at different epochs (the good ones) and their
    # standard deviations, so that they can be plotted against the results after
    # errors are taken out below.
    for n in tqdm(range(num_errors), desc="Removing Systematic #", leave=False):
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            # minimize a and c values for a number of iterations, iter
            for i in tqdm(range(iterations), desc="Converging", leave=False):
                # Using the initial guesses for each a value of each epoch, minimize c for each star
                np.nansum(a * residuals / err_squared, axis=1, out=c_numerator)
                np.nansum(a ** 2 / err_squared, axis=1, out=c_denominator)
                np.divide(c_numerator, c_denominator, out=c_loc)

                # Using the c values found above, minimize a for each epoch
                np.nansum(
                    c_loc[:, None] * residuals / err_squared, axis=0, out=a_numerator
                )
                np.nansum(c_loc[:, None] ** 2 / err_squared, axis=0, out=a_denominator)
                np.divide(a_numerator, a_denominator, out=a_loc)

                diff = np.nanmean((c_loc - c) ** 2) + np.nanmean((a_loc - a) ** 2)
                # Swap the pointers to the memory
                c, c_loc = c_loc, c
                a, a_loc = a_loc, a
                if tolerance is not None and diff < tolerance:
                    break

        # Create a matrix for the systematic errors:
        # syserr = np.zeros((stars_dim, epoch_dim))
        syserr = c[:, None] * a[None, :]

        # Remove the systematic error
        residuals = residuals - syserr

    # Reproduce the results in terms of medians and standard deviations for plotting
    # corrected_mags = residuals + median_list[:, None]
    # final_median = np.median(corrected_mags, axis=1)
    # std_dev = np.std(corrected_mags, axis=1)

    return residuals.T


class Sysrem:
    def __init__(
        self,
        input_data: np.ndarray,
        errors: np.ndarray = None,
        iterations: int = 100,
        tolerance: float = 1e-10,
        spec: np.ndarray = None,
        return_spec: bool = False,
    ):

        self.epoch_dim, self.stars_dim = input_data.shape
        self.input_data = input_data
        if errors is None:
            errors = np.sqrt(np.abs(input_data))
        self.errors = errors

        self.iterations = iterations
        self.tolerance = tolerance
        self.return_spec = return_spec
        self.spec = spec

    def run(self, num: int):

        errors_squared = np.require(self.errors ** 2, requirements="C")
        # data = np.require(self.input_data.T, requirements="C")
        # residuals, syserrors = _sysrem._sysrem_run(
        #     data, num, errors_squared, self.iterations, self.tolerance
        # )

        # residuals = np.asarray(residuals).T
        # syserrors[0] = np.asarray(syserrors[0])
        # for i in range(1, len(syserrors)):
        #     syserrors[i] = np.asarray(syserrors[i]).T
        # return residuals, syserrors

        # Create all the memory we need for SYSREM
        if self.spec is None:
            c = np.ones(self.stars_dim)
            c_loc = np.zeros(self.stars_dim)
            c_numerator = np.zeros(self.stars_dim)
            c_denominator = np.zeros(self.stars_dim)
        else:
            c = c_loc = self.spec

        a = np.ones(self.epoch_dim)
        a_loc = np.zeros(self.epoch_dim)
        a_numerator = np.zeros(self.epoch_dim)
        a_denominator = np.zeros(self.epoch_dim)

        # remove the median as the first component
        median = np.nanmedian(self.input_data, axis=0)
        residuals = self.input_data - median

        syserrors = [None] * (num + 1)
        syserrors[0] = median

        mask_r = np.isfinite(residuals) & (errors_squared != 0)
        mask_a = ~np.all(~mask_r, axis=1)
        mask_c = ~np.all(~mask_r, axis=0)

        for n in tqdm(range(num), desc="Removing Systematic #", leave=False):
            previous_diff = np.inf

            # minimize a and c values for a number of iterations, iter
            for i in tqdm(range(self.iterations), desc="Converging", leave=False):
                # Using the initial guesses for each a value of each epoch, minimize c for each star
                if self.spec is None:
                    np.sum(
                        a[:, None] * residuals / errors_squared,
                        axis=0,
                        out=c_numerator,
                        where=mask_r,
                    )
                    np.sum(
                        a[:, None] ** 2 / errors_squared,
                        axis=0,
                        out=c_denominator,
                        where=mask_r,
                    )
                    np.divide(
                        c_numerator,
                        c_denominator,
                        out=c_loc,
                        where=mask_c & (c_denominator != 0),
                    )

                # Using the c values found above, minimize a for each epoch
                np.sum(
                    c_loc[None, :] * residuals / errors_squared,
                    axis=1,
                    out=a_numerator,
                    where=mask_r,
                )
                np.sum(
                    c_loc[None, :] ** 2 / errors_squared,
                    axis=1,
                    out=a_denominator,
                    where=mask_r,
                )
                np.divide(
                    a_numerator,
                    a_denominator,
                    out=a_loc,
                    where=mask_a & (a_denominator != 0),
                )

                diff = np.mean((c_loc - c) ** 2, where=mask_c) + np.mean(
                    (a_loc - a) ** 2, where=mask_a
                )
                # Swap the pointers to the memory
                c, c_loc = c_loc, c
                a, a_loc = a_loc, a
                if (
                    self.tolerance is not None
                    and diff < self.tolerance
                    or diff > previous_diff
                ):
                    break
                previous_diff = diff

            # Create a matrix for the systematic errors:
            # syserr = np.zeros((stars_dim, epoch_dim))
            syserr = c[None, :] * a[:, None]
            # Remove the systematic error
            residuals -= syserr

            if self.return_spec:
                syserrors[n + 1] = np.copy(c)
            else:
                syserrors[n + 1] = syserr

        return residuals, syserrors


class Sars(Sysrem):
    def __init__(
        self,
        input_data: np.ndarray,
        x: np.ndarray,
        errors: np.ndarray = None,
        iterations: int = 100,
        tolerance: float = 1e-10,
        projection: np.ndarray = None,
    ):
        super().__init__(input_data, errors, iterations, tolerance, projection)
        self.x = x

    def run(self, num: int):

        errors_squared = np.require(self.errors ** 2, requirements="C")
        x = np.asarray(self.x)

        # remove the median as the first component
        median = np.nanmedian(self.input_data, axis=0)
        residuals = self.input_data - median

        syserrors = [None] * (num + 1)
        syserrors[0] = median

        # Create all the memory we need for SYSREM
        ca = np.ones(self.stars_dim)
        cr = np.ones(self.stars_dim)
        r = np.median(residuals, axis=1)
        a = np.median(residuals * x, axis=1)
        ca_loc = np.zeros(self.stars_dim)
        cr_loc = np.zeros(self.stars_dim)
        a_loc = np.zeros(self.epoch_dim)
        r_loc = np.zeros(self.epoch_dim)
        ca_numerator = np.zeros(self.stars_dim)
        ca_denominator = np.zeros(self.stars_dim)
        cr_numerator = np.zeros(self.stars_dim)
        cr_denominator = np.zeros(self.stars_dim)
        a_numerator = np.zeros(self.epoch_dim)
        a_denominator = np.zeros(self.epoch_dim)
        r_numerator = np.zeros(self.epoch_dim)
        r_denominator = np.zeros(self.epoch_dim)

        for n in tqdm(range(num), desc="Removing Systematic #", leave=False):
            previous_diff = np.inf
            np.median(residuals, axis=1, out=r)
            np.median(residuals * x, axis=1, out=a)
            ca[:] = 1
            cr[:] = 1

            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=RuntimeWarning)
                # minimize a and c values for a number of iterations, iter
                for i in tqdm(range(self.iterations), desc="Converging", leave=False):
                    # Using the initial guesses for each a value of each epoch, minimize c for each star

                    np.nansum(
                        residuals * x * ca / errors_squared
                        - r[:, None] * cr * x * ca / errors_squared,
                        axis=1,
                        out=a_numerator,
                    )
                    np.nansum(
                        x ** 2 * ca ** 2 / errors_squared, axis=1, out=a_denominator
                    )
                    np.divide(a_numerator, a_denominator, out=a_loc)
                    a_loc = np.nan_to_num(a_loc, nan=0, copy=False)

                    np.nansum(
                        residuals * cr / errors_squared
                        - a_loc[:, None] * cr * x * ca / errors_squared,
                        axis=1,
                        out=r_numerator,
                    )
                    np.nansum(cr ** 2 / errors_squared, axis=1, out=r_denominator)
                    np.divide(r_numerator, r_denominator, out=r_loc)
                    r_loc = np.nan_to_num(r_loc, nan=0, copy=False)

                    np.nansum(
                        residuals * x * a_loc[:, None] / errors_squared
                        - cr * a_loc[:, None] * x * r_loc[:, None] / errors_squared,
                        axis=0,
                        out=ca_numerator,
                    )
                    np.nansum(
                        x ** 2 * a_loc[:, None] ** 2 / errors_squared,
                        axis=0,
                        out=ca_denominator,
                    )
                    np.divide(ca_numerator, ca_denominator, out=ca_loc)
                    ca_loc = np.nan_to_num(ca_loc, nan=0, copy=False)

                    np.nansum(
                        residuals * r_loc[:, None] / errors_squared
                        - ca_loc * a_loc[:, None] * r_loc[:, None] / errors_squared,
                        axis=0,
                        out=cr_numerator,
                    )
                    np.nansum(
                        r_loc[:, None] ** 2 / errors_squared, axis=0, out=cr_denominator
                    )
                    np.divide(cr_numerator, cr_denominator, out=cr_loc)
                    cr_loc = np.nan_to_num(cr_loc, nan=0, copy=False)

                    diff = np.nanmax(
                        np.abs(a_loc[:, None] * x * ca_loc + r_loc[:, None] * cr_loc)
                    )
                    # Swap the pointers to the memory
                    ca, ca_loc = ca_loc, ca
                    cr, cr_loc = cr_loc, cr
                    a, a_loc = a_loc, a
                    r, r_loc = r_loc, r
                    if (
                        self.tolerance is not None
                        and diff < self.tolerance
                        or diff > previous_diff
                    ):
                        break
                    previous_diff = diff

            # Create a matrix for the systematic errors:
            # syserr = np.zeros((stars_dim, epoch_dim))
            syserr = a[:, None] * x * ca + r[:, None] * cr
            syserrors[n + 1] = syserr
            # Remove the systematic error
            residuals -= syserr

        return residuals, syserrors


class SysremWithProjection2(Sysrem):
    def __init__(
        self,
        input_data: np.ndarray,
        projection: np.ndarray,
        segments: np.ndarray,
        errors: np.ndarray = None,
        iterations: int = 100,
        tolerance: float = 1e-10,
        return_spec: bool = False,
    ):
        super().__init__(input_data, errors, iterations, tolerance, None, return_spec)
        self.projection = projection
        self.segments = segments

    def run(self, num: int):
        errors_squared = np.require(self.errors ** 2, requirements="C")

        # Create all the memory we need for SYSREM
        c = np.ones(self.stars_dim)
        c_loc = np.zeros(self.stars_dim)
        c_numerator = np.zeros(self.stars_dim)
        c_denominator = np.zeros(self.stars_dim)

        t = np.ones(self.stars_dim)
        t_loc = np.zeros(self.stars_dim)
        t_numerator = np.zeros(self.stars_dim)
        t_denominator = np.zeros(self.stars_dim)

        a = np.ones(self.epoch_dim)
        a_loc = np.zeros(self.epoch_dim)
        a_numerator = np.zeros(self.epoch_dim)
        a_denominator = np.zeros(self.epoch_dim)

        model = np.zeros(self.input_data.shape)

        # remove the median as the first component
        median = np.nanmedian(self.input_data, axis=0)
        residuals = np.copy(self.input_data)
        c[:] = median

        syserrors = [None] * (num + 1)
        return_spec = [None] * (num + 1)
        return_tell = [None] * (num + 1)
        return_am = [None] * (num + 1)
        syserrors[0] = median
        return_spec[0] = np.copy(median)

        for n in tqdm(range(num), desc="Removing Systematic #", leave=False):
            previous_diff = np.inf

            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=RuntimeWarning)
                # minimize a and c values for a number of iterations, iter
                for _ in tqdm(range(self.iterations), desc="Converging", leave=False):

                    # Determine the stellar spectrum for each time point
                    for i in tqdm(range(self.epoch_dim)):
                        proj = self.projection[i]
                        # For each wavelength point in the unshifted spectrum
                        for k in tqdm(range(self.stars_dim), leave=False):
                            p = proj[:, [k]]
                            idx = p.indices
                            values = p.data
                            if idx.size != 0:
                                # sum of all non k elements
                                # is the sum of all elements minus k
                                tmp = proj[idx, :] @ c - values * c[k]
                                c_numerator[k] += np.sum(
                                    a[i]
                                    * t[idx]
                                    / errors_squared[i, idx]
                                    * values
                                    * (residuals[i, idx] - a[i] * t[idx] * tmp)
                                )
                                c_denominator[k] += np.sum(
                                    a[i] ** 2
                                    * t[idx] ** 2
                                    / errors_squared[i, idx]
                                    * values ** 2
                                )

                    c_loc[:] = c_numerator / c_denominator

                    # Determine the tellurics
                    for i in range(self.epoch_dim):
                        model[i] = self.projection[i] @ c_loc

                    t_numerator[:] = np.nansum(
                        residuals * a[:, None] * model / errors_squared, axis=0
                    )
                    t_denominator[:] = np.nansum(
                        model ** 2 * a[:, None] ** 2 / errors_squared, axis=0
                    )
                    t_loc[:] = t_numerator / t_denominator

                    # Using the c values found above, minimize a for each epoch
                    a_numerator[:] = np.nansum(
                        residuals * t_loc[None, :] * model / errors_squared, axis=1
                    )
                    a_denominator[:] = np.nansum(
                        model ** 2 * t_loc[None, :] ** 2 / errors_squared, axis=1
                    )
                    a_loc[:] = a_numerator / a_denominator

                    for i in range(self.epoch_dim):
                        model[i] = (self.projection[i] @ c_loc) * t_loc * a_loc[i]

                    diff = np.nanmean((model - residuals) ** 2)
                    # Swap the pointers to the memory
                    c, c_loc = c_loc, c
                    t, t_loc = t_loc, t
                    a, a_loc = a_loc, a
                    if (
                        self.tolerance is not None
                        and diff < self.tolerance
                        or diff > previous_diff
                    ):
                        break
                    previous_diff = diff

            # Create a matrix for the systematic errors:
            syserr = np.zeros((self.epoch_dim, self.stars_dim))
            for i in range(self.epoch_dim):
                syserr[i] = (self.projection[i] @ c) * t * a[i]
            # Remove the systematic error
            residuals -= syserr

            return_spec[n + 1] = np.copy(c)
            return_tell[n + 1] = np.copy(t)
            return_am[n + 1] = np.copy(a)
            syserrors[n + 1] = syserr

        return residuals, syserrors, return_spec, return_tell, return_am


class SysremWithProjection(Sysrem):
    """
    SYSREM but accounting for barycentric velocity

    Model = AM(t) * exp(-Tell(w) * tau) * Spec(w, t)

    where Spec is shifted according the barycentric velocity
    for each observation
    """

    def __init__(
        self,
        wave: np.ndarray,
        input_data: np.ndarray,
        rv: np.ndarray,
        airmass: np.ndarray,
        errors: np.ndarray = None,
        iterations: int = 100,
        tolerance: float = 1e-10,
        return_spec: bool = False,
    ):
        super().__init__(input_data, errors, iterations, tolerance, None, return_spec)
        self.wave = wave
        self.rv = rv
        self.airmass = airmass
        self.gamma = np.sqrt((1 - rv / c_light) / (1 + rv / c_light))

    def model(self, spec, tell, am, data: np.ndarray = None):
        if data is None:
            data = np.zeros((self.epoch_dim, self.stars_dim))

        _c_interpolator = interp1d(
            self.wave, spec, kind="cubic", copy=False, fill_value="extrapolate"
        )
        c_interpolator = lambda i: _c_interpolator(self.wave * self.gamma[i])
        for i in range(self.epoch_dim):
            data[i] = c_interpolator(i) * am[i] * np.exp(-tell * self.airmass[i])

        return data

    def newton(self, obs, spec, tell, am):
        # Initial guess, assuming all Z eq 1
        old = None
        z = self.airmass
        s = np.zeros((self.epoch_dim, self.stars_dim))
        spec2 = splrep(self.wave, spec, k=3)
        for j in range(self.epoch_dim):
            s[j] = splev(self.wave * self.gamma[j], spec2, ext=3)

        for _ in tqdm(range(10), leave=False):
            t = np.exp(-tell[None, :] * z[:, None])
            a = 2 * z[:, None] ** 2 * s ** 2 * t ** 2 * am[:, None] ** 2
            a -= z[:, None] ** 2 * obs * s * t * am[:, None]
            b = z[:, None] * (am[:, None] * s * t - obs) * am[:, None] * s * t

            a = np.sum(a, axis=0)
            b = np.sum(b, axis=0)
            a[a <= 1e-8] = 1e-8
            delta_tau = b / a

            tell += delta_tau
            tell = np.clip(tell, 0, None, out=tell)
            if (
                self.tolerance is not None
                and old is not None
                and np.max(np.abs(old - tell)) < self.tolerance
            ):
                break
            old = np.copy(tell)
        return tell

    def run(self, num: int):
        errors_squared = np.require(self.errors ** 2, requirements="C")

        # Create all the memory we need for SYSREM
        # Initial guess for the spectrum
        c = np.zeros(self.stars_dim)
        c_numerator = np.zeros(self.stars_dim)
        c_denominator = np.zeros(self.stars_dim)
        t = np.zeros(self.stars_dim)
        a = np.zeros(self.epoch_dim)

        # Reserve some memory for the model
        residuals = np.copy(self.input_data)
        residuals = np.nan_to_num(residuals, nan=1, copy=False)
        model = np.zeros(self.input_data.shape)

        syserrors = [None] * (num + 1)
        return_spec = [None] * (num + 1)
        return_tell = [None] * (num + 1)
        return_am = [None] * (num + 1)

        for n in tqdm(range(num), desc="Removing Systematic #", leave=False):
            obs_bezier = [
                splrep(self.wave, residuals[i], k=3) for i in range(self.epoch_dim)
            ]
            # Initial guesses for the solution
            c[:] = np.nanmedian(residuals, axis=0)

            t[:] = 1
            t[:] = -np.log(t)
            t = np.nan_to_num(t, nan=0.2, copy=False)

            a[:] = 1
            model = self.model(c, t, a, data=model)
            a[:] = np.nanmedian(residuals / model, axis=1)
            model = self.model(c, t, a, data=model)
            previous_diff = np.nanmean((model - residuals) ** 2 / errors_squared)
            # tqdm.write(f"{previous_diff:.3g}")

            # minimize a and c values for a number of iterations, iter
            for it in tqdm(range(self.iterations), desc="Converging", leave=False):

                # Stellar Spectrum
                c_denominator[:] = 0
                c_numerator[:] = 0
                tau_bezier = splrep(self.wave, t, k=3)
                for i in range(self.epoch_dim):  # TODO: make this faster
                    tau_shift = splev(self.wave / self.gamma[i], tau_bezier, ext=3)
                    obs_shift = splev(self.wave / self.gamma[i], obs_bezier[i], ext=3)
                    c_numerator += (
                        a[i]
                        * np.exp(-tau_shift * self.airmass[i])
                        * obs_shift
                        # / errors_squared[i]
                    )
                    c_denominator += (
                        a[i] ** 2
                        * np.exp(-2 * tau_shift * self.airmass[i])
                        # / errors_squared[i]
                    )
                c[:] = c_numerator / c_denominator
                c = np.nan_to_num(c, nan=0, posinf=0, neginf=0, copy=False)
                c = np.clip(c, 0, 2, out=c)

                # Tellurics
                # TODO: Solve with Newton Raphson (or similar), #TODO: FASTER!
                t[:] = self.newton(residuals, c, t, a)

                # Scaling factor "airmass"
                c_bezier = splrep(self.wave, c, k=3)
                for i in range(self.epoch_dim):
                    c_shift = splev(self.wave * self.gamma[i], c_bezier, ext=3)
                    a_numerator = np.nansum(
                        residuals[i]
                        * c_shift
                        * np.exp(-t * self.airmass[i])
                        / errors_squared[i]
                    )
                    a_denominator = np.nansum(
                        c_shift ** 2
                        * np.exp(-2 * t * self.airmass[i])
                        / errors_squared[i]
                    )
                    a[i] = a_numerator / a_denominator
                a = np.nan_to_num(a, nan=1, posinf=1, neginf=1, copy=False)

                # Intermediate Model
                model = self.model(c, t, a, data=model)
                diff = np.nanmean((model - residuals) ** 2 / errors_squared)
                # tqdm.write(f"{diff:.3g}")

                if it > 20 and (
                    self.tolerance is not None
                    and abs(diff - previous_diff) < self.tolerance
                    or diff > previous_diff
                ):
                    break
                previous_diff = diff

            # plt.clf()
            # plt.subplot(311)
            # plt.plot(c)
            # plt.plot(self.spec, "r")
            # # plt.ylim(0, 1.1)
            # plt.subplot(312)
            # plt.plot(np.exp(-t))
            # # plt.ylim(0, 1.1)
            # plt.subplot(313)
            # plt.plot(a)
            # plt.show()

            # Create a matrix for the systematic errors:
            syserr = self.model(c, t, a)
            # Remove the systematic error
            residuals -= syserr

            return_spec[n + 1] = np.copy(c)
            return_tell[n + 1] = np.copy(t)
            return_am[n + 1] = np.copy(a)
            syserrors[n + 1] = syserr

        return residuals, syserrors, return_spec, return_tell, return_am
