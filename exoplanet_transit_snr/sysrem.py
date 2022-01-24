# -*- coding: utf-8 -*-
# The following is an algorithm to remove systematic effects in a large set of
# light curves based on a paper by O. Tamuz, T. Mazeh and S. Zucker (2004)
# titled "Correcting systematic effects in a large set of photometric light
# curves".
from __future__ import division, print_function

import warnings

import numpy as np
from tqdm import tqdm


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
        tolerance: float = 1e-6,
    ):
        self.stars_dim, self.epoch_dim = input_data.shape
        self.input_data = input_data
        if errors is None:
            errors = np.sqrt(np.abs(input_data))
        self.errors = errors
        self.errors_squared = errors ** 2

        self.iterations = iterations
        self.tolerance = tolerance

    def run(self, num: int):
        # Create all the memory we need for SYSREM
        c = np.zeros(self.stars_dim)
        a = np.ones(self.epoch_dim)
        c_loc = np.zeros(self.stars_dim)
        a_loc = np.zeros(self.epoch_dim)
        c_numerator = np.zeros(self.stars_dim)
        c_denominator = np.zeros(self.stars_dim)
        a_numerator = np.zeros(self.epoch_dim)
        a_denominator = np.zeros(self.epoch_dim)

        # remove the median as the first component
        median = np.nanmedian(self.input_data, axis=0)
        residuals = self.input_data - median

        syserrors = [None] * (num + 1)
        syserrors[0] = median

        for n in tqdm(range(num), desc="Removing Systematic #", leave=False):
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=RuntimeWarning)
                # minimize a and c values for a number of iterations, iter
                for i in tqdm(range(self.iterations), desc="Converging", leave=False):
                    # Using the initial guesses for each a value of each epoch, minimize c for each star
                    np.nansum(
                        a * residuals / self.errors_squared, axis=1, out=c_numerator
                    )
                    np.nansum(a ** 2 / self.errors_squared, axis=1, out=c_denominator)
                    np.divide(c_numerator, c_denominator, out=c_loc)

                    # Using the c values found above, minimize a for each epoch
                    np.nansum(
                        c_loc[:, None] * residuals / self.errors_squared,
                        axis=0,
                        out=a_numerator,
                    )
                    np.nansum(
                        c_loc[:, None] ** 2 / self.errors_squared,
                        axis=0,
                        out=a_denominator,
                    )
                    np.divide(a_numerator, a_denominator, out=a_loc)

                    diff = np.nanmean((c_loc - c) ** 2) + np.nanmean((a_loc - a) ** 2)
                    # Swap the pointers to the memory
                    c, c_loc = c_loc, c
                    a, a_loc = a_loc, a
                    if self.tolerance is not None and diff < self.tolerance:
                        break

            # Create a matrix for the systematic errors:
            # syserr = np.zeros((stars_dim, epoch_dim))
            syserr = c[:, None] * a[None, :]
            syserrors[n + 1] = syserr
            # Remove the systematic error
            residuals = residuals - syserr

        return residuals, syserrors
