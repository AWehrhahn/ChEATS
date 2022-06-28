# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit


def air2vac(wl_air, copy=True):
    """
    Convert wavelengths in air to vacuum wavelength
    in Angstrom
    Author: Nikolai Piskunov
    """
    if copy:
        wl_vac = np.copy(wl_air)
    else:
        wl_vac = np.asarray(wl_air)
    wl_air = np.asarray(wl_air)

    ii = np.where(wl_air > 1999.352)

    sigma2 = (1e4 / wl_air[ii]) ** 2  # Compute wavenumbers squared
    fact = (
        1e0
        + 8.336624212083e-5
        + 2.408926869968e-2 / (1.301065924522e2 - sigma2)
        + 1.599740894897e-4 / (3.892568793293e1 - sigma2)
    )
    wl_vac[ii] = wl_air[ii] * fact  # Convert to vacuum wavelength

    return wl_vac


def vac2air(wl_vac, copy=True):
    """
    Convert vacuum wavelengths to wavelengths in air
    in Angstrom
    Author: Nikolai Piskunov
    """
    if copy:
        wl_air = np.copy(wl_vac)
    else:
        wl_air = np.asarray(wl_vac)
    wl_vac = np.asarray(wl_vac)

    # Only works for wavelengths above 2000 Angstrom
    ii = np.where(wl_vac > 2e3)

    sigma2 = (1e4 / wl_vac[ii]) ** 2  # Compute wavenumbers squared
    fact = (
        1e0
        + 8.34254e-5
        + 2.406147e-2 / (130e0 - sigma2)
        + 1.5998e-4 / (38.9e0 - sigma2)
    )
    wl_air[ii] = wl_vac[ii] / fact  # Convert to air wavelength
    return wl_air


def welch_t(a, b, ua=None, ub=None):
    # t = (mean(a) - mean(b)) / sqrt(std(a)**2 + std(b)**2)
    if ua is None:
        ua = a.std()
    if ub is None:
        ub = b.std()

    xa = a.mean()
    xb = b.mean()
    t = np.abs(xa - xb) / np.sqrt(ua ** 2 + ub ** 2)
    return t


def cohen_d(a, b):
    sa = a.std()
    sb = b.std()
    s = ((a.size - 1) * sa ** 2 + (b.size - 1) * sb ** 2) / (a.size + b.size - 2)
    s = np.sqrt(s)
    d = np.abs(a.mean() - b.mean()) / s
    return d


def gauss(x, height, mu, sig, floor):
    return height * np.exp(-(((x - mu) / sig) ** 2) / 2) + floor


def gaussfit(x, y, p0=None, **kwargs):
    """
    Fit a simple gaussian to data

    gauss(x, a, mu, sigma, floor) = a * exp(-z**2/2) + floor
    with z = (x - mu) / sigma

    Parameters
    ----------
    x : array(float)
        x values
    y : array(float)
        y values
    Returns
    -------
    gauss(x), parameters
        fitted values for x, fit paramters (a, mu, sigma)
    """

    if p0 is None:
        p0 = [np.max(y) - np.min(y), 0, 1, np.min(y)]

    popt, _ = curve_fit(gauss, x, y, p0=p0, **kwargs)
    return gauss(x, *popt), popt
