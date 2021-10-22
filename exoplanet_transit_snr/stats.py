# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit


def welch_t(a, b, ua=None, ub=None):
    # t = (mean(a) - mean(b)) / sqrt(std(a)**2 + std(b)**2)
    if ua is None:
        ua = a.std()
    if ub is None:
        ub = b.std()

    xa = a.mean()
    xb = b.mean()
    t = (xa - xb) / np.sqrt(ua ** 2 + ub ** 2)
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


def gaussfit(x, y, p0=None):
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

    popt, _ = curve_fit(gauss, x, y, p0=p0)
    return gauss(x, *popt), popt
