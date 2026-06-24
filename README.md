# TrExo
Transit orbit project for Code Astro 2026

Current status: First draft of the code: you guys can try and run it on your laptop but it is not finalised as it needs the final plot part!

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

#this is basically calculating the transit depth using the following parameters: planet_radius: float and star_radius: float, which returns a float fractional brightness drop
def transit_depth(
    planet_radius,
    star_radius
):
    return (
        planet_radius / star_radius
    )**2


    def estimate_planet_radius(
    star_radius,
    normal_luminosity,
    transit_luminosity
):
    """
    Estimate planet radius from transit data.
    """

    return (
        star_radius *
        np.sqrt(
            1 -
            transit_luminosity /
            normal_luminosity
        )
    )

#this should generate the lightcurve 
def generate_lightcurve(
    depth,
    duration=2,
    points=500,
    noise_level=0
):

    time = np.linspace(
        -5,
        5,
        points
    )

    flux = np.ones(points)

    in_transit = (
        np.abs(time)
        <
        duration/2
    )

    #generating a transit light curve by creating a time array
    #reducing stellar brightness during the transit (transit method of detection for exoplanets)
    flux[in_transit] -= depth

    if noise_level > 0:

        flux += np.random.normal(
            0,
            noise_level,
            points
        )

    return time, flux


    #next should be plotting function (I have not written this yet but it should be something like:
def plot_lightcurve (
    time,
    flux
):
    #right after for the plot would be x and y labels, title, grids etc. we can add tomorrow or if any of you want to add it
