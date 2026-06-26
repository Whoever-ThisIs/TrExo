import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.patches import Circle

# =========================================================================
#**************************************************************************
#**************************************************************************
# Note: I found it easier to input directly the planet radius as a 
# parameter, rather than calculating it from the transit depth. But I 
# will keep the function here in case we want to use it later. I understand 
# the educational value of showing how the planet radius can be inferred 
# from the transit depth, but for the purpose of this simulation, 
# we can just input the planet radius directly.
# =========================================================================
# FIRST WE DO THE PLANET RADIUS ESTIMATION (Now we don't, but we can change
#  it back later)
# We are going to calculate the transit depth
# using the star_radius, normal_luminosity and 
# transit luminosity. We have:
#  
# -star_radius: float. Radius of the host star.
# -normal_luminosity: float. Normal luminosity of the 
#  host star.
# -transit_luminosity: float. Luminosity during transit, 
#  when the planet blocks part of the star.
#
# Assuming a model of transit where: 
#  $depth=(R_planet/R_star)^2$  =>  $R_planet=R_star*sqrt(depth)$
#
#def estimate_planet_radius(
#    star_radius,
#    normal_luminosity,
#    transit_luminosity
#):
#    return star_radius * np.sqrt(
#        1 - transit_luminosity / normal_luminosity
#    )
# =========================================================================
#**************************************************************************
#**************************************************************************
# =========================================================================
# We can start by selecting the initial parameters for the simulation.
#
# I want to plot two panels: the left panel will show the orbit of the 
# planet around the star, and the right panel will show the light curve 
# of the transit.
# =========================================================================
# =========================================================================
# If we start by inputing the values of each relevant parameter, here I 
# describe each one. For simplicity, star_radius will be normalized to 1.0. 
# 
# We are going to use a model of elliptical orbit, where the planet's 
# position is given by the parametric equations of an ellipse, let's 
# remember that we are seing the 2D projection of the orbit:
#
# x = a * (cos(E) - e)
# y = a * sqrt(1 - e^2) * sin(E) * cos(inc)
#
# Where:
# -a: float. Semi-major axis of the ellipse. I'm using an initial value of 2.
# -e:float. Eccentricity of the ellipse. 
# -inc: float. Is the inclination of the orbit. It can affect the 
#  total area of the transit, and therefore the depth of the light curve. 
#  Physically, it is the angle between the orbital plane and the plane of 
#  the sky. I'm using an initial value of 90 degrees, which means the 
#  orbit is edge-on.
# -planet_radius: float. Is the radius of the planet, which will affect
#  the depth of the transit. We first use a Jupiter-like planet, which is 
#  about 0.1 times the radius of the Sun. I looked up for extreme cases of
#  exoplanets, and found that "WD 1856 b" has a radius of about 8 times the 
#  radius of its host star, the white dwarf "WD 1856+534". So we can use 
#  that extreme case to see how the light curve would look like for a 
#  very large planet. 
# -E: float. Is the eccentric anomaly, in radians. 
# -time: float. Is a proxy for time, which we will use to plot the light
#  curve. We will use a range of -5 to 5, which is arbitrary but will 
#  allow us to see the transit in the light curve. We are not using 
#  real time units, but we can assume that the transit occurs in a time 
#  scale of a few days, which is typical for exoplanet transits.
#
# We will have two modes: a single image mode, where we plot the orbit 
# and the light curve for a given set of parameters, and an interactive 
# mode, where we can change the parameters and see how the orbit and 
# light curve change in real time. We must select the mode (lines 96 and 
# 97) we want and comment the other one.
# =========================================================================

star_radius = 1.0
a = 2.0
e = 0.3
inc = np.radians(90)
planet_radius = 0.1
E = np.linspace(0, 2*np.pi, 400)
time = np.linspace(0, 20, len(E))
MODE = "image"   # Two options: "image" or "interactive"
#MODE = "interactive" 

# =========================================================================
# This function computes the projected position of the planet along its
# orbit for a given value of the eccentric anomaly E. We assume an 
# elliptical orbit centered on the host star. 
# These equations generate the position of the planet in the orbital
# plane. Since an observer usually sees the system at some inclination,
# we project the orbit onto the plane of the sky by multiplying the
# y-coordinate by cos(inc).
#
# Outputs:
# -x: float or array. Projected x-coordinate of the planet.
# -y: float or array. Projected y-coordinate of the planet.
#
# The resulting coordinates describe the apparent orbit of the planet
# as seen by an observer.
# =========================================================================

def get_orbit(E):
    """Get-orbit
    Computes the projected position of the planet along its orbit for a given value of the eccentric anomaly E.
    
    Args:
        E (float or array): Eccentric anomaly in radians.

    Returns:
        x (float or array): Projected x-coordinate of the planet.
        y (float or array): Projected y-coordinate of the planet.
    
    """
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e**2) * np.sin(E)
    y *= np.cos(inc)
    return x, y

# =========================================================================
# Instead of calculating the orbital position every time we need it,
# I'm going to compute the full orbit once using all sampled values of the
# eccentric anomaly.
#
# x_orbit and y_orbit contain the coordinates of the planet along the
# complete orbit and will later be used:
#
# -To draw th orbit.
# -Calculate the projected separation between the planet and star.
# -Generate the transit light curve.
#
# Each element of x_orbit and y_orbit corresponds to one point along
# the orbital path.
# =========================================================================

x_orbit, y_orbit = get_orbit(E)

# =========================================================================
# Now we need to know how much of the stellar disk is blocked by the planet 
# at every position of the orbit. Instead of assuming a simple transit 
# depth, we can calculate the actual overlap area between the planetary 
# disk and the stellar disk.
#
# The observed stellar flux is then:
#
# flux = 1 - (A_overlap / A_star)
#
# We have three cases:
# 1) The areas do not touch and the overlap area is zero. If I input a 
#    small angle of inclination, the planet will not transit the star and 
#    the flux will remain at 1.0.
# 2) The planet is completely inside the stellar disk, and the overlap area 
#    is the area of the planet.
# 3) The planet partially overlaps the stellar disk, and we need to compute
#    the area of intersection between two circles.
# =========================================================================

def overlap_area(star_radius,planet_radius, dist):
    """ Overlap area
    Computes the area of overlap between the planetary disk and the stellar disk.

    Args:
        star_radius (float): Radius of the host star.
        planet_radius (float): Radius of the planet.
        dist (float): Projected separation between the centers of the star and the planet.

    Returns:
        float: The area of overlap between the planetary disk and the stellar disk.

    """
    if dist >= star_radius + planet_radius:
        return 0.0

    if dist <= abs(star_radius - planet_radius):
        return np.pi * min(star_radius , planet_radius)**2

    case1 = planet_radius**2 * np.arccos((dist**2 + planet_radius**2 - star_radius**2) 
                                         / (2*dist*planet_radius))
    case2 = star_radius**2 * np.arccos((dist**2 + star_radius**2 - planet_radius**2) 
                                       / (2*dist*star_radius))
    case3 = 0.5 * np.sqrt(
        (-dist + planet_radius + star_radius) *
        (dist + planet_radius - star_radius) *
        (dist - planet_radius + star_radius) *
        (dist + planet_radius + star_radius)
    )

    return case1 + case2 - case3

# =========================================================================
# Now we convert the geometric overlap between the planet and
# the star into an observable stellar flux. First we calculate:
#
# dist = sqrt(x^2 + y^2)
#
# which is the projected separation between the centers of the star
# and the planet.
#
# Then:
# -A_star: the total area of the stellar disk.
# -A_overlap: the area blocked by the planet.
#
# Therefore:
# -Flux = 1.0 when there is no transit.
# -Flux decreases as more of the star is covered.
# -The minimum flux occurs when the overlap area is maximum.
# =========================================================================

def flux_from_geometry(x, y, planet_radius):
    """ Flux from geometry
    Computes the observed stellar flux based on the geometric overlap between the planetary disk and the stellar disk.

    Args:
        x (float): Projected x-coordinate of the planet.
        y (float): Projected y-coordinate of the planet.
        planet_radius (float): Radius of the planet.

    Returns:
        float: The observed stellar flux.

    """
    dist = np.sqrt(x**2 + y**2)
    A_star = np.pi * star_radius**2
    A_overlap = overlap_area(star_radius, planet_radius, dist)

    return 1.0 - (A_overlap / A_star)

#**************************************************************************
# THE FOLLOWING LINES ARE FOR PLOTTING THE STATIC FIGURE, NOT THE ANIMATION
#**************************************************************************
# =========================================================================
# We compute the stellar flux corresponding to each point of the orbit.
# For every position:
#
# (x_orbit[i], y_orbit[i])
#
# we calculate the projected overlap between the planet and the star
# and convert it into a flux value. The resulting array 'flux' contains 
# the complete transit light curve.
###########################################################################
# IMPORTANT NOTE: 
###########################################################################
# Now I have to confess a little trick I used to make the light 
# curve look more realistic. Since the planet at the second half 
# of the orbit is movin behind the star, I set the flux to 1.0 for 
# those points.
###########################################################################
#
# We then create a figure containing two subplots:
# Left panel:
# -Displays the projected exoplanet transit.
#   *It shows:
#       -The host star represented as an orange disk.
#       -The orbital trajectory represented by a gray curve.
#       -The planet represented as a gray disk.
#
# Right panel:
# -Displays the corresponding transit light curve.
#   *We plot the light curve, normalized by the stellar flux.
#    Flux = 1.0 corresponds to the unobscured star, while values below unity 
#    indicate that part of the stellar disk is being blocked by the planet.
# =========================================================================


flux = np.array([
    flux_from_geometry(x_orbit[i], y_orbit[i], planet_radius)
    for i in range(len(E))
])

half = len(time) // 2

flux[half:] = 1.0

if MODE == "image":
    fig, (ax_orbit, ax_lc) = plt.subplots(1, 2, figsize=(11, 5))

    ax_orbit.set_aspect('equal')
    ax_orbit.set_xlim(-4.5, 4.5)
    ax_orbit.set_ylim(-3, 3)

    ax_orbit.set_title("Exoplanet Orbit")
    ax_orbit.set_xlabel("X")
    ax_orbit.set_ylabel("Y")

    star = Circle((0, 0), star_radius, color="orange")
    ax_orbit.add_patch(star)

    ax_orbit.plot(x_orbit, y_orbit, lw=1, alpha=0.6)

    planet = Circle(
        (x_orbit[-1], y_orbit[-1]),
        planet_radius,
        color="gray"
    )
    ax_orbit.add_patch(planet)

    ax_lc.set_xlim(time[0], time[-1])
    ax_lc.set_ylim(0.9, 1.02)

    ax_lc.set_title("Light Curve")
    ax_lc.set_xlabel("Time (days)")
    ax_lc.set_ylabel(" Normalized Flux")

    ax_lc.plot(time, flux, lw=2)

    plt.tight_layout()
    plt.show()
#**************************************************************************
# HERE ENDS THE STATIC FIGURE, AND THE ANIMATION STARTS
#
# We want the animation to show the planet moving along its orbit, 
# while simultaneously updating the light curve to reflect the changing 
# flux as the planet transits the star. The animation will be interactive, 
# allowing us to adjust parameters like semi-major axis, eccentricity, 
# inclination, and planet radius in real-time.
#
# We are going to use slides to adjust the parameters of the orbit and 
# the planet radius. The animation will update the orbit and light curve 
# based on the current values of the sliders. The sliders will allow us 
# to explore how different orbital configurations and planet sizes 
# affect the transit light curve. 
#**************************************************************************

if MODE == "interactive":

    fig, (ax_orbit, ax_lc) = plt.subplots(1, 2, figsize=(11, 5))
    plt.subplots_adjust(bottom=0.28)

    a_init = 2.0
    e_init = 0.3
    inc_init = 90
    rp_init = 0.1

    ax_orbit.set_aspect('equal')
    ax_orbit.set_xlim(-4.5, 4.5)
    ax_orbit.set_ylim(-3, 3)

    ax_orbit.set_title("Exoplanet Orbit (Interactive)")
    ax_orbit.set_xlabel("X")
    ax_orbit.set_ylabel("Y")

    star = Circle((0, 0), star_radius, color="orange")
    ax_orbit.add_patch(star)

    planet = Circle((x_orbit[0], y_orbit[0]), rp_init, color="gray")
    ax_orbit.add_patch(planet)

    orbit_line, = ax_orbit.plot([], [], lw=1, alpha=0.5)

    ax_lc.set_xlim(time[0], time[-1])
    ax_lc.set_ylim(0.9, 1.02)

    ax_lc.set_title("Light Curve")
    ax_lc.set_xlabel("Time (days)")
    ax_lc.set_ylabel("Normalized Flux")

    lc_line, = ax_lc.plot(time, flux, lw=2)

    ax_a   = plt.axes([0.12, 0.18, 0.7, 0.03])
    ax_e   = plt.axes([0.12, 0.13, 0.7, 0.03])
    ax_inc = plt.axes([0.12, 0.08, 0.7, 0.03])
    ax_rp  = plt.axes([0.12, 0.03, 0.7, 0.03])

    s_a   = Slider(ax_a, "Semi-major axis", 2.0, 3.0, valinit=a_init)
    s_e   = Slider(ax_e, "Eccentricity", 0.0, 0.4, valinit=e_init)
    s_inc = Slider(ax_inc, "Inclination", 0, 90, valinit=inc_init)
    s_rp  = Slider(ax_rp, "Planet radius", 0.05, 0.4, valinit=rp_init)

    frame_idx = 0
    flux_live = np.ones(len(E))

    def reset(event=None):
        """Reset
        Resets the animation to the initial state when changing any slider.

        Args:
            event: The event that triggered the reset (default is None).
        """
        global frame_idx, flux_live
        frame_idx = 0
        flux_live[:] = 1.0
        lc_line.set_data(time, flux_live)
        orbit_line.set_data([], [])
        fig.canvas.draw_idle()

    for s in [s_a, s_e, s_inc, s_rp]:
        s.on_changed(reset)

    def get_orbit_live(frame):
        """Get orbit live
        Computes the projected position of the planet along its orbit for a given frame index, using the current values of the sliders.

        Args:
            frame (int): The index of the current frame in the animation.

        Returns:
            x (float): Projected x-coordinate of the planet.
            y (float): Projected y-coordinate of the planet.
        """
        a = s_a.val
        e = s_e.val
        inc = np.radians(s_inc.val)

        E_ang = E[frame]

        x = a * (np.cos(E_ang) - e)
        y = a * np.sqrt(1 - e**2) * np.sin(E_ang)
        y *= np.cos(inc)

        return x, y

    def update(frame):
        """Update
        Updates the position of the planet and the light curve for the current frame in the animation.

        Args:
            frame (int): The index of the current frame in the animation.

        Returns:
            planet (Circle): The updated planet patch.
            lc_line (Line2D): The updated light curve line.
            orbit_line (Line2D): The updated orbit line.
        """
        global frame_idx, flux_live

        if frame_idx >= len(E):
            return planet, lc_line, orbit_line

        x, y = get_orbit_live(frame_idx)

        planet.center = (x, y)
        planet.set_radius(s_rp.val)

        xs, ys = [], []
        for i in range(frame_idx + 1):
            xi, yi = get_orbit_live(i)
            xs.append(xi)
            ys.append(yi)

        orbit_line.set_data(xs, ys)

        if frame_idx < len(E) // 2:
            flux_live[frame_idx] = flux_from_geometry(x, y, s_rp.val)
        else:
            flux_live[frame_idx] = 1.0

        lc_line.set_data(time, flux_live)

        frame_idx += 1

        return planet, lc_line, orbit_line

    ani = FuncAnimation(fig, update, frames=len(E), interval=20, blit=False)
    ani.save("exoplanet_transit.gif", writer="imagemagick", fps=30)
    
    plt.show()
