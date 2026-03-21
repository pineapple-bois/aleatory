"""
Run-and-Tumble Particle in 2D
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from aleatory.processes.base import StochasticProcess
from aleatory.utils.plotters_2d import plot_coordinate_paths
from aleatory.utils.utils import get_times


class RTP2D(StochasticProcess):
    r"""

    Run-and-Tumble Particle in Two Dimensions
    =========================================

    Notes
    -----

    A run-and-tumble particle in two dimensions is a stochastic process
    with position and orientation, where the particle moves ballistically
    at constant speed between random reorientation events called tumbles.

    In this implementation, the process
    :math:`\{(X_t, Y_t, \theta_t) : t \geq 0\}` evolves so that the
    particle moves with velocity
    :math:`v(\cos(\theta_t),\sin(\theta_t))`, while tumbles occur at
    Poisson rate :math:`\lambda`.

    Between tumble events, the orientation remains constant. At a tumble,
    the heading is reset instantaneously by drawing a new angle uniformly
    from :math:`[0,2\pi)`.

    Translational diffusion may optionally be included, in which case the
    position coordinates satisfy the discrete-time approximation

    .. math::

       X_{k+1}
       =
       X_k
       +
       v \cos(\theta_k) \Delta t
       +
       \sqrt{2 D_{\mathrm{T}} \Delta t}\,\xi_k^{(x)}

    .. math::

       Y_{k+1}
       =
       Y_k
       +
       v \sin(\theta_k) \Delta t
       +
       \sqrt{2 D_{\mathrm{T}} \Delta t}\,\xi_k^{(y)}

    where :math:`D_{\mathrm{T}}` is the translational diffusion coefficient
    and :math:`\xi_k^{(x)}` and :math:`\xi_k^{(y)}` are independent standard
    normal random variables.

    Thus the particle exhibits persistent ballistic motion on each run,
    while randomness enters through the tumble times, the post-tumble
    headings, and optionally through translational diffusion.

    When :math:`\lambda = 0`, the motion reduces to deterministic ballistic
    transport along a fixed direction, up to any translational diffusion.
    When the tumble rate is large, directional memory is lost rapidly.

    The sample paths are generated on a finite interval :math:`[0,T]` using
    a discrete-time approximation in which a tumble occurs during a timestep
    with probability :math:`\lambda \Delta t`.

    The planar sample returned by :meth:`sample` consists of the coordinate
    pair :math:`(X_t,Y_t)`. The most recently simulated orientation path is
    stored on the instance and may be accessed via the :attr:`last_theta`
    property, or returned explicitly using :meth:`sample_with_orientation`.

    Constructor, Methods, and Attributes
    ------------------------------------

    """

    def __init__(
        self,
        speed=1.0,
        tumble_rate=1.0,
        translational_diffusion=0.0,
        T=1.0,
        x0=0.0,
        y0=0.0,
        theta0=0.0,
        rng=None,
    ):
        super().__init__(T=T, rng=rng)

        if speed < 0.0:
            raise ValueError("speed must be non-negative")
        if tumble_rate < 0.0:
            raise ValueError("tumble_rate must be non-negative")
        if translational_diffusion < 0.0:
            raise ValueError("translational_diffusion must be non-negative")

        self.speed = speed
        self.tumble_rate = tumble_rate
        self.translational_diffusion = translational_diffusion
        self.x0 = x0
        self.y0 = y0
        self.theta0 = theta0

        self.name = (
            "Run-and-Tumble Particle "
            rf"$(v_0={self.speed},\ D_{{\mathrm{{T}}}}={self.translational_diffusion},\ \lambda={self.tumble_rate})$"
        )

        self.n = None
        self.times = None
        self._last_theta = None
        self._last_x = None
        self._last_y = None
        self._last_tumble_indices = None


    def __str__(self):
        return (
            f"Run-and-Tumble Particle with "
            f"speed={self.speed}, tumble_rate={self.tumble_rate}, "
            f"translational_diffusion={self.translational_diffusion}"
        )


    def __repr__(self):
        return (
            f"RTP2D("
            f"speed={self.speed}, "
            f"tumble_rate={self.tumble_rate}, "
            f"translational_diffusion={self.translational_diffusion}, "
            f"T={self.T}, "
            f"x0={self.x0}, "
            f"y0={self.y0}, "
            f"theta0={self.theta0}"
            f")"
        )


    @property
    def last_theta(self):
        """Most recently simulated orientation path."""
        return self._last_theta


    @property
    def last_position(self):
        """Most recently simulated planar path as a pair ``(x, y)``."""
        if self._last_x is None or self._last_y is None:
            return None
        return self._last_x, self._last_y


    @property
    def last_tumble_indices(self):
        """Indices at which tumbles occurred in the most recent simulation."""
        return self._last_tumble_indices


    def _sample(self, n):
        if n < 2:
            raise ValueError("n must be at least 2")

        dt = self.T / (n - 1)

        tumble_probability = self.tumble_rate * dt
        if tumble_probability > 1.0:
            raise ValueError(
                "tumble_rate * dt must not exceed 1.0 in this discrete-time approximation"
            )

        theta = np.empty(n)
        theta[0] = self.theta0
        tumble_indices = []

        for k in range(n - 1):
            if self.rng.uniform() < tumble_probability:
                theta[k + 1] = self.rng.uniform(0.0, 2.0 * np.pi)
                tumble_indices.append(k + 1)
            else:
                theta[k + 1] = theta[k]

        dx_active = self.speed * np.cos(theta[:-1]) * dt
        dy_active = self.speed * np.sin(theta[:-1]) * dt

        if self.translational_diffusion > 0.0:
            dx_noise = self.rng.normal(
                loc=0.0,
                scale=np.sqrt(2.0 * self.translational_diffusion * dt),
                size=n - 1,
            )
            dy_noise = self.rng.normal(
                loc=0.0,
                scale=np.sqrt(2.0 * self.translational_diffusion * dt),
                size=n - 1,
            )
        else:
            dx_noise = np.zeros(n - 1)
            dy_noise = np.zeros(n - 1)

        x = np.empty(n)
        y = np.empty(n)
        x[0] = self.x0
        y[0] = self.y0
        x[1:] = self.x0 + np.cumsum(dx_active + dx_noise)
        y[1:] = self.y0 + np.cumsum(dy_active + dy_noise)

        self._last_theta = theta
        self._last_x = x
        self._last_y = y
        self._last_tumble_indices = np.array(tumble_indices, dtype=int)
        return x, y


    def sample(self, n):
        self.n = n
        self.times = get_times(self.T, n)
        return self._sample(n)


    def sample_with_orientation(self, n):
        self.n = n
        self.times = get_times(self.T, n)
        x, y = self._sample(n)
        return x, y, self._last_theta.copy()


    def simulate(self, n, N):
        self.n = n
        self.times = get_times(self.T, n)
        simulations = []
        for _ in range(N):
            x, y = self._sample(n)
            simulations.append((x, y))
        return simulations


    def simulate_with_orientation(self, n, N):
        self.n = n
        self.times = get_times(self.T, n)
        simulations = []
        for _ in range(N):
            x, y = self._sample(n)
            simulations.append((x, y, self._last_theta.copy()))
        return simulations


    def plot_sample(
            self,
            n,
            coordinates=False,
            title=None,
            suptitle=None,
            mode="linear",
            ax=None,
            **fig_kw,
    ):
        if coordinates:
            return self.plot_sample_coordinates(
                n=n,
                title=title,
                suptitle=suptitle,
                mode=mode,
                ax=ax,
                **fig_kw,
            )
        return self.plot_sample_2d(
            n=n,
            title=title,
            suptitle=suptitle,
            ax=ax,
            **fig_kw,
        )


    def plot_sample_coordinates(
            self,
            n,
            title=None,
            suptitle=None,
            mode="linear",
            ax=None,
            **fig_kw,
    ):
        chart_suptitle = suptitle if suptitle is not None else self.name
        x, y, _ = self.sample_with_orientation(n)

        ax = plot_coordinate_paths(
            times=self.times,
            paths1=[x],
            paths2=[y],
            title=title,
            suptitle=chart_suptitle,
            mode=mode,
            ax=ax,
            **fig_kw,
        )
        return ax


    def plot_orientation_and_tumbles(
            self,
            n,
            theta_title="RTP orientation path",
            tumble_title="Tumble event times",
            ax_theta=None,
            ax_tumbles=None,
            **fig_kw,
    ):
        _, _, theta = self.sample_with_orientation(n)

        created_axes = False

        if ax_theta is None or ax_tumbles is None:
            fig, (ax_theta, ax_tumbles) = plt.subplots(
                2,
                1,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
                **fig_kw,
            )
            created_axes = True
        else:
            fig = ax_theta.figure

        ax_theta.plot(self.times, theta, lw=1.2)
        ax_theta.set_ylabel(r"$\theta(t)$")
        ax_theta.set_title(theta_title)
        ax_theta.grid(True)

        tumble_indices = self.last_tumble_indices
        if tumble_indices is not None and len(tumble_indices) > 0:
            tumble_times = self.times[tumble_indices]
            ax_tumbles.plot(
                tumble_times,
                np.full_like(tumble_times, 0.5, dtype=float),
                linestyle="None",
                marker="|",
                markersize=18,
                markeredgewidth=1.5,
            )

        ax_tumbles.set_xlabel("$t$")
        ax_tumbles.set_yticks([])
        ax_tumbles.set_ylim(0.0, 1.0)
        ax_tumbles.set_title(tumble_title)
        ax_tumbles.grid(True)

        if created_axes:
            fig.tight_layout()

        return ax_theta, ax_tumbles


    def plot_sample_2d(
            self,
            n,
            title=None,
            suptitle=None,
            color_by="time",
            ax=None,
            show_colorbar=True,
            **fig_kw,
    ):
        cmap = plt.get_cmap()
        chart_suptitle = suptitle if suptitle is not None else self.name
        x, y, _ = self.sample_with_orientation(n)

        if color_by == "time":
            values = self.times
            norm = Normalize(vmin=values.min(), vmax=values.max())
            colors_indices = cmap(norm(values))
            label_title = "Time"
        elif color_by == "distance":
            values = np.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2)
            norm = Normalize(vmin=values.min(), vmax=values.max())
            colors_indices = cmap(norm(values))
            label_title = "Distance to initial position"
        else:
            raise ValueError("color_by must be either 'time' or 'distance'")

        created_fig = False

        if ax is None:
            fig, ax = plt.subplots(**fig_kw)
            created_fig = True
        else:
            fig = ax.figure

        start_color = colors_indices[0]
        end_color = colors_indices[-1]
        marker_edgecolor = plt.rcParams["axes.edgecolor"]

        ax.scatter(
            x[0], y[0],
            color=start_color,
            edgecolor=marker_edgecolor,
            linewidth=0.6,
            label="Start",
            zorder=5,
        )

        ax.scatter(
            x[-1], y[-1],
            color=end_color,
            edgecolor=marker_edgecolor,
            linewidth=0.6,
            label="End",
            zorder=5,
        )

        for i in range(len(x) - 1):
            ax.plot(x[i:i + 2], y[i:i + 2], color=colors_indices[i], lw=1.2)

        if show_colorbar:
            fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                label=label_title,
                ax=ax,
            )

        if created_fig and suptitle is not False:
            fig.suptitle(chart_suptitle)

        ax.set_title(title)
        ax.set_xlabel("$X_1(t)$")
        ax.set_ylabel("$X_2(t)$")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")

        return ax
