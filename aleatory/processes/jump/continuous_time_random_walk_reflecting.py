import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from aleatory.processes.base import BaseProcess
from aleatory.utils.utils import check_positive_number, check_positive_integer


class ReflectingContinuousTimeRandomWalk(BaseProcess):
    r"""

    Reflecting Continuous-Time Random Walk
    ======================================

    Notes
    -----

    A reflecting continuous-time random walk is a nearest-neighbour jump process on
    an integer state space with one or two reflecting boundaries.

    In this implementation, the process :math:`\{X_t : t \geq 0\}` takes values in a
    subset of :math:`\mathbb{Z}` determined by the specified reflecting boundary
    conditions.

    At interior sites, the process jumps:

    - from :math:`k` to :math:`k+1` with rate :math:`\lambda`
    - from :math:`k` to :math:`k-1` with rate :math:`\mu`

    Thus, away from the boundaries, the holding times are exponentially distributed
    with parameter :math:`\lambda + \mu`, and the jump direction is chosen with
    probabilities

    .. math::

       \mathbb{P}(\Delta X = +1) = \frac{\lambda}{\lambda + \mu},
       \qquad
       \mathbb{P}(\Delta X = -1) = \frac{\mu}{\lambda + \mu}

    At a reflecting boundary, any jump that would leave the allowed state space is
    suppressed. Equivalently, the outward jump rate is set to zero at that boundary.

    So the process remains confined to its admissible state space for all times.
    When :math:`\lambda = \mu`, the walk is symmetric in the interior. When
    :math:`\lambda \neq \mu`, the walk is biased, but reflection still prevents
    escape through the boundary.

    Constructor, Methods, and Attributes
    ------------------------------------

    """
    def __init__(
            self,
            rate_up=0.5,
            rate_down=0.5,
            initial=0,
            lower=None,
            upper=None,
            rng=None
    ):
        r"""
        :param float rate_up: right-jump rate :math:`\lambda \geq 0`
        :param float rate_down: left-jump rate :math:`\mu \geq 0`
        :param int initial: initial state :math:`x_0 \in \mathbb{Z}`
        :param int or None lower: lower reflecting boundary
        :param int or None upper: upper reflecting boundary
        :param numpy.random.Generator rng: optional custom random number generator
        """
        super().__init__(rng=rng)
        self.rate_up = rate_up
        self.rate_down = rate_down
        self.initial = initial
        self.lower = lower
        self.upper = upper
        self._validate_boundaries()

        self.name = (
            rf"Continuous--Time Random Walk "
            rf"$(\lambda={self.rate_up},\ \mu={self.rate_down},\ x_0={self.initial})$"
            "\n"
            f"{self._boundary_description()}"
        )
        self.T = None
        self.N = None
        self.paths = None

    def __str__(self):
        return (
            f"Reflecting continuous-time random walk with "
            f"rate_up={self.rate_up}, rate_down={self.rate_down}, "
            f"initial={self.initial}, lower={self.lower}, upper={self.upper}"
        )

    def __repr__(self):
        return (
            f"ReflectingContinuousTimeRandomWalk("
            f"rate_up={self.rate_up}, rate_down={self.rate_down}, "
            f"initial={self.initial}, lower={self.lower}, upper={self.upper})"
        )

    def _validate_boundaries(self):
        if self.lower is None and self.upper is None:
            raise ValueError("At least one of lower or upper must be specified")

        if self.lower is not None and self.upper is not None:
            if self.lower >= self.upper:
                raise ValueError("Require lower < upper")

        if self.lower is not None and self.initial < self.lower:
            raise ValueError("initial must satisfy initial >= lower")

        if self.upper is not None and self.initial > self.upper:
            raise ValueError("initial must satisfy initial <= upper")

    def _boundary_description(self):
        if self.lower is not None and self.upper is not None:
            return rf"Reflecting boundaries: $(k_{{\mathrm{{lower}}}}^*={self.lower},\ k_{{\mathrm{{upper}}}}^*={self.upper})$"
        if self.lower is not None:
            return rf"Reflecting lower boundary: $(k_{{\mathrm{{lower}}}}^*={self.lower})$"
        return rf"Reflecting upper boundary: $(k_{{\mathrm{{upper}}}}^*={self.upper})$"

    def _in_state_space(self, x):
        if self.lower is not None and x < self.lower:
            return False
        if self.upper is not None and x > self.upper:
            return False
        return True

    @property
    def rate_up(self):
        return self._rate_up

    @rate_up.setter
    def rate_up(self, value):
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError("rate_up must be numeric")
        if value < 0:
            raise ValueError("rate_up must be nonnegative")
        self._rate_up = float(value)

    @property
    def rate_down(self):
        return self._rate_down

    @rate_down.setter
    def rate_down(self, value):
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError("rate_down must be numeric")
        if value < 0:
            raise ValueError("rate_down must be nonnegative")
        self._rate_down = float(value)

    def _check_rates(self):
        if self.rate_up == 0 and self.rate_down == 0:
            raise ValueError("At least one of rate_up or rate_down must be positive")

    def _local_rates(self, x):
        ru = self.rate_up
        rd = self.rate_down

        if self.lower is not None and x == self.lower:
            rd = 0.0

        if self.upper is not None and x == self.upper:
            ru = 0.0

        return ru, rd

    @property
    def initial(self):
        return self._initial

    @initial.setter
    def initial(self, value):
        if not isinstance(value, (int, np.integer)):
            raise TypeError("initial must be an integer")
        self._initial = int(value)

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, value):
        if value is not None and not isinstance(value, (int, np.integer)):
            raise TypeError("lower must be an integer or None")
        self._lower = None if value is None else int(value)

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, value):
        if value is not None and not isinstance(value, (int, np.integer)):
            raise TypeError("upper must be an integer or None")
        self._upper = None if value is None else int(value)

    def _sample_continuous_time_random_walk(self, T=None):
        if T is None:
            raise ValueError("T must be provided")
        check_positive_number(T, "Time")
        self._check_rates()

        t = 0.0
        x = self.initial

        times = [0.0]
        states = [x]

        while t < T:
            ru, rd = self._local_rates(x)
            total = ru + rd

            if total <= 0:
                times.append(T)
                states.append(x)
                break

            dt = self.rng.exponential(scale=1.0 / total)
            t_next = t + dt

            if t_next >= T:
                times.append(T)
                states.append(x)
                break

            u = self.rng.random()
            if u < ru / total:
                x += 1
            else:
                x -= 1

            t = t_next
            times.append(t)
            states.append(x)

        return np.asarray(times), np.asarray(states)

    def sample(self, T=None):
        return self._sample_continuous_time_random_walk(T=T)

    def simulate(self, N, T=None):
        check_positive_integer(N, "N")
        self.N = N
        self.T = T
        self.paths = [self.sample(T=T) for _ in range(N)]
        return self.paths

    def plot(
            self,
            N,
            T=None,
            style="seaborn-v0_8-whitegrid",
            mode="steps",
            title=None,
            suptitle=None,
            **fig_kw,
    ):
        chart_suptitle = suptitle if suptitle is not None else self.name
        self.simulate(N=N, T=T)
        paths = self.paths

        with plt.style.context(style):
            fig, ax = plt.subplots(**fig_kw)

            for times, states in paths:
                if mode == "points":
                    ax.scatter(times, states, s=10)
                elif mode == "steps":
                    ax.step(times, states, where="post", linewidth=1.25)
                elif mode == "linear":
                    ax.plot(times, states)
                elif mode == "points+steps":
                    ax.step(times, states, where="post", alpha=0.5)
                    color = plt.gca().lines[-1].get_color()
                    ax.plot(times, states, "o", color=color, markersize=4)
                else:
                    raise ValueError(
                        "mode can only take values 'points', 'steps', 'linear' or 'points+steps'"
                    )

            fig.suptitle(chart_suptitle)
            ax.set_title(title)
            ax.set_xlabel("$t$")
            ax.set_ylabel("$X(t)$")

            if T is not None:
                ax.set_xlim(right=T)

            if self.lower is not None:
                ax.set_ylim(bottom=self.lower - 0.5)
                ax.axhline(
                    self.lower,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=3,
                    label=r"$k_{\mathrm{lower}}^*$"
                )
            if self.upper is not None:
                ax.set_ylim(top=self.upper + 0.5)
                ax.axhline(
                    self.upper,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=3,
                    label=r"$k_{\mathrm{upper}}^*$"
                )

            plt.show()

        return fig

    @staticmethod
    def _evaluate_path_on_grid(times, states, grid):
        indices = np.searchsorted(times, grid, side="right") - 1
        indices = np.clip(indices, 0, len(states) - 1)
        return states[indices]

    def draw(
            self,
            N,
            T=10.0,
            style="seaborn-v0_8-whitegrid",
            colormap="viridis",
            envelope=False,
            marginal=True,
            mode="steps",
            colorspos=None,
            title=None,
            suptitle=None,
            **fig_kw,
    ):
        check_positive_integer(N, "N")
        check_positive_number(T, "Time")

        chart_suptitle = suptitle if suptitle is not None else self.name
        self.simulate(N=N, T=T)
        paths = self.paths

        times_grid = np.linspace(0.0, T, 400)
        path_values = np.array(
            [self._evaluate_path_on_grid(times, states, times_grid) for times, states in paths]
        )

        empirical_mean = path_values.mean(axis=0)
        lower_band = np.quantile(path_values, 0.05, axis=0)
        upper_band = np.quantile(path_values, 0.95, axis=0)

        terminal_values = np.array([states[-1] for _, states in paths], dtype=int)
        terminal_mean = terminal_values.mean()

        cm = plt.colormaps[colormap]
        n_bins = max(10, int(np.sqrt(N)))
        col = np.linspace(0, 1, n_bins, endpoint=True)

        with plt.style.context(style):
            if marginal:
                fig = plt.figure(**fig_kw)
                gs = GridSpec(1, 5)
                ax1 = fig.add_subplot(gs[:4])
                ax2 = fig.add_subplot(gs[4:], sharey=ax1)

                n, bins, patches = ax2.hist(
                    terminal_values,
                    bins=n_bins,
                    orientation="horizontal",
                    density=True,
                )

                for c, p in zip(col[: len(patches)], patches):
                    plt.setp(p, "facecolor", cm(c))

                my_bins = pd.cut(
                    terminal_values,
                    bins=bins,
                    labels=range(len(bins) - 1),
                    include_lowest=True,
                )
                colors = [col[int(b)] for b in my_bins]

                ax2.axhline(
                    y=terminal_mean,
                    linestyle="--",
                    lw=1.75,
                    color="maroon",
                    label="$E[X_T]$ (empirical)",
                )
                ax2.legend()
                plt.setp(ax2.get_yticklabels(), visible=False)
                ax2.set_title("$X_T$")

                for i, (times, states) in enumerate(paths):
                    color = cm(colors[i])

                    if mode == "points":
                        ax1.scatter(times, states, color=color, s=10)
                    elif mode == "steps":
                        ax1.step(times, states, color=color, where="post", linewidth=1.25)
                    elif mode == "points+steps":
                        ax1.step(times, states, color=color, where="post", linewidth=1.25)
                        ax1.plot(times, states, "o", color=color, markersize=4)
                    else:
                        raise ValueError(
                            "mode can only take values 'points', 'steps' or 'points+steps'"
                        )

                ax1.plot(times_grid, empirical_mean, "--", lw=1.75, label="$E[X_t]$ (empirical)")
                ax1.legend()

                if envelope:
                    ax1.fill_between(
                        times_grid,
                        upper_band,
                        lower_band,
                        alpha=0.20,
                        color="silver",
                    )

                plt.subplots_adjust(wspace=0.2, hspace=0.5)

            else:
                fig, ax1 = plt.subplots(**fig_kw)

                if colorspos is not None:
                    colors = []
                    for _, states in paths:
                        idx = min(colorspos, len(states) - 1)
                        denom = max(1.0, np.max(np.abs(states)))
                        colors.append((states[idx] / denom + 1.0) / 2.0)
                else:
                    _, bins = np.histogram(terminal_values, bins=n_bins)
                    my_bins = pd.cut(
                        terminal_values,
                        bins=bins,
                        labels=range(len(bins) - 1),
                        include_lowest=True,
                    )
                    colors = [col[int(b)] for b in my_bins]

                for i, (times, states) in enumerate(paths):
                    ax1.step(
                        times,
                        states,
                        color=cm(colors[i]),
                        lw=0.9,
                        where="post",
                    )

                ax1.plot(times_grid, empirical_mean, "--", lw=1.75, label="$E[X_t]$ (empirical)")
                ax1.legend()

                if envelope:
                    ax1.fill_between(
                        times_grid,
                        upper_band,
                        lower_band,
                        color="lightgray",
                        alpha=0.25,
                    )

            fig.suptitle(chart_suptitle)
            ax1.set_xlim(right=T)

            if self.lower is not None:
                ax1.set_ylim(bottom=self.lower - 0.5)
                ax1.axhline(
                    self.lower,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8
                )
            if self.upper is not None:
                ax1.set_ylim(top=self.upper + 0.5)
                ax1.axhline(
                    self.upper,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8
                )

            if title is None:
                ax1.set_title(r"Simulated Paths $X_t, t \leq T$")
            else:
                ax1.set_title(title)

            ax1.set_xlabel("$t$")
            ax1.set_ylabel("$X(t)$")
            plt.show()

        return fig
