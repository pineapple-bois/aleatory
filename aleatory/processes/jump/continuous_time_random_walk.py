import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import skellam

from aleatory.processes.base import BaseProcess
from aleatory.utils.utils import check_positive_number, check_positive_integer


class ContinuousTimeRandomWalk(BaseProcess):
    r"""

    Continuous-Time Random Walk
    ===========================

    Notes
    -----

    A continuous-time random walk on :math:`\mathbb{Z}` is a jump process which moves
    between neighbouring lattice sites at random times.

    In this implementation, the process :math:`\{X_t : t \geq 0\}` jumps:

    - from :math:`k` to :math:`k+1` with rate :math:`\lambda`
    - from :math:`k` to :math:`k-1` with rate :math:`\mu`

    Thus the holding times are exponentially distributed with parameter
    :math:`\lambda + \mu`, and at each jump the direction is chosen with probabilities

    .. math::

       \mathbb{P}(\Delta X = +1) = \frac{\lambda}{\lambda + \mu},
       \qquad
       \mathbb{P}(\Delta X = -1) = \frac{\mu}{\lambda + \mu}

    When :math:`\lambda = \mu`, the walk is symmetric. When
    :math:`\lambda \neq \mu`, the walk is biased and exhibits drift.

    For fixed :math:`t`, the marginal law of :math:`X_t` is the Skellam distribution,
    since the process may be represented as the difference of two independent Poisson
    processes.

    Constructor, Methods, and Attributes
    ------------------------------------

    """

    def __init__(self, rate_up=0.5, rate_down=0.5, initial=0, rng=None):
        r"""
        :param float rate_up: right-jump rate :math:`\lambda \geq 0`
        :param float rate_down: left-jump rate :math:`\mu \geq 0`
        :param int initial: initial state :math:`x_0 \in \mathbb{Z}`
        :param numpy.random.Generator rng: optional custom random number generator
        """
        super().__init__(rng=rng)
        self.rate_up = rate_up
        self.rate_down = rate_down
        self.initial = initial
        self.name = (
            rf"Continuous--Time Random Walk "
            rf"$(\lambda={self.rate_up},\ \mu={self.rate_down},\ x_0={self.initial})$"
        )
        self.T = None
        self.N = None
        self.paths = None

    def __str__(self):
        return (
            f"Continuous--time random walk with "
            f"rate_up={self.rate_up}, rate_down={self.rate_down}, initial={self.initial}"
        )

    def __repr__(self):
        return (
            f"ContinuousTimeRandomWalk("
            f"rate_up={self.rate_up}, rate_down={self.rate_down}, initial={self.initial})"
        )

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

    @property
    def total_rate(self):
        return self.rate_up + self.rate_down

    def _check_rates(self):
        if self.total_rate <= 0:
            raise ValueError("At least one of rate_up or rate_down must be positive")

    @property
    def initial(self):
        return self._initial

    @initial.setter
    def initial(self, value):
        if not isinstance(value, (int, np.integer)):
            raise TypeError("initial must be an integer")
        self._initial = int(value)


    def _sample_continuous_time_random_walk(self, T=None):
        if T is None:
            raise ValueError("T must be provided")
        check_positive_number(T, "Time")

        t = 0.0
        x = self.initial

        times = [0.0]
        states = [x]

        while t < T:
            dt = self.rng.exponential(scale=1.0 / self.total_rate)
            t_next = t + dt

            if t_next >= T:
                times.append(T)
                states.append(x)
                break

            u = self.rng.random()
            if u < self.rate_up / self.total_rate:
                x += 1
            else:
                x -= 1

            t = t_next
            times.append(t)
            states.append(x)

        return np.asarray(times), np.asarray(states)

    def sample(self, T=None):
        return self._sample_continuous_time_random_walk(T=T)

    def get_marginal(self, t):
        if not isinstance(t, (int, float, np.integer, np.floating)):
            raise TypeError("Time must be numeric")
        if t < 0:
            raise ValueError("Time must be nonnegative")
        # At fixed time t, X_t is integer-valued on Z, so its marginal law is discrete.
        # For constant jump rates, X_t is the difference of two independent Poisson variables,
        # hence Skellam-distributed.
        return skellam(self.rate_up * t, self.rate_down * t, loc=self.initial)

    def marginal_expectation(self, times):
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError("Times must be nonnegative")
        return self.initial + (self.rate_up - self.rate_down) * times

    def marginal_variance(self, times):
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError("Times must be nonnegative")
        return (self.rate_up + self.rate_down) * times

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
            plt.show()

        return fig

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
        expectations = self.marginal_expectation(times_grid)
        variances = self.marginal_variance(times_grid)
        stds = np.sqrt(variances)
        lower = expectations - 2.0 * stds
        upper = expectations + 2.0 * stds

        terminal_values = np.array([states[-1] for _, states in paths], dtype=int)
        marginalT = self.get_marginal(T)

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

                ks = np.arange(
                    int(marginalT.ppf(0.001)),
                    int(marginalT.ppf(0.999)) + 1,
                )
                ax2.plot(
                    marginalT.pmf(ks),
                    ks,
                    "o",
                    linestyle="",
                    color="maroon",
                    markersize=2,
                    label="$X_T$ pmf",
                )
                ax2.axhline(
                    y=marginalT.mean(),
                    linestyle="--",
                    lw=1.75,
                    label="$E[X_T]$",
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

                ax1.plot(times_grid, expectations, "--", lw=1.75, label="$E[X_t]$")
                ax1.legend()

                if envelope:
                    ax1.fill_between(times_grid, upper, lower, alpha=0.20, color="silver")

                plt.subplots_adjust(wspace=0.2, hspace=0.5)

            else:
                fig, ax1 = plt.subplots(**fig_kw)

                if colorspos is not None:
                    colors = []
                    for times, states in paths:
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

                ax1.plot(times_grid, expectations, "--", lw=1.75, label="$E[X_t]$")
                ax1.legend()

                if envelope:
                    ax1.fill_between(times_grid, upper, lower, color="lightgray", alpha=0.25)

            fig.suptitle(chart_suptitle)
            ax1.set_xlim(right=T)

            if title is None:
                ax1.set_title(r"Simulated Paths $X_t, t \leq T$")
            else:
                ax1.set_title(title)

            ax1.set_xlabel("$t$")
            ax1.set_ylabel("$X(t)$")
            plt.show()

        return fig
