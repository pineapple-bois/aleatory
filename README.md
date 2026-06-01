# *aleatory*

[![PyPI version fury.io](https://badge.fury.io/py/aleatory.svg)](https://pypi.org/project/aleatory/) [![Downloads](https://static.pepy.tech/personalized-badge/aleatory?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/aleatory)
![example workflow](https://github.com/quantgirluk/aleatory/actions/workflows/python-package.yml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/aleatory/badge/?version=latest)](https://aleatory.readthedocs.io/en/latest/?badge=latest)

- [Git Homepage](https://github.com/quantgirluk/aleatory)
- [Pip Repository](https://pypi.org/project/aleatory/)
- [Documentation](https://aleatory.readthedocs.io/en/latest/)
- 🆕 [Gallery of Stochastic Processes](https://aleatory.readthedocs.io/en/latest/auto_examples/index.html) 🖼️

-----

# Implemented in this fork

> Fork note: Upstream aleatory v1.2.1 merged; local ABP/RTP/CTRW additions retained.

## 1. `processes.jump`

### A. `ContinuousTimeRandomWalk`

Implements a nearest-neighbour continuous-time jump process on the full integer lattice $\mathbb{Z}$. The process jumps upward with rate $\lambda$ and downward with rate $\mu$, so it supports both symmetric and biased walks. For constant left and right jump rates, the marginal law at fixed time is given by the Skellam distribution.

**Signature**

```python
from aleatory.processes import ContinuousTimeRandomWalk

ContinuousTimeRandomWalk(
    rate_up=0.5,
    rate_down=0.5,
    initial=0,
    rng=None,
)
```

**Arguments**

- `rate_up`: right-jump rate $\lambda \geq 0$
- `rate_down`: left-jump rate $\mu \geq 0$
- `initial`: initial state $x_0 \in \mathbb{Z}$
- `rng`: optional NumPy random number generator

At least one of `rate_up` or `rate_down` must be positive.


### B. `ReflectingContinuousTimeRandomWalk`

Implements a nearest-neighbour continuous-time jump process on an integer state space with one-sided or two-sided reflecting boundaries. In the interior, the process jumps upward with rate $\lambda$ and downward with rate $\mu$. At a reflecting boundary, any jump that would leave the admissible state space is suppressed by setting the outward jump rate to zero.

**Signature**

```python
from aleatory.processes import ReflectingContinuousTimeRandomWalk

ReflectingContinuousTimeRandomWalk(
    rate_up=0.5,
    rate_down=0.5,
    initial=0,
    lower=None,
    upper=None,
    rng=None,
)
```

**Arguments**

- `rate_up`: right-jump rate $\lambda \geq 0$
- `rate_down`: left-jump rate $\mu \geq 0$
- `initial`: initial state $x_0 \in \mathbb{Z}$
- `lower`: optional lower reflecting boundary
- `upper`: optional upper reflecting boundary
- `rng`: optional NumPy random number generator

At least one of `lower` or `upper` must be specified. If both are specified, the class requires `lower < upper`, and the initial state must lie inside the admissible state space.


### C. `AbsorbingContinuousTimeRandomWalk`

Implements a nearest-neighbour continuous-time jump process on an integer state space with one-sided or two-sided absorbing boundaries. In the interior, the process jumps upward with rate $\lambda$ and downward with rate $\mu$. At an absorbing boundary, the process becomes trapped, so once a boundary is hit the state remains fixed for all subsequent times.

**Signature**

```python
from aleatory.processes import AbsorbingContinuousTimeRandomWalk

AbsorbingContinuousTimeRandomWalk(
    rate_up=0.5,
    rate_down=0.5,
    initial=0,
    lower=None,
    upper=None,
    rng=None,
)
```

**Arguments**

- `rate_up`: right-jump rate $\lambda \geq 0$
- `rate_down`: left-jump rate $\mu \geq 0$
- `initial`: initial state $x_0 \in \mathbb{Z}$
- `lower`: optional lower absorbing boundary
- `upper`: optional upper absorbing boundary
- `rng`: optional NumPy random number generator

At least one of `lower` or `upper` must be specified. If both are specified, the class requires `lower < upper`, and the initial state must lie inside the admissible state space.


## 2. `processes.multi_dimensional`

### A. `ABP2D`

Implements a two-dimensional active Brownian particle with self-propulsion, rotational diffusion, and optional translational diffusion. The particle moves at constant speed `v0` in the direction of its current heading `theta_t`, while the heading itself evolves diffusively. This produces persistent motion over short times and progressive loss of directional memory over longer times.

The process is simulated from the Langevin system

```math
\dot{X}_t = v_0 \cos(\theta_t) + \sqrt{2D_{\mathrm{T}}}\,\eta_x(t)
```

```math
\dot{Y}_t = v_0 \sin(\theta_t) + \sqrt{2D_{\mathrm{T}}}\,\eta_y(t)
```

```math
\dot{\theta}_t = \sqrt{2D_{\mathrm{R}}}\,\eta_\theta(t)
```

where `D_T` is the translational diffusion coefficient and `D_R` is the rotational diffusion coefficient.

**Signature**

```python
from aleatory.processes import ABP2D

ABP2D(
    speed=1.0,
    rotational_diffusion=1.0,
    translational_diffusion=0.0,
    T=1.0,
    x0=0.0,
    y0=0.0,
    theta0=0.0,
    rng=None,
)
```

**Arguments**

- `speed`: self-propulsion speed $v_0 \geq 0$
- `rotational_diffusion`: rotational diffusion coefficient $D_R \geq 0$
- `translational_diffusion`: translational diffusion coefficient $D_T \geq 0$
- `T`: end time of the simulation interval
- `x0`: initial x-coordinate
- `y0`: initial y-coordinate
- `theta0`: initial orientation angle
- `rng`: optional NumPy random number generator

The implementation uses an Euler discretisation on a uniform time grid over `[0, T]`. The planar sample returned by `sample(n)` consists of the coordinate pair `(x, y)`. The most recently simulated orientation path is stored on the instance and can be accessed through `last_theta`, or returned explicitly by `sample_with_orientation(n)`.


### B. `RTP2D`

Implements a two-dimensional run-and-tumble particle with self-propulsion, Poisson-distributed tumble events, and optional translational diffusion. Between tumbles, the particle moves ballistically at constant speed in a fixed direction. At each tumble, the orientation is reset instantaneously by drawing a new heading uniformly from `[0, 2π)`.

The process therefore alternates between straight runs and sudden reorientation events. In the discrete-time implementation, a tumble occurs during a timestep `dt` with probability `lambda * dt`, where `lambda` is the tumble rate.

**Signature**

```python
from aleatory.processes import RTP2D

RTP2D(
    speed=1.0,
    tumble_rate=1.0,
    translational_diffusion=0.0,
    T=1.0,
    x0=0.0,
    y0=0.0,
    theta0=0.0,
    rng=None,
)
```

**Arguments**

- `speed`: self-propulsion speed $v_0 \geq 0$
- `tumble_rate`: tumble rate $\lambda \geq 0$
- `translational_diffusion`: translational diffusion coefficient $D_T \geq 0$
- `T`: end time of the simulation interval
- `x0`: initial x-coordinate
- `y0`: initial y-coordinate
- `theta0`: initial orientation angle
- `rng`: optional NumPy random number generator

The planar sample returned by `sample(n)` consists of the coordinate pair `(x, y)`. The most recently simulated orientation path is stored on the instance and can be accessed through `last_theta`, or returned explicitly by `sample_with_orientation(n)`. The indices of the most recent tumble events are also stored on the instance and are available through `last_tumble_indices`.

----

## Backlog

Refactor all processes that hardcode
- `style="seaborn-v0_8-whitegrid"` 
- `with plt.style.context(style):`
- Explore `utils.plotters` for above

Bin `processes.jump` into 'discrete' & 'continuous'

Tidy up all `__init__.py` (every module)

----

## New Tests

### 1. `tests/test_reflecting_cont_time_random_walk.py`

Testing: `ReflectingContinuousTimeRandomWalk`

- constructor validation
- path structure invariants
- state-space invariants
- reflection-rule checks
- trapping edge cases
- comparison with the unbounded walk in a wide interval
- basic long-time bounded behaviour checks

Run it from the repo root with:

```bash
python -m unittest discover -s tests -p "test_reflecting_cont_time_random_walk.py"
```


### 2. `tests/test_absorbing_cont_time_random_walk.py`

Testing: `AbsorbingContinuousTimeRandomWalk`

- constructor validation
- path structure invariants
- state-space invariants
- absorption-rule checks
- trapping behaviour at absorbing boundaries
- comparison with the unbounded walk in a wide interval
- basic long-time absorbing behaviour checks

Run it from the repo root with:

```bash
python -m unittest discover -s tests -p "test_absorbing_cont_time_random_walk.py"
```

----

## Overview

The ***aleatory*** (/ˈeɪliətəri/) Python library provides functionality for simulating and visualising
stochastic processes. More precisely, it introduces objects representing a number of
stochastic processes and provides methods to:

- generate realizations/trajectories from each process —over discrete time sets
- create visualisations to illustrate the processes properties and behaviour

<p align="center">
<img src="https://raw.githubusercontent.com/quantgirluk/aleatory/main/docs/source/_static/vasicek_process_drawn.png"   style="display:block;float:none;margin-left:auto;margin-right:auto;width:80%">
</p>

Currently, aleatory supports the following stochastic processes in one dimension:

- Arithmetic Brownian Motion (see [Brownian Motion](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.BrownianMotion.html#aleatory.processes.BrownianMotion))
- [Bessel process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.BESProcess.html#aleatory.processes.BESProcess)
- [Brownian Bridge](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.BrownianBridge.html#aleatory.processes.BrownianBridge)
- [Brownian Excursion](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.BrownianExcursion.html#aleatory.processes.BrownianExcursion)
- [Brownian Meander](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.BrownianMeander.html#aleatory.processes.BrownianMeander)
- [Brownian Motion](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.BrownianMotion.html#aleatory.processes.BrownianMotion)
- [Constant Elasticity Variance (CEV) process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.CEVProcess.html#aleatory.processes.CEVProcess)
- [Cox–Ingersoll–Ross (CIR) process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.CIRProcess.html#aleatory.processes.CIRProcess)
- [Chan-Karolyi-Longstaff-Sanders (CKLS) process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.CKLSProcess.html#aleatory.processes.CKLSProcess)
- [Fractional Brownian Motion process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.fBM.html#aleatory.processes.fBM)
- [Galton-Watson process with Poisson branching](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.GaltonWatson.html#aleatory.processes.GaltonWatson)
- [Gamma process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.GammaProcess.html#aleatory.processes.GammaProcess)
- [Gaussian Process with  Constant Kernel](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.GPConstant.html#aleatory.processes.GPConstant)
- [Gaussian Process with Linear Kernel](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.GPLinear.html#aleatory.processes.GPLinear)
- [Gaussian Process with Matern Kernel](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.GPMatern.html#aleatory.processes.GPMatern)
- [Gaussian Process with Periodic Kernel](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.GPPeriodic.html#aleatory.processes.GPPeriodic)
- [Gaussian Process with RBF Kernel](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.GPRBF.html#aleatory.processes.GPRBF)
- [Gaussian Process with Squared Exponential Kernel](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.GPSquaredExponential.html#aleatory.processes.GPSquaredExponential)
- [General Random Walk](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.GeneralRandomWalk.html#aleatory.processes.GeneralRandomWalk)
- [Geometric Brownian Motion](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.GBM.html#aleatory.processes.GBM)
- [Hawkes process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.HawkesProcess.html#aleatory.processes.HawkesProcess)
- [Inverse Gaussian process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.InverseGaussian.html#aleatory.processes.InverseGaussian)
- [Inhomogeneous Poisson process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.InhomogeneousPoissonProcess.html#aleatory.processes.InhomogeneousPoissonProcess)
- [Mixed Poisson process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.MixedPoissonProcess.html#aleatory.processes.MixedPoissonProcess)
- [Ornstein–Uhlenbeck (OU) process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.OUProcess.html#aleatory.processes.OUProcess)
- [Poisson process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.PoissonProcess.html#aleatory.processes.PoissonProcess)
- [Random Walk](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.RandomWalk.html#aleatory.processes.RandomWalk)
- [Squared Bessel processes](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.BESQProcess.html#aleatory.processes.BESQProcess)
- [Vasicek process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.Vasicek.html#aleatory.processes.Vasicek)
- [Variance-Gamma process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.VarianceGammaProcess.html#aleatory.processes.VarianceGammaProcess)
- [White Noise](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.WhiteNoise.html#aleatory.processes.WhiteNoise)

From v1.1.1 aleatory supports the following 2-d stochastic processes:

- [Brownian Motion 2D](https://aleatory.readthedocs.io/en/latest/auto_examples/plot_brownian_2d.html#sphx-glr-auto-examples-plot-brownian-2d-py)
- [Correlated Brownian Motions](https://aleatory.readthedocs.io/en/latest/auto_examples/plot_correlated_bms.html)
- [Random Walk 2D](https://aleatory.readthedocs.io/en/latest/auto_examples/plot_random_walk_2d.html#sphx-glr-auto-examples-plot-random-walk-2d-py)

This fork additionally includes active-particle and continuous-time random-walk modules:

- `ContinuousTimeRandomWalk`: nearest-neighbour continuous-time random walk on $\mathbb{Z}$ with rates $\lambda$ and $\mu$.
- `ReflectingContinuousTimeRandomWalk`: nearest-neighbour continuous-time random walk with one-sided or two-sided reflecting boundaries.
- `AbsorbingContinuousTimeRandomWalk`: nearest-neighbour continuous-time random walk with one-sided or two-sided absorbing boundaries.
- `ABP2D`: two-dimensional active Brownian particle with self-propulsion, rotational diffusion, and optional translational diffusion.
- `RTP2D`: two-dimensional run-and-tumble particle with Poisson tumbles and optional translational diffusion.

The local CTRW boundary classes are covered by `tests/test_reflecting_cont_time_random_walk.py` and `tests/test_absorbing_cont_time_random_walk.py`.

## Installation

Aleatory is available on [pypi](https://pypi.python.org/pypi) and can be
installed as follows

```bash
pip install aleatory
```

## Dependencies

Aleatory relies heavily on

- ``numpy``  for random number generation
- ``scipy`` and ``statsmodels`` for support for a number of one-dimensional distributions.
- ``matplotlib`` for creating visualisations

## Compatibility

Aleatory is tested on Python versions 3.8, 3.9, 3.10, and 3.11

## Quick-Start

Aleatory allows you to create fancy visualisations from different stochastic processes in an easy and concise way.

For example, the following code

```python
from aleatory.processes import BrownianMotion

brownian = BrownianMotion()
brownian.draw(n=100, N=100, colormap="cool", figsize=(12,9))

```

generates a chart like this:

<p align="center">
<img src="https://raw.githubusercontent.com/quantgirluk/aleatory/main/docs/source/_static/brownian_motion_quickstart_08.png"   style="display:block;float:none;margin-left:auto;margin-right:auto;width:80%">
</p>

For more examples visit the [Quick-Start Guide](https://aleatory.readthedocs.io/en/latest/general.html).

**If you like this project, please give it a star!** ⭐️

## Thanks for Visiting! ✨

Connect with me via:

- 🦜 [Twitter](https://twitter.com/Quant_Girl)
- 👩🏽‍💼 [Linkedin](https://www.linkedin.com/in/dialidsantiago/)
- 📸 [Instagram](https://www.instagram.com/quant_girl/)
- 👾 [Personal Website](https://quantgirl.blog)
