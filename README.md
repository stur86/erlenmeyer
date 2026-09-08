# erlenmeyer

A small Python library to build chemical reaction networks and simulate them,
either deterministically (ODE, mass-action kinetics) or stochastically
(Gillespie algorithm). It also can sample a trajectory to emulate the effect of
measuring a small aliquot instead of the full system.

The API is symbolic at the front end: you declare species, combine them with
`+` and `*` to write reactions, then hand the system to a simulator.

```python
from erlenmeyer import Species, Reaction, ReactionSystem, ODESimulator

h = Species("H")
o = Species("O")
h2o = Species("H2O")

system = ReactionSystem([h, o, h2o])
system.add_reaction(Reaction(2 * h + o, h2o, 1.0))   # 2H + O => H2O, rate 1.0

traj = ODESimulator(system).run({"H": 1.0, "O": 1.0}, t_end=10.0, steps=1000)

print(traj.species)         # ['H', 'O', 'H2O']
print(traj.values.shape)    # (1000, 3)
```

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and needs Python 3.11 or
later.

```bash
uv sync                       # library and runtime dependencies
uv sync --group dev           # plus pytest and ruff
uv sync --group examples      # plus marimo and matplotlib
```

Runtime dependencies are NumPy, SciPy and Numba only.

## Concepts

### Species and reaction terms

`Species` is a named chemical species. A name must start with a letter, then
can contain letters, numbers or underscores.

Two operators build the two sides of a reaction:

* `n * species` sets a stochiometric amount. The amount must be an integer > 0.
* `term + term` joins terms into a compound term.

```python
a, b = Species("A"), Species("B")
2 * a + b        # 2A + B
a + a            # 2A  (repeated species are merged)
```

Terms compare and hash by their stochiometry, so `a + a == 2 * a` is `True`.
The species in a term are always kept sorted by name, which makes the
comparison independent of the order you wrote them in.

### Reaction

`Reaction` is a frozen dataclass with four fields:

| Field | Meaning |
| --- | --- |
| `reagents` | the term consumed |
| `products` | the term produced |
| `rate` | the rate constant |
| `bidirectional` | if `True`, the reverse reaction is added too |

```python
Reaction(2 * h + o, h2o, 1.0)                      # 2H + O => H2O [1.0]
Reaction(s + e, es, 1.0, bidirectional=True)       # S + E <=> ES [1.0]
```

**Note:** a bidirectional reaction uses the same rate constant in both
directions. If the forward and the reverse rates differ, add two separate
reactions instead.

### ReactionSystem

`ReactionSystem` holds the list of species and the set of reactions among them.

```python
system = ReactionSystem([s, e, es, p])   # order fixes the column order
system.add_reaction(Reaction(s + e, es, 1.0))
system.add_reaction(Reaction(es, s + e, 0.1))
system.add_reaction(Reaction(es, e + p, 0.5))
```

* The order of the species list is the order of the columns in every array the
  library returns.
* A species given twice raises `ValueError`.
* Reactions are kept in a set, so an identical reaction added twice raises
  `ValueError`, and `system.reactions` has no guaranteed order.
* `system.get_species_index("ES")` gives the column of a species, by object or
  by name.

`system.get_reaction_matrices()` compiles the system into three NumPy arrays,
which is the form the numerical kernels use:

* `reagents_m`, shape `[reactions, species]` — how much of each species each
  reaction consumes,
* `products_m`, same shape — how much it produces,
* `rates_v`, shape `[reactions]` — the rate constants.

A bidirectional reaction becomes two rows, forward and reverse.

### Simulators

Every simulator takes the system in its constructor and runs with:

```python
traj = Simulator(system).run(initial, **kwargs)
```

`initial` is a dictionary that maps species to their starting amount. A key can
be a `Species` or its name, and any species of the system left out starts at
zero, so you only write the ones that are actually present:

```python
ODESimulator(mm).run({"S": 10.0, e: 0.01})   # ES and P start at zero
```

`run` turns the dictionary into a vector in system species order, then calls the
simulator's own `_simulate`. It raises `ValueError` for an unknown species, for
the same species given twice under different keys (`{e: 1.0, "E": 2.0}`), or for
a negative amount.

**`ODESimulator`** integrates mass-action kinetics with
`scipy.integrate.solve_ivp`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `t_start` | `0.0` | start of the time interval |
| `t_end` | `1.0` | end of the time interval |
| `steps` | `100` | number of linearly spaced output points |

Any other keyword argument goes straight to `solve_ivp`, so you can set
`method`, `rtol`, `atol` and so on.

**`GillespieSimulator`** samples one stochastic trajectory with the Gillespie
direct method.

| Argument | Default | Meaning |
| --- | --- | --- |
| `t_end` | `1.0` | simulated time limit |
| `seed` | `None` | seed for the global NumPy RNG, for reproducible runs |

Here the values are molecule counts, not concentrations, so every amount in
`initial` must be a whole number; a fractional one raises `ValueError`. The
time points are the
times of the individual reaction events, so they are not evenly spaced and
their number changes from run to run.

### SimulationTrajectory

Both simulators return the same frozen dataclass:

| Field | Shape | Meaning |
| --- | --- | --- |
| `species` | `[species]` | species names, in system order |
| `times` | `[times]` | the sampled time points |
| `values` | `[times, species]` | concentrations or counts |

`len(traj)` is the number of time points, and `traj.slice(slice(0, None, 10))`
returns a new trajectory with every tenth point. This is useful to thin out a
Gillespie run, which can have many thousands of events.

### Sampling

`sample_trajectory(traj, sample_size, with_replacement=True, rng=None)` emulates
the measurement of a small aliquot of the system at each time point. It returns
an integer array of counts with shape `[times, species]`.

* **With replacement** (the default) each time point is treated as a set of
  proportions and sampled from a multinomial distribution. The values are
  normalized first, so this works on ODE output. A time point that is empty of
  every species gives an all-zero sample.
* **Without replacement** each time point is treated as a population of
  distinguishable items and sampled from a multivariate hypergeometric
  distribution. The values must be integers, and `sample_size` can not exceed
  the total population at any time point.

`rng` accepts a `numpy.random.Generator`, an integer seed, or `None`.

```python
from erlenmeyer import sample_trajectory

thinned = traj.slice(slice(0, None, 10))
counts = sample_trajectory(thinned, 500, rng=42)
```

## How it works

```
Species ──(+, *)──▶ reaction terms ──▶ Reaction ──▶ ReactionSystem
                                                          │
                                       get_reaction_matrices()
                                                          ▼
                                     reagents_m, products_m, rates_v
                                                          │
                              ┌───────────────────────────┴────────────┐
                              ▼                                        ▼
                       ODESimulator                           GillespieSimulator
                     (solve_ivp + Numba)                        (direct method)
                              └───────────────────┬────────────────────┘
                                                  ▼
                                        SimulationTrajectory
                                                  │
                                          sample_trajectory()
                                                  ▼
                                             counts array
```

The symbolic layer exists only to build the three matrices. Everything after
that is numerical, and the inner loops are compiled with Numba.

### The deterministic kernel

Mass-action kinetics gives each reaction *i* the rate

$$ r_i = k_i \prod_j y_j^{R_{ij}} $$

where `R` is `reagents_m` and `y` the concentration vector. The kernel computes
this product as a matrix product in log space, `exp(R @ log(y)) * k`, which
turns one loop per reaction into a single BLAS-style operation.

A concentration of zero has no logarithm, so `safe_log_fast` maps every
non-positive entry to the sentinel `-750.0`. Since `exp(-750)` underflows to
zero in double precision, a reaction whose reagent is absent gets rate zero,
which is the correct answer. The sentinel is needed because a plain `log(0)`
would give `-inf`, and a reaction that does not use the missing species at all
would then compute `0 * -inf = nan`.

Each reaction then moves each species by the difference between what it
produces and what it consumes:

$$ \frac{dy_j}{dt} = \sum_i r_i (P_{ij} - R_{ij}) $$

This right-hand side goes to `solve_ivp` with the requested output times.

### The stochastic kernel

The Gillespie direct method treats the system as an exact, discrete jump
process. The propensity of reaction *i* is

$$ a_i = k_i \prod_j \frac{n_j!}{(n_j - R_{ij})!} $$

that is, the product over its reagents of a falling factorial of the current
population. For a reagent present `n` times with stochiometric coefficient `a`,
the term `n(n-1)...(n-a+1)` counts the distinct ways to choose the colliding
molecules. A reaction whose reagents can not be satisfied gets propensity zero.

Each step then:

1. computes all propensities and their total `a_0`,
2. draws the waiting time to the next event as `-log(u) / a_0`, with `u`
   uniform — an exponential distribution with rate `a_0`,
3. picks which reaction fires by inverse-transform sampling on the cumulative
   propensities, so a reaction is chosen with probability `a_i / a_0`,
4. applies `products_m[i] - reagents_m[i]` to the population.

The loop stops at `t_end`, or earlier if the total propensity reaches zero. In
that case the system can not change any more, so the last state is repeated at
`t_end` and the run ends.

Note that the rate constants are used directly as stochastic rate constants.
The library does not convert deterministic constants into stochastic ones, so
the same rate value does not always give the same dynamics in the two
simulators for reactions of order two or higher.

## Examples

The `examples/` directory holds [marimo](https://marimo.io/) notebooks. Run one
with:

```bash
uv run --group examples marimo edit examples/example_basic.py
```

| Notebook | Subject |
| --- | --- |
| `example_basic.py` | 2H + O → H₂O, the shortest possible use of the API |
| `example_circular.py` | a closed isomer cycle A → B → C → A, with mass conservation |
| `example_michaelis_menten.py` | enzyme kinetics, S + E ⇌ ES → E + P |
| `example_lotka_volterra.py` | predator-prey oscillations |
| `example_brusselator.py` | a limit cycle from an autocatalytic model |
| `example_sir.py` | epidemic spread, with the ODE and Gillespie results compared |
| `example_sampling.py` | finite-aliquot sampling of a trajectory |

## Development

```bash
uv run pytest        # the test suite lives in src/tests
uv run ruff check    # lint
```

The package ships a `py.typed` marker, so type checkers use its annotations.

## License

MIT. See [LICENSE](LICENSE).
