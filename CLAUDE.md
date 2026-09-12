# CLAUDE.md

Notes for Claude Code working in this repository.

## What this package is

`erlenmeyer` builds chemical reaction networks and simulates them, either
deterministically (mass-action ODEs) or stochastically (Gillespie). A separate
step can sample a trajectory to emulate measuring a small aliquot instead of
the whole system.

The front end is symbolic (`Species`, `+`, `*`); everything past
`get_reaction_matrices()` is plain NumPy, with the inner loops compiled by
Numba. Runtime dependencies are NumPy, SciPy and Numba only. Keep it that way:
do not add a dependency without asking.

## Layout

```
src/erlenmeyer/
  symbols.py     Species and reaction terms; the +/* algebra
  reaction.py    Reaction, ReactionSystem, ReactionMatrices
  simulator.py   SimulationTrajectory, SimulationType, AbstractSimulator
  ode.py         ODESimulator and its Numba kernels
  gillespie.py   GillespieSimulator and its Numba kernels
  sampling.py    sample_trajectory
src/tests/       one test_<module>.py per module, pytest
examples/        marimo notebooks, one model each
```

The data flow is one way:

```
Species -> terms -> Reaction -> ReactionSystem
        -> ReactionMatrices (reagents_m, products_m, rates_v)
        -> AbstractSimulator._simulate -> SimulationTrajectory
        -> sample_trajectory -> counts
```

`ReactionMatrices` is the boundary. Everything above it is symbolic and
validated; everything below it is numeric arrays and must stay free of Python
objects, because it runs inside `numba.njit`.

## Invariants worth knowing before you edit

* **Species order fixes column order.** The list given to `ReactionSystem`
  orders the columns of every matrix, every `values` array, and the vector
  built from a `run()` dictionary.
* **Reactions live in a `set`.** `system.reactions` therefore has no
  guaranteed order, and matrix rows follow that arbitrary order. Tests that
  care must look up rows by content, not by index.
* **`run()` takes a dictionary**, keyed by `Species` or by name; anything left
  out starts at zero. `AbstractSimulator._initial_vector` turns it into the
  ordered vector that `_simulate` receives.
* **A bidirectional reaction shares one rate constant** and becomes two rows.
  Different forward and reverse rates need two separate reactions.
* **Products of `None` mean a decay**: the reagents leave and nothing takes
  their place, giving an all-zero row of `products_m`. A decay can not be
  bidirectional. Prefer it over an inert waste species, which would add a
  column to every result and grow without bound in a Gillespie run.
* **Term equality is lenient, term arithmetic is not.**
  `AbstractReactionTerm.__eq__` returns `NotImplemented` for anything that is
  not a term, so a term compares unequal to an unrelated object instead of
  raising. It has to: the products of a decay are `None`, and two `Reaction`
  dataclasses can only be compared if `None` may meet a term. `__add__` still
  raises `TypeError`, as adding a term to a number means nothing.
* **`SimulationType` on a trajectory is load-bearing**, not a label. Sampling
  reads it to choose the interpolation (`previous` for Gillespie, `linear` for
  ODE), and a Gillespie trajectory with non-integer values is rejected on
  construction.
* **Rate constants go into the Gillespie kernel as they are.** No
  deterministic-to-stochastic conversion happens, so the same constant does not
  give the same dynamics in the two simulators above first order. Do not
  "fix" this silently.
* **The ODE kernel works in log space.** `safe_log_fast` maps non-positive
  concentrations to `-750.0` so that `exp` underflows to zero; a plain `log(0)`
  would give `-inf` and then `0 * -inf = nan`. Leave the sentinel alone.

## Commands

```bash
uv sync --group dev --group examples   # full environment
uv run pytest                          # the suite lives in src/tests
uv run ruff check src/                 # must stay clean
uv run marimo edit examples/example_basic.py
```

`ruff check .` also reports about eighteen findings in `examples/`; they come
from marimo's cell format (a bare `return` closing every cell) and are
expected. `src/` must be clean.

To check every notebook still runs without opening a browser:

```python
import asyncio, importlib.util, pathlib
import matplotlib; matplotlib.use("Agg")

async def main():
    for f in sorted(pathlib.Path("examples").glob("example_*.py")):
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        await mod.app.embed()
        print("OK", f.name)

asyncio.run(main())
```

## Conventions

* **Docstrings are NumPy style** (`Parameters`, `Returns`, `Raises`) on every
  public class and method. Attributes of a dataclass are documented in the
  class docstring under `Attributes`.
* **Prose is plain, simple English.** Short sentences, no marketing. Technical
  terms of chemistry, statistics and numerics are fine.
* **Tests** are grouped in classes named for the behaviour under test
  (`TestReactionSystemMatrices`), with one assertion theme per method. They
  carry no docstrings; use a short comment where the chemistry, not the code,
  needs explaining. Keep every new behaviour covered at both levels: the unit
  that implements it, and at least one simulator run that shows it works end
  to end.
* **Examples are marimo notebooks.** Each cell's parameters and return tuple
  are marimo's dependency wiring. If you rename or drop a name in a cell, fix
  the signature and the `return` of that cell and of every cell that used it,
  then run the notebooks headlessly with the snippet above.

## When you change the API

Change these together, in one pass:

1. the module and its docstrings,
2. `src/tests/test_<module>.py`,
3. `README.md` — it documents the public API in detail and goes stale easily;
   check the argument tables of both simulators and of `sample_trajectory`,
4. the affected `examples/`,
5. `src/erlenmeyer/__init__.py` if the name is public,
6. the version in `pyproject.toml`.
