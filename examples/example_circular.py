import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from erlenmeyer.ode import ODESimulator
    from erlenmeyer.reaction import Reaction, ReactionSystem
    from erlenmeyer.symbols import Species

    return ODESimulator, Reaction, ReactionSystem, Species, np, plt


@app.cell
def _(Reaction, ReactionSystem, Species):
    # A closed, three-isomer cycle: A -> B -> C -> A. Total mass is
    # conserved. Around the loop the species relax to a stationary
    # distribution that depends only on the three rate constants.
    a = Species("A")
    b = Species("B")
    c = Species("C")

    cycle = ReactionSystem([a, b, c])
    cycle.add_reaction(Reaction(a, b, 1.0))
    cycle.add_reaction(Reaction(b, c, 2.0))
    cycle.add_reaction(Reaction(c, a, 3.0))
    return (cycle,)


@app.cell
def _(ODESimulator, cycle, np):
    traj = ODESimulator(cycle).run(np.array([100.0, 0.0, 0.0]), t_end=10.0, steps=1000)
    return (traj,)


@app.cell
def _(plt, traj):
    _f, _a = plt.subplots()
    _a.plot(traj.times, traj.values)
    _a.set_xlabel("time")
    _a.set_ylabel("population")
    _a.set_title("Circular A -> B -> C -> A")
    _a.legend(traj.species)
    return


@app.cell
def _(traj):
    # Sanity checks: mass is conserved, and the system actually moves.
    total = traj.values.sum(axis=1)
    return (total,)


@app.cell
def _(plt, total):
    _f, _a = plt.subplots()
    _a.plot(total)
    _a.set_xlabel("time index")
    _a.set_ylabel("total population")
    _a.set_title("Total mass is conserved")
    return


if __name__ == "__main__":
    app.run()
