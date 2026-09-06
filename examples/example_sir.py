import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from erlenmeyer.gillespie import GillespieSimulator
    from erlenmeyer.ode import ODESimulator
    from erlenmeyer.reaction import Reaction, ReactionSystem
    from erlenmeyer.symbols import Species

    return (
        GillespieSimulator,
        ODESimulator,
        Reaction,
        ReactionSystem,
        Species,
        np,
        plt,
    )


@app.cell
def _(Reaction, ReactionSystem, Species):
    # Classic SIR model of an epidemic. S + I -> 2I is transmission,
    # I -> R is recovery. No reinfection: one single outbreak.
    s = Species("S")
    i = Species("I")
    r = Species("R")

    sir = ReactionSystem([s, i, r])
    sir.add_reaction(Reaction(s + i, 2 * i, 3e-4))
    sir.add_reaction(Reaction(i, r, 0.1))
    return (sir,)


@app.cell
def _(ODESimulator, np, sir):
    traj = ODESimulator(sir).run(np.array([990.0, 10.0, 0.0]), t_end=150.0)
    return (traj,)


@app.cell
def _(plt, traj):
    _f, _a = plt.subplots()
    _a.plot(traj.times, traj.values)
    _a.set_xlabel("time")
    _a.set_ylabel("population")
    _a.set_title("SIR: single outbreak")
    _a.legend(traj.species)
    return


@app.cell
def _(GillespieSimulator, np, sir):
    gill_traj = GillespieSimulator(sir).run(np.array([990, 10, 0]), t_end=150.0, seed=1)
    return (gill_traj,)


@app.cell
def _(gill_traj, plt):
    _f, _a = plt.subplots()
    _a.step(gill_traj.times, gill_traj.values, where="post")
    _a.set_xlabel("time")
    _a.set_ylabel("population")
    _a.set_title("SIR: stochastic (Gillespie)")
    _a.legend(gill_traj.species)
    return


@app.cell
def _(Reaction, ReactionSystem, Species):
    # Add reinfection R -> S. The epidemic now comes in waves that slowly
    # damp out as the population settles toward an endemic equilibrium.
    s2 = Species("S")
    i2 = Species("I")
    r2 = Species("R")

    sir_rec = ReactionSystem([s2, i2, r2])
    sir_rec.add_reaction(Reaction(s2 + i2, 2 * i2, 3e-4))
    sir_rec.add_reaction(Reaction(i2, r2, 0.1))
    sir_rec.add_reaction(Reaction(r2, s2, 0.02))
    return (sir_rec,)


@app.cell
def _(ODESimulator, np, sir_rec):
    traj_waves = ODESimulator(sir_rec).run(
        np.array([990.0, 10.0, 0.0]), t_end=3000.0, steps=5000
    )
    return (traj_waves,)


@app.cell
def _(plt, traj_waves):
    _f, _a = plt.subplots()
    _a.plot(traj_waves.times, traj_waves.values)
    _a.set_xlabel("time")
    _a.set_ylabel("population")
    _a.set_title("SIR with reinfection: epidemic waves")
    _a.legend(traj_waves.species)
    return


if __name__ == "__main__":
    app.run()