import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
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
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SIR model

    This is the Susceptible-Infected-Recovered (SIR) model of epidemic spread. It has three species:

    * S, people still Susceptible to the infection
    * I, people currently Infected, and
    * R, people Recovered (and thus now immune) to the infection

    Two reactions describe the dynamics:

    * $S+I \rightarrow 2I$: a healthy individual gets infected by contagion
    * $I \rightarrow R$: an infected individual recovers and gains immunity

    In this model, the disease "burns through" the susceptible population until eventually everyone is immune.

    In a later variation of the model we also add:

    * $R \rightarrow S$: a recovered individual loses immunity over time and goes back to being susceptible

    which produces a "damping waves" dynamic with subsequent epidemics before the equilibrium is reached.
    """)
    return


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
    return i, r, s, sir


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
def _(Reaction, ReactionSystem, i, r, s):
    # Add reinfection R -> S. The epidemic now comes in waves that slowly
    # damp out as the population settles toward an endemic equilibrium.
    sirs_rec = ReactionSystem([s, i, r])
    sirs_rec.add_reaction(Reaction(s + i, 2 * i, 1e-3))
    sirs_rec.add_reaction(Reaction(i, r, 0.18))
    sirs_rec.add_reaction(Reaction(r, s, 0.008))
    return (sirs_rec,)


@app.cell
def _(ODESimulator, np, sirs_rec):
    traj_sirs = ODESimulator(sirs_rec).run(
        np.array([990.0, 10.0, 0.0]), t_end=250.0, steps=5000
    )
    return (traj_sirs,)


@app.cell
def _(plt, traj_sirs):
    _f, _a = plt.subplots()
    _a.plot(traj_sirs.times, traj_sirs.values)
    _a.set_xlabel("time")
    _a.set_ylabel("population")
    _a.set_title("SIRS (SIR + reinfection)")
    _a.legend(traj_sirs.species)
    return


if __name__ == "__main__":
    app.run()
