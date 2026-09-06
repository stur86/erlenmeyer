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

    return (ODESimulator, Reaction, ReactionSystem, Species, np, plt)


@app.cell
def _(Reaction, ReactionSystem, Species):
    h = Species("H")
    o = Species("O")
    h2o = Species("H2O")

    reaction = Reaction(2 * h + o, h2o, 1.0)
    system = ReactionSystem([h, o, h2o])
    system.add_reaction(reaction)
    return (system,)


@app.cell
def _(ODESimulator, np, system):
    simulator = ODESimulator(system)
    traj = simulator.run(np.array([1.0, 1.0, 0.0]), t_end=10.0, steps=1000)
    return (traj,)


@app.cell
def _(plt, traj):
    _f, _a = plt.subplots()
    _a.plot(traj.times, traj.values)
    _a.set_xlabel("time")
    _a.set_ylabel("concentration")
    _a.legend(traj.species)
    return


if __name__ == "__main__":
    app.run()