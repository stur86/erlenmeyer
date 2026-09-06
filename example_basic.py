import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from erlenmeyer.ode import ODESimulator
    from erlenmeyer.symbols import Species
    from erlenmeyer.reaction import Reaction, ReactionSystem
    import matplotlib.pyplot as plt

    return ODESimulator, Reaction, ReactionSystem, Species, np, plt


@app.cell
def _(Reaction, ReactionSystem, Species):
    h = Species('H')
    o = Species('O')
    h2o = Species('H2O')

    reaction = Reaction(2*h+o, h2o, 1.0)
    system = ReactionSystem([h, o, h2o])
    system.add_reaction(reaction)
    #system.add_reaction(reaction_2)
    return (system,)


@app.cell
def _(ODESimulator, np, system):
    simulator = ODESimulator(system)

    traj = simulator.run(np.array([1.0, 1.0, 0.0]), t_end=10.0)
    return (traj,)


@app.cell
def _(plt, traj):
    _f, _a = plt.subplots()

    _a.plot(traj.times, traj.values)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
