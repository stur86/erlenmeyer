import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from erlenmeyer.ode import ODESimulator
    from erlenmeyer.gillespie import GillespieSimulator, _gillespie_kernel
    from erlenmeyer.symbols import Species
    from erlenmeyer.reaction import Reaction, ReactionSystem
    import matplotlib.pyplot as plt

    print(_gillespie_kernel(np.array([1, 0]), np.array([[1.0, 0]]), np.array([[0, 1.0]]), np.array([1.0]), np.random.default_rng()))
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

    traj = simulator.run(np.array([1.0, 1.0, 0.0]), t_end=10.0, steps=1000)
    return (traj,)


@app.cell
def _(plt, traj):
    _f, _a = plt.subplots()

    _a.plot(traj.times, traj.values)
    return


@app.cell
def _(Reaction, ReactionSystem, Species):
    S = Species('S')
    I = Species('I')
    R = Species('R')

    sir_model = ReactionSystem([S, I, R])
    sir_model.add_reaction(Reaction(S+I, 2*I, 20.0))
    sir_model.add_reaction(Reaction(I, R, 10.0))
    return (sir_model,)


@app.cell
def _(ODESimulator, np, sir_model):
    sir_simulator = ODESimulator(sir_model)
    sir_traj = sir_simulator.run(np.array([1.0, 0.1, 0.0]), t_end=1.0)
    return (sir_traj,)


@app.cell
def _(plt, sir_traj):
    _f, _a = plt.subplots()

    _a.plot(sir_traj.times, sir_traj.values)
    return


@app.cell
def _(GillespieSimulator, np, sir_model):
    sir_gill_simulator = GillespieSimulator(sir_model)
    sir_gill_traj = sir_gill_simulator.run(np.array([100, 1, 0]), t_end=1.0)
    return (sir_gill_traj,)


@app.cell
def _(plt, sir_gill_traj):
    _f, _a = plt.subplots()

    _a.plot(sir_gill_traj.times, sir_gill_traj.values)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
