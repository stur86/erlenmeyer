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
    # Lotka-Volterra predator-prey system. X is the prey, Y the predator.
    # X reproduces autocatalytically, Y grows by consuming X, and the
    # predator dies into waste P. This sustains oscillations.
    x = Species("X")
    y = Species("Y")
    p = Species("P")

    lv = ReactionSystem([x, y, p])
    lv.add_reaction(Reaction(x, 2 * x, 1.0))
    lv.add_reaction(Reaction(x + y, 2 * y, 0.02))
    lv.add_reaction(Reaction(y, p, 0.8))
    return (lv,)


@app.cell
def _(ODESimulator, lv, np):
    traj = ODESimulator(lv).run(np.array([10.0, 10.0, 0.0]), t_end=40.0, steps=3000)
    return (traj,)


@app.cell
def _(plt, traj):
    _f, _a = plt.subplots()
    _a.plot(traj.times, traj.values[:, :2])
    _a.set_xlabel("time")
    _a.set_ylabel("population")
    _a.set_title("Lotka-Volterra: predator-prey oscillations")
    _a.legend(["X", "Y"])
    return


@app.cell
def _(plt, traj):
    # Phase portrait: predator population vs prey population. The closed
    # loop shows the two species chase each other periodically.
    _f, _a = plt.subplots()
    _a.plot(traj.values[:, 0], traj.values[:, 1], lw=0.5)
    _a.set_xlabel("prey X")
    _a.set_ylabel("predator Y")
    _a.set_title("Lotka-Volterra: phase portrait")
    return


if __name__ == "__main__":
    app.run()
