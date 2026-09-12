import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt

    from erlenmeyer.ode import ODESimulator
    from erlenmeyer.reaction import Reaction, ReactionSystem
    from erlenmeyer.symbols import Species

    return ODESimulator, Reaction, ReactionSystem, Species, plt


@app.cell
def _(Reaction, ReactionSystem, Species):
    # Lotka-Volterra predator-prey system. X is the prey, Y the predator.
    # X reproduces autocatalytically, Y grows by consuming X, and the
    # predator dies. This sustains oscillations.
    x = Species("X")
    y = Species("Y")

    lv = ReactionSystem([x, y])
    lv.add_reaction(Reaction(x, 2 * x, 1.0))
    lv.add_reaction(Reaction(x + y, 2 * y, 0.02))
    # Death is a decay: products of None mean the predator leaves the system
    # and nothing takes its place. Without that we would have to carry an
    # inert waste species that only ever grows.
    lv.add_reaction(Reaction(y, None, 0.8))
    return (lv,)


@app.cell
def _(ODESimulator, lv):
    traj = ODESimulator(lv).run({"X": 10.0, "Y": 10.0}, t_end=40.0, steps=3000)
    return (traj,)


@app.cell
def _(plt, traj):
    _f, _a = plt.subplots()
    _a.plot(traj.times, traj.values)
    _a.set_xlabel("time")
    _a.set_ylabel("population")
    _a.set_title("Lotka-Volterra: predator-prey oscillations")
    _a.legend(traj.species)
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
