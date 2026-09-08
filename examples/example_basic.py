import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from erlenmeyer.ode import ODESimulator
    from erlenmeyer.reaction import Reaction, ReactionSystem
    from erlenmeyer.symbols import Species

    return ODESimulator, Reaction, ReactionSystem, Species, mo, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Basic example

    This is a basic example of a chemical reaction, $\mathrm{2H+O \rightarrow H_2O}$. We simply set the two species as existing in monoatomic state to be able to show how their concentrations change, with a default transition rate of $1.0$.
    """)
    return


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
def _(ODESimulator, system):
    simulator = ODESimulator(system)
    traj = simulator.run({"H": 1.0, "O": 1.0}, t_end=10.0, steps=1000)
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
