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
    # Brusselator, a classic chemical model with a limit cycle.
    #   A -> X        B + X -> Y + D
    #   2X + Y -> 3X  X -> E
    # A and B are reservoirs: they are so abundant that their slow draw-down
    # barely shifts the dynamics on the simulated timescale, so X and Y
    # oscillate in a sustained, self-excited cycle instead of settling down.
    a = Species("A")
    b = Species("B")
    x = Species("X")
    y = Species("Y")
    d = Species("D")
    e = Species("E")

    br = ReactionSystem([a, b, x, y, d, e])
    # With [A]=1000 and [B]=10000 the effective rates are ~0.001*1000=1 and
    # ~0.0003*10000=3, putting the system above the Hopf threshold b > 1+a^2.
    br.add_reaction(Reaction(a, x, 0.001))
    br.add_reaction(Reaction(b + x, y + d, 0.0003))
    br.add_reaction(Reaction(2 * x + y, 3 * x, 1.0))
    br.add_reaction(Reaction(x, e, 1.0))
    return (br,)


@app.cell
def _(ODESimulator, br):
    traj = ODESimulator(br).run(
        {"A": 1000.0, "B": 10000.0, "X": 1.0, "Y": 1.0}, t_end=60.0, steps=3000
    )
    return (traj,)


@app.cell
def _(plt, traj):
    _f, _a = plt.subplots()
    _a.plot(traj.times, traj.values[:, [2, 3]])
    _a.set_xlabel("time")
    _a.set_ylabel("concentration")
    _a.set_title("Brusselator: chemical oscillations")
    _a.legend(["X", "Y"])
    return


@app.cell
def _(plt, traj):
    # Phase portrait: after a transient, the trajectory settles onto a
    # stable closed loop (the limit cycle).
    _f, _a = plt.subplots()
    _a.plot(traj.values[:, 2], traj.values[:, 3], lw=0.5)
    _a.set_xlabel("X")
    _a.set_ylabel("Y")
    _a.set_title("Brusselator: limit cycle")
    return


if __name__ == "__main__":
    app.run()
