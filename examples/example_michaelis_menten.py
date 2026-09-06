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
    # Michaelis-Menten enzyme mechanism.
    #   S + E <-> ES  (binding and unbinding)
    #   ES -> E + P   (turnover)
    # The enzyme E is regenerated, so it acts as a catalyst.
    s = Species("S")
    e = Species("E")
    es = Species("ES")
    p = Species("P")

    mm = ReactionSystem([s, e, es, p])
    mm.add_reaction(Reaction(s + e, es, 1.0))
    mm.add_reaction(Reaction(es, s + e, 0.1))
    mm.add_reaction(Reaction(es, e + p, 0.5))
    return (mm,)


@app.cell
def _(ODESimulator, mm, np):
    # Use little enzyme so the enzyme cycle is the slow, rate-limiting part.
    traj = ODESimulator(mm).run(np.array([10.0, 0.01, 0.0, 0.0]), t_end=10.0, steps=2000)
    return (traj,)


@app.cell
def _(plt, traj):
    _f, _a = plt.subplots()
    _a.semilogy(traj.times, traj.values)
    _a.set_xlabel("time")
    _a.set_ylabel("concentration")
    _a.set_title("Michaelis-Menten enzyme kinetics")
    _a.legend(traj.species)
    return


@app.cell
def _(ODESimulator, mm, np):
    # Initial velocity as a function of substrate concentration. Even with
    # more and more substrate, the rate levels off: that is the signature
    # saturation behaviour described by the Michaelis-Menten equation
    #   v0 = Vmax * [S0] / (KM + [S0]).
    s0 = np.geomspace(0.03, 30.0, 8)
    km = (0.1 + 0.5) / 1.0  # KM = (k_unbind + k_cat) / k_bind
    vmax = 0.5 * 0.01  # vmax = k_cat * [E0]
    v0 = []
    for s_val in s0:
        _traj = ODESimulator(mm).run(
            np.array([s_val, 0.01, 0.0, 0.0]), t_end=10.0, steps=2000
        )
        # Skip the pre-steady-state burst, then take the slope of P(t)
        _start, _stop = int(3 * 2000 / 10), int(8 * 2000 / 10)
        _t = _traj.times[_start:_stop]
        _p = _traj.values[_start:_stop, 3]
        v0.append(np.polyfit(_t, _p, 1)[0])
    return km, s0, v0, vmax


@app.cell
def _(km, np, plt, s0, v0, vmax):
    _f, _a = plt.subplots()
    _a.semilogx(s0, np.asarray(v0), "o-", label="simulated")
    _s = np.geomspace(s0.min(), s0.max(), 200)
    _a.semilogx(_s, vmax * _s / (km + _s), "--", label="Michaelis-Menten")
    _a.set_xlabel("initial substrate [S0]")
    _a.set_ylabel("initial rate v0")
    _a.set_title("Michaelis-Menten saturation curve")
    _a.legend()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
