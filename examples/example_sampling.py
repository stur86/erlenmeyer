import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from erlenmeyer import Species, Reaction, ReactionSystem, ODESimulator, GillespieSimulator, sample_trajectory
    import matplotlib.pyplot as plt

    return (
        ODESimulator,
        Reaction,
        ReactionSystem,
        Species,
        np,
        plt,
        sample_trajectory,
    )


@app.cell
def _(Reaction, ReactionSystem, Species):
    h2 = Species('H2')
    o2 = Species('O2')
    h2o = Species('H2O')

    water_reaction = Reaction(2*h2+o2, 2*h2o, 1.0)
    wrsys = ReactionSystem([h2, o2, h2o])
    wrsys.add_reaction(water_reaction)
    return (wrsys,)


@app.cell
def _(ODESimulator, sample_trajectory, wrsys):
    ode_sim = ODESimulator(wrsys)
    ode_traj = ode_sim.run({"H2": 0.5, "O2": 0.5}, t_end=20.0)
    ode_traj_sample = sample_trajectory(ode_traj.slice(slice(0, None, 10)), 500)
    return ode_traj, ode_traj_sample


@app.cell
def _(np, ode_traj, ode_traj_sample, plt):
    _f, _a = plt.subplots()

    _a.plot(ode_traj.times, ode_traj.values)
    for _i in range(3):
        _a.plot(ode_traj.times[::10], ode_traj_sample[:,_i]*1.0/np.sum(ode_traj_sample, axis=1), ls='--', c=f'C{_i}', marker='x')
    _a.legend(ode_traj.species)
    return


if __name__ == "__main__":
    app.run()
