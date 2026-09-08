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
        GillespieSimulator,
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
    return h2, h2o, o2, wrsys


@app.cell
def _(ODESimulator, np, sample_trajectory, wrsys):
    ode_sim = ODESimulator(wrsys)
    ode_traj = ode_sim.run({"H2": 0.5, "O2": 0.5}, t_end=20.0)
    sample_t = np.linspace(0, 20, 20)
    ode_traj_sample = sample_trajectory(ode_traj, sample_size=500, times=sample_t)
    return ode_traj, ode_traj_sample, sample_t


@app.cell
def _(np, ode_traj, ode_traj_sample, plt, sample_t):
    _f, _a = plt.subplots()

    _a.plot(ode_traj.times, ode_traj.values)
    for _i in range(3):
        _a.plot(sample_t, ode_traj_sample[:,_i]*1.0/np.sum(ode_traj_sample, axis=1), ls='--', c=f'C{_i}', marker='x')
    _a.legend(ode_traj.species)
    return


@app.cell
def _(
    GillespieSimulator,
    Reaction,
    ReactionSystem,
    h2,
    h2o,
    o2,
    sample_t,
    sample_trajectory,
):
    # Gillespie
    water_reaction_slow = Reaction(2*h2+o2, 2*h2o, 1e-6)
    wrsys_slow = ReactionSystem([h2, o2, h2o])
    wrsys_slow.add_reaction(water_reaction_slow)

    gill_sim = GillespieSimulator(wrsys_slow)
    gill_traj = gill_sim.run({"H2": 500, "O2": 500}, t_end=20.0)
    gill_v_frac = 0.1
    gill_traj_sample = sample_trajectory(gill_traj, times=sample_t, volume_fraction=gill_v_frac)
    return gill_traj, gill_traj_sample, gill_v_frac


@app.cell
def _(gill_traj, gill_traj_sample, gill_v_frac, plt, sample_t):
    _f, _a = plt.subplots()

    _a.plot(gill_traj.times, gill_traj.values)
    for _i in range(3):
        _a.plot(sample_t, gill_traj_sample[:,_i]/gill_v_frac, ls='--', c=f'C{_i}', marker='x')
    _a.legend(gill_traj.species)
    return


if __name__ == "__main__":
    app.run()
