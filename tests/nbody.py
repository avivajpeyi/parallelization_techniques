from datetime import datetime
from tqdm.auto import trange, tqdm
from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from utils import make_gif, remove_spines, get_runtime, make_colormap, collect_runtimes

G = 1.0  # Gravitational Constant
OUTDIR = 'orbit_out/'


### Basic Lists ###
def newtonian_acceleration_basic(pos, mass, G, softening):
    # positions r = [x,y,z] for all particles
    x = [p[0] for p in pos]
    y = [p[1] for p in pos]
    z = [p[2] for p in pos]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = [[x[j] - x[i] for j in range(len(pos))] for i in range(len(pos))]
    dy = [[y[j] - y[i] for j in range(len(pos))] for i in range(len(pos))]
    dz = [[z[j] - z[i] for j in range(len(pos))] for i in range(len(pos))]

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = [
        [
            (dx[i][j] ** 2 + dy[i][j] ** 2 + dz[i][j] ** 2 + softening ** 2)
            for j in range(len(pos))
        ]
        for i in range(len(pos))
    ]
    for i in range(len(pos)):
        for j in range(len(pos)):
            if inv_r3[i][j] > 0:
                inv_r3[i][j] = inv_r3[i][j] ** (-1.5)

    ax = [G * sum(dx[i][j] * inv_r3[i][j] for j in range(len(pos))) * mass[i] for i in range(len(pos))]
    ay = [G * sum(dy[i][j] * inv_r3[i][j] for j in range(len(pos))) * mass[i] for i in range(len(pos))]
    az = [G * sum(dz[i][j] * inv_r3[i][j] for j in range(len(pos))) * mass[i] for i in range(len(pos))]

    # pack together the acceleration components
    a = [(ax[i], ay[i], az[i]) for i in range(len(pos))]

    return a


def nbody_runner_basic(
        N=5,
        tEnd=5.0,
        dt=0.01,
        softening=0.1,
        random_seed=17,
        max_runtime=50,
):
    """ N-body simulation """

    random.seed(random_seed)
    # Initialisation
    mass = [20.0 / N] * N  # total mass of particles is 20
    pos = [[random.random() for _ in range(3)] for _ in range(N)]  # randomly selected positions and velocities
    vel = [[random.random() for _ in range(3)] for _ in range(N)]
    t = 0
    Nt = int(np.ceil(tEnd / dt))
    runtime_start = datetime.now()
    vel_mean = [sum(m * v for m, v in zip(mass, v)) / sum(mass) for v in zip(*vel)]
    vel = [[v[i] - vel_mean[i] for i in range(3)] for v in vel]
    acc = newtonian_acceleration_basic(pos, mass, G, softening)
    pos_save = [[[0.0 for _ in range(3)] for _ in range(N)] for _ in range(Nt + 1)]
    for i in range(N):
        for j in range(3):
            pos_save[i][j][0] = pos[i][j]

    # Simulation loop
    for i in trange(Nt):
        vel = [[v[j] + acc[i][j] * dt / 2.0 for j in range(3)] for i, v in enumerate(vel)]
        pos = [[pos[i][j] + vel[i][j] * dt for j in range(3)] for i in range(N)]
        acc = newtonian_acceleration_basic(pos, mass, G, softening)
        vel = [[v[j] + acc[i][j] * dt / 2.0 for j in range(3)] for i, v in enumerate(vel)]
        t += dt
        for k in range(N):
            for j in range(3):
                pos_save[i][k][j] = pos[k][j]
        runtime = get_runtime(runtime_start)
        if runtime > max_runtime:
            print(f"Runtime exceeded {max_runtime} seconds. Stopping simulation.")
            break

    pos_save = np.array(pos_save)
    pos_save = np.transpose(pos_save, (1, 2, 0))
    return pos_save


### NP ###
def newtonian_acceleration_np(pos, mass, G, softening):
    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = (dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2)
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

    # pack together the acceleration components
    return G * np.hstack((
        np.matmul(dx * inv_r3, mass),
        np.matmul(dy * inv_r3, mass),
        np.matmul(dz * inv_r3, mass)
    ))


def nbody_runner_np(
        N: int = 5,
        tEnd: float = 10.0,
        dt: float = 0.01,
        softening: float = 0.1,
        random_seed: int = 17,
        max_runtime: int = 2,
):
    """ N-body simulation """

    # Initialisation
    random.seed(random_seed)
    mass = 20.0 * np.ones((N, 1)) / N  # total mass of particles is 20
    pos = np.random.randn(N, 3)  # randomly selected positions and velocities
    vel = np.random.randn(N, 3)
    t = 0
    Nt = int(np.ceil(tEnd / dt))
    runtime_start = datetime.now()
    vel -= np.mean(mass * vel, 0) / np.mean(mass)
    acc = newtonian_acceleration_np(pos, mass, G, softening)
    pos_save = np.zeros((N, 3, Nt + 1))
    pos_save[:, :, 0] = pos

    # Simulation loop
    for i in trange(Nt):
        vel += acc * dt / 2.0
        pos += vel * dt
        acc = newtonian_acceleration_np(pos, mass, G, softening)
        vel += acc * dt / 2.0
        t += dt
        pos_save[:, :, i + 1] = pos
        runtime = get_runtime(runtime_start)
        if runtime > max_runtime:
            print(f"Runtime exceeded {max_runtime} seconds. Stopping simulation.")
            break
    return pos_save


### Cupy ###


### Plotting ###
def plot_runtimes(n_particles, n_trials=3):
    vectorized = collect_runtimes(nbody_runner_np, n_particles, n_trials)
    basic = collect_runtimes(nbody_runner_basic, n_particles, n_trials)

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')
    for i, (data, label) in enumerate(zip([basic, vectorized], ['Basic', 'Vectorized'])):
        ax.plot(n_particles, np.quantile(data, 0.5, axis=1), label=label, color=f'C{i}')
        ax.fill_between(
            n_particles,
            np.quantile(data, 0.05, axis=1),
            np.quantile(data, 0.95, axis=1),
            alpha=0.3,
            color=f'C{i}',
        )
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Runtime (s)')
    ax.legend()
    plt.tight_layout()
    fig.savefig('runtime.png', dpi=300)


def plot_particles(positions: Union[List, np.ndarray], n_time_total: int = 0, color='tab:blue'):
    """Plot the positions of particles in 2D

    Parameters
    ----------
    positions : Union[List, np.ndarray]
        List of positions of particles. Should be of shape (n_particles, {xyz}, n_time).
    n_time_total : int, optional
        Total number of time steps (used for plotting the trail), by default 0
    color : str, optional
        Color of the particles, by default 'tab:blue'
    """
    n_part, _, _ = positions.shape

    fig = plt.figure(figsize=(4, 4), dpi=80)
    ax = fig.gca()

    # plot the particle orbits
    idx_end = np.argmax(np.where(np.sum(positions, axis=(0, 1)) != 0)[0])
    idx_start = np.max([int(idx_end - 0.1 * n_time_total), 0])
    nidx = idx_end - idx_start

    max_size = 10
    ax.scatter(
        positions[:, 0, idx_end], positions[:, 1, idx_end],
        s=max_size, color=color, ec='k', lw=0.5
    )

    # plot the trail
    if nidx > 1:
        ms = np.geomspace(1e-4, max_size, nidx)
        # set ms < 0.05 to 0
        mask = ms < 0.05
        ms[mask] = 0

        # colors = np.array([make_colormap('tab:blue', 'white')(i) for i in np.linspace(0, 1, nidx)])
        ax.scatter(
            positions[:, 0, idx_start:idx_end], positions[:, 1, idx_start:idx_end],
            s=[ms] * n_part, zorder=-10,
            c=[ms] * n_part,
            cmap=make_colormap(color),

        )

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax = plt.gca()
    remove_spines(ax)
    ax.set_aspect('equal', 'box')
    # remove white border around figure
    fig.tight_layout(pad=0)
    return fig


def plot_particle_gif(pos, outdir, dur):
    os.makedirs(outdir, exist_ok=True)
    n_part, _, n_time = pos.shape
    for i in trange(10, n_time, 10):
        fig = plot_particles(pos[:, :, 0:i], n_time_total=n_time, color='tab:blue')
        # add textbox in top left corner
        ax = plt.gca()
        ax.text(
            0.05, 0.95,
            f't={i:003d}', transform=ax.transAxes,
            fontsize=14, verticalalignment='top',
            fontstyle='italic',
            alpha=0.5,
        )
        fig.savefig(f'{outdir}/orbit_{i:003d}.png')
        plt.close(fig)
    make_gif(f'{outdir}/orbit_*.png', f'{outdir}/orbit.gif', duration=dur)


### Main ###


def main():
    # pos = nbody_runner_basic(N=3, random_seed=4)
    # plot_particle_gif(pos, 'out_basic', dur=0.1)
    pos = nbody_runner_np(N=10, random_seed=4)
    plot_particle_gif(pos, 'out_vectorized', dur=0.1)

    # plot_runtimes(n_particles=[10, 20, 50, 100, 200], n_trials=3)


if __name__ == "__main__":
    main()
