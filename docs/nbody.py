import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tqdm.auto import trange

T0 = datetime.now()
OUTDIR = 'orbit_out/'
os.makedirs(OUTDIR, exist_ok=True)


def remove_spines(ax):
    """Remove all spines and ticks from an axis"""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])



def timestamp():
    ts = (datetime.now() - T0).total_seconds()
    return int(ts * 1000)


def newtonian_acceleration(pos, mass, G, softening):
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

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components
    a = np.hstack((ax, ay, az))

    return a


def plot_orbits(pos_save, color='tab:blue', outdir=OUTDIR):
    # plot the particle orbits
    if isinstance(pos_save, list):
        pos_save = np.array(pos_save)
    n_part, _, n_time = pos_save.shape

    # if fig does not exist, create one
    if not plt.get_fignums():
        plt.figure(figsize=(4, 4), dpi=80)
    plt.cla()

    # nonzero pos idx
    idx = np.argmax(np.where(np.sum(pos_save, axis=(0,1)) != 0)[0])
    max_size = 10
    plt.scatter(pos_save[:, 0, idx], pos_save[:, 1, idx], s=max_size, color=color, edgecolor='k')

    # marker size increasing from 0 to max_size until the idx
    ms = np.geomspace(1e-4, max_size, idx)
    # set ms < 0.05 to 0
    ms[ms < 0.05] = 0
    for i in range(n_part):
        plt.scatter(pos_save[i, 0, :idx], pos_save[i, 1, :idx], s=ms, color=color, zorder=-10)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    ax = plt.gca()
    remove_spines(ax)
    ax.set_aspect('equal', 'box')
    # plt.savefig(f'{outdir}/orbits_{timestamp():003d}.png', dpi=80)
    plt.show()


def main():
    """ N-body simulation """

    # Simulation parameters
    N = 5  # Number of particles
    t = 0  # current time of the simulation
    tEnd = 2.0  # time at which simulation ends
    dt = 0.01  # timestep
    softening = 0.1  # softening length
    G = 1.0  # Newton's Gravitational Constant
    plotRealTime = True  # switch on for plotting as the simulation goes along

    # Generate Initial Conditions
    np.random.seed(17)  # set the random number generator seed

    mass = 20.0 * np.ones((N, 1)) / N  # total mass of particles is 20
    pos = np.random.randn(N, 3)  # randomly selected positions and velocities
    vel = np.random.randn(N, 3)

    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, 0) / np.mean(mass)

    # calculate initial gravitational accelerations
    acc = newtonian_acceleration(pos, mass, G, softening)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # save energies, particle orbits for plotting trails
    pos_save = np.zeros((N, 3, Nt + 1))
    pos_save[:, :, 0] = pos

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=80)

    # Simulation Main Loop
    for i in trange(Nt):
        # (1/2) kick
        vel += acc * dt / 2.0

        # drift
        pos += vel * dt

        # update accelerations
        acc = newtonian_acceleration(pos, mass, G, softening)

        # (1/2) kick
        vel += acc * dt / 2.0

        # update time
        t += dt

        # save energies, positions for plotting trail
        pos_save[:, :, i + 1] = pos

        plot_orbits(pos_save, color='tab:blue')
        # # plot in real time
        # if plotRealTime or (i == Nt - 1):
        #     plot_orbits(pos_save)

    # Save figure
    plot_orbits(pos_save, color='tab:blue')
    plt.savefig('nbody.png', dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
