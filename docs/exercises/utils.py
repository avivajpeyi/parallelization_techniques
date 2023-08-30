import matplotlib.pyplot as plt
import imageio
import os
import glob
import re
import shutil
from datetime import datetime
import numpy as np
from tqdm.auto import trange, tqdm

from matplotlib import rcParams

import colorsys
from matplotlib.colors import ColorConverter, LinearSegmentedColormap

rcParams.update({"xtick.major.pad": "7.0"})
rcParams.update({"xtick.major.size": "7.5"})
rcParams.update({"xtick.major.width": "1.5"})
rcParams.update({"xtick.minor.pad": "7.0"})
rcParams.update({"xtick.minor.size": "3.5"})
rcParams.update({"xtick.minor.width": "1.0"})
rcParams.update({"ytick.major.pad": "7.0"})
rcParams.update({"ytick.major.size": "7.5"})
rcParams.update({"ytick.major.width": "1.5"})
rcParams.update({"ytick.minor.pad": "7.0"})
rcParams.update({"ytick.minor.size": "3.5"})
rcParams.update({"ytick.minor.width": "1.0"})
rcParams.update({"font.size": 20})
rcParams.update({"xtick.top": True})
rcParams.update({"ytick.right": True})
rcParams.update({"xtick.direction": "in"})
rcParams.update({"ytick.direction": "in"})


def collect_runtimes(func, n_vals, n_trials=2, kwargs={}) -> np.ndarray:
    """
    Collect runtimes for a function with different input sizes

    Parameters
    ----------
    func : function
        Function to run
    n_vals : list
        List of input sizes
    n_trials : int, optional
        Number of trials to run, by default 2
    kwargs : dict, optional
        Keyword arguments to pass to func, by default {}

    Returns
    -------
    np.ndarray
        Array of runtimes with shape (len(n_vals), n_trials)
    """
    runtimes = np.zeros((len(n_vals), n_trials))
    for npart_i in trange(len(n_vals), desc="Collecting runtimes"):
        for trial_i in range(n_trials):
            start = datetime.now()
            n = n_vals[npart_i]
            func(N=n, **kwargs)
            runtimes[npart_i, trial_i] = get_runtime(start)
    return runtimes


def get_runtime(start: datetime):
    return (datetime.now() - start).total_seconds()


def make_gif(im_regex, outname, duration=0.1):
    imgs = glob.glob(im_regex)
    imgs = sorted(imgs, key=lambda x: int(re.findall(r"\d+", x)[0]))
    frames = [imageio.imread_v2(f) for f in imgs]
    imageio.mimsave(outname, frames, duration=duration)
    for f in imgs:
        os.remove(f)


def remove_spines(ax):
    """Remove all spines and ticks from an axis"""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])


def scale_color_brightness(color, scale_l=1.0):
    rgb = ColorConverter.to_rgb(color)
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


def make_colormap(color):
    rgb = [
        (i, scale_color_brightness(color, scale_l=i + 1)) for i in np.linspace(0, 1, 30)
    ]
    cmap = LinearSegmentedColormap.from_list(f"custom_{color}", colors=rgb, N=256)
    cmap = cmap.reversed()
    return cmap
