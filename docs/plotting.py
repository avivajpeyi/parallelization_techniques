import matplotlib.pyplot as plt


def remove_spines(ax):
    """Remove all spines from an axis"""
    for spine in ax.spines.values():
        spine.set_visible(False)
