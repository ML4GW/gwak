import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
import matplotlib as mpl
import inspect


def make_corner(
    embd,
    label_name_list,
    axwidth=2,
    return_fig=False,
    hist_bar: list = ["Glitch", "Background"],
    contour_bins: int = 10,
    contor_levels: int = 1,
    dpi=300,
    # hist_bar: list = ["Background", "Glitch"]
):

    em_dim = embd.shape[1]
    colors = list(mpl.colormaps['Set1'].colors) \
        + list(mpl.colormaps['Dark2'].colors)

    fig, axes = plt.subplots(
        em_dim, em_dim,
        figsize = (em_dim * axwidth, em_dim * axwidth),
        dpi=dpi
    )
    for ax in axes.flat:
        ax.set_axis_off()

    if not all(label in label_name_list for label in hist_bar):
        raise ValueError("All hist_bar labels must be present in label_name_list.")

    unique_labels = list(set(label_name_list))
    unique_labels = hist_bar + [
        label for label in unique_labels
        if label not in hist_bar
    ]

    patches = []

    for il, label in enumerate(unique_labels):
        mask = np.array(label_name_list)==label
        xlims = []

        # Diagonal line
        for i in range(em_dim):

            histtype = "bar" if label in hist_bar else "step"
            alpha = 0.75 if label in hist_bar else 1

            plt.sca(axes[i,i])
            plt.axis('on')
            plt.hist(
                embd[mask, i], bins=20,
                density=False,
                color=colors[il],
                histtype=histtype,
                alpha=alpha,
                linewidth=1.5
            )
            plt.yscale("log")

        # Scatter & Contour
        for i in range(1,em_dim):
            for j in range(i):
                plt.sca(axes[i,j])

                plt.scatter(
                    embd[mask,j], embd[mask,i],
                    color=colors[il],
                    s=0.5,
                )
                plt.xlim(axes[j,j].get_xlim())
                if label in hist_bar:
                    counts, xedges, yedges = np.histogram2d(
                        embd[mask,j],
                        embd[mask,i],
                        bins=contour_bins
                    )
                    xcenters = (xedges[:-1] + xedges[1:]) / 2
                    ycenters = (yedges[:-1] + yedges[1:]) / 2
                    X, Y = np.meshgrid(xcenters, ycenters)
                    plt.contour(
                        X, Y, counts.T,
                        levels=contor_levels,
                        colors=colors[il],
                        alpha=1
                    )

        patches.append(Patch(
            label=label,
            color=colors[il]
        ))

    plt.sca(axes[0,-1])
    plt.legend(handles=patches,ncol=1, fontsize = 18)
    if return_fig:
        return fig
