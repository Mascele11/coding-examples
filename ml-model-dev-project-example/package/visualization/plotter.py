# ======================================================================================================================
#   Libraries
# ======================================================================================================================
import logging
from logging import Logger

from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from plotly.graph_objs import Figure

import seaborn as sns


# ======================================================================================================================
#   Global Configuration
# ======================================================================================================================
# Seaborn theme
sns.set_theme(style="darkgrid")

# setup logging
logger: Logger = logging.getLogger(__name__)


# ======================================================================================================================
#   Functions
# ======================================================================================================================

def lineplot(df: pd.DataFrame, x: str, y: str, x_label: str = None, y_label: str = None,
             hue: str = None, hue_label: str = None, title: str = None, colors: Union[dict, list] = None,
             rotate_x: bool = False, rotate_y: bool = False, filename: Union[Path, str] = None) -> Axes:
    # check input parameters
    if not x_label: x_label = x
    if not y_label: y_label = y
    # parse input parameters
    if filename: filename = Path(filename)
    # setup plotly
    fig: Figure = plt.figure(constrained_layout=True)
    # plot actual chart
    ax: Axes = sns.lineplot(x=x, y=y, hue=hue, data=df, palette=colors)
    # rotate labels
    if rotate_x: plt.xticks(rotation=40, ha="right")
    if rotate_y: plt.yticks(rotation=40, ha="right")
    # write label alias
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # set legend
    if hue_label:
        ax.legend(title=hue_label)
    # set title
    if title:
        ax.set_title(title, fontdict={'fontsize': 13, 'fontweight': 'bold'})
    # save to disk
    if filename:
        plt.savefig(str(filename))
    # show plot
    plt.show()
    # return chart object
    return ax


def barplot(df: Union[pd.DataFrame, np.ndarray], x: str, y: str, x_label: str = None, y_label: str = None, hue: str = None, hue_label: str = None,
            title: str = None, filename: Union[Path, str] = None, rotate_x: bool = False, rotate_y: bool = False,
            colors: Union[dict, list] = None) -> Axes:
    # check input parameters
    if not x_label: x_label = x
    if not y_label: y_label = y
    # parse input parameters
    if filename: filename = Path(filename)
    # setup plotly
    fig: Figure = plt.figure(constrained_layout=True)
    # plot actual chart
    ax: Axes = sns.barplot(x=x, y=y, hue=hue, data=df, palette=colors)
    # rotate labels
    if rotate_x: plt.xticks(rotation=40, ha="right")
    if rotate_y: plt.yticks(rotation=40, ha="right")
    # write label alias
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # set legend
    if hue_label:
        ax.legend(title=hue_label, ncol=int(df[hue].unique().shape[0]/3)+1)
    # set title
    if title:
        ax.set_title(title, fontdict={'fontsize': 13, 'fontweight': 'bold'})
    # save to disk
    if filename:
        plt.savefig(str(filename))
    # show plot
    plt.show()
    # clear matplotlib
    #plt.close()
    # return chart object
    return ax


def boxplot(df: Union[pd.DataFrame, np.ndarray], x: str, y: str, x_label: str = None, y_label: str = None, hue: str = None, hue_label: str = None,
             title: str = None, filename: Union[Path, str] = None, rotate_x: bool = False, rotate_y: bool = False,
            colors: Union[dict, list] = None) -> Axes:
    # check input parameters
    if not x_label: x_label = x
    if not y_label: y_label = y
    # parse input parameters
    if filename: filename = Path(filename)
    # setup plotly
    fig: Figure = plt.figure(constrained_layout=True)
    # plot actual chart
    ax: Axes = sns.boxplot(x=x, y=y, hue=hue, data=df, showfliers=True, palette=colors, whis=[5, 95])
    # rotate labels
    if rotate_x: plt.xticks(rotation=40, ha="right")
    if rotate_y: plt.yticks(rotation=40, ha="right")
    # write label alias
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # set legend
    if hue_label:
        ax.legend(title=hue_label)
    # set title
    if title:
        ax.set_title(title, fontdict={'fontsize': 13, 'fontweight': 'bold'})
    # save to disk
    if filename:
        plt.savefig(str(filename))
    # show plot
    plt.show()
    # clear matplotlib
    #plt.close()
    # return chart object
    return ax


def scatterplot(df: Union[pd.DataFrame, np.ndarray], x: str, y: str, x_label: str = None, y_label: str = None, hue: str = None, hue_label: str = None,
             title: str = None, filename: Union[Path, str] = None, rotate_x: bool = False, rotate_y: bool = False,
            colors: Union[dict, list] = None) -> Axes:
    # check input parameters
    if not x_label: x_label = x
    if not y_label: y_label = y
    # parse input parameters
    if filename: filename = Path(filename)
    # setup plotly
    fig: Figure = plt.figure(constrained_layout=True)
    # plot actual chart
    ax: Axes = sns.scatterplot(x=x, y=y, hue=hue, data=df, palette=colors, alpha=0.8)#, s=df.bubble_size)
    # rotate labels
    if rotate_x: plt.xticks(rotation=40, ha="right")
    if rotate_y: plt.yticks(rotation=40, ha="right")
    # write label alias
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # set legend
    if hue_label:
        ax.legend(title=hue_label, ncol=3, loc='upper right', fontsize='6')
    else:
        plt.legend([], [], frameon=False)
    # set background gradient
    map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#8df796af", "#f0e173af", "#fa6161af"])
    margins: tuple = (df[x].min()-75, df[x].max()+75, df[y].min()-75, df[y].max()+75)
    _gradient_image(ax, direction=-0.5, extent=margins,
                    cmap=map, cmap_range=(0, 1))
    # set title
    if title:
        ax.set_title(title, fontdict={'fontsize': 13, 'fontweight': 'bold'})
    # save to disk
    if filename:
        plt.savefig(str(filename))
    # show plot
    plt.show()
    # clear matplotlib
    #plt.close()
    # return chart object
    return ax


def lineplot_temp(self: pd.DataFrame, x: str, y: str, x_label: str = None, y_label: str = None,
             hue: str = None, hue_label: str = None, title: str = None, colors: Union[dict, list] = None,
             rotate_x: bool = False, rotate_y: bool = False, filename: Union[Path, str] = None) -> Axes:
    # setup plotly
    #fig: Figure = plt.figure(constrained_layout=True)
    # plot actual chart
    ax: Axes = sns.lineplot(x=x, y=y, hue=hue, data=df, palette=colors)
    _decorations(ax, x, y, x_label, y_label, hue, hue_label, title, filename, rotate_x, rotate_y)
    # return chart object
    return ax


# ======================================================================================================================
#   Internal Facilities
# ======================================================================================================================

def _decorations(ax: Axes, x: str, y: str, x_label: str = None, y_label: str = None, hue: str = None, hue_label: str = None,
             title: str = None, filename: Union[Path, str] = None, rotate_x: bool = False, rotate_y: bool = False) -> Axes:
    # check input parameters
    if not x_label: x_label = x
    if not y_label: y_label = y
    # parse input parameters
    if filename: filename = Path(filename)
    # setup plotly
    fig: Figure = plt.figure(constrained_layout=True)
    # rotate labels
    if rotate_x: plt.xticks(rotation=40, ha="right")
    if rotate_y: plt.yticks(rotation=40, ha="right")
    # write label alias
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # set legend
    if hue_label:
        ax.legend(title=hue_label)
    # set title
    if title:
        ax.set_title(title, fontdict={'fontsize': 13, 'fontweight': 'bold'})
    # save to disk
    if filename:
        plt.savefig(str(filename))
    # show plot
    plt.show()
    # return chart object
    return ax


def _gradient_image(ax, extent, direction=0.3, cmap_range=(0, 1), **kwargs):
    """
    Draw a gradient image based on a colormap.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    extent
        The extent of the image as (xmin, xmax, ymin, ymax).
        By default, this is in Axes coordinates but may be
        changed using the *transform* kwarg.
    direction : float
        The direction of the gradient. This is a number in
        range 0 (=vertical) to 1 (=horizontal).
    cmap_range : float, float
        The fraction (cmin, cmax) of the colormap that should be
        used for the gradient, where the complete colormap is (0, 1).
    **kwargs
        Other parameters are passed on to `.Axes.imshow()`.
        In particular useful is *cmap*.
    """
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [1, 0], v @ [1, 1]],
                  [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    im = ax.imshow(X, extent=extent, interpolation='bicubic',
                   vmin=0, vmax=1, aspect='auto', **kwargs)
    return im


# ======================================================================================================================
#   Debugging Entrypoint
# ======================================================================================================================
if __name__ == '__main__':
    """ Debug entrypoint """
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    pd.DataFrame.plot = lineplot_temp

    lineplot(df, x='A', y='B')
    df.plot(x='A', y='B')
