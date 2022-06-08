from typing import List
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pylab as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.colors import n_colors

sns.set_context("paper", font_scale=2)
sns.axes_style({"xtick.direction": "out", "ytick.direction": "out"})
sns.set_style("darkgrid")

PLOT_PATH = Path(__file__).parent.parent / "plots"


def plot_single_column_maps(
    df: gpd.GeoDataFrame,
    column_name: str,
    cmap_label: str,
    roundoff: int = -1,
    annotate: bool = True,
    savefig: bool = False,
):
    """
     Plot single column from a geoPandas DataFrame

    :param df: geopandas dataframe
    :param column_name: column to use for plotting
    :param cmap_label: colour map labels
    :param roundoff: degree to which we round colour map
    :param annotate: flag to annotate plot or not
    :param savefig: flag to export figure or not
    :return:
    """

    _, axes = plt.subplots(ncols=1, figsize=(15, 12), sharey=True)

    df.plot(
        ax=axes,
        column=column_name,
        figsize=(15, 12),
        alpha=0.9,
        edgecolor="grey",
        legend=True,
        cmap="Oranges",
        vmin=np.round(df[column_name].min(), roundoff),
        vmax=np.round(df[column_name].max(), roundoff),
        legend_kwds={"label": cmap_label, "orientation": "horizontal"},
    )
    axes.set_axis_off()
    plt.margins(y=0)
    if annotate:
        for _, row in df.iterrows():
            axes.annotate(
                text=np.round(row[column_name], roundoff),
                xy=(row["centre"].x, row["centre"].y),
                horizontalalignment="center",
                fontsize=10,
                weight="bold",
            )
    if savefig:
        figure_name = PLOT_PATH / "review_price.png"
        plt.savefig(figure_name, dpi=600)


def plot_multiple_column_maps(
    df: gpd.GeoDataFrame,
    column_list: List[str],
    cmap_label_list: List[str],
    cmap_roundoff: int = -1,
    figname: str = "average_price",
    annotate: bool = True,
    savefig: bool = False,
):
    """
    Plot multiple columns from a geoPandas DataFrame

    :param figname: name of figure
    :param df: geopandas dataframe
    :param column_list: columns to use for plotting
    :param cmap_label_list: colour map labels
    :param cmap_roundoff: degree to which we round colour map
    :param figname: name of figure
    :param annotate: flag to annotate plot or not
    :param savefig: flag to export figure or not
    :return:
    """

    n_cols = len(column_list)
    _, axes = plt.subplots(ncols=n_cols, figsize=(25, 12), sharey=True)
    for i, col in enumerate(column_list):
        df.plot(
            ax=axes[i],
            column=col,
            figsize=(15, 12),
            alpha=0.9,
            edgecolor="grey",
            legend=True,
            cmap="Oranges",
            vmin=np.round(df[col].min(), cmap_roundoff),
            vmax=np.round(df[col].max(), cmap_roundoff),
            legend_kwds={"label": cmap_label_list[i], "orientation": "horizontal"},
        )
        axes[i].set_axis_off()
        plt.margins(y=0)
        if annotate:
            for _, row in df.iterrows():
                axes[i].annotate(
                    text=int(row[col]),
                    xy=(row["centre"].x, row["centre"].y),
                    horizontalalignment="center",
                    fontsize=10,
                    weight="bold",
                )

    if savefig:
        figure_name = PLOT_PATH / f"{figname}.png"
        plt.savefig(figure_name, dpi=600)


def plot_ridge_plot(
    df: pd.DataFrame,
    column_name: str,
    xlabel: str,
    log_xvalue: bool = False,
    savefig: bool = False,
):
    """
    Plot a Plotly ridge plot of a column from a dataframe.

    :param df: pandas dataframe to use
    :param column_name: column of interest for ridge plot
    :param xlabel: label for x-axis
    :param log_xvalue: flag to decide whether to log10 the x-axis variable
    :param savefig: flag to export figure or not
    :return:
    """

    n_boroughs = len(df["neighbourhood_cleansed"].unique())
    colors = n_colors(
        "rgb(5, 200, 200)", "rgb(200, 10, 10)", n_boroughs, colortype="rgb"
    )

    fig = go.Figure()

    for borough, color in zip(set(df["neighbourhood_cleansed"].unique()), colors):
        if log_xvalue:
            x_value = np.log10(df[df["neighbourhood_cleansed"] == borough][column_name])
        else:
            x_value = df[df["neighbourhood_cleansed"] == borough][column_name]

        fig.add_trace(
            go.Violin(
                x=x_value,
                line_color=color,
                name=borough,
                spanmode="soft",
            )
        )

    fig.update_traces(orientation="h", side="positive", width=3, points=False)
    fig.update_layout(
        xaxis_showgrid=False,
        xaxis_zeroline=False,
        width=1500,
        height=1200,
        xaxis_title=xlabel,
        font=dict(size=20),
    )
    if savefig:
        figure_name = PLOT_PATH / "price_ridge.png"
        fig.write_image(figure_name)
    else:
        fig.show()


def plot_pred_vs_test(
    y_test1,
    y_pred1,
    labels_list: List[str],
    target_limit=1000,
    y_test2=None,
    y_pred2=None,
    savefig: bool = False,
):
    """
    Plot test data versus predictions for two models

    :param y_test1: test target variable from model 1
    :param y_test2: test target variable from model 2
    :param y_pred1: predictions from model 1
    :param y_pred2: predictions from model 2
    :param labels_list: line labels for each model
    :param savefig: flag to export figure or not
    :return:
    """
    _, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(y_test1, y_pred1, alpha=0.5, label=labels_list[0])
    if y_test2 and y_pred2:
        ax.scatter(y_test2, y_pred2, alpha=0.5, label=labels_list[1])
    ax.plot(np.arange(target_limit), np.arange(target_limit), "-k", linewidth=2)
    plt.xlabel("Actual / £")
    plt.ylabel("Predicted / £")
    plt.legend()

    if savefig:
        figure_name = PLOT_PATH / "model_scatter.png"
        plt.savefig(figure_name, dpi=300)
