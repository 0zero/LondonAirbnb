import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pylab as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.colors import n_colors
import plotly.express as px

sns.set_context("paper", font_scale = 2)
sns.axes_style({ 'xtick.direction': 'out', 'ytick.direction': 'out',})
sns.set_style("darkgrid")


def plot_single_column_maps(df, column_name, cmap_label, roundoff=-1, annotate=True):
    """
        Plot multiple columns from a geoPandas DataFrame
    """

    _, axes = plt.subplots(ncols=1, figsize=(15, 12), sharey=True)

    df.plot(
        ax=axes,
        column=column_name,
        figsize=(15, 12),
        alpha=0.9,
        edgecolor='grey',
        legend=True,
        cmap="Oranges",
        vmin=np.round(df[column_name].min(), roundoff),
        vmax=np.round(df[column_name].max(), roundoff),
        legend_kwds={'label': cmap_label, 'orientation': "horizontal"},
    );
    axes.set_axis_off();
    plt.margins(y=0)
    if annotate:
        for _, row in df.iterrows():
            axes.annotate(
                text=np.round(row[column_name], roundoff),
                xy=(row["centre"].x,row["centre"].y),
                horizontalalignment='center',
                fontsize=10,
                weight="bold",
            )


def plot_multiple_column_maps(df, column_list, cmap_label_list, cmap_roundoff=-1, annotate=True, savefig=False):
    """
        Plot multiple columns from a geoPandas DataFrame
    """

    n_cols = len(column_list)
    _, axes = plt.subplots(ncols=n_cols, figsize=(25, 12), sharey=True)
    for i, col in enumerate(column_list):
        df.plot(
            ax=axes[i],
            column=col,
            figsize=(15, 12),
            alpha=0.9,
            edgecolor='grey',
            legend=True,
            cmap="Oranges",
            vmin=np.round(df[col].min(), cmap_roundoff),
            vmax=np.round(df[col].max(), cmap_roundoff),
            legend_kwds={'label': cmap_label_list[i], 'orientation': "horizontal"},
        );
        axes[i].set_axis_off();
        plt.margins(y=0)
        if annotate:
            for _, row in df.iterrows():
                axes[i].annotate(
                    text=int(row[col]),
                    xy=(row["centre"].x ,row["centre"].y),
                    horizontalalignment='center',
                    fontsize=10,
                    weight="bold",
                )

        if savefig:
            plt.savefig("average_price.png" ,dpi=600)


def plot_ridge_plot(df, column_name, xlabel, log_xvalue=False):
    """
        Plot a Plotly ridge plot of a column from a dataframe.
    """

    n_boroughs = len(df["neighbourhood_cleansed"].unique())
    colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', n_boroughs, colortype='rgb')

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

    fig.update_traces(orientation='h', side='positive', width=3, points=False)
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False, width=1500, height=1200, xaxis_title=xlabel,
                      font=dict(size=20))
    fig.show()


def plot_pred_vs_test():
    fig3, ax3 = plt.subplots(figsize=(12,10));

    ax3.scatter(y_test_new_lt1000, forest_preds_lt1000, alpha=0.5, label="price < £1000: $r^2$=0.42");
    ax3.scatter(y_test_new_lt400, forest_preds_400, alpha=0.5, label="price < £400: $r^2$=0.49");
    ax3.plot(np.arange(1000), np.arange(1000), '-k', linewidth=2);
    plt.xlabel("Actual / £");
    plt.ylabel("Predicted / £");
    plt.legend();
    # plt.savefig("scatter.png",dpi=300)
