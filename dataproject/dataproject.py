import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from ipywidgets import interact, IntSlider

global ratio
ratio = "ratio_65year"
global column
column = ""
merged2 = pd.DataFrame()
dfK_new1 = pd.DataFrame()

def keep_regs(df, regs):
    """ Example function. Keep only the subset regs of regions in data.

    Args:
        df (pd.DataFrame): pandas dataframe 

    Returns:
        df (pd.DataFrame): pandas dataframe

    """ 
    
    for r in regs:
        I = df.reg.str.contains(r)
        df = df.loc[I == False] # keep everything else
    
    return df


    
def plot(self):

    fig, ax = plt.subplots()

    geomerged = gpd.GeoDataFrame(merged2)

    geomerged.plot(
    ax=ax,
    alpha=0.7,
    column=ratio,
    linewidth=0.1,
    edgecolor="#555",
    categorical=False,
    legend=True,
        # cmap="autumn_r",
        # This is a key decision here. Lovely background info:
        # https://seaborn.pydata.org/tutorial/color_palettes.html
        # Use a sequential one.
    cmap=seaborn.color_palette("rocket_r", as_cmap=True),
    )

def interactive_plot(self):

    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    geomerged = gpd.GeoDataFrame(merged2)

    geomerged.plot(
    ax=ax,
    alpha=0.7,
    column=column,
    linewidth=0.1,
    edgecolor="#555",
    categorical=False,
    legend=True,
        # cmap="autumn_r",
        # This is a key decision here. Lovely background info:
        # https://seaborn.pydata.org/tutorial/color_palettes.html
        # Use a sequential one.
    cmap=seaborn.color_palette("rocket_r", as_cmap=True),
    )

    # create a slider for selecting columns
    slider = IntSlider(min=1, max=len(geomerged.columns)-1, step=1, value=1, description='Column:')

    def update_plot(column):
        line.set_data(geomerged['x'], geomerged[geomerged.columns[column]])
        ax.set_title(geomerged.columns[column])
        fig.canvas.draw()

    # create an interactive widget with the slider and the update function
    interact(update_plot, column=slider)

    # show the plot
    plt.show()


def interactive_plot1(self):

    geomerged = gpd.GeoDataFrame(merged2)

    # create a slider for selecting columns
    slider = IntSlider(min=1, max=len(geomerged.columns)-1, step=1, value=1, description='Column:')

    def update_plot(column):

        fig, ax = plt.subplots()
        line, = ax.plot([], [])
        ax.set_title(geomerged.columns[column])

        # update the plot using the selected column
        #print("update_plot called")  # check if function is being called
        #print(column)
        print("column:", geomerged.columns[column])
        ax.clear()
        geomerged.plot(
            ax=ax,
            alpha=0.7,
            column=geomerged.columns[column],
            linewidth=0.1,
            edgecolor="#555",
            categorical=False,
            legend=True,
            cmap=seaborn.color_palette("rocket_r", as_cmap=True),
        )

        fig.canvas.draw()

    # create an interactive widget with the slider and the update function
    interact(update_plot, column=slider)