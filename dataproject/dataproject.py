import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from ipywidgets import interact, IntSlider

#set default variables
global ratio
ratio = ""
global column
column = ""
merged2 = pd.DataFrame()
dfK_new1 = pd.DataFrame()
min1 = 0
max1 = 10

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


 #plot for manual columns   
def plot(self, ax=None):
    
    if ax is None:
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
    cmap=seaborn.color_palette("rocket_r", as_cmap=True),
    )


#plot with sliders
def interactive_plot1(self):

    geomerged = gpd.GeoDataFrame(merged2)

    # create a slider for selecting columns
    slider = IntSlider(min=min1, max=max1, step=1, value=1, description='Column:')

    def update_plot(column):

        fig, ax = plt.subplots()
        line, = ax.plot([], [])
        ax.set_title(geomerged.columns[column])

        # update the plot using the selected column
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