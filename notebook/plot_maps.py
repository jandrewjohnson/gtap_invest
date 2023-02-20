import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from colormap import Colormap
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os.path

wdir = "./"

# Cartopy is only needed if you want to add map features like borders
#from cartopy import config
#import cartopy

# log is a Boolean: True if the plot should be on a logarithmic scale
# pc is a Boolean: True if the data is in percentages (i.e., 0-100+); False if
# the data is from 0.0-1.0+.
def plot_maps(fp,variable,vmin,vmax,outfilename,title,log,pc):

    map_df = gpd.read_file(fp)
    if pc == False:
        map_df['percent'] = map_df[variable].apply(lambda x: x*100.0)
        variable = 'percent'
    else:
        pass
    ax = plt.axes(projection=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0)

    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    gl.xlocator = mticker.FixedLocator([])
    gl.ylocator = mticker.FixedLocator([])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    #gl.xlabel_style = {'size': 10, 'color': 'black'}
    #gl.ylabel_style = {'size': 10, 'color': 'black'}
    #gl.xlabel_style = {'color': 'red', 'weight': 'bold'}

    c = Colormap()
    mycmap = c.cmap_linear('red', 'white', 'blue')

    ax.coastlines('10m')

    # Add map features using cartopy:
    #ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
    #ax.add_feature(cartopy.feature.RIVERS)

    # Title
    ax.set_title(title, pad=35, fontdict={'fontsize': '10', 'fontweight' : '3'})

    # Colour bar legend
    if log == False:
        sm = plt.cm.ScalarMappable(cmap=mycmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    else:
        sm = plt.cm.ScalarMappable(cmap=mycmap, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))

    cbar = plt.colorbar(sm, fraction=0.03, pad=0.04)
    cbar.set_label('% impact', rotation=270, labelpad=10)

    # create map
    if log == False:
        map_df.plot(column=variable, cmap=mycmap, linewidth=0.8, ax=ax, edgecolor='None', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    else:
        map_df.plot(column=variable, cmap=mycmap, linewidth=0.8, ax=ax, edgecolor='None', norm=matplotlib.colors.LogNorm())

    # Save as PNG
    plt.savefig(outfilename+'.png', bbox_inches = "tight")


fp = os.path.join(wdir,"gtap2_shockfile.gpkg")
vmin, vmax = 95, 105.0

# Plot Forest map
variable_forest = 'gtap2_rcp45_ssp2_2030_SR_RnD_20p_PESGC_carbon_forestry_shock'
outfilename_forest = "forest_map"
title_forest = 'Forestry impact from changing forest cover'
log = False
pc = True
plot_maps(fp,variable_forest,vmin,vmax,outfilename_forest,title_forest,log,pc)

# Plot Pollination map
variable_pollination = 'gtap2_rcp45_ssp2_2030_SR_PESLC_pollination_shock'
outfilename_pollination = "pollination_map"
title_pollination = 'Pollination impact on crop yield'
log = False
pc = False
plot_maps(fp,variable_pollination,vmin,vmax,outfilename_pollination,title_pollination,log,pc)


# Plot Fisheries map
variable_fisheries = 'gtap2_rcp45_ssp2_2030_BAU_marine_fisheries_shock'
outfilename_fisheries = "fisheries_map"
title_fisheries = 'Fisheries impact from reduced total-catch biomass'
log = False
pc = True
plot_maps(fp,variable_fisheries,vmin,vmax,outfilename_fisheries,title_fisheries,log,pc)
