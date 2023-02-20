import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.colors
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from colormap import Colormap
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Users shall define their own "value" and "plt_details" choice in the plot_run or plot_trail
# value is a dataframe save the specific value the user would like to graph 
# plt_details is a dataframe save all the ploting details such as legend, title, x_lable, y_lable...

def plot_hist(fp,value, plt_details, save):
    df = pd.read_csv(fp)
    df_fig = df.loc[(df['EVCHANGE'] == value['var']) & (df['SCEN'] == value['scen'])]
    df_fig.plot(x=value['xval'], y=value['yval'], kind=plt_details['kind'], legend=plt_details['legend'], color = plt_details['color'])
    plt.xlabel(plt_details['xlabel']) 
    plt.ylabel(plt_details['ylabel']) 
    plt.title(label=plt_details['title'])
    if save == 1: 
        plt.savefig(plt_details['fig_output'])
        df_pivot=pd.pivot_table(df_fig, values='Value', index=['SCEN'], columns=['AREGWLD'])
        df_pivot.to_csv(plt_details['csv_output'])
        print('Figure and CSV are saved under current directory named as: ', plt_details['fig_output'], 'and', plt_details['csv_output'])
        plt.close()
    else: 
        plt.show()
        print('Figure is not saved. To save the Figure: Change the last input into 1')
        plt.close()


def plot_histscat(fp,value, plt_details, save):
    df = pd.read_csv(fp)
    df = df.loc[(df['SCEN'] == value['scen']) & (df['AREGWLD'] == value['area']) & (df['GDPEX'].isin(value["var"]))]
    df.loc[len(df.index)] = [value['area'], value['new_var'], value['scen'], df['Value'].sum()] 

    df_his = df.loc[df['GDPEX'] == value['new_var']]
    df_sca = df.loc[df['GDPEX'] != value['new_var']]

    sns.set(style="whitegrid")
    sns.barplot(x=value["xval"], y=value["yval"], data=df_his, capsize=.1, errorbar="sd", color = plt_details['color'])
    sns.swarmplot(y=value["yval"], data=df_sca, hue=value["xval"])
    
    if save == 1: 
        plt.savefig(plt_details['fig_output'])
        df.to_csv(plt_details['csv_output'])
        print('Figure and CSV are saved under current directory named as: ', plt_details['fig_output'], 'and', plt_details['csv_output'])
        plt.close()
    else: 
        plt.show()
        print('Figure is not saved. To save the Figure: Change the last input into 1')
        plt.close()


def plot_maps(fp,plt_details, save):

    map_df = gpd.read_file(fp)
    if plt_details['pc'] == False:
        map_df['percent'] = map_df[plt_details['variable']].apply(lambda x: x*100.0)
        plt_details['variable'] = 'percent'
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
    ax.set_title(plt_details['title'], pad=35, fontdict={'fontsize': '10', 'fontweight' : '3'})

    # Colour bar legend
    if plt_details['log'] == False:
        sm = plt.cm.ScalarMappable(cmap=mycmap, norm=plt.Normalize(vmin=plt_details['vmin'], vmax=plt_details['vmax']))
    else:
        sm = plt.cm.ScalarMappable(cmap=mycmap, norm=matplotlib.colors.LogNorm(vmin=plt_details['vmin'], vmax=plt_details['vmax']))

    cbar = plt.colorbar(sm, fraction=0.03, pad=0.04)
    cbar.set_label('% impact', rotation=270, labelpad=10)

    # create map
    if plt_details['log'] == False:
        map_df.plot(column=plt_details['variable'], cmap=mycmap, linewidth=0.8, ax=ax, edgecolor='None', norm=plt.Normalize(vmin=plt_details['vmin'], vmax=plt_details['vmax']))
    else:
        map_df.plot(column=plt_details['variable'], cmap=mycmap, linewidth=0.8, ax=ax, edgecolor='None', norm=matplotlib.colors.LogNorm())

    # Save as PNG
    if save == 1: 
        plt.savefig(plt_details['out_filename'])
        print('Map is saved under current directory named as: ', plt_details['out_filename'])
        plt.close()
    else: 
        plt.show()
        print('Map is not saved. To save the Map: Change the last input into 1.')
        plt.close()


def plot_clustack(fp,value, plt_details, save):
    df = pd.read_csv(fp)
    df = df.loc[df['SCEN'].isin(value["scen"])]

    df['SCEN'] = df['SCEN'].replace(['GlOBPES'],'Global PES')
    df['SCEN'] = df['SCEN'].replace(['NATPES'],'National PES')
    df['SCEN'] = df['SCEN'].replace(['SRLand'],'Subs to Land')
    df['SCEN'] = df['SCEN'].replace(['SRnD20'],'Subs to R&D')
    df['SCEN'] = df['SCEN'].replace(['GPESSRRnD20'],'Combined Policies')

    df_pivot = pd.pivot_table(df, values='Value', index=['AREGWLD', 'SCEN'], columns=['EVDEC'], fill_value=0)
    df_pivot[value['new_var']] = df_pivot['Efficiency']+ df_pivot['Market'] + df_pivot['PES']
    
    ## Plot
    fig = go.Figure()
    
    #Total Welfare: Marker Part
    df_TW = pd.melt(df_pivot.reset_index(), col_level=0, id_vars=['AREGWLD', 'SCEN'], value_vars=['Total Welfare', 'Efficiency','Market','PES'], value_name='Value')
    df_TW = df_TW.loc[(df_TW['EVDEC'] == value['new_var'])]
    fig.add_trace(go.Scatter(x=[df_TW.AREGWLD,df_TW.SCEN], y=df_TW.Value, mode='markers', name='Total Welfare', marker_color = 'green', marker_symbol = "diamond"))
    
    #Bar Chart Part
    df['EVDEC'] = df['EVDEC'].replace(['Efficiency'],'Efficiency Changes')
    df['EVDEC'] = df['EVDEC'].replace(['Market'],'Market-mediated interregional transfers')
    df['EVDEC'] = df['EVDEC'].replace(['PES'],'Fianacial Transfers for PES')

    for n, c in zip(df.EVDEC.unique(), plt_details['color']):
        plot_EVAC_sub = df[df.EVDEC == n]
        fig.add_trace(
        go.Bar(x=[plot_EVAC_sub.AREGWLD,plot_EVAC_sub.SCEN], y=plot_EVAC_sub.Value, name=n, marker_color=c)
    )
    
    fig.update_layout(barmode="relative", title=plt_details['title'], )

    if save == 1: 
        #fig.write_image(plt_details['fig_output'])
        df_pivot.to_csv(plt_details['csv_output'])
        print('Figure and CSV are saved under current directory named as: ', plt_details['fig_output'], 'and', plt_details['csv_output'])
        plt.close()
    else: 
        fig.show()
        print('Figure is not saved. To save the Figure: Change the last input into 1.')


def plot_panel(fp_1a, fp_1b,fp_1c,panel_details, panel_val, save):

    fig = plt.figure(figsize=(8.5,11),dpi=300) #this panel figure is designed to present on a US letter
    #plt.title('Figure 1: Failing to invest in natural capital lowers economic growth, impacting low-income countries most', fontsize=10, pad = 50)
    
    #define the subfigures shape
    subfigs = fig.subfigures(2,1, height_ratios=[0.7, 2], wspace=0.5, hspace = 0.5) #2rows, 1 col; Row1: Panel a&b; Row2: Panel c

    #the upper axes
    ax_up = subfigs[0].subplots(1, 2, sharex=False, sharey = False) #top part has 2cols and 1row

    #the lower axes 
    ax_down = subfigs[1].subplots(3, 1, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()}) #down part has 3rows and 1col


    #read the file for each part
    df_1a = pd.read_csv(fp_1a)
    df_1b = pd.read_csv(fp_1b)
    map_df = gpd.read_file(fp_1c)
  
    #Fig-1a
    df_fig_1a = df_1a.loc[(df_1a['EVCHANGE'] == panel_val['val_fig1a']['var']) & (df_1a['SCEN'] == panel_val['val_fig1a']['scen'])]
    df_fig_1a.plot(x=panel_val['val_fig1a']['xval'], y=panel_val['val_fig1a']['yval'], kind=panel_details['details_fig1a']['kind'], legend=panel_details['details_fig1a']['legend'], ax = ax_up[0], color = '#228471',fontsize=7)
    ax_up[0].set_xlabel(panel_details['details_fig1a']['xlabel'], fontsize = 4) 
    ax_up[0].set_ylabel(panel_details['details_fig1a']['ylabel'] , fontsize = 5) 
    ax_up[0].set_title(label = panel_details['details_fig1a']['title'], fontsize = 9)   
    ax_up[0].set_xticks(np.arange(len(df_fig_1a['AREGWLD'])), fontsize = 1)
    ax_up[0].tick_params(axis="y",direction="in", pad=0, labelsize = 7)
    ax_up[0].tick_params(axis="x",direction="out", pad=5, labelsize = 5,  rotation=00)

    #Fig-1b
    df_1b = df_1b.loc[(df_1b['SCEN'] == panel_val['val_fig1b']['scen']) & (df_1b['AREGWLD'] == panel_val['val_fig1b']['area']) & (df_1b['GDPEX'].isin(panel_val['val_fig1b']["var"]))]
    df_1b.loc[len(df_1b.index)] = [panel_val['val_fig1b']['area'], panel_val['val_fig1b']['new_var'], panel_val['val_fig1b']['scen'], df_1b['Value'].sum()] 

    df1b_his = df_1b.loc[df_1b['GDPEX'] == panel_val['val_fig1b']['new_var']]
    df1b_sca = df_1b.loc[df_1b['GDPEX'] != panel_val['val_fig1b']['new_var']]


    sns.barplot(x=panel_val['val_fig1b']["xval"], y=panel_val['val_fig1b']["yval"], data=df1b_his,capsize=.1, errorbar="sd", color = panel_details['details_fig1b']['color'],ax=ax_up[1])
    sns.swarmplot(y=panel_val['val_fig1b']["yval"], data=df1b_sca, hue=panel_val['val_fig1b']["xval"],ax=ax_up[1])
    ax_up[1].set_title(panel_details['details_fig1b']['title'], fontsize = 9)
    ax_up[1].set_xlabel(panel_details['details_fig1b']['xlabel'], fontsize = 7) 
    ax_up[1].xaxis.set_label_coords(0.5, -0.1)
    ax_up[1].set_ylabel(panel_details['details_fig1b']['ylabel'] , fontsize = 5) 
    ax_up[1].legend(fontsize=4, bbox_to_anchor=(1.02, 0.25), loc='upper left', borderaxespad=0, markerscale=0.5)
    ax_up[1].tick_params(axis="y",direction="in", pad=-25, labelsize=7)
    ax_up[1].tick_params(axis="x",direction="out", pad=5, labelsize=7, labelright = True)

    # Maps
    subfigs[1].suptitle('C. Pollination impact on crop yield', fontsize = 10, y = 0.9, x = 0.15)
    
    for i in range(0, 3):
        c = Colormap()
        mycmap = c.cmap_linear('red', 'white', 'blue')
        ax_down[i].coastlines('10m')

        # Title
        ax_down[i].set_title(panel_details[list(panel_details)[i+2]]['title'], pad=3, fontdict={'fontsize': '6'})

        # create map
        map_df.plot(column=panel_details[list(panel_details)[i+2]]['variable'], cmap=mycmap, linewidth=0.8, ax=ax_down[i], edgecolor='None', norm=plt.Normalize(vmin=panel_details[list(panel_details)[i+2]]['vmin'], vmax=panel_details[list(panel_details)[i+2]]['vmax']))

    # Color Bar

    sm = plt.cm.ScalarMappable(cmap=mycmap, norm=plt.Normalize(vmin=panel_details[list(panel_details)[1+2]]['vmin'], vmax=panel_details[list(panel_details)[1+2]]['vmax']))

    cbar_ax = subfigs[1].add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = subfigs[1].colorbar(sm, fraction=0.0006, pad=0.004, cax=cbar_ax)
    cbar.set_label('Percentage impact by agro-ecological zone and region', rotation=270, labelpad=10)
    
    # Save as PNG
    if save == 1: 
        plt.savefig('Fig1_panel.png')
        print('Figure is saved under current directory named as: ', 'Figure1_panel.png')
        plt.close()
    else: 
        plt.show()
        print('Figure is not saved. To save the Map: Change the last input into 1.')
        plt.close()









