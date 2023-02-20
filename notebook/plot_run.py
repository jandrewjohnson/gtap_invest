import os
from plot_fun import plot_hist, plot_histscat, plot_clustack, plot_maps, plot_panel



wdir_map = "C:\\Users\\Lifeng Ren\\Projects\\matlibplot_928\\"
wdir_csv = "C:\\Users\\Lifeng Ren\\Projects\\matlibplot_928\\PNAS-Sep22\\results\\res\\"
fp_1a = os.path.join(wdir_csv,"EVAR.csv")
fp_1b = os.path.join(wdir_csv,"GEAC.csv")
fp_1c = os.path.join(wdir_map,"gtap2_shockfile.gpkg")
fp_2b = os.path.join(wdir_csv,"EVAC.csv")

value_fig1a = {
    'xval': 'AREGWLD',
    'yval': 'Value',
    'var': 'EVpct',
    'scen': 'BAU_aES'}

plt_details_fig1a = {
    'kind': "bar",
    'color': "#228471",
    'legend': None,
    'ylabel': "$\%$ change compared to baseline without ecosystem service damages",
    'xlabel': None,
    'title': "A. Regional Welfare Losses from degraded ecosystem services",
    'fig_output': "Fig1a.png",
    'csv_output': "Fig1a_pivot.csv"}

value_fig1b = {
    'xval': 'GDPEX',
    'yval': 'Value',
    'var': ['Household', 'Investment', 'Government', 'Exports', 'Imports'], #variable foucsed on 
    'scen': 'BAU_aES', #scenario foucsed on 
    'area': 'World',
    'new_var': 'GDP' #your new variable
    }

plt_details_fig1b = {
    #'kind': "bar",
    'color': "#228471",
    'legend': "None",
    'ylabel': "$\%$ change compared to baseline without ecosystem service damages",
    'xlabel': "Include ES: World",
    'title': "B. GDP change and decomposition",
    'fig_output': "Fig1b.png",
    'csv_output': "Fig1b_pivot.csv"}

plt_details_forest = {
    'variable': 'gtap2_rcp45_ssp2_2030_SR_RnD_20p_PESGC_carbon_forestry_shock',
    'title': "Forestry impact from changing forest cover",
    'out_filename': "forest_map.png",
    'vmin': 95,
    'vmax': 105.0,
    'log' : False,
    'pc' : True,
    }

plt_details_pollination = {
    'variable': 'gtap2_rcp45_ssp2_2030_SR_PESLC_pollination_shock',
    'title': 'Pollination impact on crop yield',
    'out_filename': "pollination_map",
    'vmin': 95,
    'vmax': 105.0,
    'log' : False,
    'pc' : True,
    }

plt_details_fisheries = {
    'variable': 'gtap2_rcp45_ssp2_2030_BAU_marine_fisheries_shock',
    'title': 'Fisheries impact from reduced total-catch biomass',
    'out_filename': "fisheries_map",
    'vmin': 95,
    'vmax': 105.0,
    'log' : False,
    'pc' : True,
    }

value_fig2b = {
    'xval': 'GDPEX',
    'yval': 'Value',
    'var': ['Household', 'Investment', 'Government', 'Exports', 'Imports'], #variable foucsed on 
    'scen': ['GlOBPES','NATPES','SRLand','SRnD20','GPESSRRnD20'], #scenario foucsed on 
    'area': 'All',
    'new_var': 'Total Welfare' #your new variable
    }

plt_details_fig2b = {
    #'kind': "bar",
    'color': ["#4472C4", "#ED7D31", "#A5A5A5"],
    'legend': ['Efficiency Changes','Market-mediated interregional transfers','Fianacial Transfers for PES'],
    'ylabel': "$\%$ change compared to baseline without ecosystem service damages",
    'xlabel': "Include ES: World",
    'title': "Regional welfare change with nature-smart policies",
    'fig_output': "Fig2b.png",
    'csv_output': "Fig2b_pivot.csv"}

panel_details = {
    'details_fig1a': plt_details_fig1a,
    'details_fig1b': plt_details_fig1b,
    'details_pol': plt_details_pollination,
    'details_for': plt_details_forest,
    'details_fis': plt_details_fisheries
}

panel_val = {
    'val_fig1a': value_fig1a,
    'val_fig1b': value_fig1b
}

#RUN AND SAVE: 
plot_hist(fp_1a,value_fig1a, plt_details_fig1a, 1) #fig1a
plot_histscat(fp_1b,value_fig1b, plt_details_fig1b, 1) #fig1b
plot_maps(fp_1c, plt_details_forest, 1) #fig1c-forest
plot_maps(fp_1c, plt_details_pollination, 1) #fig1c-pollination
plot_maps(fp_1c, plt_details_fisheries, 1) #fig1c-fisheries
plot_clustack(fp_2b,value_fig2b, plt_details_fig2b, 1) #fig2b
plot_panel(fp_1a, fp_1b,fp_1c,panel_details, panel_val, 1) #fig1_panel










