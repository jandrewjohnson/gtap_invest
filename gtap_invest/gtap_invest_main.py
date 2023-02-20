"""Important interoperability note among versions: this version did not implement the actual gtap-invest logic used for the shocks. This was an old version testing purposes only. The next version merges back in the final shock code used for eg dasgupta tipping points and wb versions."""

from collections import OrderedDict
import hazelbean as hb
import os, sys, time
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
import netCDF4
import pygeoprocessing
# import geoecon as ge
import os
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shapely
from shapely.geometry import Point, MultiPolygon, Polygon
from collections import OrderedDict
import multiprocessing
import subprocess
import threading
from subprocess import Popen, PIPE
import dask
import dask.array

import gtap_invest
import gtap_invest.visualization
from gtap_invest.visualization import visualization
from gtap_invest.visualization.visualization import *

from pygeoprocessing import geoprocessing as gp

L = hb.get_logger()

# TODOO Two major tasks remain, ensure all pyramids are constructed identically wrt ndvs, test/validtate seals
# Recompile cython file if needed.
recompile_cython = 0
if recompile_cython:
    old_cwd = os.getcwd()
    script_dir = os.path.split(os.path.realpath(__file__))[0]

    os.chdir(script_dir)
    cython_command = "python compile_cython_functions.py build_ext -i clean"  #
    returned = os.system(cython_command)
    os.chdir(old_cwd)
    if returned:
        raise NameError('Cythonization failed.')

import global_invest
from global_invest import global_invest_main
import carbon_biophysical
import gtap_invest_integration_functions
from global_invest import pollination_sufficiency
from global_invest.pollination_sufficiency import make_poll_suff

import pes_policy_identification as pes_policy_identification_script

# pollination_sufficiency.make_poll_suff



global_processing_steps = 'NOTE: Global preprocessing steps are not intended to be rerun by users (who instead just use' \
                          'the file generated). They are included in the code for clarity, however.'

# def initialize_paths(p):
#     # TODOO Where to put these? Not really a run config...
#     # To easily convert between per-ha and per-cell terms, these very accurate ha_per_cell maps are defined.
#     p.ha_per_cell_10sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_10sec.tif")
#     p.ha_per_cell_300sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_300sec.tif")
#     p.ha_per_cell_900sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_900sec.tif")
#     p.ha_per_cell_1800sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_1800sec.tif")
#
#     # The ha per cell paths also can be used when writing new tifs as the match path.
#     p.match_10sec_path = p.ha_per_cell_10sec_path
#     p.match_300sec_path = p.ha_per_cell_300sec_path
#     p.match_900sec_path = p.ha_per_cell_900sec_path
#     p.match_1800sec_path = p.ha_per_cell_1800sec_path
#
#     p.ha_per_cell_column_10sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_10sec.tif")
#     p.ha_per_cell_column_300sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_300sec.tif")
#     p.ha_per_cell_column_900sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_900sec.tif")
#     p.ha_per_cell_column_1800sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_1800sec.tif")
#
#     p.visualization_base_path = os.path.join(p.model_base_data_dir, "GTAP_Visualizations_Data")
#     p.visualization_base_path = os.path.join(p.model_base_data_dir, "GTAP_Visualizations_Data")
#     # p.visualization_base_path = os.path.join(p.script_dir, "visualization")
#     p.visualization_config_file_path = os.path.join(p.model_base_data_dir, 'GTAP_Visualizations_Data', 'default_config\config.yaml')
#
#     # If you want to use R to post-process the GTAP sl4 files, include your path to the Rscript.exe file on your system here.
#     p.r_executable_path = r"C:\Program Files\R\R-4.0.3\bin\Rscript.exe"
#
#


def country_gdp_and_pop(p):
    p.country_data_csv_input_path = r"C:\Users\jajohns\Files\Research\base_data\ssps_database\SspDb_country_data_2013-06-12.csv\SspDb_country_data_2013-06-12.csv"
    p.country_data_xlsx_path = os.path.join(p.cur_dir, 'pop_and_gdp.xlsx')

    if p.run_this:
        # Read in data from the ssp database hosted at IIASA
        df = pd.read_csv(p.country_data_csv_input_path)

        # Drop some irrelevant stuff
        df.drop('MODEL', axis=1, inplace=True)

        # Drop everything besides 2015, 2030 and 2050
        df.drop([
            '1950', '1955', '1960', '1965', '1970', '1975', '1980', '1985', '1990', '1995', '2000', '2005', '2010', '2020', '2025', '2035', '2040', '2045', '2055', '2060', '2065', '2070', '2075', '2080', '2085', '2090', '2095', '2100', '2105', '2110', '2115', '2120', '2125', '2130', '2135', '2140', '2145', '2150', ],
            axis=1, inplace=True)

        # Select out population (for later joining)
        df_pop = df[(df['VARIABLE'] == 'Population') | (df['VARIABLE'] == 'GDP|PPP')]

        df_pop['UNIT'] = ['people'] * len(df_pop['VARIABLE'])
        df_pop['UNIT'][df_pop['VARIABLE'] == 'GDP|PPP'] = '2010 USD'
        df_pop_melted = pd.melt(df_pop, id_vars=['SCENARIO', 'REGION', 'VARIABLE'], value_vars=['2015', '2030', '2050'], var_name='YEAR')
        df_pop_melted = df_pop_melted.rename(columns={"REGION": "iso3"})
        df_pop_melted['SCENARIO'] = df_pop_melted['SCENARIO'].str.split('_').str[0]

        df_pop_melted['SCENARIO'] = df_pop_melted['SCENARIO'].str.replace('ssp1', 'rcp26_ssp1')
        df_pop_melted['SCENARIO'] = df_pop_melted['SCENARIO'].str.replace('ssp2', 'rcp45_ssp2')
        df_pop_melted['SCENARIO'] = df_pop_melted['SCENARIO'].str.replace('ssp3', 'rcp70_ssp3')
        df_pop_melted['SCENARIO'] = df_pop_melted['SCENARIO'].str.replace('ssp4', 'rcp34_ssp4')
        df_pop_melted['SCENARIO'] = df_pop_melted['SCENARIO'].str.replace('ssp4d', 'rcp60_ssp4')
        df_pop_melted['SCENARIO'] = df_pop_melted['SCENARIO'].str.replace('ssp5', 'rcp85_ssp5')

        df_pop_melted.to_excel(p.country_data_xlsx_path)


def luh_projections_by_region_aez(p):
    p.gtap37_aez18_stats_vector_path = os.path.join(p.cur_dir, 'gtap37_aez18_stats.gpkg')
    p.gtap37_aez18_path = os.path.join(p.model_base_data_dir, 'gtap_vector_pyramid', "GTAP37_AEZ18.gpkg")

    match_path = r"C:\Users\jajohns\Files\Research\base_data\pyramids\country_ids_15m.tif"
    p.scenario_seals_5_paths = {}

    for year in p.base_years:
        for c, class_name in enumerate(p.class_labels):
            label = class_name + '_baseline_' + str(year)
            file_path = os.path.join(p.luh2_as_seals7_proportion_dir, 'baseline', str(year), class_name + '.tif')
            p.scenario_seals_5_paths[label] = file_path

    for luh_scenario_label in p.luh_scenario_labels:
        for scenario_year in p.scenario_years:
            for c, class_name in enumerate(p.class_labels):
                dir = os.path.join(p.seals7_difference_from_base_year_dir, luh_scenario_label, str(scenario_year))
                label = class_name + '_' + luh_scenario_label.lower() + '_' + str(scenario_year)
                file_path = os.path.join(p.luh2_as_seals7_proportion_dir, luh_scenario_label, str(scenario_year), class_name + '.tif')
                p.scenario_seals_5_paths[label] = file_path


    p.gtap37_aez18_ids_15m_path = os.path.join(p.cur_dir, 'gtap37_aez18_ids_15m.tif')

    # Generate unique_zone_ids early and only once for speed. Note that it must start with a zero.
    # unique_zone_ids = np.asarray([0] + hb.unique_raster_values_path(p.gtap37_aez18_ids_15m_path), dtype=np.int64)
    unique_zone_ids = None # Actually it was faster to do it this way because that pulls from the GDF rather than read the whole raster.

    df = pd.DataFrame()
    p.luh_projections_by_region_aez_path = os.path.join(p.cur_dir, 'luc_projections_by_region_aez.xlsx')
    p.luh_projections_by_region_aez_vector_path = os.path.join(p.cur_dir, 'luc_projections_by_region_aez.gpkg')

    if p.run_this:
        if not hb.path_exists(p.luh_projections_by_region_aez_path):
            for name, path in p.scenario_seals_5_paths.items():
                to_merge = hb.zonal_statistics_flex(path,
                                                    p.gtap37_aez18_path,
                                                    p.gtap37_aez18_ids_15m_path,
                                                    unique_zone_ids=unique_zone_ids,
                                                    id_column_label='pyramid_id',
                                                    zones_raster_data_type=5,
                                                    verbose=True, assert_projections_same=True)
                to_merge = to_merge.rename(columns={'sums': name})
                df = df.merge(to_merge, how='outer', left_index=True, right_index=True)

            gdf = gpd.read_file(p.gtap37_aez18_path)
            gdf = gdf.merge(df, left_on='pyramid_id', right_index=True, how='outer')

            for luh_scenario_label in p.luh_scenario_labels:
                for scenario_year in p.scenario_years:
                    for c, class_name in enumerate(p.class_labels):
                        dir = os.path.join(p.seals7_difference_from_base_year_dir, luh_scenario_label, str(scenario_year))
                        policy_scenario_label = class_name + '_' + luh_scenario_label.lower() + '_' + str(scenario_year)
                        new_label = class_name + '_' + luh_scenario_label.lower() + '_' + str(scenario_year) + '_prop_change'
                        baseline_label = class_name + '_' + 'baseline' + '_' + str(p.base_year)
                        gdf[new_label] = (gdf[policy_scenario_label] / gdf[baseline_label])

            gdf.to_file(p.luh_projections_by_region_aez_vector_path, driver='GPKG')
            df = gdf.drop('geometry', axis=1)
            df.to_excel(p.luh_projections_by_region_aez_path)


def available_land(p):

    p.gtap37_aez18_zones_raster_path = os.path.join(p.cur_dir, 'gtap37_aez18_zones.tif')

    p.available_land_spreadsheet_path = os.path.join(p.cur_dir, 'available_land.csv')
    p.available_land_vector_path = os.path.join(p.cur_dir, 'available_land.gpkg')

    p.final_available_land_spreadsheet_path = os.path.join(p.cur_dir, 'final_available_land.csv')
    p.final_available_land_vector_path = os.path.join(p.cur_dir, 'final_available_land.gpkg')


    def available_land_op(
            parameters,
            nutrient_retention_index,
            oxygen_availability_index,
            rooting_conditions_index,
            toxicity_index,
            workability_index,
            excess_salts_index,
            nutrient_availability_index,
            caloric_yield,
            TRI,
            crop_suitability,
            lulc,
            ha_per_cell,
    ):
        output_array = np.where((nutrient_retention_index <= parameters[0]) & (oxygen_availability_index <= parameters[1]) & (rooting_conditions_index <= parameters[2]) & (toxicity_index <= parameters[3]) &
                                (workability_index <= parameters[4]) & (excess_salts_index <= parameters[5]) & (nutrient_availability_index <= parameters[6])
                                & (caloric_yield > parameters[7]) & (TRI <= parameters[8]) & (crop_suitability >= parameters[9]) & (lulc >= 2) & (lulc <= 5), 1, 0) * ha_per_cell

        return output_array


    if p.run_this:
        p.enumeration_classes = list(range(0, 7+1))
        # NOTE gtap37_aez18_stats_vector_path did not work because some were dropped.

        parameters_dict = {}
        parameters_dict[0] = [1, 1, 1, 1, 1, 1, 1, 160000000000, 5, 60]
        parameters_dict[1] = [2, 2, 2, 2, 2, 2, 2, 80000000000, 8, 50]
        parameters_dict[2] = [3, 3, 3, 3, 3, 3, 3, 40000000000, 10, 40]
        parameters_dict[3] = [4, 4, 4, 4, 4, 4, 4, 20000000000, 20, 30]
        parameters_dict[4] = [5, 5, 5, 5, 5, 5, 5, 10000000000, 30, 20]
        parameters_dict[5] = [6, 6, 6, 6, 6, 6, 6, 5000000000, 40, 10]
        parameters_dict[6] = [7, 7, 7, 7, 7, 7, 7, 1000000000, 10000, 1]



        for k, v in parameters_dict.items():
            metric_path = os.path.join(p.cur_dir, 'arable_definition_' + str(k) + '.tif')
            if not hb.path_exists(metric_path):
                inputs = [
                    v,
                    p.available_land_inputs['nutrient_retention_index'],
                    p.available_land_inputs['oxygen_availability_index'],
                    p.available_land_inputs['rooting_conditions_index'],
                    p.available_land_inputs['toxicity_index'],
                    p.available_land_inputs['workability_index'],
                    p.available_land_inputs['excess_salts_index'],
                    p.available_land_inputs['nutrient_availability_index'],
                    p.available_land_inputs['caloric_yield'],
                    p.available_land_inputs['TRI'],
                    p.available_land_inputs['crop_suitability'],
                    p.base_year_lulc_path,
                    p.ha_per_cell_10sec_path,
                ]
                hb.raster_calculator_af_flex(inputs, available_land_op, metric_path, datatype=6, ndv=-9999.0)

            hb.make_path_global_pyramid(metric_path, verbose=False)

            metric_spreadsheet_path = os.path.join(p.cur_dir, 'arable_definition_' + str(k) + '.csv')
            if not hb.path_exists(metric_spreadsheet_path):
                hb.zonal_statistics_flex(metric_path, p.gtap37_aez18_path, p.gtap37_aez18_zones_raster_path, id_column_label='pyramid_id',
                                         zones_ndv=-9999, values_ndv=-9999., zones_raster_data_type=5,
                                         stats_to_retrieve='sums',
                                         csv_output_path=metric_spreadsheet_path)

        # Also stamp on actually used land, which is obviously arable
        def combine_existing_with_arable(lulc, arable_land, ha_per_cell):
            return np.where((lulc==2) | (arable_land > 0), ha_per_cell, 0)



        for k, v in parameters_dict.items():
            metric_path = os.path.join(p.cur_dir, 'arable_definition_' + str(k) + '.tif')
            augmented_metric_path = os.path.join(p.cur_dir, 'current_and_arable_definition_' + str(k) + '.tif')
            input_list = [p.base_year_simplified_lulc_path, metric_path, p.ha_per_cell_10sec_path]

            if not hb.path_exists(augmented_metric_path):
                L.info('Adding cultivated land ONTO arable land.')
                hb.raster_calculator_af_flex(input_list, combine_existing_with_arable, augmented_metric_path, datatype=7, ndv=-9999.0)
                hb.add_overviews_to_path(augmented_metric_path, specific_overviews_to_add=[6, 60])

            augmented_metric_spreadsheet_path = os.path.join(p.cur_dir, 'current_and_arable_definition_' + str(k) + '.csv')
            if not hb.path_exists(augmented_metric_spreadsheet_path):
                hb.zonal_statistics_flex(augmented_metric_path, p.gtap37_aez18_path, p.gtap37_aez18_zones_raster_path, id_column_label='pyramid_id',
                                         zones_ndv=-9999, values_ndv=-9999., zones_raster_data_type=5,
                                         stats_to_retrieve='sums',
                                         csv_output_path=augmented_metric_spreadsheet_path)


        # Enumerate classes present in each zone. This was slow and so redid just for ag.
        # if not hb.path_exists(p.available_land_vector_path):
        #     hb.zonal_statistics_flex(p.base_year_simplified_lulc_path, p.gtap37_aez18_path, p.gtap37_aez18_zones_raster_path, id_column_label='pyramid_id',
        #                              zones_ndv=-9999, values_ndv=-9999., zones_raster_data_type=5,
        #                              stats_to_retrieve='enumeration', enumeration_classes=p.enumeration_classes, multiply_raster_path=None,
        #                              csv_output_path=p.available_land_spreadsheet_path, vector_output_path=p.available_land_vector_path)

        if not hb.path_exists(p.final_available_land_spreadsheet_path):

            final_df = gpd.read_file(p.gtap37_aez18_path)
            # lulc_ha_df = pd.read_csv(p.available_land_spreadsheet_path, index_col=0)


            for k, v in parameters_dict.items():
                metric_spreadsheet_path = os.path.join(p.cur_dir, 'arable_definition_' + str(k) + '.csv')
                augmented_metric_spreadsheet_path = os.path.join(p.cur_dir, 'current_and_arable_definition_' + str(k) + '.csv')

                arable_def_df = pd.read_csv(metric_spreadsheet_path, index_col=0)
                arable_def_df = arable_def_df.rename(columns={'sums': 'arable_definition_' + str(k)})
                final_df = final_df.merge(arable_def_df, how='outer', left_on='pyramid_id', right_index=True)

                arable_def_df = pd.read_csv(augmented_metric_spreadsheet_path, index_col=0)
                arable_def_df = arable_def_df.rename(columns={'sums': 'current_and_arable_definition_' + str(k)})
                final_df = final_df.merge(arable_def_df, how='outer', left_on='pyramid_id', right_index=True)

                final_df['ha_pch_' + str(k)] = final_df['arable_definition_' + str(k)] / (final_df['current_and_arable_definition_' + str(k)] - final_df['arable_definition_' + str(k)]) * 100

                # final_df['ha_pch_' + str(k)] = (final_df['arable_definition_' + str(k)] / final_df['ha_cropland_2']) * 100

            final_df.to_file(p.final_available_land_vector_path, driver='GPKG')

            final_df = final_df.drop(columns='geometry')
            final_df.to_csv(p.final_available_land_spreadsheet_path)



def gtap1_aez_uris(p):
    """Run a precompiled GTAPAEZ.exe file by calling the and a cmf file.
    Additionally and somewhat confusingly, this is also where I enable the ability to run GTAP-AEZ via an external process. The way this works is it copies the
    code for GTAP-InVEST into the ProjectFlow p.cur_dir, runs it, and saves a result in a logical place. HOWEVER, if the process is instead run via a batch file
    the way it was done before, it saves it to a different location. Subsequent tasks choose the manual .bat result over the internal one IF IT EXISTS. So
    if you have previously run a manual run and now want it to be in ProjectFlow, just make sure to delete tehe gtap_result directory.
    """


    p.gtap1_aez_invest_local_model_dir = os.path.join(p.cur_dir, p.gtap_aez_invest_release_string)

    if p.run_this:


        # Extract a gtap-aez-invest zipfile into the curdir
        if not hb.path_exists(p.gtap1_aez_invest_local_model_dir):

            # Redundant step here. Need to eliminate the copy in base_data dir and just have erwin/uris push the code to the repository? But what about the sl4 and data files?
            # If the gtap-aez code doesn't exist in the repository, copy it from the base data to the project dir (wierd I know)
            if not hb.path_exists(p.gtap_aez_invest_code_dir):
                L.info('Unzipping all files in ' + p.gtap_aez_invest_zipfile_path + ' to ' + p.gtap1_aez_invest_local_model_dir)
                hb.unzip_file(p.gtap_aez_invest_zipfile_path, p.cur_dir, verbose=False)

            # If it does exist in the repo, copy it from there.
            else:
                L.info('Creating project-specific copy of GTAP files, copying from ' + p.gtap_aez_invest_code_dir + ' to ' + p.gtap1_aez_invest_local_model_dir)
                hb.copy_file_tree_to_new_root(p.gtap_aez_invest_code_dir, p.gtap1_aez_invest_local_model_dir)

        hb.create_directories(os.path.join(p.gtap1_aez_invest_local_model_dir, 'work'))

        # There was a typo in Uris' scenario names, fixd here.
        src = os.path.join(p.gtap1_aez_invest_local_model_dir, 'cmfs', '2021_30_SR_RnD_20p_PESGB_30_noES.cmf')
        dst = os.path.join(p.gtap1_aez_invest_local_model_dir, 'cmfs', '2021_30_SR_RnD_20p_PESGC_30_noES.cmf')
        if hb.path_exists(src):
            hb.copy_shutil_flex(src, dst)

        # One trivial error I was never able to trace down was that that a file was missing, triggerering the following error
        # %% UNABLE TO OPEN EXISTING FILE '2021_30_BAU_rigid_noES_SUPP.har'.
        # Uris said I can just:
        # if there is a file “2021_30_BAU_noES_supp.HAR” you can just duplicate that and rename to “'2021_30_BAU_rigid_noES_SUPP.har”
        # so yeah, here I do that:
        src = os.path.join(p.gtap1_aez_invest_local_model_dir, '2021_30_BAU_allES_supp.har')
        dst = os.path.join(p.gtap1_aez_invest_local_model_dir, '2021_30_BAU_rigid_noES_SUPP.har')
        hb.copy_shutil_flex(src, dst)


        gtapaez_executable_abs_path = os.path.join(p.gtap1_aez_invest_local_model_dir, 'GTAPAEZ.exe')

        # Define paths for the source cmf file (extracted from GTAP-AEZ integration zipfile) and the modified one that will be run
        gtap_policy_baseline_scenario_label = str(p.base_year) + '_' + str(p.policy_base_year)[2:] + '_BAU'
        gtap_policy_baseline_scenario_source_cmf_path = os.path.join(p.gtap1_aez_invest_local_model_dir, 'cmfs', gtap_policy_baseline_scenario_label + '.cmf')
        gtap_policy_baseline_scenario_cmf_path = os.path.join(p.gtap1_aez_invest_local_model_dir, gtap_policy_baseline_scenario_label + '_local.cmf')
        gtap_policy_baseline_solution_file_path = os.path.join(p.cur_dir, p.gtap_aez_invest_release_string, 'work', gtap_policy_baseline_scenario_label + '.sl4')

        L.info('gtap_policy_baseline_scenario_cmf_path', gtap_policy_baseline_scenario_cmf_path)
        L.info('gtap_policy_baseline_solution_file_path', gtap_policy_baseline_solution_file_path)

        if not hb.path_exists(gtap_policy_baseline_solution_file_path, verbose=True):
            # Generate a new cmf file with updated paths.
            gtap_invest_integration_functions.generate_policy_baseline_cmf_file(gtap_policy_baseline_scenario_source_cmf_path, gtap_policy_baseline_scenario_cmf_path)

            # Run the gtap executable pointing to the new cmf file
            call_list = [gtapaez_executable_abs_path, '-cmf', gtap_policy_baseline_scenario_cmf_path]
            gtap_invest_integration_functions.run_gtap_cmf(gtap_policy_baseline_scenario_label, call_list)

        run_parallel = 1
        parallel_iterable = []
        for gtap_scenario_label in p.gtap1_scenario_labels:

            current_scenario_source_cmf_path = os.path.join(p.gtap1_aez_invest_local_model_dir, 'cmfs', gtap_scenario_label + '.cmf')
            current_scenario_cmf_path = os.path.join(p.gtap1_aez_invest_local_model_dir, gtap_scenario_label + '_local.cmf')

            current_solution_file_path = os.path.join(p.cur_dir, p.gtap_aez_invest_release_string, 'work', gtap_scenario_label + '.sl4')

            # Hack to fix uris scenario typo
            src = os.path.join(p.gtap1_aez_invest_local_model_dir, 'cmfs', '2021_30_SR_RnD_20p_PESGB_30_allES.cmf')
            dst = os.path.join(p.gtap1_aez_invest_local_model_dir, 'cmfs', '2021_30_SR_RnD_20p_PESGC_30_allES.cmf')
            hb.copy_shutil_flex(src, dst)

            possible_file_names = [current_solution_file_path]
            if 'PESGC' in current_solution_file_path:
                possible_file_names.append(current_solution_file_path.replace('PESGC', 'PESGB'))


            if not any([hb.path_exists(i, verbose=True) for i in possible_file_names]):
                # Generate a new cmf file with updated paths.
                # Currently this just uses the policy_baseline version.
                gtap_invest_integration_functions.generate_policy_baseline_cmf_file(current_scenario_source_cmf_path, current_scenario_cmf_path)

                # Run the gtap executable pointing to the new cmf file
                call_list = [gtapaez_executable_abs_path, '-cmf', current_scenario_cmf_path]
                parallel_iterable.append(tuple([gtap_scenario_label, call_list]))

                # If the model is run sequentially, just call it here.
                if not run_parallel:
                    gtap_invest_integration_functions.run_gtap_cmf(gtap_scenario_label, call_list)


        if run_parallel:
            # Performance note: it takes about 3 seconds to run this block even with nothing in the iterable, I guess just from launching the worker pool
            if len(parallel_iterable) > 0:

                worker_pool = multiprocessing.Pool(p.num_workers)  # NOTE, worker pool and results are LOCAL variabes so that they aren't pickled when we pass the project object.

                finished_results = []
                result = worker_pool.starmap_async(gtap_invest_integration_functions.run_gtap_cmf, parallel_iterable)
                for i in result.get():
                    finished_results.append(i)
                worker_pool.close()
                worker_pool.join()

        # Now call the R code to pull these
        src_r_postsim_script_path = os.path.join(p.gtap1_aez_invest_local_model_dir, 'postsims', '01_output_csv.r')

        # Create a local copy of the R file with modifications specific to this run (i.e., the shanging the workspace)
        r_postsim_script_path = os.path.join(p.gtap1_aez_invest_local_model_dir, 'postsims', '01_output_csv_local.r')



        # Copy the time-stamped gtap2 results to a non-time-stamped version to clarify which one to use for eg plotting.
        p.gtap1_results_path = os.path.join(p.cur_dir, 'GTAP_Results.csv')
        p.gtap1_land_use_change_path = os.path.join(p.cur_dir, 'GTAP_AEZ_LCOVER_ha.csv')
        current_date = hb.pretty_time('year_month_day_hyphens')
        dated_expected_files = [os.path.join(p.cur_dir, current_date +'_GTAP_Results.csv'), os.path.join(p.cur_dir, current_date +'_GTAP_AEZ_LCOVER_ha.csv')]
        expected_files = [p.gtap2_results_path, p.gtap2_land_use_change_path]

        for c, file_path in enumerate(dated_expected_files):
            if hb.path_exists(file_path):
                hb.copy_shutil_flex(file_path, expected_files[c])

        gtap1_aez_results_exist = all([True if hb.path_exists(i) else False for i in expected_files])

        if not gtap1_aez_results_exist:
            L.info('Starting r script at ' + str(r_postsim_script_path) + ' to create output files.')

            os.makedirs(os.path.join(os.path.split(src_r_postsim_script_path)[0], 'temp'), exist_ok=True)
            os.makedirs(os.path.join(os.path.split(src_r_postsim_script_path)[0], 'temp', 'merge'), exist_ok=True)

            # TODOO, This is a silly duplication of non-small files that could be eliminated. originally it was in so that i didn't have to modify Uris' r code.
            # TODOO Also, most of this R code is just running har2csv.exe, which is a license-constrained gempack file. replace with python har2csv
            hb.copy_file_tree_to_new_root(os.path.join(p.gtap1_aez_invest_local_model_dir, 'work'), os.path.join(p.gtap1_aez_invest_local_model_dir, 'postsims', 'in', 'gtap'))

            # working_dir = implied_r_working_dir = os.path.split(src_r_script_path)[0]
            gtap_invest_integration_functions.generate_postsims_r_script_file(src_r_postsim_script_path, r_postsim_script_path)

            hb.execute_r_script(p.r_executable_path, os.path.abspath(r_postsim_script_path))


            # The two CSVs generated by the script file are key outputs. Copy them to the cur_dir root as well as the output dir
            files_to_copy = [os.path.join(p.gtap1_aez_invest_local_model_dir, 'postsims', 'out', os.path.split(i)[1]) for i in dated_expected_files]

            for file_path in files_to_copy:
                hb.copy_shutil_flex(file_path, os.path.join(p.cur_dir, os.path.split(file_path)[1]), verbose=True)
                hb.copy_shutil_flex(file_path, os.path.join(p.output_dir, 'gtap1_aez', os.path.split(file_path)[1]), verbose=True)
                hb.copy_shutil_flex(file_path, os.path.join(p.cur_dir, os.path.split(expected_files[c])[1]), verbose=True)

def gtap1_aez(p):
    """Run a precompiled GTAPAEZ.exe file by calling the and a cmf file.
    Additionally and somewhat confusingly, this is also where I enable the ability to run GTAP-AEZ via an external process. The way this works is it copies the
    code for GTAP-InVEST into the ProjectFlow p.cur_dir, runs it, and saves a result in a logical place. HOWEVER, if the process is instead run via a batch file
    the way it was done before, it saves it to a different location. Subsequent tasks choose the manual .bat result over the internal one IF IT EXISTS. So
    if you have previously run a manual run and now want it to be in ProjectFlow, just make sure to delete tehe gtap_result directory.
    """


    p.gtap1_aez_invest_local_model_dir = os.path.join(p.cur_dir, p.gtap_aez_invest_release_string)

    if p.run_this:


        # Extract a gtap-aez-invest zipfile into the curdir
        if not hb.path_exists(p.gtap1_aez_invest_local_model_dir):

            # Redundant step here. Need to eliminate the copy in base_data dir and just have erwin/uris push the code to the repository? But what about the sl4 and data files?
            # If the gtap-aez code doesn't exist in the repository, copy it from the base data to the project dir (wierd I know)
            if not hb.path_exists(p.gtap_aez_invest_code_dir):
                L.info('Unzipping all files in ' + p.gtap_aez_invest_zipfile_path + ' to ' + p.gtap1_aez_invest_local_model_dir)
                hb.unzip_file(p.gtap_aez_invest_zipfile_path, p.cur_dir, verbose=False)

            # If it does exist in the repo, copy it from there.
            else:
                L.info('Creating project-specific copy of GTAP files, copying from ' + p.gtap_aez_invest_code_dir + ' to ' + p.gtap1_aez_invest_local_model_dir)
                hb.copy_file_tree_to_new_root(p.gtap_aez_invest_code_dir, p.gtap1_aez_invest_local_model_dir)

        hb.create_directories(os.path.join(p.gtap1_aez_invest_local_model_dir, 'work'))

        # There was a typo in Uris' scenario names, fixd here.
        src = os.path.join(p.gtap1_aez_invest_local_model_dir, 'cmfs', '2021_30_SR_RnD_20p_PESGB_30_noES.cmf')
        dst = os.path.join(p.gtap1_aez_invest_local_model_dir, 'cmfs', '2021_30_SR_RnD_20p_PESGC_30_noES.cmf')
        if hb.path_exists(src):
            hb.copy_shutil_flex(src, dst)

        # One trivial error I was never able to trace down was that that a file was missing, triggerering the following error
        # %% UNABLE TO OPEN EXISTING FILE '2021_30_BAU_rigid_noES_SUPP.har'.
        # Uris said I can just:
        # if there is a file “2021_30_BAU_noES_supp.HAR” you can just duplicate that and rename to “'2021_30_BAU_rigid_noES_SUPP.har”
        # so yeah, here I do that:
        src = os.path.join(p.gtap1_aez_invest_local_model_dir, '2021_30_BAU_allES_supp.har')
        dst = os.path.join(p.gtap1_aez_invest_local_model_dir, '2021_30_BAU_rigid_noES_SUPP.har')
        hb.copy_shutil_flex(src, dst)


        gtapaez_executable_abs_path = os.path.join(p.gtap1_aez_invest_local_model_dir, 'GTAPAEZ.exe')

        # Define paths for the source cmf file (extracted from GTAP-AEZ integration zipfile) and the modified one that will be run
        gtap_policy_baseline_scenario_label = str(p.base_year) + '_' + str(p.policy_base_year)[2:] + '_BAU'
        gtap_policy_baseline_scenario_source_cmf_path = os.path.join(p.gtap1_aez_invest_local_model_dir, 'cmfs', gtap_policy_baseline_scenario_label + '.cmf')
        gtap_policy_baseline_scenario_cmf_path = os.path.join(p.gtap1_aez_invest_local_model_dir, gtap_policy_baseline_scenario_label + '_local.cmf')
        gtap_policy_baseline_solution_file_path = os.path.join(p.cur_dir, p.gtap_aez_invest_release_string, 'work', gtap_policy_baseline_scenario_label + '.sl4')

        L.info('gtap_policy_baseline_scenario_cmf_path', gtap_policy_baseline_scenario_cmf_path)
        L.info('gtap_policy_baseline_solution_file_path', gtap_policy_baseline_solution_file_path)

        if not hb.path_exists(gtap_policy_baseline_solution_file_path, verbose=True):
            # Generate a new cmf file with updated paths.
            gtap_invest_integration_functions.generate_policy_baseline_cmf_file(gtap_policy_baseline_scenario_source_cmf_path, gtap_policy_baseline_scenario_cmf_path)

            # Run the gtap executable pointing to the new cmf file
            call_list = [gtapaez_executable_abs_path, '-cmf', gtap_policy_baseline_scenario_cmf_path]
            gtap_invest_integration_functions.run_gtap_cmf(gtap_policy_baseline_scenario_label, call_list)

        run_parallel = 1
        parallel_iterable = []
        for gtap_scenario_label in p.gtap1_scenario_labels:

            current_scenario_source_cmf_path = os.path.join(p.gtap1_aez_invest_local_model_dir, 'cmfs', gtap_scenario_label + '.cmf')
            current_scenario_cmf_path = os.path.join(p.gtap1_aez_invest_local_model_dir, gtap_scenario_label + '_local.cmf')

            current_solution_file_path = os.path.join(p.cur_dir, p.gtap_aez_invest_release_string, 'work', gtap_scenario_label + '.sl4')

            # Hack to fix uris scenario typo
            src = os.path.join(p.gtap1_aez_invest_local_model_dir, 'cmfs', '2021_30_SR_RnD_20p_PESGB_30_allES.cmf')
            dst = os.path.join(p.gtap1_aez_invest_local_model_dir, 'cmfs', '2021_30_SR_RnD_20p_PESGC_30_allES.cmf')
            hb.copy_shutil_flex(src, dst)

            possible_file_names = [current_solution_file_path]
            if 'PESGC' in current_solution_file_path:
                possible_file_names.append(current_solution_file_path.replace('PESGC', 'PESGB'))


            if not any([hb.path_exists(i, verbose=True) for i in possible_file_names]):
                # Generate a new cmf file with updated paths.
                # Currently this just uses the policy_baseline version.
                gtap_invest_integration_functions.generate_policy_baseline_cmf_file(current_scenario_source_cmf_path, current_scenario_cmf_path)

                # Run the gtap executable pointing to the new cmf file
                call_list = [gtapaez_executable_abs_path, '-cmf', current_scenario_cmf_path]
                parallel_iterable.append(tuple([gtap_scenario_label, call_list]))

                # If the model is run sequentially, just call it here.
                if not run_parallel:
                    gtap_invest_integration_functions.run_gtap_cmf(gtap_scenario_label, call_list)


        if run_parallel:
            # Performance note: it takes about 3 seconds to run this block even with nothing in the iterable, I guess just from launching the worker pool
            if len(parallel_iterable) > 0:

                worker_pool = multiprocessing.Pool(p.num_workers)  # NOTE, worker pool and results are LOCAL variabes so that they aren't pickled when we pass the project object.

                finished_results = []
                result = worker_pool.starmap_async(gtap_invest_integration_functions.run_gtap_cmf, parallel_iterable)
                for i in result.get():
                    finished_results.append(i)
                worker_pool.close()
                worker_pool.join()

        # Now call the R code to pull these
        src_r_postsim_script_path = os.path.join(p.gtap1_aez_invest_local_model_dir, 'postsims', '01_output_csv.r')

        # Create a local copy of the R file with modifications specific to this run (i.e., the shanging the workspace)
        r_postsim_script_path = os.path.join(p.gtap1_aez_invest_local_model_dir, 'postsims', '01_output_csv_local.r')



        # Copy the time-stamped gtap2 results to a non-time-stamped version to clarify which one to use for eg plotting.
        p.gtap1_results_path = os.path.join(p.cur_dir, 'GTAP_Results.csv')
        p.gtap1_land_use_change_path = os.path.join(p.cur_dir, 'GTAP_AEZ_LCOVER_ha.csv')
        current_date = hb.pretty_time('year_month_day_hyphens')
        dated_expected_files = [os.path.join(p.cur_dir, current_date +'_GTAP_Results.csv'), os.path.join(p.cur_dir, current_date +'_GTAP_AEZ_LCOVER_ha.csv')]
        expected_files = [p.gtap1_results_path, p.gtap1_land_use_change_path]

        for c, file_path in enumerate(dated_expected_files):
            if hb.path_exists(file_path):
                hb.copy_shutil_flex(file_path, expected_files[c])

        gtap1_aez_results_exist = all([True if hb.path_exists(i) else False for i in expected_files])

        if not gtap1_aez_results_exist:
            L.info('Starting r script at ' + str(r_postsim_script_path) + ' to create output files.')

            os.makedirs(os.path.join(os.path.split(src_r_postsim_script_path)[0], 'temp'), exist_ok=True)
            os.makedirs(os.path.join(os.path.split(src_r_postsim_script_path)[0], 'temp', 'merge'), exist_ok=True)

            # TODOO, This is a silly duplication of non-small files that could be eliminated. originally it was in so that i didn't have to modify Uris' r code.
            # TODOO Also, most of this R code is just running har2csv.exe, which is a license-constrained gempack file. replace with python har2csv
            hb.copy_file_tree_to_new_root(os.path.join(p.gtap1_aez_invest_local_model_dir, 'work'), os.path.join(p.gtap1_aez_invest_local_model_dir, 'postsims', 'in', 'gtap'))

            # working_dir = implied_r_working_dir = os.path.split(src_r_script_path)[0]
            gtap_invest_integration_functions.generate_postsims_r_script_file(src_r_postsim_script_path, r_postsim_script_path)

            hb.execute_r_script(p.r_executable_path, os.path.abspath(r_postsim_script_path))


            # The two CSVs generated by the script file are key outputs. Copy them to the cur_dir root as well as the output dir
            files_to_copy = [os.path.join(p.gtap1_aez_invest_local_model_dir, 'postsims', 'out', os.path.split(i)[1]) for i in dated_expected_files]

            for file_path in files_to_copy:
                hb.copy_shutil_flex(file_path, os.path.join(p.cur_dir, os.path.split(file_path)[1]), verbose=True)
                hb.copy_shutil_flex(file_path, os.path.join(p.output_dir, 'gtap1_aez', os.path.split(file_path)[1]), verbose=True)
                hb.copy_shutil_flex(file_path, os.path.join(p.cur_dir, os.path.split(expected_files[c])[1]), verbose=True)





def gtap1_extracts_from_solution(p):
    if p.run_this:
        hb.create_directories(os.path.join(p.cur_dir, 'raw_sl4_extracts'))

        # Extract .sl4 format to a raw_csv
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:

                    # Hack to fix rename
                    if policy_scenario_label == 'SR_RnD_20p_PESGC_30':
                        policy_scenario_label = policy_scenario_label.replace('SR_RnD_20p_PESGC_30', 'SR_RnD_20p_PESGB_30')

                    gtap_scenario_label = '2021_30_' + policy_scenario_label + '_noES'
                    gtap_sl4_path_no_extension = os.path.join(p.gtap1_aez_invest_local_model_dir, 'work', gtap_scenario_label)
                    gtap_sl4_path = gtap_sl4_path_no_extension + '.sl4'

                    current_all_vars_output_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label + '_all_vars_raw.csv')
                    current_select_vars_output_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label + '_select_vars_raw.csv')
                    required_paths = [current_all_vars_output_path, current_select_vars_output_path]
                    if not all([hb.path_exists(i) for i in required_paths]):
                        selected_vars = ['qgdp', 'pgdp']

                        gtap_invest_integration_functions.extract_raw_csv_from_sl4(gtap_sl4_path, current_all_vars_output_path)
                        gtap_invest_integration_functions.extract_raw_csv_from_sl4(gtap_sl4_path, current_select_vars_output_path, vars_to_extract=selected_vars)

        # Keep track of output dimensions to see if any scenarios have different number of outputs (which you may want to troubleshoot)
        run_dimensions = {}

        # ## THIS IS DATED NOW. Update with GTAP2 version.
        # # Loop through generated raw CSVs and extract more useful results.
        # for luh_scenario_label in p.luh_scenario_labels:
        #     run_dimensions[luh_scenario_label] = {}
        #     for scenario_year in p.scenario_years:
        #         run_dimensions[luh_scenario_label][scenario_year] = {}
        #         for policy_scenario_label in p.policy_scenario_labels:
        # 
        #             run_dimensions[luh_scenario_label][scenario_year][policy_scenario_label] = {}
        #             gtap_scenario_label = '2021_30_' + policy_scenario_label + '_noES'
        #             raw_csv_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label + '_all_vars_raw.csv')
        # 
        #             # HACK yet again to rename.
        #             if '2021_30_SR_RnD_20p_PESGC_30_noES' in gtap_scenario_label:
        #                 raw_csv_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label.replace('2021_30_SR_RnD_20p_PESGC_30_noES', '2021_30_SR_RnD_20p_PESGB_30_noES') + '_all_vars_raw.csv')
        # 
        #             hb.create_directories(os.path.join(p.cur_dir, 'solution_shapes'))
        #             solution_summary_path = os.path.join(p.cur_dir, 'solution_shapes', gtap_scenario_label + '_solution_variables_description.csv')
        #             if not hb.path_exists(solution_summary_path):
        #                 output_lines = []
        #                 var_names = []
        #                 var_descriptions = []
        #                 shapes = []
        #                 axes = []
        #                 with open(raw_csv_path, 'r') as fp:
        #                     for line in fp:
        #                         if '! Variable' in line:
        #                             if ' of size ' not in line:
        #                                 var_name = line.split(' ')[3]
        #                                 var_desc = line.split('#')[1]
        # 
        #                                 var_names.append(var_name)
        #                                 var_descriptions.append(var_desc)
        #                             else:
        #                                 if '(' in line:
        #                                     current_axes = line.split('(')[1].split(')')[0]
        #                                     shape = line.split(' size ')[1]
        #                                     shape = shape.replace('\n', '')
        #                                 else:
        #                                     current_axes = 'singular'
        #                                     shape = 1
        # 
        #                                 axes.append(current_axes)
        #                                 shapes.append(shape)
        # 
        #                     solution_variables_description_df = pd.DataFrame(data={'var_name': var_names, 'var_desc': var_descriptions, 'axes': axes, 'shape': shapes})
        #                     solution_variables_description_df['axes_with_shape'] = solution_variables_description_df['axes'].astype(str) + '_' + solution_variables_description_df['shape'].astype(str)
        #                     L.info('Generated solution variables description dataframe and saved it to ' + str(solution_summary_path))
        #                     L.info(solution_variables_description_df)
        # 
        #                     # Note, this method of only calculating it once assumes every solution has all the same variables for all scenarios. That seems wrong.
        #                     solution_variables_description_df.to_csv(solution_summary_path)
        #             else:
        #                 solution_variables_description_df = pd.read_csv(solution_summary_path)
        # 
        #             unique_axes_with_shape = hb.enumerate_array_as_odict(np.asarray(solution_variables_description_df['axes'], dtype='str'))
        #             for shape, count in unique_axes_with_shape.items():
        #                 run_dimensions[luh_scenario_label][scenario_year][policy_scenario_label][shape] = count
        # 
        # # validate that all are the same shape (optional)
        # assert_all_solutions_identical_shapes = True
        # if assert_all_solutions_identical_shapes:
        #     initial_v3 = None
        #     for k, v in run_dimensions.items():
        #         for k2, v2 in v.items():
        #             for k3, v3 in v2.items():
        #                 if initial_v3 is None:
        #                     initial_v3 = v3
        #                 else:
        #                     if v3 != initial_v3:
        #                         L.critical('Solution files are not the same output dimensions!!! RUN!!!!')
        #                         L.info(initial_v3)
        #                         L.info(v3)
        #                         assert NameError('Solution files are not the same output dimensions!!! RUN!!!!')
        # 
        # Loop through generated raw CSVs and make well-formated tables
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    gtap_scenario_label = '2021_30_' + policy_scenario_label + '_noES'

                    # Hack yet again to rename
                    if '2021_30_SR_RnD_20p_PESGC_30_noES' in gtap_scenario_label:
                        gtap_scenario_label = gtap_scenario_label.replace('2021_30_SR_RnD_20p_PESGC_30_noES', '2021_30_SR_RnD_20p_PESGB_30_noES')

                    raw_csv_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label + '_all_vars_raw.csv')

                    write_dir = os.path.join(p.cur_dir, 'output_tables', luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label)
                    hb.create_directories(write_dir)

                    gtap_invest_integration_functions.extract_vertical_csvs_from_multidimensional_sl4_csv(raw_csv_path, write_dir, gtap_scenario_label)

        # Merge the singular variables across scenarios
        df = None
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:

                for policy_scenario_label in p.policy_scenario_labels:
                    gtap_scenario_label = '2021_30_' + policy_scenario_label + '_noES'

                    # Hack yet again to rename
                    if '2021_30_SR_RnD_20p_PESGC_30_noES' in gtap_scenario_label:
                        gtap_scenario_label = gtap_scenario_label.replace('2021_30_SR_RnD_20p_PESGC_30_noES', '2021_30_SR_RnD_20p_PESGB_30_noES')

                    scenario_label = luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                    singular_csv_path = os.path.join(p.cur_dir, 'output_tables', scenario_label, gtap_scenario_label + '_singular_vars.csv')

                    current_df = pd.read_csv(singular_csv_path)
                    if df is None:
                        df = current_df
                        df.rename(columns={'value': scenario_label}, inplace=True)  # GET BACK SUMMARY
                    else:
                        current_df[scenario_label] = current_df['value']
                        df = df.merge(current_df[['var_name', scenario_label]], left_on='var_name', right_on='var_name')
        df.to_csv(os.path.join(p.cur_dir, 'global_results_across_scenarios.csv'), index=False)

        # Merge single variables against region-scenarios
        df = None

        for var in ['qgdp', 'pgdp']:
            for luh_scenario_label in p.luh_scenario_labels:
                for scenario_year in p.scenario_years:

                    for policy_scenario_label in p.policy_scenario_labels:
                        gtap_scenario_label = '2021_30_' + policy_scenario_label + '_noES'

                        # Hack yet again to rename
                        if '2021_30_SR_RnD_20p_PESGC_30_noES' in gtap_scenario_label:
                            gtap_scenario_label = gtap_scenario_label.replace('2021_30_SR_RnD_20p_PESGC_30_noES', '2021_30_SR_RnD_20p_PESGB_30_noES')

                        scenario_label = luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        singular_csv_path = os.path.join(p.cur_dir, 'output_tables', scenario_label, gtap_scenario_label + '_one_dim_vars.csv')

                        current_df = pd.read_csv(singular_csv_path)

                        if df is None:
                            current_df = current_df[current_df['var_name'] == var]
                            current_df[scenario_label] = current_df['value']
                            df = current_df[['dim_value', scenario_label]]
                            # df.rename(columns={'value': scenario_label}, inplace=True) #GET BACK SUMMARY
                        else:
                            current_df[scenario_label] = current_df['value']
                            current_df = current_df[current_df['var_name'] == var]
                            # current_df = current_df[['dim_value', 'value']]
                            df = df.merge(current_df[['dim_value', scenario_label]], left_on='dim_value', right_on='dim_value')
            df.to_csv(os.path.join(p.cur_dir, var + '_across_scenarios_and_regions.csv'), index=False)


def gtap_results_joined_with_luh_change(p):
    """Join the GTAP genereated endogenous land-use change projections per 37 region and 18 AEZs with the equivilent aggregation of LUH2 data.
    Then, shift the LUH2 results so that they sum up to the GTAP projeciotn but keep the spatial distribution of LUH2 at 15min."""
    p.gtap_with_luh_change_path = os.path.join(p.cur_dir, 'gtap_with_luh_change.xlsx')
    p.gtap1_projections_path = os.path.join(p.cur_dir, 'gtap1_projections.xlsx')
    p.gtap_luh_available_land_path = os.path.join(p.cur_dir, "gtap_luh_available_land.gpkg")
    p.full_projection_results_vector_path = os.path.join(p.cur_dir, 'full_projection_results.gpkg')
    p.full_projection_results_spreadsheet_path = os.path.join(p.cur_dir, 'full_projection_results.xlsx')


    if p.run_this:
        luh_df = pd.read_excel(p.luh_projections_by_region_aez_path, index_col=0)
        # Input AEZ-Reg landuse baseline data
        p.gtap_baseline_2014_lulc_path = os.path.join(p.input_dir, "2014_21_baseline_Y2014_REG_AEZ.csv")
        if not hb.path_exists(p.gtap_baseline_2014_lulc_path):
            p.gtap_baseline_2014_lulc_path = os.path.join(p.model_base_data_dir, 'gtap_inputs', '2014_21_baseline_Y2014_REG_AEZ.csv')


        p.gtap_baseline_2021_lulc_path = os.path.join(p.input_dir, "2014_21_baseline_Y2021_REG_AEZ.csv")
        if not hb.path_exists(p.gtap_baseline_2021_lulc_path):
            p.gtap_baseline_2021_lulc_path = os.path.join(p.model_base_data_dir, 'gtap_inputs', '2014_21_baseline_Y2021_REG_AEZ.csv')

        gtap_baseline_2014_lulc_df = pd.read_csv(p.gtap_baseline_2014_lulc_path)


        # LEARNING POINT: Using
        # gtap_baseline_2014_lulc_df.loc[gtap_baseline_2014_lulc_df['LAND_COVER'] == 'CROPLAND', 'LAND_COVER'] = 'cropland'
        # is greatly preferred over
        # gtap_baseline_2014_lulc_df['LAND_COVER'][gtap_baseline_2014_lulc_df['LAND_COVER'] == 'CROPLAND'] = 'cropland'
        # as the later can create ambiguious assignment.

        # Rename some of the old-stye labels from the AEZ database to match how the newer results come in from gtap.
        gtap_baseline_2014_lulc_df.loc[gtap_baseline_2014_lulc_df['LAND_COVER'] == 'CROPLAND', 'LAND_COVER'] = 'cropland'
        gtap_baseline_2014_lulc_df.loc[gtap_baseline_2014_lulc_df['LAND_COVER'] == 'ruminant', 'LAND_COVER'] = 'grassland'
        gtap_baseline_2014_lulc_df.loc[gtap_baseline_2014_lulc_df['LAND_COVER'] == 'forestsec', 'LAND_COVER'] = 'forest'
        gtap_baseline_2014_lulc_df.loc[gtap_baseline_2014_lulc_df['LAND_COVER'] == 'UNMNGLAND', 'LAND_COVER'] = 'natural'
        gtap_baseline_2014_lulc_df.set_index(['REG', 'AEZ_COMM', 'LAND_COVER'], inplace=True)
        gtap_baseline_2014_lulc_df = gtap_baseline_2014_lulc_df.rename(columns={'Value': 'gtap1_baseline_2014_ha'})

        gtap_baseline_2021_lulc_df = pd.read_csv(p.gtap_baseline_2021_lulc_path)

        gtap_baseline_2021_lulc_df.loc[gtap_baseline_2021_lulc_df['LAND_COVER'] == 'CROPLAND', 'LAND_COVER'] = 'cropland'
        gtap_baseline_2021_lulc_df.loc[gtap_baseline_2021_lulc_df['LAND_COVER'] == 'ruminant', 'LAND_COVER'] = 'grassland'
        gtap_baseline_2021_lulc_df.loc[gtap_baseline_2021_lulc_df['LAND_COVER'] == 'forestsec', 'LAND_COVER'] = 'forest'
        gtap_baseline_2021_lulc_df.loc[gtap_baseline_2021_lulc_df['LAND_COVER'] == 'UNMNGLAND', 'LAND_COVER'] = 'natural'
        gtap_baseline_2021_lulc_df.set_index(['REG', 'AEZ_COMM', 'LAND_COVER'], inplace=True)
        gtap_baseline_2021_lulc_df = gtap_baseline_2021_lulc_df.rename(columns={'Value': 'gtap1_baseline_2021_ha'})

        # Merge gtap baselines 2014 and 2021 into one dataframe
        gtap1_projections_df = gtap_baseline_2014_lulc_df.merge(gtap_baseline_2021_lulc_df, left_index=True, right_index=True)
        gtap1_projections_df.to_excel(p.gtap1_projections_path)

        # # Map the PROJETION files from gtap to a dictionary
        # p.gtap1_lulc_scenario_projections = {}
        # for luh_scenario_label in p.luh_scenario_labels:
        #     p.gtap1_lulc_scenario_projections[luh_scenario_label] = {}
        #     for scenario_year in p.scenario_years:
        #         p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year] = {}
        #
        #         p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['bau'] = os.path.join(p.input_dir, "2021_30_baseline_REG_AEZ.csv")
        #         p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['subs'] = os.path.join(p.input_dir, "2021_30_IOsubsidy_one_shot_REG_AEZ.csv")
        #         p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['rd'] = os.path.join(p.input_dir, "2021_30_RnDP_2pa_REG_AEZ.csv")
        #         p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['pes'] = os.path.join(p.input_dir, "2021_30_PES_REG_AEZ.csv")
        #         p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['zdc'] = os.path.join(p.input_dir, "2021_30_zdc_REG_AEZ_BEFORE_GTAP1.csv")
        #         # p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['PESGC'] = os.path.join(p.input_dir, "2020-12-28_GTAP_AEZ_LCOVER_ha.csv")
        #         p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['PESLC'] = os.path.join(p.input_dir, "2020-12-28_GTAP_AEZ_LCOVER_ha.csv")
        #         p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['RnD'] = os.path.join(p.input_dir, "2020-12-28_GTAP_AEZ_LCOVER_ha.csv")
        #         p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['SR_Land'] = os.path.join(p.input_dir, "2020-12-28_GTAP_AEZ_LCOVER_ha.csv")
        #         p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['SR_Land_PESGC'] = os.path.join(p.input_dir, "2020-12-28_GTAP_AEZ_LCOVER_ha.csv")
        #         p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['SR_PESLC'] = os.path.join(p.input_dir, "2020-12-28_GTAP_AEZ_LCOVER_ha.csv")
        #         p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['SR_RnD'] = os.path.join(p.input_dir, "2020-12-28_GTAP_AEZ_LCOVER_ha.csv")
        #         p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['SR_RnD_PESGC'] = os.path.join(p.input_dir, "2020-12-28_GTAP_AEZ_LCOVER_ha.csv")
        #         p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['SR_RnD_PESLC'] = os.path.join(p.input_dir, "2020-12-28_GTAP_AEZ_LCOVER_ha.csv")
        #

        #


        # Join gtap1_AEZ_LCOVER change with spatial results.


        # Rename gtap input columns
        df_combined_scenarios_input = None
        p.gtap1_lulc_scenario_projection_dfs = {}
        for luh_scenario_label in p.luh_scenario_labels:
            p.gtap1_lulc_scenario_projection_dfs[luh_scenario_label] = {}
            for scenario_year in p.scenario_years:
                p.gtap1_lulc_scenario_projection_dfs[luh_scenario_label][scenario_year] = {}

                for policy_scenario_label in p.policy_scenario_labels:
                # for policy_scenario_label, scenario_path in p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year].items():
                #     scenario_path = p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year][policy_scenario_label]
                    scenario_path = p.gtap1_land_use_change_path

                    # if not hb.path_exists(scenario_path):
                    #     scenario_path = os.path.join(p.model_base_data_dir, 'gtap_inputs', 'example_GTAP_AEZ_LCOVER_ha.csv')

                    L.info('Reading ' + scenario_path, policy_scenario_label)

                    if policy_scenario_label not in ['zdc']:
                        if df_combined_scenarios_input is None:
                            df_combined_scenarios_input = pd.read_csv(scenario_path)

                        # HACK to fix typo in Uris' latest naming of SR_RnD_PESGC_30 vs SR_RnD_PESGB_30
                        if policy_scenario_label == 'SR_RnD_PESGC_30':
                            modded_policy_scenario_label = 'SR_RnD_PESGB_30'
                            src_path = os.path.join(p.cur_dir, p.gtap_aez_invest_release_string, 'work', gtap_scenario_label + '.sl4')
                            dst_path = os.path.join(p.cur_dir, p.gtap_aez_invest_release_string, 'work', gtap_scenario_label + '.sl4')

                            hb.copy_shutil_flex(src_path, dst_path)
                        elif policy_scenario_label == 'SR_RnD_20p_PESGC_30':
                            modded_policy_scenario_label = 'SR_RnD_20p_PESGB_30'
                        else:
                            modded_policy_scenario_label = policy_scenario_label


                        full_modded_policy_scenario_label = '2021_30_' + modded_policy_scenario_label + '_noES'

                        if full_modded_policy_scenario_label not in df_combined_scenarios_input['SCENARIO'].unique():
                            raise NameError('Could not find scenario ' + full_modded_policy_scenario_label + ' in csv.')
                        df = df_combined_scenarios_input.loc[df_combined_scenarios_input['SCENARIO'] == full_modded_policy_scenario_label]

                        df.loc[df['VARNAME'] == 'Cropland Cover', 'VARNAME'] = 'cropland'
                        df.loc[df['VARNAME'] == 'Managed Forest Cover', 'VARNAME'] = 'forest'
                        df.loc[df['VARNAME'] == 'Pasture Cover', 'VARNAME'] = 'grassland'
                        df.loc[df['VARNAME'] == 'Other Land', 'VARNAME'] = 'natural'
                        df = df.rename({'VARNAME': 'LAND_COVER'}, axis=1)


                    elif policy_scenario_label in ['zdc']: # Then it is Floris' input.
                        L.info('Converting FLORIS inputs to GTAP/SEALS style. NOTE!!! This is just a proof of concept to ensure running works as we are still waiting for Uris endogenization')
                        df3 = pd.read_csv(p.gtap1_land_use_change_path)
                        # df3 = pd.read_csv(p.gtap1_lulc_scenario_projections[luh_scenario_label][scenario_year]['bau'])

                        df2 = gtap1_projections_df.copy()
                        # df2 = df.copy()
                        df_input = pd.read_csv(scenario_path)

                        df2 = df2.reset_index() # REQUIRED because otherwize AEZ_COMM is not found (because it's an index)
                        # Change AEZ to AZ so that there is an AZREG column.
                        df2['to_merge'] = df2['AEZ_COMM'].map(str) + '-' + df2['REG'].map(str)
                        df2['to_merge'] = df2['to_merge'].str.replace('AEZ', 'AZ')

                        df_input['to_merge'] = df_input['AZREG']
                        df_input = df_input[['to_merge', 'AZREG', 'gtap37gtap', 'oil_crop_suit_area_km2', 'oil_crop_suit_area_ZDC_km2']]

                        df2 = df2.merge(df_input, 'outer', on='to_merge')

                        # Rename the BAU column to the zdc scenario so that it is what we keep.
                        df2['gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_ha'] = df2['gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + 'bau' + '_ha']
                        # df2 = df2.rename(columns={'gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + 'bau' + '_ha': 'gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_ha'})


                        value_key = 'gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_ha'

                        def transform_row(r):
                            if r['LAND_COVER'] == 'forest':
                                r[value_key] = r[value_key] + (r['oil_crop_suit_area_km2'] * 100 - r['oil_crop_suit_area_ZDC_km2'] * 100)
                            elif r['LAND_COVER'] == 'cropland':
                                r[value_key] = r[value_key] - (r['oil_crop_suit_area_km2'] * 100 - r['oil_crop_suit_area_ZDC_km2'] * 100)
                            else:
                                r[value_key] = r[value_key]
                            return r

                        df2 = df2.apply(transform_row, axis=1)
                        df = df2
                        # df2[value_key].apply(lambda x: x[value_key] - (x['oil_crop_suit_area_km2'] * 100 - x['oil_crop_suit_area_ZDC_km2'] * 100) if x['LAND_COVER'] == 'forest' else x[value_key])
                        # a = df2.loc[df2['LAND_COVER'] == 'forest']['gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_ha']

                    df.set_index(['REG', 'AEZ_COMM', 'LAND_COVER'], inplace=True)
                    full_scenario_label = 'gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_ha'
                    df = df.rename(columns={'Value': full_scenario_label})
                    df.to_excel(os.path.join(p.cur_dir, policy_scenario_label + '_output.xlsx'))
                    gtap1_projections_df = gtap1_projections_df.merge(df[full_scenario_label], left_index=True, right_index=True)

                    # L.info(policy_scenario_label, '\n', gtap1_projections_df)
        # Move the land-use type from an column value to a column header. This creates a multi-indexed column header.
        gtap1_projections_collapsed_df = pd.pivot_table(gtap1_projections_df, index=['REG', 'AEZ_COMM'], columns=['LAND_COVER'])
        # LEARNING POINT: Pivot table created a multi-level column, which I wanted to collapse for excel.
        # To do this, I used the columns.map(lambda) noting that x is a length 2 list where x[0[ is the top layer in the
        # multi level index and x[1] is the lower.
        # Collapse the multi-index column into a single index column with useful names.
        def joiner(x):

            if 'baseline' not in x[0]:
                joined = '_'.join(x[0].split('_')[:-1]) + '_' + x[1] + '_ha'
            else:
                joined = x[0].split('_')[0] + '_' + x[0].split('_')[2] + '_' + x[0].split('_')[1] + '_' + x[1] + '_ha'
            return joined
        # def joiner(x):
        #     if 'baseline' not in x[0]:
        #         joined = x[0].split('_')[0] + '_' + x[0].split('_')[1] + '_' + x[0].split('_')[2] + '_' + x[0].split('_')[3] + '_' + x[0].split('_')[4] + '_' + x[1] + '_ha'
        #     else:
        #         joined = x[0].split('_')[0] + '_' + x[0].split('_')[2] + '_' + x[0].split('_')[1] + '_' + x[1] + '_ha'
        #     return joined
        gtap1_projections_collapsed_df.columns = gtap1_projections_collapsed_df.columns.map(joiner)

        ## If we want to reinclude ssp rcp this is how:
        # gtap1_projections_collapsed_df.columns = gtap1_projections_collapsed_df.columns.map(
        #     lambda x: x[0].split('_')[0] + '_' + x[0].split('_')[1] + '_' + x[0].split('_')[2] + '_' + x[0].split('_')[3] + '_' + x[0].split('_')[4] + '_' + x[1] + '_ha')
        gtap1_projections_collapsed_df = gtap1_projections_collapsed_df.reset_index()
        # Save a preliminary xls for error checking on merge
        gtap1_projections_collapsed_df.to_excel(p.gtap1_projections_path)

        # Add in difference from 2021
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    for class_label in p.class_labels_that_differ_between_ssp_and_gtap:
                        baseline_col = 'gtap1_' + str(p.base_year) + '_baseline_' + class_label + '_ha'
                        policy_baseline_col = 'gtap1_' + str(p.policy_base_year) + '_baseline_' + class_label + '_ha'
                        scenario_col = 'gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_' + class_label + '_ha'
                        baseline_change_col = 'gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_minus_' + str(p.base_year) + '_' + policy_scenario_label + '_' + class_label + '_ha_change'
                        policy_baseline_change_col = 'gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_minus_' + str(p.policy_base_year) + '_' + policy_scenario_label + '_' + class_label + '_ha_change'
                        gtap1_projections_collapsed_df = gtap1_projections_collapsed_df.fillna(0)

                        gtap1_projections_collapsed_df[baseline_change_col] = gtap1_projections_collapsed_df[scenario_col] - gtap1_projections_collapsed_df[baseline_col]
                        gtap1_projections_collapsed_df[policy_baseline_change_col] = gtap1_projections_collapsed_df[scenario_col] - gtap1_projections_collapsed_df[policy_baseline_col]

        # Change AEZ to AZ so that there is an AZREG column.
        gtap1_projections_collapsed_df['merge_name'] = gtap1_projections_collapsed_df['AEZ_COMM'].map(str) + '_' + gtap1_projections_collapsed_df['REG'].map(str)
        gtap1_projections_collapsed_df['merge_name'] = gtap1_projections_collapsed_df['merge_name'].str.replace('AEZ', 'AZ')

        # Add the AZREG column to the luh data
        luh_df['merge_name'] = 'AZ' + luh_df['aez_pyramid_id'].map(str) + '_' + luh_df['gtap37v10_pyramid_name'].map(str)

        # Rename luh columns
        new_columns = []

        rename_dict = {}
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    for base_year in p.base_years:
                        for class_label in p.class_labels_that_differ_between_ssp_and_gtap:
                            old_label = class_label + '_baseline_' + str(base_year)
                            new_label = 'luh_baseline_' + str(base_year) + '_' + class_label + '_ha'
                            rename_dict[old_label] = new_label

                    for scenario_year in p.scenario_years:
                        for class_label in p.class_labels_that_differ_between_ssp_and_gtap:
                            old_label = class_label + '_' + luh_scenario_label + '_' + str(scenario_year)
                            new_label = 'luh_' + luh_scenario_label + '_' + str(scenario_year) + '_' + class_label + '_ha'
                            rename_dict[old_label] = new_label

                            old_label = class_label + '_' + luh_scenario_label + '_' + str(scenario_year) + '_prop_change'
                            new_label = 'luh_' + luh_scenario_label + '_' + str(scenario_year) + '_' + class_label + '_prop_change'
                            rename_dict[old_label] = new_label


        # for i in luh_df.columns:
        #     if '2030' in i and 'prop' not in i:
        #         to_append = 'luh_' + i.split('_')[1] + '_' + i.split('_')[2] + '_' + i.split('_')[3] + '_' + i.split('_')[0] + '_ha'
        #         new_columns.append(to_append)
        #     elif '2014' in i or '2021' in i and 'prop' not in i:
        #         to_append = 'luh_' + i.split('_')[1] + '_' + i.split('_')[2] + '_' + i.split('_')[0]  + '_ha'
        #         new_columns.append(to_append)
        #     elif 'prop_change' in i:
        #         to_append = 'luh_' + i.split('_')[1] + '_' + i.split('_')[2] + '_' + i.split('_')[0]  + '_prop_change'
        #         new_columns.append(to_append)
        #     else:
        #         new_columns.append(i)
        # luh_df.columns = new_columns
        luh_df = luh_df.rename(columns=rename_dict)

        # Export luh projections for error checking.
        p.luh_projections_spreadsheet_path = os.path.join(p.cur_dir, 'luh_projections.xlsx')
        luh_df.to_excel(p.luh_projections_spreadsheet_path)

        # Merge luh summed by AEZ with gtap projections
        gtap_with_luh_df = gtap1_projections_collapsed_df.merge(luh_df, how='outer', left_on='merge_name', right_on='merge_name')
        # Calculate the dif_luh_change for each scenario future year. This will be what we shift by.
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    for class_label in p.class_labels_that_differ_between_ssp_and_gtap:
                        gtap_with_luh_df['gtap1_' + luh_scenario_label + '_' +str(scenario_year)+'_'+policy_scenario_label+'_'+class_label+'_dif_luh'] \
                            = gtap_with_luh_df['gtap1_' + luh_scenario_label + '_' +str(scenario_year)+'_'+policy_scenario_label+'_' + class_label + '_ha'] \
                              - gtap_with_luh_df['luh_' + luh_scenario_label + '_' +str(scenario_year)+'_' + class_label + '_ha']
                        gtap_with_luh_df['luh_' + luh_scenario_label + '_' +str(scenario_year)+'_' + class_label + '_ha_change'] = gtap_with_luh_df['luh_rcp45_ssp2_'+str(scenario_year)+'_' + class_label + '_ha'] - gtap_with_luh_df['luh_baseline_2021_' + class_label + '_ha']
                        gtap_with_luh_df['gtap1_' + luh_scenario_label + '_' +str(scenario_year)+'_'+policy_scenario_label+'_'+class_label+'_dif_luh_change'] \
                            = (gtap_with_luh_df['gtap1_' + luh_scenario_label + '_'  + str(scenario_year) + '_minus_' + str(p.policy_base_year) + '_' + policy_scenario_label + '_' + class_label + '_ha_change']) \
                              - (gtap_with_luh_df['luh_' + luh_scenario_label + '_' +str(scenario_year)+'_' + class_label + '_ha'] - gtap_with_luh_df['luh_baseline_2021_' + class_label + '_ha'])


        gtap_with_luh_df = gtap_with_luh_df.fillna(0)
        gtap_with_luh_df.to_excel(p.gtap_with_luh_change_path)

        # TEMPORARILY I do not use available land calculations, but this would greatly improve predictions of where the expansion could happen.
        p.final_available_land_vector_path = hb.get_existing_path_from_nested_sources(p.final_available_land_vector_path, p, 1)
        available_land_df = gpd.read_file(p.final_available_land_vector_path)

        # VERSION INCOMPATIBILITY: I didn't want to regen available_land_df simply so that it had the correct naming of pyramid_ordered_id being pyramid_id, thus i manually changed it here.
        available_land_df = available_land_df.drop(columns='pyramid_id')
        available_land_df['AZREG'] = 'AZ' + available_land_df['aez_pyramid_id'].map(str) + '_' + available_land_df['gtap37v10_pyramid_name']
        available_land_df = available_land_df.rename(columns={'pyramid_ordered_id': 'pyramid_id'})

        gtap_with_luh_df['AZREG'] = gtap_with_luh_df['AEZ_COMM'].str.replace('AEZ', 'AZ') + '_' +  gtap_with_luh_df['REG']

        merged_df = available_land_df.merge(gtap_with_luh_df[[i for i in list(gtap_with_luh_df.columns) if i not in list(available_land_df.columns)] + ['AZREG']], on='AZREG', how='outer')
        merged_df.to_file(p.gtap_luh_available_land_path, driver='GPKG')

        # Identify correct LUH rasters to shift from base year.
        baseline_ha_total_paths = {}
        baseline_ha_total_paths['urban'] = os.path.join(p.luh2_as_seals7_proportion_dir, 'baseline', str(p.policy_base_year), 'urban.tif')
        baseline_ha_total_paths['cropland'] = os.path.join(p.luh2_as_seals7_proportion_dir, 'baseline', str(p.policy_base_year), 'cropland.tif')
        baseline_ha_total_paths['grassland'] = os.path.join(p.luh2_as_seals7_proportion_dir, 'baseline', str(p.policy_base_year), 'grassland.tif')
        baseline_ha_total_paths['forest'] = os.path.join(p.luh2_as_seals7_proportion_dir, 'baseline', str(p.policy_base_year), 'forest.tif')
        baseline_ha_total_paths['nonforestnatural'] = os.path.join(p.luh2_as_seals7_proportion_dir, 'baseline', str(p.policy_base_year), 'nonforestnatural.tif')

        # Identify correct LUH rasters to shift from policy base year.
        policy_baseline_ha_total_paths = {}
        policy_baseline_ha_total_paths['urban'] = os.path.join(p.luh2_as_seals7_proportion_dir, 'baseline', str(p.policy_base_year), 'urban.tif')
        policy_baseline_ha_total_paths['cropland'] = os.path.join(p.luh2_as_seals7_proportion_dir, 'baseline', str(p.policy_base_year), 'cropland.tif')
        policy_baseline_ha_total_paths['grassland'] = os.path.join(p.luh2_as_seals7_proportion_dir, 'baseline', str(p.policy_base_year), 'grassland.tif')
        policy_baseline_ha_total_paths['forest'] = os.path.join(p.luh2_as_seals7_proportion_dir, 'baseline', str(p.policy_base_year), 'forest.tif')
        policy_baseline_ha_total_paths['nonforestnatural'] = os.path.join(p.luh2_as_seals7_proportion_dir, 'baseline', str(p.policy_base_year), 'nonforestnatural.tif')

        p.luh_scenario_ha_total_paths = {}
        for luh_scenario_label in p.luh_scenario_labels:
            p.luh_scenario_ha_total_paths[luh_scenario_label] = {}
            for scenario_year in p.scenario_years:
                p.luh_scenario_ha_total_paths[luh_scenario_label][scenario_year] = {}
                p.luh_scenario_ha_total_paths[luh_scenario_label][scenario_year]['urban'] = os.path.join(p.luh2_as_seals7_proportion_dir, luh_scenario_label, str(scenario_year), "urban.tif")
                p.luh_scenario_ha_total_paths[luh_scenario_label][scenario_year]['cropland'] = os.path.join(p.luh2_as_seals7_proportion_dir, luh_scenario_label, str(scenario_year), "cropland.tif")
                p.luh_scenario_ha_total_paths[luh_scenario_label][scenario_year]['grassland'] = os.path.join(p.luh2_as_seals7_proportion_dir, luh_scenario_label, str(scenario_year), "grassland.tif")
                p.luh_scenario_ha_total_paths[luh_scenario_label][scenario_year]['forest'] = os.path.join(p.luh2_as_seals7_proportion_dir, luh_scenario_label, str(scenario_year), "forest.tif")
                p.luh_scenario_ha_total_paths[luh_scenario_label][scenario_year]['nonforestnatural'] = os.path.join(p.luh2_as_seals7_proportion_dir, luh_scenario_label, str(scenario_year), "nonforestnatural.tif")


        # Create rastetr of luh gridded difference. This is what we will shift up or down until it matches gtap.
        p.luh_scenario_ha_change_paths = {}
        for luh_scenario_label in p.luh_scenario_labels:
            p.luh_scenario_ha_change_paths[luh_scenario_label] = {}
            for scenario_year in p.scenario_years:
                p.luh_scenario_ha_change_paths[luh_scenario_label][scenario_year] = {}
                for class_label in p.class_labels:
                    a_future = hb.as_array(p.luh_scenario_ha_total_paths[luh_scenario_label][scenario_year][class_label])
                    a_baseline = hb.as_array(policy_baseline_ha_total_paths[class_label])
                    a_future = np.where(a_future == -9999., 0, a_future)
                    a_baseline = np.where(a_baseline == -9999., 0, a_baseline)

                    change = a_future - a_baseline

                    # NOTE! This is a kinda awkward duplication of the seals7_difference_from_base_year, but that wasn't iterated over
                    # the different scenarios/years/etc
                    new_path = os.path.join(p.cur_dir, luh_scenario_label, str(scenario_year), 'luh_' + class_label + '_' + str(scenario_year) + '_' + str(p.policy_base_year) + '_ha_change.tif')
                    try:
                        hb.create_directories(os.path.join(p.cur_dir, luh_scenario_label, str(scenario_year)))
                    except:
                        pass
                    p.luh_scenario_ha_change_paths[luh_scenario_label][scenario_year][class_label] = new_path
                    hb.save_array_as_geotiff(change, new_path, policy_baseline_ha_total_paths['urban'])

        p.gtap1_ha_total_paths = {}

        for luh_scenario_label in p.luh_scenario_labels:
            p.gtap1_ha_total_paths[luh_scenario_label] = {}
            for scenario_year in p.scenario_years:
                p.gtap1_ha_total_paths[luh_scenario_label][scenario_year] = {}
                for policy_scenario_label in p.policy_scenario_labels:
                    p.gtap1_ha_total_paths[luh_scenario_label][scenario_year][policy_scenario_label] = {}
                    for class_label in p.class_labels:
                        if class_label in p.class_labels_that_differ_between_ssp_and_gtap:
                            p.gtap1_ha_total_paths[luh_scenario_label][scenario_year][policy_scenario_label][class_label] = {}
                            luh_change = hb.as_array(p.luh_scenario_ha_change_paths[luh_scenario_label][scenario_year][class_label])

                            p.gtap37_aez18_zones_raster_15min_path = os.path.join(p.cur_dir, 'gtap37_aez18_zones_raster_15min.tif')
                            cells_per_zone = hb.unique_raster_values_count(p.gtap37_aez18_ids_15m_path) # NOTE, unique_raster_values_count returns a defaultdict which will fail if not as a defaultdict.

                            # Create a raster that takes the total shift amount and scales it down by all the cells present in the zone to get the per-cell shift.
                            rules = {
                                merged_df['pyramid_id'].iloc[j]:
                                    np.float32(merged_df['gtap1_'+luh_scenario_label+'_'+str(scenario_year)+'_'+policy_scenario_label+'_'+class_label+'_dif_luh_change'].iloc[j]) / cells_per_zone[merged_df['pyramid_id'].iloc[j]]
                                        for j in range(len(merged_df['gtap1_'+luh_scenario_label+'_'+str(scenario_year)+'_'+policy_scenario_label+'_'+class_label+'_dif_luh_change']))
                            }
                            # rules[-9999] = 0.0

                            rules[-9999.] = np.float64(0.0)
                            rules = {np.nan_to_num(k): np.nan_to_num(v) for k, v in rules.items()}

                            L.info('Reclassifying rules onto regions: ' + str(rules))


                            shift_path = os.path.join(p.cur_dir, luh_scenario_label, str(scenario_year), policy_scenario_label, str(class_label) + '_shift.tif')

                            if not hb.path_exists(shift_path):
                                input_array = hb.as_array(p.gtap37_aez18_ids_15m_path).astype(np.float32)

                                gp.reclassify_raster((p.gtap37_aez18_ids_15m_path, 1), rules, shift_path, 6, -9999, hb.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS_HB)
                                # hb.reclassify_flex(p.gtap37_aez18_ids_15m_path, rules, shift_path, output_data_type=6, verbose=False)

                                shifter = hb.as_array(shift_path)
                                shifter = np.where(shifter == -9999., 0, shifter)

                                # Shift the luh spatial distribution around so that it matches the GTAP total.
                                shifted_change = luh_change + shifter
                                shifted_change = np.where(shifted_change == -9999, 0, shifted_change)
                                shifted_change = np.where(input_array > 0, shifted_change, 0)
                                gtap1_class_total_15min_path = os.path.join(p.cur_dir, luh_scenario_label, str(scenario_year), policy_scenario_label, 'gtap1_' + str(class_label) + '_ha_change_15min.tif')

                                # Keep track of the tree of scenario paths for later use in seals.
                                p.gtap1_ha_total_paths[luh_scenario_label][scenario_year][policy_scenario_label][class_label] = gtap1_class_total_15min_path
                                hb.save_array_as_geotiff(shifted_change, gtap1_class_total_15min_path, p.luh_scenario_ha_total_paths[p.luh_scenario_labels[0]][scenario_year][class_label])

                                # To validate that it actually does sum up, take the zonal stats of the new rasters
                                df = hb.zonal_statistics_flex(gtap1_class_total_15min_path, p.gtap37_aez18_path, p.gtap37_aez18_ids_15m_path, id_column_label='pyramid_id',
                                                         zones_ndv=-9999, values_ndv=-9999., zones_raster_data_type=5,
                                                         stats_to_retrieve='sums',)
                                # LEARNING POINT, the following two were not identical because I only wrote the new values and instead got junk memory for ndvs.
                                # df = hb.zonal_statistics_flex(gtap1_class_total_15min_path, p.gtap37_aez18_path, p.gtap37_aez18_zones_raster_15min_path, id_column_label='pyramid_id',
                                #                          zones_ndv=-9999, values_ndv=-9999., zones_raster_data_type=5,
                                #                          stats_to_retrieve='sums',)
                                df = df.rename(columns={'sums': 'gtap1_' +luh_scenario_label+'_'+ str(scenario_year) + '_' + policy_scenario_label + '_' + class_label + '_ha_change_spatialized_validation'})
                                merged_df = merged_df.merge(df, left_on='pyramid_id', right_index=True, how='outer')
                                merged_df = merged_df[merged_df['pyramid_id'].notna()]

        # merged_df['validate2'] = merged_df['gtap1_rcp45_ssp2_2030_bau_cropland_ha_change_spatialized_validation'] / merged_df['gtap1_rcp45_ssp2_2030_minus_2021_bau_cropland_ha_change']


        if not hb.path_exists(p.full_projection_results_spreadsheet_path):
            merged_df.rename({'GTAP': 'ISO3'}, axis=1, inplace=True)

            put_at_front = [
                'pyramid_id',
                'pyramid_ids_concatenated',
                'pyramid_ids_multiplied',
                'gtap37v10_pyramid_id',
                'aez_pyramid_id',
                'gtap37v10_pyramid_name',
                'ISO3',
                'AZREG',
                'AEZ_COMM',
                'merge_name',
                'admin',
                'GTAP226',
                'GTAP140v9a',
                'GTAPv9a',
                'GTAP141v9p',
                'GTAPv9p',
                'AreaCode',
                'names',
                'bb',
                'minx',
                'miny',
                'maxx',
                'maxy',
            ]


            merged_df = hb.dataframe_reorder_columns(merged_df, put_at_front)
            # ISO3 upper
            merged_df['ISO3'] = merged_df['ISO3'].str.upper()

            merged_df.rename(columns={'admin': 'gtap9_admin_id', 'names': 'gtap9_admin_name'}, inplace=True)
            merged_df.drop(['merge_name',
                            'arable_definition_0', 'ha_pch_0',
                            'arable_definition_1', 'ha_pch_1',
                            'arable_definition_2', 'ha_pch_2',
                            'arable_definition_3', 'ha_pch_3',
                            'arable_definition_4', 'ha_pch_4',
                            'arable_definition_5', 'ha_pch_5',
                            'arable_definition_6', 'ha_pch_6',
                            ], axis=1, inplace=True)

            merged_df.to_file(p.full_projection_results_vector_path, driver='GPKG')

            merged_df[[i for i in merged_df.columns if i != 'geometry']].to_excel(p.full_projection_results_spreadsheet_path)

            # TODOO IDEA, incorporate a validate function like this that determines whether it needs to rerun.
            validate_correct = False
            if validate_correct:
                # Test that in each country the sum of changes in the spatialized equals that of the total gtap1 by region.
                df = pd.read_excel(p.full_projection_results_spreadsheet_path)
                validation_array_path = os.path.join(p.cur_dir, p.luh_scenario_labels[0], str(p.scenario_years[0]), p.policy_scenario_labels[0], 'gtap1_cropland_ha_change_15min.tif')

                validation_array = hb.as_array(validation_array_path)
                a = df['gtap1_rcp45_ssp2_2030_bau_cropland_ha'] - df['gtap1_2021_baseline_cropland_ha']
                b = df['gtap1_rcp45_ssp2_2030_minus_2021_bau_cropland_ha_change']
                a_sum = np.sum(a)
                b_sum = np.sum(b)
                array_sum = np.sum(validation_array)

                if np.sum(a - b) / a_sum > 0.00001:
                    raise NameError('Difference bigger than floatingpoint error.')

def pes_policy_identification(p):
    # Create maps assocaited with FFN PES payments

    if p.run_this:
        args = {}
        args['input_folder'] = os.path.join(p.model_base_data_dir, 'pes_policy_identification_inputs')
        args['parameters_uri'] = os.path.join(args['input_folder'], 'parameters.csv')

        args['crop_statistics_folder'] = os.path.join(hb.BASE_DATA_DIR, 'crops/earthstat/crop_production/')
        args['soil_carbon_tons_per_cell_uri'] = os.path.join(args['input_folder'], 'soil_carbon_tons_per_cell.tif')
        args['potential_carbon_if_no_cultivation'] = os.path.join(args['input_folder'], 'potential_carbon_if_no_cultivation.tif')
        args['crop_carbon_tons_per_cell_uri'] = os.path.join(args['input_folder'], 'crop_carbon_tons_per_cell.tif')

        args['potential_natural_vegetation_carbon_tons_per_cell_uri'] = os.path.join(args['input_folder'], 'potential_carbon_if_no_cultivation.tif')
        args['match_uri'] = os.path.join(hb.BASE_DATA_DIR, 'pyramids/ha_per_cell_300sec.tif')
        args['produce_base_data_figs'] = False
        # args['n_method'] = 'ones'
        args['calculate_ffn_c_payments'] = True
        args['calculate_ffn_f_payments'] = True
        args['calculate_ffn_a_payments'] = True
        args['calculate_ffn_f_country_level_payments'] = True
        args['produce_base_mechanism_figs'] = False
        args['produce_fractional_value_plot'] = False
        args['produce_auction_mechaism_figs'] = False

        args['calorie_increase'] = 0.7
        args['assumed_intensification'] = .75,
        args['transition_function_alpha'] = 1.0
        args['transition_function_beta'] = 0.5
        args['solution_precision'] = 0.01
        args['min_extensification'] = 0.05
        args['max_extensification'] = 0.95

        args['social_cost_of_carbon'] = 171.
        args['asymmetric_scatter'] = .06
        args['npv_adjustment'] = 20.91744579
        args['n_method'] = 'profit'

        pes_scenario_definition = {}
        pes_scenario_definition['subs_budget'] = 471163000000
        pes_scenario_definition['subs_budget_half'] = 471163000000 / 2

        for pes_scenario_name, budget in pes_scenario_definition.items():
            args['run_dir'] = os.path.join(p.cur_dir, pes_scenario_name)
            args['run_folder'] = os.path.join(p.cur_dir, pes_scenario_name)
            args['workspace'] = os.path.join(p.cur_dir, pes_scenario_name)
            args['output_folder'] = os.path.join(p.cur_dir, pes_scenario_name)
            args['intermediate_folder'] = os.path.join(p.cur_dir, pes_scenario_name)

            args['conservation_budget'] = budget


            hb.create_directories(args['run_dir'])

            L.info('Running pes_policy_identificaiton wtih args', args)
            execute_args = args

            file_that_should_exist = os.path.join(p.cur_dir, pes_scenario_name, "payment_set_f_per_ha.tif")
            if not hb.path_exists(file_that_should_exist):
                pes_policy_identification_script.execute(execute_args)

def pes_policy_endogenous_land_shock(p):
    # FOR THIS TASK TO RUN:
    # You need to run pes_policy_identification.py first, which will generate the geotiffs used below.
    p.zone_ids_raster_path = os.path.join(p.cur_dir, 'GTAP37_AEZ18_ids_10sec.tif')
    p.zone_ids_raster_300sec_path = os.path.join(p.cur_dir, 'GTAP37_AEZ18_ids_300sec.tif')
    p.zone_ids_raster_900sec_path = os.path.join(p.cur_dir, 'GTAP37_AEZ18_ids_900sec.tif')

    pes_budget_scenario_to_use = 'subs_budget_half'

    # TODOO HACK, I beleive this wasnt properly done seperately per scenario, so confirm before reusing.
    p.pes_model_dir = os.path.join(p.pes_policy_identification_dir, pes_budget_scenario_to_use)

    p.desired_ha_conserved_path = os.path.join(p.pes_model_dir, "desired_ha_conserved.tif")

    p.pes_shockfile_parameters_path = os.path.join(p.cur_dir, 'pes_shockfile_parameters.csv')

    if p.run_this:
        # First for gloabl
        # p.payment_set_f_per_ha = "payment_set_f_per_ha.tif"

        p.payment_set_f_per_ha_path = os.path.join(p.pes_model_dir, "payment_set_f_per_ha.tif")
        csv_output_path = os.path.join(p.cur_dir, 'payment_set_f_per_ha.csv')
        vector_output_path = os.path.join(p.cur_dir, 'payment_set_f_per_ha.gpkg')

        p.ha_removed_from_production_path = os.path.join(p.cur_dir, 'ha_removed_from_production.tif')
        #PNAS ISSUE: this was implemented here as the args['conservation_budget'] = 471163000000 / 2. Need to clarify how this was calculated and decide which is the CORRECT scenario.
        if not hb.path_exists(csv_output_path):
            hb.zonal_statistics_flex(p.payment_set_f_per_ha_path, p.gtap37_aez18_path, p.zone_ids_raster_300sec_path, 'pyramid_id', 5, 6, -9999, -9999.,
                                     assert_projections_same=False,  csv_output_path=csv_output_path, vector_output_path=vector_output_path, stats_to_retrieve='sums_counts')


            payments = hb.as_array(p.payment_set_f_per_ha_path)
            ha_changed = hb.as_array(p.desired_ha_conserved_path)
            ha_removed = np.where((payments > 0) & (ha_changed > 0), ha_changed, 0)
            hb.save_array_as_geotiff(ha_removed, p.ha_removed_from_production_path, p.zone_ids_raster_300sec_path, ndv=-9999., data_type=6)

        csv_output_path = os.path.join(p.cur_dir, 'ha_removed_from_production.csv')
        if not hb.path_exists(csv_output_path):

            vector_output_path = os.path.join(p.cur_dir, 'ha_removed_from_production.gpkg')
            hb.zonal_statistics_flex(p.ha_removed_from_production_path, p.gtap37_aez18_path, p.zone_ids_raster_300sec_path, 'pyramid_id', 5, 6, -9999, -9999.,
                                     assert_projections_same=False,  csv_output_path=csv_output_path, vector_output_path=vector_output_path, stats_to_retrieve='sums_counts')

        p.expenditure_f_per_grid_path = os.path.join(p.pes_model_dir, r"expenditure_f_per_grid.tif")
        csv_output_path = os.path.join(p.cur_dir, 'expenditure_f_per_grid.csv')
        if not hb.path_exists(csv_output_path):

            vector_output_path = os.path.join(p.cur_dir, 'expenditure_f_per_grid.gpkg')
            hb.zonal_statistics_flex(p.expenditure_f_per_grid_path, p.gtap37_aez18_path, p.zone_ids_raster_300sec_path, 'pyramid_id', 5, 6, -9999, -9999.,
                                     assert_projections_same=False,  csv_output_path=csv_output_path, vector_output_path=vector_output_path, stats_to_retrieve='sums_counts')

        p.value_protected_per_grid_path =  os.path.join(p.pes_model_dir, "value_protected_given_budget_per_grid_f.tif")
        csv_output_path = os.path.join(p.cur_dir, 'value_protected_per_grid.csv')
        if not hb.path_exists(csv_output_path):

            vector_output_path = os.path.join(p.cur_dir, 'value_protected_per_grid.gpkg')
            hb.zonal_statistics_flex(p.value_protected_per_grid_path, p.gtap37_aez18_path, p.zone_ids_raster_300sec_path, 'pyramid_id', 5, 6, -9999, -9999.,
                                     assert_projections_same=False,  csv_output_path=csv_output_path, vector_output_path=vector_output_path, stats_to_retrieve='sums_counts')

        p.profit_forgone_f_per_ha_conserved_path = os.path.join(p.pes_model_dir, "value_protected_given_budget_per_grid_f.tif")
        csv_output_path = os.path.join(p.cur_dir, 'profit_forgone_f_per_ha_conserved.csv')
        if not hb.path_exists(csv_output_path):

            vector_output_path = os.path.join(p.cur_dir, 'profit_forgone_f_per_ha_conserved.gpkg')
            hb.zonal_statistics_flex(p.profit_forgone_f_per_ha_conserved_path, p.gtap37_aez18_path, p.zone_ids_raster_300sec_path, 'pyramid_id', 5, 6, -9999, -9999.,
                                     assert_projections_same=False,  csv_output_path=csv_output_path, vector_output_path=vector_output_path, stats_to_retrieve='sums_counts')

        if not hb.path_exists(p.pes_shockfile_parameters_path):
            # TODOO HACK, I beleive this wasnt properly done seperately per scenario, so confirm before reusing.
            payment_set_f_per_ha = pd.read_csv(os.path.join(p.cur_dir, 'payment_set_f_per_ha.csv'), index_col=0)
            payment_set_f_per_ha.rename(columns={'sums': 'payment_set_f_per_ha_sum', 'counts': 'payment_set_f_per_ha_count'}, inplace=True)
            payment_set_f_per_ha['payment_set_f_per_ha'] = payment_set_f_per_ha['payment_set_f_per_ha_sum'] / payment_set_f_per_ha['payment_set_f_per_ha_count']
            payment_set_f_per_ha.drop(['payment_set_f_per_ha_sum', 'payment_set_f_per_ha_count'], axis=1, inplace=True)

            ha_removed_from_production = pd.read_csv(os.path.join(p.cur_dir, 'ha_removed_from_production.csv'), index_col=0)
            ha_removed_from_production.rename(columns={'sums': 'ha_removed_from_production', 'counts': 'ha_removed_from_production_count'}, inplace=True)
            ha_removed_from_production.drop(['ha_removed_from_production_count'], axis=1, inplace=True)

            expenditure_f_per_grid = pd.read_csv(os.path.join(p.cur_dir, 'expenditure_f_per_grid.csv'), index_col=0)
            expenditure_f_per_grid.rename(columns={'sums': 'expenditure_f_per_grid', 'counts': 'expenditure_f_per_grid_count'}, inplace=True)
            expenditure_f_per_grid.drop(['expenditure_f_per_grid_count'], axis=1, inplace=True)

            value_protected_per_grid = pd.read_csv(os.path.join(p.cur_dir, 'value_protected_per_grid.csv'), index_col=0)
            value_protected_per_grid.rename(columns={'sums': 'value_protected_per_grid', 'counts': 'value_protected_per_grid_count'}, inplace=True)
            value_protected_per_grid.drop(['value_protected_per_grid_count'], axis=1, inplace=True)

            profit_forgone_f_per_ha_conserved = pd.read_csv(os.path.join(p.cur_dir, 'profit_forgone_f_per_ha_conserved.csv'), index_col=0)
            profit_forgone_f_per_ha_conserved.rename(columns={'sums': 'profit_forgone_f_per_ha_conserved_sum', 'counts': 'profit_forgone_f_per_ha_conserved_count'}, inplace=True)
            profit_forgone_f_per_ha_conserved['profit_forgone_f_per_ha_conserved'] = profit_forgone_f_per_ha_conserved['profit_forgone_f_per_ha_conserved_sum'] / profit_forgone_f_per_ha_conserved['profit_forgone_f_per_ha_conserved_count']
            profit_forgone_f_per_ha_conserved.drop(['profit_forgone_f_per_ha_conserved_sum', 'profit_forgone_f_per_ha_conserved_count'], axis=1, inplace=True)

            df = pd.merge(payment_set_f_per_ha, ha_removed_from_production, right_index=True, left_index=True)
            df = pd.merge(df, expenditure_f_per_grid, right_index=True, left_index=True)
            df = pd.merge(df, value_protected_per_grid, right_index=True, left_index=True)
            df = pd.merge(df, profit_forgone_f_per_ha_conserved, right_index=True, left_index=True)

            # NEGATIVE PROFIT IRELAND MESSES THIS UP WTF IRELAND?
            df.loc[df['expenditure_f_per_grid'] < 0, 'expenditure_f_per_grid'] = 0
            df.loc[df['value_protected_per_grid'] < 0, 'value_protected_per_grid'] = 0
            df.loc[df['profit_forgone_f_per_ha_conserved'] < 0, 'profit_forgone_f_per_ha_conserved'] = 0

            gdf = gpd.read_file(p.gtap37_aez18_path, driver='GPKG')

            out_df = pd.merge(gdf[[i for i in gdf.columns if i != 'geometry']], df, left_on='pyramid_id', right_index=True, how='outer')
            out_df.to_csv(p.pes_shockfile_parameters_path)


            out_gdf = gdf.merge(df, left_on='pyramid_id', right_index=True, how='outer')
            out_gdf.to_file(p.pes_shockfile_parameters_path.replace('.csv', '.gpkg'), driver='GPKG')

        # Now for country-constrained
        p.pes_shockfile_parameters_country_level_path = os.path.join(p.cur_dir, 'pes_shockfile_parameters_country_level.csv')
        p.payment_set_f_per_ha_country_level_path = os.path.join(p.pes_model_dir, "payment_set_f_per_ha_country_level.tif")
        if not hb.path_exists(p.pes_shockfile_parameters_country_level_path):

            csv_output_path = os.path.join(p.cur_dir, 'payment_set_f_per_ha_country_level.csv')
            vector_output_path = os.path.join(p.cur_dir, 'payment_set_f_per_ha_country_level.gpkg')
            hb.zonal_statistics_flex(p.payment_set_f_per_ha_country_level_path, p.gtap37_aez18_path, p.zone_ids_raster_300sec_path, 'pyramid_id', 5, 6, -9999, -9999.,
                                     assert_projections_same=False,  csv_output_path=csv_output_path, vector_output_path=vector_output_path, stats_to_retrieve='sums_counts')

            p.ha_removed_from_production_path = os.path.join(p.cur_dir, 'ha_removed_from_production_country_level.tif')
            payments = hb.as_array(p.payment_set_f_per_ha_country_level_path)
            ha_changed = hb.as_array(p.desired_ha_conserved_path)
            ha_removed = np.where((payments > 0) & (ha_changed > 0), ha_changed, 0)
            hb.save_array_as_geotiff(ha_removed, p.ha_removed_from_production_path, p.zone_ids_raster_300sec_path, ndv=-9999., data_type=6)

            csv_output_path = os.path.join(p.cur_dir, 'ha_removed_from_production_country_level.csv')
            vector_output_path = os.path.join(p.cur_dir, 'ha_removed_from_production_country_level.gpkg')
            hb.zonal_statistics_flex(p.ha_removed_from_production_path, p.gtap37_aez18_path, p.zone_ids_raster_300sec_path, 'pyramid_id', 5, 6, -9999, -9999.,
                                     assert_projections_same=False,  csv_output_path=csv_output_path, vector_output_path=vector_output_path, stats_to_retrieve='sums_counts')

            p.expenditure_f_per_grid_path = os.path.join(p.pes_model_dir, r"expenditure_f_per_grid_country_level.tif")
            csv_output_path = os.path.join(p.cur_dir, 'expenditure_f_per_grid_country_level.csv')
            vector_output_path = os.path.join(p.cur_dir, 'expenditure_f_per_grid_country_level.gpkg')
            hb.zonal_statistics_flex(p.expenditure_f_per_grid_path, p.gtap37_aez18_path, p.zone_ids_raster_300sec_path, 'pyramid_id', 5, 6, -9999, -9999.,
                                     assert_projections_same=False,  csv_output_path=csv_output_path, vector_output_path=vector_output_path, stats_to_retrieve='sums_counts')

            p.value_protected_per_grid_path = os.path.join(p.pes_model_dir, "value_protected_given_budget_per_grid_f_country_level.tif")
            csv_output_path = os.path.join(p.cur_dir, 'value_protected_per_grid_country_level.csv')
            vector_output_path = os.path.join(p.cur_dir, 'value_protected_per_grid_country_level.gpkg')
            hb.zonal_statistics_flex(p.value_protected_per_grid_path, p.gtap37_aez18_path, p.zone_ids_raster_300sec_path, 'pyramid_id', 5, 6, -9999, -9999.,
                                     assert_projections_same=False,  csv_output_path=csv_output_path, vector_output_path=vector_output_path, stats_to_retrieve='sums_counts')

            p.profit_forgone_f_per_ha_conserved_path  = os.path.join(p.pes_model_dir, "value_protected_given_budget_per_grid_f_country_level.tif")
            csv_output_path = os.path.join(p.cur_dir, 'profit_forgone_f_per_ha_conserved_country_level.csv')
            vector_output_path = os.path.join(p.cur_dir, 'profit_forgone_f_per_ha_conserved_country_level.gpkg')
            hb.zonal_statistics_flex(p.profit_forgone_f_per_ha_conserved_path, p.gtap37_aez18_path, p.zone_ids_raster_300sec_path, 'pyramid_id', 5, 6, -9999, -9999.,
                                     assert_projections_same=False,  csv_output_path=csv_output_path, vector_output_path=vector_output_path, stats_to_retrieve='sums_counts')

            payment_set_f_per_ha = pd.read_csv(os.path.join(p.cur_dir, 'payment_set_f_per_ha_country_level.csv'), index_col=0)
            payment_set_f_per_ha.rename(columns={'sums': 'payment_set_f_per_ha_country_level_sum', 'counts': 'payment_set_f_per_ha_country_level_count'}, inplace=True)
            payment_set_f_per_ha['payment_set_f_per_ha_country_level'] = payment_set_f_per_ha['payment_set_f_per_ha_country_level_sum'] / payment_set_f_per_ha['payment_set_f_per_ha_country_level_count']
            payment_set_f_per_ha.drop(['payment_set_f_per_ha_country_level_sum', 'payment_set_f_per_ha_country_level_count'], axis=1, inplace=True)

            ha_removed_from_production = pd.read_csv(os.path.join(p.cur_dir, 'ha_removed_from_production_country_level.csv'), index_col=0)
            ha_removed_from_production.rename(columns={'sums': 'ha_removed_from_production_country_level_sum', 'counts': 'ha_removed_from_production_country_level_count'}, inplace=True)
            ha_removed_from_production.drop(['ha_removed_from_production_country_level_count'], axis=1, inplace=True)

            expenditure_f_per_grid = pd.read_csv(os.path.join(p.cur_dir, 'expenditure_f_per_grid.csv'), index_col=0)
            expenditure_f_per_grid.rename(columns={'sums': 'expenditure_f_per_grid_country_level', 'counts': 'expenditure_f_per_grid_country_level_count'}, inplace=True)
            expenditure_f_per_grid.drop(['expenditure_f_per_grid_country_level_count'], axis=1, inplace=True)

            value_protected_per_grid = pd.read_csv(os.path.join(p.cur_dir, 'value_protected_per_grid_country_level.csv'), index_col=0)
            value_protected_per_grid.rename(columns={'sums': 'value_protected_per_grid_country_level', 'counts': 'value_protected_per_grid_country_level_count'}, inplace=True)
            value_protected_per_grid.drop(['value_protected_per_grid_country_level_count'], axis=1, inplace=True)

            profit_forgone_f_per_ha_conserved = pd.read_csv(os.path.join(p.cur_dir, 'profit_forgone_f_per_ha_conserved_country_level.csv'), index_col=0)
            profit_forgone_f_per_ha_conserved.rename(columns={'sums': 'profit_forgone_f_per_ha_conserved_country_level_sum', 'counts': 'profit_forgone_f_per_ha_conserved_country_level_count'}, inplace=True)
            profit_forgone_f_per_ha_conserved['profit_forgone_f_per_ha_conserved_country_level'] = profit_forgone_f_per_ha_conserved['profit_forgone_f_per_ha_conserved_country_level_sum'] / profit_forgone_f_per_ha_conserved['profit_forgone_f_per_ha_conserved_country_level_count']
            profit_forgone_f_per_ha_conserved.drop(['profit_forgone_f_per_ha_conserved_country_level_sum', 'profit_forgone_f_per_ha_conserved_country_level_count'], axis=1, inplace=True)

            df = pd.merge(payment_set_f_per_ha, ha_removed_from_production, right_index=True, left_index=True)
            df = pd.merge(df, expenditure_f_per_grid, right_index=True, left_index=True)
            df = pd.merge(df, value_protected_per_grid, right_index=True, left_index=True)
            df = pd.merge(df, profit_forgone_f_per_ha_conserved, right_index=True, left_index=True)

            # NEGATIVE PROFIT IRELAND MESSES THIS UP WTF IRELAND?
            df.loc[df['expenditure_f_per_grid_country_level'] < 0, 'expenditure_f_per_grid_country_level'] = 0
            df.loc[df['value_protected_per_grid_country_level'] < 0, 'value_protected_per_grid_country_level'] = 0
            df.loc[df['profit_forgone_f_per_ha_conserved_country_level'] < 0, 'profit_forgone_f_per_ha_conserved_country_level'] = 0





            gdf = gpd.read_file(p.gtap37_aez18_path, driver='GPKG')

            out_df = pd.merge(gdf[[i for i in gdf.columns if i != 'geometry']], df, left_on='pyramid_id', right_index=True, how='outer')
            out_df.to_csv(p.pes_shockfile_parameters_country_level_path)


            out_gdf = gdf.merge(df, left_on='pyramid_id', right_index=True, how='outer')
            out_gdf.to_file(p.pes_shockfile_parameters_country_level_path.replace('.csv', '.gpkg'), driver='GPKG')

def protect_30_by_30_endogenous_land_shock(p):
    # TODOO Make this be done once at the top and then made into base data srsly
    p.zone_ids_raster_path = os.path.join(p.cur_dir, 'GTAP37_AEZ18_ids_10sec.tif')
    p.zone_ids_raster_300sec_path = os.path.join(p.cur_dir, 'GTAP37_AEZ18_ids_300sec.tif')
    p.zone_ids_raster_900sec_path = os.path.join(p.cur_dir, 'GTAP37_AEZ18_ids_900sec.tif')


    p.biodiversity_bau_path = os.path.join(p.model_base_data_dir, 'biodiversity', 'biodiversity_bau_10sec.tif')
    p.biodiversity_baseline_input_path = os.path.join(p.model_base_data_dir, 'biodiversity', 'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_biodiversity.tif')
    p.biodiversity_baseline_path = os.path.join(p.model_base_data_dir, 'biodiversity', 'biodiversity_baseline_10sec.tif')
    p.biodiversity_baseline_norm_path = os.path.join(p.model_base_data_dir, 'biodiversity', 'biodiversity_baseline_norm_10sec.tif')


    p.global_protected_land_input_path = os.path.join(p.model_base_data_dir, 'protected_areas', 'StrictPAs.tif')
    p.protected_areas_baseline_10sec_path = os.path.join(p.model_base_data_dir, 'protected_areas', 'protected_areas_baseline_10sec.tif')


    p.carbon_storage_input_path = os.path.join(p.model_base_data_dir, 'carbon_storage', 'carbon_above_ground_mg_per_ha_global_30s_johnson_compressed.tif')
    p.carbon_storage_baseline_path = os.path.join(p.model_base_data_dir, 'carbon_storage', 'carbon_storage_baseline_10sec.tif')
    p.carbon_storage_baseline_norm_path = os.path.join(p.model_base_data_dir, 'carbon_storage', 'carbon_storage_baseline_norm_10sec.tif')


    p.biodiversity_baseline_norm_300sec_path = os.path.join(p.cur_dir, 'biodiversity_baseline_norm_300sec.tif')
    p.carbon_storage_baseline_norm_300sec_path = os.path.join(p.cur_dir, 'carbon_storage_baseline_norm_300sec.tif')
    p.protected_areas_baseline_300sec_path = os.path.join(p.cur_dir, 'protected_areas_baseline_300sec.tif')

    p.conservation_value_index_path = os.path.join(p.cur_dir, 'conservation_value_index.tif')
    p.conservation_value_ranked_path = os.path.join(p.cur_dir, 'conservation_value_ranked.tif')
    p.conservation_value_ranked_keys_path = os.path.join(p.cur_dir, 'conservation_value_ranked_keys.tif')

    p.ha_to_protect_10sec_path = os.path.join(p.cur_dir, 'ha_to_protect_10sec.tif')
    if p.run_this:

        # Process to make Pyramidal
        if not hb.path_exists(p.protected_areas_baseline_10sec_path):
            hb.resample_to_match_pyramid(p.global_protected_land_input_path, p.biodiversity_bau_path, p.protected_areas_baseline_10sec_path,
                                         s_srs_wkt=hb.mollweide_wkt, verbose=True)
        if not hb.path_exists(p.biodiversity_baseline_path):
            hb.resample_to_match_pyramid(p.biodiversity_baseline_input_path, p.biodiversity_bau_path, p.biodiversity_baseline_path, verbose=True)
        if not hb.path_exists(p.carbon_storage_baseline_path):
            hb.resample_to_match_pyramid(p.carbon_storage_input_path, p.biodiversity_bau_path, p.carbon_storage_path, verbose=True)

        # Normalize Biodiv and carbon

        ## DONT NEED to normalize biodiv as it already is, despite 30 outlier values of 1.5, which i think are mistakes.
        # if not hb.path_exists(p.biodiversity_baseline_norm_path):
        #     # biodiversity_baseline_array = hb.as_array(p.biodiversity_baseline_path)
        #     # biodiversity_baseline_norm_array = hb.normalize_array_memsafe(biodiversity_baseline_array, ndv=-9999.)
        #     hb.normalize_array_memsafe(p.biodiversity_baseline_path, p.biodiversity_baseline_norm_path, ndv=-9999.)
        #
        #     hb.save_array_as_geotiff(biodiversity_baseline_norm_array, p.biodiversity_baseline_norm_path, p.biodiversity_baseline_path)

        if not hb.path_exists(p.carbon_storage_baseline_norm_path):
            hb.normalize_array_memsafe(p.carbon_storage_baseline_path, p.carbon_storage_baseline_norm_path, ndv=-9999., log_transform=False)

        pa1_path = os.path.join(p.model_base_data_dir, "protected_areas\pa1.tif")
        pa2_path = os.path.join(p.model_base_data_dir, "protected_areas\pa2.tif")
        pa3_path = os.path.join(p.model_base_data_dir, "protected_areas\pa3.tif")

        p.protected_areas_all_baseline_10sec_path = os.path.join(p.model_base_data_dir, "protected_areas\protected_areas_all_baseline_10sec.tif")
        if not hb.path_exists(p.protected_areas_all_baseline_10sec_path):

            hb.raster_calculator_af_flex([pa1_path, pa2_path, pa3_path], lambda x, y, z: np.where((x==1) | (y==1) | (z==1), 1, 0), p.protected_areas_all_baseline_10sec_path)


        hb.make_path_global_pyramid(p.carbon_storage_baseline_norm_path)
        hb.make_path_global_pyramid(p.biodiversity_baseline_path)

        # LEARNING POINT, classification of 0-1 variables like below need to have average resampling done and NOT ignore their NDV.
        # this means the ndv needs to be sensible, like correctly as zero. Otherwise, it just takes the average of cells that are 1
        # and upsampling resample methods cause incorrectness. Note that this same flaw is true of regular gdal bilinear resampling
        # and is just an error most people don't catch
        hb.make_path_global_pyramid(p.protected_areas_baseline_10sec_path)

        p.protected_areas_all_baseline_10sec_floats_path = os.path.join(p.model_base_data_dir, r"protected_areas\protected_areas_all_baseline_10sec_floats.tif")
        if not hb.path_exists(p.protected_areas_all_baseline_10sec_floats_path):
            hb.change_array_datatype_and_ndv(p.protected_areas_all_baseline_10sec_path, p.protected_areas_all_baseline_10sec_floats_path, data_type=6, output_ndv=-9999.0)

        hb.make_path_global_pyramid(p.protected_areas_all_baseline_10sec_floats_path)

        # CHOICE HERE: I switched everything to 300s res from here on out and so everything below is not memory safe. This
        # was just done for speed reasons.
        if not hb.path_exists(p.carbon_storage_baseline_norm_300sec_path):
            # TODOO Note that currently as envisioned, this shifts everything by half a pixel, per implementation of GDAL.GRA_AVERAGE. Consider fixing by either shifting input prior to creation of overviews or by testing if other resampling methods are sum-valid while not shifting.
            hb.resample_via_pyramid_overviews(p.carbon_storage_baseline_norm_path, 300, p.carbon_storage_baseline_norm_300sec_path)
        if not hb.path_exists(p.biodiversity_baseline_norm_300sec_path):
            hb.resample_via_pyramid_overviews(p.biodiversity_baseline_path, 300, p.biodiversity_baseline_norm_300sec_path)
        if not hb.path_exists(p.protected_areas_baseline_300sec_path):
            hb.resample_via_pyramid_overviews(p.protected_areas_baseline_10sec_path, 300, p.protected_areas_baseline_300sec_path)
            # hb.resample_via_pyramid_overviews(p.protected_areas_baseline_10sec_path, 300, p.protected_areas_baseline_300sec_path,
            #                                   force_overview_rewrite=False, new_ndv=0, overview_resampling_algorithm='average')
        p.protected_areas_all_baseline_300sec_path = os.path.join(p.cur_dir, 'protected_areas_all_baseline_300sec.tif')
        if not hb.path_exists(p.protected_areas_all_baseline_300sec_path):
            hb.resample_via_pyramid_overviews(p.protected_areas_all_baseline_10sec_floats_path, 300, p.protected_areas_all_baseline_300sec_path)

            # hb.resample_via_pyramid_overviews(p.protected_areas_all_baseline_10sec_path, 300, p.protected_areas_all_baseline_300sec_path,
            #                                   force_overview_rewrite=True, new_ndv=0, overview_resampling_algorithm='average', overview_data_types=6)
            #
        carbon_array = hb.as_array(p.carbon_storage_baseline_norm_300sec_path)
        biodiv_array = hb.as_array(p.biodiversity_baseline_norm_300sec_path)
        pa_array = hb.as_array(p.protected_areas_baseline_300sec_path)
        pa_all_array = hb.as_array(p.protected_areas_all_baseline_300sec_path)

        if not hb.path_exists(p.conservation_value_index_path):
            conservation_value = carbon_array * biodiv_array
            hb.save_array_as_geotiff(conservation_value, p.conservation_value_index_path, p.biodiversity_baseline_norm_300sec_path)
        else:
            conservation_value = hb.as_array(p.conservation_value_index_path)

        if not hb.path_exists(p.conservation_value_ranked_path) or not hb.path_exists(p.conservation_value_ranked_keys_path):
            conservation_value_ranked, keys = hb.get_rank_array_and_keys(conservation_value, ndv=0)

            hb.save_array_as_geotiff(conservation_value_ranked, p.conservation_value_ranked_path, p.biodiversity_baseline_norm_300sec_path)
            hb.save_array_as_geotiff(keys, p.conservation_value_ranked_keys_path, p.biodiversity_baseline_norm_300sec_path, n_cols=keys.shape[1], n_rows=keys.shape[0])

        df_results = None
        if df_results is None:
            # Use this opportunity to clean the projections spreadsheet.
            df_results = pd.read_excel(p.full_projection_results_spreadsheet_path, index_col=0)





        # From Waldron 2020 currently 16% of land and 7.4% of ocean is designated as protected.
        # Calibrate observed PA to equal the no-PA-expansion scenario in waldron
        # Biodiversity wilderness consensus covers 43% of terrestrial land, 26% marine
        # T5 adds 5% more of the land surface to existing PA network (up to 20% planetary land).
        # T6, expand 30% terrestiral land that reduces global species extinction

        p.protected_areas_all_baseline_hectares_300sec_path = os.path.join(p.cur_dir, 'protected_areas_all_baseline_hectares_300sec.tif')
        p.protected_areas_per_aezreg_spreadsheet_path = os.path.join(p.cur_dir, 'protected_areas_per_aezreg.xlsx')
        if not hb.path_exists(p.protected_areas_per_aezreg_spreadsheet_path):
            # Calculate protected area in HECTARES.

            ha_per_cell_300sec = hb.as_array(p.ha_per_cell_300sec_path)
            pa_haectares_array = pa_all_array * ha_per_cell_300sec
            hb.save_array_as_geotiff(pa_haectares_array, p.protected_areas_all_baseline_hectares_300sec_path, p.protected_areas_baseline_300sec_path)

            # Start by calculating the stric-PA total area in each gtap-AEZ.

            df = hb.zonal_statistics_flex(p.protected_areas_all_baseline_hectares_300sec_path,
                            p.gtap37_aez18_path,
                            zone_ids_raster_path=p.zone_ids_raster_300sec_path,
                            id_column_label='pyramid_id',
                            zones_raster_data_type=5,
                            values_raster_data_type=6,
                            zones_ndv=-9999,
                            values_ndv=-9999,
                            all_touched=None,
                            assert_projections_same=False, )
            df['pyramid_id'] = df.index
            df.rename(columns={'sums': 'protected_ha_sum'}, inplace=True)
            df = df.fillna(0.0)

            df_results = pd.merge(df_results, df, on='pyramid_id', how='outer')

            ha_in_region = hb.as_array(p.ha_per_cell_300sec_path)

            zone_ids = hb.as_array(p.zone_ids_raster_300sec_path)
            zone_ones = np.where(zone_ids > 0, 1, 0)
            ha_in_zone = ha_in_region * zone_ones
            temp_path = hb.temp_filename()
            hb.save_array_as_geotiff(ha_in_zone, temp_path, p.carbon_storage_baseline_norm_300sec_path)
            df = hb.zonal_statistics_flex(temp_path,
                            p.gtap37_aez18_path,
                            zone_ids_raster_path=p.zone_ids_raster_300sec_path,
                            id_column_label='pyramid_id',
                            zones_raster_data_type=5,
                            values_raster_data_type=6,
                            zones_ndv=-9999,
                            values_ndv=-9999,
                            all_touched=None,
                            assert_projections_same=False, )
            df['pyramid_id'] = df.index
            df.rename(columns={'sums': 'total_land_ha'}, inplace=True)
            df_results = pd.merge(df_results, df, on='pyramid_id', how='outer')

            df_results['total_land_ha_no_barren'] = df_results['gtap1_2014_baseline_cropland_ha'] \
                                       + df_results['gtap1_2014_baseline_forest_ha'] \
                                       + df_results['gtap1_2014_baseline_grassland_ha'] \
                                       + df_results['gtap1_2014_baseline_natural_ha'] \
                                       + df_results['gtap1_2021_baseline_cropland_ha']

            df_results['percent_protected_exclude_barren_2014'] = df_results['protected_ha_sum'] / df_results['total_land_ha_no_barren']
            df_results['percent_protected_all_2014'] = df_results['protected_ha_sum'] / df_results['total_land_ha']


            necessary_increase_multiplier = 2.06925939 # Found in excel

            # JUST FOR REFERENCE
            df_results['new_protected_ha_based_on_proportion_protected'] = np.where((
                        df_results['protected_ha_sum'] * necessary_increase_multiplier) / df_results['total_land_ha'] < 0.95,
                        df_results['protected_ha_sum'] * (necessary_increase_multiplier-1), # new land
                        (df_results['total_land_ha'] - df_results['protected_ha_sum']))

            # THIS IS THE ONE USED, Left new_protected_ha_based_on_proportion_protected for reference
            required_increase_proportion_remaining = .1798 # found in excel
            df_results['new_protected_ha_based_on_proportion_remaining'] = (df_results['total_land_ha'] - df_results['protected_ha_sum']) * required_increase_proportion_remaining
            df_results.fillna(0.0, inplace=True)
            df_results.to_excel(p.protected_areas_per_aezreg_spreadsheet_path)

            to_group_cols = ['pyramid_id', 'gtap37v10_pyramid_id', 'gtap37v10_pyramid_name', 'aez_pyramid_id', 'AZREG', 'total_land_ha', 'protected_ha_sum', 'new_protected_ha_based_on_proportion_remaining']

            to_group = df_results[to_group_cols]
            to_group = to_group.set_index(['pyramid_id', 'gtap37v10_pyramid_id', 'aez_pyramid_id', 'gtap37v10_pyramid_name'])

            df_grouped = pd.pivot_table(to_group, index=['pyramid_id', 'gtap37v10_pyramid_id', 'aez_pyramid_id', 'gtap37v10_pyramid_name'], aggfunc=np.mean)
            df_grouped = df_grouped[df_grouped['new_protected_ha_based_on_proportion_remaining'] > 0]

            df_grouped['dropzero'] = df_grouped.index.get_level_values(1)

            df_grouped = df_grouped[df_grouped['dropzero'] > 0]
            df_grouped = df_grouped.drop(columns='dropzero')
            p.protected_areas_per_aezreg_spreadsheet_path = os.path.join(p.cur_dir, 'protected_areas_per_aezreg_spreadsheet.xlsx')
            df_grouped.to_excel(p.protected_areas_per_aezreg_spreadsheet_path)


            df_grouped = pd.pivot_table(df_grouped, index=['gtap37v10_pyramid_id', 'gtap37v10_pyramid_name'], aggfunc=np.sum)

            df_grouped = df_grouped.rename(columns={'protected_ha_sum': 'protected_ha_2014_sum', 'new_protected_ha_based_on_proportion_remaining': 'new_protected_ha_2030_sum'})
            df_grouped['proportion_protected_in_2014'] = df_grouped['protected_ha_2014_sum'] / df_grouped['total_land_ha']
            df_grouped['proportion_newly_protected_by_2030'] = df_grouped['new_protected_ha_2030_sum'] / df_grouped['total_land_ha']

            p.protected_areas_per_reg_spreadsheet_path = os.path.join(p.cur_dir, 'protected_areas_per_reg_spreadsheet.xlsx')
            df_grouped.to_excel(p.protected_areas_per_reg_spreadsheet_path)

        possible_path = os.path.join(p.available_land_dir, 'current_and_arable_definition_4.tif')

        p.available_land_10sec_path = hb.get_existing_path_from_nested_sources(possible_path, p)
        p.available_land_300sec_path = os.path.join(p.cur_dir, 'current_and_arable_definition_4_300sec.tif')
        if not hb.path_exists(p.available_land_300sec_path):
            temppath = hb.temp(remove_at_exit=True)
            hb.resample_via_pyramid_overviews(p.available_land_10sec_path, 300, p.available_land_300sec_path, scale_array_by_resolution_change=True)


        # Confirm that the difference in sum of the unsampled and pyramid sampled are the same:"
        # LEARNING POINT: It takes 2.5 minutes to call as_array on a 10sec 64float. Conversely, it takes 2.0 mins to calculate the sum of that array generated. I need to switch to xarrays or dask.
        # 2591961924.59536 - 2879957.694236822* (300/10)^2 = -.217789
        # which i guess is 10 digits of accuracy.

        p.ha_to_protect_raster_path = os.path.join(p.cur_dir, 'ha_to_protect.tif')
        p.displaced_ag_land_raster_path = os.path.join(p.cur_dir, 'displaced_ag_land.tif')
        if not hb.path_exists(p.displaced_ag_land_raster_path):
            existing_or_potential_displaced_ag_land = hb.as_array(p.available_land_300sec_path)
            zone_ids_array = hb.as_array(p.zone_ids_raster_300sec_path)
            unique_zones = np.unique(zone_ids_array)
            unique_zones = [i for i in unique_zones if i >= 0] # drop ndv

            # Iterate through ranking to make protections
            df = pd.read_excel(p.protected_areas_per_aezreg_spreadsheet_path, index_col=0)
            ha_per_cell = hb.as_array(p.ha_per_cell_300sec_path)
            ha_already_protected = hb.as_array(p.protected_areas_all_baseline_hectares_300sec_path)

            df['pyramid_id_as_col'] = df.index
            to_protect_array = np.zeros(max(df['pyramid_id_as_col']) + 1)
            for i in df.index:
                zone_id = df.at[i, 'pyramid_id_as_col']
                ha_to_protect = df.at[i, 'new_protected_ha_based_on_proportion_remaining']
                to_protect_array[zone_id] = ha_to_protect
                L.info('Processing position, zone', i, zone_id, 'which will protect another', ha_to_protect, 'hectares')

            ranked_keys = hb.as_array(p.conservation_value_ranked_keys_path)
            ha_to_protect_array = np.zeros_like(ha_per_cell)
            displaced_ag_land = np.zeros_like(ha_per_cell)

            for i in range(ranked_keys.shape[1]):
                rc = (int(ranked_keys[0, i]), int(ranked_keys [1, i]))

                current_aezreg = zone_ids_array[rc]
                if current_aezreg > 0:
                    remaining_ha_to_protect = to_protect_array[current_aezreg]
                    if remaining_ha_to_protect > 0:

                        ha_to_protect = ha_per_cell[rc] - ha_already_protected[rc]

                        to_protect_array[current_aezreg] -= ha_to_protect

                        ha_to_protect_array[rc] = ha_to_protect
                        displaced_ag_land[rc] = existing_or_potential_displaced_ag_land[rc]



                        if i % 100000 == 0:
                            L.info(i, 'rc', rc, 'current_aezreg', current_aezreg, 'ha_already_protected:', ha_already_protected[rc], 'ha_to_protect:', ha_to_protect)



            hb.save_array_as_geotiff(ha_to_protect_array, p.ha_to_protect_raster_path, p.ha_per_cell_300sec_path)
            hb.save_array_as_geotiff(displaced_ag_land, p.displaced_ag_land_raster_path, p.ha_per_cell_300sec_path)

        p.full_results_with_ha_removed_from_production_spreadsheet_path = os.path.join(p.cur_dir, 'full_results_with_ha_removed_from_production.xlsx')
        p.ha_removed_from_production_spreadsheet_path = os.path.join(p.cur_dir, 'ha_removed_from_production.xlsx')
        if not hb.path_exists(p.ha_removed_from_production_spreadsheet_path):
            df_results = pd.read_excel(p.protected_areas_per_aezreg_spreadsheet_path)

            df = hb.zonal_statistics_flex(p.ha_to_protect_raster_path,
                                          p.gtap37_aez18_path,
                                          zone_ids_raster_path=p.zone_ids_raster_300sec_path,
                                          id_column_label='pyramid_id',
                                          zones_raster_data_type=5,
                                          values_raster_data_type=6,
                                          zones_ndv=-9999,
                                          values_ndv=-9999,
                                          all_touched=None,
                                          assert_projections_same=False, )
            df['pyramid_id'] = df.index
            df.rename(columns={'sums': '30_by_30_ha_newly_protected'}, inplace=True)
            df_results = pd.merge(df_results, df, on='pyramid_id', how='outer')

            df = hb.zonal_statistics_flex(p.displaced_ag_land_raster_path,
                                          p.gtap37_aez18_path,
                                          zone_ids_raster_path=p.zone_ids_raster_300sec_path,
                                          id_column_label='pyramid_id',
                                          zones_raster_data_type=5,
                                          values_raster_data_type=6,
                                          zones_ndv=-9999,
                                          values_ndv=-9999,
                                          all_touched=None,
                                          assert_projections_same=False, )
            df['pyramid_id'] = df.index
            df.rename(columns={'sums': '30_by_30_displaced_ag_land_ha'}, inplace=True)
            df_results = pd.merge(df_results, df, on='pyramid_id', how='outer')
            df_results.fillna(0.0, inplace=True)
            df_results['AZREG'] = 'AZ' + df_results['aez_pyramid_id'].map(int).map(str) + '_' + df_results['gtap37v10_pyramid_name'].map(str)
            df_results.to_excel(p.full_results_with_ha_removed_from_production_spreadsheet_path)

            df_just_ha_removed_from_production = df_results[['pyramid_id', 'aez_pyramid_id', 'gtap37v10_pyramid_name', 'AZREG',
                                                             '30_by_30_ha_newly_protected',  '30_by_30_displaced_ag_land_ha']]

            df_just_ha_removed_from_production.to_excel(p.ha_removed_from_production_spreadsheet_path)

        if not hb.path_exists(p.ha_to_protect_10sec_path):

            hb.resample_to_match_pyramid(p.ha_to_protect_raster_path, p.protected_areas_all_baseline_10sec_path, p.ha_to_protect_10sec_path, output_data_type=6, ndv=-9999.)

seals_processing_steps = 'starts here'

def align_land_available_inputs(p):
    """Manually moved into base data."""


    p.unaligned_inputs = {}
    p.unaligned_inputs['nutrient_retention_index'] = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\nutrient_retention_index.tif"
    p.unaligned_inputs['oxygen_availability_index'] = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\oxygen_availability_index.tif"
    p.unaligned_inputs['rooting_conditions_index'] = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\rooting_conditions_index.tif"
    p.unaligned_inputs['toxicity_index'] = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\toxicity_index.tif"
    p.unaligned_inputs['workability_index'] = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\workability_index.tif"
    p.unaligned_inputs['excess_salts_index'] = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\excess_salts_index.tif"
    p.unaligned_inputs['nutrient_availability_index'] = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\nutrient_availability_index.tif"
    p.unaligned_inputs['TRI'] = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\TRI_30s.tif"
    p.unaligned_inputs['caloric_yield'] = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\available_land_inputs\caloric_yield_total_all_filled.tif"
    p.unaligned_inputs['crop_suitability'] = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\soil_suitability\cropsuitability_rainfed_and_irrigated\1981-2010\overall_cropsuit_i_1981-2010\overall_cropsuit_i_1981-2010.bil"

    p.aligned_inputs = {}
    p.aligned_inputs['nutrient_retention_index'] = os.path.join(p.cur_dir, 'nutrient_retention_index_10s.tif')
    p.aligned_inputs['oxygen_availability_index'] = os.path.join(p.cur_dir, 'oxygen_availability_index_10s.tif')
    p.aligned_inputs['rooting_conditions_index'] = os.path.join(p.cur_dir, 'rooting_conditions_index_10s.tif')
    p.aligned_inputs['toxicity_index'] = os.path.join(p.cur_dir, 'toxicity_index_10s.tif')
    p.aligned_inputs['workability_index'] = os.path.join(p.cur_dir, 'workability_index_10s.tif')
    p.aligned_inputs['excess_salts_index'] = os.path.join(p.cur_dir, 'excess_salts_index_10s.tif')
    p.aligned_inputs['nutrient_availability_index'] = os.path.join(p.cur_dir, 'nutrient_availability_index_10s.tif')
    p.aligned_inputs['TRI'] = os.path.join(p.cur_dir, 'TRI_10s.tif')
    p.aligned_inputs['caloric_yield'] = os.path.join(p.cur_dir, 'caloric_yield_10s.tif')
    p.aligned_inputs['crop_suitability'] = os.path.join(p.cur_dir, 'crop_suitability_10s.tif')

    p.resample_methods = {}
    p.resample_methods['nutrient_retention_index'] = 'bilinear'
    p.resample_methods['oxygen_availability_index'] = 'bilinear'
    p.resample_methods['rooting_conditions_index'] = 'bilinear'
    p.resample_methods['toxicity_index'] = 'bilinear'
    p.resample_methods['workability_index'] = 'bilinear'
    p.resample_methods['excess_salts_index'] = 'bilinear'
    p.resample_methods['nutrient_availability_index'] = 'bilinear'
    p.resample_methods['TRI'] = 'bilinear'
    p.resample_methods['caloric_yield'] = 'bilinear'
    p.resample_methods['crop_suitability'] = 'bilinear'

    p.available_land_inputs = {}
    p.available_land_inputs['nutrient_retention_index'] = os.path.join(hb.GTAP_INVEST_BASE_DATA_DIR, 'available_land_inputs', 'nutrient_retention_index_10s.tif')
    p.available_land_inputs['oxygen_availability_index'] = os.path.join(hb.GTAP_INVEST_BASE_DATA_DIR, 'available_land_inputs', 'oxygen_availability_index_10s.tif')
    p.available_land_inputs['rooting_conditions_index'] = os.path.join(hb.GTAP_INVEST_BASE_DATA_DIR, 'available_land_inputs', 'rooting_conditions_index_10s.tif')
    p.available_land_inputs['toxicity_index'] = os.path.join(hb.GTAP_INVEST_BASE_DATA_DIR, 'available_land_inputs', 'toxicity_index_10s.tif')
    p.available_land_inputs['workability_index'] = os.path.join(hb.GTAP_INVEST_BASE_DATA_DIR, 'available_land_inputs', 'workability_index_10s.tif')
    p.available_land_inputs['excess_salts_index'] = os.path.join(hb.GTAP_INVEST_BASE_DATA_DIR, 'available_land_inputs', 'excess_salts_index_10s.tif')
    p.available_land_inputs['nutrient_availability_index'] = os.path.join(hb.GTAP_INVEST_BASE_DATA_DIR, 'available_land_inputs', 'nutrient_availability_index_10s.tif')
    p.available_land_inputs['TRI'] = os.path.join(hb.GTAP_INVEST_BASE_DATA_DIR, 'available_land_inputs', 'TRI_10s.tif')
    p.available_land_inputs['minutes_to_market'] = os.path.join(hb.GTAP_INVEST_BASE_DATA_DIR, 'available_land_inputs', 'minutes_to_market_10s.tif')
    p.available_land_inputs['caloric_yield'] = os.path.join(hb.GTAP_INVEST_BASE_DATA_DIR, 'available_land_inputs', 'caloric_yield_10s.tif')
    p.available_land_inputs['crop_suitability'] = os.path.join(hb.GTAP_INVEST_BASE_DATA_DIR, 'available_land_inputs', 'crop_suitability_10s.tif')
    # p.available_land_inputs['caloric_yield'] = r"C:\Users\jajohns\Files\Research\invest_crop_model\projects\global\intermediate\caloric_yield_total_all_filled.tif"
    if p.run_this:

        # p.unaligned_inputs = {}
        # p.unaligned_inputs['caloric_yield'] = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\available_land_inputs\caloric_yield_total_all_filled.tif"
        #
        # p.aligned_inputs = {}
        # p.aligned_inputs['caloric_yield'] = os.path.join(p.cur_dir, 'caloric_yield_10s.tif')
        #
        # p.resample_methods = {}
        # p.resample_methods['caloric_yield'] = 'bilinear'


        if False:

            parsed_iterable = []

            parsed_iterable = [(p.unaligned_inputs[k],
                                p.match_path,
                                p.aligned_inputs[k],
                                p.resample_methods[k],
                                7,
                                -9999.,
                                -9999.,
                                )
                                    for k, v in p.unaligned_inputs.items() if not hb.path_exists(p.aligned_inputs[k]) and not hb.path_exists(p.available_land_inputs[k])]
            L.info('Starting to resample from align_inputs')

            # TODOO Idea: have a function that wraps the above type dictionaries into another call that check which exists, creating a new iterable to run the missing. Further incorporate this with the hb.Path object.
            finished_results = []
            worker_pool = multiprocessing.Pool(p.num_workers)
            result = worker_pool.starmap_async(hb.resample_to_match, parsed_iterable)
            # result = worker_pool.starmap_async(hb.align_dataset_to_match, parsed_iterable)
            for i in result.get():
                finished_results.append(i)
            worker_pool.close()
            worker_pool.join()

        for k, v in p.available_land_inputs.items():
            hb.make_path_global_pyramid(v, verbose=False)


def country_summary_of_luc_in_ssps(p):
    """estimates of land use by country in a baseline year, and in future representative years, e.g. 2030 and 2050"""

    p.country_summary_of_luc_in_ssps_vector_path = os.path.join(p.cur_dir, 'country_summary_of_luc_in_ssps.gpkg')
    p.vertical_country_summary_of_luc_in_ssps_spreadsheet_path = os.path.join(p.cur_dir, 'vertical_country_summary_of_luc_in_ssps.xlsx')
    p.country_summary_of_luc_in_ssps_spreadsheet_path = os.path.join(p.cur_dir, 'country_summary_of_luc_in_ssps.xlsx')
    p.countries_shapefile_path = os.path.join(hb.BASE_DATA_DIR, "pyramids", "countries.shp")
    p.countries_raster_path =  os.path.join(hb.BASE_DATA_DIR, "pyramids", "country_ids_15m.tif")
    if p.run_this:

        # Add baseline class ha sums to df
        vertical_results_df = pd.DataFrame(columns=['SCENARIO', 'YEAR', 'VARIABLE', 'UNIT', 'value'])
        horizontal_results_df = None

        # for luh_scenario_label in p.luh_scenario_labels:
        #     p.gtap1_ha_total_paths[luh_scenario_label] = {}
        #     for scenario_year in p.scenario_years:
        #         p.gtap1_ha_total_paths[luh_scenario_label][scenario_year] = {}
        #         for policy_scenario_label in p.policy_scenario_labels:
        #             p.gtap1_ha_total_paths[luh_scenario_label][scenario_year][policy_scenario_label] = {}
        #             for class_label in p.class_labels:

        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:

                # NOTE: Remember that LUH scenario processing doesn't have a policy label.
                # for policy_scenario_label in p.policy_scenario_labels:
                L.info('    Processing scenario_year' + str(scenario_year))
                for c, class_name in enumerate(p.class_labels):

                    L.info('        Processing class ' + str(class_name))
                    baseline_tif = os.path.join(p.luh2_as_seals7_proportion_dir, luh_scenario_label, str(scenario_year), 'ha_total', str(c + 1) + '_' + class_name + '.tif')


                    results_odict = hb.zonal_statistics_flex(baseline_tif, p.gtap37_aez18_path,
                                                             zone_ids_raster_path=p.countries_raster_path, id_column_label='pyramid_id',
                                                             zones_ndv=None, values_ndv=None,
                                                             all_touched=None, assert_projections_same=True,
                                                             unique_zone_ids=None, csv_output_path=None, verbose=False)
                    # results_odict = hb.zonal_statistics_flex(baseline_tif, p.countries_shapefile_path, zone_ids_raster_path=p.countries_raster_path, id_column_label=None,
                    #                                          zones_ndv=None, values_ndv=None,
                    #                                          all_touched=None, assert_projections_same=True,
                    #                                          unique_zone_ids=None, csv_output_path=None, verbose=False)

                    d = OrderedDict()
                    d['SCENARIO'] = [str(luh_scenario_label)] * len(results_odict.values())
                    d['YEAR'] = [str(scenario_year)] * len(results_odict.values())
                    d['VARIABLE'] = [class_name + '_area'] * len(results_odict.values())
                    d['UNIT'] = ['hectares'] * len(results_odict.values())
                    d['value'] = [i['sum'] for i in results_odict.values()]
                    current_df = pd.DataFrame(data=d, index=results_odict.keys())

                    vertical_results_df = vertical_results_df.append(current_df, ignore_index=False, sort=False)

                    col_name = luh_scenario_label + '_' + str(scenario_year) + '_' + class_name + '_ha'
                    if horizontal_results_df is None:
                        horizontal_results_df = pd.DataFrame(data=[i['sum'] for i in results_odict.values()], index=results_odict.keys(), columns=[col_name])
                    else:
                        to_merge_df = pd.DataFrame(data=[i['sum'] for i in results_odict.values()], index=results_odict.keys(), columns=[col_name])
                        horizontal_results_df = pd.merge(horizontal_results_df, to_merge_df, how='outer', left_index=True, right_index=True)

        # Have to add the index back as a column to join on it.
        vertical_results_df['id'] = vertical_results_df.index
        horizontal_results_df['id'] = horizontal_results_df.index

        gdf = gpd.read_file(p.countries_shapefile_path)
        gdf = gdf[['id', 'iso3', 'admin', 'geometry']]

        gdf_names_only = gdf[['id', 'iso3', 'admin']]

        # Complex line to ensure all items in this col are the same type.
        gdf["geometry"] = [shapely.geometry.MultiPolygon([feature]) if type(feature) == shapely.geometry.Polygon else feature for feature in gdf["geometry"]]
        # to_drop = ['id', 'iso3', 'nev_name', 'fao_name', 'fao_id_c', 'gtap140', 'continent', 'region_un', 'region_wb', 'geom_index', 'abbrev', 'adm0_a3', 'adm0_a3_is', 'adm0_a3_un', 'adm0_a3_us', 'adm0_a3_wb', 'admin', 'brk_a3', 'brk_group', 'brk_name', 'country', 'disp_name', 'economy', 'fao_id', 'fao_reg', 'fips_10_', 'formal_en', 'formal_fr', 'gau', 'gdp_md_est', 'gdp_year', 'geounit', 'gu_a3', 'income_grp', 'iso', 'iso2_cull', 'iso3_cull', 'iso_3digit', 'iso_a2', 'iso_a3', 'iso_a3_eh', 'iso_n3', 'lastcensus', 'name', 'name_alt', 'name_ar', 'name_bn', 'name_cap', 'name_ciawf', 'name_de', 'name_el', 'name_en', 'name_es', 'name_fr', 'name_hi', 'name_hu', 'name_id', 'name_it', 'name_ja', 'name_ko', 'name_long', 'name_nl', 'name_pl', 'name_pt', 'name_ru', 'name_sort', 'name_sv', 'name_tr', 'name_vi', 'name_zh', 'ne_id', 'nev_lname', 'nev_sname', 'note_adm0', 'note_brk', 'official', 'olympic', 'pop_est', 'pop_rank', 'pop_year', 'postal', 'sov_a3', 'sovereignt', 'su_a3', 'subregion', 'subunit', 'type', 'un_a3', 'un_iso_n', 'un_vehicle', 'undp', 'uni', 'wb_a2', 'wb_a3', 'wiki1', 'wikidataid', 'wikipedia', 'woe_id', 'woe_id_eh', 'woe_note']
        # Merge results into vector file
        gdf = gdf.merge(horizontal_results_df, on='id')
        vertical_results_df = vertical_results_df.merge(gdf_names_only, on='id')

        # NOTE FUN USAGE OF GPKG, much faster than shapefiles and is open standard
        gdf.to_file(p.country_summary_of_luc_in_ssps_vector_path, driver='GPKG')

        # Also save an xlsx
        df = gdf.drop('geometry', axis=1)
        df.to_excel(p.country_summary_of_luc_in_ssps_spreadsheet_path)

        vertical_results_df['id'] = vertical_results_df.index

        # Load GDP and Pop
        aux_data_df = pd.read_excel(p.country_data_xlsx_path)
        vertical_results_df = vertical_results_df.append(aux_data_df, ignore_index=False, sort=False)

        vertical_results_df.to_excel(p.vertical_country_summary_of_luc_in_ssps_spreadsheet_path)


def gridded_change_vectors(p):

    if p.run_this:
        gdf = gpd.read_file(p.country_summary_of_luc_in_ssps_vector_path)


        gdf.replace(0.0, 100.0, True)
        gdf.replace(0, 100.0, True)
        gdf.fillna(100.0, inplace=True)

        for scenario in p.policy_scenario_labels[1:]:  # SKIP Observed

            for year in ['2030', '2050']:

                for class_name in p.class_labels:
                    # The following plot uses a hacky approach to specifying the vmin vmax TWICE rather than just sharing an AX. this is beause I coulndt figure out how to overwrite geopandas ax.
                    fig, ax = plt.subplots(1, 1)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="2%", pad=.1)

                    # rcp26_ssp1_2030_cropland_pct_change
                    col_name = scenario + '_' + str(year) + '_' + class_name + '_pct_change'
                    title_name = col_name.replace('_', ' ').replace('pct', 'percent')
                    title_name = title_name[:4] + '.' + title_name[4:]

                    vmin, vmax = 40, 160
                    # vmin, vmax = 0, np.max(gdf[col_name])
                    gdf.plot(column=col_name, cmap='BrBG', ax=ax, vmin=vmin, vmax=vmax)

                    # sm = plt.cm.ScalarMappable(cmap='Spectral_r', norm=colors.LogNorm(vmin=vmin+1, vmax=vmax)) # CANNOT DO THIS without also adjusting the vmin plot noralization above.
                    sm = plt.cm.ScalarMappable(cmap='BrBG', norm=colors.Normalize(vmin=vmin + 1, vmax=vmax))
                    sm._A = []  # fake up the array of the scalar mappable. Urgh...
                    cbar = fig.colorbar(sm, cax=cax)

                    ax.set_title(title_name, fontdict={'fontsize': '10', 'fontweight': '3'})
                    cbar.set_label('percent', fontdict={'fontsize': '7', 'fontweight': '1'})
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.axis('off')
                    gdf.geometry.boundary.plot(color=None, edgecolor='k', linewidth=.05, ax=ax)
                    # plt.show()
                    current_path = os.path.join(p.cur_dir, col_name + '.png')
                    plt.savefig(current_path, dpi=400)


biophysical_processing_steps = 'starts here'

def land_cover_change_analysis(p):


    hb.timer()
    p.land_cover_change_analysis_csv_path = os.path.join(p.cur_dir, 'land_cover_change_analysis.csv')

    if p.run_this:
        do_esa_enumeration = 0
        if do_esa_enumeration:
            df = None
            enumeration_classes = hb.esacci_extended_classes
            policy_scenario_label = 'gtap1_baseline_2014'
            current_csv_path = os.path.join(p.cur_dir, 'land_cover_change_' + policy_scenario_label + '.csv')


            # current_raster_zone_ids_path = p.zone_ids_raster_path
            current_raster_zone_ids_path = os.path.join(p.cur_dir, 'zone_ids.tif')

            if not hb.path_exists(current_csv_path):
                df = hb.zonal_statistics_flex(p.base_year_lulc_path,
                                              p.gtap37_aez18_path,
                                              zone_ids_raster_path=current_raster_zone_ids_path,
                                              id_column_label='pyramid_id',
                                              zones_raster_data_type=5,
                                              values_raster_data_type=6,
                                              zones_ndv=-9999,
                                              values_ndv=-9999,
                                              all_touched=None,
                                              assert_projections_same=False,
                                              stats_to_retrieve='enumeration',
                                              enumeration_classes=enumeration_classes,
                                              multiply_raster_path=p.ha_per_cell_column_10sec_path) #multiply_raster_path=p.ha_per_cell_10sec_pathmultiply_raster_path=p.ha_per_cell_10sec_path

                df.to_csv(current_csv_path)
            else:
                df = pd.read_csv(current_csv_path, index_col=0)

            # These were wrong because note below.
            # rename_dict = {k: policy_scenario_label + '_' + v for k, v in hb.esacci_extended_short_class_descriptions.items()}
            # rename_dict = {str(k): policy_scenario_label + '_' + v for k, v in hb.esacci_extended_short_class_descriptions.items()}

            final_vertical_cols = []
            # NOTE: Awkward omission in zonal_stats is that instead of the ESACCI class ids, it is just 0-37, so need to remap prior to remap.
            rename_dict = {str(c): policy_scenario_label + '_' + hb.esacci_extended_short_class_descriptions[i] for c, i in enumerate(hb.esacci_extended_short_class_descriptions.keys())}
            full_rename_dict_pre = {policy_scenario_label + '_' + hb.esacci_extended_short_class_descriptions[i]: policy_scenario_label + '_' + hb.esacci_to_habitat_quality_simplified_correspondence[i][2] for c, i in enumerate(hb.esacci_extended_short_class_descriptions.keys())}
            full_rename_dict = full_rename_dict_pre # for later use of reclassing.
            df.rename(columns=rename_dict, inplace=True)

            # Save these for eventual data stacking/unstacking to get into vertical format.
            final_vertical_cols += list(rename_dict.values())

            for luh_scenario_label in p.luh_scenario_labels:
                for scenario_year in p.scenario_years:
                    for policy_scenario_label in p.policy_scenario_labels:
                        include_string = 'lulc_esa_gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        luc_scenario_path = os.path.join(p.intermediate_dir, 'map_esa_simplified_back_to_esa', include_string + '.tif')

                        current_csv_path = os.path.join(p.cur_dir, 'land_cover_change_' + policy_scenario_label + '.csv')
                        if not hb.path_exists(current_csv_path):
                            current_df = hb.zonal_statistics_flex(luc_scenario_path,
                                                          p.gtap37_aez18_path,
                                                          zone_ids_raster_path=current_raster_zone_ids_path,
                                                          id_column_label='pyramid_id',
                                                          zones_raster_data_type=5,
                                                          values_raster_data_type=6,
                                                          zones_ndv=-9999,
                                                          values_ndv=-9999,
                                                          all_touched=None,
                                                          assert_projections_same=False,
                                                                  stats_to_retrieve='enumeration',
                                                          enumeration_classes=enumeration_classes,
                                                          multiply_raster_path=p.ha_per_cell_column_10sec_path)


                            current_df.to_csv(current_csv_path)

                        else:
                            current_df = pd.read_csv(current_csv_path, index_col=0)

                        scenario_vertical_name = '2021_2030_'+ policy_scenario_label + '_noES'
                        # rename_dict = {str(k): policy_scenario_label + '_' + v for k, v in hb.esacci_extended_short_class_descriptions.items()}
                        rename_dict = {str(c): scenario_vertical_name + '_' + hb.esacci_extended_short_class_descriptions[i] for c, i in enumerate(hb.esacci_extended_short_class_descriptions.keys())}
                        full_rename_dict_pre = {scenario_vertical_name + '_' + hb.esacci_extended_short_class_descriptions[i]: scenario_vertical_name + '_' + hb.esacci_to_habitat_quality_simplified_correspondence[i][2] for c, i in enumerate(hb.esacci_extended_short_class_descriptions.keys())}


                        full_rename_dict.update(full_rename_dict_pre)
                        current_df.rename(columns=rename_dict, inplace=True)

                        # Save these for eventual data stacking/unstacking to get into vertical format.
                        final_vertical_cols += list(rename_dict.values())

                        df = pd.merge(df, current_df, how='outer', left_index=True, right_index=True)

            df.to_csv(hb.suri(p.land_cover_change_analysis_csv_path, 'pre'))

            full_gdf = gpd.read_file(p.full_projection_results_vector_path)
            index_cols_to_include = ['pyramid_id', 'pyramid_ids_concatenated', 'pyramid_ids_multiplied', 'gtap37v10_pyramid_id', 'aez_pyramid_id', 'gtap37v10_pyramid_name', 'ISO3', 'AZREG', 'AEZ_COMM', 'gtap9_admin_id', 'GTAP226', 'GTAP140v9a', 'GTAPv9a', 'GTAP141v9p', 'GTAPv9p', 'AreaCode', 'gtap9_admin_name', 'bb', 'minx', 'miny', 'maxx', 'maxy']
            output_df = pd.merge(full_gdf[index_cols_to_include], df, how='outer', left_on='pyramid_id', right_index=True)

            output_df.to_csv(p.land_cover_change_analysis_csv_path)
            output_df.set_index(['gtap37v10_pyramid_name', 'aez_pyramid_id'], inplace=True)
            # output_vertical_df = output_df.stack()
            index_cols_to_include = ['pyramid_id', 'pyramid_ids_concatenated', 'pyramid_ids_multiplied', 'gtap37v10_pyramid_id', 'ISO3', 'AZREG', 'AEZ_COMM', 'gtap9_admin_id', 'GTAP226', 'GTAP140v9a', 'GTAPv9a', 'GTAP141v9p', 'GTAPv9p', 'AreaCode', 'gtap9_admin_name', 'bb', 'minx', 'miny', 'maxx', 'maxy']
            index_cols_to_include = ['pyramid_id', 'pyramid_ids_concatenated']

            output_vertical_df = output_df.pivot_table(index=['gtap37v10_pyramid_name', 'aez_pyramid_id'], columns=[], values=index_cols_to_include + final_vertical_cols)

            output_vertical_df = output_vertical_df.stack()
            output_vertical_df.index.rename(['REG', 'AEZ_COMM', 'SCENARIO'], inplace=True)
            output_vertical_df = output_vertical_df.to_frame(name='Value')
            # output_vertical_df = output_vertical_df[final_vertical_cols]

            output_df.dropna(inplace=True)

            # output_vertical_df = output_vertical_df.merge(output_df[[i for i in output_df.columns if i not in output_vertical_df.columns]], left_index=True, right_index=True, how='outer')


            output_vertical_df.to_csv(hb.suri(p.land_cover_change_analysis_csv_path, 'vertical'))

            grouped_df = output_vertical_df.groupby(['REG', 'SCENARIO']).sum()
            grouped_df.to_csv(hb.suri(p.land_cover_change_analysis_csv_path, 'by_region'))
            rename_dict = hb.esacci_extended_short_class_descriptions


            output_vertical_df['SCENARIO_col'] = output_vertical_df.index.get_level_values(2)
            grouped_df = output_vertical_df.replace({'SCENARIO_col': full_rename_dict})
            # grouped_df = output_vertical_df.rename(columns=full_rename_dict)
            grouped_df = grouped_df.groupby(['REG', 'SCENARIO_col']).sum()
            grouped_df.to_csv(hb.suri(p.land_cover_change_analysis_csv_path, 'by_region_few_classes'))


        #################################
        ### Now dow it for seals7 and at the COUNTRY level
        ####################

        # # HACK SHORTCUT: Only need to do it for BAU per raffaelo's instructions
        # p.policy_scenario_labels = ['BAU'] # LOLOOLOL


        countries_vector_path = r"C:\Users\jajohns\Files\Research\base_data\pyramids\countries_iso3.gpkg"
        # ha_per_cell_path = r"C:\Users\jajohns\Files\Research\base_data\pyramids\ha_per_cell_10sec.tif"
        # ha_per_cell_column_path = r"C:\Users\jajohns\Files\Research\base_data\pyramids\ha_per_cell_column_10sec.tif"
        country_ids_path = r"C:\Users\jajohns\Files\Research\base_data\pyramids\country_ids_10sec.tif"

        df = None
        enumeration_classes = list(hb.seals_simplified_labels.keys())
        enumeration_labels = list(hb.seals_simplified_labels.values())
        policy_scenario_label = 'gtap1_baseline_2014'
        current_csv_path = os.path.join(p.cur_dir, 'land_cover_change_SEALS5_' + policy_scenario_label + '.csv')


        # current_raster_zone_ids_path = p.zone_ids_raster_path
        current_raster_zone_ids_path = os.path.join(p.cur_dir, 'zone_ids.tif')

        if not hb.path_exists(current_csv_path):
            unique_zone_ids = np.asarray(list(range(256)), dtype=np.int64) # Note the + 1
            df = hb.zonal_statistics_flex(p.base_year_simplified_lulc_path,
                                          countries_vector_path,
                                          zone_ids_raster_path=country_ids_path,
                                          id_column_label='id',
                                          zones_raster_data_type=5,
                                          values_raster_data_type=6,
                                          zones_ndv=-9999,
                                          values_ndv=-9999,
                                          all_touched=None,
                                          unique_zone_ids=unique_zone_ids,
                                          assert_projections_same=False,
                                          vector_columns_to_include_in_output=['iso3'],
                                          vector_index_column='id',
                                          stats_to_retrieve='enumeration',
                                          enumeration_classes=enumeration_classes,
                                          enumeration_labels=enumeration_labels,
                                          multiply_raster_path=p.ha_per_cell_10sec_path) #multiply_raster_path=p.ha_per_cell_10sec_pathmultiply_raster_path=p.ha_per_cell_10sec_path

            df.to_csv(current_csv_path, index=False)
        else:
            df = pd.read_csv(current_csv_path, index_col=0)

        # These were wrong because note below.
        # rename_dict = {k: policy_scenario_label + '_' + v for k, v in hb.esacci_extended_short_class_descriptions.items()}
        # rename_dict = {str(k): policy_scenario_label + '_' + v for k, v in hb.esacci_extended_short_class_descriptions.items()}

        scenario_vertical_name = '2021_2030_' + policy_scenario_label + '_noES'
        # rename_dict = {str(k): policy_scenario_label + '_' + v for k, v in hb.esacci_extended_short_class_descriptions.items()}
        rename_dict = {enumeration_labels[k]: policy_scenario_label + '_' + v for k, v in hb.seals_simplified_labels.items()}
        df.rename(columns=rename_dict, inplace=True)

        # Save these for eventual data stacking/unstacking to get into vertical format.
        final_vertical_cols = []
        final_vertical_cols += list(rename_dict.values())

        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    include_string = 'lulc_seals7_gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                    luc_scenario_path = os.path.join(p.intermediate_dir, 'stitched_lulcs', include_string + '.tif')

                    current_csv_path = os.path.join(p.cur_dir, 'land_cover_change_SEALS5_' + policy_scenario_label + '.csv')
                    if not hb.path_exists(current_csv_path):
                        unique_zone_ids = np.asarray(list(range(256)), dtype=np.int64)
                        current_df = hb.zonal_statistics_flex(luc_scenario_path,
                                                      countries_vector_path,
                                                      zone_ids_raster_path=country_ids_path,
                                                      id_column_label='id',
                                                      zones_raster_data_type=5,
                                                      values_raster_data_type=6,
                                                      zones_ndv=-9999,
                                                      values_ndv=-9999,
                                                      all_touched=None,
                                                      unique_zone_ids=unique_zone_ids,
                                                      assert_projections_same=False,
                                                      vector_columns_to_include_in_output=['iso3'],
                                                      vector_index_column='id',
                                                      stats_to_retrieve='enumeration',
                                                      enumeration_classes=enumeration_classes,
                                                      enumeration_labels=enumeration_labels,
                                                      multiply_raster_path=p.ha_per_cell_column_10sec_path)


                        current_df.to_csv(current_csv_path, index=False)

                    else:
                        current_df = pd.read_csv(current_csv_path, index_col=0)


                    scenario_vertical_name = '2021_2030_'+ policy_scenario_label + '_noES'
                    # rename_dict = {str(k): policy_scenario_label + '_' + v for k, v in hb.esacci_extended_short_class_descriptions.items()}
                    rename_dict = {enumeration_labels[k]: policy_scenario_label + '_' + v for k, v in hb.seals_simplified_labels.items()}
                    current_df.rename(columns=rename_dict, inplace=True)

                    # Save these for eventual data stacking/unstacking to get into vertical format.
                    final_vertical_cols += list(rename_dict.values())

                    df = hb.df_merge(df, current_df, how='outer', left_on='id', right_on='id')
                    # df = pd.merge(df, current_df, how='outer', left_on='id', right_on='id')

        df.to_csv(hb.suri(p.land_cover_change_analysis_csv_path, 'by_country'), index=False)

        # full_gdf = gpd.read_file(p.full_projection_results_vector_path)
        # index_cols_to_include = ['pyramid_id', 'pyramid_ids_concatenated', 'pyramid_ids_multiplied', 'gtap37v10_pyramid_id', 'aez_pyramid_id', 'gtap37v10_pyramid_name', 'ISO3', 'AZREG', 'AEZ_COMM', 'gtap9_admin_id', 'GTAP226', 'GTAP140v9a', 'GTAPv9a', 'GTAP141v9p', 'GTAPv9p', 'AreaCode', 'gtap9_admin_name', 'bb', 'minx', 'miny', 'maxx', 'maxy']
        # output_df = pd.merge(full_gdf[index_cols_to_include], df, how='outer', left_on='pyramid_id', right_index=True)
        #
        # output_df.to_csv(p.land_cover_change_analysis_csv_path, index=False)
        # output_df.set_index(['gtap37v10_pyramid_name', 'aez_pyramid_id'], inplace=True)
        # # output_vertical_df = output_df.stack()
        # index_cols_to_include = ['pyramid_id', 'pyramid_ids_concatenated', 'pyramid_ids_multiplied', 'gtap37v10_pyramid_id', 'ISO3', 'AZREG', 'AEZ_COMM', 'gtap9_admin_id', 'GTAP226', 'GTAP140v9a', 'GTAPv9a', 'GTAP141v9p', 'GTAPv9p', 'AreaCode', 'gtap9_admin_name', 'bb', 'minx', 'miny', 'maxx', 'maxy']
        # index_cols_to_include = ['pyramid_id', 'pyramid_ids_concatenated']
        #
        # output_vertical_df = output_df.pivot_table(index=['gtap37v10_pyramid_name', 'aez_pyramid_id'], columns=[], values=index_cols_to_include + final_vertical_cols)
        #
        # output_vertical_df = output_vertical_df.stack()
        # output_vertical_df.index.rename(['REG', 'AEZ_COMM', 'SCENARIO'], inplace=True)
        # output_vertical_df = output_vertical_df.to_frame(name='Value')
        # # output_vertical_df = output_vertical_df[final_vertical_cols]
        #
        # output_df.dropna(inplace=True)
        #
        # # output_vertical_df = output_vertical_df.merge(output_df[[i for i in output_df.columns if i not in output_vertical_df.columns]], left_index=True, right_index=True, how='outer')
        #
        #
        # output_vertical_df.to_csv(hb.suri(p.land_cover_change_analysis_csv_path, 'vertical'))
        #
        # grouped_df = output_vertical_df.groupby(['REG', 'SCENARIO']).sum()
        # grouped_df.to_csv(hb.suri(p.land_cover_change_analysis_csv_path, 'by_region'))
        # rename_dict = hb.esacci_extended_short_class_descriptions
        #
        #
        # output_vertical_df['SCENARIO_col'] = output_vertical_df.index.get_level_values(2)
        # grouped_df = output_vertical_df.replace({'SCENARIO_col': full_rename_dict})
        # # grouped_df = output_vertical_df.rename(columns=full_rename_dict)
        # grouped_df = grouped_df.groupby(['REG', 'SCENARIO_col']).sum()
        # grouped_df.to_csv(hb.suri(p.land_cover_change_analysis_csv_path, 'by_region_few_classes'))



        # LEARNING POINT TypeError: '<' not supported between instances of 'float' and 'str' arises when you have na values. simply drop them then merge.



def carbon_biophysical(p):
    """Calculate carbon storage present list of LULC maps."""

    # def complete():
    #     # TODOO Implement a logic like this that has every task have a complete method to test if it exists rather than just dir exists.
    #     if all([hb.path_exists(os.path.join(p.cur_dir, 'carbon_' + policy_scenario_label + '.tif')) for i in p.scenario.keys()]):
    #         return True
    #     else:
    #         return False

    # # SHORTCUT HACK
    # p.policy_scenario_labels = ['BAU']

    p.exhaustive_carbon_table_path = os.path.join(p.model_base_data_dir, "carbon_storage", "exhaustive_carbon_table.csv")
    # p.exhaustive_carbon_table_path = r"C:\Users\jajohns\Files\Research\carbon_unilever\intermediate\join_carbon_tables_to_rasterized_zone_ids\exhaustive_carbon_table.csv"

    p.carbon_zones_path = os.path.join(p.model_base_data_dir, 'carbon_storage', 'carbon_zones_rasterized.tif')
    p.joined_carbon_table_stacked_with_seals_simplified_classes_path = os.path.join(p.cur_dir, 'joined_carbon_table_stacked_with_seals_simplified_classes.csv')


    if p.run_this:
        p.match_floats_path_clipped = os.path.join(p.cur_dir, 'match_floats.tif')
        p.soyo_bb = [12.050412203, -6.229749114, 13.030121546, -5.557749114]  # Interesting area with mangroves, a city and some terrain\

        # First run baseline for reference
        policy_scenario_label = 'gtap1_baseline_' + str(p.base_year)
        # p.baseline_lulc_paths[policy_scenario_label] = os.path.join(hb.SEALS_BASE_DATA_DIR, 'lulc_esa', 'full', 'ESACCI-LC-L4-LCCS-Map-300m-P1Y-' + str(p.base_year) + '-v2.0.7.tif')

        raster_to_check_bb_path = hb.list_filtered_paths_nonrecursively(p.stitched_lulc_esa_scenarios, include_extensions='.tif')[0]

        stitched_bb = hb.get_bounding_box(raster_to_check_bb_path)


        global_bb = hb.get_bounding_box(p.base_year_lulc_path)
        if stitched_bb != global_bb:
            current_carbon_zones_path = os.path.join(p.cur_dir, 'carbon_zones.tif')
            hb.clip_raster_by_bb(p.carbon_zones_path, stitched_bb, current_carbon_zones_path)

            current_ha_per_cell_path = os.path.join(p.cur_dir, 'ha_per_cell_10sec.tif')
            hb.clip_raster_by_bb(p.ha_per_cell_10sec_path, stitched_bb, current_ha_per_cell_path)

            luc_scenario_path = os.path.join(p.cur_dir, 'lulc_clipped.tif')
            hb.clip_raster_by_bb(p.base_year_lulc_path, stitched_bb, luc_scenario_path)


        else:
            current_carbon_zones_path = p.carbon_zones_path
            current_ha_per_cell_path = p.ha_per_cell_10sec_path
            luc_scenario_path = p.base_year_lulc_path


        baseline_carbon_Mg_per_ha_output_path = os.path.join(p.cur_dir, 'carbon_Mg_per_ha_' + policy_scenario_label + '.tif')
        if not hb.path_exists(baseline_carbon_Mg_per_ha_output_path):
            L.info('Running global_invest_main.carbon_global on LULC: ' + str(luc_scenario_path) + ' and saving results to ' + str(baseline_carbon_Mg_per_ha_output_path))
            global_invest_main.carbon(luc_scenario_path, current_carbon_zones_path, p.exhaustive_carbon_table_path, baseline_carbon_Mg_per_ha_output_path)
        baseline_carbon_per_cell_output_path = os.path.join(p.cur_dir, 'carbon_Mg_per_cell_10s_' + policy_scenario_label + '.tif')
        if not hb.path_exists(baseline_carbon_per_cell_output_path):
            hb.raster_calculator_af_flex([baseline_carbon_Mg_per_ha_output_path, current_ha_per_cell_path], lambda x, y: np.where((x >= 0) & (y >= 0), x * y, -9999.), baseline_carbon_per_cell_output_path)


        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:


                for policy_scenario_label in p.policy_scenario_labels:
                    include_string = 'lulc_esa_gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                    luc_scenario_path = os.path.join(p.intermediate_dir, 'map_esa_simplified_back_to_esa', include_string + '.tif')

                    # TODOO Make everything in 3 nested dirs.
                    carbon_Mg_per_ha_output_path = os.path.join(p.cur_dir, 'carbon_Mg_per_ha_' + policy_scenario_label + '.tif')

                    # TODOO Augment project_flow to replace hb.path_exists with project_flow.is_done()
                    if not hb.path_exists(carbon_Mg_per_ha_output_path):
                        L.info('Running global_invest_main.carbon_global on LULC: ' + str(luc_scenario_path) + ' and saving results to ' + str(carbon_Mg_per_ha_output_path))
                        global_invest_main.carbon(luc_scenario_path, current_carbon_zones_path, p.exhaustive_carbon_table_path, carbon_Mg_per_ha_output_path)

                    carbon_Mg_per_cell_10s_path = os.path.join(p.cur_dir, 'carbon_Mg_per_cell_10s_' + policy_scenario_label + '.tif')
                    if not hb.path_exists(carbon_Mg_per_cell_10s_path):
                        L.info('Converting to per-cell and saving results to ' + str(carbon_Mg_per_cell_10s_path))
                        hb.raster_calculator_af_flex([carbon_Mg_per_ha_output_path, current_ha_per_cell_path], lambda x, y: np.where((x >= 0) & (y >= 0), x * y, -9999.), carbon_Mg_per_cell_10s_path)

                    if p.build_overviews_and_stats:
                        if not hb.path_exists(carbon_Mg_per_ha_output_path + '.ovr'):
                            hb.add_overviews_to_path(carbon_Mg_per_ha_output_path, specific_overviews_to_add=[30], overview_resampling_algorithm='average')
                            hb.calculate_raster_stats(carbon_Mg_per_ha_output_path)

                    if p.build_overviews_and_stats:
                        if not hb.path_exists(carbon_Mg_per_cell_10s_path + '.ovr'):
                            hb.add_overviews_to_path(carbon_Mg_per_cell_10s_path, specific_overviews_to_add=[30], overview_resampling_algorithm='average')
                            hb.calculate_raster_stats(carbon_Mg_per_cell_10s_path)

                    carbon_change_path = os.path.join(p.cur_dir, 'carbon_change_' + policy_scenario_label + '.tif')
                    if not hb.path_exists(carbon_change_path):
                        hb.raster_calculator_af_flex([carbon_Mg_per_cell_10s_path, baseline_carbon_Mg_per_ha_output_path], lambda x, y: np.where((x >= 0) & (y >= 0), x - y, -9999.), carbon_change_path)
                    if p.build_overviews_and_stats:
                        if not hb.path_exists(carbon_change_path + '.ovr'):
                            # hb.add_overviews_to_path(carbon_change_path, overview_resampling_algorithm='average')
                            hb.add_overviews_to_path(carbon_change_path, specific_overviews_to_add=[30], overview_resampling_algorithm='average')
                            hb.calculate_raster_stats(carbon_change_path)


def carbon_shock(p):
    """Convert carbon storage map into shockfile, which is basically just the percentage changes."""

    p.carbon_shock_csv_path = os.path.join(p.cur_dir, 'carbon_shock.csv')
    if p.run_this:


        df = None

        # CRITICAL TODOO Last run had some border-irregularities which caused this zone_ids to be incorrect, per cur_task_zone_ids_raster_path. For next run, and already, just to be sure, I've updated it with the correct file.

        # cur_task_zone_ids_raster_path = os.path.join(p.cur_dir, 'zone_ids.tif')
        # TODOO. just a thought, but i'm not sure my testing paradigm can be made scale agnostic becasue of the complexity added by having to have paths used in raster-calculator and
        # especially zonal stats, which would have to generate a new zone-ids raster for each bb. Perhaps have cached ones? but the obvious improvement would be to have it be
        # able to read cr_width_heights and then use something like DASK! YES!!!
        cur_task_zone_ids_raster_path = p.zone_ids_raster_path
        # cur_task_zone_ids_raster_path = p.zone_ids_raster_path
        value_names = []
        baseline_scenario_label = 'gtap1_baseline_2014'

        value_names.append(baseline_scenario_label + '_sum')
        current_csv_path = os.path.join(p.cur_dir, 'carbon_stats_' + baseline_scenario_label + '.csv')
        if not hb.path_exists(current_csv_path):
            # BASELINE:

            carbon_mg_ha_path = os.path.join(p.carbon_biophysical_dir, 'carbon_Mg_per_cell_10s_' + baseline_scenario_label + '.tif')
            df = hb.zonal_statistics_flex(carbon_mg_ha_path,
                            p.gtap37_aez18_path,
                            zone_ids_raster_path=cur_task_zone_ids_raster_path,
                            id_column_label='pyramid_id',
                            zones_raster_data_type=5,
                            values_raster_data_type=6,
                            zones_ndv=-9999,
                            values_ndv=-9999,
                            all_touched=None,
                            assert_projections_same=False, )
            df.rename(columns={'sums': baseline_scenario_label + '_sum'}, inplace=True)
            df.to_csv(current_csv_path)
        else:
            df = pd.read_csv(current_csv_path, index_col=0)

        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    current_csv_path = os.path.join(p.cur_dir, 'carbon_zonal_stats_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '.csv')
                    if not hb.path_exists(current_csv_path):
                        carbon_mg_ha_path = os.path.join(p.carbon_biophysical_dir, 'carbon_Mg_per_cell_10s_' + policy_scenario_label + '.tif')
                        current_df = hb.zonal_statistics_flex(carbon_mg_ha_path,
                                                             p.gtap37_aez18_path,
                                                             zone_ids_raster_path=cur_task_zone_ids_raster_path,
                                                             id_column_label='pyramid_id',
                                                              zones_raster_data_type=5,
                                                              values_raster_data_type=6,
                                                              zones_ndv=-9999,
                                                              values_ndv=-9999,
                                                             all_touched=None,
                                                             assert_projections_same=False,
                                                              verbose=True)
                        'gtap2_rcp45_ssp2_2030_bau_carbon_storage_total_ha'
                        current_df.rename(columns={'sums': 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_sum'}, inplace=True)
                        value_names.append('gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_sum')
                        current_df.to_csv(current_csv_path)
                    else:
                        current_df = pd.read_csv(current_csv_path, index_col=0)
                    df = pd.merge(df, current_df, how='outer', left_index=True, right_index=True)


        full_gdf = gpd.read_file(p.full_projection_results_vector_path)
        cols_to_include = ['pyramid_id', 'pyramid_ids_concatenated', 'pyramid_ids_multiplied', 'gtap37v10_pyramid_id', 'aez_pyramid_id', 'gtap37v10_pyramid_name', 'ISO3', 'AZREG', 'AEZ_COMM', 'gtap9_admin_id', 'GTAP226', 'GTAP140v9a', 'GTAPv9a', 'GTAP141v9p', 'GTAPv9p', 'AreaCode', 'gtap9_admin_name', 'bb', 'minx', 'miny', 'maxx', 'maxy']
        output_df = pd.merge(full_gdf[cols_to_include], df, how='outer', left_on='pyramid_id', right_index=True)

        output_df.to_csv(p.carbon_shock_csv_path)

        output_df_grouped = output_df.groupby('gtap37v10_pyramid_name').agg('sum')
        output_df_grouped.to_csv(hb.suri(p.carbon_shock_csv_path, 'region'))

        output_df_grouped_vertical = output_df.pivot_table(index=['gtap37v10_pyramid_name', 'aez_pyramid_id'], values=value_names)
        output_df_grouped_vertical.to_csv(hb.suri(p.carbon_shock_csv_path, 'vertical'))

        # Calculate NPV (ported code from scripts directory)

        output_df_grouped

        #################################
        ### Now dow it for seals7 and at the COUNTRY level
        ####################

        # Also, optionally, calculate the carbon change PER COUNTRY (rather than per GTAP region)
        calculate_by_country = 1
        if calculate_by_country:
            #
            # # HACK SHORTCUT: Only need to do it for BAU per raffaelo's instructions
            # p.policy_scenario_labels = ['BAU']  # LOLOOLOL

            countries_vector_path = r"C:\Users\jajohns\Files\Research\base_data\pyramids\countries_iso3.gpkg"
            # ha_per_cell_path = r"C:\Users\jajohns\Files\Research\base_data\pyramids\ha_per_cell_10sec.tif"
            # ha_per_cell_column_path = r"C:\Users\jajohns\Files\Research\base_data\pyramids\ha_per_cell_column_10sec.tif"
            country_ids_path = r"C:\Users\jajohns\Files\Research\base_data\pyramids\country_ids_10sec.tif"

            df = None
            enumeration_classes = list(hb.seals_simplified_labels.keys())
            policy_scenario_label = 'gtap1_baseline_2014'
            current_csv_path = os.path.join(p.cur_dir, 'carbon_by_country_' + policy_scenario_label + '.csv')

            base_year_carbon_path = os.path.join(p.carbon_biophysical_dir, 'carbon_Mg_per_cell_10s_' + baseline_scenario_label + '.tif')

            if not hb.path_exists(current_csv_path):
                unique_zone_ids = np.asarray(list(range(256)), dtype=np.int64)  # Note the + 1
                df = hb.zonal_statistics_flex(base_year_carbon_path,
                                              countries_vector_path,
                                              zone_ids_raster_path=country_ids_path,
                                              id_column_label='id',
                                              zones_raster_data_type=5,
                                              values_raster_data_type=6,
                                              zones_ndv=-9999,
                                              values_ndv=-9999,
                                              all_touched=None,
                                              unique_zone_ids=unique_zone_ids,
                                              assert_projections_same=False,
                                              output_column_prefix=policy_scenario_label,
                                              vector_columns_to_include_in_output=['iso3'],
                                              vector_index_column='id',
                                              stats_to_retrieve='sums',)

                df.to_csv(current_csv_path, index=False)
            else:
                df = pd.read_csv(current_csv_path, index_col=0)

            # These were wrong because note below.
            # rename_dict = {k: policy_scenario_label + '_' + v for k, v in hb.esacci_extended_short_class_descriptions.items()}
            # rename_dict = {str(k): policy_scenario_label + '_' + v for k, v in hb.esacci_extended_short_class_descriptions.items()}

            # final_vertical_cols = []
            # # NOTE: Awkward omission in zonal_stats is that instead of the ESACCI class ids, it is just 0-37, so need to remap prior to remap.
            # rename_dict = {str(c): policy_scenario_label + '_' + hb.esacci_extended_short_class_descriptions[i] for c, i in enumerate(hb.esacci_extended_short_class_descriptions.keys())}
            # full_rename_dict_pre = {policy_scenario_label + '_' + hb.esacci_extended_short_class_descriptions[i]: policy_scenario_label + '_' + hb.esacci_to_habitat_quality_simplified_correspondence[i][2] for c, i in enumerate(hb.esacci_extended_short_class_descriptions.keys())}
            # full_rename_dict = full_rename_dict_pre  # for later use of reclassing.
            # df.rename(columns=rename_dict, inplace=True)
            #
            # # Save these for eventual data stacking/unstacking to get into vertical format.
            # final_vertical_cols += list(rename_dict.values())

            for luh_scenario_label in p.luh_scenario_labels:
                for scenario_year in p.scenario_years:
                    for policy_scenario_label in p.policy_scenario_labels:
                        scenario_carbon_path = os.path.join(p.carbon_biophysical_dir, 'carbon_Mg_per_cell_10s_' + policy_scenario_label + '.tif')


                        current_csv_path = os.path.join(p.cur_dir, 'carbon_by_country_' + policy_scenario_label + '.csv')
                        if not hb.path_exists(current_csv_path):
                            # current_df = hb.zonal_statistics_flex(scenario_carbon_path,
                            #                                       p.gtap37_aez18_path,
                            #                                       zone_ids_raster_path=country_ids_path,
                            #                                       id_column_label='pyramid_id',
                            #                                       zones_raster_data_type=5,
                            #                                       values_raster_data_type=6,
                            #                                       zones_ndv=-9999,
                            #                                       values_ndv=-9999,
                            #                                       all_touched=None,
                            #                                       assert_projections_same=False,
                            #                                       stats_to_retrieve='sums')
                            unique_zone_ids = np.asarray(list(range(256)), dtype=np.int64)
                            current_df = hb.zonal_statistics_flex(scenario_carbon_path,
                                                          countries_vector_path,
                                                          zone_ids_raster_path=country_ids_path,
                                                          id_column_label='id',
                                                          zones_raster_data_type=5,
                                                          values_raster_data_type=6,
                                                          zones_ndv=-9999,
                                                          values_ndv=-9999,
                                                          all_touched=None,
                                                          unique_zone_ids=unique_zone_ids,
                                                          assert_projections_same=False,
                                                          output_column_prefix=policy_scenario_label,
                                                          vector_columns_to_include_in_output=['iso3'],
                                                          vector_index_column='id',
                                                          stats_to_retrieve='sums', )

                            current_df.to_csv(current_csv_path, index=False)

                        else:
                            current_df = pd.read_csv(current_csv_path, index_col=0)
                        #
                        # scenario_vertical_name = '2021_2030_' + policy_scenario_label + '_noES'
                        # # rename_dict = {str(k): policy_scenario_label + '_' + v for k, v in hb.esacci_extended_short_class_descriptions.items()}
                        # rename_dict = {str(c): scenario_vertical_name + '_' + hb.esacci_extended_short_class_descriptions[i] for c, i in enumerate(hb.esacci_extended_short_class_descriptions.keys())}
                        # full_rename_dict_pre = {scenario_vertical_name + '_' + hb.esacci_extended_short_class_descriptions[i]: scenario_vertical_name + '_' + hb.esacci_to_habitat_quality_simplified_correspondence[i][2] for c, i in enumerate(hb.esacci_extended_short_class_descriptions.keys())}
                        #
                        # full_rename_dict.update(full_rename_dict_pre)
                        # current_df.rename(columns=rename_dict, inplace=True)
                        #
                        # # Save these for eventual data stacking/unstacking to get into vertical format.
                        # final_vertical_cols += list(rename_dict.values())

                        df = hb.df_merge(df, current_df, how='outer', left_on='id', right_on='id')
                        # df = pd.merge(df, current_df, how='outer', left_index=True, right_ex=True)

            p.carbon_by_country_stats_path = os.path.join(p.cur_dir, 'carbon_by_country.csv')
            df.to_csv(p.carbon_by_country_stats_path, index=False)

            full_gdf = gpd.read_file(p.full_projection_results_vector_path)
            index_cols_to_include = ['pyramid_id', 'pyramid_ids_concatenated', 'pyramid_ids_multiplied', 'gtap37v10_pyramid_id', 'aez_pyramid_id', 'gtap37v10_pyramid_name', 'ISO3', 'AZREG', 'AEZ_COMM', 'gtap9_admin_id', 'GTAP226', 'GTAP140v9a', 'GTAPv9a', 'GTAP141v9p', 'GTAPv9p', 'AreaCode', 'gtap9_admin_name', 'bb', 'minx', 'miny', 'maxx', 'maxy']
            output_df = pd.merge(full_gdf[index_cols_to_include], df, how='outer', left_on='pyramid_id', right_index=True)

            output_df.to_csv(p.land_cover_change_analysis_csv_path)
            # output_df.set_index(['gtap37v10_pyramid_name', 'aez_pyramid_id'

def water_yield_biophysical(p):
    if p.run_this:
        policy_scenario_label = 'gtap1_baseline_2014'
        water_yield_mm_input_path = os.path.join(p.water_yield_biophysical_dir, 'water_yield_' + policy_scenario_label + '_input.tif')
        water_yield_mm_path = os.path.join(p.water_yield_biophysical_dir, 'water_yield_' + policy_scenario_label + '.tif')

        # Placeholder, but currently one still just pastes the results into this folder. Need to keep the task though to CREATE the folder on a new projcet.
        # if not hb.path_exists(water_yield_mm_path):
        #     hb.resample_to_match(water_yield_mm_input_path, p.match_10sec_path, water_yield_mm_path)
        #
        # for luc_scenario in p.luc_scenarios:
        #     for year in p.scenario_years:
        #         policy_scenario_label = luc_scenario + '_' + str(year)
        #         water_yield_mm_input_path = os.path.join(p.water_yield_biophysical_dir, 'water_yield_' + policy_scenario_label + '_input.tif')
        #         current_path = os.path.join(p.water_yield_biophysical_dir, 'water_yield_' + policy_scenario_label + '.tif')
        #         if not hb.path_exists(current_path):
        #             hb.resample_to_match(water_yield_mm_input_path, p.match_10sec_path, current_path)


def water_yield_shock(p):
    """Convert water_yield into shockfile, which is basically just the percentage changes."""

    p.water_yield_shock_csv_path = os.path.join(p.cur_dir, 'water_yield_shock.csv')

    p.water_yield_biophysical_dir = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\projects\feedback_with_policies\intermediate\water_yield_biophysical"

    if p.run_this:
        df = None

        # Placeholder, but currently one still just pastes the results into this folder. Need to keep the task though to CREATE the folder on a new projcet.
        # if not hb.path_exists(p.water_yield_shock_csv_path): # TODOO Could extend this to check that all scenarios exist as cols.
        #     # BASELINE:
        #     policy_scenario_label = 'gtap1_baseline_2014'
        #     water_yield_mm_path = os.path.join(p.water_yield_biophysical_dir, 'water_yield_' + policy_scenario_label + '.tif')
        #     df = hb.zonal_statistics_flex(water_yield_mm_path,
        #                                           p.gtap37_aez18_path,
        #                                           zone_ids_raster_path=p.zone_ids_raster_path,
        #                                           id_column_label='pyramid_id',
        #                                   zones_raster_data_type=5,
        #                                   values_raster_data_type=6,
        #                                   zones_ndv=-9999,
        #                                   values_ndv=-9999,
        #                                           all_touched=None,
        #                                           assert_projections_same=False, )
        #     df.rename(columns={'sums': policy_scenario_label + '_sum'}, inplace=True)
        #
        #     for luc_scenario in p.luc_scenarios:
        #         for year in p.scenario_years:
        #             policy_scenario_label = luc_scenario + '_' + str(year)
        #             current_path = os.path.join(p.water_yield_biophysical_dir, 'water_yield_' + policy_scenario_label + '.tif')
        #             current_df = hb.zonal_statistics_flex(current_path,
        #                                      p.gtap37_aez18_path,
        #                                      zone_ids_raster_path=p.zone_ids_raster_path,
        #                                      id_column_label='pyramid_id',
        #                                                   zones_raster_data_type=5,
        #                                                   values_raster_data_type=6,
        #                                                   zones_ndv=-9999,
        #                                                   values_ndv=-9999,
        #                                      all_touched=None,
        #                                      assert_projections_same=False,)
        #             current_df.rename(columns={'sums': policy_scenario_label + '_sum'}, inplace=True)
        #             if df is None:
        #                 df = current_df
        #             else:
        #                 df = pd.merge(df, current_df, how='outer', left_index=True, right_index=True)
        #     df.to_csv(p.water_yield_shock_csv_path)


def pollination_biophysical(p):

    # TODOO CURRENTLY SAVES IN WEIRD WORISPACE PLACE
    # C:\Files\Research\cge\gtap_invest\gtap_invest_dev\gtap_invest\workspace_poll_suff\lulc_esa_gtap1_rcp45_ssp2_2030_SR_RnD_20p

    baseline_scenario_label = 'lulc_esa_gtap1_baseline_' + str(p.base_year)
    p.baseline_clipped_lulc_path = os.path.join(p.stitched_lulc_esa_scenarios_dir, baseline_scenario_label + '.tif')
    if p.run_this:

        luc_scenario_path = p.base_year_lulc_path
        # base_year_lulc_label = 'lulc_esa_gtap1_baseline_' + str(p.base_year)


        final_raster_path = os.path.join(p.cur_dir, 'poll_suff_ag_coverage_prop_' + baseline_scenario_label + '.tif')
        if not hb.path_exists(final_raster_path):
            L.info('Running global_invest_main.make_poll_suff on LULC: ' + str(luc_scenario_path) + ' and saving results to ' + str(final_raster_path))
            current_landuse_path = os.path.join(p.stitched_lulc_esa_scenarios, baseline_scenario_label + '.tif')


            # base_year_lulc_label = 'lulc_seals7_gtap1_baseline_' + str(p.base_year)
            # esa_include_string = 'lulc_esa_gtap1_' + luh_scenario_label + '_' + str(year) + '_' + policy_scenario_label
            # p.lulc_projected_stitched_path = os.path.join(p.cur_dir, esa_include_string + '.tif')
            #


            pollination_sufficiency.make_poll_suff.execute(current_landuse_path, p.cur_dir)

            created_raster_path = os.path.join(p.cur_dir, 'churn\poll_suff_hab_ag_coverage_rasters', "poll_suff_ag_coverage_prop_10s_" + baseline_scenario_label + ".tif")

            hb.copy_shutil_flex(created_raster_path, final_raster_path)

        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    current_scenario_label = luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                    final_raster_path = os.path.join(p.cur_dir, 'poll_suff_ag_coverage_prop_gtap1_' + current_scenario_label + '.tif')
                    current_landuse_path = os.path.join(p.stitched_lulc_esa_scenarios, 'lulc_esa_gtap1_' + current_scenario_label + '.tif')

                    if not hb.path_exists(final_raster_path):


                        L.info('Running pollination model for ' + current_scenario_label + ' from ' + current_landuse_path + ' to ' + final_raster_path)

                        pollination_sufficiency.make_poll_suff.execute(current_landuse_path, p.cur_dir)

                        # After it finishes, move the file to the root dir and get rid of the cruft.
                        created_raster_path = os.path.join(p.cur_dir, 'churn\poll_suff_hab_ag_coverage_rasters',  'poll_suff_ag_coverage_prop_10s_lulc_esa_gtap1_' + current_scenario_label + '.tif')
                        hb.copy_shutil_flex(created_raster_path, final_raster_path)
                    else:
                        L.info('Skipping running pollination model for ' + current_scenario_label + ' from ' + current_landuse_path + ' to ' + final_raster_path)

def pollination_shock(p):
    """Convert pollination into shockfile."""
    # # OLD SHORTCUT
    # p.policy_scenario_labels = [p.policy_scenario_labels[0], p.policy_scenario_labels[8]]

    ### Thought: think again about adding a penalty to only crediting pollination losses in bau
    p.pollination_shock_change_per_region_path = os.path.join(p.cur_dir, 'pollination_shock_change_per_region.gpkg')
    p.pollination_shock_csv_path = os.path.join(p.cur_dir, 'pollination_shock.csv')
    p.crop_data_dir = r"C:\Users\jajohns\Files\Research\base_data\crops\earthstat\crop_production"
    p.crop_prices_dir = r"C:\Users\jajohns\Files\Research\base_data\pyramids\crops\price"
    # p.pollination_biophysical_dir = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\projects\feedback_with_policies\intermediate\pollination_biophysical"
    p.pollination_dependence_spreadsheet_input_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\pollination\rspb20141799supp3.xls" # Note had to fix pol.dep for cofee and greenbroadbean as it was 25 not .25

    p.crop_value_baseline_path = os.path.join(p.cur_dir, 'crop_value_baseline.tif')
    p.crop_value_no_pollination_path = os.path.join(p.cur_dir, 'crop_value_no_pollination.tif')
    p.crop_value_max_lost_path = os.path.join(p.cur_dir, 'crop_value_max_lost.tif')
    p.crop_value_max_lost_10s_path = os.path.join(p.cur_dir, 'crop_value_max_lost_10s.tif')
    p.crop_value_baseline_10s_path = os.path.join(p.cur_dir, 'crop_value_baseline_10s.tif')

    if p.run_this:
        df = None

        # # TODO HACK: scenario subset
        # p.policy_scenario_labels = p.gtap_bau_and_combined_labels

        ###########################################
        ###### Calculate base-data necessary to do conversion of biophysical to shockfile
        ###########################################

        if not all([hb.path_exists(i) for i in [p.crop_value_baseline_path,
                                                p.crop_value_no_pollination_path,
                                                p.crop_value_max_lost_path,]]):
            df = pd.read_excel(p.pollination_dependence_spreadsheet_input_path, sheet_name='Crop nutrient content')

            crop_names = list(df['Crop map file name'])[:-3] # Drop last three which were custom addons in manuscript and don't seem to have earthstat data for.
            pollination_dependence = list(df['poll.dep'])
            crop_value_baseline = np.zeros(hb.get_shape_from_dataset_path(p.ha_per_cell_300sec_path))
            crop_value_no_pollination = np.zeros(hb.get_shape_from_dataset_path(p.ha_per_cell_300sec_path))
            for c, crop_name in enumerate(crop_names):
                L.info('Calculating value yield effect from pollination for ' + str(crop_name) + ' with pollination dependence ' + str(pollination_dependence[c]))
                crop_price_path = os.path.join(p.crop_prices_dir, crop_name + '_prices_per_ton.tif')
                crop_price = hb.as_array(crop_price_path)
                crop_price = np.where(crop_price > 0, crop_price, 0.0)
                crop_yield = hb.as_array(os.path.join(p.crop_data_dir, crop_name + '_HarvAreaYield_Geotiff', crop_name + '_Production.tif'))
                crop_yield = np.where(crop_yield > 0, crop_yield, 0.0)

                crop_value_baseline += (crop_yield * crop_price)
                crop_value_no_pollination += (crop_yield * crop_price) * (1 - float(pollination_dependence[c]))

            crop_value_max_lost = crop_value_baseline - crop_value_no_pollination
            #
            # crop_value_baseline_path = os.path.join(p.cur_dir, 'crop_value_baseline.tif')
            # crop_value_no_pollination_path = os.path.join(p.cur_dir, 'crop_value_no_pollination.tif')
            # crop_value_max_lost_path = os.path.join(p.cur_dir, 'crop_value_max_lost.tif')

            hb.save_array_as_geotiff(crop_value_baseline, p.crop_value_baseline_path, p.match_300sec_path, ndv=-9999, data_type=6)
            hb.save_array_as_geotiff(crop_value_no_pollination, p.crop_value_no_pollination_path, p.match_300sec_path, ndv=-9999, data_type=6)
            hb.save_array_as_geotiff(crop_value_max_lost, p.crop_value_max_lost_path, p.match_300sec_path, ndv=-9999, data_type=6)


        ### Resample the base data to match LULC
        global_bb = hb.get_bounding_box(p.base_year_lulc_path)
        stitched_bb = hb.get_bounding_box(p.baseline_clipped_lulc_path)
        if stitched_bb != global_bb:
            current_path = os.path.join(p.cur_dir, 'crop_value_max_lost_clipped.tif')
            hb.clip_raster_by_bb(p.crop_value_max_lost_path, stitched_bb, current_path)
            p.crop_value_max_lost_path = current_path

        if not hb.path_exists(p.crop_value_baseline_10s_path):
            hb.resample_to_match(p.crop_value_baseline_path, p.baseline_clipped_lulc_path, p.crop_value_baseline_10s_path, ndv=-9999., output_data_type=6)


        if not hb.path_exists(p.crop_value_max_lost_10s_path):
            hb.resample_to_match(p.crop_value_max_lost_path, p.baseline_clipped_lulc_path, p.crop_value_max_lost_10s_path, ndv=-9999., output_data_type=6)


        ###########################################
        ###### Calculate crop_value_pollinator_adjusted.
        ###########################################

        # Incorporate the "sufficient pollination threshold" of 30%
        # TODOO Go through and systematically pull into config files to initialize model and write output summary of what were used.
        sufficient_pollination_threshold = 0.3

        ### BASELINE crop_value_pollinator_adjusted:
        policy_scenario_label = 'gtap1_baseline_' + str(p.base_year)
        current_output_excel_path = os.path.join(p.cur_dir, 'crop_value_pollinator_adjusted_' + policy_scenario_label + '_zonal_stats.xlsx')
        suff_path = os.path.join(p.pollination_biophysical_dir, 'poll_suff_ag_coverage_prop_lulc_esa_' + policy_scenario_label + '.tif')
        crop_value_pollinator_adjusted_path = os.path.join(p.cur_dir, 'crop_value_pollinator_adjusted_' + policy_scenario_label + '.tif')

        if not hb.path_exists(crop_value_pollinator_adjusted_path):
            hb.raster_calculator_af_flex([p.crop_value_baseline_10s_path, p.crop_value_max_lost_10s_path, suff_path, p.base_year_simplified_lulc_path], lambda baseline_value, max_loss, suff, lulc:
                    np.where((max_loss > 0) & (suff < sufficient_pollination_threshold) & (lulc == 2), baseline_value - max_loss * (1 - (1/sufficient_pollination_threshold) * suff),
                        np.where((max_loss > 0) & (suff >= sufficient_pollination_threshold) & (lulc == 2), baseline_value, -9999.)), output_path=crop_value_pollinator_adjusted_path)

        # Do zonal statistics on outputed raster by AEZ-REG. Note that we need sum and count for when/if we calculate mean ON GRIDCELLS WITH AG.
        if not hb.path_exists(current_output_excel_path):
            df = hb.zonal_statistics_flex(crop_value_pollinator_adjusted_path,
                                          p.gtap37_aez18_path,
                                          zone_ids_raster_path=p.zone_ids_raster_path,
                                          id_column_label='pyramid_id',
                                          zones_raster_data_type=5,
                                          values_raster_data_type=6,
                                          zones_ndv=-9999,
                                          values_ndv=-9999,
                                          all_touched=None,
                                          stats_to_retrieve='sums_counts',
                                          assert_projections_same=False, )
            generated_scenario_label = 'gtap1_baseline_2014'
            df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
            df.to_excel(current_output_excel_path)
        else:
            generated_scenario_label = 'gtap1_baseline_2014'
            df = pd.read_excel(current_output_excel_path, index_col=0)
            df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
        merged_df = df

        ### SCENARIO crop_value_pollinator_adjusted
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    current_output_excel_path = os.path.join(p.cur_dir, 'crop_value_pollinator_adjusted_' + policy_scenario_label + '_zonal_stats.xlsx')
                    suff_path = os.path.join(p.pollination_biophysical_dir, 'poll_suff_ag_coverage_prop_gtap1_' + luh_scenario_label
                                                + '_' + str(scenario_year) + '_' + policy_scenario_label + '.tif')
                    lulc_path = os.path.join(p.stitched_lulc_esa_scenarios_dir, 'lulc_seals7_gtap1_' + luh_scenario_label
                                                + '_' + str(scenario_year) + '_' + policy_scenario_label + '.tif')

                    crop_value_pollinator_adjusted_path = os.path.join(p.cur_dir, 'crop_value_pollinator_adjusted_' + policy_scenario_label + '.tif')

                    if not hb.path_exists(crop_value_pollinator_adjusted_path):
                        hb.raster_calculator_af_flex([p.crop_value_baseline_10s_path, p.crop_value_max_lost_10s_path, suff_path, lulc_path], lambda baseline_value, max_loss, suff, lulc:
                                np.where((max_loss > 0) & (suff < sufficient_pollination_threshold) & (lulc == 2), baseline_value - max_loss * (1 - (1/sufficient_pollination_threshold) * suff),
                                        np.where((max_loss > 0) & (suff >= sufficient_pollination_threshold) & (lulc == 2), baseline_value, -9999.)), output_path=crop_value_pollinator_adjusted_path)


                        # START HERE: Continue thinking about what the right shock is overall. Is it the average on NEW land? Or the aggregate value
                        # To isolate the effect, maybe calculate the average value of crop loss on cells that are cultivated in both scenarios? Start on a dask function that does that?

                    if not hb.path_exists(current_output_excel_path):
                        df = hb.zonal_statistics_flex(crop_value_pollinator_adjusted_path,
                                                      p.gtap37_aez18_path,
                                                      zone_ids_raster_path=p.zone_ids_raster_path,
                                                      id_column_label='pyramid_id',
                                                      zones_raster_data_type=5,
                                                      values_raster_data_type=6,
                                                      zones_ndv=-9999,
                                                      values_ndv=-9999,
                                                      all_touched=None,
                                                      stats_to_retrieve='sums_counts',
                                                      assert_projections_same=False, )

                        generated_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
                        df.to_excel(current_output_excel_path)
                    else:
                        generated_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        df = pd.read_excel(current_output_excel_path, index_col=0)
                        df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count', generated_scenario_label + '_total': generated_scenario_label + '_sum',}, inplace=True)
                    merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)

        ###########################################
        ###### Calculate change from scenario to baseline, on and not on existing ag
        ###########################################

        baseline_policy_scenario_label = 'gtap1_baseline_' + str(p.base_year)
        baseline_crop_value_pollinator_adjusted_path = os.path.join(p.cur_dir, 'crop_value_pollinator_adjusted_' + baseline_policy_scenario_label + '.tif')
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:

                    # Calculate difference between scenario and BASELINE for crop value adjusted
                    bau_crop_value_pollinator_adjusted_path = os.path.join(p.cur_dir, 'crop_value_pollinator_adjusted_BAU.tif')
                    current_crop_value_pollinator_adjusted_path = os.path.join(p.cur_dir, 'crop_value_pollinator_adjusted_' + policy_scenario_label + '.tif')

                    crop_value_difference_from_baseline_path = os.path.join(p.cur_dir, 'crop_value_difference_from_baseline_' + policy_scenario_label + '.tif')
                    if not hb.path_exists(crop_value_difference_from_baseline_path):
                        hb.dask_compute([baseline_crop_value_pollinator_adjusted_path, current_crop_value_pollinator_adjusted_path], lambda x, y: y - x, crop_value_difference_from_baseline_path)

                    # Zonal stats for difference from Baseline
                    current_output_excel_path = os.path.join(p.cur_dir, 'crop_value_difference_from_baseline_' + policy_scenario_label + '_zonal_stats.xlsx')
                    if not hb.path_exists(current_output_excel_path):
                        df = hb.zonal_statistics_flex(crop_value_difference_from_baseline_path,
                                                      p.gtap37_aez18_path,
                                                      zone_ids_raster_path=p.zone_ids_raster_path,
                                                      id_column_label='pyramid_id',
                                                      zones_raster_data_type=5,
                                                      values_raster_data_type=6,
                                                      zones_ndv=-9999,
                                                      values_ndv=-9999,
                                                      all_touched=None,
                                                      stats_to_retrieve='sums_counts',
                                                      assert_projections_same=False, )
                        generated_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
                        df.to_excel(current_output_excel_path)

                    # Calc difference between scenario and BASELINE for crop_value on grid-cells that were agri in both lulc maps.
                    lulc_path = os.path.join(p.stitched_lulc_esa_scenarios_dir, 'lulc_seals7_gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '.tif')
                    crop_value_difference_from_baseline_existing_ag_path = os.path.join(p.cur_dir, 'crop_value_difference_from_baseline_existing_ag_' + policy_scenario_label + '.tif')

                    def op(x, y, w, z):
                        r = dask.array.where((w == 2) & (z == 2), y - x, 0.)
                        rr = (z * 0.0) + r # HACK. Dask.array.where was returning a standard xarray rather than a rioxarray. This dumb hack makes it inherit the rioxarray parameters of z
                        return rr

                    if not hb.path_exists(crop_value_difference_from_baseline_existing_ag_path):
                        op_paths = [
                            baseline_crop_value_pollinator_adjusted_path,
                            crop_value_pollinator_adjusted_path,
                            lulc_path,
                            p.base_year_simplified_lulc_path,
                        ]

                        hb.dask_compute(op_paths, op, crop_value_difference_from_baseline_existing_ag_path)

                    # Zonal stats for difference from Baseline
                    current_output_excel_path = os.path.join(p.cur_dir, 'crop_value_difference_from_baseline_existing_ag_' + policy_scenario_label + '_zonal_stats.xlsx')
                    if not hb.path_exists(current_output_excel_path):
                        df = hb.zonal_statistics_flex(crop_value_difference_from_baseline_existing_ag_path,
                                                      p.gtap37_aez18_path,
                                                      zone_ids_raster_path=p.zone_ids_raster_path,
                                                      id_column_label='pyramid_id',
                                                      zones_raster_data_type=5,
                                                      values_raster_data_type=6,
                                                      zones_ndv=-9999,
                                                      values_ndv=-9999,
                                                      all_touched=None,
                                                      stats_to_retrieve='sums_counts',
                                                      assert_projections_same=False, )
                        generated_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_existing_ag'
                        df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
                        df.to_excel(current_output_excel_path)
                    else:
                        df = pd.read_excel(current_output_excel_path, index_col=0)
                    merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)

                    # Also need to compute the value on that cropland that was cropland in both
                    # crop_value_baseline_existing_ag_path = os.path.join(p.cur_dir, 'crop_value_baseline_existing_ag_' + policy_scenario_label + '.tif')

                    def op(y, w, z):
                        r = dask.array.where((w == 2) & (z == 2), y, 0.)
                        rr = (z * 0.0) + r  # HACK. Dask.array.where was returning a standard xarray rather than a rioxarray. This dumb hack makes it inherit the rioxarray parameters of z
                        return rr

                    if not hb.path_exists(crop_value_baseline_existing_ag_path):
                        op_paths = [
                            baseline_crop_value_pollinator_adjusted_path,
                            lulc_path,
                            p.base_year_simplified_lulc_path,
                        ]

                        hb.dask_compute(op_paths, op, crop_value_baseline_existing_ag_path)

                    # Zonal stats for difference from Baseline
                    current_output_excel_path = os.path.join(p.cur_dir, 'crop_value_baseline_existing_ag_' + policy_scenario_label + '_zonal_stats.xlsx')
                    if not hb.path_exists(current_output_excel_path):
                        df = hb.zonal_statistics_flex(crop_value_baseline_existing_ag_path,
                                                      p.gtap37_aez18_path,
                                                      zone_ids_raster_path=p.zone_ids_raster_path,
                                                      id_column_label='pyramid_id',
                                                      zones_raster_data_type=5,
                                                      values_raster_data_type=6,
                                                      zones_ndv=-9999,
                                                      values_ndv=-9999,
                                                      all_touched=None,
                                                      stats_to_retrieve='sums_counts',
                                                      assert_projections_same=False, )
                        generated_scenario_label = 'crop_value_baseline_existing_ag_compared_to_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
                        df.to_excel(current_output_excel_path)
                    else:
                        df = pd.read_excel(current_output_excel_path, index_col=0)
                    merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)


                    if policy_scenario_label != 'BAU':
                        pass

                        # Calc difference between scenario and BAU for crop_value adjusted
                        # IMPORTANT NOTE: This is really just for plotting and visualization. The shockfiles themselves are all defined relative to the baseline, not relative to bau.
                        crop_value_difference_from_bau_path = os.path.join(p.cur_dir, 'crop_value_difference_from_bau_' + policy_scenario_label + '.tif')
                        bau_lulc_path = os.path.join(p.stitched_lulc_esa_scenarios_dir, 'lulc_seals7_gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_BAU.tif')
                        if not hb.path_exists(crop_value_difference_from_bau_path):
                            hb.dask_compute([bau_crop_value_pollinator_adjusted_path, current_crop_value_pollinator_adjusted_path], lambda x, y: y - x, crop_value_difference_from_bau_path)

                        # Zonal stats for difference from BAU
                        current_output_excel_path = os.path.join(p.cur_dir, 'crop_value_difference_from_bau_' + policy_scenario_label + '_zonal_stats.xlsx')
                        if not hb.path_exists(current_output_excel_path):
                            df = hb.zonal_statistics_flex(crop_value_difference_from_bau_path,
                                                          p.gtap37_aez18_path,
                                                          zone_ids_raster_path=p.zone_ids_raster_path,
                                                          id_column_label='pyramid_id',
                                                          zones_raster_data_type=5,
                                                          values_raster_data_type=6,
                                                          zones_ndv=-9999,
                                                          values_ndv=-9999,
                                                          all_touched=None,
                                                          stats_to_retrieve='sums_counts',
                                                          assert_projections_same=False, )
                            generated_scenario_label = 'gtap2_crop_value_difference_from_bau_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                            df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
                            df.to_excel(current_output_excel_path)
                        else:
                            df = pd.read_excel(current_output_excel_path, index_col=0)
                        merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)

                        # Calc difference between scenario and BAU for crop_value adjusted ON EXISTING AG
                        # ie for crop_value on grid-cells that were agri in both lulc maps.
                        lulc_path = os.path.join(p.stitched_lulc_esa_scenarios_dir, 'lulc_seals7_gtap1_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '.tif')
                        crop_value_difference_from_bau_existing_ag_path = os.path.join(p.cur_dir, 'crop_value_difference_from_bau_existing_ag_' + policy_scenario_label + '.tif')

                        def op(x, y, w, z):
                            r = dask.array.where((w == 2) & (z == 2), y - x, 0.)
                            rr = (z * 0.0) + r  # HACK. Dask.array.where was returning a standard xarray rather than a rioxarray. This dumb hack makes it inherit the rioxarray parameters of z
                            return rr

                        if not hb.path_exists(crop_value_difference_from_bau_existing_ag_path):
                            op_paths = [
                                bau_crop_value_pollinator_adjusted_path,
                                crop_value_pollinator_adjusted_path,
                                lulc_path,
                                bau_lulc_path,
                            ]

                            hb.dask_compute(op_paths, op, crop_value_difference_from_bau_existing_ag_path)

                        current_output_excel_path = os.path.join(p.cur_dir, 'crop_value_difference_from_bau_existing_ag_' + policy_scenario_label + '_zonal_stats.xlsx')
                        if not hb.path_exists(current_output_excel_path):
                            df = hb.zonal_statistics_flex(crop_value_difference_from_bau_existing_ag_path,
                                                          p.gtap37_aez18_path,
                                                          zone_ids_raster_path=p.zone_ids_raster_path,
                                                          id_column_label='pyramid_id',
                                                          zones_raster_data_type=5,
                                                          values_raster_data_type=6,
                                                          zones_ndv=-9999,
                                                          values_ndv=-9999,
                                                          all_touched=None,
                                                          stats_to_retrieve='sums_counts',
                                                          assert_projections_same=False, )


                            generated_scenario_label = 'gtap2_crop_value_difference_from_bau_existing_ag_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                            df.rename(columns={'sums': generated_scenario_label + '_sum', 'counts': generated_scenario_label + '_count'}, inplace=True)
                            df.to_excel(current_output_excel_path)
                        else:
                            df = pd.read_excel(current_output_excel_path, index_col=0)
                        merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)

        ###########################################
        ###### Calculate the actual shock as the mean change.
        ###########################################

        scenario_shock_column_names_to_keep = []
        # scenario_shock_column_names_to_keep = ['pyramid_id', 'pyramid_ids_concatenated', 'pyramid_ids_multiplied', 'gtap37v10_pyramid_id', 'aez_pyramid_id', 'gtap37v10_pyramid_name', 'ISO3', 'AZREG', 'AEZ_COMM']
        if not hb.path_exists(p.pollination_shock_csv_path):
            baseline_generated_scenario_label = 'gtap1_baseline_2014'
            baseline_generated_scenario_label_existing_ag = 'gtap1_baseline_2014_existing_ag'

            scenario_shock_column_names_to_keep.append(baseline_generated_scenario_label + '_sum')
            # scenario_shock_column_names_to_keep.append(baseline_generated_scenario_label_existing_ag + '_sum')

            for luh_scenario_label in p.luh_scenario_labels:
                for scenario_year in p.scenario_years:
                    for policy_scenario_label in p.policy_scenario_labels:

                        generated_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_pollination_shock'
                        merged_df[generated_label] = merged_df[generated_scenario_label + '_sum'] / merged_df[baseline_generated_scenario_label + '_sum']
                        scenario_shock_column_names_to_keep.append(generated_label)

                        # # NOTE: When calculating the value only on existing cells, cannot use the sum / sum method above. Need to use the new rasters created ad calculate their mean.
                        # generated_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_existing_ag'
                        # merged_df[generated_scenario_label + '_mean'] = merged_df[generated_scenario_label_existing_ag + '_sum'] / merged_df[baseline_generated_scenario_label_existing_ag + '_count']

                        # START HERE: ALMOST got the full sim ready to run on the new pollination method but didn't finish getting the averages here calculated.

                        # generated_scenario_label_existing_ag = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_existing_ag'
                        # generated_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_pollination_shock'
                        # merged_df[generated_label] = merged_df[generated_scenario_label_existing_ag + '_sum'] / merged_df[baseline_generated_scenario_label_existing_ag + '_sum']
                        # scenario_shock_column_names_to_keep.append(generated_label)

                        # generated_scenario_existing_ag_label = 'gtap2_crop_value_difference_from_baseline_existing_ag_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        # generated_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_pollination_shock_existing_ag'
                        #
                        # merged_df[generated_label] = merged_df[generated_scenario_existing_ag_label + '_sum'] / merged_df[baseline_generated_scenario_label + '_sum']

                        # # Also subtract the difference with BAU for each other policy
                        # if policy_scenario_label != 'BAU':
                        #     bau_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_BAU_pollination_shock'
                        #     scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_pollination_shock'
                        #     new_label = luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_shock_minus_bau'
                        #     merged_df[new_label] = merged_df[scenario_label] - merged_df[bau_label]
                        #
                        #     # generated_scenario_existing_ag_label = 'gtap2_crop_value_difference_from_bau_existing_ag_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        #     # merged_df[generated_scenario_existing_ag_label + '_mean'] = merged_df[generated_scenario_existing_ag_label + '_sum'] / merged_df[generated_scenario_existing_ag_label + '_count']
                        #     # merged_df[generated_scenario_existing_ag_label + '_mean_minus_baseline'] = merged_df[generated_scenario_existing_ag_label + '_mean'] - merged_df[baseline_generated_scenario_label + '_mean']
                        #
                        #
                        #     generated_bau_label  = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_BAU'
                        #     generated_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        #
                        #     generated_bau_existing_ag_label  = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_BAU'
                        #     merged_df[generated_scenario_label + '_sum_div_bau'] = merged_df[generated_scenario_label + '_sum'] / merged_df[generated_bau_label + '_sum']
                        #
                        #     generated_scenario_existing_ag_label = 'gtap2_crop_value_difference_from_bau_existing_ag_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        #     # merged_df[generated_scenario_existing_ag_label + '_mean_minus_bau'] = merged_df[generated_scenario_existing_ag_label + '_mean'] - merged_df[generated_scenario_existing_ag_label + '_mean']
                        #
                        #     # merged_df[generated_scenario_existing_ag_label + '_sum_div_bau'] = merged_df[generated_scenario_existing_ag_label + '_sum'] / merged_df[generated_bau_existing_ag_label + '_sum']
                        #
                        #     generated_scenario_label= 'gtap2_crop_value_difference_from_bau_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label


            # write to csv and gpkg
            merged_df.to_csv(hb.suri(p.pollination_shock_csv_path, 'comprehensive'))
            gdf = gpd.read_file(p.gtap37_aez18_path)
            gdf = gdf.merge(merged_df, left_on='pyramid_id', right_index=True, how='outer')
            gdf.to_file(hb.suri(p.pollination_shock_change_per_region_path, 'comprehensive'), driver='GPKG')

            merged_df = merged_df[scenario_shock_column_names_to_keep]
            merged_df.to_csv(p.pollination_shock_csv_path)
            gdf = gpd.read_file(p.gtap37_aez18_path)
            gdf = gdf.merge(merged_df, left_on='pyramid_id', right_index=True, how='outer')
            gdf.to_file(p.pollination_shock_change_per_region_path, driver='GPKG')


def pollination_shock_old(p):
    """Convert pollination into shockfile."""

    p.pollination_shock_csv_path = os.path.join(p.cur_dir, 'pollination_shock.csv')
    p.crop_data_dir = r"C:\Users\jajohns\Files\Research\base_data\crops\earthstat\crop_production"
    p.crop_prices_dir = r"C:\Users\jajohns\Files\Research\base_data\pyramids\crops\price"
    # p.pollination_biophysical_dir = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\projects\feedback_with_policies\intermediate\pollination_biophysical"
    p.pollination_dependence_spreadsheet_input_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\pollination\rspb20141799supp3.xls" # Note had to fix pol.dep for cofee and greenbroadbean as it was 25 not .25

    p.crop_value_baseline_path = os.path.join(p.cur_dir, 'crop_value_baseline.tif')
    p.crop_value_no_pollination_path = os.path.join(p.cur_dir, 'crop_value_no_pollination.tif')
    p.crop_value_max_lost_path = os.path.join(p.cur_dir, 'crop_value_max_lost.tif')
    p.crop_value_max_lost_10s_path = os.path.join(p.cur_dir, 'crop_value_max_lost_10s.tif')

    if p.run_this:
        df = None

        if not all([hb.path_exists(i) for i in [p.crop_value_baseline_path,
                                                p.crop_value_no_pollination_path,
                                                p.crop_value_max_lost_path,]]):
            df = pd.read_excel(p.pollination_dependence_spreadsheet_input_path, sheet_name='Crop nutrient content')

            crop_names = list(df['Crop map file name'])[:-3] # Drop last three which were custom addons in manuscript and don't seem to have earthstat data for.
            pollination_dependence = list(df['poll.dep'])
            crop_value_baseline = np.zeros(hb.get_shape_from_dataset_path(p.ha_per_cell_300sec_path))
            crop_value_no_pollination = np.zeros(hb.get_shape_from_dataset_path(p.ha_per_cell_300sec_path))
            for c, crop_name in enumerate(crop_names):
                L.info('Calculating value yield effect from pollination for ' + str(crop_name) + ' with pollination dependence ' + str(pollination_dependence[c]))
                crop_price_path = os.path.join(p.crop_prices_dir, crop_name + '_prices_per_ton.tif')
                crop_price = hb.as_array(crop_price_path)
                crop_yield = hb.as_array(os.path.join(p.crop_data_dir, crop_name + '_HarvAreaYield_Geotiff', crop_name + '_Production.tif'))

                crop_value_baseline += crop_yield
                crop_value_no_pollination += crop_yield * (1 - float(pollination_dependence[c]))

            crop_value_max_lost = crop_value_baseline - crop_value_no_pollination
            crop_value_baseline_path = os.path.join(p.cur_dir, 'crop_value_baseline.tif')
            crop_value_no_pollination_path = os.path.join(p.cur_dir, 'crop_value_no_pollination.tif')
            crop_value_max_lost_path = os.path.join(p.cur_dir, 'crop_value_max_lost.tif')

            hb.save_array_as_geotiff(crop_value_baseline, p.crop_value_baseline_path, p.match_300sec_path, ndv=-9999, data_type=6)
            hb.save_array_as_geotiff(crop_value_no_pollination, p.crop_value_no_pollination_path, p.match_300sec_path, ndv=-9999, data_type=6)
            hb.save_array_as_geotiff(crop_value_max_lost, p.crop_value_max_lost_path, p.match_300sec_path, ndv=-9999, data_type=6)

        # Resample max_loss
        global_bb = hb.get_bounding_box(p.base_year_lulc_path)
        stitched_bb = hb.get_bounding_box(p.baseline_clipped_lulc_path)
        if stitched_bb != global_bb:
            current_path = os.path.join(p.cur_dir, 'crop_value_max_lost_clipped.tif')
            hb.clip_raster_by_bb(p.crop_value_max_lost_path, stitched_bb, current_path)
            p.crop_value_max_lost_path = current_path

        if not hb.path_exists(p.crop_value_max_lost_10s_path):
            hb.resample_to_match(p.crop_value_max_lost_path, p.baseline_clipped_lulc_path, p.crop_value_max_lost_10s_path, ndv=-9999., output_data_type=6)

        # BASELINE:
        policy_scenario_label = 'gtap1_baseline_' + str(p.base_year)
        current_output_excel_path = os.path.join(p.cur_dir, policy_scenario_label + '_zonal_stats.xlsx')
        pollination_baseline_path = os.path.join(p.pollination_biophysical_dir, 'poll_suff_ag_coverage_prop_lulc_esa_' + policy_scenario_label + '.tif')

        if not hb.path_exists(current_output_excel_path):
            df = hb.zonal_statistics_flex(pollination_baseline_path,
                                          p.gtap37_aez18_path,
                                          zone_ids_raster_path=p.zone_ids_raster_path,
                                          id_column_label='pyramid_id',
                                          zones_raster_data_type = 5,
                                          values_raster_data_type = 6,
                                          zones_ndv = -9999,
                                          values_ndv = 9999,
                                          all_touched=None,
                                          assert_projections_same=False,
                                          csv_output_path=current_output_excel_path)
            generated_scenario_label = 'gtap1_baseline_2014'
            df.rename(columns={'sums': generated_scenario_label + '_total'}, inplace=True)
            df.to_excel(current_output_excel_path)
        else:
            df = pd.read_excel(current_output_excel_path, index_col=0)
        merged_df = df

        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    current_output_excel_path = os.path.join(p.cur_dir, policy_scenario_label + '_zonal_stats.xlsx')
                    current_path = os.path.join(p.pollination_biophysical_dir, 'poll_suff_ag_coverage_prop_gtap1_' + luh_scenario_label
                                                + '_' + str(scenario_year) + '_' + policy_scenario_label + '.tif')
                    amount_lost_path = os.path.join(p.cur_dir, 'pollination_value_lost_' + luh_scenario_label
                                                    + '_' + policy_scenario_label + '_' + str(scenario_year) + '.tif')

                    if not hb.path_exists(current_output_excel_path):
                        hb.raster_calculator_af_flex([p.crop_value_max_lost_10s_path, current_path], lambda x, y: np.where((x > 0) & (y < sufficient_pollination_threshold), x * (1 - (1./sufficient_pollination_threshold) * y), x), output_path=amount_lost_path)

                        df = hb.zonal_statistics_flex(current_path,
                                                      p.gtap37_aez18_path,
                                                      zone_ids_raster_path=p.zone_ids_raster_path,
                                                      id_column_label='pyramid_id',
                                                      zones_raster_data_type = 5,
                                                      values_raster_data_type = 6,
                                                      zones_ndv = -9999,
                                                      values_ndv = 9999,
                                                      all_touched=None,
                                                      assert_projections_same=False, )
                        generated_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '_total'
                        df.rename(columns={'sums': generated_scenario_label}, inplace=True)
                        df.to_excel(current_output_excel_path)
                    else:
                        df = pd.read_excel(current_output_excel_path, index_col=0)
                    merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)

        merged_df.to_csv(p.pollination_shock_csv_path)


def marine_fisheries_shock(p):
    """Calculate shockfile from ensemble of FISHMIP"""
    p.gtap37_marine_pyramid_path = os.path.join(p.model_base_data_dir, "gtap_marine_regions", "gtap37_marine_pyramid.gpkg")

    # DATA NOTE: These were very hard to download. Had to extract the download link from the wget file. Here they are:

    "http://esg.pik-potsdam.de/thredds/fileServer/isimip_dataroot/isimip2a/output/marine-fishery_global/BOATS/GFDL-ESM2M/rcp45/wo-diaz/no-fishing/no-oa/global/tcb/monthly/v20181005/boats_gfdl-esm2m_rcp4p5_wo-diaz_no-fishing_no-oa_tcb_global_monthly_2006_2099.nc"
    "http://esg.pik-potsdam.de/thredds/fileServer/isimip_dataroot/isimip2a/output/marine-fishery_global/BOATS/GFDL-ESM2M/rcp45/wo-diaz/fishing/no-oa/global/tcb/monthly/v20181005/boats_gfdl-esm2m_rcp4p5_wo-diaz_fishing_no-oa_tcb_global_monthly_2006_2099.nc"
    "http://esg.pik-potsdam.de/thredds/fileServer/isimip_dataroot/isimip2a/output/marine-fishery_global/BOATS/GFDL-ESM2M/rcp60/wo-diaz/no-fishing/no-oa/global/tcb/monthly/v20181005/boats_gfdl-esm2m_rcp6p0_wo-diaz_no-fishing_no-oa_tcb_global_monthly_2006_2099.nc"
    "http://esg.pik-potsdam.de/thredds/fileServer/isimip_dataroot/isimip2a/output/marine-fishery_global/BOATS/GFDL-ESM2M/rcp60/wo-diaz/fishing/no-oa/global/tcb/monthly/v20181005/boats_gfdl-esm2m_rcp6p0_wo-diaz_fishing_no-oa_tcb_global_monthly_2006_2099.nc"

    "http://esg.pik-potsdam.de/thredds/fileServer/isimip_dataroot/isimip2a/output/marine-fishery_global/BOATS/IPSL-CM5A-LR/rcp45/wo-diaz/no-fishing/no-oa/global/tcb/monthly/v20181005/boats_ipsl-cm5a-lr_rcp4p5_wo-diaz_no-fishing_no-oa_tcb_global_monthly_2006_2099.nc"
    "http://esg.pik-potsdam.de/thredds/fileServer/isimip_dataroot/isimip2a/output/marine-fishery_global/BOATS/IPSL-CM5A-LR/rcp45/wo-diaz/fishing/no-oa/global/tcb/monthly/v20181005/boats_ipsl-cm5a-lr_rcp4p5_wo-diaz_fishing_no-oa_tcb_global_monthly_2006_2099.nc"
    "http://esg.pik-potsdam.de/thredds/fileServer/isimip_dataroot/isimip2a/output/marine-fishery_global/BOATS/IPSL-CM5A-LR/rcp60/wo-diaz/no-fishing/no-oa/global/tcb/monthly/v20181005/boats_ipsl-cm5a-lr_rcp6p0_wo-diaz_no-fishing_no-oa_tcb_global_monthly_2006_2099.nc"
    "http://esg.pik-potsdam.de/thredds/fileServer/isimip_dataroot/isimip2a/output/marine-fishery_global/BOATS/IPSL-CM5A-LR/rcp60/wo-diaz/fishing/no-oa/global/tcb/monthly/v20181005/boats_ipsl-cm5a-lr_rcp6p0_wo-diaz_fishing_no-oa_tcb_global_monthly_2006_2099.nc"

    p.fisheries_shockfile_path = os.path.join(p.cur_dir, 'fisheries_shockfile.xlsx')
    p.fishing_scenario_paths = OrderedDict()
    p.fishing_scenario_paths['rcp85_fishing'] = OrderedDict()
    # p.fishing_scenario_paths['rcp85_nofishing'] = OrderedDict()
    # p.fishing_scenario_paths['rcp60_fishing'] = OrderedDict()
    # p.fishing_scenario_paths['rcp60_nofishing'] = OrderedDict()
    p.fishing_scenario_paths['rcp45_fishing'] = OrderedDict()
    # p.fishing_scenario_paths['rcp45_nofishing'] = OrderedDict()
    p.fishing_scenario_paths['rcp26_fishing'] = OrderedDict()
    p.fishing_scenario_paths['rcp26_nofishing'] = OrderedDict()

    # TODOO SUB IN rcp4.5
    p.fishing_scenario_paths['rcp85_fishing']['boats_gfdl'] = os.path.join(p.model_base_data_dir, "marine_fisheries", "monthly", "boats_gfdl-esm2m_rcp8p5_wo-diaz_fishing_no-oa_tcb_global_monthly_2006_2099.nc")
    p.fishing_scenario_paths['rcp85_fishing']['boats_ipsl'] = os.path.join(p.model_base_data_dir, "marine_fisheries", "monthly", "boats_ipsl-cm5a-lr_rcp8p5_wo-diaz_fishing_no-oa_tcb_global_monthly_2006_2099.nc")
    # p.fishing_scenario_paths['rcp60_fishing']['boats_gfdl'] = os.path.join(p.model_base_data_dir, "marine_fisheries", "monthly", "boats_gfdl-esm2m_rcp8p5_wo-diaz_fishing_no-oa_tcb_global_monthly_2006_2099.nc")
    # p.fishing_scenario_paths['rcp60_fishing']['boats_ipsl'] = os.path.join(p.model_base_data_dir, "marine_fisheries", "monthly", "boats_ipsl-cm5a-lr_rcp8p5_wo-diaz_fishing_no-oa_tcb_global_monthly_2006_2099.nc")
    p.fishing_scenario_paths['rcp45_fishing']['boats_gfdl'] = os.path.join(p.model_base_data_dir, "marine_fisheries", "monthly", "boats_gfdl-esm2m_rcp4p5_wo-diaz_fishing_no-oa_tcb_global_monthly_2006_2099.nc")
    p.fishing_scenario_paths['rcp45_fishing']['boats_ipsl'] = os.path.join(p.model_base_data_dir, "marine_fisheries", "monthly", "boats_ipsl-cm5a-lr_rcp4p5_wo-diaz_fishing_no-oa_tcb_global_monthly_2006_2099.nc")

    p.fishing_scenario_paths['rcp26_fishing']['boats_gfdl'] = os.path.join(p.model_base_data_dir, "marine_fisheries", "monthly", "boats_gfdl-esm2m_rcp2p6_wo-diaz_fishing_no-oa_tcb_global_monthly_2006_2099.nc")
    p.fishing_scenario_paths['rcp26_fishing']['boats_ipsl'] = os.path.join(p.model_base_data_dir, "marine_fisheries", "monthly", "boats_ipsl-cm5a-lr_rcp2p6_wo-diaz_fishing_no-oa_tcb_global_monthly_2006_2099.nc")
    p.fishing_scenario_paths['rcp26_nofishing']['boats_gfdl'] = os.path.join(p.model_base_data_dir, "marine_fisheries", "monthly", "boats_gfdl-esm2m_rcp2p6_wo-diaz_no-fishing_no-oa_tcb_global_monthly_2006_2099.nc")
    p.fishing_scenario_paths['rcp26_nofishing']['boats_ipsl'] = os.path.join(p.model_base_data_dir, "marine_fisheries", "monthly", "boats_ipsl-cm5a-lr_rcp2p6_wo-diaz_no-fishing_no-oa_tcb_global_monthly_2006_2099.nc")

    p.all_years = p.base_years + [2015] + p.scenario_years
    window_radius = 12

    if p.run_this:
        if not hb.path_exists(p.fisheries_shockfile_path):
            base_year = 2015
            target_year = 2030
            assumed_irg = 0.2 # from lit review
            dim_selection_indices = [base_year - target_year, None, None]
            output_df = None
            for policy_scenario_label, odict in p.fishing_scenario_paths.items():
                for year in p.all_years:
                    moving_average = None
                    n = 0

                    for model_name, path in odict.items():

                        target_dir = os.path.join(p.cur_dir, policy_scenario_label, model_name)



                        for shift in range(window_radius * 2 + 1):
                            current_time = (int(year) - 1901) * 12 - 1260 + shift

                            hb.extract_global_netcdf_to_geotiff(path, target_dir, vars_to_extract=None, time_indices_to_extract=current_time)
                            # hb.extract_global_netcdf_to_geotiff(path, target_dir, vars_to_extract=None, time_indices_to_extract=1788 - 1260)
                            generated_path = os.path.join(target_dir, hb.file_root(path) + '_tcb_' + str(current_time + 1260) + '.tif')

                            a = hb.as_array(generated_path)
                            a = np.where(a != -9999., a, 0)
                            if moving_average is None:
                                moving_average = np.zeros(a.shape)
                            moving_average += a
                            n += 1
                    moving_average /= n
                    output_path = os.path.join(p.cur_dir, policy_scenario_label + '_' + str(year) + '_moving_average.tif')
                    hb.save_array_as_geotiff(moving_average, output_path, generated_path)


                for base_year in p.base_years:
                    for scenario_year in p.scenario_years:
                        baseline_array = hb.as_array(os.path.join(p.cur_dir, policy_scenario_label + '_' + str(base_year) + '_moving_average.tif'))
                        scenario_array = hb.as_array(os.path.join(p.cur_dir, policy_scenario_label + '_' + str(scenario_year) + '_moving_average.tif'))

                        zone_ids_raster_path = os.path.join(p.cur_dir, 'marine_ids_10s.tif')
                        stats_path = os.path.join(p.cur_dir, policy_scenario_label + '_' + str(base_year) + '.csv')

                        baseline_array_stats_df = hb.zonal_statistics_flex(os.path.join(p.cur_dir, policy_scenario_label + '_' + str(base_year) + '_moving_average.tif'),
                                                                  p.gtap37_marine_pyramid_path, zone_ids_raster_path=zone_ids_raster_path, id_column_label='pyramid_id',
                                                                  values_ndv=-9999., csv_output_path=stats_path)

                        stats_path = os.path.join(p.cur_dir, policy_scenario_label + '_' + str(scenario_year) + '.csv')
                        scenario_array_stats_df = hb.zonal_statistics_flex(os.path.join(p.cur_dir, policy_scenario_label + '_' + str(scenario_year) + '_moving_average.tif'),
                                                                  p.gtap37_marine_pyramid_path, zone_ids_raster_path=zone_ids_raster_path, id_column_label='pyramid_id',
                                                                  values_ndv=-9999., csv_output_path=stats_path)

                        if output_df is None:
                            output_df = baseline_array_stats_df
                            # output_df.drop('count', axis=1, inplace=True)
                            output_df.rename(columns={'sums': policy_scenario_label + '_' + str(base_year)}, inplace=True)

                            next_df = scenario_array_stats_df
                            # next_df.drop('count', axis=1, inplace=True)
                            next_df.rename(columns={'sums': policy_scenario_label + '_' + str(scenario_year)}, inplace=True)
                            output_df = output_df.merge(next_df[[i for i in next_df.columns if i not in output_df.columns]], left_index=True, right_index=True)


                        else:
                            next_df = baseline_array_stats_df
                            # next_df.drop('count', axis=1, inplace=True)
                            next_df.rename(columns={'sums': policy_scenario_label + '_' + str(base_year)}, inplace=True)
                            output_df = output_df.merge(next_df[[i for i in next_df.columns if i not in output_df.columns]], left_index=True, right_index=True)

                            next_df = scenario_array_stats_df
                            # next_df.drop('count', axis=1, inplace=True)
                            next_df.rename(columns={'sums': policy_scenario_label + '_' + str(scenario_year)}, inplace=True)
                            output_df = output_df.merge(next_df[[i for i in next_df.columns if i not in output_df.columns]], left_index=True, right_index=True)

            for policy_scenario_label, odict in p.fishing_scenario_paths.items():
                for scenario_year in p.scenario_years:

                    output_df[policy_scenario_label + '_' + str(scenario_year) + '_pch'] = (output_df[policy_scenario_label + '_' + str(scenario_year)] - output_df[policy_scenario_label + '_' + str(p.base_years[0])]) / output_df[policy_scenario_label + '_' + str(p.base_years[0])]
                    # output_df['rcp26_fishing_pch'] = (output_df['rcp26_fishing_2030'] - output_df['rcp26_fishing_2015']) / output_df['rcp26_fishing_2015']
                    # output_df['rcp26_nofishing_pch'] = (assumed_irg * output_df['rcp26_nofishing_2030'] - output_df['rcp26_fishing_2015']) / output_df['rcp26_fishing_2015']
                    # output_df['rcp26_nofishing_pch'] = (output_df['rcp26_nofishing_2030'] - output_df['rcp26_nofishing_2015']) / output_df['rcp26_nofishing_2015']

            gtap37_all_pyramid = gpd.read_file(p.gtap37_marine_pyramid_path, driver='GPKG')
            gtap37_all_pyramid = gtap37_all_pyramid.merge(output_df, left_on='pyramid_id', right_index=True)
            gtap37_all_pyramid.to_excel(p.fisheries_shockfile_path)

        # TODOO check if I have or not a off by 1 pyramid id error because there was -9999 included as a region.


def coastal_protection_biophysical(p):
    # Dummy to create dir
    if p.run_this:
        pass

def coastal_protection_shock(p):

    # Main output
    p.coastal_protection_shockfile_path = os.path.join(p.cur_dir, 'coastal_protection_shockfile_vector.gpkg')
    # Load CP input paths into scenario dict
    p.coastal_protection_biophysical_paths = {}
    for luh_scenario_label in p.luh_scenario_labels:
        p.coastal_protection_biophysical_paths[luh_scenario_label] = {}
        for scenario_year in p.scenario_years:
            p.coastal_protection_biophysical_paths[luh_scenario_label][scenario_year] = {}
            for policy_scenario_label in p.policy_scenario_labels:
                #
                # "lulc_gtap1_pes_esa_classes_md5_219590831d116ff455bb1273ee5f1236.gpkg"
                # "lulc_gtap1_rd_esa_classes_md5_9b24cfc51fefc8362058ec656af091f2.gpkg"
                # "lulc_gtap1_subs_esa_classes_md5_fde00f7f77e143bca0541ca2f8cfd9e6.gpkg"
                path = os.path.join(p.coastal_protection_biophysical_dir, 'gtap1_coastal_protection_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label + '.gpkg')
                p.coastal_protection_biophysical_paths[luh_scenario_label][scenario_year][policy_scenario_label] = path

    p.rt_points_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\shared_inputs\coastal vulnerability_2018-10-15\shapefile_Rt\Rt.shp"
    p.gtap_regions_path = os.path.join(p.input_dir, "gtapv9140_07-09-2018.shp")
    p.marine_zones_path = os.path.join(hb.BASE_DATA_DIR, r"cartographic\marineregions\EEZ_land_union_v2_201410\EEZ_land_v2_201410.shp")
    p.nev_countries_path = r"C:\Users\jajohns\Files\Research\base_data\cartographic\naturalearth_v3\10m_cultural\ne_10m_admin_0_countries.shp"

    p.marine_inclusive_gtap_regions_path = os.path.join(p.cur_dir, 'marine_inclusive_gtap_regions.shp')

    p.points_in_poly_shapefile_path = os.path.join(p.cur_dir, 'pip.shp')
    p.points_in_poly_excel_path = os.path.join(p.cur_dir, 'pip.xlsx')

    p.cv_results_by_ssp_path = os.path.join(p.cur_dir, "cv_results_by_ssp.shp")

    p.esa_path = r"C:\Users\jajohns\Files\Research\base_data\lulc\esacci\ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992-v2.0.7.tif"
    p.land_binary_path = os.path.join(p.cur_dir, 'esa_land_binary.tif')
    p.country_marine_polygons_path = r"C:\Users\jajohns\Files\Research\base_data\cartographic\country_marine_polygons.shp"

    if p.run_this:
        # marine_zones = gpd.read_file(p.marine_zones_path)
        # rt_points = gpd.read_file(p.rt_points_path)
        # gtap_regions = gpd.read_file(p.gtap_regions_path)
        # nev_countries = gpd.read_file(p.nev_countries_path)
        # Append each scenario's raw results and a binary for if it passed a threshold
        scenarios_shockfile_gdf = None

        # First load baseline
        cp_input_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\projects\feedback_policies_and_tipping_points\intermediate\coastal_protection_biophysical\gtap1_coastal_protection_baseline_2021.gpkg"
        L.info('Loading Coastal Protection input file: ' + cp_input_path)
        scenarios_shockfile_gdf = gpd.read_file(cp_input_path)
        scenarios_shockfile_gdf['Rt_' + str(2021)] = scenarios_shockfile_gdf['Rt_nohab_all'] - scenarios_shockfile_gdf['service_ESA']
        scenarios_shockfile_gdf = scenarios_shockfile_gdf[['Rt_' + str(2021), 'geometry']]
        scenarios_shockfile_gdf['hr_' + str(2021)] = np.where(scenarios_shockfile_gdf['Rt_' + str(2021)].values >= 3.3, 1, 0)
        scenarios_shockfile_gdf = scenarios_shockfile_gdf[['Rt_' + str(2021), 'hr_' + str(2021), 'geometry']]
        baseline_gdf = scenarios_shockfile_gdf.copy()


        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:

                    cp_input_path = p.coastal_protection_biophysical_paths[luh_scenario_label][scenario_year][policy_scenario_label]
                    L.info('Loading Coastal Protection input file: ' + cp_input_path)
                    current_gdf = gpd.read_file(cp_input_path)
                    L.info('    Loaded Coastal Protection input file: ' + str(list(current_gdf)))

                    current_gdf = current_gdf[['Rt', 'Rt_nohab_all', 'geometry']]
                    current_gdf = current_gdf.rename(columns={'Rt': 'Rt_' + str(scenario_year) + '_' + str(policy_scenario_label),
                                                              'Rt_nohab_all': 'Rt_nohab_all_' + str(scenario_year) + '_' + str(policy_scenario_label),
                                                              })

                    # LEARNING Point, this appears to be the fastest way to calculate a new column, ie vectorizing the funciton over numpy arrays, which is better than using apply()
                    current_gdf['hr_' + str(2030) + '_' + policy_scenario_label] = np.where(current_gdf['Rt_' + str(2030) + '_' + policy_scenario_label].values >= 3.3, 1, 0)
                    current_gdf['hr_nohab_all_' + str(2030) + '_' + policy_scenario_label] = np.where(current_gdf['Rt_nohab_all_' + str(2030) + '_' + policy_scenario_label].values >= 3.3, 1, 0)

                    # Number of new cels at high risk
                    current_gdf['hrd_' + str(2030) + '_' + policy_scenario_label] = current_gdf['hr_' + str(2030) + '_' + policy_scenario_label] - baseline_gdf['hr_' + str(2021)]

                    # LEARNING POINT, geopandas sjoin is implemented such that it automatically gives it an index_right. Thus a second sjoin on the same one will fail cause it has an index_right already.
                    scenarios_shockfile_gdf = scenarios_shockfile_gdf[[i for i in scenarios_shockfile_gdf.columns if i not in ['index_right', 'index_left']]]
                    scenarios_shockfile_gdf = gpd.sjoin(scenarios_shockfile_gdf, current_gdf, how='inner')

        scenarios_shockfile_gdf.to_file(p.coastal_protection_shockfile_path, driver='GPKG')
        # old but indicative of goals.
        do = False
        if do:
            # Create land-binary path.

            start = time.time()
            esa_ds = gdal.Open(p.esa_path)
            esa_array = esa_ds.GetRasterBand(1).ReadAsArray()  # .astype(np.int8)  # callback=reporter
            land_binary = np.where((esa_array != 210) & (esa_array != 0), 1, 0)
            hb.save_array_as_geotiff(land_binary, p.land_binary_path, p.esa_path, data_type=1, ndv=255)

            # Merge GTAP Regions into marine zones.
            joined = marine_zones.merge(gtap_regions[['ISO', 'ID', 'GTAP140']], left_on='ISO_3digit', right_on='ISO', how='left')
            nev_countries = nev_countries.drop(columns=['geometry'])

            # Merge NEV into marine zones
            columns = ['featurecla', 'scalerank', 'LABELRANK', 'SOVEREIGNT', 'SOV_A3', 'ADM0_DIF', 'LEVEL', 'TYPE', 'ADMIN', 'ADM0_A3', 'GEOU_DIF', 'GEOUNIT', 'GU_A3', 'SU_DIF', 'SUBUNIT', 'SU_A3', 'BRK_DIFF', 'NAME', 'NAME_LONG', 'BRK_A3', 'BRK_NAME', 'BRK_GROUP', 'ABBREV', 'POSTAL', 'FORMAL_EN', 'FORMAL_FR', 'NAME_CIAWF', 'NOTE_ADM0', 'NOTE_BRK', 'NAME_SORT', 'NAME_ALT', 'POP_EST', 'POP_RANK', 'GDP_MD_EST', 'POP_YEAR', 'LASTCENSUS', 'GDP_YEAR', 'ECONOMY', 'INCOME_GRP', 'WIKIPEDIA', 'FIPS_10_', 'ISO_A2', 'ISO_A3', 'ISO_A3_EH', 'ISO_N3', 'UN_A3', 'WB_A2', 'WB_A3', 'WOE_ID', 'WOE_ID_EH', 'WOE_NOTE', 'ADM0_A3_IS', 'ADM0_A3_US', 'ADM0_A3_UN', 'ADM0_A3_WB', 'CONTINENT', 'REGION_UN', 'SUBREGION', 'REGION_WB', 'NAME_LEN', 'LONG_LEN', 'ABBREV_LEN', 'TINY', 'HOMEPART', 'MIN_ZOOM', 'MIN_LABEL', 'MAX_LABEL', 'NE_ID', 'WIKIDATAID']
            joined = joined.merge(nev_countries[columns], left_on='ISO_3digit', right_on='ADM0_A3', how='left')
            joined.to_file(p.marine_inclusive_gtap_regions_path)

            country_marine_polygons = gpd.read_file(p.country_marine_polygons_path)
            country_marine_polygons = country_marine_polygons[['ISO_3digit', 'GTAP140', 'geometry']]

            if not os.path.exists(p.points_in_poly_shapefile_path):
                points_in_polys = gpd.sjoin(rt_points, country_marine_polygons, how='inner')
                points_in_polys.to_file(p.points_in_poly_shapefile_path)
                # points_in_polys.to_excel(p.points_in_poly_excel_path)
            else:
                points_in_polys = gpd.read_file(p.points_in_poly_shapefile_path)

            # sum_per_country = points_in_polys.groupby('PolyGroupByField')['fields', 'in', 'grouped', 'output'].agg(['sum'])
            if not os.path.exists(p.cv_results_by_ssp_path):
                cv = points_in_polys
                # Scenario differences from current. (not very useful because index)
                cv['ssp1_d'] = cv['Rt_ssp1'] - cv['Rt_cur']
                cv['ssp3_d'] = cv['Rt_ssp3'] - cv['Rt_cur']
                cv['ssp5_d'] = cv['Rt_ssp5'] - cv['Rt_cur']

                # # Scenario means
                # cv['cur_m'] = cv['Rt_cur'] / cv['NUMPOINTS']
                # cv['ssp1_m'] = cv['Rt_ssp1'] / cv['NUMPOINTS']
                # cv['ssp3_m'] = cv['Rt_ssp3'] / cv['NUMPOINTS']
                # cv['ssp5_m'] = cv['Rt_ssp5'] / cv['NUMPOINTS']
                #
                # # Mean difference
                # cv['ssp1_md'] = cv['ssp1_m'] - cv['cur_m']
                # cv['ssp3_md'] = cv['ssp3_m'] - cv['cur_m']
                # cv['ssp5_md'] = cv['ssp5_m'] - cv['cur_m']

                # Calcualte number that are high-risk, ie above 3.3.
                def op(x, y):
                    np.where(x > y, 1, 0)

                # LEARNING Point, this appears to be the fastest way to calculate a new column, ie vectorizing the funciton over numpy arrays, which is better than using apply()
                cv['cur_hr'] = np.where(cv['Rt_cur'].values >= 3.3, 1, 0)
                cv['ssp1_hr'] = np.where(cv['Rt_ssp1'].values >= 3.3, 1, 0)
                cv['ssp3_hr'] = np.where(cv['Rt_ssp3'].values >= 3.3, 1, 0)
                cv['ssp5_hr'] = np.where(cv['Rt_ssp5'].values >= 3.3, 1, 0)

                # Number of new cels at high risk
                cv['ssp1_hrd'] = cv['ssp1_hr'] - cv['cur_hr']
                cv['ssp3_hrd'] = cv['ssp3_hr'] - cv['cur_hr']
                cv['ssp5_hrd'] = cv['ssp5_hr'] - cv['cur_hr']

                cv.to_file(p.cv_results_by_ssp_path)
                # # Number of new cels at high risk
                # cv['ssp1_if'] = 1.0 - (cv['ssp1_hrd'] / cv['NUMPOINTS'])
                # cv['ssp3_if'] = 1.0 - (cv['ssp3_hrd'] / cv['NUMPOINTS'])
                # cv['ssp5_if'] = 1.0 - (cv['ssp5_hrd'] / cv['NUMPOINTS'])
            else:
                cv = gpd.read_file(p.cv_results_by_ssp_path)

            p.cv_aggregated_path = os.path.join(p.cur_dir, 'cv_aggregated.xlsx')
            if p.run_this:
                cv = gpd.read_file(p.cv_results_by_ssp_path)
                cv_aggregated = cv.groupby('ISO_3digit', as_index=False)['Rt_cur', 'Rt_ssp1', 'Rt_ssp3', 'Rt_ssp5', 'ssp1_hrd', 'ssp3_hrd', 'ssp5_hrd'].agg(['sum', 'count'])

                new_columns = ['Rt_cur_sum', 'Rt_cur_count', 'Rt_ssp1_sum', 'Rt_ssp1_count', 'Rt_ssp3_sum', 'Rt_ssp3_count', 'Rt_ssp5_sum', 'Rt_ssp5_count', 'ssp1_hrd_sum', 'ssp1_hrd_count', 'ssp3_hrd_sum', 'ssp3_hrd_count', 'ssp5_hrd_sum', 'ssp5_hrd_count']
                cv_aggregated.columns = new_columns

                cv_aggregated.rename({'Rt_cur_count': 'num_coastal_cells'}, axis='columns', inplace=True)

                final_columns = ['num_coastal_cells', 'Rt_cur_sum', 'Rt_ssp1_sum', 'Rt_ssp3_sum', 'Rt_ssp5_sum', 'ssp1_hrd_sum', 'ssp3_hrd_sum', 'ssp5_hrd_sum']
                cv_aggregated = cv_aggregated[final_columns]
                cv_aggregated.to_excel(p.cv_aggregated_path)


            p.cv_impact_factors_path = os.path.join(p.cur_dir, 'cv_impact_factors.xlsx')
            if p.run_this:
                df = pd.read_excel(p.cv_aggregated_path)
                df['ssp1_if'] = 1.0 - (df['ssp1_hrd_sum'] / df['num_coastal_cells'])
                df['ssp3_if'] = 1.0 - (df['ssp3_hrd_sum'] / df['num_coastal_cells'])
                df['ssp5_if'] = 1.0 - (df['ssp5_hrd_sum'] / df['num_coastal_cells'])

                df.to_excel(p.cv_impact_factors_path)


def gtap2_combined_shockfile(p):
    p.gtap2_exhaustive_shockfile_path = os.path.join(p.cur_dir, "gtap2_exhaustive_shockfile.gpkg")
    p.gtap2_shockfile_path = os.path.join(p.cur_dir, "gtap2_shockfile.gpkg")

    p.old_shockfile_path_for_comparison_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\projects\combined_policies\reference\gtap2_shockfile_08_27.xlsx"
    if p.run_this:
        gdf = gpd.read_file(p.full_projection_results_vector_path)
        gdf_original = gdf.copy()

        # NOTE: Excluded marine boundaries for now because marine countries have many extra zones.
        # marine_df = pd.read_excel(p.fisheries_shockfile_path, index_col=0)
        #
        # LEARNING POINT: When merging two GDFs, the geometry columns will be renamed geometry_x and geometry_y. This will present as the gdf being interpreted as a df cause no geometry and so the error will be df has not method to_file()
        # gdf = gdf.merge(marine_df[[i for i in marine_df.columns if i not in gdf.columns]], left_on='pyramid_id', right_index=True, how='outer')

        # TODOO For full rerun, make sure Greenland is dropped at the correct stage if desired. Currently is' blank in a lot of these shockfile columns.
        #Drop antarctica
        gdf = gdf[gdf['pyramid_id'] != 0]

        gdf['aez_pyramid_id'].fillna(0.0, inplace=True)
        gdf['AZREG'] = 'AZ' + gdf['aez_pyramid_id'].map(int).map(str) + '_' + gdf['gtap37v10_pyramid_name'].map(str)

        # Merge in PES shockfile. This is the information that went into GTAP endogenous land-use calculation.
        pes_gdf = gpd.read_file(p.pes_shockfile_parameters_path.replace('.csv', '.gpkg'), driver='GPKG')
        pes_gdf['AZREG'] = 'AZ' + pes_gdf['aez_pyramid_id'].map(int).map(str) + '_' + pes_gdf['gtap37v10_pyramid_name'].map(str)
        gdf = gdf.merge(pes_gdf[[i for i in pes_gdf.columns if i not in gdf.columns or i == 'AZREG']], left_on='AZREG', right_on='AZREG', how='outer')


        # Read in carbon shocks
        df = pd.read_csv(p.carbon_shock_csv_path, index_col=0)

        # Rename all carbon storage columns so that they are sufficiently detailed when it merges back in with to full GDF.
        rename_dict = {}
        rename_dict['gtap1_baseline_2014_sum'] = 'gtap2_rcp45_ssp2_2014_baseline_carbon_storage_total_ha'
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    current_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label

                    rename_dict[current_scenario_label + '_sum'] = current_scenario_label + '_carbon_storage_total_ha'
        df = df.rename(columns=rename_dict)

        gdf = gdf.merge(df[[i for i in df.columns if i not in gdf.columns or i == 'pyramid_id']], left_on='pyramid_id', right_index=True, how='outer')

        # Calculate carbon shock as percentage change from baseline
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    current_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label

                    gdf[current_scenario_label + '_carbon_forestry_shock'] = \
                        (gdf[current_scenario_label + '_carbon_storage_total_ha'] / gdf['gtap2_rcp45_ssp2_2014_baseline_carbon_storage_total_ha']) * 100.


        # For speed, and because we aren't affecting fisheries in this run, I just re-used the old fisheries shocks.

        use_old_fisheries_shockfiles = 0
        if use_old_fisheries_shockfiles:
            if hb.path_exists(p.old_shockfile_path_for_comparison_path):
                old_shockfile_df = pd.read_excel(p.old_shockfile_path_for_comparison_path)
                old_cols = [
                    'AZREG', 'marine_fisheries_baseline_quantity', 'marine_fisheries_pes_shock',
                ]
                # old_cols = [
                #     'AZREG',
                #     'marine_fisheries_baseline_quantity', 'marine_fisheries_bau_shock', 'marine_fisheries_subs_shock', 'marine_fisheries_rd_shock', 'marine_fisheries_pes_shock',
                #     'pollination_baseline_quantity', 'pollination_bau_shock', 'pollination_subs_shock', 'pollination_rd_shock', 'pollination_pes_shock',
                #     'carbon_forestry_baseline_quantity', 'carbon_forestry_bau_shock', 'carbon_forestry_subs_shock', 'carbon_forestry_rd_shock', 'carbon_forestry_pes_shock',
                # ]
                old_shockfile_df = old_shockfile_df[old_cols]
                old_shockfile_df['AZREG'] = old_shockfile_df['AZREG'].str.replace('-', '_')
                gdf = gdf.merge(old_shockfile_df[[i for i in old_shockfile_df.columns if i not in gdf.columns or i == 'AZREG']], left_on='AZREG', right_on='AZREG', how='outer')

                # Copy the old fisheries columns so that to each new scenario
                for luh_scenario_label in p.luh_scenario_labels:
                    for scenario_year in p.scenario_years:
                        for policy_scenario_label in p.policy_scenario_labels:
                            current_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                            gdf[current_scenario_label + '_marine_fisheries_shock'] = gdf['marine_fisheries_pes_shock']
            else:
                fisheries_gdf = pd.read_excel(p.fisheries_shockfile_path)
                gdf = gdf.merge(fisheries_gdf[[i for i in fisheries_gdf.columns if i not in gdf.columns or i == 'GTAP140v9a']], left_on='GTAP140v9a', right_on='GTAP140v9a', how='outer')
        # Use NEW fisheries shockfiles.
        else:
            fisheries_gdf = pd.read_excel(p.fisheries_shockfile_path)
            gdf = gdf.merge(fisheries_gdf[[i for i in fisheries_gdf.columns if i not in gdf.columns or i == 'GTAP140v9a']], left_on='GTAP140v9a', right_on='GTAP140v9a', how='outer')

        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    current_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                    old_label = luh_scenario_label.split('_')[0] + '_fishing_' + str(scenario_year) + '_pch'
                    gdf[current_scenario_label + '_marine_fisheries_shock'] = gdf[old_label] * 100 + 100

        # Collapse shock, added for completeness but is unchanged
        # gdf['carbon_forestry_fishcol_shock'] = gdf['carbon_forestry_bau_shock']

        # # Collapse of amazon shock
        # # Total collapse loses 88.2% where 82.2 is the percent on average that shrubland in that ecoregion has in carbon content compared to forest. of carbon due to much lower carbon content of shrubland.
        # gdf['carbon_forestry_amazoncol_shock'] = gdf['carbon_forestry_bau_shock']
        # gdf['carbon_forestry_amazoncol_shock'][gdf['aez_pyramid_id'] == 5] = gdf['carbon_forestry_amazoncol_shock'] * (1 - .882)
        # gdf['carbon_forestry_amazoncol_shock'][gdf['aez_pyramid_id'] == 6] = gdf['carbon_forestry_amazoncol_shock'] * (1 - .882)
        # gdf['carbon_forestry_polcol_shock'] = gdf['carbon_forestry_bau_shock']


        # Read in pollination and merge with gdf
        df = pd.read_csv(p.pollination_shock_csv_path, index_col=0)
        gdf = gdf.merge(df[[i for i in df.columns if i not in gdf.columns or i == 'pyramid_id']], left_on='pyramid_id', right_index=True, how='outer')

        # Iterate through scenarios to calculate percentage change in POLLINATION
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    current_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                    gdf.rename(columns={'gtap1_baseline_2014_sum': 'gtap1_baseline_2014_pollination_sum'}, inplace=True)
                    gdf.rename(columns={'gtap1_baseline_2014_count': 'gtap1_baseline_2014_pollination_count'}, inplace=True)
                    gdf.rename(columns={current_scenario_label + '_sum': current_scenario_label + '_pollination_sum'}, inplace=True)
                    gdf.rename(columns={current_scenario_label + '_count': current_scenario_label + '_pollination_count'}, inplace=True)

                    # # OLD METHOD FOR VALIDATION
                    # gdf[current_scenario_label + '_pollination_shock_old'] = np.where(((gdf[current_scenario_label + '_pollination_sum'] / gdf[current_scenario_label + '_pollination_count']) / (gdf['gtap1_baseline_2014_pollination_sum'] / gdf['gtap1_baseline_2014_pollination_count'])) * 100. < 140.,
                    #                                                               ((gdf[current_scenario_label + '_pollination_sum'] / gdf[current_scenario_label + '_pollination_count']) / (gdf['gtap1_baseline_2014_pollination_sum'] / gdf['gtap1_baseline_2014_pollination_count'])) * 100.,
                    #                                                               ((gdf[current_scenario_label + '_pollination_sum'] / gdf[current_scenario_label + '_pollination_count']) / (gdf['gtap1_baseline_2014_pollination_sum'] / gdf['gtap1_baseline_2014_pollination_count'])) * 100.)
                    #
                    # gdf[current_scenario_label + '_pollination_shock_old'] = np.where(((gdf[current_scenario_label + '_pollination_sum'] / gdf[current_scenario_label + '_pollination_count']) / (gdf['gtap1_baseline_2014_pollination_sum'] / gdf['gtap1_baseline_2014_pollination_count'])) * 100. < 140.,
                    #                                                               ((gdf[current_scenario_label + '_pollination_sum'] / gdf[current_scenario_label + '_pollination_count']) / (gdf['gtap1_baseline_2014_pollination_sum'] / gdf['gtap1_baseline_2014_pollination_count'])) * 100.,
                    #                                                               ((gdf[current_scenario_label + '_pollination_sum'] / gdf[current_scenario_label + '_pollination_count']) / (gdf['gtap1_baseline_2014_pollination_sum'] / gdf['gtap1_baseline_2014_pollination_count'])) * 100.)
        to_drop = ['marine_fisheries_pes_shock']
        gdf[[i for i in gdf.columns if i != 'geometry' and i not in to_drop]].to_excel(p.gtap2_exhaustive_shockfile_path.replace('.gpkg', '.xlsx'), index=False)

        gdf = gdf.rename(columns={
            'rcp45_fishing_' + str(p.base_years[0]): 'marine_fisheries_baseline_quantity',
            'gtap1_baseline_carbon_storage_' + str(p.base_years[0]) + '_sum': 'carbon_forestry_baseline_quantity',
            # 'gtap1_baseline_old_carbon_forestry_2014_sum': 'carbon_forestry_baseline_quantity',
            'gtap1_baseline_old_pollination_' + str(p.base_years[0]) + '_sum': 'pollination_baseline_quantity',
        }        )

        shockfile_indices = [
            'pyramid_id',
            'pyramid_ids_concatenated',
            'pyramid_ids_multiplied',
            'gtap37v10_pyramid_id',
            'aez_pyramid_id',
            'gtap37v10_pyramid_name',
            'ISO3',
            'AZREG',
            'AEZ_COMM',
        ]


        # shockfile_column_headers =
        #
        # shockfile_column_headers = [i for i in rename_dict.values()]
        # gdf = gdf[shockfile_column_headers]

        # values = {i: 100.0 for i in shock_cols}
        # gdf = np.where(gdf.isna(), 100.0, gdf)

        shockfile_column_headers = ['gtap2_rcp45_ssp2_2014_baseline_carbon_storage_total_ha', 'marine_fisheries_baseline_quantity', 'gtap1_baseline_2014_pollination_sum']
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    current_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                    # shockfile_column_headers.append(current_scenario_label + '_pollination_sum')
                    shockfile_column_headers.append(current_scenario_label + '_pollination_shock')
                    shockfile_column_headers.append(current_scenario_label + '_marine_fisheries_shock')
                    shockfile_column_headers.append(current_scenario_label + '_carbon_forestry_shock')
                    shockfile_column_headers.append(current_scenario_label + '_carbon_storage_total_ha')
        shockfile_column_headers = list(sorted(shockfile_column_headers))
        gdf = gdf[shockfile_indices + shockfile_column_headers]

        gdf = gdf.loc[gdf['pyramid_id'] > 0]
        for i in [i for i in gdf.columns if i not in shockfile_indices]:
            gdf[i] = np.where(gdf[i].isna(), 100.0, gdf[i])
            gdf[i].fillna(100.0, inplace=True)
        # gdf = gdf.fillna(value=values, axis=0)
        gdf = gdf.loc[gdf['gtap37v10_pyramid_name'].notna()]

        # gdf.to_file(p.gtap2_shockfile_path, driver='GPKG')

        gdf[[i for i in gdf.columns if i != 'geometry']].to_excel(p.gtap2_shockfile_path.replace('.gpkg', '.xlsx').replace('.xlsx', '_prior_to_gtap_rename.xlsx'), index=False)


        # Last step is to then rename all things EXACTLY like uris wants and save as a CSV ready for copying.
        rename_dict = {}
        rename_dict['gtap37v10_pyramid_name'] = 'REG'
        rename_dict['AEZ_COMM'] = 'AEZ'
        rename_dict['marine_fisheries_baseline_quantity'] = 'gtap2_rcp45_ssp2_2014_baseline_marine_fisheries_total'
        rename_dict['gtap1_baseline_2014_pollination_sum'] = 'gtap2_rcp45_ssp2_2014_baseline_pollination_sum'
        rename_dict['gtap2_rcp45_ssp2_2014_baseline_carbon_storage_total_ha'] = 'gtap2_rcp45_ssp2_2014_baseline_carbon_forestry_total'
        gdf.rename(columns=rename_dict, inplace=True)

        gtap_style_shockfile_column_headers = ['REG', 'ISO3', 'AZREG', 'AEZ']

        shocked_es_names = ['marine_fisheries', 'pollination', 'carbon_forestry']
        for shocked_es_name in shocked_es_names:

            for luh_scenario_label in p.luh_scenario_labels:
                if shocked_es_name == 'pollination':
                    current_baseline_total_label = 'gtap2_' + luh_scenario_label + '_2014_baseline_' + shocked_es_name  + '_sum'
                else:
                    current_baseline_total_label = 'gtap2_' + luh_scenario_label + '_2014_baseline_' + shocked_es_name  + '_total'
                gtap_style_shockfile_column_headers.append(current_baseline_total_label)

                for scenario_year in p.scenario_years:
                    for policy_scenario_label in p.policy_scenario_labels:
                        current_scenario_label = 'gtap2_' + luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        current_scenario_with_es_label = current_scenario_label + '_' + shocked_es_name + '_shock'
                        gtap_style_shockfile_column_headers.append(current_scenario_with_es_label)

        gdf['gtap37v10_pyramid_name'] = gdf['REG'] # Add it back in cause when we renamed off of this we lost it.
        gtap_style_shockfile_column_headers.extend(['pyramid_id', 'pyramid_ids_concatenated', 'pyramid_ids_multiplied', 'gtap37v10_pyramid_id', 'aez_pyramid_id', 'gtap37v10_pyramid_name'])
        gdf = gdf[gtap_style_shockfile_column_headers]
        gdf_original

        gdf_out = hb.df_merge(gdf_original, gdf, left_on='pyramid_id', right_on='pyramid_id')
        gdf_out = gdf_out[[i for i in gdf_out.columns if i != 'geometry'] + ['geometry']]
        hb.pp(gdf_out.columns)
        gdf_out.to_file(p.gtap2_shockfile_path, driver='GPKG')
        gdf[[i for i in gdf.columns if i != 'geometry']].to_csv(p.gtap2_shockfile_path.replace('.gpkg', '.csv'), index=False)

def gtap2_process_shockfiles(p):
    p.gtap2_aez_invest_proc_local_model_dir = os.path.join(p.cur_dir, '00_InVEST_proc')

    if p.run_this:



        # Extract a gtap-aez-invest zipfile into the curdir
        if not hb.path_exists(p.gtap2_aez_invest_proc_local_model_dir):
            if not hb.path_exists(p.gtap_aez_invest_proc_code_dir):
                L.info('Unzipping all files in ' + p.gtap_aez_invest_proc_zipfile_path + ' to ' + p.gtap2_aez_invest_proc_local_model_dir)
                hb.unzip_file(p.gtap_aez_invest_proc_zipfile_path, p.cur_dir, verbose=False)
            else:
                L.info('Creating project-specific copy of GTAP files, copying from ' + p.gtap_aez_invest_proc_code_dir + ' to ' + p.gtap2_aez_invest_proc_local_model_dir)
                hb.copy_file_tree_to_new_root(p.gtap_aez_invest_proc_code_dir, p.gtap2_aez_invest_proc_local_model_dir)

        hb.create_directories(os.path.join(p.gtap2_aez_invest_proc_local_model_dir, 'work'))

        # Copy InVEST Shockfile to expected input dir, saving it as a csv as expected.
        invest_shockfile_path = os.path.join(p.gtap2_combined_shockfile_dir, 'gtap2_shockfile.csv')
        dst_path = os.path.join(p.cur_dir, '00_InVEST_proc', 'in', 'InVEST', 'gtap2_shockfile.csv')
        hb.copy_shutil_flex(invest_shockfile_path, dst_path)

        # Now call the R code to pull these
        src_r_proc_script_path = os.path.join(p.gtap2_aez_invest_proc_local_model_dir, '01_data_proc.r')

        # Create a local copy of the R file with modifications specific to this run (i.e., the shanging the workspace)
        r_proc_script_path = os.path.join(p.gtap2_aez_invest_proc_local_model_dir, '01_data_proc_local.r')

        # hb.copy_shutil_flex(src_r_proc_script_path, r_proc_script_path)
        gtap_invest_integration_functions.generate_procs_r_script_file(src_r_proc_script_path, r_proc_script_path)

        gtap2_process_shockfiles_results_exist = hb.path_exists(os.path.join(p.gtap2_aez_invest_proc_local_model_dir, 'out'))
        if not gtap2_process_shockfiles_results_exist or 1: # HACK becasue directory was getting written first somehow
            L.info('Starting r script at ' + str(r_proc_script_path) + ' to create scenario-specific shockfiles.')

            hb.execute_r_script(p.r_executable_path, os.path.abspath(r_proc_script_path))


def gtap2_shockfile_analysis(p):
    p.shockfile_labels = [
        'carbon_forestry_shock',
        'marine_fisheries_shock',
        'pollination_shock_gro',
        'pollination_shock_ocr',
        'pollination_shock_osd',
        'pollination_shock_pfb',
        'pollination_shock_v_f',
    ]

    p.shockfiles_summary_spreadsheet_path = os.path.join(p.cur_dir, 'gtap2_shockfiles_summary.xlsx')
    if p.run_this:
        df = None
        run_label = 'gtap2'

        multi_index_columns = []
        for luh_label in p.luh_scenario_labels:
            for year in p.scenario_years:
                for scenario_label in p.policy_scenario_labels:
                    for shockfile_label in p.shockfile_labels:
                        file_name_standard = '_'.join([run_label, 'luh_label', str(year), scenario_label, shockfile_label]) + '.txt'
                        shock_dir = os.path.join(p.gtap2_process_shockfiles_dir, '00_InVEST_proc', 'out')
                        input_path = os.path.join(shock_dir, '_'.join([run_label, luh_label, str(year), scenario_label, shockfile_label]) + '.txt')

                        df_current = gtap_invest_integration_functions.gtap_shockfile_to_df(input_path)

                        multi_index_columns.append((luh_label, year, scenario_label, shockfile_label))

                        if df is None:
                            df = df_current
                        else:
                            df = df.merge(df_current, left_index=True, right_index=True)
        df.to_excel(p.shockfiles_summary_spreadsheet_path)

        do_comparison_to_ecn_shockfile = 1
        if do_comparison_to_ecn_shockfile:
            df_compare = None
            # ecn_project_dir = os,path.join(p.cur_dir, '../../../economic_case_for_nature')
            ecn_dir = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\projects\gtap_invest_pnas\intermediate\gtap2_aez_WITH_URIS_GENERATED_SHOCKS_1"
            ecn_gtap_root_dir = os.path.join(ecn_dir, p.gtap_aez_invest_release_string)

            for luh_label in p.luh_scenario_labels:
                for year in p.scenario_years:

                    # HACK: Note that this is not the set of scenario labels used elsewhere.
                    # for scenario_label in p.gtap_bau_and_pesgc_label:
                    for scenario_label in p.policy_scenario_labels:
                        for shockfile_label in p.shockfile_labels:
                            file_name_standard = '_'.join([run_label, 'luh_label', str(year), scenario_label, shockfile_label]) + '.txt'
                            # shock_dir = os.path.join(p.gtap2_process_shockfiles_dir, '00_InVEST_proc', 'out')
                            input_path = os.path.join(ecn_gtap_root_dir, 'X' + str(year) + '_' + scenario_label + '_' + shockfile_label + '.txt')

                            df_current = gtap_invest_integration_functions.gtap_shockfile_to_df(input_path)

                            multi_index_columns.append((luh_label + '_COMPARE', year, scenario_label, shockfile_label))


                            if df_compare is None:
                                df_compare = df.merge(df_current, left_index=True, right_index=True) # Start with previous df
                            else:
                                df_compare = df_compare.merge(df_current, left_index=True, right_index=True)


            df_compare.to_excel(hb.suri(p.shockfiles_summary_spreadsheet_path, 'comparison_flat'))
            hb.pp(multi_index_columns)
            df_compare.columns = pd.MultiIndex.from_tuples(multi_index_columns)
            df_compare.to_excel(hb.suri(p.shockfiles_summary_spreadsheet_path, 'comparison'))


def gtap2_ae_uris(p):
    """Run a precompiled GTAPAEZ.exe file by calling the and a cmf file.
    Additionally and somewhat confusingly, this is also where I enable the ability to run GTAP-AEZ via an external process. The way this works is it copies the
    code for GTAP-InVEST into the ProjectFlow p.cur_dir, runs it, and saves a result in a logical place. HOWEVER, if the process is instead run via a batch file
    the way it was done before, it saves it to a different location. Subsequent tasks choose the manual .bat result over the internal one IF IT EXISTS. So
    if you have previously run a manual run and now want it to be in ProjectFlow, just make sure to delete tehe gtap_result directory.

    TODOO: Generate specific tab files for each scenario dynamically based on a template. Might not be needed?

    """
    p.gtap2_results_path = os.path.join(p.cur_dir, 'GTAP_Results.csv')
    p.gtap2_land_use_change_path = os.path.join(p.cur_dir, 'GTAP_AEZ_LCOVER_ha.csv')

    p.gtap2_aez_invest_local_model_dir = os.path.join(p.cur_dir, p.gtap_aez_invest_release_string)
    if p.run_this:

        gtap_run_copy_exclude_list = [i for i in p.gtap_combined_policy_scenario_labels if not any([i in j for j in p.gtap2_scenario_labels])]
        gtap_run_copy_include_list = p.gtap2_scenario_labels + ['GTAPAEZ', 'gtapaez', '2014_21_BAU_supp','sets', '2014_21_BAU.upd', 'Default.prm', 'default.prm', 'base', 'Base', '2021_2030', 'TOHAT2']

        # Extract a gtap-aez-invest zipfile into the curdir
        # TODOO rethink this overly complex structure and bring it 'in-house' rather than assuming releases coming out of purdue will match this pattern.
        if not hb.path_exists(p.gtap2_aez_invest_local_model_dir):
            if not hb.path_exists(p.gtap_aez_invest_code_dir):
                L.info('Unzipping all files in ' + p.gtap_aez_invest_zipfile_path + ' to ' + p.gtap2_aez_invest_local_model_dir)

                hb.unzip_file(p.gtap_aez_invest_zipfile_path, p.cur_dir, verbose=False)
            else:
                L.info('Creating project-specific copy of GTAP files, copying from ' + p.gtap_aez_invest_code_dir + ' to ' + p.gtap2_aez_invest_local_model_dir)
                # HACK (sorta), I should ultimately get rid of this copy_file_tree approach as it is as easy to break as list_filtered_paths_recursively
                hb.copy_file_tree_to_new_root(p.gtap_aez_invest_code_dir, p.gtap2_aez_invest_local_model_dir, exclude_strings=gtap_run_copy_exclude_list)
                # hb.copy_file_tree_to_new_root(p.gtap_aez_invest_code_dir, p.gtap2_aez_invest_local_model_dir, include_strings=gtap_run_copy_include_list)

        hb.create_directories(os.path.join(p.gtap2_aez_invest_local_model_dir, 'work'))

        # There was a typo in Uris' scenario names, fixd here.
        src = os.path.join(p.gtap2_aez_invest_local_model_dir, 'cmfs', '2021_30_SR_RnD_20p_PESGB_30_noES.cmf')
        dst = os.path.join(p.gtap2_aez_invest_local_model_dir, 'cmfs', '2021_30_SR_RnD_20p_PESGC_30_noES.cmf')
        if hb.path_exists(src):
            hb.copy_shutil_flex(src, dst)

        # for a GTAP2 run, also need to copy in the results from gtap1 as it build on those solution files.
        # HACK (sorta), I should ultimately get rid of this copy_file_tree approach as it is as easy to break as list_filtered_paths_recursively
        hb.copy_file_tree_to_new_root(p.gtap1_aez_invest_local_model_dir, p.gtap2_aez_invest_local_model_dir, exclude_strings=gtap_run_copy_exclude_list, exclude_extensions='.r', verbose=False)

        # Copy in (and replace Uris' old) shock files from the process_shockfile dir
        src_dir = os.path.join(p.gtap2_process_shockfiles_dir, '00_InVEST_proc', 'out')
        dst_dir = os.path.join(p.cur_dir, '04_20_2021_GTAP_AEZ_INVEST')


        L.warning('HACK, this will only work if the gtap labels are formatted exactly right, ie the year is the first 4 digits before the underscore.')
        for gtap_scenario_label in p.gtap2_scenario_labels:
            current_year = int(gtap_scenario_label.split('_')[0][0:2] + gtap_scenario_label.split('_')[1])
            current_scenario_label = '_'.join(gtap_scenario_label.split('_')[2:-1])

            for shockfile_label in p.shockfile_labels:
                src_file = os.path.join(src_dir, 'gtap2_rcp45_ssp2_' + str(current_year) + '_' + current_scenario_label + '_' + shockfile_label + '.txt')
                dst_file = os.path.join(dst_dir, 'X' + str(current_year) + '_' + current_scenario_label + '_' + shockfile_label + '.txt')

                if hb.path_exists(dst_file) and not hb.path_exists(hb.suri(dst_file, 'validation')):
                    os.rename(dst_file, hb.suri(dst_file, 'validation'))
                hb.copy_shutil_flex(src_file, dst_file)


        # %% UNABLE TO OPEN EXISTING FILE '2021_30_BAU_rigid_noES_SUPP.har'.
        # if there is a file “2021_30_BAU_noES_supp.HAR” you can just duplicate that and rename to “'2021_30_BAU_rigid_noES_SUPP.har”

        gtapaez_executable_abs_path = os.path.join(p.gtap2_aez_invest_local_model_dir, 'GTAPAEZ.exe')

        # # Define paths for the source cmf file (extracted from GTAP-AEZ integration zipfile) and the modified one that will be run
        # gtap_policy_baseline_scenario_label = str(p.base_year) + '_' + str(p.policy_base_year)[2:] + '_BAU'
        # gtap_policy_baseline_scenario_source_cmf_path = os.path.join(p.gtap2_aez_invest_local_model_dir, 'cmfs', gtap_policy_baseline_scenario_label + '.cmf')
        # gtap_policy_baseline_scenario_cmf_path = os.path.join(p.gtap2_aez_invest_local_model_dir, gtap_policy_baseline_scenario_label + '_local.cmf')
        # gtap_policy_baseline_solution_file_path = os.path.join(p.cur_dir, p.gtap_aez_invest_release_string, 'work', gtap_policy_baseline_scenario_label + '.sl4')
        #
        # L.info('gtap_policy_baseline_scenario_cmf_path', gtap_policy_baseline_scenario_cmf_path)
        # L.info('gtap_policy_baseline_solution_file_path', gtap_policy_baseline_solution_file_path)
        #
        # if not hb.path_exists(gtap_policy_baseline_solution_file_path, verbose=True):
        #     # Generate a new cmf file with updated paths.
        #     gtap_invest_integration_functions.generate_policy_baseline_cmf_file(gtap_policy_baseline_scenario_source_cmf_path, gtap_policy_baseline_scenario_cmf_path)
        #
        #     # Run the gtap executable pointing to the new cmf file
        #     call_list = [gtapaez_executable_abs_path, '-cmf', gtap_policy_baseline_scenario_cmf_path]
        #     gtap_invest_integration_functions.run_gtap_cmf(gtap_policy_baseline_scenario_label, call_list)

        run_parallel = 1
        parallel_iterable = []
        # HACK: Note that this is not the set of scenario labels used elsewhere.
        # ORIGINAL: for gtap_scenario_label in p.gtap2_scenario_labels:
        # subset_scenarios = [i for i in p.gtap2_scenario_labels if i in p.gtap_bau_and_pesgc_label]
        # subset_scenarios = ['2021_30_' + i + '_allES' for i in p.gtap_bau_and_pesgc_label]
        # for gtap_scenario_label in subset_scenarios:
        for gtap_scenario_label in p.gtap2_scenario_labels:

            current_scenario_source_cmf_path = os.path.join(p.gtap2_aez_invest_local_model_dir, 'cmfs', gtap_scenario_label + '.cmf')
            current_scenario_cmf_path = os.path.join(p.gtap2_aez_invest_local_model_dir, gtap_scenario_label + '_local.cmf')

            current_solution_file_path = os.path.join(p.cur_dir, p.gtap_aez_invest_release_string, 'work', gtap_scenario_label + '.sl4')

            # Hack to fix uris scenario typo

            src = os.path.join(p.gtap2_aez_invest_local_model_dir, 'cmfs', '2021_30_SR_RnD_20p_PESGB_30_allES.cmf')
            dst = os.path.join(p.gtap2_aez_invest_local_model_dir, 'cmfs', '2021_30_SR_RnD_20p_PESGC_30_allES.cmf')
            if hb.path_exists(src):
                hb.copy_shutil_flex(src, dst)

            # Hack to rename shocks.txt typo
            for i in [os.path.join(p.gtap2_aez_invest_local_model_dir, "2021_30_SR_Land_allES_shocks.txt"), os.path.join(p.gtap2_aez_invest_local_model_dir, "2021_30_SR_Land_noES_shocks.txt")]:
                if hb.path_exists(i):
                    hb.copy_shutil_flex(i, i.replace('shocks.txt', 'shock.txt'))

            if '2021_30_SR_RnD_20p_PESGC_30_allES' in current_solution_file_path:
                current_solution_file_path = current_solution_file_path.replace('2021_30_SR_RnD_20p_PESGC_30_allES', '2021_30_SR_RnD_20p_PESGB_30_allES')

            if not hb.path_exists(current_solution_file_path, verbose=True):
                # Generate a new cmf file with updated paths.
                # Currently this just uses the policy_baseline version.
                gtap_invest_integration_functions.generate_policy_baseline_cmf_file(current_scenario_source_cmf_path, current_scenario_cmf_path)


                # Run the gtap executable pointing to the new cmf file
                # FOR MANUAL: Open command prompt at release dir, enter command: ".\GTAPAEZ.exe -cmf 2021_30_PESGC_allES_local.cmf"
                call_list = [gtapaez_executable_abs_path, '-cmf', current_scenario_cmf_path]
                parallel_iterable.append(tuple([gtap_scenario_label, call_list]))


                # HACK
                do_validation = 0
                if do_validation:
                    L.critical('Running validation as well. This means running uris old shockfiles for each scenario from the ECN release')
                    current_scenario_validation_cmf_path = os.path.join(p.gtap2_aez_invest_local_model_dir, gtap_scenario_label + '_validation.cmf')
                    gtap_invest_integration_functions.generate_policy_baseline_cmf_file(current_scenario_cmf_path, current_scenario_validation_cmf_path, run_on_validation=True)

                    call_list = [gtapaez_executable_abs_path, '-cmf', current_scenario_validation_cmf_path]
                    validation_label = gtap_scenario_label + '_validation'
                    if '2021_30_SR_Land_allES_validation' not in current_scenario_validation_cmf_path:
                        parallel_iterable.append(tuple([validation_label, call_list]))

                # If the model is run sequentially, just call it here.
                if not run_parallel:
                    L.critical('About to run non-parallel' + str(parallel_iterable))

                    gtap_invest_integration_functions.run_gtap_cmf(gtap_scenario_label, call_list)

        if run_parallel:
            # Performance note: it takes about 3 seconds to run this block even with nothing in the iterable, I guess just from launching the worker pool
            if len(parallel_iterable) > 0:
                L.critical('About to run in parallel' + str(hb.pp(parallel_iterable)))

                worker_pool = multiprocessing.Pool(p.num_workers)  # NOTE, worker pool and results are LOCAL variabes so that they aren't pickled when we pass the project object.

                finished_results = []

                result = worker_pool.starmap_async(gtap_invest_integration_functions.run_gtap_cmf, parallel_iterable)
                for i in result.get():
                    try:
                        finished_results.append(i)
                    except:
                        finished_results.append('FAILED!!!   ' + str(i))
                worker_pool.close()
                worker_pool.join()

        current_date = hb.pretty_time('year_month_day_hyphens')


        # Copy the time-stamped gtap2 results to a non-time-stamped version to clarify which one to use for eg plotting.

        dated_expected_files = [os.path.join(p.cur_dir, current_date +'_GTAP_Results.csv'), os.path.join(p.cur_dir, current_date +'_GTAP_AEZ_LCOVER_ha.csv')]
        expected_files = [p.gtap2_results_path, p.gtap2_land_use_change_path]

        for c, file_path in enumerate(dated_expected_files):
            if hb.path_exists(file_path):
                hb.copy_shutil_flex(file_path, expected_files[c])

        gtap2_aez_results_exist = all([True if hb.path_exists(i) else False for i in expected_files])

        if not gtap2_aez_results_exist:
            # Now call the R code to pull these
            src_r_postsim_script_path = os.path.join(p.gtap2_aez_invest_local_model_dir, 'postsims', '01_output_csv.r')

            # Create a local copy of the R file with modifications specific to this run (i.e., the shanging the workspace)
            r_postsim_script_path = os.path.join(p.gtap2_aez_invest_local_model_dir, 'postsims', '01_output_csv_local.r')

            L.info('Starting r script at ' + str(r_postsim_script_path) + ' to create output files.')

            os.makedirs(os.path.join(os.path.split(src_r_postsim_script_path)[0], 'temp'), exist_ok=True)
            os.makedirs(os.path.join(os.path.split(src_r_postsim_script_path)[0], 'temp', 'merge'), exist_ok=True)

            # TODOO, This is a silly duplication of non-small files that could be eliminated. originally it was in so that i didn't have to modify Uris' r code.
            # TODOO Also, most of this R code is just running har2csv.exe, which is a license-constrained gempack file. replace with python har2csv
            hb.copy_file_tree_to_new_root(os.path.join(p.gtap2_aez_invest_local_model_dir, 'work'), os.path.join(p.gtap2_aez_invest_local_model_dir, 'postsims', 'in', 'gtap'))

            gtap_invest_integration_functions.generate_postsims_r_script_file(src_r_postsim_script_path, r_postsim_script_path)

            hb.execute_r_script(p.r_executable_path, os.path.abspath(r_postsim_script_path))

            # The two CSVs generated by the script file are key outputs. Copy them to the cur_dir root as well as the output dir
            files_to_copy = [os.path.join(p.gtap2_aez_invest_local_model_dir, 'postsims', 'out', os.path.split(i)[1]) for i in dated_expected_files]

            for c, file_path in enumerate(files_to_copy):
                hb.copy_shutil_flex(file_path, os.path.join(p.cur_dir, os.path.split(file_path)[1]), verbose=True)
                hb.copy_shutil_flex(file_path, os.path.join(p.output_dir, 'gtap2_aez', os.path.split(file_path)[1]), verbose=True)
                hb.copy_shutil_flex(file_path, os.path.join(p.cur_dir, os.path.split(expected_files[c])[1]), verbose=True)

def gtap2_aez(p):
    """Run a precompiled GTAPAEZ.exe file by calling the and a cmf file.
    Additionally and somewhat confusingly, this is also where I enable the ability to run GTAP-AEZ via an external process. The way this works is it copies the
    code for GTAP-InVEST into the ProjectFlow p.cur_dir, runs it, and saves a result in a logical place. HOWEVER, if the process is instead run via a batch file
    the way it was done before, it saves it to a different location. Subsequent tasks choose the manual .bat result over the internal one IF IT EXISTS. So
    if you have previously run a manual run and now want it to be in ProjectFlow, just make sure to delete tehe gtap_result directory.

    TODOO: Generate specific tab files for each scenario dynamically based on a template. Might not be needed?

    """
    p.gtap2_results_path = os.path.join(p.cur_dir, 'GTAP_Results.csv')
    p.gtap2_land_use_change_path = os.path.join(p.cur_dir, 'GTAP_AEZ_LCOVER_ha.csv')

    p.gtap2_aez_invest_local_model_dir = os.path.join(p.cur_dir, p.gtap_aez_invest_release_string)
    if p.run_this:

        gtap_run_copy_exclude_list = [i for i in p.gtap_combined_policy_scenario_labels if not any([i in j for j in p.gtap2_scenario_labels])]
        gtap_run_copy_include_list = p.gtap2_scenario_labels + ['GTAPAEZ', 'gtapaez', '2014_21_BAU_supp','sets', '2014_21_BAU.upd', 'Default.prm', 'default.prm', 'base', 'Base', '2021_2030', 'TOHAT2']

        # Extract a gtap-aez-invest zipfile into the curdir
        # TODOO rethink this overly complex structure and bring it 'in-house' rather than assuming releases coming out of purdue will match this pattern.
        if not hb.path_exists(p.gtap2_aez_invest_local_model_dir):
            if not hb.path_exists(p.gtap_aez_invest_code_dir):
                L.info('Unzipping all files in ' + p.gtap_aez_invest_zipfile_path + ' to ' + p.gtap2_aez_invest_local_model_dir)

                hb.unzip_file(p.gtap_aez_invest_zipfile_path, p.cur_dir, verbose=False)
            else:
                L.info('Creating project-specific copy of GTAP files, copying from ' + p.gtap_aez_invest_code_dir + ' to ' + p.gtap2_aez_invest_local_model_dir)
                # HACK (sorta), I should ultimately get rid of this copy_file_tree approach as it is as easy to break as list_filtered_paths_recursively
                hb.copy_file_tree_to_new_root(p.gtap_aez_invest_code_dir, p.gtap2_aez_invest_local_model_dir)
                # hb.copy_file_tree_to_new_root(p.gtap_aez_invest_code_dir, p.gtap2_aez_invest_local_model_dir, include_strings=gtap_run_copy_include_list)

        # for a GTAP2 run, also need to copy in the results from gtap1 as it build on those solution files.
        # HACK (sorta), I should ultimately get rid of this copy_file_tree approach as it is as easy to break as list_filtered_paths_recursively
        hb.copy_file_tree_to_new_root(p.gtap1_aez_invest_local_model_dir, p.gtap2_aez_invest_local_model_dir, exclude_strings=gtap_run_copy_exclude_list, exclude_extensions='.r', verbose=False)

        # Copy in (and replace Uris' old) shock files from the process_shockfile dir
        src_dir = os.path.join(p.gtap2_process_shockfiles_dir, '00_InVEST_proc', 'out')
        dst_dir = os.path.join(p.cur_dir, '04_20_2021_GTAP_AEZ_INVEST')


        L.warning('HACK, this will only work if the gtap labels are formatted exactly right, ie the year is the first 4 digits before the underscore.')
        for gtap_scenario_label in p.gtap2_scenario_labels:
            current_year = int(gtap_scenario_label.split('_')[0][0:2] + gtap_scenario_label.split('_')[1])
            current_scenario_label = '_'.join(gtap_scenario_label.split('_')[2:-1])

            for shockfile_label in p.shockfile_labels:
                src_file = os.path.join(src_dir, 'gtap2_rcp45_ssp2_' + str(current_year) + '_' + current_scenario_label + '_' + shockfile_label + '.txt')
                dst_file = os.path.join(dst_dir, 'X' + str(current_year) + '_' + current_scenario_label + '_' + shockfile_label + '.txt')
                hb.copy_shutil_flex(src_file, dst_file)


        # %% UNABLE TO OPEN EXISTING FILE '2021_30_BAU_rigid_noES_SUPP.har'.
        # if there is a file “2021_30_BAU_noES_supp.HAR” you can just duplicate that and rename to “'2021_30_BAU_rigid_noES_SUPP.har”

        gtapaez_executable_abs_path = os.path.join(p.gtap2_aez_invest_local_model_dir, 'GTAPAEZ.exe')

        # # Define paths for the source cmf file (extracted from GTAP-AEZ integration zipfile) and the modified one that will be run
        # gtap_policy_baseline_scenario_label = str(p.base_year) + '_' + str(p.policy_base_year)[2:] + '_BAU'
        # gtap_policy_baseline_scenario_source_cmf_path = os.path.join(p.gtap2_aez_invest_local_model_dir, 'cmfs', gtap_policy_baseline_scenario_label + '.cmf')
        # gtap_policy_baseline_scenario_cmf_path = os.path.join(p.gtap2_aez_invest_local_model_dir, gtap_policy_baseline_scenario_label + '_local.cmf')
        # gtap_policy_baseline_solution_file_path = os.path.join(p.cur_dir, p.gtap_aez_invest_release_string, 'work', gtap_policy_baseline_scenario_label + '.sl4')
        #
        # L.info('gtap_policy_baseline_scenario_cmf_path', gtap_policy_baseline_scenario_cmf_path)
        # L.info('gtap_policy_baseline_solution_file_path', gtap_policy_baseline_solution_file_path)
        #
        # if not hb.path_exists(gtap_policy_baseline_solution_file_path, verbose=True):
        #     # Generate a new cmf file with updated paths.
        #     gtap_invest_integration_functions.generate_policy_baseline_cmf_file(gtap_policy_baseline_scenario_source_cmf_path, gtap_policy_baseline_scenario_cmf_path)
        #
        #     # Run the gtap executable pointing to the new cmf file
        #     call_list = [gtapaez_executable_abs_path, '-cmf', gtap_policy_baseline_scenario_cmf_path]
        #     gtap_invest_integration_functions.run_gtap_cmf(gtap_policy_baseline_scenario_label, call_list)

        run_parallel = 1
        parallel_iterable = []
        # HACK: Note that this is not the set of scenario labels used elsewhere.
        # ORIGINAL: for gtap_scenario_label in p.gtap2_scenario_labels:
        # subset_scenarios = [i for i in p.gtap2_scenario_labels if i in p.gtap_bau_and_pesgc_label]
        # subset_scenarios = ['2021_30_' + i + '_allES' for i in p.gtap_bau_and_pesgc_label]
        # for gtap_scenario_label in subset_scenarios:
        for gtap_scenario_label in p.gtap2_scenario_labels:

            current_scenario_source_cmf_path = os.path.join(p.gtap2_aez_invest_local_model_dir, 'cmfs', gtap_scenario_label + '.cmf')
            current_scenario_cmf_path = os.path.join(p.gtap2_aez_invest_local_model_dir, gtap_scenario_label + '_local.cmf')

            current_solution_file_path = os.path.join(p.cur_dir, p.gtap_aez_invest_release_string, 'work', gtap_scenario_label + '.sl4')

            # Hack to fix uris scenario typo

            src = os.path.join(p.gtap2_aez_invest_local_model_dir, 'cmfs', '2021_30_SR_RnD_20p_PESGB_30_allES.cmf')
            dst = os.path.join(p.gtap2_aez_invest_local_model_dir, 'cmfs', '2021_30_SR_RnD_20p_PESGC_30_allES.cmf')
            if hb.path_exists(src):
                hb.copy_shutil_flex(src, dst)

            # Hack to rename shocks.txt typo
            for i in [os.path.join(p.gtap2_aez_invest_local_model_dir, "2021_30_SR_Land_allES_shocks.txt"), os.path.join(p.gtap2_aez_invest_local_model_dir, "2021_30_SR_Land_noES_shocks.txt")]:
                if hb.path_exists(i):
                    hb.copy_shutil_flex(i, i.replace('shocks.txt', 'shock.txt'))

            if '2021_30_SR_RnD_20p_PESGC_30_allES' in current_solution_file_path:
                current_solution_file_path = current_solution_file_path.replace('2021_30_SR_RnD_20p_PESGC_30_allES', '2021_30_SR_RnD_20p_PESGB_30_allES')

            if not hb.path_exists(current_solution_file_path, verbose=True):
                # Generate a new cmf file with updated paths.
                # Currently this just uses the policy_baseline version.
                gtap_invest_integration_functions.generate_policy_baseline_cmf_file(current_scenario_source_cmf_path, current_scenario_cmf_path)


                # Run the gtap executable pointing to the new cmf file
                # FOR MANUAL: Open command prompt at release dir, enter command: ".\GTAPAEZ.exe -cmf 2021_30_PESGC_allES_local.cmf"
                call_list = [gtapaez_executable_abs_path, '-cmf', current_scenario_cmf_path]
                parallel_iterable.append(tuple([gtap_scenario_label, call_list]))


                # HACK
                do_validation = 0
                if do_validation:
                    L.critical('Running validation as well. This means running uris old shockfiles for each scenario from the ECN release')
                    current_scenario_validation_cmf_path = os.path.join(p.gtap2_aez_invest_local_model_dir, gtap_scenario_label + '_validation.cmf')
                    gtap_invest_integration_functions.generate_policy_baseline_cmf_file(current_scenario_cmf_path, current_scenario_validation_cmf_path, run_on_validation=True)
                    print('current_scenario_validation_cmf_path', current_scenario_validation_cmf_path)
                    call_list = [gtapaez_executable_abs_path, '-cmf', current_scenario_validation_cmf_path]
                    validation_label = gtap_scenario_label + '_validation'
                    if '2021_30_SR_Land_allES_validation' not in current_scenario_validation_cmf_path:
                        parallel_iterable.append(tuple([validation_label, call_list]))

                # If the model is run sequentially, just call it here.
                if not run_parallel:
                    L.critical('About to run non-parallel' + str(parallel_iterable))

                    gtap_invest_integration_functions.run_gtap_cmf(gtap_scenario_label, call_list)

        if run_parallel:
            # Performance note: it takes about 3 seconds to run this block even with nothing in the iterable, I guess just from launching the worker pool
            if len(parallel_iterable) > 0:
                L.critical('About to run in parallel' + str(hb.pp(parallel_iterable)))

                worker_pool = multiprocessing.Pool(p.num_workers)  # NOTE, worker pool and results are LOCAL variabes so that they aren't pickled when we pass the project object.

                finished_results = []

                result = worker_pool.starmap_async(gtap_invest_integration_functions.run_gtap_cmf, parallel_iterable)
                for i in result.get():
                    try:
                        finished_results.append(i)
                    except:
                        finished_results.append('FAILED!!!   ' + str(i))
                worker_pool.close()
                worker_pool.join()

        current_date = hb.pretty_time('year_month_day_hyphens')


        # Copy the time-stamped gtap2 results to a non-time-stamped version to clarify which one to use for eg plotting.

        dated_expected_files = [os.path.join(p.cur_dir, current_date +'_GTAP_Results.csv'), os.path.join(p.cur_dir, current_date +'_GTAP_AEZ_LCOVER_ha.csv')]
        expected_files = [p.gtap2_results_path, p.gtap2_land_use_change_path]

        for c, file_path in enumerate(dated_expected_files):
            if hb.path_exists(file_path):
                hb.copy_shutil_flex(file_path, expected_files[c])

        gtap2_aez_results_exist = all([True if hb.path_exists(i) else False for i in expected_files])

        if not gtap2_aez_results_exist:
            # Now call the R code to pull these
            src_r_postsim_script_path = os.path.join(p.gtap2_aez_invest_local_model_dir, 'postsims', '01_output_csv.r')

            # Create a local copy of the R file with modifications specific to this run (i.e., the shanging the workspace)
            r_postsim_script_path = os.path.join(p.gtap2_aez_invest_local_model_dir, 'postsims', '01_output_csv_local.r')

            L.info('Starting r script at ' + str(r_postsim_script_path) + ' to create output files.')

            os.makedirs(os.path.join(os.path.split(src_r_postsim_script_path)[0], 'temp'), exist_ok=True)
            os.makedirs(os.path.join(os.path.split(src_r_postsim_script_path)[0], 'temp', 'merge'), exist_ok=True)

            # TODOO, This is a silly duplication of non-small files that could be eliminated. originally it was in so that i didn't have to modify Uris' r code.
            # TODOO Also, most of this R code is just running har2csv.exe, which is a license-constrained gempack file. replace with python har2csv
            hb.copy_file_tree_to_new_root(os.path.join(p.gtap2_aez_invest_local_model_dir, 'work'), os.path.join(p.gtap2_aez_invest_local_model_dir, 'postsims', 'in', 'gtap'))
            print('src_r_postsim_script_path', src_r_postsim_script_path)
            gtap_invest_integration_functions.generate_postsims_r_script_file(src_r_postsim_script_path, r_postsim_script_path)

            hb.execute_r_script(p.r_executable_path, os.path.abspath(r_postsim_script_path))

            # The two CSVs generated by the script file are key outputs. Copy them to the cur_dir root as well as the output dir
            files_to_copy = [os.path.join(p.gtap2_aez_invest_local_model_dir, 'postsims', 'out', os.path.split(i)[1]) for i in dated_expected_files]

            for c, file_path in enumerate(files_to_copy):
                hb.copy_shutil_flex(file_path, os.path.join(p.cur_dir, os.path.split(file_path)[1]), verbose=True)
                hb.copy_shutil_flex(file_path, os.path.join(p.output_dir, 'gtap2_aez', os.path.split(file_path)[1]), verbose=True)
                hb.copy_shutil_flex(file_path, os.path.join(p.cur_dir, os.path.split(expected_files[c])[1]), verbose=True)




def gtap2_extracts_from_solution(p):

    if p.run_this:
        hb.create_directories(os.path.join(p.cur_dir, 'raw_sl4_extracts'))

        # Extract .sl4 format to a raw_csv
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:

                    # Hack to fix rename
                    if policy_scenario_label == 'SR_RnD_20p_PESGC_30':
                        policy_scenario_label = policy_scenario_label.replace('SR_RnD_20p_PESGC_30', 'SR_RnD_20p_PESGB_30')

                    gtap_scenario_label = '2021_30_' + policy_scenario_label + '_allES'
                    gtap_sl4_path_no_extension = os.path.join(p.gtap2_aez_invest_local_model_dir, 'work', gtap_scenario_label)
                    gtap_sl4_path = gtap_sl4_path_no_extension + '.sl4'

                    current_all_vars_output_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label + '_all_vars_raw.csv')
                    current_select_vars_output_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label + '_select_vars_raw.csv')
                    required_paths = [current_all_vars_output_path, current_select_vars_output_path]
                    if not all([hb.path_exists(i) for i in required_paths]) and hb.path_exists(gtap_sl4_path):
                        selected_vars = ['qgdp', 'pgdp']

                        gtap_invest_integration_functions.extract_raw_csv_from_sl4(gtap_sl4_path, current_all_vars_output_path)
                        # gtap_invest_integration_functions.extract_raw_csv_from_sl4(gtap_sl4_path, current_all_vars_output_path, additional_options=['-alltsbt'])
                        gtap_invest_integration_functions.extract_raw_csv_from_sl4(gtap_sl4_path, current_select_vars_output_path, vars_to_extract=selected_vars)

        # Keep track of output dimensions to see if any scenarios have different number of outputs (which you may want to troubleshoot)
        run_dimensions = {}

        # Loop through generated raw CSVs and extract more useful results.
        for luh_scenario_label in p.luh_scenario_labels:
            run_dimensions[luh_scenario_label] = {}
            for scenario_year in p.scenario_years:
                run_dimensions[luh_scenario_label][scenario_year] = {}
                for policy_scenario_label in p.policy_scenario_labels:

                    run_dimensions[luh_scenario_label][scenario_year][policy_scenario_label] = {}
                    gtap_scenario_label = '2021_30_' + policy_scenario_label + '_allES'
                    raw_csv_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label + '_all_vars_raw.csv')

                    # HACK yet again to rename.
                    if '2021_30_SR_RnD_20p_PESGC_30_allES' in gtap_scenario_label:
                        raw_csv_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label.replace('2021_30_SR_RnD_20p_PESGC_30_allES', '2021_30_SR_RnD_20p_PESGB_30_allES') + '_all_vars_raw.csv')

                    hb.create_directories(os.path.join(p.cur_dir, 'solution_shapes'))
                    solution_summary_path = os.path.join(p.cur_dir, 'solution_shapes', gtap_scenario_label + '_solution_variables_description.csv')
                    hb.create_directories(solution_summary_path)
                    if not hb.path_exists(solution_summary_path):
                        output_lines = []
                        var_names = []
                        var_descriptions = []
                        shapes = []
                        axes = []
                        with open(raw_csv_path, 'r') as fp:
                            for line in fp:
                                if '! Variable' in line:
                                    if ' of size ' not in line:
                                        var_name = line.split(' ')[3]
                                        var_desc = line.split('#')[1]

                                        var_names.append(var_name)
                                        var_descriptions.append(var_desc)
                                    else:
                                        if '(' in line:
                                            current_axes = line.split('(')[1].split(')')[0]
                                            shape = line.split(' size ')[1]
                                            shape = shape.replace('\n', '')
                                        else:
                                            current_axes = 'singular'
                                            shape = 1

                                        axes.append(current_axes)
                                        shapes.append(shape)

                            solution_variables_description_df = pd.DataFrame(data={'var_name': var_names, 'var_desc': var_descriptions, 'axes': axes, 'shape': shapes})
                            solution_variables_description_df['axes_with_shape'] = solution_variables_description_df['axes'].astype(str) + '_' + solution_variables_description_df['shape'].astype(str)
                            L.info('Generated solution variables description dataframe and saved it to ' + str(solution_summary_path))
                            L.info(solution_variables_description_df)

                            # Note, this method of only calculating it once assumes every solution has all the same variables for all scenarios. That seems wrong.
                            solution_variables_description_df.to_csv(solution_summary_path)
                    else:
                        solution_variables_description_df = pd.read_csv(solution_summary_path)

                    unique_axes_with_shape = hb.enumerate_array_as_odict(np.asarray(solution_variables_description_df['axes'], dtype='str'))
                    for shape, count in unique_axes_with_shape.items():
                        run_dimensions[luh_scenario_label][scenario_year][policy_scenario_label][shape] = count

        # hb.pp(run_dimensions)

        # validate that all are the same shape (optional)
        assert_all_solutions_identical_shapes = True
        if assert_all_solutions_identical_shapes:
            initial_v3 = None
            for k, v in run_dimensions.items():
                for k2, v2 in v.items():
                    for k3, v3 in v2.items():
                        if initial_v3 is None:
                            initial_v3 = v3
                        else:
                            if v3 != initial_v3:
                                L.critical('Solution files are not the same output dimensions!!! RUN!!!!')
                                L.info(initial_v3)
                                L.info(v3)
                                assert NameError('Solution files are not the same output dimensions!!! RUN!!!!')





        # Loop through generated raw CSVs and make well-formated tables
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    gtap_scenario_label = '2021_30_' + policy_scenario_label + '_allES'

                    # Hack yet again to rename
                    if '2021_30_SR_RnD_20p_PESGC_30_allES' in gtap_scenario_label:
                        gtap_scenario_label = gtap_scenario_label.replace('2021_30_SR_RnD_20p_PESGC_30_allES', '2021_30_SR_RnD_20p_PESGB_30_allES')

                    raw_csv_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label + '_all_vars_raw.csv')

                    write_dir = os.path.join(p.cur_dir, 'output_tables', luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label)
                    hb.create_directories(write_dir)

                    gtap_invest_integration_functions.extract_vertical_csvs_from_multidimensional_sl4_csv(raw_csv_path, write_dir, gtap_scenario_label)

        # Merge the singular variables across scenarios
        df = None
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:

                for policy_scenario_label in p.policy_scenario_labels:
                    gtap_scenario_label = '2021_30_' + policy_scenario_label + '_allES'

                    # Hack yet again to rename
                    if '2021_30_SR_RnD_20p_PESGC_30_allES' in gtap_scenario_label:
                        gtap_scenario_label = gtap_scenario_label.replace('2021_30_SR_RnD_20p_PESGC_30_allES', '2021_30_SR_RnD_20p_PESGB_30_allES')


                    scenario_label = luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                    singular_csv_path = os.path.join(p.cur_dir, 'output_tables', scenario_label, gtap_scenario_label + '_singular_vars.csv')

                    current_df = pd.read_csv(singular_csv_path)
                    if df is None:
                        df = current_df
                        df.rename(columns={'value': scenario_label}, inplace=True) #GET BACK SUMMARY
                    else:
                        current_df[scenario_label] = current_df['value']
                        df = df.merge(current_df[['var_name', scenario_label]] , left_on='var_name', right_on='var_name')
        df.to_csv(os.path.join(p.cur_dir, 'global_results_across_scenarios.csv'), index=False)

        # Merge single variables against region-scenarios
        df = None

        for var in ['qgdp', 'pgdp']:
            for luh_scenario_label in p.luh_scenario_labels:
                for scenario_year in p.scenario_years:

                    for policy_scenario_label in p.policy_scenario_labels:
                        gtap_scenario_label = '2021_30_' + policy_scenario_label + '_allES'

                        # Hack yet again to rename
                        if '2021_30_SR_RnD_20p_PESGC_30_allES' in gtap_scenario_label:
                            gtap_scenario_label = gtap_scenario_label.replace('2021_30_SR_RnD_20p_PESGC_30_allES', '2021_30_SR_RnD_20p_PESGB_30_allES')


                        scenario_label = luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        singular_csv_path = os.path.join(p.cur_dir, 'output_tables', scenario_label, gtap_scenario_label + '_one_dim_vars.csv')

                        current_df = pd.read_csv(singular_csv_path)

                        if df is None:
                            current_df = current_df[current_df['var_name'] == var]
                            current_df[scenario_label] = current_df['value']
                            df = current_df[['dim_value', scenario_label]]
                            # df.rename(columns={'value': scenario_label}, inplace=True) #GET BACK SUMMARY
                        else:
                            current_df[scenario_label] = current_df['value']
                            current_df = current_df[current_df['var_name'] == var]
                            # current_df = current_df[['dim_value', 'value']]
                            df = df.merge(current_df[['dim_value', scenario_label]] , left_on='dim_value', right_on='dim_value')
            df.to_csv(os.path.join(p.cur_dir, var + '_across_scenarios_and_regions.csv'), index=False)


def gtap2_valdation_extracts_from_solution(p):

    if p.run_this:
        hb.create_directories(os.path.join(p.cur_dir, 'raw_sl4_extracts'))

        # Extract .sl4 format to a raw_csv
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:

                    # Hack to fix rename
                    if policy_scenario_label == 'SR_RnD_20p_PESGC_30':
                        policy_scenario_label = policy_scenario_label.replace('SR_RnD_20p_PESGC_30', 'SR_RnD_20p_PESGB_30')

                    gtap_scenario_label = '2021_30_' + policy_scenario_label + '_allES_validation'
                    gtap_sl4_path_no_extension = os.path.join(p.gtap2_aez_invest_local_model_dir, 'work', gtap_scenario_label)
                    gtap_sl4_path = gtap_sl4_path_no_extension + '.sl4'

                    current_all_vars_output_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label + '_all_vars_raw.csv')
                    current_select_vars_output_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label + '_select_vars_raw.csv')
                    required_paths = [current_all_vars_output_path, current_select_vars_output_path]

                    hb.pp(required_paths)
                    if not all([hb.path_exists(i) for i in required_paths]) and hb.path_exists(gtap_sl4_path):
                        selected_vars = ['qgdp', 'pgdp', ]

                        gtap_invest_integration_functions.extract_raw_csv_from_sl4(gtap_sl4_path, current_all_vars_output_path)
                        gtap_invest_integration_functions.extract_raw_csv_from_sl4(gtap_sl4_path, current_select_vars_output_path, vars_to_extract=selected_vars)

        do_shapes_analysis = 0
        if do_shapes_analysis:
            # Keep track of output dimensions to see if any scenarios have different number of outputs (which you may want to troubleshoot)
            run_dimensions = {}

            # Loop through generated raw CSVs and extract more useful results.
            for luh_scenario_label in p.luh_scenario_labels:
                run_dimensions[luh_scenario_label] = {}
                for scenario_year in p.scenario_years:
                    run_dimensions[luh_scenario_label][scenario_year] = {}
                    for policy_scenario_label in p.policy_scenario_labels:

                        run_dimensions[luh_scenario_label][scenario_year][policy_scenario_label] = {}
                        gtap_scenario_label = '2021_30_' + policy_scenario_label + '_allES_validation'
                        raw_csv_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label + '_all_vars_raw.csv')

                        # HACK yet again to rename.
                        if '2021_30_SR_RnD_20p_PESGC_30_allES_validation' in gtap_scenario_label:
                            raw_csv_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label.replace('2021_30_SR_RnD_20p_PESGC_30_allES_validation', '2021_30_SR_RnD_20p_PESGB_30_allES_validation') + '_all_vars_raw.csv')

                        hb.create_directories(os.path.join(p.cur_dir, 'solution_shapes'))
                        solution_summary_path = os.path.join(p.cur_dir, 'solution_shapes', gtap_scenario_label + '_solution_variables_description.csv')
                        if not hb.path_exists(solution_summary_path) and hb.path_exists(raw_csv_path):
                            output_lines = []
                            var_names = []
                            var_descriptions = []
                            shapes = []
                            axes = []
                            with open(raw_csv_path, 'r') as fp:
                                for line in fp:
                                    if '! Variable' in line:
                                        if ' of size ' not in line:
                                            var_name = line.split(' ')[3]
                                            var_desc = line.split('#')[1]

                                            var_names.append(var_name)
                                            var_descriptions.append(var_desc)
                                        else:
                                            if '(' in line:
                                                current_axes = line.split('(')[1].split(')')[0]
                                                shape = line.split(' size ')[1]
                                                shape = shape.replace('\n', '')
                                            else:
                                                current_axes = 'singular'
                                                shape = 1

                                            axes.append(current_axes)
                                            shapes.append(shape)

                                solution_variables_description_df = pd.DataFrame(data={'var_name': var_names, 'var_desc': var_descriptions, 'axes': axes, 'shape': shapes})
                                solution_variables_description_df['axes_with_shape'] = solution_variables_description_df['axes'].astype(str) + '_' + solution_variables_description_df['shape'].astype(str)
                                L.info('Generated solution variables description dataframe and saved it to ' + str(solution_summary_path))
                                L.info(solution_variables_description_df)

                                # Note, this method of only calculating it once assumes every solution has all the same variables for all scenarios. That seems wrong.
                                solution_variables_description_df.to_csv(solution_summary_path)
                        else:
                            solution_variables_description_df = pd.read_csv(solution_summary_path)

                        unique_axes_with_shape = hb.enumerate_array_as_odict(np.asarray(solution_variables_description_df['axes'], dtype='str'))
                        for shape, count in unique_axes_with_shape.items():
                            run_dimensions[luh_scenario_label][scenario_year][policy_scenario_label][shape] = count

            # hb.pp(run_dimensions)

            # validate that all are the same shape (optional)
            assert_all_solutions_identical_shapes = True
            if assert_all_solutions_identical_shapes:
                initial_v3 = None
                for k, v in run_dimensions.items():
                    for k2, v2 in v.items():
                        for k3, v3 in v2.items():
                            if initial_v3 is None:
                                initial_v3 = v3
                            else:
                                if v3 != initial_v3:
                                    L.critical('Solution files are not the same output dimensions!!! RUN!!!!')
                                    L.info(initial_v3)
                                    L.info(v3)
                                    assert NameError('Solution files are not the same output dimensions!!! RUN!!!!')





        # Loop through generated raw CSVs and make well-formated tables
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    gtap_scenario_label = '2021_30_' + policy_scenario_label + '_allES_validation'

                    # Hack yet again to rename
                    if '2021_30_SR_RnD_20p_PESGC_30_allES_validation' in gtap_scenario_label:
                        gtap_scenario_label = gtap_scenario_label.replace('2021_30_SR_RnD_20p_PESGC_30_allES_validation', '2021_30_SR_RnD_20p_PESGB_30_allES_validation')

                    raw_csv_path = os.path.join(p.cur_dir, 'raw_sl4_extracts', gtap_scenario_label + '_all_vars_raw.csv')

                    write_dir = os.path.join(p.cur_dir, 'output_tables', luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label)
                    hb.create_directories(write_dir)

                    if hb.path_exists(raw_csv_path):
                        gtap_invest_integration_functions.extract_vertical_csvs_from_multidimensional_sl4_csv(raw_csv_path, write_dir, gtap_scenario_label)

        # Merge the singular variables across scenarios
        df = None
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:

                for policy_scenario_label in p.policy_scenario_labels:
                    gtap_scenario_label = '2021_30_' + policy_scenario_label + '_allES_validation'

                    # Hack yet again to rename
                    if '2021_30_SR_RnD_20p_PESGC_30_allES_validation' in gtap_scenario_label:
                        gtap_scenario_label = gtap_scenario_label.replace('2021_30_SR_RnD_20p_PESGC_30_allES_validation', '2021_30_SR_RnD_20p_PESGB_30_allES_validation')


                    scenario_label = luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                    singular_csv_path = os.path.join(p.cur_dir, 'output_tables', scenario_label, gtap_scenario_label + '_singular_vars.csv')

                    if hb.path_exists(singular_csv_path):
                        current_df = pd.read_csv(singular_csv_path)
                        if df is None:
                            df = current_df
                            df.rename(columns={'value': scenario_label}, inplace=True) #GET BACK SUMMARY
                        else:
                            current_df[scenario_label] = current_df['value']
                            df = df.merge(current_df[['var_name', scenario_label]] , left_on='var_name', right_on='var_name')
        df.to_csv(os.path.join(p.cur_dir, 'global_results_across_scenarios.csv'), index=False)

        # Merge single variables against region-scenarios
        df = None

        for var in ['qgdp', 'pgdp']:
            for luh_scenario_label in p.luh_scenario_labels:
                for scenario_year in p.scenario_years:

                    for policy_scenario_label in p.policy_scenario_labels:
                        gtap_scenario_label = '2021_30_' + policy_scenario_label + '_allES_validation'

                        # Hack yet again to rename
                        if '2021_30_SR_RnD_20p_PESGC_30_allES_validation' in gtap_scenario_label:
                            gtap_scenario_label = gtap_scenario_label.replace('2021_30_SR_RnD_20p_PESGC_30_allES_validation', '2021_30_SR_RnD_20p_PESGB_30_allES_validation')


                        scenario_label = luh_scenario_label + '_' + str(scenario_year) + '_' + policy_scenario_label
                        singular_csv_path = os.path.join(p.cur_dir, 'output_tables', scenario_label, gtap_scenario_label + '_one_dim_vars.csv')

                        if hb.path_exists(singular_csv_path):
                            current_df = pd.read_csv(singular_csv_path)

                            if df is None:
                                current_df = current_df[current_df['var_name'] == var]
                                current_df[scenario_label] = current_df['value']
                                df = current_df[['dim_value', scenario_label]]
                                # df.rename(columns={'value': scenario_label}, inplace=True) #GET BACK SUMMARY
                            else:
                                current_df[scenario_label] = current_df['value']
                                current_df = current_df[current_df['var_name'] == var]
                                # current_df = current_df[['dim_value', 'value']]
                                df = df.merge(current_df[['dim_value', scenario_label]] , left_on='dim_value', right_on='dim_value')
            df.to_csv(os.path.join(p.cur_dir, var + '_across_scenarios_and_regions.csv'), index=False)


def gtap2_results_as_tables(p):
    " HUGE FERENCE FROM extracts_from_solution method is this uses URIS' extraction technique, whish was better because it was more general."
    if p.run_this:

        L.info('Reading gtap_aez results file at ' + p.gtap2_results_path)
        results_df = pd.read_csv(p.gtap2_results_path)

        vars_to_plot = ['qgdp']
        for var in vars_to_plot:

            # add in baseline value first.
            baseline_gtap_scenario_label = '2014_21_BAU'
            current_scenario_label = baseline_gtap_scenario_label

            subset_scenario_var = results_df.loc[(results_df['SCENARIO'] == current_scenario_label) & (results_df['VAR'] == var)]
            df = pd.pivot_table(subset_scenario_var, index='REG', columns='TYPE', values='Value')
            rename_dict = {i: current_scenario_label + '_' + i for i in df.columns}
            df.rename(columns=rename_dict, inplace=True)

            # Also will be converting to to GTAP-InVEST style labels
            baseline_label = 'baseline_2014_sum'
            original_baseline_label = 'gtap1_baseline_2014_sum'
            baseline_policy_label = 'baseline_2021_sum'
            original_baseline_policy_label = 'gtap1_baseline_2021_sum'
            rename_dict = {}
            rename_dict[baseline_gtap_scenario_label + '_BaseValue'] = baseline_label

            for luh_scenario_label in p.luh_scenario_labels:
                for scenario_year in p.scenario_years:
                    for policy_scenario_label in p.policy_scenario_labels:

                        # Get labels of current scenario in SEALS and GTAP format.
                        current_scenario_label = '_'.join([luh_scenario_label, str(scenario_year), policy_scenario_label])
                        current_gtap1_scenario_label = '2021_30_' + policy_scenario_label + '_noES'
                        current_gtap1_scenario_label = current_gtap1_scenario_label.replace('20p_PESGC_30', '20p_PESGB_30')
                        current_gtap2_scenario_label = '2021_30_' + policy_scenario_label + '_allES'
                        current_gtap2_scenario_label = current_gtap2_scenario_label.replace('20p_PESGC_30', '20p_PESGB_30')

                        # Get results without ES shocks
                        subset_scenario_var = results_df.loc[(results_df['SCENARIO'] == current_gtap1_scenario_label) & (results_df['VAR'] == var)]
                        subset_pivoted = pd.pivot_table(subset_scenario_var, index='REG', columns='TYPE', values='Value')
                        rename_dict = {i: current_scenario_label + '_noES_' + i for i in subset_pivoted.columns}
                        subset_pivoted.rename(columns=rename_dict, inplace=True)
                        df = df.merge(subset_pivoted, left_on='REG', right_on='REG')

                        # Get results without ES shocks
                        subset_scenario_var = results_df.loc[(results_df['SCENARIO'] == current_gtap2_scenario_label) & (results_df['VAR'] == var)]
                        subset_pivoted = pd.pivot_table(subset_scenario_var, index='REG', columns='TYPE', values='Value')
                        rename_dict = {i: current_scenario_label + '_allES_' + i for i in subset_pivoted.columns}
                        subset_pivoted.rename(columns=rename_dict, inplace=True)
                        df = df.merge(subset_pivoted, left_on='REG', right_on='REG')

            current_path = os.path.join(p.cur_dir, '_'.join([luh_scenario_label, str(scenario_year)]) + '_' + var + '_gtap_labels.csv')
            df.to_csv(current_path, index=True)

            current_path = os.path.join(p.cur_dir, '_'.join([luh_scenario_label, str(scenario_year)]) + '_' + var + '_gtap_labels_transposed.csv')
            df.T.to_csv(current_path, index=True)

            initial_labels_to_report = [baseline_label, baseline_policy_label]

            current_scenario_BaseValue_label = baseline_gtap_scenario_label + '_BaseValue'
            current_scenario_UpdValue_label = baseline_gtap_scenario_label+ '_UpdValue'
            baseline_policy_label_2014_prices = 'baseline_2021_with_2014_prices_sum'
            df[baseline_label] = df[current_scenario_BaseValue_label]
            df[baseline_policy_label_2014_prices] = df[current_scenario_UpdValue_label]
            # df[baseline_policy_label] = df[current_scenario_UpdValue_label]

            # LEARNING POINT, even lists are passed as references not copies. Needed to make this a copy or both would update.
            labels_to_report = initial_labels_to_report.copy()


            for luh_scenario_label in p.luh_scenario_labels:
                for scenario_year in p.scenario_years:
                    for policy_scenario_label in p.policy_scenario_labels:

                        current_scenario_label = '_'.join([luh_scenario_label, str(scenario_year), policy_scenario_label])



                        # HACK, this assumes BAU will be first in the list, which right now I am okay requiring.
                        if policy_scenario_label == 'BAU':

                            # For just the BAU (the first time through) also add the 2021 baseline as recalculated with 2021 prices and relabel it for general use.
                            current_scenario_BaseValue_label = current_scenario_label+ '_noES_BaseValue'
                            current_scenario_noes_UpdValue_label = current_scenario_label+ '_noES_UpdValue'
                            current_scenario_alles_UpdValue_label = current_scenario_label+ '_allES_UpdValue'
                            df[baseline_policy_label] = df[current_scenario_BaseValue_label]

                            # labels_to_report.append(baseline_label)
                            # labels_to_report.append(baseline_policy_label)

                            # Add scenario level label for noES (GTAP1 run)
                            current_scenario_UpdValue_label = current_scenario_label + '_noES_UpdValue'
                            labels_to_report.append(current_scenario_UpdValue_label)

                            # # Calculate and add difference from baseline label
                            # new_label = 'gtap1_' + current_scenario_label + '_minus_baseline'
                            # df[new_label] = df[current_scenario_UpdValue_label] - df[baseline_policy_label]
                            # labels_to_report.append(new_label)

                            # Add scenario level label for allES (GTAP2 run)
                            current_scenario_UpdValue_label = current_scenario_label + '_allES_UpdValue'
                            bau_noES_UpdValue_label = current_scenario_noes_UpdValue_label
                            bau_allES_UpdValue_label = current_scenario_alles_UpdValue_label
                            bau_noes_label = 'gtap1_' + current_scenario_label + '_sum'
                            df[bau_noes_label] = df[bau_noES_UpdValue_label]
                            labels_to_report.append(bau_noES_UpdValue_label)

                            bau_alles_label = 'gtap2_' + current_scenario_label + '_sum'
                            df[bau_alles_label] = df[bau_allES_UpdValue_label]
                            # labels_to_report.append(bau_alles_label)

                            # Calculate and add difference from baseline label
                            new_label = 'gtap1_' + current_scenario_label + '_minus_baseline'
                            df[new_label] = df[bau_noES_UpdValue_label] - df[baseline_policy_label]
                            labels_to_report.append(new_label)

                            # Calculate and add difference from baseline label
                            new_label = 'gtap2_' + current_scenario_label + '_minus_baseline'
                            df[new_label] = df[bau_allES_UpdValue_label] - df[baseline_policy_label]
                            labels_to_report.append(new_label)

                        else:
                            # Interesting choice, but do we actually care about the scenario where we protect ES but dont include their impact? excluding for now for parsimony

                            # Add scenario level label for allES (GTAP2 run)
                            current_scenario_UpdValue_label = current_scenario_label + '_allES_UpdValue'
                            new_label = 'gtap2_' + current_scenario_label + '_sum'
                            df[new_label] = df[current_scenario_UpdValue_label]
                            labels_to_report.append(new_label)

                            # Calculate and add difference from baseline label
                            new_label = 'gtap2_' + current_scenario_label + '_minus_baseline'
                            df[new_label] = df[current_scenario_UpdValue_label] - df[baseline_policy_label]
                            labels_to_report.append(new_label)

                            # Calculate and add difference from BAU label
                            new_label = 'gtap2_' + current_scenario_label + '_minus_bau'

                            df[new_label] = df[current_scenario_UpdValue_label] - df[bau_allES_UpdValue_label]

                            labels_to_report.append(new_label)


                    df_differences = df[[bau_noes_label, bau_alles_label] + labels_to_report]

                    current_path = os.path.join(p.cur_dir, '_'.join([luh_scenario_label, str(scenario_year)]) + '_' + var + '_minus_baseline.csv')
                    df_differences_out = df_differences[initial_labels_to_report + [bau_noes_label, bau_alles_label] + [i for i in df_differences.columns if '_minus_baseline' in i]]
                    df_differences_out.to_csv(current_path, index=True)

                    # Drop all the extra values and only compare policy performance versus bau
                    current_path = os.path.join(p.cur_dir, '_'.join([luh_scenario_label, str(scenario_year)]) + '_' + var + '_minus_bau.csv')
                    df_differences[initial_labels_to_report + [bau_noes_label, bau_alles_label] + [i for i in df_differences.columns if 'minus_bau' in i]].to_csv(current_path, index=True)



        L.info('Reading gtap_aez LULC results file at ' + p.gtap2_land_use_change_path)
        results_lulc_df = pd.read_csv(p.gtap2_land_use_change_path)



        carbon_df = pd.read_csv(p.carbon_shock_csv_path, index_col=0)



        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:

                drop_cols = ['pyramid_id', 'pyramid_ids_multiplied', 'gtap37v10_pyramid_id', 'aez_pyramid_id', 'gtap9_admin_id', 'GTAP226', 'GTAPv9a', 'GTAPv9p', 'AreaCode', 'minx', 'miny', 'maxx', 'maxy']
                carbon_by_region = carbon_df[[i for i in carbon_df.columns if i not in drop_cols]].groupby('gtap37v10_pyramid_name').sum()
                carbon_by_region[baseline_label] = carbon_by_region[original_baseline_label]
                # carbon_by_region[] = carbon_by_region[]



                for policy_scenario_label in p.policy_scenario_labels:

                    current_data_label = 'gtap2_' + '_'.join([luh_scenario_label, str(scenario_year), policy_scenario_label]) + '_sum'
                    current_new_label = 'gtap2_' + '_'.join([luh_scenario_label, str(scenario_year), policy_scenario_label]) + '_minus_baseline'

                    carbon_by_region[current_new_label] = carbon_by_region[current_data_label] - carbon_by_region[baseline_label]

                    # carbon_by_region.T.to_csv(os.path.join(p.cur_dir, '_'.join([luh_scenario_label, str(scenario_year)]) + '_carbon_minus_baseline.csv'), index=True)

                    if policy_scenario_label != 'BAU':
                        current_bau_label = 'gtap2_' + '_'.join([luh_scenario_label, str(scenario_year), 'BAU']) + '_sum'
                        current_data_label = 'gtap2_' + '_'.join([luh_scenario_label, str(scenario_year), policy_scenario_label]) + '_sum'
                        current_new_label = 'gtap2_' + '_'.join([luh_scenario_label, str(scenario_year), policy_scenario_label]) + '_minus_bau'

                        carbon_by_region[current_new_label] = carbon_by_region[current_data_label] - carbon_by_region[current_bau_label]

                carbon_by_region.to_csv(os.path.join(p.cur_dir, '_'.join([luh_scenario_label, str(scenario_year)]) + '_carbon.csv'), index=True)
                carbon_by_region.T.to_csv(os.path.join(p.cur_dir, '_'.join([luh_scenario_label, str(scenario_year)]) + '_carbon_transposed.csv'), index=True)
                carbon_by_region[[baseline_label] + [i for i in carbon_by_region.columns if 'minus_baseline' in i]].to_csv(os.path.join(p.cur_dir, '_'.join([luh_scenario_label, str(scenario_year)]) + '_carbon_minus_baseline.csv'), index=True)
                carbon_by_region[[baseline_label, current_bau_label] + [i for i in carbon_by_region.columns if 'minus_bau' in i]].to_csv(os.path.join(p.cur_dir, '_'.join([luh_scenario_label, str(scenario_year)]) + '_carbon_minus_bau.csv'), index=True)

def output_visualizations(p):
    import seaborn as sns

    sns.set_theme()

    if p.run_this:

        # Read in the vertical, full results produced in gtap2_aez task.
        df = pd.read_csv(p.gtap2_results_path)
        df_income_categories = pd.read_excel(os.path.join(p.model_base_data_dir, 'gtap37_continents_and_income_class.xlsx'), index_col='gtap37_region_label')

        dfp = df
        df = df.merge(df_income_categories, left_on='REG', right_index=True)

        lengths_dict, keys_dict = hb.get_unique_keys_from_vertical_dataframe(df)
        L.info('Analyized vertical results and got key-lengths of\n')
        L.info(hb.pp(lengths_dict, return_as_string=True))

        L.info('Analyized vertical results and got key-uniques of\n')
        L.info(hb.pp(keys_dict, return_as_string=True))

        scenario_names = {}
        scenario_names['2014_21_BAU'] = 'Baseline'
        scenario_names['2021_30_BAU_noES'] = 'BAU Excluding ES'
        scenario_names['2021_30_BAU_allES'] = 'BAU'
        scenario_names['2021_30_PESGC_allES'] = 'Global PES'
        scenario_names['2021_30_PESLC_allES'] = 'National PES'
        scenario_names['2021_30_SR_Land_allES'] = 'Subsidy repurposed to land'
        scenario_names['2021_30_SR_RnD_20p_allES'] = 'Subsidies repurposed to R&D'
        scenario_names['2021_30_SR_RnD_20p_PESGC_allES'] = 'Combined'

        vars_to_plot = {}
        vars_to_plot['qgdp'] = 'Real GDP'
        vars_to_plot['p_lc_cropl'] = 'Cropland Cover'
        vars_to_plot['pfactrl_land'] = 'Cropland Cover'
        vars_to_plot['EV'] = 'Economic Welfare via EV'  # DROPPED FOR NOW BECAUSE DOESNT HAVE UpdValue reported.
        vars_to_plot['p_qxw_pdyr'] = 'Paddy Rice Sector Export at World Mkt'
        vars_to_plot['p_qow_pdyr'] = 'Paddy Rice Sector Output at World Mkt'
        vars_to_plot['p_lc_pasture'] = 'Pasture Cover'
        vars_to_plot['p_qmt_wheat'] = 'Wheat Sector Production'

        nature_matters_scenarios = ['2014_21_BAU', '2021_30_BAU_noES', '2021_30_BAU_allES']

        # Merge in GDP, Pop, etc. for variable scaling
        input_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\GTAP_Visualizations_Data\gdp_and_other_scaling_variables.csv"
        scaling_vars_input = pd.read_csv(input_path)
        svs = scaling_vars_input[['REG', 'GDP 2014']]
        scaling_vars = svs.groupby('REG').sum()
        global_gdp_2014 = scaling_vars['GDP 2014'].sum()
        scaling_vars['global_gdp_weight'] = scaling_vars['GDP 2014'] / global_gdp_2014
        df = df.merge(scaling_vars, left_on='REG', right_on='REG')
        ### GENERATE SINGLE VARIABLE PLOTS
        # This was an ambitious but ultimately limited method for summarizing results. It used the structure of the vertical data to create plots,
        # but it wasn't flexible enough as is without lots of manual additions (like difference between vars).
        # Kept in for reference and because it's good enough for single variable summarization.
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                plot_root_dir = os.path.join(p.cur_dir, luh_scenario_label, str(scenario_year), 'single_variable_plots')
                hb.create_directories(plot_root_dir)

                # Set up which variables are to which dimensions
                plot_variable_folder_levels = {'VAR': [
                    'qgdp',
                    'p_lc_cropl',
                    'pfactrl_land',
                    'EV',
                    'p_qxw_pdyr',
                    'p_qow_pdyr',
                    'p_lc_pasture',
                    'p_qmt_wheat',
                ]}
                plot_variable_path_levels = {'TYPE' : ['UpdValue'],
                                             'UNITS': [],}
                xaxis_level = {'REG': []}
                grouped_level = {'SCENARIO': nature_matters_scenarios}
                group_by = ['']

                gtap_invest.visualization.visualization.plot_from_vertical_results(df, plot_variable_folder_levels, plot_variable_path_levels, xaxis_level, grouped_level, group_by, plot_root_dir)

        ### Create cross-scenario custom plots
        # In addition to the vertical data, read in also the processed files that horizontally sstacked the scenarios and subtracted them from baseline and bau.
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for var, variable_name in vars_to_plot.items():
                    var_output_dir = os.path.join(p.cur_dir, luh_scenario_label, str(scenario_year), 'cross-scenario comparisons', var)
                    hb.create_directories(var_output_dir)

                    gtap_invest.visualization.visualization.plot_scenario_comparison(df, nature_matters_scenarios, scenario_names, var, variable_name, var_output_dir)

        # Plot GDP and Carbon together.
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:

                output_dir = os.path.join(p.cur_dir, luh_scenario_label, str(scenario_year), 'gdp_with_carbon')
                hb.create_directories(output_dir)

                # Subset out Baseline
                bau_comparison_scenario_names = {}
                bau_comparison_scenario_names['2021_30_BAU_allES'] = 'BAU'
                bau_comparison_scenario_names['2021_30_PESGC_allES'] = 'Global PES'
                bau_comparison_scenario_names['2021_30_PESLC_allES'] = 'National PES'
                bau_comparison_scenario_names['2021_30_SR_Land_allES'] = 'Subsidy repurposed to land'
                bau_comparison_scenario_names['2021_30_SR_RnD_20p_allES'] = 'Subsidies repurposed to R&D'
                bau_comparison_scenario_names['2021_30_SR_RnD_20p_PESGC_allES'] = 'Subsidies to R&D with Global PES'

                # Read in post-processed tables. TODOO Make this update per scenario.
                carbon_path = os.path.join(p.gtap2_results_as_tables_dir, 'rcp45_ssp2_2030_carbon_minus_bau.csv')
                df_carbon = pd.read_csv(carbon_path, index_col=0)

                # Identify the relevant columns as the difference from BAU
                minus_bau_col_headers = [i for i in df_carbon.columns if 'minus_bau' in i]

                new_cols = []
                multidim_cols = []
                for i in minus_bau_col_headers:
                    # HACK at scenario labels to get correct gtap scenario. TODOO fix.
                    before, after = i.split('_minus_bau')
                    _, _, _, _, after2 = before.split('_', 4)
                    gtap_scenario_label = '2021_30_' + after2 + '_allES'
                    multidim_cols.append((gtap_scenario_label, 'Carbon Mt'))
                    new_cols.append(gtap_scenario_label)

                # LEARNING POINT: Create multiindex for later joining. Ended up not using this because the scenarios were simply enough named.
                new_cols_index = pd.MultiIndex.from_tuples(multidim_cols)
                df_carbon = df_carbon[minus_bau_col_headers]
                df_carbon.columns = new_cols

                # Process df so that it can merge with main GTAP results
                df_carbon_stacked = df_carbon.stack()
                df_carbon_stacked = df_carbon_stacked.reset_index()
                df_carbon_stacked.columns = ['REG', 'SCENARIO', 'carbon_mt']
                df_carbon_stacked = df_carbon_stacked.set_index(['REG', 'SCENARIO'])

                # Set GTAP results index to match carbon and select GDP
                df = df.set_index(['REG', 'SCENARIO'])
                df = df.loc[(df['VAR'] == 'qgdp') & ((df['TYPE'] == 'UpdValue'))]

                # Actually, set a bunch of other stuff to be index so that the results are technically vertical.
                df = df.set_index(['TYPE', 'UNITS', 'income_class', 'REGNAME', 'VAR'], append=True)
                df = df[['Value']]

                # Calculate the difference in GDP compared to BAU and add it as a column.
                dfpa = hb.calculate_on_vertical_df(df, 1, 'difference_from_row', '2021_30_BAU_allES', '_minus_bau')

                # Select and rename so matches carbon columns
                dfpar = dfpa.reset_index()
                dfpars = dfpar.loc[dfpar['SCENARIO'].str.contains('_minus_bau')]
                dfpars['SCENARIO'] = dfpars['SCENARIO'].str.replace('_minus_bau', '')
                dfpars = dfpars.set_index(['REG', 'SCENARIO', 'TYPE', 'UNITS', 'income_class', 'REGNAME', 'VAR'])

                # Merge carbon with GTAP main results
                dfm = dfpars.merge(df_carbon_stacked, left_index=True, right_index=True)
                dfm = dfm.rename(columns={0: 'qgdp'})
                dfm = dfm.rename(columns={'Value': 'qgdp'})
                dfm = dfm[['qgdp', 'carbon_mt']]

                # Calcualte dollar-valued damage of carbon Mt change using social cost of carbon. Also convert to million USD
                ssc = 185.0 # Updated to Rennert et al 2022 Nature
                dfm['carbon_m_usd'] = dfm['carbon_mt'].apply(lambda x: x * ssc / 1000000)
                dfm = dfm.drop(columns='carbon_mt')
                dfm.to_csv(os.path.join(output_dir, 'gdp_with_carbon_by_region.csv'))

                # Process plot DF
                dfp = pd.DataFrame(dfm.groupby(level=1, axis=0).sum())
                dfp = dfp.rename(columns={'qgdp': 'GDP', 'carbon_m_usd': 'Avoided carbon damages'})
                dfp = dfp.loc[dfp.index.isin(bau_comparison_scenario_names.keys())]
                dfp.index = [bau_comparison_scenario_names[i] for i in dfp.index]
                dfp['sum'] = dfp['GDP'] + dfp['Avoided carbon damages']
                dfp = dfp.sort_values(by='sum', ascending=True)
                dfp = dfp.drop(columns='sum')
                dfp.to_csv(os.path.join(output_dir, 'gdp_with_carbon_by_region.csv'))

                # do plot.
                fig, ax = plt.subplots()
                dfp.plot.bar(ax=ax, stacked=True)
                desc = 'Policy scenario improvement over BAU'
                ax.set_ylabel('Million 2020 USD')
                ax.set_xlabel('Scenario')
                plt.xticks(rotation=20, ha='right')
                ax.set_title(desc)
                output_path = os.path.join(output_dir, desc + '.png')
                plt.setp(ax.patches, linewidth=0)
                plt.savefig(output_path, dpi=200, bbox_inches="tight")


        # Plot GDP and Carbon together.
        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:

                output_dir = os.path.join(p.cur_dir, luh_scenario_label, str(scenario_year), 'gdp_with_land')
                hb.create_directories(output_dir)

                # Subset out Baseline
                bau_comparison_scenario_names = {}
                bau_comparison_scenario_names['2021_30_BAU_allES'] = 'BAU'
                bau_comparison_scenario_names['2021_30_PESGC_allES'] = 'Global PES'
                bau_comparison_scenario_names['2021_30_PESLC_allES'] = 'National PES'
                bau_comparison_scenario_names['2021_30_SR_Land_allES'] = 'Subsidy repurposed to land'
                bau_comparison_scenario_names['2021_30_SR_RnD_20p_allES'] = 'Subsidies repurposed to R&D'
                bau_comparison_scenario_names['2021_30_SR_RnD_20p_PESGC_allES'] = 'Subsidies to R&D with Global PES'

                # Set GTAP results index to match carbon and select GDP
                # Read in the vertical, full results produced in gtap2_aez task.
                df = pd.read_csv(p.gtap2_results_path)
                df_income_categories = pd.read_excel(os.path.join(p.model_base_data_dir, 'gtap37_continents_and_income_class.xlsx'), index_col='gtap37_region_label')
                dfp = df
                df = df.merge(df_income_categories, left_on='REG', right_index=True)

                lengths_dict, keys_dict = hb.get_unique_keys_from_vertical_dataframe(df)

                df = df.set_index(['REG', 'SCENARIO'])

                vars_to_extract = ['qgdp', 'p_lc_cropl', 'p_lc_pasture', 'p_lc_mngfo']
                df = df.loc[(df['VAR'].isin(vars_to_extract)) & (df['TYPE'] == 'UpdValue')]

                # Actually, set a bunch of other stuff to be index so that the results are technically vertical.
                df = df.set_index(['TYPE', 'UNITS', 'income_class', 'REGNAME', 'VAR'], append=True)
                df = df[['Value']]

                # Calculate the difference in GDP compared to BAU and add it as a column.
                dfpa = hb.calculate_on_vertical_df(df, 1, 'difference_from_row', '2021_30_BAU_allES', '_minus_bau')

                # Select and rename so matches carbon columns
                dfpar = dfpa.reset_index()
                dfpars = dfpar.loc[dfpar['SCENARIO'].str.contains('_minus_bau')]
                dfpars['SCENARIO'] = dfpars['SCENARIO'].str.replace('_minus_bau', '')
                dfpars = dfpars.set_index(['REG', 'SCENARIO', 'TYPE', 'UNITS', 'income_class', 'REGNAME', 'VAR'])

                new_index = [i for i in dfpars.index if i not in ['UNITS']]
                # dfpars.index = new_index
                dfparsu = dfpars.unstack(level=-1)

                # This generates data that are vertically misaligned because GDP is in dollars and LUC is in ha.
                # which leaves a lot of nans when unstacked.
                # Fix this by unstacking again on the units col and dropping na.
                dfparsuu = dfparsu.unstack(level=3)
                dfparsuud = dfparsuu.dropna(axis=1)

                # Unstacking always generates a multiindex column when coming from vertical data.
                # Grab the correct one and reassign.
                dfparsuud.columns = dfparsuud.columns.get_level_values(1)


                # Process plot DF
                new_index = [i for i in dfparsuud.index if i !='REG']
                dfparsuud = dfparsuud.reset_index()
                dfp = pd.DataFrame(dfparsuud.groupby(by='SCENARIO', axis=0).sum())
                # dfp = pd.DataFrame(dfparsuud.groupby(level=0, axis=0).sum())
                dfp = dfp.rename(columns={'qgdp': 'GDP', 'p_lc_cropl': 'Cropland', 'p_lc_pasture': 'Pasture', 'p_lc_mngfo': 'Managed forestry'})
                dfp = dfp.loc[dfp.index.isin(bau_comparison_scenario_names.keys())]
                dfp['Natural'] = -1. * (dfp['Cropland'] + dfp['Pasture'] + dfp['Managed forestry'])
                # dfp = dfp.drop('2021_30_BAU_allES')
                dfp.to_csv(os.path.join(output_dir, 'gdp_with_luc_by_region.csv'))

                # do plot.
                fig, ax = plt.subplots()
                dfp.plot.bar(ax=ax, stacked=True)
                desc = 'Policy scenario improvement over BAU'
                ax.set_ylabel('Million 2020 USD')
                ax.set_xlabel('Scenario')
                plt.xticks(rotation=20, ha='right')
                ax.set_title(desc)
                output_path = os.path.join(output_dir, desc + '.png')
                plt.setp(ax.patches, linewidth=0)
                plt.savefig(output_path, dpi=200, bbox_inches="tight")
                plt.close()

                dfp['Natural'] = dfp['Natural'].apply(lambda x: x / 1000000)
                dfp['GDP'] = dfp['GDP'].apply(lambda x: x / 1000)

                fig, ax = plt.subplots()
                x = dfp['GDP']
                y = dfp['Natural']
                labels = [bau_comparison_scenario_names[i] for i in dfp.index]
                dfp.plot.scatter(ax=ax, x='GDP', y='Natural', s=100, alpha=.5, edgecolors='none') # x, y, c=color, s=scale, label=color, alpha=0.3, edgecolors='none'
                desc = 'Policy scenario improvement over BAU for GDP and Natural Land protection'
                ax.set_xlabel('Billion 2020 USD')
                ax.set_ylabel('Million hectares preserved')

                for x_pos, y_pos, label in zip(x, y, labels):
                    ax.annotate(label,  # The label for this point
                                xy=(x_pos, y_pos),  # Position of the corresponding point
                                xytext=(7, 0),  # Offset text by 7 points to the right
                                textcoords='offset points',  # tell it to use offset points
                                ha='left',  # Horizontally aligned to the left
                                va='center')  # Vertical alignment is centered
                # plt.xticks(rotation=20, ha='right')
                ax.set_title(desc)
                output_path = os.path.join(output_dir, desc + '_scatter.png')
                plt.setp(ax.patches, linewidth=0)
                plt.savefig(output_path, dpi=200, bbox_inches="tight")

def output_visualizations_secondary(p):
    import seaborn as sns

    sns.set_theme()

    if p.run_this:

        # Read in the vertical, full results produced in gtap2_aez task.
        df = pd.read_csv(p.gtap2_results_path)
        run_subtotals = 1
        if run_subtotals:
            for luh_scenario_label in p.luh_scenario_labels:
                for scenario_year in p.scenario_years:
                    output_dir = os.path.join(p.cur_dir, luh_scenario_label, str(scenario_year), 'subtotals')
                    hb.create_directories(output_dir)

                    # Subset out Baseline
                    bau_comparison_scenario_names = {}
                    bau_comparison_scenario_names['2021_30_BAU_allES'] = 'BAU'
                    bau_comparison_scenario_names['2021_30_PESGC_allES'] = 'Global PES'
                    bau_comparison_scenario_names['2021_30_PESLC_allES'] = 'National PES'
                    bau_comparison_scenario_names['2021_30_SR_Land_allES'] = 'Subsidy repurposed to land'
                    bau_comparison_scenario_names['2021_30_SR_RnD_20p_allES'] = 'Subsidies repurposed to R&D'
                    bau_comparison_scenario_names['2021_30_SR_RnD_20p_PESGC_allES'] = 'Subsidies to R&D with Global PES'

                    subtotals_to_consider = ['SubTot_fish', 'SubTot_polli', 'SubTot_forest', 'SubTot_others']
                    # Set GTAP results index to match carbon and select GDP
                    # Read in the vertical, full results produced in gtap2_aez task.

                    df_income_categories = pd.read_excel(os.path.join(p.model_base_data_dir, 'gtap37_continents_and_income_class.xlsx'), index_col='gtap37_region_label')
                    dfm = df.merge(df_income_categories, left_on='REG', right_index=True)

                    dfms = dfm.set_index(['REGNAME', 'SCENARIO'])

                    vars_to_extract = ['qgdp']
                    # vars_to_extract = ['qgdp', 'p_lc_cropl', 'p_lc_pasture', 'p_lc_mngfo']
                    dfmss = dfms.loc[(dfms['VAR'].isin(vars_to_extract)) & (dfms['TYPE'].isin(subtotals_to_consider))]

                    # # Actually, set a bunch of other stuff to be index so that the results are technically vertical.
                    # dfmssi = dfmss.set_index(['TYPE', 'UNITS', 'income_class', 'REGNAME', 'VAR'])
                    dfmssi = dfmss.set_index(['TYPE', 'VAR', 'income_class'], append=True)
                    # dfmssi = dfmss.set_index(['TYPE', 'UNITS', 'income_class', 'REGNAME', 'VAR'], append=True)
                    dfmssiv = dfmssi[['Value']]

                    # Calculate the difference in GDP compared to BAU and add it as a column.
                    dfpa = hb.calculate_on_vertical_df(dfmssiv, 1, 'difference_from_row', '2021_30_BAU_allES', '_minus_bau')

                    # Select and rename so matches carbon columns
                    dfpar = dfpa.reset_index()
                    # dfpar['SCENARIO_with_var'] = dfpar['SCENARIO'].copy()

                    # dfpars = dfpar.loc[dfpar['SCENARIO'].str.contains('_minus_bau')]
                    # dfpars['SCENARIO'] = dfpars['SCENARIO'].str.replace('_minus_bau', '')
                    dfparss = dfpar.loc[dfpar['SCENARIO'].isin(bau_comparison_scenario_names.keys())] # Selecting on the scenario names directly meanse we don't get the difference vars.

                    # First make the plot for subtotals difference from baseline
                    for var in dfparss['VAR'].unique():
                        dfs = dfparss.loc[(dfparss['VAR'] == var)]
                        for scenario in dfs['SCENARIO'].unique():
                            dfss = dfs.loc[(dfparss['SCENARIO'] == scenario)]
                            dfss = dfss.set_index(['REGNAME', 'TYPE', 'income_class'], append=False) # LEARNING POINT, append=True made a checkerboard outcome because it kept the integer index
                            dfss = dfss[['Value']]
                            dfssu = dfss.unstack(level=1)
                            dfssu.columns = dfssu.columns.get_level_values(1)


                            # TODOO ugly way to order income categories how I wanted. make elegent.
                            ordered_income_categories = {'Low income': '0Low income', 'Lower middle income': '1Lower middle income', 'Upper middle income': '2Upper middle income', 'High income': '3High income'}
                            reversed_ordered_income_categories = dict(zip(ordered_income_categories.values(), ordered_income_categories.keys()))
                            dfssu = dfssu.rename(index=ordered_income_categories, level=1)
                            dfssu.sort_index(axis=0, level=0, ascending=True, inplace=True)
                            dfssu.sort_index(axis=0, level=1, ascending=True, inplace=True)
                            dfssu = dfssu.rename(index=reversed_ordered_income_categories, level=1)
                            dfssu = dfssu.reset_index()

                            from collections import Counter
                            income_group_counts = list(Counter(dfssu['income_class']).values())

                            dfssu = dfssu.set_index(['REGNAME'])
                            dfssu = dfssu[[i for i in dfssu.columns if 'SubTot_others' not in i]]

                            # dfs = dfparss.loc[(dfpars['VAR'] == var) & (dfpars['SCENARIO'] == '2021_30_PESGC_allES')  & (dfpars['REG'] == 'ARG')]

                            # do plot.
                            colors = ['tab:blue',  'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

                            subtotals_to_consider = {'SubTot_fish': 'Fisheries', 'SubTot_polli': 'Pollination', 'SubTot_forest': 'Forestry'}

                            fig, ax = plt.subplots()
                            dfssu.plot.bar(ax=ax, stacked=True)
                            ax.legend(list(subtotals_to_consider.values()))

                            cumulative_width = 0
                            y_lim = ax.get_ylim()
                            min_y = y_lim[0] * 1.05
                            ax.set_ylim(y_lim[0] * 1.1, y_lim[1])
                            for c, width in enumerate(income_group_counts):
                                plt.axvspan(cumulative_width - .5, cumulative_width + width - .5, facecolor=colors[c], alpha=0.05, zorder=-100)
                                plt.annotate(list(ordered_income_categories.keys())[c], xy=(cumulative_width, min_y), size=9)
                                cumulative_width += width

                            formatted_scenario = bau_comparison_scenario_names[scenario]
                            desc = var + '_' + scenario + '_subtotals_diff_from_baseline'
                            ax.set_ylabel('Percent difference')
                            ax.set_xlabel(None)
                            plt.xticks(rotation=90, ha='right', size=10)
                            for tick in ax.xaxis.get_majorticklabels():
                                tick.set_horizontalalignment("center")

                            ax.set_title(formatted_scenario + ' effect of including ecosystem services')

                            output_path = os.path.join(output_dir, desc + '.png')
                            L.info('Plotting ' + str(output_path))
                            plt.setp(ax.patches, linewidth=0)
                            plt.savefig(output_path, dpi=200, bbox_inches="tight")
                            plt.close()

                    # Second make the plot for subtotals minus bau
                    dfpars = dfpar.loc[dfpar['SCENARIO'].str.contains('_minus_bau')]
                    dfpars['SCENARIO'] = dfpars['SCENARIO'].str.replace('_minus_bau', '')
                    dfparss = dfpars.loc[dfpars['SCENARIO'].isin(bau_comparison_scenario_names.keys())]

                    for var in dfparss['VAR'].unique():
                        dfs = dfparss.loc[(dfpars['VAR'] == var)]
                        for scenario in dfs['SCENARIO'].unique():
                            dfss = dfs.loc[(dfpars['SCENARIO'] == scenario)]
                            # dfss = dfss.set_index(['REGNAME', 'TYPE'], append=False) # LEARNING POINT, append=True made a checkerboard outcome because it kept the integer index
                            dfss = dfss.set_index(['REGNAME', 'TYPE', 'income_class'], append=False) # LEARNING POINT, append=True made a checkerboard outcome because it kept the integer index

                            dfss = dfss[['Value']]
                            dfssu = dfss.unstack(level=1)
                            dfssu.columns = dfssu.columns.get_level_values(1)

                            # TODOO ugly way to order income categories how I wanted. make elegent.
                            ordered_income_categories = {'Low income': '0Low income', 'Lower middle income': '1Lower middle income', 'Upper middle income': '2Upper middle income', 'High income': '3High income'}
                            reversed_ordered_income_categories = dict(zip(ordered_income_categories.values(), ordered_income_categories.keys()))
                            dfssu = dfssu.rename(index=ordered_income_categories, level=1)
                            dfssu.sort_index(axis=0, level=0, ascending=True, inplace=True)
                            dfssu.sort_index(axis=0, level=1, ascending=True, inplace=True)
                            dfssu = dfssu.rename(index=reversed_ordered_income_categories, level=1)
                            dfssu = dfssu.reset_index()

                            from collections import Counter
                            income_group_counts = list(Counter(dfssu['income_class']).values())

                            dfssu = dfssu.set_index(['REGNAME'])

                            colors = ['tab:blue',  'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
                            subtotals_to_consider = {'SubTot_fish': 'Fisheries', 'SubTot_polli': 'Pollination', 'SubTot_forest': 'Forestry', 'SubTot_others': 'Other equilibrium effects'}


                            # do plot.
                            fig, ax = plt.subplots()
                            dfssu.plot.bar(ax=ax, stacked=True)
                            ax.legend(list(subtotals_to_consider.values()))

                            cumulative_width = 0
                            y_lim = ax.get_ylim()
                            min_y = y_lim[0] * 1.05
                            ax.set_ylim(y_lim[0] * 1.1, y_lim[1])
                            for c, width in enumerate(income_group_counts):
                                plt.axvspan(cumulative_width - .5, cumulative_width + width - .5, facecolor=colors[c], alpha=0.05, zorder=-100)
                                plt.annotate(list(ordered_income_categories.keys())[c], xy=(cumulative_width, min_y), size=9)
                                cumulative_width += width


                            desc = var + '_' + scenario + '_subtotals_minus_bau'
                            ax.set_ylabel('Percent difference')
                            # ax.set_xlabel('Region')
                            ax.set_xlabel(None)
                            plt.xticks(rotation=90, ha='right', size=10)
                            for tick in ax.xaxis.get_majorticklabels():
                                tick.set_horizontalalignment("center")

                            subtotals_to_consider = {'SubTot_fish': 'Fisheries', 'SubTot_polli': 'Pollination', 'SubTot_forest': 'Forestry', 'SubTot_others': 'Other equilibrium effects'}

                            formatted_scenario = bau_comparison_scenario_names[scenario]

                            ax.set_title(formatted_scenario + ' GDP difference from BAU')
                            output_path = os.path.join(output_dir, desc + '.png')
                            plt.setp(ax.patches, linewidth=0)
                            plt.savefig(output_path, dpi=200, bbox_inches="tight")
                            plt.close()


def output_visualizations_tertiary(p):
    import seaborn as sns
    sns.set_theme()

    # Define project-level pretty-names for scenarios
    p.bau_comparison_scenario_names = {}
    p.bau_comparison_scenario_names['2021_30_BAU_allES'] = 'BAU'
    p.bau_comparison_scenario_names['2021_30_PESGC_allES'] = 'Global PES'
    p.bau_comparison_scenario_names['2021_30_PESLC_allES'] = 'National PES'
    p.bau_comparison_scenario_names['2021_30_SR_Land_allES'] = 'Subsidy repurposed to land'
    p.bau_comparison_scenario_names['2021_30_SR_RnD_20p_allES'] = 'Subsidies repurposed to R&D'
    p.bau_comparison_scenario_names['2021_30_SR_RnD_20p_PESGC_allES'] = 'Subsidies to R&D with Global PES'

    p.all_subtotal_labels = ['SubTot_fish', 'SubTot_polli', 'SubTot_forest', 'SubTot_others']
    p.es_subtotal_labels = ['SubTot_fish', 'SubTot_polli', 'SubTot_forest']
    p.totals_labels = ['UpdValue']
    p.ordered_income_categories = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
    # p.ordered_income_categories = {'Low income': '0Low income', 'Lower middle income': '1Lower middle income', 'Upper middle income': '2Upper middle income', 'High income': '3High income'}
    if p.run_this:

        # Read in the vertical, full results produced in gtap2_aez task.
        df = pd.read_csv(p.gtap2_results_path)

        # Also load income categories and merge
        df_income_categories = pd.read_excel(os.path.join(p.model_base_data_dir, 'gtap37_continents_and_income_class.xlsx'), index_col='gtap37_region_label')
        dfi = df.merge(df_income_categories, left_on='REG', right_index=True)


        # Start here, need to have discussion with group about how to interpret subtotals and why they differ when
        # marine fisheries doesn't change. Can the graph of subtotals minus BAU be used?
        # Then, finish this plotting approach for the other key figures, namely making the per-secenario plot have the sorted income groups.

        ### Plot difference between BAU and Baseline to establish how much nature matters with substotals
        run_subtotals = 1
        if run_subtotals:
            for luh_scenario_label in p.luh_scenario_labels:
                for scenario_year in p.scenario_years:
                    output_dir = os.path.join(p.cur_dir, luh_scenario_label, str(scenario_year))
                    hb.create_directories(output_dir)

                    # visualization.plot_var_by_region_with_subtotals(df)


                    vars_to_extract = ['qgdp']
                    dfi = dfi.loc[(dfi['VAR'].isin(vars_to_extract)) & (dfi['TYPE'].isin(p.totals_labels))]
                    dfii = dfi.set_index(['REGNAME', 'SCENARIO', 'income_class', 'TYPE', 'VARNAME', 'VAR'])
                    dfiis = dfii[['Value']]
                    dfii_the_rest = dfii[[i for i in dfii.columns if i != 'Value']]

                    # Calculate the difference in GDP compared to BAU and add it as a column.
                    dfiisc = hb.calculate_on_vertical_df(dfiis, 1, 'percentage_change_from_row', '2021_30_BAU_allES', '_percentage_change')

                    # TODOO ugly way to order income categories how I wanted. make elegent.
                    ordered_income_categories_rename_dict = {v: str(c) + ' ' + v for c, v in enumerate(p.ordered_income_categories)}
                    # ordered_income_categories_rename_dict = dict(zip(range(len(p.ordered_income_categories)), p.ordered_income_categories))
                    reversed_ordered_income_categories_rename_dict = dict(zip(ordered_income_categories_rename_dict.values(), ordered_income_categories_rename_dict.keys()))
                    dfiiscs = dfiisc.rename(index=ordered_income_categories_rename_dict, level=2)
                    dfiiscs.sort_index(axis=0, level=0, ascending=True, inplace=True)
                    dfiiscs.sort_index(axis=0, level=2, ascending=True, inplace=True)
                    dfiiscs = dfiiscs.rename(index=reversed_ordered_income_categories_rename_dict, level=2)
                    dfiiscs = dfiiscs.reset_index()




                    dfiiscs = dfiiscs.set_index(['REGNAME'])
                    # dfiiscs = dfiiscs[[i for i in dfssu.columns if 'SubTot_others' not in i]]

                    dfiiscss = dfiiscs.loc[dfiiscs['SCENARIO'].isin([i + '_percentage_change' for i in p.bau_comparison_scenario_names.keys()])]  # Selecting on the scenario names directly meanse we don't get the difference vars.

                    # dfiiscs = dfiiscs.loc[dfiiscs['SCENARIO'] == ]
                    # do plot.
                    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

                    subtotals_to_consider = {'SubTot_fish': 'Fisheries', 'SubTot_polli': 'Pollination', 'SubTot_forest': 'Forestry'}

                    # Actually only need combined scenario
                    dfiiscss = dfiiscs.loc[dfiiscs['SCENARIO'].isin(['2021_30_SR_RnD_20p_PESGC_allES_percentage_change'])]
                    from collections import Counter
                    income_group_counts = list(Counter(dfiiscss['income_class']).values())
                    dfiiscssp = dfiiscss[['Value']]



                    fig, ax = plt.subplots()
                    dfiiscssp.plot.bar(ax=ax, stacked=True)
                    # ax.legend(None)
                    # ax.legend(list(subtotals_to_consider.values()))
                    ax.legend().set_visible(False)

                    cumulative_width = 0
                    y_lim = ax.get_ylim()
                    min_y = y_lim[0] * 1.05
                    ax.set_ylim(y_lim[0] * 1.1, y_lim[1])
                    for c, width in enumerate(income_group_counts):
                        plt.axvspan(cumulative_width - .5, cumulative_width + width - .5, facecolor=colors[c], alpha=0.05, zorder=-100)
                        plt.annotate(list(ordered_income_categories_rename_dict.keys())[c], xy=(cumulative_width, min_y), size=9)
                        cumulative_width += width

                    formatted_scenario = p.bau_comparison_scenario_names['2021_30_SR_RnD_20p_PESGC_allES']
                    desc = 'combined_subtotals_diff_from_baseline'
                    ax.set_ylabel('Percent difference')
                    ax.set_xlabel(None)
                    plt.xticks(rotation=90, ha='right', size=10)
                    for tick in ax.xaxis.get_majorticklabels():
                        tick.set_horizontalalignment("center")

                    ax.set_title('Combined policy improvement over BAU')

                    output_path = os.path.join(output_dir, desc + '.png')
                    L.info('Plotting ' + str(output_path))
                    plt.setp(ax.patches, linewidth=0)
                    plt.savefig(output_path, dpi=200, bbox_inches="tight")
                    plt.close()



old = "starts_here"
def regional_luc():
    p.gtap37_aez18_stats_vector_path = os.path.join(p.cur_dir, 'gtap37_aez18_stats.gpkg')


    match_path = r"C:\OneDrive\Projects\base_data\pyramids\country_ids_15m.tif"
    p.scenario_seals_5_paths = {}
    p.scenario_seals_5_paths['urban_rcp26_ssp1_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP26_SSP1\2030\ha_total\1_urban.tif"
    p.scenario_seals_5_paths['urban_rcp45_ssp2_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP45_SSP2\2030\ha_total\1_urban.tif"
    p.scenario_seals_5_paths['urban_rcp85_ssp5_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP85_SSP5\2030\ha_total\1_urban.tif"
    p.scenario_seals_5_paths['cropland_rcp26_ssp1_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP26_SSP1\2030\ha_total\2_cropland.tif"
    p.scenario_seals_5_paths['cropland_rcp45_ssp2_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP45_SSP2\2030\ha_total\2_cropland.tif"
    p.scenario_seals_5_paths['cropland_rcp85_ssp5_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP85_SSP5\2030\ha_total\2_cropland.tif"
    p.scenario_seals_5_paths['grassland_rcp26_ssp1_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP26_SSP1\2030\ha_total\3_grassland.tif"
    p.scenario_seals_5_paths['grassland_rcp45_ssp2_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP45_SSP2\2030\ha_total\3_grassland.tif"
    p.scenario_seals_5_paths['grassland_rcp85_ssp5_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP85_SSP5\2030\ha_total\3_grassland.tif"
    p.scenario_seals_5_paths['forest_rcp26_ssp1_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP26_SSP1\2030\ha_total\4_forest.tif"
    p.scenario_seals_5_paths['forest_rcp45_ssp2_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP45_SSP2\2030\ha_total\4_forest.tif"
    p.scenario_seals_5_paths['forest_rcp85_ssp5_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP85_SSP5\2030\ha_total\4_forest.tif"
    p.scenario_seals_5_paths['nonforestnatural_rcp26_ssp1_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP26_SSP1\2030\ha_total\5_nonforestnatural.tif"
    p.scenario_seals_5_paths['nonforestnatural_rcp45_ssp2_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP45_SSP2\2030\ha_total\5_nonforestnatural.tif"
    p.scenario_seals_5_paths['nonforestnatural_rcp85_ssp5_2030'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP85_SSP5\2030\ha_total\5_nonforestnatural.tif"
    p.scenario_seals_5_paths['urban_baseline_2014'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP26_SSP1\2015\ha_total\1_urban.tif"
    p.scenario_seals_5_paths['cropland_baseline_2014'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP26_SSP1\2015\ha_total\2_cropland.tif"
    p.scenario_seals_5_paths['grassland_baseline_2014'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP26_SSP1\2015\ha_total\3_grassland.tif"
    p.scenario_seals_5_paths['forest_baseline_2014'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP26_SSP1\2015\ha_total\4_forest.tif"
    p.scenario_seals_5_paths['nonforestnatural_baseline_2014'] = r"C:\OneDrive\Projects\cge\worldbank\projects\second_report\intermediate\luh2_as_seals7_proportion\RCP26_SSP1\2015\ha_total\5_nonforestnatural.tif"

    p.gtap19_aez18_ids_15m_path = os.path.join(p.cur_dir, 'gtap19_aez18_ids_15m.tif')
    p.gtap37_aez18_ids_15m_path = os.path.join(p.cur_dir, 'gtap37_aez18_ids_15m.tif')

    # df = pd.DataFrame()
    # p.gtap19_aez18_stats_path = os.path.join(p.cur_dir, 'gtap19_aez18_stats.xlsx')
    # if not hb.path_exists(p.gtap19_aez18_stats_path):
    #
    #     for name, path in p.scenario_seals_5_paths.items():
    #         # NOTE gtap_aez_dissolved_path hasn't been updated to leverage pyramid gpkg code
    #         to_merge = hb.zonal_statistics_flex(path, p.gtap_aez_dissolved_path, p.gtap19_aez18_ids_15m_path, id_column_label='GTAP19GTAP_as_id', verbose=True)
    #         to_merge = to_merge.rename(columns={'sum': name})
    #         df = df.merge(to_merge, how='outer', left_index=True, right_index=True)
    #
    #     df.to_excel(p.gtap19_aez18_stats_path)


    df = pd.DataFrame()
    p.gtap37_aez18_stats_path = os.path.join(p.cur_dir, 'gtap37_aez18_stats.xlsx')
    p.gtap37_aez18_pch_stats_path = os.path.join(p.cur_dir, 'gtap37_aez18_pch_stats.xlsx')
    if not hb.path_exists(p.gtap37_aez18_stats_path):

        for name, path in p.scenario_seals_5_paths.items():

            to_merge = hb.zonal_statistics_flex(path,
                                                p.gtap37_aez18_vector_path,
                                                p.gtap37_aez18_ids_15m_path,
                                                id_column_label='pyramid_ordered_id',
                                                zones_raster_data_type=5,
                                                verbose=True)
            to_merge = to_merge.rename(columns={'sum': name})
            df = df.merge(to_merge, how='outer', left_index=True, right_index=True)

        gdf = gpd.read_file(p.gtap37_aez18_vector_path)

        gdf = gdf.merge(df, left_on='pyramid_ordered_id', right_index=True)
        # df = df.merge(gdf, left_index=True, right_on='pyramid_ordered_id')

        gdf['urban_rcp26_ssp1_2030_pch'] = ((gdf['urban_rcp26_ssp1_2030'] - gdf['urban_baseline_2014']) / gdf['urban_baseline_2014']) * 100.0
        gdf['cropland_rcp26_ssp1_2030_pch'] = ((gdf['cropland_rcp26_ssp1_2030'] - gdf['cropland_baseline_2014']) / gdf['cropland_baseline_2014']) * 100.0
        gdf['grassland_rcp26_ssp1_2030_pch'] = ((gdf['grassland_rcp26_ssp1_2030'] - gdf['grassland_baseline_2014']) / gdf['grassland_baseline_2014']) * 100.0
        gdf['forest_rcp26_ssp1_2030_pch'] = ((gdf['forest_rcp26_ssp1_2030'] - gdf['forest_baseline_2014']) / gdf['forest_baseline_2014']) * 100.0
        gdf['nonforestnatural_rcp26_ssp1_2030_pch'] = ((gdf['nonforestnatural_rcp26_ssp1_2030'] - gdf['nonforestnatural_baseline_2014']) / gdf['nonforestnatural_baseline_2014']) * 100.0

        gdf['urban_rcp45_ssp2_2030_pch'] = ((gdf['urban_rcp26_ssp1_2030'] - gdf['urban_baseline_2014']) / gdf['urban_baseline_2014']) * 100.0
        gdf['cropland_rcp45_ssp2_2030_pch'] = ((gdf['cropland_rcp26_ssp1_2030'] - gdf['cropland_baseline_2014']) / gdf['cropland_baseline_2014']) * 100.0
        gdf['grassland_rcp45_ssp2_2030_pch'] = ((gdf['grassland_rcp26_ssp1_2030'] - gdf['grassland_baseline_2014']) / gdf['grassland_baseline_2014']) * 100.0
        gdf['forest_rcp45_ssp2_2030_pch'] = ((gdf['forest_rcp26_ssp1_2030'] - gdf['forest_baseline_2014']) / gdf['forest_baseline_2014']) * 100.0
        gdf['nonforestnatural_rcp45_ssp2_2030_pch'] = ((gdf['nonforestnatural_rcp26_ssp1_2030'] - gdf['nonforestnatural_baseline_2014']) / gdf['nonforestnatural_baseline_2014']) * 100.0

        gdf['urban_rcp85_ssp5_2030_pch'] = ((gdf['urban_rcp26_ssp1_2030'] - gdf['urban_baseline_2014']) / gdf['urban_baseline_2014']) * 100.0
        gdf['cropland_rcp85_ssp5_2030_pch'] = ((gdf['cropland_rcp26_ssp1_2030'] - gdf['cropland_baseline_2014']) / gdf['cropland_baseline_2014']) * 100.0
        gdf['grassland_rcp85_ssp5_2030_pch'] = ((gdf['grassland_rcp26_ssp1_2030'] - gdf['grassland_baseline_2014']) / gdf['grassland_baseline_2014']) * 100.0
        gdf['forest_rcp85_ssp5_2030_pch'] = ((gdf['forest_rcp26_ssp1_2030'] - gdf['forest_baseline_2014']) / gdf['forest_baseline_2014']) * 100.0
        gdf['nonforestnatural_rcp85_ssp5_2030_pch'] = ((gdf['nonforestnatural_rcp26_ssp1_2030'] - gdf['nonforestnatural_baseline_2014']) / gdf['nonforestnatural_baseline_2014']) * 100.0

        gdf.to_file(p.gtap37_aez18_stats_vector_path, driver='GPKG')
        df = gdf.drop('geometry', axis=1)
        df.to_excel(p.gtap37_aez18_stats_path)



        df.to_excel(p.gtap37_aez18_pch_stats_path)


def water_depletion_factor(p):
    """UNUSED but maybe worthwile? Uses Kate Brauman's watergap 3 results to see where there is a change in depleted zones. We didn't use it because we used Jing's water basin data."""


    p.water_depletion_factor_5m_path = os.path.join(p.cur_dir, 'water_depletion_factor_5m.tif')
    p.water_depletion_factor_10s_path = os.path.join(p.cur_dir, 'water_depletion_factor_10s.tif')

    if p.run_this:
        ndv = -9999.
        water_depletion_categories = hb.as_array(p.water_depletion_categories_path)
        water_depletion_factor = np.where(water_depletion_categories == 8, 1.0, 0.0)
        water_depletion_factor = np.where(water_depletion_categories == 7, .875, water_depletion_factor)
        water_depletion_factor = np.where(water_depletion_categories == 6, .75, water_depletion_factor)
        water_depletion_factor = np.where(water_depletion_categories == 5, .675, water_depletion_factor)
        water_depletion_factor = np.where(water_depletion_categories == 2, .25, water_depletion_factor)
        # water_depletion_factor = np.where(water_depletion_categories == 0, ndv, water_depletion_factor)

        hb.save_array_as_geotiff(water_depletion_factor, p.water_depletion_factor_5m_path, p.water_depletion_categories_path, ndv=ndv, compress=True)

        # hb.align_dataset_to_match(p.water_depletion_factor_5m_path, p.water_yield_cur_path, p.water_depletion_factor_10s_path, resample_method='near', compress=True)
        hb.resample_to_match(p.water_depletion_factor_5m_path, p.water_yield_cur_path, p.water_depletion_factor_10s_path, resample_method='near', compress=False)

    """


    Key metric: water depletion = fraction of renewable water consumptively used for human activities

    Water Depletion and WaterGap3 Basins
    Water depletion is a measure of the fraction of available renewable water consumptively used by human activities within a watershed.
    Our characterization of water depletion uses calculations from WaterGAP3 to assess long-term average annual consumed fraction of
    renewably available water, then integrates seasonal depletion and dry-year depletion, also based on WaterGAP3 calculations, with
    average annual depletion into a unified scale. There are 8 water depletion categories: <5% depleted, 5-25% depleted, 25-50% depleted,
    50-75% depleted, dry-year depleted, seasonally depleted, 75-100% depleted, and >100% depleted. For data reliability reasons, we
    include only the 15,091 watersheds larger than 1,000 km2, which constitute 90% of total land area. A large number of small
    coastal watersheds are excluded. Detailed information can be found in the open-access paper “Water Depletion: An improved
    metric for incorporating seasonal and dry-year water scarcity into water risk assessments” online at Elementa: Science of the Anthropocene < http://elementascience.org/>

    Annual water depletion categories

    1= <5%
    2= 5-25%
    5= Dry-Year
    6= Seasonal
    7= 75-100%
    8= >100%

    In order to evaluate water depletion by category, it was necessary to identify a threshold to define a
    “depleted” condition on a seasonal basis and in dry years. Using categories allows us to integrate inter- and
    intra-annual variation with average annual variation into a single scale, the simplicity of which is important
    for decision-
    makers. We employ a threshold of 75% based on a natural division we identified in our data
    and discuss the implications of setting the threshold at this level in the uncertainties and limitations section
    of the discussion.
    We define “seasonal depletion” for watersheds as occurring when annual depletion is below our
    75% threshold
    but at least one month has a consumption-to-availability ratio greater than 75%. Because of
    the strong relationship
    between monthly and annual depletion, we integrate this category into our unified
    depletion scale just below 75% annual depletion. We classify watersheds as having dry-year depletion by
    evaluating seasonal depletion over each year of the historic range of water availability and evaporative demand
    (1971–2000). Watersheds are identified as dry-year depleted if they experience one month more than 75%
    depleted in at least 10% of years during the historic period but on average are not annually or seasonally
    depleted. We integrate dry-year depletion into our unified depletion scale just below seasonal depletion.

    Citation:
    Brauman, KA, BD Richter, S Postel, M Malby, M Flörke. (2016) “Water Depletion: An improved metric for incorporating seasonal and dry-year water scarcity into water risk assessments.” Elementa: Science of the Anthropocene. Doi: http://doi.org/10.12952/journal.elementa.000083"""

    p.water_depletion_factor_ssp1_path = os.path.join(p.cur_dir, 'water_depletion_factor_ssp1.tif')
    p.water_depletion_factor_ssp3_path = os.path.join(p.cur_dir, 'water_depletion_factor_ssp3.tif')
    p.water_depletion_factor_ssp5_path = os.path.join(p.cur_dir, 'water_depletion_factor_ssp5.tif')

    def local_op(depletion_factor, scenario_wy, current_wy):
        a = (scenario_wy - current_wy) * depletion_factor

        return a

    if p.run_this:
        temp_path = hb.temp('.tif', 'global', False)
        hb.fill_to_match_extent(p.water_yield_cur_path, p.water_depletion_factor_10s_path, remove_temporary_files=False)
        hb.fill_to_match_extent(p.water_yield_ssp1_path, p.water_depletion_factor_10s_path, remove_temporary_files=False)
        hb.fill_to_match_extent(p.water_yield_gc_path, p.water_depletion_factor_10s_path, remove_temporary_files=False)
        hb.fill_to_match_extent(p.water_yield_ssp5_path, p.water_depletion_factor_10s_path, remove_temporary_files=False)

        # hb.set_geotransform_to_tuple(temp_path, hb.geotransform_global_10s)

        # hb.raster_calculator_af_flex([p.water_depletion_factor_10s_path, p.water_yield_ssp1_path, p.water_yield_cur_path], local_op, p.water_depletion_factor_ssp1_path, compress=True, add_overviews=True)
        hb.raster_calculator_af_flex([p.water_depletion_factor_10s_path, p.water_yield_gc_path, p.water_yield_cur_path], local_op, p.water_depletion_factor_ssp3_path, compress=True, add_overviews=True)
        hb.raster_calculator_af_flex([p.water_depletion_factor_10s_path, p.water_yield_ssp5_path, p.water_yield_cur_path], local_op, p.water_depletion_factor_ssp5_path, compress=True, add_overviews=True)
        # hb.raster_calculator_hb()

    p.countries_marine_polygons_path = os.path.join(p.model_base_data_dir, r"countries_marine_polygons.shp")
    p.countries_marine_polygons_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\countries_marine_polygons.shp"
    p.countries_marine_polygons_path = r"C:\Users\jajohns\Files\Research\base_data\pyramids\country_marine_polygons.shp"
    p.water_yield_cur_aggregated_path = os.path.join(p.cur_dir, 'water_yield_cur_aggregated.csv')
    # p.zone_ids_raster_path = os.path.join(p.cur_dir, 'countries_marine_polygon_raster.tif')
    match = hb.ArrayFrame(r"C:\Users\jajohns\Files\Research\base_data\pyramids\ha_per_cell_300sec.tif")

    p.water_yield_cur_r_path = hb.suri(p.water_yield_cur_path, 'r')
    p.water_yield_ssp1_r_path = hb.suri(p.water_yield_ssp1_path, 'r')
    p.water_yield_gc_r_path = hb.suri(p.water_yield_gc_path, 'r')
    p.water_yield_ssp5_r_path = hb.suri(p.water_yield_ssp5_path, 'r')

    p.water_impact_aggregated_ssp1_path = os.path.join(p.cur_dir, 'water_impact_ssp1.csv')
    p.water_impact_aggregated_gc_path = os.path.join(p.cur_dir, 'water_impact_gc.csv')
    p.water_impact_aggregated_ssp5_path = os.path.join(p.cur_dir, 'water_impact_ssp5.csv')
    # p.water_basin_gtap_19_shapefile_path = os.path.join(p.model_base_data_dir, "shapefiles/WaterBasinGTAP19.shp")
    p.water_basin_gtap_19_shapefile_path = os.path.join(p.model_base_data_dir, "shapefiles/gtap19_basins.shp")
    # p.water_basin_gtap_19_shapefile_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\shapefiles\BasinGTAP19_simple\BasinGTAP19_simple.shp"

    p.unique_zone_ids = np.arange(1, 216 + 1, 1, np.int64)

    # p.zone_ids_raster_path = None  # Forcepgp
    p.zone_ids_raster_path = os.path.join(p.cur_dir, 'gtap19_basin_ids.tif')
    p.zone_ids_column_name = 'g19b_id'

    p.new_zone_ids_raster_path = os.path.join(p.cur_dir, 'mg140_ids.tif')

    if p.run_this:

        # First get baseline zonal stats
        if not os.path.exists(p.water_yield_cur_aggregated_path):
            csv_output_path = os.path.join(p.cur_dir, 'cur_water_zonal_stats.csv')
            stats_cur = hb.zonal_statistics_flex(p.water_yield_cur_r_path, p.mgtap140_path,
                                                 zone_ids_raster_path=p.new_zone_ids_raster_path, id_column_label='pyramid_id', data_type=6, ndv=-9999,
                                                 all_touched=None, assert_projections_same=False, use_iterblocks=True, csv_output_path=csv_output_path)

            hb.python_object_to_csv(OrderedDict(stats_cur), output_uri=p.water_yield_cur_aggregated_path, csv_type='rc_2d_odict')

        if not os.path.exists(p.water_impact_aggregated_ssp1_path):
            csv_output_path = os.path.join(p.cur_dir, 'sp_water_zonal_stats.csv')
            stats_1 = hb.zonal_statistics_flex(p.water_yield_ssp1_r_path, p.mgtap140_path,
                                               zone_ids_raster_path=p.new_zone_ids_raster_path, id_column_label='pyramid_id', data_type=6, ndv=-9999,
                                               all_touched=None, assert_projections_same=False, use_iterblocks=True, csv_output_path=csv_output_path)
            hb.python_object_to_csv(OrderedDict(stats_1), output_uri=p.water_impact_aggregated_ssp1_path, csv_type='rc_2d_odict')

        if not os.path.exists(p.water_impact_aggregated_gc_path):
            csv_output_path = os.path.join(p.cur_dir, 'gc_water_zonal_stats.csv')
            stats_3 = hb.zonal_statistics_flex(p.water_yield_gc_r_path, p.mgtap140_path,
                                               zone_ids_raster_path=p.new_zone_ids_raster_path, id_column_label='pyramid_id', data_type=6, ndv=-9999,
                                               all_touched=None, assert_projections_same=False, use_iterblocks=True, csv_output_path=csv_output_path)
            hb.python_object_to_csv(OrderedDict(stats_3), output_uri=p.water_impact_aggregated_gc_path, csv_type='rc_2d_odict')

        if not os.path.exists(p.water_impact_aggregated_ssp5_path):
            csv_output_path = os.path.join(p.cur_dir, 'bau_water_zonal_stats.csv')
            stats_5 = hb.zonal_statistics_flex(p.water_yield_ssp5_r_path, p.mgtap140_path,
                                               zone_ids_raster_path=p.new_zone_ids_raster_path, id_column_label='seals_id', data_type=6, ndv=-9999,
                                               all_touched=None, assert_projections_same=False, use_iterblocks=True, csv_output_path=csv_output_path)
            hb.python_object_to_csv(OrderedDict(stats_5), output_uri=p.water_impact_aggregated_ssp5_path, csv_type='rc_2d_odict')

    p.water_yield_results_excel_path = os.path.join(p.cur_dir, 'water_yield_results.xlsx')
    p.water_yield_results_shapefile_path = os.path.join(p.cur_dir, 'water_yield_results.shp')

    if p.run_this:
        df_cur = pd.read_csv(p.water_yield_cur_aggregated_path, index_col=0)

        df1 = pd.read_csv(p.water_impact_aggregated_ssp1_path, index_col=0)
        df3 = pd.read_csv(p.water_impact_aggregated_gc_path, index_col=0)
        df5 = pd.read_csv(p.water_impact_aggregated_ssp5_path, index_col=0)

        df = pd.merge(df1[['sum', 'count']], df3[['sum', 'count']], left_index=True, right_index=True, suffixes=[1, 3])
        df = pd.merge(df, df5[['sum', 'count']], left_index=True, right_index=True)
        df.rename({'sum': 'sum5', 'count': 'count5'}, axis=1, inplace=True)

        df = pd.merge(df, df_cur[['sum', 'count']], left_index=True, right_index=True)
        df.rename({'sum': 'sum_cur', 'count': 'count_cur'}, axis=1, inplace=True)

        df['wy_d1'] = (df['sum1']) / df['sum_cur']
        df['wy_d3'] = (df['sum3']) / df['sum_cur']
        df['wy_d5'] = (df['sum5']) / df['sum_cur']

        gdf = gpd.read_file(p.water_basin_gtap_19_shapefile_path)
        merged_gpd = gdf.merge(df, left_on='g19b_id', right_index=True)

        merged_gpd.to_file(p.water_yield_results_shapefile_path)

        excel_df = merged_gpd.drop('geometry', axis=1)
        excel_df.to_excel(p.water_yield_results_excel_path)


def gtap_outputs_and_produce_figures(p):

    p.gtap_results_xlsx_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\projects\calc_biophysical_shocks_0_5_0\input\UNGA Figures 10-14-2019_CN.xlsx"
    p.gtap_aez_vector_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\GTAPv9_AEZ18.shp"
    p.lulc_gc_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\lulc\lulc_gc_esa_classes.tif"
    p.lulc_gc_five_class_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\lulc\lulc_gc.tif"
    p.country_ids_10s_path = r"C:\Users\jajohns\Files\Research\base_data\pyramids\country_ids_10s.tif"
    p.lulc_sp_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\base_data\lulc\GLOBIO4_LU_10sec_2050_ssp1_rcp26_md5_803166420f51e5ef7dcaa970faa98173.tif"
    if p.run_this:
        df = pd.read_excel(p.gtap_results_xlsx_path, 'WorldSupply', skiprows=2)

        def op(x):
            return np.where((x >= 50) & (x <= 180), 1, 0)

        p.natural_land_gc_mask_path = os.path.join(p.cur_dir, 'natural_land_gc_mask.tif')
        p.natural_land_gc_mask_r_path = os.path.join(p.cur_dir, 'natural_land_gc_mask_r.tif')
        # hb.raster_calculator_af_flex(p.lulc_gc_path, op, p.natural_land_gc_mask_path, compress=True)

        p.natural_land_sp_mask_path = os.path.join(p.cur_dir, 'natural_land_sp_mask.tif')
        p.natural_land_sp_mask_r_path = os.path.join(p.cur_dir, 'natural_land_sp_mask_r.tif')
        # hb.raster_calculator_af_flex(p.lulc_sp_path, op, p.natural_land_sp_mask_path, compress=True)

        p.new_zone_ids_raster_path = os.path.join(p.aggregate_water_impact_factors_dir, 'mg140_ids.tif')
        p.upsampled_zone_ids_raster_path = os.path.join(p.cur_dir, 'mg140_ids_upsampled.tif')

        # hb.resample_to_match(p.natural_land_gc_mask_path, p.new_zone_ids_raster_path, p.natural_land_gc_mask_r_path, resample_method='near')
        # hb.resample_to_match(p.natural_land_sp_mask_path, p.new_zone_ids_raster_path, p.natural_land_sp_mask_r_path, resample_method='near')

        # a = hb.as_array(p.new_zone_ids_raster_path).astype(np.int)
        # b = hb.naive_upsample_byte(a, 30)
        # hb.save_array_as_geotiff(b, p.upsampled_zone_ids_raster_path, p.natural_land_gc_mask_path)

        csv_output_path = os.path.join(p.cur_dir, 'coastal_adjacency_gc_stats.csv')
        stats_cur = hb.zonal_statistics_flex(p.natural_land_gc_mask_r_path, p.mgtap140_path,
                                             zone_ids_raster_path=p.new_zone_ids_raster_path, id_column_label='seals_id',
                                             all_touched=None, assert_projections_same=False, use_iterblocks=True, csv_output_path=csv_output_path)
        hb.python_object_to_csv(OrderedDict(stats_cur), output_uri=p.water_yield_cur_aggregated_path, csv_type='rc_2d_odict')

        csv_output_path = os.path.join(p.cur_dir, 'coastal_adjacency_sp_stats.csv')
        stats_cur = hb.zonal_statistics_flex(p.natural_land_sp_mask_r_path, p.mgtap140_path,
                                             zone_ids_raster_path=p.new_zone_ids_raster_path, id_column_label='seals_id',
                                             all_touched=None, assert_projections_same=False, use_iterblocks=True, csv_output_path=csv_output_path)
        hb.python_object_to_csv(OrderedDict(stats_cur), output_uri=p.water_yield_cur_aggregated_path, csv_type='rc_2d_odict')

        gdf = gpd.read_file(p.countries_marine_polygons_path)
        gdf = gdf[['ISO_3digit', 'ID', 'GTAP140']]
        gdf['seals_id'] = gdf.index + 1

        df = pd.read_csv(os.path.join(p.cur_dir, 'coastal_adjacency_gc_stats.csv'), index_col=0)
        df['seals_id'] = df.index
        merged_gpd1 = gdf.merge(df, left_on='seals_id', right_on='seals_id')
        csv_output_path = os.path.join(p.cur_dir, 'coastal_adjacency_gc_stats_merged.xlsx')
        merged_gpd1.to_excel(csv_output_path)

        gdf = gpd.read_file(p.countries_marine_polygons_path)
        gdf = gdf[['ISO_3digit', 'ID', 'GTAP140']]
        gdf['seals_id'] = gdf.index + 1

        df = pd.read_csv(os.path.join(p.cur_dir, 'coastal_adjacency_sp_stats.csv'), index_col=0)
        df['seals_id'] = df.index
        merged_gpd2 = gdf.merge(df, left_on='seals_id', right_on='seals_id')
        csv_output_path = os.path.join(p.cur_dir, 'coastal_adjacency_sp_stats_merged.xlsx')
        merged_gpd2.to_excel(csv_output_path)

        df3 = pd.read_excel(p.gtap_results_xlsx_path, 'GDP in M USD', skiprows=1)
        df3['RegionsCap'] = df3['Regions'].str.upper()
        df4 = df3.merge(merged_gpd1, left_on='RegionsCap', right_on='GTAP140', how='left')
        df5 = df4.merge(merged_gpd2, left_on='RegionsCap', right_on='GTAP140', how='left')

        df6 = pd.read_excel(os.path.join(p.calc_marine_fisheries_shock_dir, 'results_joined.xlsx'), index_col=0)  # , 'GDP in M USD', skiprows=1
        df6['seals_id'] = df6.index
        merged_gpd3 = gdf.merge(df6, left_on='seals_id', right_on='seals_id')
        csv_output_path = os.path.join(p.cur_dir, 'asdf.xlsx')
        merged_gpd3.to_excel(csv_output_path)

        df7 = df5.merge(merged_gpd3, left_on='RegionsCap', right_on='GTAP140', how='left')
        p.combined_results_xlsx_path = os.path.join(p.cur_dir, 'combined_results.xlsx')
        df7.to_excel(p.combined_results_xlsx_path)


def marine_boundaries(p):
    p.rt_points_path = r"C:\Users\jajohns\Files\Research\cge\gtap_invest\shared_inputs\coastal vulnerability_2018-10-15\shapefile_Rt\Rt.shp"
    p.gtap_regions_path = os.path.join(p.input_dir, "gtapv9140_07-09-2018.shp")
    p.marine_zones_path = os.path.join(hb.BASE_DATA_DIR, r"cartographic\marineregions\EEZ_land_union_v2_201410\EEZ_land_v2_201410.shp")
    p.nev_countries_path = r"C:\Users\jajohns\Files\Research\base_data\cartographic\naturalearth_v3\10m_cultural\ne_10m_admin_0_countries.shp"

    p.marine_inclusive_gtap_regions_path = os.path.join(p.cur_dir, 'marine_inclusive_gtap_regions.shp')

    p.points_in_poly_shapefile_path = os.path.join(p.cur_dir, 'pip.shp')
    p.points_in_poly_excel_path = os.path.join(p.cur_dir, 'pip.xlsx')

    p.cv_results_by_ssp_path = os.path.join(p.cur_dir, "cv_results_by_ssp.shp")

    p.cv_results_aggregated_to_gtap_zones_path = os.path.join(p.cur_dir, "cv_results_aggregated_to_gtap_zones.shp")

    p.esa_path = r"C:\Users\jajohns\Files\Research\base_data\lulc\esacci\ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992-v2.0.7.tif"
    p.land_binary_path = os.path.join(p.cur_dir, 'esa_land_binary.tif')
    p.country_marine_polygons_path = r"C:\Users\jajohns\Files\Research\base_data\cartographic\country_marine_polygons.shp"

    if p.run_this:
        marine_zones = gpd.read_file(p.marine_zones_path)
        rt_points = gpd.read_file(p.rt_points_path)
        gtap_regions = gpd.read_file(p.gtap_regions_path)
        nev_countries = gpd.read_file(p.nev_countries_path)

        do = False
        if do:
            # Create land-binary path.

            start = time.time()
            esa_ds = gdal.Open(p.esa_path)
            esa_array = esa_ds.GetRasterBand(1).ReadAsArray()  # .astype(np.int8)  # callback=reporter
            land_binary = np.where((esa_array != 210) & (esa_array != 0), 1, 0)
            hb.save_array_as_geotiff(land_binary, p.land_binary_path, p.esa_path, data_type=1, ndv=255)

            # Merge GTAP Regions into marine zones.
            joined = marine_zones.merge(gtap_regions[['ISO', 'ID', 'GTAP140']], left_on='ISO_3digit', right_on='ISO', how='left')
            nev_countries = nev_countries.drop(columns=['geometry'])

            # Merge NEV into marine zones
            columns = ['featurecla', 'scalerank', 'LABELRANK', 'SOVEREIGNT', 'SOV_A3', 'ADM0_DIF', 'LEVEL', 'TYPE', 'ADMIN', 'ADM0_A3', 'GEOU_DIF', 'GEOUNIT', 'GU_A3', 'SU_DIF', 'SUBUNIT', 'SU_A3', 'BRK_DIFF', 'NAME', 'NAME_LONG', 'BRK_A3', 'BRK_NAME', 'BRK_GROUP', 'ABBREV', 'POSTAL', 'FORMAL_EN', 'FORMAL_FR', 'NAME_CIAWF', 'NOTE_ADM0', 'NOTE_BRK', 'NAME_SORT', 'NAME_ALT', 'POP_EST', 'POP_RANK', 'GDP_MD_EST', 'POP_YEAR', 'LASTCENSUS', 'GDP_YEAR', 'ECONOMY', 'INCOME_GRP', 'WIKIPEDIA', 'FIPS_10_', 'ISO_A2', 'ISO_A3', 'ISO_A3_EH', 'ISO_N3', 'UN_A3', 'WB_A2', 'WB_A3', 'WOE_ID', 'WOE_ID_EH', 'WOE_NOTE', 'ADM0_A3_IS', 'ADM0_A3_US', 'ADM0_A3_UN', 'ADM0_A3_WB', 'CONTINENT', 'REGION_UN', 'SUBREGION', 'REGION_WB', 'NAME_LEN', 'LONG_LEN', 'ABBREV_LEN', 'TINY', 'HOMEPART', 'MIN_ZOOM', 'MIN_LABEL', 'MAX_LABEL', 'NE_ID', 'WIKIDATAID']
            joined = joined.merge(nev_countries[columns], left_on='ISO_3digit', right_on='ADM0_A3', how='left')
            joined.to_file(p.marine_inclusive_gtap_regions_path)

        country_marine_polygons = gpd.read_file(p.country_marine_polygons_path)
        country_marine_polygons = country_marine_polygons[['ISO_3digit', 'GTAP140', 'geometry']]

        if not os.path.exists(p.points_in_poly_shapefile_path):
            points_in_polys = gpd.sjoin(rt_points, country_marine_polygons, how='inner')
            points_in_polys.to_file(p.points_in_poly_shapefile_path)
            # points_in_polys.to_excel(p.points_in_poly_excel_path)
        else:
            points_in_polys = gpd.read_file(p.points_in_poly_shapefile_path)

        # sum_per_country = points_in_polys.groupby('PolyGroupByField')['fields', 'in', 'grouped', 'output'].agg(['sum'])

        if not os.path.exists(p.cv_results_by_ssp_path):
            cv = points_in_polys
            # Scenario differences from current. (not very useful because index)
            cv['ssp1_d'] = cv['Rt_ssp1'] - cv['Rt_cur']
            cv['ssp3_d'] = cv['Rt_ssp3'] - cv['Rt_cur']
            cv['ssp5_d'] = cv['Rt_ssp5'] - cv['Rt_cur']

            # # Scenario means
            # cv['cur_m'] = cv['Rt_cur'] / cv['NUMPOINTS']
            # cv['ssp1_m'] = cv['Rt_ssp1'] / cv['NUMPOINTS']
            # cv['ssp3_m'] = cv['Rt_ssp3'] / cv['NUMPOINTS']
            # cv['ssp5_m'] = cv['Rt_ssp5'] / cv['NUMPOINTS']
            #
            # # Mean difference
            # cv['ssp1_md'] = cv['ssp1_m'] - cv['cur_m']
            # cv['ssp3_md'] = cv['ssp3_m'] - cv['cur_m']
            # cv['ssp5_md'] = cv['ssp5_m'] - cv['cur_m']

            # Calcualte number that are high-risk, ie above 3.3.
            def op(x, y):
                np.where(x > y, 1, 0)

            # LEARNING Point, this appears to be the fastest way to calculate a new column, ie vectorizing the funciton over numpy arrays, which is better than using apply()
            cv['cur_hr'] = np.where(cv['Rt_cur'].values >= 3.3, 1, 0)
            cv['ssp1_hr'] = np.where(cv['Rt_ssp1'].values >= 3.3, 1, 0)
            cv['ssp3_hr'] = np.where(cv['Rt_ssp3'].values >= 3.3, 1, 0)
            cv['ssp5_hr'] = np.where(cv['Rt_ssp5'].values >= 3.3, 1, 0)

            # Number of new cels at high risk
            cv['ssp1_hrd'] = cv['ssp1_hr'] - cv['cur_hr']
            cv['ssp3_hrd'] = cv['ssp3_hr'] - cv['cur_hr']
            cv['ssp5_hrd'] = cv['ssp5_hr'] - cv['cur_hr']

            cv.to_file(p.cv_results_by_ssp_path)
            # # Number of new cels at high risk
            # cv['ssp1_if'] = 1.0 - (cv['ssp1_hrd'] / cv['NUMPOINTS'])
            # cv['ssp3_if'] = 1.0 - (cv['ssp3_hrd'] / cv['NUMPOINTS'])
            # cv['ssp5_if'] = 1.0 - (cv['ssp5_hrd'] / cv['NUMPOINTS'])
        else:
            cv = gpd.read_file(p.cv_results_by_ssp_path)


def extract_hars_to_xls():
    global p

    if p.run_this:
        target_dir = os.path.join(p.input_dir, "GTAP10_GTAP_2014_65x141_1to1")
        # input_dir = r"C:\GTPAg2\GTAP10\GTAP\2014"

        hb.create_directories(p.output_dir)
        # ge.gtap_interface.extract_gtap_to_xls(target_dir, p.output_dir)

        csv_dir = os.path.join(p.cur_dir, 'csvs')
        hb.create_directories(csv_dir)



        for file_path in hb.list_filtered_paths_nonrecursively(p.output_dir, include_extensions='.xls', include_strings='co2'):

            wb = xlrd.open_workbook(file_path)
            sheet_names = wb.sheet_names()

            for sheet_name in sheet_names[0:7]:

                sh = wb.sheet_by_name(sheet_name)
                num_cols = sh.ncols

                if num_cols == 1:
                    print ('Is history tab, skipping')
                else:
                    csv_path = os.path.join(csv_dir, sheet_name + '.csv')
                    csv_file = open(csv_path, 'w')
                    # csv_file = open(csv_path, 'wb')
                    wr = csv.writer(csv_file, quoting=csv.QUOTE_NONE)  # quoting=csv.QUOTE_ALL

                    num_rows = sh.nrows
                    num_cols = len(sh.row_values(0))

                    if all(cell == '' for cell in sh.row_values(0)):
                        first_row_blank = True
                    else:
                        first_row_blank = False

                    if sh.row_values(0)[1] == 'Real Matrix':
                        is_real_matrix = True
                    else:
                        is_real_matrix = False

                    if str(sh.row_values(5)[0]).startswith('Next block shows'):
                        multi_dimensional = True
                    else:
                        multi_dimensional = False

                    if first_row_blank:
                        for n_r in range(1, num_rows):
                            wr.writerow(sh.row_values(n_r))



                    # wr.writerow(sh.row_values(n_r))
                    csv_file.close()



if __name__ == '__main__':
    print ('You called gtap_invest_main.py directly. Perhaps you want to run a config file like run_getap_invest_project_name.py')