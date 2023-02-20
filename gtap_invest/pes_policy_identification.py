"""Module for calculating the selective set of grid-cells for extensification 
and comparing it to a buisiness as usual scenario"""

import logging
import os
import csv
import math, time

import numpy
import scipy as sp
import matplotlib.pyplot as plt
from collections import OrderedDict

np = numpy
from osgeo import gdal, gdalconst
# from geoecon import utils as gu
# from geoecon import visualization as gv
# import geoecon as ge
import hazelbean as hb
import pandas as pd

import multiprocessing
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('classic')


#setup cython functions
import pyximport
pyximport.install(setup_args={"script_args":["--compiler=mingw32"],"include_dirs":numpy.get_include()},reload_support=True)
#from geoecon_utils import geoecon_cython_utils as gcu

log_id = hb.pretty_time()
LOGGER = logging.getLogger('ffn')
LOGGER.setLevel(logging.DEBUG) # warn includes the final output of the whole model, carbon saved.


def execute(args, ui=None):
    """Wrapper function that pulls the model functions (e.g. bau/selective) and
    compares them. The model functions are responsible for producing output, 
    using optional args passed in via args dict"""

    # Set default args dictionary. Parts will be overridden with the three possible input methods (GeoEcon, standalone
    # application or runtime script).
    default_args = build_default_args()

    # Overwrite default_args with user-inputted args
    for name in args:
        default_args[name] = args[name]
    args = default_args

    args['parameters_summary_string'] = str(args['conservation_budget']) + '_' +  str(args['social_cost_of_carbon']) + '_' +  str(args['asymmetric_scatter']) + '_' +  str(args['npv_adjustment'])
    args.update({'run_id': hb.pretty_time()}) #the run_id is a unique identifier for each run, generated via time-stamp to the second.

    if 'run_folder' in args:
        5
    else:
        args.update({'run_folder': r'C:\Files\Research\cge\gtap_invest\projects\pes_policy_identification\subs_budget_half'}) #This script generates a new folder of this name in the output_folder
    # args.update({'run_folder': os.path.join(args['bulk_output_folder'], 'run_at_' + args['run_id'] + '_' + args['parameters_summary_string'])}) #This script generates a new folder of this name in the output_folder

    # Make the run_folder if it doesn't exist.
    try: os.makedirs(args['run_folder'])
    except: LOGGER.warn('run_folder is already there. This shouldnt happen because the run_folder is postpended with the run_id timestamp. Are you over-riding something you shouldnt?')

    print('args2', args)
    file_handler = logging.FileHandler(os.path.join(args['run_folder'], 'run_summary_' + log_id + '.txt'))
    LOGGER.addHandler(file_handler)

    # # Load default files from input_folder
    # proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'), verbose = args['verbose'])
    # ha_per_cell = hb.as_array(os.path.join(args['input_folder'], 'ha_per_cell.tif'), verbose = args['verbose'])


    # Create per-crop information and save in the bulk_output_folder
    if args['calculate_yield_tons_per_cell']: # Not needed right now.
        LOGGER.debug('Summing per-cell yield over 172 crops')
        add_yield_tons_per_cell_across_crops(args)

    #Create Soil Carbon info
    if args['aggregate_soil_carbon_data']:
        LOGGER.debug('Aggregating 1km soil carbon data to 10km.')
        aggregate_soil_carbon_data(args)
    else:
        args['soil_carbon_uri'] = os.path.join(args['input_folder'], 'soil_carbon_tons_per_ha.tif')


    # Convert yield to calories
    if args['calculate_calories_per_cell']:
        LOGGER.debug('Calculating new values for calories_per_cell.')
        add_calories_per_cell_across_crops(args)

    # Calculate change in carbon
    if args['calculate_change_in_carbon']:
        LOGGER.debug('Calculating change in carbon.')
        calculate_change_in_carbon(args)

    if args['calculate_bau_and_selective']:
        LOGGER.debug('Calculating BAU and Selective scenarios')
        calculate_bau_and_selective(args)

    # Output publication figures
    if args['produce_ag_tradeoffs_publication_figures']:
        LOGGER.debug('Running produce_ag_tradeoffs_publication_figures')
        produce_ag_tradeoffs_publication_figures(args)

    if args['create_base_ffn_inputs']:
        LOGGER.debug('Calculating create_base_ffn_inputs.')
        create_base_ffn_inputs(args)

    if args['produce_base_data_figs']:
        LOGGER.debug('Running produce_base_data_figs')
        produce_base_data_figs(args)


    if args['calculate_ffn_b_payments']:
        LOGGER.debug('Calculating FFN.')
        calculate_ffn_b_payments(args)


    if args['calculate_ffn_c_payments']:
        LOGGER.debug('Calculating FFN.')
        calculate_ffn_c_payments(args)

    if args['calculate_ffn_f_payments']:
        LOGGER.debug('Calculating calculate_ffn_f_payments.')
        args = calculate_ffn_f_payments(args)

    if args['calculate_ffn_a_payments']:
        LOGGER.debug('Calculatingn calculate_ffn_a_payments')
        calculate_ffn_a_payments(args)

    if args['calculate_ffn_f_country_level_payments']:
        LOGGER.debug('Calculating calculate_ffn_f_country_level_payments.')
        calculate_ffn_f_country_level_payments(args)



    if args['produce_base_mechanism_figs']:
        LOGGER.debug('Running produce_base_mechanism_figs')
        produce_base_mechanism_figs(args)

    if args['produce_fractional_value_plot']:
        LOGGER.debug('Running produce_fractional_value_plot')
        produce_fractional_value_plot(args)

    if args['produce_auction_mechaism_figs']:
        LOGGER.debug('Running produce_auction_mechaism_figs')
        produce_auction_mechaism_figs(args)

    LOGGER.info('Script finished.\n\n\nParameters used:\n')

    for name, value in enumerate(args):
        LOGGER.debug(str(value) + ': ' + str(args[value]))

    return

def build_default_args():
    default_args = {}
    default_args.update({'workspace': r"c:\Files/research\cge\gtap_invest\projects\\pes_policy_identification//"}) #sets the default location
    default_args.update({'output_folder': os.path.join(default_args['workspace'] + 'output/')}) #base folder where all run_folders are placed
    default_args.update({'input_folder': os.path.join(default_args['workspace'] + 'input/')}) #put pre-generated, non-bulk input data (such as those you may have created in previous runs but don't want to regenerate each time) here
    default_args.update({'intermediate_folder': os.path.join(default_args['workspace'] + 'intermediate/')}) #put pre-generated, non-bulk input data (such as those you may have created in previous runs but don't want to regenerate each time) here
    default_args.update({'base_data_dir': os.path.join('C:/Files/research/base_data')})
    default_args.update({'bulk_input_folder': os.path.join(default_args['workspace'] + 'input/')}) #for data larger than 1 gb
    default_args.update({'bulk_output_folder': os.path.join('c:/temp')}) #for data larger than 1 gb
    default_args.update({'crop_statistics_folder': os.path.join(default_args['base_data_dir'], 'crops/earthstat/crop_production/')}) #location of Earthstat data. Must have 172 crop-specific folders (as provided by earthstat.org)
    default_args.update({'soil_carbon_input_uri': os.path.join(default_args['workspace'] + 'input/1kmsoilgrids/OCSTHA_sd6_M_02_apr_2014.tif')}) #the current version of Ag_tradeoffs uses 1km soil carbon data, located in this geotiff.
    default_args.update({'soil_carbon_tons_per_cell_uri': os.path.join(default_args['input_folder'],'soil_carbon_tons_per_cell.tif')}) #The 1km soil carbon data has to be aggregated up to 10km. The results are saved here.
    default_args.update({'potential_natural_vegetation_carbon_tons_per_cell_uri': os.path.join(default_args['input_folder'],'potential_carbon_if_no_cultivation.tif')})
    default_args.update({'crop_carbon_tons_per_cell_uri': os.path.join(default_args['input_folder'],'crop_carbon_tons_per_cell.tif')})
    default_args.update({'match_uri': 'c:/Files/research/base_data/pyramids/ha_per_cell_300sec.tif'}) #The match file identifies the geotransform, extent, resolution, projection etc. that all other files must be in.

    default_args.update({'parameters_uri': os.path.join(default_args['input_folder'],'parameters.csv')}) #all crop-specific parameters are defined here. Non-crop parameters are defined in this run file.

    default_args.update({'verbose': True}) #If true, will show extra information about what arrays were loaded and from where, with stats.

    # Most of these variables will be set by the user, but in the event that they want to run using the same parameters that went into PNAS2014, these are set as default.
    default_args.update({'calorie_increase': 0.7,
                         'assumed_intensification': .75, #what portion of the calorie increase will be met with higher yields?
                         'transition_function_alpha': 1.0, #the transition function is post_extensification_prop_cultivated = alpha*initial_prop_cultivated ^ beta
                         'transition_function_beta': 0.5,
                         'solution_precision': 0.01, #lower values result in more answer precision
                         'min_extensification': 0.05, #assume no extensification can happen in cells with less current extensifcation than this
                         'max_extensification': 0.95, #assume no extensification can happen in cells with more current extensifcation than this
                         'conservation_budget': 1000000000,
                         'social_cost_of_carbon': 75.0,
                         'asymmetric_scatter': 0.03,
                         'npv_adjustment': 10.91744579, #based on 0.08 discount rate into exp(-rt) over 30 years.
                         })

    # Because several of the steps are computationally slow, the following default_args enable/disable each of the steps.
    # and instead load pregenerated files saved in the results_folder. By default, all parts will be run.
    default_args.update({'calculate_yield_tons_per_cell': True, #Not needed becuase we use calories only.
                         'aggregate_soil_carbon_data': False, # Decided to just use land-econ version
                         'calculate_calories_per_cell': True,
                         'calculate_change_in_carbon': True,
                         'calculate_bau_and_selective': True,
                         'produce_ag_tradeoffs_publication_figures': False,
                         'create_base_ffn_inputs': True,
                         'calculate_ffn_b_payments': True,
                         'calculate_ffn_f_payments': True,
                         'calculate_ffn_a_payments': True,
                         'produce_ffn_publication_figures': True,
                         })
    return default_args

def add_yield_tons_per_cell_across_crops(args):
    crop_names = []
    for folder in os.listdir(args['crop_statistics_folder']):
        if not os.path.splitext(folder)[1]:  # select anything without file extensions (folders)
            crop_names.append(folder.split('_')[0])

    if not hb.path_exists(os.path.join(args['input_folder'],'yield_tons_per_cell.tif')):
        aggregate_yield_sum = 0
        aggregate_yield_from_input_sum = 0
        yield_tons_per_cell = np.zeros((2160, 4320))
        for crop_name in crop_names:
            crop_folder_uri = os.path.join(args['crop_statistics_folder'], crop_name + '_HarvAreaYield_Geotiff')
            hectares_harvested = hb.as_array(os.path.join(crop_folder_uri, crop_name + '_HarvestedAreaHectares.tif'), verbose = args['verbose'])
            yield_tons_per_ha = hb.as_array(os.path.join(crop_folder_uri, crop_name + '_YieldPerHectare.tif'), verbose = args['verbose'])
            yield_tons_per_cell_crop = hb.as_array(os.path.join(crop_folder_uri, crop_name + '_Production.tif'), verbose = args['verbose'])

            if crop_name == 'chicory': #there is a large data error in Chicory. To get the correct results, I divided everything by 10E+17, which then made it match publicaiton maps.
                yield_tons_per_cell_crop = hectares_harvested * yield_tons_per_ha
            yield_tons_per_cell_sum = np.nansum(yield_tons_per_cell_crop)
            yield_tons_per_cell_from_input_sum = np.nansum(yield_tons_per_cell_crop)
            aggregate_yield_sum += yield_tons_per_cell_sum
            yield_tons_per_cell += yield_tons_per_cell_crop
            LOGGER.info('Summed yield per cell for,' + crop_name + ',' + str(yield_tons_per_cell_sum) + ',' + str(yield_tons_per_cell_from_input_sum)+ ',' + str(aggregate_yield_sum)+ ',' + str(aggregate_yield_from_input_sum))

        hb.save_array_as_geotiff(yield_tons_per_cell,os.path.join(args['run_folder'],'yield_tons_per_cell.tif'),args['match_uri'], data_type=6, ndv = -255)
    return_string = 'Finished sum_yield_tons_per_cell.'

    return return_string


def aggregate_soil_carbon_data(args):
    print('Not needed')
    print('run_folder', )
    # ha_per_cell = hb.as_array(os.path.join(args['base_data_dir'], 'pyramids', 'ha_per_cell_300sec.tif'))
    # uri_out = os.path.join(args['run_folder'], 'soil_carbon_tons_per_ha.tif')
    #
    # #SKIPPING THIS WONT WORK BEACUSE I DIDNT BRINGIN THIS TO  NUMDAL
    # # BORT.aggregate_geotiff(args['soil_carbon_input_uri'], uri_out, match_uri = args['match_uri'])
    # soil_carbon_tons_per_ha = hb.as_array(uri_out, verbose = args['verbose'])
    # soil_carbon_tons_per_cell = ha_per_cell * soil_carbon_tons_per_ha
    # hb.save_array_as_geotiff(soil_carbon_tons_per_cell, os.path.join(args['run_folder'], 'soil_carbon_tons_per_cell.tif'),args['match_uri'], data_type=6,  ndv=-255)


def add_calories_per_cell_across_crops(args):
    ha_per_cell = hb.as_array(os.path.join(args['base_data_dir'], 'pyramids', 'ha_per_cell_300sec.tif'))
    proportion_cultivated = hb.as_array(os.path.join(args['base_data_dir'], 'publications\\ag_tradeoffs\\land_econ', 'proportion_cultivated.tif'))
    crop_names = []

    for folder in os.listdir(args['crop_statistics_folder']):
        if not os.path.splitext(folder)[1]:  # select anything without file extensions (folders)
            crop_names.append(folder.split('_')[0])

    parameters = hb.file_to_python_object(args['parameters_uri'], declare_type='DD')

    if not hb.path_exists(os.path.join(args['input_folder'],'calories_per_cell.tif')):
        calories_per_cell = np.zeros((2160, 4320))
        calories_per_ha_weighted_by_present_mix = np.zeros((2160, 4320))
        for crop_name in crop_names:
            crop_folder_uri = os.path.join(args['crop_statistics_folder'], crop_name + '_HarvAreaYield_Geotiff')
            hectares_harvested = hb.as_array(os.path.join(crop_folder_uri, crop_name + '_HarvestedAreaHectares.tif'), verbose = args['verbose'])
            yield_tons_per_ha = hb.as_array(os.path.join(crop_folder_uri, crop_name + '_YieldPerHectare.tif'), verbose = args['verbose'])
            current_crop_calories_per_ha = yield_tons_per_ha * float(parameters[crop_name]['Kcal/Kg']) * 1000.0
            calories_per_ha_weighted_by_present_mix += np.where(proportion_cultivated>0, current_crop_calories_per_ha * (hectares_harvested / (ha_per_cell * proportion_cultivated)),0.0)
            current_crop_yield_tons_per_cell = hectares_harvested * yield_tons_per_ha
            current_crop_calories_per_cell = current_crop_yield_tons_per_cell * float(parameters[crop_name]['Kcal/Kg']) * 1000.0
            current_crop_yield_tons_per_cell_sum = np.nansum(current_crop_calories_per_cell)
            calories_per_cell += current_crop_calories_per_cell
            calories_per_cell_running_sum = np.nansum(calories_per_cell)
            hb.save_array_as_geotiff(current_crop_calories_per_ha,os.path.join(args['run_folder'], crop_name + '_calories_per_ha.tif'),args['match_uri'], data_type=6, ndv = -255)
            LOGGER.info('Summed calories per cell for,' + crop_name + ',' + str(current_crop_yield_tons_per_cell_sum) + ',' + 'calorie_sum' + ',' + str(calories_per_cell_running_sum)+',np sum calories_per_ha_weighted_by_present_mix,'+str(np.nansum(calories_per_ha_weighted_by_present_mix)))
        hb.save_array_as_geotiff(calories_per_ha_weighted_by_present_mix,os.path.join(args['run_folder'],'calories_per_ha_weighted_by_present_mix.tif'),args['match_uri'], data_type=6, ndv = -255, verbose=True)

        hb.save_array_as_geotiff(calories_per_cell,os.path.join(args['input_folder'],'calories_per_cell.tif'),args['match_uri'], data_type=6,  ndv = -255, verbose=True)

    return 'Finished sum_calories_per_cell.'


def calculate_change_in_carbon(args):
    change_in_carbon_complete_conversion = hb.as_array(args['crop_carbon_tons_per_cell_uri']) - hb.as_array(args['potential_natural_vegetation_carbon_tons_per_cell_uri']) - 0.25 * hb.as_array(args['soil_carbon_tons_per_cell_uri'])
    hb.save_array_as_geotiff(change_in_carbon_complete_conversion,os.path.join(args['run_folder'],'change_in_carbon_complete_conversion.tif'),args['match_uri'], data_type=6, ndv = -255)


def calc_proportional_increase_bau_extensification(args):
    # in the execute function (i.e., newly written vs. loaded.) Currently, I am approaching it as the either function
    # can be called and completed fully with just the args

    # Load arrays
    proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'), verbose = args['verbose'])
    if args['calculate_calories_per_cell']:
        calories_per_cell = hb.as_array(os.path.join(args['input_folder'],'calories_per_cell.tif'), verbose = args['verbose'])
    else:
        calories_per_cell = hb.as_array(os.path.join(args['input_folder'],'calories_per_cell.tif'), verbose = True)

    # Calculate and set initial values
    initial_calories = np.nansum(calories_per_cell)
    goal_calories = (args['calorie_increase'] + 1) * initial_calories
    calories_per_cell_post_intensification = calories_per_cell + calories_per_cell * args['calorie_increase'] * args['assumed_intensification']
    calories_per_proportion_cultivated_post_intensification = np.where(proportion_cultivated > 0, np.divide(calories_per_cell_post_intensification, proportion_cultivated), 0)
    total_calories_produced = 0
    previous_total_calories_produced = 0
    feasible_proportion_cultivated_increase = np.where((proportion_cultivated < args['max_extensification']) & (proportion_cultivated > args['min_extensification']),proportion_cultivated, 0)
    extensification_factor = .99  #initial guess, takes the place of beta

    # Simulation logic
    iteration_step = 0
    while 1:
        iteration_step += 1
        if (1 - args['solution_precision']) * goal_calories < total_calories_produced < (1 + args['solution_precision']) * goal_calories:
            LOGGER.info("Met goal within solution precision when calculating BAU")
            break
        if abs(previous_total_calories_produced - total_calories_produced) < total_calories_produced * (args['solution_precision'] / 20000000000000):
            LOGGER.info("Couldn't find any new lands when calculating BAU. Produced" + str(total_calories_produced))
            break
        proportion_cultivated_bau = np.where(feasible_proportion_cultivated_increase > 0, args['transition_function_alpha'] * feasible_proportion_cultivated_increase ** extensification_factor, proportion_cultivated)
        calories_per_cell_bau = calories_per_proportion_cultivated_post_intensification * proportion_cultivated_bau
        previous_total_calories_produced = total_calories_produced
        total_calories_produced = np.nansum(calories_per_cell_bau)
        extensification_factor *= total_calories_produced / goal_calories
        running_stats = 'Iteration: ' + str(iteration_step) + ', calories: ' + str(total_calories_produced)
        LOGGER.info(running_stats)

    hb.save_array_as_geotiff(proportion_cultivated_bau,os.path.join(args['run_folder'], 'proportion_cultivated_bau.tif'), args['match_uri'])

    return proportion_cultivated_bau

def calc_carbon_selective_extensification(args):
    # Load arrays

    print('dirs1', args['input_folder'])
    print('dirs2', args['run_folder'])
    proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
    ha_per_cell = hb.as_array(os.path.join(args['base_data_dir'], 'pyramids', 'ha_per_cell_300sec.tif'))

    if args['calculate_calories_per_cell']:
        calories_per_cell = hb.as_array(os.path.join(args['input_folder'],'calories_per_cell.tif')).astype(np.float32)
    else:
        calories_per_cell = hb.as_array(os.path.join(args['input_folder'],'calories_per_cell.tif')).astype(np.float32)
    if args['calculate_change_in_carbon']:
        change_in_carbon_complete_conversion = hb.as_array(os.path.join(args['run_folder'],'change_in_carbon_complete_conversion.tif'))
    else:
        change_in_carbon_complete_conversion = hb.as_array(os.path.join(args['intermediate_folder'],'change_in_carbon_complete_conversion.tif'))

    # Calculate and set initial values
    min_extensification = args['min_extensification']
    max_extensification = args['max_extensification']
    alpha = args['transition_function_alpha']
    beta = args['transition_function_beta']

    initial_calories = np.nansum(calories_per_cell)
    goal_calories = (args['calorie_increase'] + 1) * initial_calories
    calories_per_cell_post_intensification = calories_per_cell + calories_per_cell * args['calorie_increase'] * args['assumed_intensification'] # is args['calorie_increase']  the problenm
    calories_per_ha_post_intensification = calories_per_cell_post_intensification / ha_per_cell
    calories_per_proportion_cultivated_post_intensification = np.where(proportion_cultivated > 0, np.divide(calories_per_cell_post_intensification, proportion_cultivated), 0)
    change_in_carbon_complete_conversion_per_ha = change_in_carbon_complete_conversion / ha_per_cell

    # if carbon reduces when the cell is converted to crop (the normal case), use the ratio as per normal
    calories_per_carbon_by_ha = np.where(change_in_carbon_complete_conversion_per_ha < 0, calories_per_ha_post_intensification / ( -1.0 * change_in_carbon_complete_conversion_per_ha), 0).astype(np.float32)
    # but if carbon goes up (usually because irrigation), instead CA
    calories_per_carbon_by_ha = np.where(change_in_carbon_complete_conversion_per_ha > 0, calories_per_ha_post_intensification * change_in_carbon_complete_conversion_per_ha + 1, calories_per_carbon_by_ha) # The plus 1 is debatable, but in the event that there is a VERY close to zero change in carbon, this ensures the cell will be evaluated at least at the calories produced.

    hb.save_array_as_geotiff(calories_per_carbon_by_ha,os.path.join(args['run_folder'],'calories_per_carbon_by_ha.tif'),args['match_uri'], data_type=6, ndv = -255)

    total_calories_produced = 0
    previous_total_calories_produced = 0
    proportion_cultivated_selective = np.copy(proportion_cultivated)
    calories_per_carbon_threshold = np.nanmean(calories_per_carbon_by_ha)  #initialing thresholdcalories_per_carbonThreshold = np.nanmean(calories_per_carbon) #starting threshold

    #simulation logic
    iteration_step = 0
    while 1:
        iteration_step += 1
        if (1 - args['solution_precision']) * goal_calories < total_calories_produced < (
                    1 + args['solution_precision']) * goal_calories:
            LOGGER.info("Arrived at precision goal and produced (cal)when calculating Selective: "+str(total_calories_produced))
            break
        if abs(previous_total_calories_produced - total_calories_produced) < total_calories_produced * args[
            'solution_precision'] / 2000000:
            LOGGER.info("Couldn't find any new lands when calculating Selective. Produced" + str(total_calories_produced))
        proportion_cultivated_selective = np.where((calories_per_carbon_by_ha > calories_per_carbon_threshold)
                                                   & (proportion_cultivated < max_extensification)
                                                   & (proportion_cultivated > min_extensification),
                                                   alpha * proportion_cultivated ** beta,
                                                   proportion_cultivated)

        calories_per_cell_selective = calories_per_proportion_cultivated_post_intensification * proportion_cultivated_selective

        previous_total_calories_produced = total_calories_produced
        total_calories_produced = np.nansum(calories_per_cell_selective)

        calories_per_carbon_threshold = calories_per_carbon_threshold * ((total_calories_produced / goal_calories) ** 4)  #the 8 is a volitility parameter. if too low, takes long. if too high, bounces around not converging. 8 seems about right
        running_stats = 'Iteration: ' + str(iteration_step) + ', CA Threshold: ' + str(calories_per_carbon_threshold) + ', Calories produced: ' + str(total_calories_produced) + ', Precision: ' + str(total_calories_produced / goal_calories)
        LOGGER.info(running_stats)

    hb.save_array_as_geotiff(proportion_cultivated_selective, os.path.join(args['run_folder'],'proportion_cultivated_selective.tif'), args['match_uri'])

    #     #Per-iteration graphics display
    #     if ui != None:
    #         ui.parent_application.update_layer_stats(running_stats)
    #     if ui != None and 'show_in_results_log' in args:
    #         if args['show_in_results_log'] > 0:
    #             ui.parent_application.update_results(".")
    #
    #
    #             #save output_string as a txt file in the project's output folder
    #         #    open(os.path.join(args['run_folder'],"selective_model_results_" + args['run_id'] + ".csv"), "w").write(output_string)
    #
    # #export desired maps
    # output_uri = os.path.join(args['run_folder'],
    #                           "proportion_cultivated_selective.tif")
    # # proportion_cultivated_selective_ds = raster_utils.new_raster_from_base(proportion_cultivated_ds, output_uri,
    # #                                                                        'GTiff', -255, gdal.GDT_Float32)
    # # proportion_cultivated_selective_band = proportion_cultivated_selective_ds.GetRasterBand(1)
    # # proportion_cultivated_selective_band.WriteArray(proportion_cultivated_selective)
    # # if ui != None: ui.parent_application.matrix_viewer_dock.add_item(output_uri)
    #
    # if ui != None and 'show_in_results_log' in args:
    #    if args[ 'show_in_results_log'] > 0:
    #         ui.parent_application.update_results("total_calories_produced_selective," + str(total_calories_produced))
    #         ui.parent_application.update_results("calories_per_carbon_threshold," + str(calories_per_carbon_threshold))
    return proportion_cultivated_selective


def calculate_bau_and_selective(args):
    proportion_cultivated_bau = calc_proportional_increase_bau_extensification(args)
    proportion_cultivated_selective = calc_carbon_selective_extensification(args)
    proportion_cultivated = hb.as_array(os.path.join(args['input_folder'],'proportion_cultivated.tif'))

    if args['calculate_change_in_carbon']:
        change_in_carbon_complete_conversion = hb.as_array(os.path.join(args['run_folder'],'change_in_carbon_complete_conversion.tif'))
    else:
        change_in_carbon_complete_conversion = hb.as_array(os.path.join(args['intermediate_folder'],'change_in_carbon_complete_conversion.tif'))


    hb.save_array_as_geotiff(proportion_cultivated_bau, os.path.join(args['run_folder'], 'proportion_cultivated_bau.tif'), args['match_uri'], data_type=6,  ndv = -255)
    hb.save_array_as_geotiff(proportion_cultivated_selective, os.path.join(args['run_folder'], 'proportion_cultivated_selective.tif'), args['match_uri'], data_type=6,  ndv = -255)

    proportion_conserved = proportion_cultivated_bau - proportion_cultivated_selective
    hb.save_array_as_geotiff(proportion_conserved, os.path.join(args['run_folder'], 'proportion_conserved.tif'), args['match_uri'], data_type=6,  ndv = -255)

    carbon_change_bau = (proportion_cultivated_bau - proportion_cultivated) * change_in_carbon_complete_conversion
    hb.save_array_as_geotiff(carbon_change_bau, os.path.join(args['run_folder'], 'carbon_change_bau.tif'), args['match_uri'], data_type=6,  ndv = -255)

    carbon_change_selective = (proportion_cultivated_selective - proportion_cultivated) * change_in_carbon_complete_conversion
    hb.save_array_as_geotiff(carbon_change_selective, os.path.join(args['run_folder'], 'carbon_change_selective.tif'), args['match_uri'], data_type=6,  ndv = -255)
    carbon_conserved = -1 * proportion_conserved * change_in_carbon_complete_conversion #the -1 is because the variable is carbon CONSERVED whereas everything else was net carbon change.
    hb.save_array_as_geotiff(carbon_conserved, os.path.join(args['run_folder'], 'carbon_conserved.tif'), args['match_uri'], data_type=6,  ndv = -255)

    #calculte post-run results
    carbon_conserved_sum = np.nansum(carbon_conserved)
    LOGGER.info('Calculation of BAU and Selective scenarios complete!')
    LOGGER.info('Carbon Saved: ' + str(carbon_conserved_sum))
    LOGGER.info('Carbon change in BAU: ' + str(np.nansum(carbon_change_bau)))
    LOGGER.info('Carbon change in Selective: ' + str(np.nansum(carbon_change_selective)))


def produce_ag_tradeoffs_publication_figures(args):
    # Set general figure parameters
    title = None
    show_lat_lon = False #if True, show best 5 graticules.
    show_when_generated = True
    resolution = 'i' #c (coarse), l, i, h
    center_cbar_at = 0
    num_cbar_ticks = 5
    no_data_value = -255
    crop_inf = True
    reverse_colorbar = False
    max_array_size = 5000000
    insert_white_divergence_point = False

    if args['calculate_bau_and_selective']:
        proportion_conserved = hb.as_array(os.path.join(args['run_folder'],'proportion_conserved.tif'))
        carbon_conserved = hb.as_array(os.path.join(args['run_folder'],'carbon_conserved.tif'))
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'],'proportion_cultivated.tif'))
        proportion_cultivated_bau = hb.as_array(os.path.join(args['run_folder'],'proportion_cultivated_bau.tif'))
        proportion_cultivated_selective = hb.as_array(os.path.join(args['run_folder'],'proportion_cultivated_selective.tif'))
    else:
        proportion_conserved = hb.as_array(os.path.join(args['intermediate_folder'],'proportion_conserved.tif'))
        carbon_conserved = hb.as_array(os.path.join(args['intermediate_folder'],'carbon_conserved.tif'))
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'],'proportion_cultivated.tif'))
        proportion_cultivated_bau = hb.as_array(os.path.join(args['intermediate_folder'],'proportion_cultivated_bau.tif'))
        proportion_cultivated_selective = hb.as_array(os.path.join(args['intermediate_folder'],'proportion_cultivated_selective.tif'))




    # Set figure-specific parameters and run it
    # Figure 4
    input_array = proportion_conserved
    output_uri = os.path.join(args['run_folder'], 'figure_4a_global_proportion_conserved.png')
    use_basemap = True
    bounding_box = 'clip_poles'
    cbar_label = 'Proportion conserved (proportion of grid-cell left natural in carbon-preserving scenario \nbut cultivated in unselective scenario. Here and elsewhere, zero and no-data plotted white)'
    vmin = 0.0
    vmax = 0.08
    color_scheme = 'spectral_contrast' #spectral, spectral_contrast, bw, prbg
    hb.show(input_array, output_uri=output_uri, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    input_array = proportion_conserved * -1.0
    vmin = 0.0
    vmax = 0.24
    bounding_box = 'clip_poles'
    cbar_label = 'Proportion lost (proportion of grid-cell cultivated in carbon-preserving \nscenario but not cultivated in unselective scenario)'
    output_uri = os.path.join(args['run_folder'], 'figure_4b_global_proportion_lost.png')
    hb.show(input_array, output_uri=output_uri, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    input_array = proportion_conserved
    vmin = -.24
    vmax = 0.08
    color_scheme = 'spectral_contrast'
    cbar_label = 'Proportion difference (proportion of grid-cell left natural in carbon-preserving \nscenario but cultivated in unselective scenario)'
    output_uri = os.path.join(args['run_folder'], 'figure_4c_global_proportion_difference.png')
    hb.show(input_array, output_uri=output_uri, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    # Figure 5
    max_array_size = 10000000
    show_state_boundaries = True
    input_array = proportion_conserved
    output_uri = os.path.join(args['run_folder'], 'figure_5a_us_midwest_proportion_conserved.png')
    use_basemap = True
    cbar_label = 'Proportion conserved (proportion of grid-cell left natural in carbon-preserving scenario \nbut cultivated in unselective scenario)'
    bounding_box = 'us_midwest'
    vmin = 0.0
    vmax = 0.08
    color_scheme = 'spectral_contrast' #spectral, spectral_contrast, bw, prbg
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, show_state_boundaries=show_state_boundaries, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    input_array = proportion_conserved * -1.0
    vmin = 0.0
    vmax = 0.24
    cbar_label = 'Proportion lost (proportion of grid-cell cultivated in carbon-preserving \nscenario but not cultivated in unselective scenario)'
    output_uri = os.path.join(args['run_folder'], 'figure_5b_us_midwest_proportion_lost.png')
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, show_state_boundaries=show_state_boundaries, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)



    input_array = proportion_conserved
    output_uri = os.path.join(args['run_folder'], 'figure_5c_se_asia_proportion_conserved.png')
    cbar_label = 'Proportion conserved (proportion of grid-cell left natural in carbon-preserving \nscenario but cultivated in unselective scenario)'
    bounding_box = 'se_asia'
    vmin = 0.0
    vmax = 0.08
    color_scheme = 'spectral_contrast' #spectral, spectral_contrast, bw, prbg
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    input_array = proportion_conserved * -1.0
    vmin = 0.0
    vmax = 0.24
    cbar_label = 'Proportion lost (proportion of grid-cell cultivated in carbon-preserving \nscenario but not cultivated in unselective scenario)'
    output_uri = os.path.join(args['run_folder'], 'figure_5d_se_asia_proportion_lost.png')
    hb.show(input_array, output_uri=output_uri, vertical_shift=.1, fig_height = 6, fig_width=8, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    input_array = proportion_conserved
    vmin = -.24
    vmax = 0.08
    bounding_box = 'us_midwest'
    color_scheme = 'spectral_contrast'
    cbar_label = 'Proportion difference (proportion of grid-cell left natural in carbon-preserving \nscenario but cultivated in unselective scenario)'
    output_uri = os.path.join(args['run_folder'], 'figure_5e_us_midwest_proportion_difference.png')
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    input_array = proportion_conserved
    vmin = -.24
    vmax = 0.08
    bounding_box = 'se_asia'
    color_scheme = 'spectral_contrast'
    cbar_label = 'Proportion difference (proportion of grid-cell left natural in carbon-preserving \nscenario but cultivated in unselective scenario)'
    output_uri = os.path.join(args['run_folder'], 'figure_5f_se_asia_proportion_difference.png')
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)


    # Figure 6
    max_array_size = 10000000
    show_state_boundaries = True
    input_array = carbon_conserved
    output_uri = os.path.join(args['run_folder'], 'figure_6a_us_midwest_carbon_conserved.png')
    use_basemap = True
    cbar_label = 'Carbon storage conserved (metric tonnes of carbon lost in unselective \nscenario but not lost in carbon-preserving scenario)'
    bounding_box = 'us_midwest'
    vmin = 0
    vmax = 120000
    color_scheme = 'spectral_contrast' #spectral, spectral_contrast, bw, prbg
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, show_state_boundaries=show_state_boundaries, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    input_array = carbon_conserved * -1.0
    cbar_label = 'Carbon storage lost (metric tonnes of carbon lost in carbon-preserving \nscenario but not lost in unselective scenario)'
    vmin = 0
    vmax = 120000
    output_uri = os.path.join(args['run_folder'], 'figure_6b_us_midwest_carbon_lost.png')
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, show_state_boundaries=show_state_boundaries, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)


    input_array = carbon_conserved
    output_uri = os.path.join(args['run_folder'], 'figure_6c_se_asia_carbon_conserved.png')
    bounding_box = 'se_asia'
    cbar_label = 'Carbon storage conserved (metric tonnes of carbon lost in unselective scenario but\nnot lost in carbon-preserving scenario)'
    vmin = 0
    vmax = 120000
    color_scheme = 'spectral_contrast' #spectral, spectral_contrast, bw, prbg
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    input_array = carbon_conserved * -1.0
    cbar_label = 'Carbon storage lost (metric tonnes of carbon lost in carbon-preserving scenario but\nnot lost in unselective scenario)'
    vmin = 0
    vmax = 120000
    output_uri = os.path.join(args['run_folder'], 'figure_6d_se_asia_carbon_lost.png')
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    input_array = carbon_conserved
    vmin = -120000
    vmax = 120000
    bounding_box = 'us_midwest'
    color_scheme = 'spectral_contrast'
    cbar_label = 'Carbon storage difference (carbon-preserving scenario minus unselective)'
    output_uri = os.path.join(args['run_folder'], 'figure_6e_us_midwest_carbon_difference.png')
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    input_array = carbon_conserved
    bounding_box = 'se_asia'
    color_scheme = 'spectral_contrast'
    cbar_label = 'Carbon storage difference (carbon-preserving scenario minus unselective)'
    output_uri = os.path.join(args['run_folder'], 'figure_6f_se_asia_carbon_difference.png')
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    # Figure 7
    input_array = proportion_cultivated
    vmin = 0.0
    vmax = 1.0
    bounding_box = 'se_asia'
    color_scheme = 'spectral_contrast'
    cbar_label = 'Current proportion cultivated'
    output_uri = os.path.join(args['run_folder'], 'figure_7a_current_proportion_cultivated.png')
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    input_array = proportion_cultivated_selective
    vmin = 0.0
    vmax = 1.0
    bounding_box = 'se_asia'
    color_scheme = 'spectral_contrast'
    cbar_label = 'Proportion cultivated in optimized scenario'
    output_uri = os.path.join(args['run_folder'], 'figure_7b_optimal_proportion_cultivated.png')
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)


    input_array = proportion_cultivated_bau
    vmin = 0.0
    vmax = 1.0
    bounding_box = 'se_asia'
    color_scheme = 'spectral_contrast'
    cbar_label = 'Proportion cultivated in business as usual (BAU) scenario'
    output_uri = os.path.join(args['run_folder'], 'figure_7c_bau_proportion_cultivated.png')
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)


    input_array = proportion_conserved
    vmin = -.24
    vmax = 0.08
    bounding_box = 'se_asia'
    color_scheme = 'spectral_contrast'
    cbar_label = 'Proportion conserved (difference between optimal and BAU)'
    output_uri = os.path.join(args['run_folder'], 'figure_7d_proportion_conserved.png')
    hb.show(input_array, output_uri=output_uri, vertical_shift = .1, fig_height = 6, fig_width=8, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

def create_base_ffn_inputs(args):
    LOGGER.info('Creating base FFN inputs.')
    ha_per_cell = hb.as_array(os.path.join(args['base_data_dir'], 'pyramids', 'ha_per_cell_300sec.tif'))
    proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))



    if args['calculate_yield_tons_per_cell']:
        yield_tons_per_cell = hb.as_array(os.path.join(args['input_folder'], 'yield_tons_per_cell.tif'))
    else:
        yield_tons_per_cell = hb.as_array(os.path.join(args['input_folder'], 'yield_tons_per_cell.tif'))

    if args['calculate_calories_per_cell']:
        calories_per_cell = hb.as_array(os.path.join(args['input_folder'], 'calories_per_cell.tif'))
        calories_per_ha_weighted_by_present_mix = hb.as_array(os.path.join(args['input_folder'], 'calories_per_ha_weighted_by_present_mix.tif'))
    else:
        calories_per_cell = hb.as_array(os.path.join(args['input_folder'], 'calories_per_cell.tif'))
        calories_per_ha_weighted_by_present_mix = hb.as_array(os.path.join(args['input_folder'], 'calories_per_ha_weighted_by_present_mix.tif'))

    if args['calculate_bau_and_selective']:
        proportion_conserved = hb.as_array(os.path.join(args['run_folder'], 'proportion_conserved.tif'))
        carbon_conserved = hb.as_array(os.path.join(args['run_folder'], 'carbon_conserved.tif'))
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        proportion_cultivated_bau = hb.as_array(os.path.join(args['run_folder'], 'proportion_cultivated_bau.tif'))
        proportion_cultivated_selective = hb.as_array(os.path.join(args['run_folder'], 'proportion_cultivated_selective.tif'))
    else:
        proportion_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_conserved.tif'))
        carbon_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'carbon_conserved.tif'))
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        proportion_cultivated_bau = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_cultivated_bau.tif'))
        proportion_cultivated_selective = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_cultivated_selective.tif'))

    ha_cultivated = proportion_cultivated * ha_per_cell
    ha_cultivated_bau = proportion_cultivated_bau * ha_per_cell
    ha_cultivated_selective = proportion_cultivated_selective * ha_per_cell
    desired_ha_conserved = ha_cultivated_bau - ha_cultivated_selective
    hb.save_array_as_geotiff(desired_ha_conserved, os.path.join(args['run_folder'], 'desired_ha_conserved.tif'), args['match_uri'], data_type=6,  ndv = -255)

    additional_carbon_value_available_per_grid = (carbon_conserved * args['social_cost_of_carbon']) #Mistake here: this is not a legit definition to devide it by ha and then later devide it by ha conserved. need to stick to one or the other when talking about a subset of ha in a gridcell #note that I switch back to positive payments
    hb.save_array_as_geotiff(additional_carbon_value_available_per_grid, os.path.join(args['run_folder'], 'additional_carbon_value_available_per_grid.tif'), args['match_uri'], data_type=6,  ndv = -255)
    additional_carbon_value_available_per_ha_conserved = additional_carbon_value_available_per_grid / desired_ha_conserved
    hb.save_array_as_geotiff(additional_carbon_value_available_per_ha_conserved, os.path.join(args['run_folder'], 'additional_carbon_value_available_per_ha_conserved.tif'), args['match_uri'], data_type=6,  ndv = -255)

    calories_forgone_per_grid  = (calories_per_cell / ha_per_cell) * desired_ha_conserved
    hb.save_array_as_geotiff(calories_forgone_per_grid, os.path.join(args['run_folder'], 'calories_forgone_per_grid.tif'), args['match_uri'], data_type=6,  ndv = -255)

    calories_forgone_per_ha = calories_forgone_per_grid / ha_per_cell
    calories_forgone_per_ha_conserved = np.where(desired_ha_conserved != 0, calories_forgone_per_grid / desired_ha_conserved,0.0)
    hb.save_array_as_geotiff(calories_forgone_per_ha_conserved, os.path.join(args['run_folder'], 'calories_forgone_per_ha_conserved.tif'), args['match_uri'], data_type=6,  ndv = -255)

    ha_per_cell = hb.as_array(os.path.join(args['base_data_dir'], 'pyramids', 'ha_per_cell_300sec.tif'))

    # Calculate ag value proxy
    mean_calories_per_ha = np.nansum(calories_per_cell)/np.nansum(ha_cultivated)
    mean_calories_per_ton = np.nansum(calories_per_cell) / np.nansum(yield_tons_per_cell)
    price_per_ton_corn = 45.9296 * 4.65
    value_per_calorie = price_per_ton_corn / mean_calories_per_ton
    value_per_cell_proxy = value_per_calorie * calories_per_cell
    value_per_ha_proxy = value_per_cell_proxy / ha_per_cell
    value_per_ha_cultivated_proxy = value_per_ha_proxy * (1 / proportion_cultivated)  * args['npv_adjustment']
    profit_forgone_per_ha_conserved_proxy = value_per_ha_cultivated_proxy
    hb.save_array_as_geotiff(profit_forgone_per_ha_conserved_proxy, os.path.join(args['run_folder'], 'profit_forgone_per_ha_conserved_proxy.tif'), args['match_uri'], data_type=6,  ndv=-255)

    # Calculate real ag value
    value_per_cell = hb.as_array('c:/Files/research/base_data/crops/ag_value_2000.tif')
    value_per_ha = value_per_cell / ha_per_cell
    value_per_cultivated_ha = value_per_ha * (1 / proportion_cultivated)  * args['npv_adjustment']
    profit_forgone_per_ha_conserved = value_per_cultivated_ha
    hb.save_array_as_geotiff(profit_forgone_per_ha_conserved, os.path.join(args['run_folder'], 'profit_forgone_per_ha_conserved.tif'), args['match_uri'], data_type=6,  ndv = -255)

    return

def calculate_ffn_b_payments(args):
    print('dir3', args['intermediate_folder'])

    if args['calculate_yield_tons_per_cell']:
        yield_tons_per_cell = hb.as_array(os.path.join(args['input_folder'], 'yield_tons_per_cell.tif'))
    else:
        yield_tons_per_cell = hb.as_array(os.path.join(args['input_folder'], 'yield_tons_per_cell.tif'))

    if args['calculate_calories_per_cell']:
        calories_per_cell = hb.as_array(os.path.join(args['input_folder'], 'calories_per_cell.tif'))
        calories_per_ha_weighted_by_present_mix = hb.as_array(os.path.join(args['input_folder'], 'calories_per_ha_weighted_by_present_mix.tif'))
    else:
        calories_per_cell = hb.as_array(os.path.join(args['input_folder'], 'calories_per_cell.tif'))
        calories_per_ha_weighted_by_present_mix = hb.as_array(os.path.join(args['input_folder'], 'calories_per_ha_weighted_by_present_mix.tif'))

    # if args['calculate_bau_and_selective']:
    #     carbon_conserved = hb.as_array(os.path.join(args['run_folder'], 'carbon_conserved.tif'))
    #     proportion_cultivated_bau = hb.as_array(os.path.join(args['run_folder'], 'carbon_conserved.tif'))
    #     proportion_cultivated_selective = hb.as_array(os.path.join(args['run_folder'], 'proportion_cultivated_selective.tif'))
    #     proportion_conserved = hb.as_array(os.path.join(args['run_folder'], 'proportion_conserved.tif'))
    #
    # else:
    #     carbon_conserved = hb.as_array(os.path.join(args['input_folder'], 'carbon_conserved.tif'))
    #     proportion_cultivated_bau = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated_bau.tif'))
    #     proportion_cultivated_selective = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated_selective.tif'))
    #     proportion_conserved = hb.as_array(os.path.join(args['input_folder'], 'proportion_conserved.tif'))


    if args['calculate_bau_and_selective']:
        proportion_conserved = hb.as_array(os.path.join(args['run_folder'], 'proportion_conserved.tif'))
        carbon_conserved = hb.as_array(os.path.join(args['run_folder'], 'carbon_conserved.tif'))
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        proportion_cultivated_bau = hb.as_array(os.path.join(args['run_folder'], 'proportion_cultivated_bau.tif'))
        proportion_cultivated_selective = hb.as_array(os.path.join(args['run_folder'], 'proportion_cultivated_selective.tif'))
    else:
        proportion_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_conserved.tif'))
        carbon_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'carbon_conserved.tif'))
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        proportion_cultivated_bau = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_cultivated_bau.tif'))
        proportion_cultivated_selective = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_cultivated_selective.tif'))


    if args['create_base_ffn_inputs']:
        additional_carbon_value_available_per_grid = hb.as_array(os.path.join(args['run_folder'], 'additional_carbon_value_available_per_grid.tif'))
        desired_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'desired_ha_conserved.tif'))
        additional_carbon_value_available_per_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'additional_carbon_value_available_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'profit_forgone_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved_proxy = hb.as_array(os.path.join(args['run_folder'], 'profit_forgone_per_ha_conserved_proxy.tif'))

    else:
        additional_carbon_value_available_per_grid = hb.as_array(os.path.join(args['intermediate_folder'], 'additional_carbon_value_available_per_grid.tif'))
        desired_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'desired_ha_conserved.tif'))
        additional_carbon_value_available_per_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'additional_carbon_value_available_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'profit_forgone_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved_proxy = hb.as_array(os.path.join(args['intermediate_folder'], 'profit_forgone_per_ha_conserved_proxy.tif'))

    analyze_nmb_methods = False # see C:\Files/research\ag_tradeoffs\food_for_nature_payments\output\does logging nmb affect order - yes
    if analyze_nmb_methods:
        nmb_denominator = np.log10(np.where(profit_forgone_per_ha_conserved > 0, profit_forgone_per_ha_conserved + 1.0, 1.0)) + 1.0
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
        hb.show(nmb, title='non proxy, logged', cbar_percentiles=[5,50,95])

        unraveled_keys = np.unravel_index(nmb.argsort(axis=None), shape=nmb.shape)  # returns TWO arrays in a tuple of all the rows, then all the cols keys
        nmb_sorted_keys = (unraveled_keys[0][::-1], unraveled_keys[1][::-1])  # thus here we need to reverse each seperately
        order_of_payments = np.zeros(nmb.shape)
        for i in range(len(nmb_sorted_keys[0])):
            current_key = (nmb_sorted_keys[0][i], nmb_sorted_keys[1][i])
            if nmb[current_key] <= 0:
                break
            order_of_payments[current_key] = i + 1
        hb.show(order_of_payments, title='order_of_payments non proxy, logged')

        nmb_denominator = np.where(profit_forgone_per_ha_conserved > 0, profit_forgone_per_ha_conserved + 1.0, 1.0)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
        hb.show(nmb, title='non proxy, not logged', cbar_percentiles=[5,50,95])

        unraveled_keys = np.unravel_index(nmb.argsort(axis=None), shape=nmb.shape)  # returns TWO arrays in a tuple of all the rows, then all the cols keys
        nmb_sorted_keys = (unraveled_keys[0][::-1], unraveled_keys[1][::-1])  # thus here we need to reverse each seperately
        order_of_payments = np.zeros(nmb.shape)
        for i in range(len(nmb_sorted_keys[0])):
            current_key = (nmb_sorted_keys[0][i], nmb_sorted_keys[1][i])
            if nmb[current_key] <= 0:
                break
            order_of_payments[current_key] = i + 1
        hb.show(order_of_payments, title='order_of_payments non proxy, not logged')

        nmb_denominator = np.log10(np.where(profit_forgone_per_ha_conserved_proxy > 0, profit_forgone_per_ha_conserved_proxy + 1.0, 1.0)) + 1.0
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
        hb.show(nmb, title='proxy, logged', cbar_percentiles=[5,50,95])

        unraveled_keys = np.unravel_index(nmb.argsort(axis=None), shape=nmb.shape)  # returns TWO arrays in a tuple of all the rows, then all the cols keys
        nmb_sorted_keys = (unraveled_keys[0][::-1], unraveled_keys[1][::-1])  # thus here we need to reverse each seperately
        order_of_payments = np.zeros(nmb.shape)
        for i in range(len(nmb_sorted_keys[0])):
            current_key = (nmb_sorted_keys[0][i], nmb_sorted_keys[1][i])
            if nmb[current_key] <= 0:
                break
            order_of_payments[current_key] = i + 1
        hb.show(order_of_payments, title='order_of_payments proxy, logged')

        nmb_denominator = np.where(profit_forgone_per_ha_conserved_proxy > 0, profit_forgone_per_ha_conserved_proxy + 1.0, 1.0)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
        hb.show(nmb, title='proxy, not logged', cbar_percentiles=[5,50,95])

        unraveled_keys = np.unravel_index(nmb.argsort(axis=None), shape=nmb.shape)  # returns TWO arrays in a tuple of all the rows, then all the cols keys
        nmb_sorted_keys = (unraveled_keys[0][::-1], unraveled_keys[1][::-1])  # thus here we need to reverse each seperately
        order_of_payments = np.zeros(nmb.shape)
        for i in range(len(nmb_sorted_keys[0])):
            current_key = (nmb_sorted_keys[0][i], nmb_sorted_keys[1][i])
            if nmb[current_key] <= 0:
                break
            order_of_payments[current_key] = i + 1
        hb.show(order_of_payments, title='order_of_payments proxy, not logged')


    if args['n_method'] == 'ones':
        # Just equal to 1
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        nmb_denominator = np.ones(proportion_cultivated.shape)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
    elif args['n_method'] == 'proportion_cultivated':
       # Based on prop cultivated
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        nmb_denominator = np.where(proportion_cultivated > 0, proportion_cultivated, -255)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
    elif args['n_method'] == 'profit_proxy':
        # Based on dollars per calorie using corn price
        nmb_denominator = np.where(profit_forgone_per_ha_conserved_proxy > 0, profit_forgone_per_ha_conserved_proxy, 0.0)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
    elif args['n_method'] == 'profit':
        # Based on dollars per calorie using corn price
        nmb_denominator = np.where(profit_forgone_per_ha_conserved > 0, profit_forgone_per_ha_conserved, 0.0)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)


    hb.save_array_as_geotiff(nmb, os.path.join(args['run_folder'], 'nmb_b.tif'), args['match_uri'], data_type=6,  ndv = -255)

    unraveled_keys = np.unravel_index(nmb.argsort(axis=None), shape=nmb.shape) #returns TWO arrays in a tuple of all the rows, then all the cols keys
    nmb_sorted_keys = (unraveled_keys[0][::-1], unraveled_keys[1][::-1]) # thus here we need to reverse each seperately

    order_of_payments = np.zeros(nmb.shape)
    for i in range(len(nmb_sorted_keys[0])):
        current_key = (nmb_sorted_keys[0][i], nmb_sorted_keys[1][i])
        if nmb[current_key] <= 0:
            break
        order_of_payments[current_key] = i + 1
    hb.save_array_as_geotiff(order_of_payments, os.path.join(args['run_folder'], 'order_of_payments_b.tif'), args['match_uri'], data_type=6,  ndv = -255)

    order_of_payments = None

    value_protected_given_budget_per_grid = np.zeros(nmb.shape)
    order_of_payments_given_budget = np.zeros(nmb.shape)
    payment_set_given_budget_per_ha = np.zeros(nmb.shape)
    sum_spent = 0
    num_rejected_payments = 0

    for i in range(len(nmb_sorted_keys[0])):
        current_key = (nmb_sorted_keys[0][i], nmb_sorted_keys[1][i])
        if nmb[current_key] <= 0:
            LOGGER.warn('N = 0 in all remaining cells. (You have run out of locations that are worth protecting')
            break
        if additional_carbon_value_available_per_ha_conserved[current_key] < profit_forgone_per_ha_conserved[current_key]:
            num_rejected_payments += 1
        else:
            value_in_grid = additional_carbon_value_available_per_grid[current_key]# * desired_ha_conserved[current_key] #I made a mistake here by reducing the additional carboni value further by the desired ha conserved, double reducing it.
            sum_spent += value_in_grid
            if sum_spent > args['conservation_budget']:
                LOGGER.debug('All budget spent. BREAK.')
                break
            value_protected_given_budget_per_grid[current_key] = value_in_grid
            order_of_payments_given_budget[current_key] = i + 1

            payment_set_given_budget_per_ha[current_key] = additional_carbon_value_available_per_ha_conserved[current_key]


    LOGGER.info('   FFN based on SMV (social marginal value) solved.')
    LOGGER.info('   Carbon value Saved,' + str(np.nansum(value_protected_given_budget_per_grid)))
    LOGGER.info('   Money Spent,' + str(sum_spent))
    LOGGER.info('   num_rejected_payments,' + str(num_rejected_payments))

    hb.save_array_as_geotiff(payment_set_given_budget_per_ha, os.path.join(args['run_folder'], 'payment_set_b_per_ha.tif'), args['match_uri'], data_type=6,  ndv = -255)
    hb.save_array_as_geotiff(order_of_payments_given_budget, os.path.join(args['run_folder'], 'order_of_payments_given_budget_b.tif'), args['match_uri'], data_type=6,  ndv = -255)
    hb.save_array_as_geotiff(value_protected_given_budget_per_grid, os.path.join(args['run_folder'], 'value_protected_given_budget_per_grid_b.tif'), args['match_uri'], data_type=6,  ndv = -255)

    return


def calculate_ffn_c_payments(args):
    if args['calculate_yield_tons_per_cell']:
        yield_tons_per_cell = hb.as_array(os.path.join(args['input_folder'], 'yield_tons_per_cell.tif'))
    else:
        yield_tons_per_cell = hb.as_array(os.path.join(args['input_folder'], 'yield_tons_per_cell.tif'))

    if args['calculate_calories_per_cell']:
        calories_per_cell = hb.as_array(os.path.join(args['input_folder'], 'calories_per_cell.tif'))
        calories_per_ha_weighted_by_present_mix = hb.as_array(os.path.join(args['input_folder'], 'calories_per_ha_weighted_by_present_mix.tif'))
    else:
        calories_per_cell = hb.as_array(os.path.join(args['input_folder'], 'calories_per_cell.tif'))
        calories_per_ha_weighted_by_present_mix = hb.as_array(os.path.join(args['input_folder'], 'calories_per_ha_weighted_by_present_mix.tif'))

    if args['calculate_bau_and_selective']:
        proportion_conserved = hb.as_array(os.path.join(args['run_folder'], 'proportion_conserved.tif'))
        carbon_conserved = hb.as_array(os.path.join(args['run_folder'], 'carbon_conserved.tif'))
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        proportion_cultivated_bau = hb.as_array(os.path.join(args['run_folder'], 'proportion_cultivated_bau.tif'))
        proportion_cultivated_selective = hb.as_array(os.path.join(args['run_folder'], 'proportion_cultivated_selective.tif'))
    else:
        proportion_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_conserved.tif'))
        carbon_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'carbon_conserved.tif'))
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        proportion_cultivated_bau = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_cultivated_bau.tif'))
        proportion_cultivated_selective = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_cultivated_selective.tif'))


    if args['create_base_ffn_inputs']:
        additional_carbon_value_available_per_grid = hb.as_array(os.path.join(args['run_folder'], 'additional_carbon_value_available_per_grid.tif'))
        desired_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'desired_ha_conserved.tif'))
        additional_carbon_value_available_per_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'additional_carbon_value_available_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'profit_forgone_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved_proxy = hb.as_array(os.path.join(args['run_folder'], 'profit_forgone_per_ha_conserved_proxy.tif'))

    else:
        additional_carbon_value_available_per_grid = hb.as_array(os.path.join(args['intermediate_folder'], 'additional_carbon_value_available_per_grid.tif'))
        desired_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'desired_ha_conserved.tif'))
        additional_carbon_value_available_per_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'additional_carbon_value_available_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'profit_forgone_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved_proxy = hb.as_array(os.path.join(args['intermediate_folder'], 'profit_forgone_per_ha_conserved_proxy.tif'))

    if args['n_method'] == 'ones':
        # Just equal to 1
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        nmb_denominator = np.ones(proportion_cultivated.shape)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
    elif args['n_method'] == 'proportion_cultivated':
        # Based on prop cultivated
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        nmb_denominator = np.where(proportion_cultivated > 0, proportion_cultivated, -255)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
    elif args['n_method'] == 'profit_proxy':
        # Based on dollars per calorie using corn price
        nmb_denominator = np.where(profit_forgone_per_ha_conserved_proxy > 0, profit_forgone_per_ha_conserved_proxy, 0.0)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
    elif args['n_method'] == 'profit':
        # Based on dollars per calorie using corn price
        nmb_denominator = np.where(profit_forgone_per_ha_conserved > 0, profit_forgone_per_ha_conserved, 0.0)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)

    # hb.save_array_as_geotiff(nmb, os.path.join(args['run_folder'], 'nmb.tif'), args['match_uri'], data_type=6,  ndv=-255)
    hb.save_array_as_geotiff(nmb, os.path.join(args['run_folder'], 'nmb_c.tif'), args['match_uri'], data_type=6,  ndv = -255)

    unraveled_keys = np.unravel_index(nmb.argsort(axis=None), shape=nmb.shape) #returns TWO arrays in a tuple of all the rows, then all the cols keys
    nmb_sorted_keys = (unraveled_keys[0][::-1], unraveled_keys[1][::-1]) # thus here we need to reverse each seperately

    order_of_payments = np.zeros(nmb.shape)
    for i in range(len(nmb_sorted_keys[0])):
        current_key = (nmb_sorted_keys[0][i], nmb_sorted_keys[1][i])
        if nmb[current_key] <= 0:
            break
        order_of_payments[current_key] = i + 1
    hb.save_array_as_geotiff(order_of_payments, os.path.join(args['run_folder'], 'order_of_payments_c.tif'), args['match_uri'], data_type=6,  ndv = -255)

    order_of_payments = None

    value_protected_given_budget_per_grid = np.zeros(nmb.shape)
    order_of_payments_given_budget = np.zeros(nmb.shape)
    payment_set_given_budget_per_ha = np.zeros(nmb.shape)
    sum_spent = 0
    num_rejected_payments = 0

    for i in range(len(nmb_sorted_keys[0])):
        current_key = (nmb_sorted_keys[0][i], nmb_sorted_keys[1][i])
        if nmb[current_key] <= 0:
            LOGGER.warn('N = 0 in all remaining cells. (You have run out of locations that are worth protecting')
            break

        if additional_carbon_value_available_per_ha_conserved[current_key] < profit_forgone_per_ha_conserved[current_key]:
            num_rejected_payments += 1
        else:
            value_in_grid = additional_carbon_value_available_per_grid[current_key]# * desired_ha_conserved[current_key] #I made a mistake here by reducing the additional carboni value further by the desired ha conserved, double reducing it.
            # sum_spent += value_in_grid

            # KEY DIFFERENCE, here, we pay not the value in grid, but the Opportunity Cost.
            sum_spent += profit_forgone_per_ha_conserved[current_key] * desired_ha_conserved[current_key]

            if sum_spent > args['conservation_budget']:
                LOGGER.debug('All budget spent. BREAK.')
                break
            value_protected_given_budget_per_grid[current_key] = value_in_grid
            order_of_payments_given_budget[current_key] = i + 1

            # payment_set_given_budget_per_ha[current_key] = additional_carbon_value_available_per_ha_conserved[current_key]
            payment_set_given_budget_per_ha[current_key] = profit_forgone_per_ha_conserved[current_key]


    LOGGER.info('\nFFN base on paying OC (opportunity cost) solved.')
    LOGGER.info('Carbon value Saved,' + str(np.nansum(value_protected_given_budget_per_grid)))
    LOGGER.info('Money Spent,' + str(sum_spent))
    LOGGER.info('num_rejected_payments,' + str(num_rejected_payments))

    hb.save_array_as_geotiff(payment_set_given_budget_per_ha, os.path.join(args['run_folder'], 'payment_set_c_per_ha.tif'), args['match_uri'], data_type=6,  ndv = -255)
    hb.save_array_as_geotiff(order_of_payments_given_budget, os.path.join(args['run_folder'], 'order_of_payments_given_budget_c.tif'), args['match_uri'], data_type=6,  ndv = -255)
    hb.save_array_as_geotiff(value_protected_given_budget_per_grid, os.path.join(args['run_folder'], 'value_protected_given_budget_per_grid_c.tif'), args['match_uri'], data_type=6,  ndv = -255)

    return


def calculate_ffn_f_payments(args):
    # BUG I don't know why, but this only gets the correct values if run in isolation. Otherwise, it uses the profits map that isn't the proxy.

    # Load inputs
    ha_per_cell = hb.as_array(os.path.join(args['base_data_dir'], 'pyramids', 'ha_per_cell_300sec.tif'))
    proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
    # if args['calculate_yield_tons_per_cell']:
    #     #yield_tons_per_cell = hb.as_array(os.path.join(args['run_folder'], 'yield_tons_per_cell.tif'))
    # else:
    #     #yield_tons_per_cell = hb.as_array(os.path.join(args['input_folder'], 'yield_tons_per_cell.tif'))
    # if args['calculate_calories_per_cell']:
    #     #calories_per_cell = hb.as_array(os.path.join(args['run_folder'], 'calories_per_cell.tif'))
    #     calories_per_ha_weighted_by_present_mix = hb.as_array(os.path.join(args['run_folder'], 'calories_per_ha_weighted_by_present_mix.tif'))
    # else:
    #     #calories_per_cell = hb.as_array(os.path.join(args['input_folder'], 'calories_per_cell.tif'))
    #     calories_per_ha_weighted_by_present_mix = hb.as_array(os.path.join(args['input_folder'], 'calories_per_ha_weighted_by_present_mix.tif'))
    # if args['calculate_bau_and_selective']:
    #     carbon_conserved = hb.as_array(os.path.join(args['run_folder'], 'carbon_conserved.tif'))
    #     proportion_cultivated_bau = hb.as_array(os.path.join(args['run_folder'], 'carbon_conserved.tif'))
    #     proportion_cultivated_selective = hb.as_array(os.path.join(args['run_folder'], 'proportion_cultivated_selective.tif'))
    #     proportion_conserved = hb.as_array(os.path.join(args['run_folder'], 'proportion_conserved.tif'))
    # else:
    #     carbon_conserved = hb.as_array(os.path.join(args['input_folder'], 'carbon_conserved.tif'))
    #     proportion_cultivated_bau = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated_bau.tif'))
    #     proportion_cultivated_selective = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated_selective.tif'))
    #     proportion_conserved = hb.as_array(os.path.join(args['input_folder'], 'proportion_conserved.tif'))

    if args['calculate_bau_and_selective']:
        proportion_conserved = hb.as_array(os.path.join(args['run_folder'], 'proportion_conserved.tif'))
        carbon_conserved = hb.as_array(os.path.join(args['run_folder'], 'carbon_conserved.tif'))
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        proportion_cultivated_bau = hb.as_array(os.path.join(args['run_folder'], 'proportion_cultivated_bau.tif'))
        proportion_cultivated_selective = hb.as_array(os.path.join(args['run_folder'], 'proportion_cultivated_selective.tif'))
    else:
        proportion_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_conserved.tif'))
        carbon_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'carbon_conserved.tif'))
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        proportion_cultivated_bau = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_cultivated_bau.tif'))
        proportion_cultivated_selective = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_cultivated_selective.tif'))

    if args['create_base_ffn_inputs']:
        additional_carbon_value_available_per_grid = hb.as_array(os.path.join(args['run_folder'], 'additional_carbon_value_available_per_grid.tif'))
        desired_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'desired_ha_conserved.tif'))
        additional_carbon_value_available_per_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'additional_carbon_value_available_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'profit_forgone_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved_proxy = hb.as_array(os.path.join(args['run_folder'], 'profit_forgone_per_ha_conserved_proxy.tif'))
    else:
        additional_carbon_value_available_per_grid = hb.as_array(os.path.join(args['intermediate_folder'], 'additional_carbon_value_available_per_grid.tif'))
        desired_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'desired_ha_conserved.tif'))
        additional_carbon_value_available_per_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'additional_carbon_value_available_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'profit_forgone_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved_proxy = hb.as_array(os.path.join(args['intermediate_folder'], 'profit_forgone_per_ha_conserved_proxy.tif'))


    if args['n_method'] == 'ones':
        # Just equal to 1
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        nmb_denominator = np.ones(proportion_cultivated.shape)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
    elif args['n_method'] == 'proportion_cultivated':
        # Based on prop cultivated
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        nmb_denominator = np.where(proportion_cultivated > 0, proportion_cultivated, -255)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
    elif args['n_method'] == 'profit_proxy':
        # Based on dollars per calorie using corn price
        nmb_denominator = np.where(profit_forgone_per_ha_conserved_proxy > 0, profit_forgone_per_ha_conserved_proxy, 0.0)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
    elif args['n_method'] == 'profit':
        # Based on dollars per calorie using corn price
        nmb_denominator = np.where(profit_forgone_per_ha_conserved > 0, profit_forgone_per_ha_conserved, 0.0)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)

    # nmb_denominator = np.log10(np.where(profit_forgone_per_ha_conserved_proxy > 0, profit_forgone_per_ha_conserved_proxy + 1.0, 1.0)) + 1.0
    # hb.save_array_as_geotiff(nmb_denominator, os.path.join(args['run_folder'], 'nmb_denominator.tif'), args['match_uri'], data_type=6,  ndv = -255)
    # nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)



    hb.save_array_as_geotiff(nmb, os.path.join(args['run_folder'], 'nmb_f_payments.tif'), args['match_uri'], data_type=6,  ndv=-255)

    unraveled_keys = np.unravel_index(nmb.argsort(axis=None), shape=nmb.shape) #returns TWO arrays in a tuple of all the rows, then all the cols keys
    nmb_sorted_keys = (unraveled_keys[0][::-1], unraveled_keys[1][::-1]) # thus here we need to reverse each seperately

    unraveled_keys = None

    def calc_best_reduction_factor_given_range(min, max, step):

        # Record the alpha-protected value combinations for later plotting
        value_obtained_points = OrderedDict()
        value_missed_points = OrderedDict()

        if max > 1.0:
            max = 1.0
            LOGGER.warn('Attempted to test a reduction factor greater than 1. Why are you asking this?')

        if min < 0.0:
            min = 0.0
            LOGGER.warn('Attempted to test a reduction factor less than 0. Why are you asking this?')

        highest_value_yet = 0
        best_reduction_factor = 0
        for reduction_factor in np.arange(min, max+step, step):
            payment_offer_f_per_ha = reduction_factor * additional_carbon_value_available_per_ha_conserved
            sum_spent = 0
            value_obtained = 0
            value_missed = 0
            num_rejected_payments = 0
            for i in range(len(nmb_sorted_keys[0])):
                current_key = (nmb_sorted_keys[0][i], nmb_sorted_keys[1][i])
                if nmb[current_key] <= 0:
                    LOGGER.warn('N = 1 in all remaining cells. (You have run out of locations that are worth protecting')
                    break

                value_in_grid = additional_carbon_value_available_per_grid[current_key]
                expenditure_in_grid = payment_offer_f_per_ha[current_key] * desired_ha_conserved[current_key]


                if payment_offer_f_per_ha[current_key] < profit_forgone_per_ha_conserved_proxy[current_key]:
                    num_rejected_payments += 1
                    if payment_offer_f_per_ha[current_key] > profit_forgone_per_ha_conserved[current_key]:
                        value_missed += value_in_grid


                else:
                    value_obtained += value_in_grid
                    sum_spent += expenditure_in_grid
                    if sum_spent > args['conservation_budget']:
                        break
                value_obtained_points[float(reduction_factor)] = float(value_obtained)
                value_missed_points[float(reduction_factor)] = float(value_missed)

            if value_obtained > highest_value_yet:
                highest_value_yet = value_obtained
                best_reduction_factor = reduction_factor
            LOGGER.info('Testing of reduction factor ' + str(reduction_factor) + ' resulted in conservation value of ' + str(value_obtained) + ' at expense of ' + str(sum_spent))
        return best_reduction_factor, value_obtained_points, value_missed_points


    args['best_reduction_factor'] = 0.7

    if 'best_reduction_factor' in args:
        best_reduction_factor = args['best_reduction_factor']
        skip = True
    else:
        skip = False


    if not skip:
        fractional_reduction_to_value = OrderedDict()
        fractional_reduction_to_value_missed = OrderedDict()
        current_best, value_obtained_points, value_missed_points = calc_best_reduction_factor_given_range(0,1,.025)
        fractional_reduction_to_value.update(value_obtained_points)
        fractional_reduction_to_value_missed.update(value_missed_points)

        current_best, value_obtained_points, value_missed_points = calc_best_reduction_factor_given_range(current_best - .1, current_best + .1,.01)
        fractional_reduction_to_value.update(value_obtained_points)
        fractional_reduction_to_value_missed.update(value_missed_points)

        # current_best, value_obtained_points, value_missed_points = calc_best_reduction_factor_given_range(current_best - .01, current_best + .01,.001)
        # fractional_reduction_to_value.update(value_obtained_points)
        # fractional_reduction_to_value_missed.update(value_missed_points)
        #
        # best_reduction_factor, value_obtained_points, value_missed_points = calc_best_reduction_factor_given_range(current_best - .001, current_best + .001,.0001)
        # fractional_reduction_to_value.update(value_obtained_points)
        # fractional_reduction_to_value_missed.update(value_missed_points)

        best_reduction_factor = current_best
        args['best_reduction_factor'] = best_reduction_factor
        # Make a new sorted dict
        fractional_reduction_to_value = OrderedDict(sorted(fractional_reduction_to_value.items()))
        fractional_reduction_to_value_missed = OrderedDict(sorted(fractional_reduction_to_value_missed.items()))

        fractional_reduction_to_value_uri = os.path.join(args['run_folder'], 'fractional_reduction_to_value.csv')
        hb.python_object_to_csv(fractional_reduction_to_value, fractional_reduction_to_value_uri)

        fractional_reduction_to_value_missed_uri = os.path.join(args['run_folder'], 'fractional_reduction_to_value_missed.csv')
        hb.python_object_to_csv(fractional_reduction_to_value_missed, fractional_reduction_to_value_missed_uri)



        #best_reduction_factor, current_points = calc_best_reduction_factor_given_range(current_best - .0001, current_best + .0001,.00001)


    payment_offer_f_per_ha = best_reduction_factor * additional_carbon_value_available_per_ha_conserved

    #additional_carbon_value_available_per_ha_conserved = None

    sum_spent = 0
    value_obtained = 0
    num_rejected_payments = 0
    value_protected_f_per_grid = np.zeros(nmb.shape)
    expenditure_f_per_grid = np.zeros(nmb.shape)
    order_of_payments_f = np.zeros(nmb.shape)
    payment_set_f_per_ha = np.zeros(nmb.shape)
    for i in range(len(nmb_sorted_keys[0])):
        current_key = (nmb_sorted_keys[0][i], nmb_sorted_keys[1][i])
        if nmb[current_key] <= 0:
            LOGGER.warn('N = 1 in all remaining cells. (You have run out of locations that are worth protecting')
            break
        if payment_offer_f_per_ha[current_key] < profit_forgone_per_ha_conserved[current_key]:
            num_rejected_payments += 1
        else:
            value_in_grid = additional_carbon_value_available_per_grid[current_key]
            expenditure_in_grid = payment_offer_f_per_ha[current_key] * desired_ha_conserved[current_key]
            value_obtained += value_in_grid
            sum_spent += expenditure_in_grid
            if sum_spent > args['conservation_budget']:
                break
            value_protected_f_per_grid[current_key] = value_in_grid
            expenditure_f_per_grid[current_key] = expenditure_in_grid
            order_of_payments_f[current_key] = i + 1

            payment_set_f_per_ha[current_key] = payment_offer_f_per_ha[current_key]

    LOGGER.info('\nFFN with reduced price solved.')
    LOGGER.info('Best reduction factor was ' + str(best_reduction_factor))
    LOGGER.info('Carbon value Saved,' + str(np.nansum(value_protected_f_per_grid)))
    LOGGER.info('Money Spent,' + str(np.nansum(expenditure_f_per_grid)))
    LOGGER.info('num_rejected_payments,' + str(num_rejected_payments))

    hb.save_array_as_geotiff(payment_set_f_per_ha, os.path.join(args['run_folder'], 'payment_set_f_per_ha.tif'), args['match_uri'], data_type=6,  ndv = -255)
    hb.save_array_as_geotiff(order_of_payments_f, os.path.join(args['run_folder'], 'order_of_payments_f.tif'), args['match_uri'], data_type=6,  ndv = -255)
    # hb.save_array_as_geotiff(value_protected_f_per_grid, os.path.join(args['run_folder'], 'value_protected_f_per_grid.tif'), args['match_uri'], data_type=6,  ndv = -255)
    hb.save_array_as_geotiff(value_protected_f_per_grid, os.path.join(args['run_folder'], 'value_protected_given_budget_per_grid_f.tif'), args['match_uri'], data_type=6,  ndv = -255)
    hb.save_array_as_geotiff(expenditure_f_per_grid, os.path.join(args['run_folder'], 'expenditure_f_per_grid.tif'), args['match_uri'], data_type=6,  ndv = -255)

    hb.save_array_as_geotiff(payment_offer_f_per_ha, os.path.join(args['run_folder'], 'payment_offer_f_per_ha.tif'), args['match_uri'], data_type=6,  ndv = -255)

    return args


def calculate_ffn_a_payments(args):
    if args['create_base_ffn_inputs']:
        additional_carbon_value_available_per_grid = hb.as_array(os.path.join(args['run_folder'], 'additional_carbon_value_available_per_grid.tif'))
        desired_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'desired_ha_conserved.tif'))
        additional_carbon_value_available_per_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'additional_carbon_value_available_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'profit_forgone_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved_proxy = hb.as_array(os.path.join(args['run_folder'], 'profit_forgone_per_ha_conserved_proxy.tif'))
    else:
        additional_carbon_value_available_per_grid = hb.as_array(os.path.join(args['intermediate_folder'], 'additional_carbon_value_available_per_grid.tif'))
        desired_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'desired_ha_conserved.tif'))
        additional_carbon_value_available_per_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'additional_carbon_value_available_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'profit_forgone_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved_proxy = hb.as_array(os.path.join(args['intermediate_folder'], 'profit_forgone_per_ha_conserved_proxy.tif'))
    if args['calculate_calories_per_cell']:
        calories_per_cell = hb.as_array(os.path.join(args['input_folder'], 'calories_per_cell.tif'))
        calories_per_ha_weighted_by_present_mix = hb.as_array(os.path.join(args['input_folder'], 'calories_per_ha_weighted_by_present_mix.tif'))
    else:
        calories_per_cell = hb.as_array(os.path.join(args['input_folder'], 'calories_per_cell.tif'))
        calories_per_ha_weighted_by_present_mix = hb.as_array(os.path.join(args['input_folder'], 'calories_per_ha_weighted_by_present_mix.tif'))
    # if args['calculate_ffn_b_payments']:
    #     nmb = hb.as_array(os.path.join(args['run_folder'], 'nmb.tif'))
    # else:
    #     nmb = hb.as_array(os.path.join(args['intermediate_folder'], 'nmb.tif'))

    if args['calculate_ffn_f_payments']:
        payment_offer_f_per_ha = hb.as_array(os.path.join(args['run_folder'], 'payment_offer_f_per_ha.tif'))
    else:
        payment_offer_f_per_ha = hb.as_array(os.path.join(args['intermediate_folder'], 'payment_offer_f_per_ha.tif'))


    # To introduce some unknown information, I let the true value of the value forgone be defined as the mean with a deviation
    # equal to a truncated normal distribution bounded to be += 0.5 of the mean.
    profit_forgone_a_per_ha_conserved = np.where(profit_forgone_per_ha_conserved != 0, profit_forgone_per_ha_conserved * sp.stats.truncnorm.rvs(1. - args['asymmetric_scatter'], 1. + args['asymmetric_scatter'], size=profit_forgone_per_ha_conserved.shape),0)
    hb.save_array_as_geotiff(profit_forgone_a_per_ha_conserved, os.path.join(args['run_folder'], 'profit_forgone_a_per_ha_conserved.tif'), args['match_uri'], data_type=6,  ndv = -255)

    # nmb_a_denominator = np.log10(np.where(profit_forgone_a_per_ha_conserved > 0, profit_forgone_a_per_ha_conserved + 1.0, 1.0)) + 1.0
    nmb_a_denominator = np.where(profit_forgone_a_per_ha_conserved > 0, profit_forgone_a_per_ha_conserved + 1.0, 1.0)
    hb.save_array_as_geotiff(nmb_a_denominator, os.path.join(args['run_folder'], 'nmb_a_denominator.tif'), args['match_uri'], data_type=6,  ndv = -255)
    nmb_a = np.where(additional_carbon_value_available_per_ha_conserved > 0, additional_carbon_value_available_per_ha_conserved / nmb_a_denominator, 0)

    hb.save_array_as_geotiff(nmb_a, os.path.join(args['run_folder'], 'nmb_a.tif'), args['match_uri'], data_type=6,  ndv = -255)

    # unraveled_keys = np.unravel_index(nmb.argsort(axis=None), shape=nmb.shape) #returns TWO arrays in a tuple of all the rows, then all the cols keys
    # nmb_sorted_keys = (unraveled_keys[0][::-1], unraveled_keys[1][::-1]) # thus here we need to reverse each seperately

    unraveled_keys = np.unravel_index(nmb_a.argsort(axis=None), shape=nmb_a.shape) #returns TWO arrays in a tuple of all the rows, then all the cols keys
    nmb_a_sorted_keys = (unraveled_keys[0][::-1], unraveled_keys[1][::-1]) # thus here we need to reverse each seperately

    unraveled_keys = None

    value_protected_a_per_grid = np.zeros(nmb_a.shape)
    expenditure_a_per_grid = np.zeros(nmb_a.shape)
    order_of_payments_a = np.zeros(nmb_a.shape)
    payment_set_a_per_ha = np.zeros(nmb_a.shape)
    sum_spent = 0
    num_rejected_payments = 0
    for i in range(len(nmb_a_sorted_keys[0])):
        current_key = (nmb_a_sorted_keys[0][i], nmb_a_sorted_keys[1][i])
        # At first i thought this couldn't have a break here in a VCG, but really it's just that you can't have the price change. you can still have the
        # if False and payment_offer_f_per_ha[current_key] * desired_ha_conserved[current_key] > 0 and nmb[current_key] > 0:
        # if nmb_a[current_key] <= 0:
        #     LOGGER.warn('N = 0 in all remaining cells. (You have run out of locations that are worth protecting')
        #     break

        if nmb_a[current_key] <= 0:
            LOGGER.warn('N = 0 in all remaining cells. (You have run out of locations that are worth protecting')
            break
        # Note that the payment_offer must be calculated in the previous step and not based on bids.
        #if payment_offer_f_per_ha[current_key] < profit_forgone_a_per_ha[current_key]:
        if payment_offer_f_per_ha[current_key] < profit_forgone_a_per_ha_conserved[current_key]:
            # LOGGER.debug('Value of payment less than opportunity cost, and so rejected the bid')
            num_rejected_payments += 1
        else:
            value_in_grid = additional_carbon_value_available_per_grid[current_key]
            expenditure_in_grid = payment_offer_f_per_ha[current_key] * desired_ha_conserved[current_key]
            sum_spent += expenditure_in_grid

            if sum_spent > args['conservation_budget']:
                break
            value_protected_a_per_grid[current_key] = value_in_grid
            expenditure_a_per_grid[current_key] = expenditure_in_grid
            order_of_payments_a[current_key] = i + 1
            payment_set_a_per_ha[current_key] = payment_offer_f_per_ha[current_key]

    LOGGER.info('\nFFN with private_info information AND eduction in prices solved.')
    LOGGER.info('Carbon value Saved,' + str(np.nansum(value_protected_a_per_grid)))
    LOGGER.info('Money Spent,' + str(np.nansum(expenditure_a_per_grid)))
    LOGGER.info('num_rejected_payments,' + str(num_rejected_payments))

    hb.save_array_as_geotiff(payment_set_a_per_ha, os.path.join(args['run_folder'], 'payment_set_a_per_ha.tif'), args['match_uri'], data_type=6,  ndv = -255)
    hb.save_array_as_geotiff(order_of_payments_a, os.path.join(args['run_folder'], 'order_of_payments_a'), args['match_uri'], data_type=6,  ndv = -255)
    hb.save_array_as_geotiff(value_protected_a_per_grid, os.path.join(args['run_folder'], 'value_protected_given_budget_per_grid_a.tif'), args['match_uri'], data_type=6,  ndv = -255)
    hb.save_array_as_geotiff(expenditure_a_per_grid, os.path.join(args['run_folder'], 'expenditure_a_per_grid.tif'), args['match_uri'], data_type=6,  ndv = -255)

def calculate_ffn_f_country_level_payments(args):
    # BUG I don't know why, but this only gets the correct values if run in isolation. Otherwise, it uses the profits map that isn't the proxy.

    # Load inputs
    if args['calculate_bau_and_selective']:
        proportion_conserved = hb.as_array(os.path.join(args['run_folder'], 'proportion_conserved.tif'))
        carbon_conserved = hb.as_array(os.path.join(args['run_folder'], 'carbon_conserved.tif'))
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        proportion_cultivated_bau = hb.as_array(os.path.join(args['run_folder'], 'proportion_cultivated_bau.tif'))
        proportion_cultivated_selective = hb.as_array(os.path.join(args['run_folder'], 'proportion_cultivated_selective.tif'))
    else:
        proportion_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_conserved.tif'))
        carbon_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'carbon_conserved.tif'))
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        proportion_cultivated_bau = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_cultivated_bau.tif'))
        proportion_cultivated_selective = hb.as_array(os.path.join(args['intermediate_folder'], 'proportion_cultivated_selective.tif'))

    if args['create_base_ffn_inputs']:
        additional_carbon_value_available_per_grid = hb.as_array(os.path.join(args['run_folder'], 'additional_carbon_value_available_per_grid.tif'))
        desired_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'desired_ha_conserved.tif'))
        additional_carbon_value_available_per_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'additional_carbon_value_available_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved = hb.as_array(os.path.join(args['run_folder'], 'profit_forgone_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved_proxy = hb.as_array(os.path.join(args['run_folder'], 'profit_forgone_per_ha_conserved_proxy.tif'))
    else:
        additional_carbon_value_available_per_grid = hb.as_array(os.path.join(args['intermediate_folder'], 'additional_carbon_value_available_per_grid.tif'))
        desired_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'desired_ha_conserved.tif'))
        additional_carbon_value_available_per_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'additional_carbon_value_available_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved = hb.as_array(os.path.join(args['intermediate_folder'], 'profit_forgone_per_ha_conserved.tif'))
        profit_forgone_per_ha_conserved_proxy = hb.as_array(os.path.join(args['intermediate_folder'], 'profit_forgone_per_ha_conserved_proxy.tif'))


    if args['n_method'] == 'ones':
        # Just equal to 1
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        nmb_denominator = np.ones(proportion_cultivated.shape)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
    elif args['n_method'] == 'proportion_cultivated':
        # Based on prop cultivated
        proportion_cultivated = hb.as_array(os.path.join(args['input_folder'], 'proportion_cultivated.tif'))
        nmb_denominator = np.where(proportion_cultivated > 0, proportion_cultivated, -255)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
    elif args['n_method'] == 'profit_proxy':
        # Based on dollars per calorie using corn price
        nmb_denominator = np.where(profit_forgone_per_ha_conserved_proxy > 0, profit_forgone_per_ha_conserved_proxy, 0.0)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)
    elif args['n_method'] == 'profit':
        # Based on dollars per calorie using corn price
        nmb_denominator = np.where(profit_forgone_per_ha_conserved > 0, profit_forgone_per_ha_conserved, 0.0)
        nmb = np.where((additional_carbon_value_available_per_ha_conserved > 0) & (desired_ha_conserved > 0), additional_carbon_value_available_per_ha_conserved / nmb_denominator, 0)

    subsidy_savings_path = os.path.join(args['workspace'], '../../../../../base_data', 'pes_policy_identification_inputs', '20201209_SubsidySavings_v1_with_npv.xlsx')

    xls = pd.ExcelFile(subsidy_savings_path)
    region_names = ['ARG', 'BGD', 'BRAZIL', 'C_C_Amer', 'CAN', 'CHIHKG', 'COL', 'E_Asia', 'EGY', 'ETH', 'EU27', 'IDN', 'INDIA', 'JAPAN', 'KOR', 'MAR', 'MDG', 'MEAS_NAfr', 'MEX', 'MYS', 'NGA', 'Oceania', 'Oth_CEE_CIS', 'Oth_Europe', 'PAK', 'PHL', 'POL', 'R_S_Asia', 'R_SE_Asia', 'Russia', 'S_o_Amer', 'S_S_AFR', 'TUR', 'USA', 'VNM', 'XAC', 'ZAF', ]
    df = xls.parse('Annualized')
    # df = xls.parse('Annualized', skiprows=4, index_col=None, na_values=['NA'])


    array = df.iloc[4:41, 1:11].values

    npvs = []
    for i in range(len(region_names)):
        current_npv = 0
        for j in range(10):
            current_npv += (array[i, j] / (1+.08)**(j+1)) * 1000000 / 2

        npvs.append(current_npv)

    budget = dict(zip(region_names, npvs))
    budget.update({0: 0})
    gtap_37_path = r"C:\Files\Research\cge\gtap_invest\base_data\gtap37.gpkg"
    gtap_zones_raster_path = os.path.join(args['run_folder'], 'gtap_zones_raster.tif')
    gdf = gpd.read_file(gtap_37_path)
    correspondence = dict(zip([int(i) for i in gdf['AreaCode']], gdf['GTAP37']))
    correspondence.update({-9999: 0})
    print('correspondence', correspondence)
    hb.convert_polygons_to_id_raster(gtap_37_path, gtap_zones_raster_path, args['match_uri'],
                                  id_column_label='AreaCode', data_type=7, ndv=-9999, all_touched=None, compress=True)
    gtap_zones_raster = hb.as_array(gtap_zones_raster_path)
    # df = df.loc[df['Savings from Subsidies'].isin(region_names)]
    # print('df', df)

    # # Hackish way of selecting the npv row
    # df = df.iloc[42, 2:11]
    # npv_total = df.sum()
    # print('npv_total', npv_total)



    hb.save_array_as_geotiff(nmb, os.path.join(args['run_folder'], 'nmb_f_payments_country_level.tif'), args['match_uri'], data_type=6,  ndv=-255)

    unraveled_keys = np.unravel_index(nmb.argsort(axis=None), shape=nmb.shape) #returns TWO arrays in a tuple of all the rows, then all the cols keys
    nmb_sorted_keys = (unraveled_keys[0][::-1], unraveled_keys[1][::-1]) # thus here we need to reverse each seperately

    unraveled_keys = None

    def calc_best_reduction_factor_given_range(min, max, step):

        # Record the alpha-protected value combinations for later plotting
        value_obtained_points = OrderedDict()
        value_missed_points = OrderedDict()

        if max > 1.0:
            max = 1.0
            LOGGER.warn('Attempted to test a reduction factor greater than 1. Why are you asking this?')

        if min < 0.0:
            min = 0.0
            LOGGER.warn('Attempted to test a reduction factor less than 0. Why are you asking this?')

        highest_value_yet = 0
        best_reduction_factor = 0
        for reduction_factor in np.arange(min, max+step, step):
            payment_offer_f_per_ha = reduction_factor * additional_carbon_value_available_per_ha_conserved
            sum_spent = 0
            value_obtained = 0
            value_missed = 0
            num_rejected_payments = 0
            for i in range(len(nmb_sorted_keys[0])):
                current_key = (nmb_sorted_keys[0][i], nmb_sorted_keys[1][i])
                if nmb[current_key] <= 0:
                    LOGGER.warn('N = 1 in all remaining cells. (You have run out of locations that are worth protecting')
                    break

                value_in_grid = additional_carbon_value_available_per_grid[current_key]
                expenditure_in_grid = payment_offer_f_per_ha[current_key] * desired_ha_conserved[current_key]


                if payment_offer_f_per_ha[current_key] < profit_forgone_per_ha_conserved_proxy[current_key]:
                    num_rejected_payments += 1
                    if payment_offer_f_per_ha[current_key] > profit_forgone_per_ha_conserved[current_key]:
                        value_missed += value_in_grid


                else:
                    value_obtained += value_in_grid
                    sum_spent += expenditure_in_grid
                    if sum_spent > args['conservation_budget']:
                        break
                value_obtained_points[float(reduction_factor)] = float(value_obtained)
                value_missed_points[float(reduction_factor)] = float(value_missed)

            if value_obtained > highest_value_yet:
                highest_value_yet = value_obtained
                best_reduction_factor = reduction_factor
            LOGGER.info('Testing of reduction factor ' + str(reduction_factor) + ' resulted in conservation value of ' + str(value_obtained) + ' at expense of ' + str(sum_spent))
        return best_reduction_factor, value_obtained_points, value_missed_points

    #the following has to be EXACTLY right or you get crazy results.

    args['best_reduction_factor'] = 0.7

    if 'best_reduction_factor' in args:
        best_reduction_factor = args['best_reduction_factor']
        skip = True
    else:
        skip = False
    if not skip:
        fractional_reduction_to_value = OrderedDict()
        fractional_reduction_to_value_missed = OrderedDict()
        current_best, value_obtained_points, value_missed_points = calc_best_reduction_factor_given_range(0,1,.025)
        fractional_reduction_to_value.update(value_obtained_points)
        fractional_reduction_to_value_missed.update(value_missed_points)

        current_best, value_obtained_points, value_missed_points = calc_best_reduction_factor_given_range(current_best - .1, current_best + .1,.01)
        fractional_reduction_to_value.update(value_obtained_points)
        fractional_reduction_to_value_missed.update(value_missed_points)

        best_reduction_factor = current_best

        # current_best, value_obtained_points, value_missed_points = calc_best_reduction_factor_given_range(current_best - .01, current_best + .01,.001)
        # fractional_reduction_to_value.update(value_obtained_points)
        # fractional_reduction_to_value_missed.update(value_missed_points)
        #
        # best_reduction_factor, value_obtained_points, value_missed_points = calc_best_reduction_factor_given_range(current_best - .001, current_best + .001,.0001)
        # fractional_reduction_to_value.update(value_obtained_points)
        # fractional_reduction_to_value_missed.update(value_missed_points)

        # Make a new sorted dict
        fractional_reduction_to_value = OrderedDict(sorted(fractional_reduction_to_value.items()))
        fractional_reduction_to_value_missed = OrderedDict(sorted(fractional_reduction_to_value_missed.items()))

        fractional_reduction_to_value_uri = os.path.join(args['run_folder'], 'fractional_reduction_to_value.csv')
        hb.python_object_to_csv(fractional_reduction_to_value, fractional_reduction_to_value_uri)

        fractional_reduction_to_value_missed_uri = os.path.join(args['run_folder'], 'fractional_reduction_to_value_missed.csv')
        hb.python_object_to_csv(fractional_reduction_to_value_missed, fractional_reduction_to_value_missed_uri)



        #best_reduction_factor, current_points = calc_best_reduction_factor_given_range(current_best - .0001, current_best + .0001,.00001)


    payment_offer_f_per_ha = best_reduction_factor * additional_carbon_value_available_per_ha_conserved

    #additional_carbon_value_available_per_ha_conserved = None

    sum_spent = 0
    value_obtained = 0
    num_rejected_payments = 0
    value_protected_f_per_grid = np.zeros(nmb.shape)
    expenditure_f_per_grid = np.zeros(nmb.shape)
    order_of_payments_f = np.zeros(nmb.shape)
    payment_set_f_per_ha = np.zeros(nmb.shape)


    for i in range(len(nmb_sorted_keys[0])):
        current_key = (nmb_sorted_keys[0][i], nmb_sorted_keys[1][i])
        if nmb[current_key] <= 0:
            LOGGER.warn('N = 1 in all remaining cells. (You have run out of locations that are worth protecting')
            break

        current_zone = gtap_zones_raster[current_key]
        current_zone_name = correspondence[current_zone]

        if payment_offer_f_per_ha[current_key] < profit_forgone_per_ha_conserved[current_key]:
            num_rejected_payments += 1
        else:
            value_in_grid = additional_carbon_value_available_per_grid[current_key]
            expenditure_in_grid = payment_offer_f_per_ha[current_key] * desired_ha_conserved[current_key]

            if budget[current_zone_name] > 0:
                value_obtained += value_in_grid
                budget[current_zone_name] -= expenditure_in_grid
                value_protected_f_per_grid[current_key] = value_in_grid
                expenditure_f_per_grid[current_key] = expenditure_in_grid
                order_of_payments_f[current_key] = i + 1

                payment_set_f_per_ha[current_key] = payment_offer_f_per_ha[current_key]
            else:
                pass
                # print('skipping', current_zone_name)

    LOGGER.info('\nFFN with reduced price solved.')
    LOGGER.info('Best reduction factor was ' + str(best_reduction_factor))
    LOGGER.info('Carbon value Saved,' + str(np.nansum(value_protected_f_per_grid)))
    LOGGER.info('Money Spent,' + str(np.nansum(expenditure_f_per_grid)))
    LOGGER.info('num_rejected_payments,' + str(num_rejected_payments))

    hb.save_array_as_geotiff(payment_set_f_per_ha, os.path.join(args['run_folder'], 'payment_set_f_per_ha_country_level.tif'), args['match_uri'], data_type=6,  ndv = -255)
    hb.save_array_as_geotiff(order_of_payments_f, os.path.join(args['run_folder'], 'order_of_payments_f_country_level.tif'), args['match_uri'], data_type=6,  ndv = -255)
    # hb.save_array_as_geotiff(value_protected_f_per_grid, os.path.join(args['run_folder'], 'value_protected_f_per_grid.tif'), args['match_uri'], data_type=6,  ndv = -255)
    hb.save_array_as_geotiff(value_protected_f_per_grid, os.path.join(args['run_folder'], 'value_protected_given_budget_per_grid_f_country_level.tif'), args['match_uri'], data_type=6,  ndv = -255)
    hb.save_array_as_geotiff(expenditure_f_per_grid, os.path.join(args['run_folder'], 'expenditure_f_per_grid_country_level.tif'), args['match_uri'], data_type=6,  ndv = -255)

    hb.save_array_as_geotiff(payment_offer_f_per_ha, os.path.join(args['run_folder'], 'payment_offer_f_per_ha_country_level.tif'), args['match_uri'], data_type=6,  ndv = -255)

    return




def produce_ffn_publication_figures(args):
    #         if args['calculate_ffn_b_payments']:
    #             order_of_payments = hb.as_array(os.path.join(args['run_folder'], 'order_of_payments.tif'))
    #             optimal_payments_given_budget = hb.as_array(os.path.join(args['run_folder'], 'optimal_payments_given_budget.tif'))
    #             order_of_payments_given_budget = hb.as_array(os.path.join(args['run_folder'], 'order_of_payments_given_budget.tif'))
    #             filelist = os.listdir(args['run_folder'])
    #         else:
    #             order_of_payments = hb.as_array(os.path.join(args['input_folder'], 'order_of_payments.tif'))
    #             optimal_payments_given_budget = hb.as_array(os.path.join(args['input_folder'], 'optimal_payments_given_budget.tif'))
    #             order_of_payments_given_budget = hb.as_array(os.path.join(args['input_folder'], 'order_of_payments_given_budget.tif'))
    #             filelist = os.listdir(args['input_folder'])
    file_list = [
        'order_of_payments',
        'n',
        'expenditure_f_per_grid',
        'expenditure_a_per_grid',
        'payment_set_given_budget_per_ha',
        'payment_set_f_per_ha',
        'payment_set_a_per_ha',
        'value_protected_given_budget_per_grid',
        'value_protected_f_per_grid',
        'value_protected_a_per_grid',
    ]

    # arrays_to_plot = []
    # for file_name in file_list:
    #     try:
    #         arrays_to_plot.append(hb.as_array(os.path.join(args['run_folder'], file_name + '.tif')))
    #     except:
    #         arrays_to_plot.append(hb.as_array(os.path.join(args['input_folder'], file_name + '.tif')))

    # Kludge.
    if args['create_base_ffn_inputs']:
        base_ffn_inputs_folder = args['run_folder']
    else:
        base_ffn_inputs_folder = args['intermediate_folder']

    # Set general figure parameters
    title = None
    show_lat_lon = False #if True, show best 5 graticules.
    use_basemap = True
    show_when_generated = True
    resolution = 'i' #c (coarse), l, i, h
    center_cbar_at = 0
    num_cbar_ticks = 3
    no_data_value = -255
    crop_inf = True
    reverse_colorbar = True
    insert_white_divergence_point = False
    show_output = False
    generate_cmap_from_data = True

    # Figure 1
    input_array = hb.as_array(os.path.join(base_ffn_inputs_folder, 'nmb.tif'))
    output_uri = os.path.join(args['run_folder'], 'fig_1_' + file_list[1] + '.png')
    bounding_box = 'has_gli_data'
    cbar_label = 'N (higher values indicate higher conservation priority)'
    vmin = 0
    vmax = 5000
    vmin = None
    vmax = 6000
    num_cbar_ticks = 2
    # tick_labels = ['< 0 or no data', 2.5]
    color_scheme = 'bold_spectral' #jet spectral, spectral_contrast, bw, prbg
    hb.show_array(input_array=input_array, show_output=show_output, generate_cmap_from_data = generate_cmap_from_data,
                  output_uri=output_uri, use_basemap=use_basemap,
                  title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution,
                  bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks,
                  no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  insert_white_divergence_point=insert_white_divergence_point)

    # Figure 2
    input_array = hb.as_array(os.path.join(base_ffn_inputs_folder, 'order_of_payments.tif'))
    output_uri = os.path.join(args['run_folder'], 'fig_2_' + file_list[0] + '.png')
    bounding_box = 'has_gli_data'
    cbar_label = 'Order of payments (lower = sooner)'
    vmin = 0
    vmax = 500000
    # vmin = None
    # vmax = None
    color_scheme = 'bold_spectral' #spectral, spectral_contrast, bw, prbg
    hb.show_array(input_array=input_array, show_output=show_output, generate_cmap_from_data = generate_cmap_from_data, output_uri=output_uri, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    # Figure 3a
    num_cbar_ticks = 3
    input_array = hb.as_array(os.path.join(base_ffn_inputs_folder, 'value_protected_given_budget_per_grid.tif'))
    output_uri = os.path.join(args['run_folder'], 'fig_3a_global_value_protected_given_budget_per_grid.png')
    use_basemap = True
    bounding_box = 'has_gli_data'
    cbar_label = 'Payments to offer given perfect information ($ per grid-cell)'
    vmin = 12000
    vmax = 24000
    # vmin = None
    # vmax = None
    generate_cmap_from_data = True
    color_scheme = 'bold_spectral' #spectp[=ral, spectral_contrast, bw, prbg
    hb.show_array(input_array=input_array, show_output=show_output, generate_cmap_from_data = generate_cmap_from_data, output_uri=output_uri, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    # Figure 3b
    input_array = hb.as_array(os.path.join(base_ffn_inputs_folder, 'value_protected_given_budget_per_grid.tif'))
    output_uri = os.path.join(args['run_folder'], 'fig_3b_se_asia_value_protected_given_budget_per_grid.png')
    use_basemap = True
    bounding_box = 'se_asia'
    cbar_label = 'Payments to offer given perfect information ($ per grid-cell)'
    vmin = 12000
    vmax = 24000
    # vmin = None
    # vmax = None
    generate_cmap_from_data = True
    color_scheme = 'bold_spectral' #spectp[=ral, spectral_contrast, bw, prbg
    hb.show_array(input_array=input_array, show_output=show_output, generate_cmap_from_data = generate_cmap_from_data, output_uri=output_uri, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)





    # Figure 4a
    num_cbar_ticks = 3
    input_array = hb.as_array(os.path.join(base_ffn_inputs_folder, 'value_protected_f_per_grid.tif'))
    output_uri = os.path.join(args['run_folder'], 'fig_4a_global_value_protected_f_per_grid.png')
    use_basemap = True
    bounding_box = 'has_gli_data'
    cbar_label = 'Payments to offer given price reduction ($ per grid-cell)'
    vmin = 0
    vmax = 3800
    # vmin = None
    # vmax = None
    generate_cmap_from_data = True
    color_scheme = 'bold_spectral' #spectp[=ral, spectral_contrast, bw, prbg
    hb.show_array(input_array=input_array, show_output=show_output, generate_cmap_from_data = generate_cmap_from_data, output_uri=output_uri, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    # Figure 4b
    input_array = hb.as_array(os.path.join(base_ffn_inputs_folder, 'value_protected_f_per_grid.tif'))
    output_uri = os.path.join(args['run_folder'], 'fig_4b_se_asia_value_protected_f_per_grid.png')
    use_basemap = True
    bounding_box = 'se_asia'
    cbar_label = 'Payments to offer given price reduction ($ per grid-cell)'
    vmin = 0
    vmax = 3800
    # vmin = None
    # vmax = None
    generate_cmap_from_data = True
    color_scheme = 'bold_spectral' #spectp[=ral, spectral_contrast, bw, prbg
    hb.show_array(input_array=input_array, show_output=show_output, generate_cmap_from_data = generate_cmap_from_data, output_uri=output_uri, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)






    # Figure 5a
    num_cbar_ticks = 3
    input_array = hb.as_array(os.path.join(base_ffn_inputs_folder, 'value_protected_a_per_grid.tif'))
    output_uri = os.path.join(args['run_folder'], 'fig_5a_global_value_protected_a_per_grid.png')
    use_basemap = True
    bounding_box = 'has_gli_data'
    cbar_label = 'Payments to offer given price reduction and asymmetric info ($ per grid-cell)'
    vmin = 0
    vmax = 3800
    # vmin = None
    # vmax = None
    generate_cmap_from_data = True
    color_scheme = 'bold_spectral' #spectp[=ral, spectral_contrast, bw, prbg
    hb.show_array(input_array=input_array, show_output=show_output, generate_cmap_from_data = generate_cmap_from_data, output_uri=output_uri, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    # Figure 5b
    input_array = hb.as_array(os.path.join(base_ffn_inputs_folder, 'value_protected_a_per_grid.tif'))
    output_uri = os.path.join(args['run_folder'], 'fig_5b_se_asia_value_protected_a_per_grid.png')
    use_basemap = True
    bounding_box = 'se_asia'
    cbar_label = 'Payments to offer given price reduction and asymmetric info ($ per grid-cell)'
    vmin = 0
    vmax = 3800
    # vmin = None
    # vmax = None
    generate_cmap_from_data = True
    color_scheme = 'bold_spectral' #spectp[=ral, spectral_contrast, bw, prbg
    hb.show_array(input_array=input_array, show_output=show_output, generate_cmap_from_data = generate_cmap_from_data, output_uri=output_uri, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    # Figure 6
    input_array = hb.as_array(os.path.join(base_ffn_inputs_folder, 'expenditure_f_per_grid.tif'))
    output_uri = os.path.join(args['run_folder'], 'fig_6_expenditure_f_per_grid.png')
    use_basemap = True
    bounding_box = 'has_gli_data'
    cbar_label = 'Expenditures given price reduction ($ per grid-cell)'
    vmin = 0
    vmax = 8500000
    # vmin = None
    # vmax = None
    generate_cmap_from_data = True
    color_scheme = 'bold_spectral' #spectral, spectral_contrast, bw, prbg
    hb.show_array(input_array=input_array, show_output=show_output, generate_cmap_from_data = generate_cmap_from_data, output_uri=output_uri, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    #Figure 7
    input_array = hb.as_array(os.path.join(base_ffn_inputs_folder, 'expenditure_a_per_grid.tif'))
    output_uri = os.path.join(args['run_folder'], 'fig_7_expenditure_a_per_grid.png')
    use_basemap = True
    bounding_box = 'has_gli_data'
    cbar_label = 'Expenditures given price reduction and asymmetric info ($ per grid-cell)'
    vmin = 0
    vmax = 8500000
    # vmin = None
    # vmax = None
    generate_cmap_from_data = True
    color_scheme = 'bold_spectral' #spectral, spectral_contrast, bw, prbg
    hb.show_array(input_array=input_array, show_output=show_output, generate_cmap_from_data = generate_cmap_from_data, output_uri=output_uri, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

    #Figure 8
    input_array = np.where(hb.as_array(os.path.join(base_ffn_inputs_folder, 'expenditure_a_per_grid.tif')) > 0, hb.as_array(os.path.join(base_ffn_inputs_folder, 'desired_ha_conserved.tif')), 0.0)
    output_uri = os.path.join(args['run_folder'], 'fig_8_hectares_prevented_from_being_extensified.png')
    use_basemap = True
    bounding_box = 'has_gli_data'
    cbar_label = 'Hectares per grid-cell prevented from being extensified\ngiven price reduction and asymmetric info'
    # vmin = None
    # vmax = None
    generate_cmap_from_data = True
    color_scheme = 'bold_spectral' #spectral, spectral_contrast, bw, prbg
    hb.show_array(input_array=input_array, show_output=show_output, generate_cmap_from_data = generate_cmap_from_data, output_uri=output_uri, use_basemap=use_basemap, title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box, vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar, insert_white_divergence_point=insert_white_divergence_point)

def produce_ffn_publication_figures_v2(args):
    primary_folder = args['run_folder']
    secondary_folder = args['intermediate_folder']



    # Figure 1
    if os.path.exists(os.path.join(primary_folder, 'payment_set_b_per_ha.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, 'payment_set_b_per_ha.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, 'payment_set_b_per_ha.tif'))
    output_uri = os.path.join(args['run_folder'], 'payment_set_b_per_ha.png')
    a = hb.show_array(input_array, output_uri=output_uri, use_basemap=True)
    # hb.show_array(input_array=input_array, show_output=show_output, generate_cmap_from_data = generate_cmap_from_data, output_uri=output_uri, use_basemap=use_basemap,
    #               title=title, cbar_label=cbar_label, show_lat_lon=show_lat_lon, resolution=resolution, bounding_box=bounding_box,
    #               vmin=vmin, vmax=vmax, center_cbar_at=center_cbar_at, num_cbar_ticks=num_cbar_ticks, no_data_value=no_data_value, crop_inf=crop_inf, color_scheme=color_scheme,
    #               reverse_colorbar=reverse_colorbar)
    # Figure 2
    if os.path.exists(os.path.join(primary_folder, 'payment_set_c_per_ha.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, 'payment_set_c_per_ha.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, 'payment_set_c_per_ha.tif'))
    output_uri = os.path.join(args['run_folder'], 'payment_set_c_per_ha.png')
    a = hb.show_array(input_array, output_uri=output_uri, use_basemap=True)

def produce_base_data_figs(args):
    uris = [
    os.path.join(args['run_folder'], 'desired_ha_conserved.tif'),
    os.path.join(args['run_folder'], 'additional_carbon_value_available_per_grid.tif'),
    os.path.join(args['run_folder'], 'additional_carbon_value_available_per_ha_conserved.tif'),
            ]

    for uri in uris:
        hb.show(uri, output_uri=uri.replace('.tif', '.png'),use_basemap = True,
        resolution = 'i',
        num_cbar_ticks = 3, bounding_box = 'has_gli_data')



def produce_base_mechanism_figs(args):
    primary_folder = args['run_folder']
    secondary_folder = args['intermediate_folder']

    use_basemap = True
    resolution = 'i'
    num_cbar_ticks = 3
    color_scheme = None
    reverse_colorbar = False
    bounding_box = 'has_gli_data'
    vmin = None
    vmax = None
    vmid = None

    # Figure 1
    name = 'nmb_b'
    cbar_label = 'N (net marginal benefit of public value per private value forgone)'
    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], 'nmb_b.png')

    vmin = 0
    vmax = 5

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=True,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    # Figure 1
    name = 'order_of_payments_b'
    cbar_label = 'Order to offer payments (lower is sooner)'
    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], 'order_of_payments.png')

    vmin = 0
    vmax = 500000

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=True,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    # Figure 1
    name = 'payment_set_b_per_ha'
    cbar_label = 'Payment value per ha (2010 $US)'

    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '.png')

    vmin = 4000
    vmax = 24000


    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    # Figure 1
    name = 'order_of_payments_b'
    cbar_label = 'Order to offer payments (lower is sooner)'

    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '.png')

    vmin = None
    vmax = None

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    # Figure 1
    name = 'value_protected_given_budget_per_grid_b'
    cbar_label = 'Carbon value protected (2010 $US)'

    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '.png')

    vmin = 0
    vmax = 8000000

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    # # Figure 1
    # name = 'nmb_c'
    # if os.path.exists(os.path.join(primary_folder, name + '.tif')):
    #     input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    # else:
    #     input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    # output_uri = os.path.join(args['run_folder'], name + '.png')
    #
    # vmin = None
    # vmax = None
    #
    # hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
    #               title=None, cbar_label=cbar_label, bounding_box=bounding_box,
    #               vmin=vmin, vmax=vmax, vmid=vmid)

    # Figure 1
    name = 'payment_set_c_per_ha'
    cbar_label = 'Payment value per ha (2010 $US)'

    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '.png')

    vmin = 0
    vmax = 10000

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    # Figure 1
    name = 'order_of_payments_c'
    cbar_label = 'Order to offer payments (lower is sooner)'

    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '.png')

    vmin = None
    vmax = None

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    # Figure 1
    name = 'value_protected_given_budget_per_grid_c'
    cbar_label = 'Carbon value protected (2010 $US)'

    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '.png')

    vmin = 0
    vmax = 8000000

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)


    # Figure 1
    name = 'payment_set_c_per_ha'
    cbar_label = 'Payment value per ha (2010 $US)'


    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '_se_asia.png')

    vmin = 0
    vmax = 10000
    bounding_box = 'se_asia'

    resolution = 'h'

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    resolution = 'i'

    # PRICE_REDUCTION
    name = 'payment_set_f_per_ha'
    cbar_label = 'Payment value per ha (2010 $US)'

    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '.png')

    vmin = 0
    vmax = 10000

    bounding_box = 'has_gli_data'

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    # Figure 1
    name = 'order_of_payments_f'
    cbar_label = 'Order to offer payments (lower is sooner)'

    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '.png')

    vmin = None
    vmax = None

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    # Figure 1
    name = 'value_protected_given_budget_per_grid_f'
    cbar_label = 'Carbon value protected (2010 $US)'

    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '.png')

    vmin = 0
    vmax = 8000000

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)


    # Figure 1
    name = 'payment_set_f_per_ha'
    cbar_label = 'Payment value per ha (2010 $US)'

    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '_se_asia.png')

    vmin = 0
    vmax = 10000
    bounding_box = 'se_asia'

    resolution = 'h'

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    resolution = 'i'

def produce_fractional_value_plot(args):
    matplotlib.style.use('ggplot')

    fractional_reduction_to_value_uri = os.path.join(args['run_folder'], 'fractional_reduction_to_value.csv')
    if os.path.exists(fractional_reduction_to_value_uri):
        df = pd.read_csv(fractional_reduction_to_value_uri, names=['Fractional reduction','Carbon value saved (2010 $US)'])
    else:
        df = pd.read_csv(os.path.join(args['intermediate_folder'], 'fractional_reduction_to_value.csv'), names=['Fractional reduction','Carbon value saved (2010 $US)'])

    plt.plot(df['Fractional reduction'], df['Carbon value saved (2010 $US)'])
    plt.xlabel('Fractional reduction')
    plt.ylabel('Carbon value saved (2010 $US)')
    plt.xlim([0,1])
    plt.gca().set_ylim(bottom=0)

    fractional_reduction_to_value_plot_uri = os.path.join(args['run_folder'], 'fractional_reduction_to_value.png')
    plt.savefig(fractional_reduction_to_value_plot_uri)

    fractional_reduction_to_value_uri = os.path.join(args['run_folder'], 'fractional_reduction_to_value_missed.csv')
    if os.path.exists(fractional_reduction_to_value_uri):
        df = pd.read_csv(fractional_reduction_to_value_uri, names=['Fractional reduction','Carbon value missed (2010 $US)'])
    else:
        df = pd.read_csv(os.path.join(args['intermediate_folder'], 'fractional_reduction_to_value_missed.csv'), names=['Fractional reduction','Carbon value missed (2010 $US)'])

    plt.plot(df['Fractional reduction'], df['Carbon value missed (2010 $US)'])
    plt.xlabel('Fractional reduction')
    plt.ylabel('Carbon value missed (2010 $US)')
    plt.xlim([0,1])
    plt.gca().set_ylim(bottom=0)

    fractional_reduction_to_value_plot_uri = os.path.join(args['run_folder'], 'fractional_reduction_to_value_with_missed.png')
    plt.savefig(fractional_reduction_to_value_plot_uri)
    # plt.show()

def produce_auction_mechaism_figs(args):
    matplotlib.style.use('classic')

    primary_folder = args['run_folder']
    secondary_folder = args['intermediate_folder']

    use_basemap = True
    resolution = 'i'
    num_cbar_ticks = 3
    color_scheme = None
    reverse_colorbar = False
    bounding_box = 'has_gli_data'
    vmin = None
    vmax = None
    vmid = None

    # Figure 1
    name = 'payment_set_a_per_ha'
    cbar_label = 'Payment value per ha (2010 $US)'

    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '.png')

    vmin = 0
    vmax = 10000

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    # Figure 1
    name = 'order_of_payments_a'
    cbar_label = 'Order to offer payments (lower is sooner)'

    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '.png')

    vmin = None
    vmax = None

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    # Figure 1
    name = 'value_protected_given_budget_per_grid_a'
    cbar_label = 'Carbon value protected (2010 $US)'

    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '.png')

    vmin = 0
    vmax = 8000000

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    # Figure 1
    name = 'nmb_a'
    cbar_label = 'N'

    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '.png')

    vmin = 0
    vmax = 5

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    # Figure 1
    name = 'payment_set_a_per_ha'
    cbar_label = 'Payment value per ha (2010 $US)'


    if os.path.exists(os.path.join(primary_folder, name + '.tif')):
        input_array = hb.as_array(os.path.join(primary_folder, name + '.tif'))
    else:
        input_array = hb.as_array(os.path.join(secondary_folder, name + '.tif'))
    output_uri = os.path.join(args['run_folder'], name + '_se_asia.png')

    vmin = 0
    vmax = 10000
    bounding_box = 'se_asia'

    resolution = 'h'

    hb.show_array(input_array=input_array, output_uri=output_uri, use_basemap=use_basemap, resolution=resolution, num_cbar_ticks=num_cbar_ticks, color_scheme=color_scheme, reverse_colorbar=reverse_colorbar,
                  title=None, cbar_label=cbar_label, bounding_box=bounding_box,
                  vmin=vmin, vmax=vmax, vmid=vmid)

    resolution = 'i'


def main():
    pass


if __name__ == '__main__':

    # all information required by this code is passed to ffn.execute via the dictionary 'args'. Default values for args are included in ffn_pre_numdal.py but they will be overwritten by any run-specific args defined here.
    args = {}

    # Override default dataset locations with user-defined versions. May be useful if you want to reference datasets stored elsewhere without having to creat a new local version
    args.update({'bulk_input_folder': 'E:/bulk_data/'})
    # args.update({'crop_statistics_folder': os.path.join(args['bulk_input_folder'], 'earthstat/HarvestedAreaYield175Crops/')}) #location of Earthstat data. Must have 172 crop-specific folders (as provided by earthstat.org)
    args.update({'soil_carbon_input_uri': os.path.join(args['bulk_input_folder'], 'soil/1kmsoilgrids/OCSTHA_sd6_M_02_apr_2014.tif')})  # the current version of Ag_tradeoffs uses 1km soil carbon data, located in this geotiff.

    # Input the parameters that are specific to this run.
    args.update({'calorie_increase': 0.7,
                 'assumed_intensification': .75,  # wh5t portion of the calorie increase will be met with higher yields?
                 'transition_function_alpha': 1.0,  # the transition function is post_extensification_prop_cultivated = alpha*initial_prop_cultivated ^ beta
                 'transition_function_beta': 0.5,
                 'solution_precision': 0.01,  # lower values result in more answer precision
                 'min_extensification': 0.05,  # assume no extensification can happen in cells with less current extensifcation than this
                 'max_extensification': 0.95,  # assume no extensification can happen in cells with more current extensifcation than this
                 'conservation_budget': 471163000000/2,  # 1654,000,000,000 to implement all. 100000000000 was paper default 176,959,000,000 # Subs replacement 471163000000
                 'social_cost_of_carbon': 171.,  # tol 2009, 3%, 171 is from 1%
                 'asymmetric_scatter': .06,
                 'npv_adjustment': 20.91744579,  # based on 0.05 discount rate into exp(-rt) over 30 years.
                 'n_method': 'profit',  # ones, proportion_cultivated, profit_proxy
                 })

    # Choose which elements of the model to run anew vs. load a pre-calculated version.

    args_ffn = {}
    args_ffn.update({'calculate_yield_tons_per_cell': 1,
                     'aggregate_soil_carbon_data': 0,
                     'calculate_calories_per_cell': 1,
                     'calculate_change_in_carbon': 1,
                     'calculate_bau_and_selective': 1,
                     'produce_ag_tradeoffs_publication_figures': 0,
                     'create_base_ffn_inputs': 1,
                     'produce_base_data_figs': 0,
                     'calculate_ffn_b_payments': 0,
                     'calculate_ffn_c_payments': 0,
                     'calculate_ffn_f_payments': 1,
                     'calculate_ffn_a_payments': 0,
                     'calculate_ffn_f_country_level_payments': 1,
                     'produce_base_mechanism_figs': 0,
                     'produce_fractional_value_plot': 0,
                     'produce_auction_mechaism_figs': 0,
                     })

    args.update(args_ffn)

    execute(args)



