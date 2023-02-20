# initialize_project defines the run() command for the whole project, which takes the project object as its only function.
# It is responsible for:
#   initialize_paths(p)
#

import os
import hazelbean as hb

import seals_main
import seals_generate_base_data
import seals_process_luh2
import config


import gtap_invest_main
import seals_main
import seals_process_luh2
from gtap_invest.visualization import visualization

def run(p):

    # Configure the logger that captures all the information generated.
    p.L = hb.get_logger('test_run_gtap_invest')

    p.L.info('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script + '\n    with base_data set at ' + p.base_data_dir)

    initialize_paths(p)

    # Sometimes runs fail mid run. This checks for that and picks up where there is a completed file for that zone.
    p.skip_created_downscaling_zones = 0

    # As it calibrates, optionally write each calibration allocation step
    p.write_calibration_generation_arrays = 1



    # Before training and running the model, the LULC map is simplified to fewer classes. Each one currently has a label
    # such as 'seals7', defined in the config code.
    p.lulc_remap_labels = {}
    p.lulc_simplification_label = 'seals7'
    p.lulc_label = 'lulc_esa'
    p.lulc_simplified_label = 'lulc_esa_' + p.lulc_simplification_label
    p.lulc_simplification_remap = config.esa_to_seals7_correspondence

    # THIS CODE IS REDUNDANT AND NEEDS TO BE REFACTORED to draw from the simplifcation remap or a more robust type of input CSV.
    # SEALS-simplified classes are defined here, which can be iterated over. We also define what classes are shifted by GTAP's endogenous land-calcualtion step.
    p.class_indices = [1, 2, 3, 4, 5]  # These are the indices of classes THAT CAN EXPAND/CONTRACT
    p.nonchanging_class_indices = [6, 7]  # These add other lulc classes that might have an effect on LUC but cannot change themselves (e.g. water, barren)
    p.regression_input_class_indices = p.class_indices + p.nonchanging_class_indices

    p.class_labels = ['urban', 'cropland', 'grassland', 'forest', 'nonforestnatural', ]
    p.nonchanging_class_labels = ['water', 'barren_and_other']
    p.regression_input_class_labels = p.class_labels + p.nonchanging_class_labels

    p.shortened_class_labels = ['urban', 'crop', 'past', 'forest', 'other', ]

    p.class_indices_that_differ_between_ssp_and_gtap = [2, 3, 4, ]
    p.class_labels_that_differ_between_ssp_and_gtap = ['cropland', 'grassland', 'forest', ]

    # Specifies which sigmas should be used in a gaussian blur of the class-presence tifs in order to regress on adjacency.
    # Note this will have a huge impact on performance as full-extent gaussian blurs for each class will be generated for
    # each sigma.
    p.gaussian_sigmas_to_test = [1, 5]

    # Change how many generations of training to allow. A generation is an exhaustive search so relatievely few generations are required to get to a point
    # where no more improvements can be found.
    p.num_generations = 1

    # Provided by GTAP team.
    # TODOO This is still based on the file below, which was from Purdue. It is a vector of 300sec gridcells and should be replaced with continuous vectors
    p.gtap37_aez18_input_vector_path = os.path.join(p.base_data_dir, "pyramids", "GTAP37_AEZ18.gpkg")
    p.use_calibration_from_zone_centroid_tile = 1
    p.use_calibration_created_coefficients = 1
    p.calibration_zone_polygons_path = os.path.join(p.gtap37_aez18_input_vector_path)  # Only needed if use_calibration_from_zone_centroid_tile us True.

    #### Here is very ugly code that makes sure the different run configs point to the right files. This need to be fixed as an override of a default.
    # TODOO For magpie integration: make this an override.
    p.baseline_labels = ['baseline']
    p.baseline_coarse_state_paths = {}
    p.baseline_coarse_state_paths['baseline'] = {}
    p.baseline_coarse_state_paths['baseline'][p.base_year] = os.path.join(p.input_dir, "SSP2_BiodivPol_LPJmL5_2021-05-21_15.08.06", "cell.land_0.5_share_to_seals_SSP2_BiodivPol_LPJmL5.nc")

    # These are the POLICY scenarios. The model will iterate over these as well.
    p.gtap_combined_policy_scenario_labels = ['BAU', 'BAU_rigid', 'PESGC', 'SR_Land', 'PESLC', 'SR_RnD_20p', 'SR_Land_PESGC', 'SR_PESLC', 'SR_RnD_20p_PESGC', 'SR_RnD_PESLC', 'SR_RnD_20p_PESGC_30']
    p.gtap_just_bau_label = ['BAU']
    p.gtap_bau_and_30_labels = ['BAU', 'SR_RnD_20p_PESGC_30']
    p.luh_labels = ['no_policy']

    p.magpie_policy_scenario_labels = [
        'SSP2_BiodivPol_LPJmL5',
        'SSP2_BiodivPol_ClimPol_LPJmL5',
        'SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5',
        'SSP2_ClimPol_LPJmL5',
        'SSP2_NPI_base_LPJmL5',
    ]

    p.magpie_test_policy_scenario_labels = [
        # 'SSP2_BiodivPol_LPJmL5',
        # 'SSP2_BiodivPol_ClimPol_LPJmL5',
        'SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5',
        # 'SSP2_ClimPol_LPJmL5',
        # 'SSP2_NPI_base_LPJmL5',
    ]

    # Scenarios are defined by a combination of meso-level focusing layer that defines coarse LUC and Climate with the policy scenarios (or just scenarios) below.
    p.magpie_scenario_coarse_state_paths = {}
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'] = {}
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050] = {}
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_BiodivPol_LPJmL5'] = os.path.join(p.input_dir, "SSP2_BiodivPol_LPJmL5_2021-05-21_15.08.06", "cell.land_0.5_share_to_seals_SSP2_BiodivPol_LPJmL5.nc")
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_BiodivPol_ClimPol_LPJmL5'] = os.path.join(p.input_dir, "SSP2_BiodivPol+ClimPol_LPJmL5_2021-05-21_15.09.32", "cell.land_0.5_share_to_seals_SSP2_BiodivPol+ClimPol_LPJmL5.nc")
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5'] = os.path.join(p.input_dir, "SSP2_BiodivPol+ClimPol+NCPpol_LPJmL5_2021-05-21_15.10.54", "cell.land_0.5_share_to_seals_SSP2_BiodivPol+ClimPol+NCPpol_LPJmL5.nc")
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_ClimPol_LPJmL5'] = os.path.join(p.input_dir, "SSP2_ClimPol_LPJmL5_2021-05-21_15.12.19", "cell.land_0.5_share_to_seals_SSP2_ClimPol_LPJmL5.nc")
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_NPI_base_LPJmL5'] = os.path.join(p.input_dir, "SSP2_NPI_base_LPJmL5_2021-05-21_15.05.56", "cell.land_0.5_share_to_seals_SSP2_NPI_base_LPJmL5.nc")

    p.gtap_scenario_coarse_state_paths = {}
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'] = {}
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'][2030] = {}
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'][2050] = {}
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'][2030]['BAU'] = p.luh_scenario_states_paths['rcp45_ssp2']
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'][2050]['BAU'] = p.luh_scenario_states_paths['rcp45_ssp2']

    p.luh_scenario_coarse_state_paths = {}
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'] = {}
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'] = {}
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'][2030] = {}
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'][2050] = {}
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'][2030] = {}
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'][2050] = {}
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'][2030]['no_policy'] = p.luh_scenario_states_paths['rcp45_ssp2']
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'][2050]['no_policy'] = p.luh_scenario_states_paths['rcp45_ssp2']
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'][2030]['no_policy'] = p.luh_scenario_states_paths['rcp85_ssp5']
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'][2050]['no_policy'] = p.luh_scenario_states_paths['rcp85_ssp5']

    # TODOO: figure out is_calibration_run vs p.calibrate. I need to specify the different run achetypes (just luh2, alternative to luh2, shapefile ON TOP of luh2)

    if p.is_magpie_run:
        p.scenario_coarse_state_paths = p.magpie_scenario_coarse_state_paths
    elif p.is_gtap1_run:
        p.scenario_coarse_state_paths = p.gtap_scenario_coarse_state_paths
    elif p.is_calibration_run:
        p.scenario_coarse_state_paths = p.luh_scenario_coarse_state_paths

    if p.test_mode:
        if p.is_magpie_run:
            if p.policy_scenario_labels is None:
                p.policy_scenario_labels = p.magpie_test_policy_scenario_labels
        elif p.is_gtap1_run:
            if p.policy_scenario_labels is None:
                p.policy_scenario_labels = p.gtap_bau_and_30_labels
        elif p.is_calibration_run:
            if p.policy_scenario_labels is None:
                p.policy_scenario_labels = p.luh_labels
    else:
        if p.is_magpie_run:
            if p.policy_scenario_labels is None:
                p.policy_scenario_labels = p.magpie_policy_scenario_labels
        elif p.is_gtap1_run:
            if p.policy_scenario_labels is None:
                p.policy_scenario_labels = p.gtap_combined_policy_scenario_labels
        elif p.is_calibration_run:
            if p.policy_scenario_labels is None:
                p.policy_scenario_labels = p.luh_labels

    if p.is_gtap1_run:
        # HACK, because I don't yet auto-generate the cmf files and other GTAP modelled inputs, and instead just take the files out of the zipfile Uris
        # provides, I still have to follow his naming scheme. This list comprehension converts a policy_scenario_label into a gtap1 or gtap2 label.
        p.gtap1_scenario_labels = [str(p.policy_base_year) + '_' + str(p.scenario_years[0])[2:] + '_' + i + '_noES' for i in p.policy_scenario_labels]
        p.gtap2_scenario_labels = [str(p.policy_base_year) + '_' + str(p.scenario_years[0])[2:] + '_' + i + '_allES' for i in p.policy_scenario_labels]

    # This is a zipfile I received from URIS that has all the packaged GTAP files ready to run. Extract these to a project dir.
    p.gtap_aez_invest_release_string = '04_20_2021_GTAP_AEZ_INVEST'
    p.gtap_aez_invest_zipfile_path = os.path.join(p.base_data_dir, 'gtap_aez_invest_releases', p.gtap_aez_invest_release_string + '.zip')
    p.gtap_aez_invest_code_dir = os.path.join(p.script_dir, 'gtap_aez', p.gtap_aez_invest_release_string)

    # Associate each luh, year, and policy scenario with a set of seals input parameters. This can be used if, for instance, the policy you
    # are analyzing involves focusing land-use change into certain types of gridcells.
    p.gtap_pretrained_coefficients_path_dict = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'] = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030] = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2050] = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['BAU'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['BAU'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['PESGC'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['RnD'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_Land'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['PESLC'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_Land_PESGC'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_PESLC'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_PESGC'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_PESLC'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_20p'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_20p_PESGC'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_PESGC_30'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints_and_protected_areas.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_20p_PESGC_30'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints_and_protected_areas.xlsx')

    p.magpie_pretrained_coefficients_path_dict = {}
    p.magpie_pretrained_coefficients_path_dict['baseline'] = {}
    p.magpie_pretrained_coefficients_path_dict['baseline'][2015] = {}
    p.magpie_pretrained_coefficients_path_dict['baseline'][2015]['baseline'] = {}
    p.magpie_pretrained_coefficients_path_dict['baseline'][2015]['baseline'] = os.path.join(p.base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')

    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'] = {}
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050] = {}
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_BiodivPol_LPJmL5'] = os.path.join(p.base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_BiodivPol_ClimPol_LPJmL5'] = os.path.join(p.base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5'] = os.path.join(p.base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_ClimPol_LPJmL5'] = os.path.join(p.base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_NPI_base_LPJmL5'] = os.path.join(p.base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')

    # Define exact floating point representations of arcdegrees
    p.fine_resolution_degrees = hb.pyramid_compatible_resolutions[p.fine_resolution_arcseconds]
    p.coarse_resolution_degrees = hb.pyramid_compatible_resolutions[p.coarse_resolution_arcseconds]
    p.fine_resolution = p.fine_resolution_degrees
    p.coarse_resolution = p.coarse_resolution_degrees

    p.coarse_ha_per_cell_path = p.ha_per_cell_paths[p.coarse_resolution_arcseconds]
    p.coarse_match_path = p.coarse_ha_per_cell_path

    # A little awkward, but I used to use integers and list counting to keep track of the actual lulc class value. Now i'm making it expicit with dicts.
    p.class_indices_to_labels_correspondence = dict(zip(p.class_indices, p.class_labels))
    p.class_labels_to_indices_correspondence = dict(zip(p.class_labels, p.class_indices))

    p.calibrate = 1  # UNUSED EXCEPT IN Development features

    if p.is_gtap1_run:
        p.pretrained_coefficients_path_dict = p.gtap_pretrained_coefficients_path_dict
    elif p.is_magpie_run:
        p.pretrained_coefficients_path_dict = p.magpie_pretrained_coefficients_path_dict
    elif p.is_calibration_run:
        p.pretrained_coefficients_path_dict = 'use_generated'  # TODOO Make this point somehow to the generated one.

    p.static_regressor_paths = {}
    p.static_regressor_paths['sand_percent'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'sand_percent.tif')
    p.static_regressor_paths['silt_percent'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'silt_percent.tif')
    p.static_regressor_paths['soil_bulk_density'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'soil_bulk_density.tif')
    p.static_regressor_paths['soil_cec'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'soil_cec.tif')
    p.static_regressor_paths['soil_organic_content'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'soil_organic_content.tif')
    p.static_regressor_paths['strict_pa'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'strict_pa.tif')
    p.static_regressor_paths['temperature_c'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'temperature_c.tif')
    p.static_regressor_paths['travel_time_to_market_mins'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'travel_time_to_market_mins.tif')
    p.static_regressor_paths['wetlands_binary'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'wetlands_binary.tif')
    p.static_regressor_paths['alt_m'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'alt_m.tif')
    p.static_regressor_paths['carbon_above_ground_mg_per_ha_global'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'carbon_above_ground_mg_per_ha_global.tif')
    p.static_regressor_paths['clay_percent'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'clay_percent.tif')
    p.static_regressor_paths['ph'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'ph.tif')
    p.static_regressor_paths['pop'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'pop.tif')
    p.static_regressor_paths['precip_mm'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'precip_mm.tif')

    p.global_esa_lulc_paths_by_year = {}
    p.global_esa_lulc_paths_by_year[p.training_start_year] = os.path.join(p.base_data_dir, "lulc", "esa", "lulc_esa_" + str(p.training_start_year) + '.tif')
    p.global_esa_lulc_paths_by_year[p.base_year] = os.path.join(p.base_data_dir, "lulc", "esa", "lulc_esa_" + str(p.base_year) + '.tif')

    p.training_start_year_lulc_path = p.global_esa_lulc_paths_by_year[p.training_start_year]
    p.training_end_year_lulc_path = p.global_esa_lulc_paths_by_year[p.base_year]
    p.base_year_lulc_path = p.global_esa_lulc_paths_by_year[p.base_year]

    p.global_esa_seals7_lulc_paths_by_year = {}

    # START HERE, finish adding seals7 base data to google cloud andor fixing the paths.
    p.global_esa_seals7_lulc_paths_by_year[p.training_start_year] = os.path.join(p.base_data_dir, "lulc", "esa", "seals7", "lulc_esa_seals7_" + str(p.training_start_year) + '.tif')
    p.global_esa_seals7_lulc_paths_by_year[p.base_year] = os.path.join(p.base_data_dir, "lulc", "esa", "seals7", "lulc_esa_seals7_" + str(p.base_year) + '.tif')

    p.training_start_year_seals7_lulc_path = p.global_esa_seals7_lulc_paths_by_year[p.training_start_year]
    p.training_end_year_seals7_lulc_path = p.global_esa_seals7_lulc_paths_by_year[p.base_year]
    p.base_year_seals7_lulc_path = p.global_esa_seals7_lulc_paths_by_year[p.base_year]

    p.base_year_lulc_path = p.global_esa_lulc_paths_by_year[p.base_year]

    # SEALS results will be tiled on top of output_base_map_path, filling in areas potentially outside of the zones run (e.g., filling in small islands that were skipped_
    p.base_year_simplified_lulc_path = p.global_esa_seals7_lulc_paths_by_year[p.base_year]
    p.lulc_training_start_year_10sec_path = p.global_esa_seals7_lulc_paths_by_year[p.base_year]
    p.output_base_map_path = p.base_year_simplified_lulc_path

    if p.test_mode:
        p.stitch_tiles_to_global_basemap = 0
        if p.is_gtap1_run:
            run_1deg_subset = 1
            run_5deg_subset = 0
            magpie_subset = 0
        elif p.is_magpie_run:
            run_1deg_subset = 0
            run_5deg_subset = 0
            magpie_subset = 1
        elif p.is_calibration_run:
            run_1deg_subset = 1
            run_5deg_subset = 0
            magpie_subset = 0

    else:
        p.stitch_tiles_to_global_basemap = 1
        run_1deg_subset = 0
        run_5deg_subset = 0
        magpie_subset = 0


    p.subset_of_blocks_to_run = None
    # # If a a subset is defined, set its tiles here.
    # if run_1deg_subset:
    #     if p.processing_block_size == 1.0:
    #     p.processing_block_size = 1.0  # arcdegrees
    #     p.subset_of_blocks_to_run = [15526]  # mn. Has urban displaced by natural error.
    #     p.subset_of_blocks_to_run = [15708]  # slightly more representative  yet heterogenous zone.
    #     p.subset_of_blocks_to_run = [15526, 15708]  # combined. # DEFINED GLOBALLY
    #     p.subset_of_blocks_to_run = [0, 1, 2]  # Now defined wrt clipped aoi
    #
    #     # p.subset_of_blocks_to_run = [
    #     #     15526, 15526 + 180 * 1, 15526 + 180 * 2,
    #     #     15527, 15527 + 180 * 1, 15527 + 180 * 2,
    #     #     15528, 15528 + 180 * 1, 15528 + 180 * 2,
    #     # ]  # 3x3 mn tiles
    #     p.force_to_global_bb = False
    # elif run_5deg_subset:
    #     p.processing_block_size = 5.0  # arcdegrees
    #     p.subset_of_blocks_to_run = [476, 476 + 1 + (36 * 2), 476 + 3 + (36 * 4), 476 + 9 + (36 * 8), 476 + 1 + (36 * 25)]  # Montana
    #     p.force_to_global_bb = False
    # elif magpie_subset:
    #     p.processing_block_size = 5.0  # arcdegrees
    #     # p.subset_of_blocks_to_run = [476, 476 + 1 + (36 * 2)]
    #     # p.subset_of_blocks_to_run = [476 + 9 + (36 * 8)]  # Montana
    #     p.subset_of_blocks_to_run = [476]
    #     p.force_to_global_bb = False
    # else:
    #     p.subset_of_blocks_to_run = None
    #     p.processing_block_size = 5.0  # arcdegrees
    #     p.force_to_global_bb = True

    # Define which paths need to be in the base_data. Missing paths will be downloaded.
    p.required_base_data_paths = {}
    p.required_base_data_paths['global_countries_iso3_path'] = p.countries_iso3_path

    p.required_base_data_paths['coarse_state_paths'] = p.scenario_coarse_state_paths
    p.required_base_data_paths['pyramids'] = os.path.join(p.base_data_dir, 'pyramids', 'ha_per_cell_300sec.tif')
    p.required_base_data_paths['pyramids2'] = os.path.join(p.base_data_dir, 'pyramids', 'ha_per_cell_900sec.tif')  # TODOO I made an idiotic choice to make this a nested dict which requires unique ids.... Eventually i have this flatten to a list but here i need to give it a temporary unique index.
    p.required_base_data_paths['pyramids3'] = os.path.join(p.base_data_dir, 'pyramids', 'ha_per_cell_10sec.tif')  # TODOO I made an idiotic choice to make this a nested dict which requires unique ids.... Eventually i have this flatten to a list but here i need to give it a temporary unique index.
    p.required_base_data_paths['pyramids4'] = p.match_paths[3600.0]  # TODOO I made an idiotic choice to make this a nested dict which requires unique ids.... Eventually i have this flatten to a list but here i need to give it a temporary unique index.
    p.required_base_data_paths['pretrained_coefficients_paths'] = p.pretrained_coefficients_path_dict

    p.required_base_data_paths['gtap37_aez18_path'] = p.gtap37_aez18_input_vector_path

    if p.is_calibration_run:
        p.required_base_data_paths['static_regressor_paths'] = p.static_regressor_paths

        p.required_base_data_paths['training_start_year_lulc_path'] = p.training_start_year_lulc_path
        p.required_base_data_paths['training_end_year_lulc_path'] = p.training_end_year_lulc_path
        p.required_base_data_paths['base_year_lulc_path'] = p.base_year_lulc_path

        p.required_base_data_paths['training_start_year_seals7_lulc_path'] = p.training_start_year_seals7_lulc_path
        p.required_base_data_paths['training_end_year_seals7_lulc_path'] = p.training_end_year_seals7_lulc_path
        p.required_base_data_paths['base_year_seals7_lulc_path'] = p.base_year_seals7_lulc_path

    # Sometimes runs fail mid run. This checks for that and picks up where there is a completed file for that zone.
    p.skip_created_downscaling_zones = 1





    p.project_aoi_task = p.add_task(project_aoi)
    # project_aoi(p)

    # TODOO Make this based on a run parameter
    if p.run_type == 'just_figs':
        build_just_figs_task_tree(p)
    elif p.run_type == 'complete_run':
        build_complete_run_task_tree(p)
    else:
        raise NameError('No valid run_type was set.')
        # build_allocation_run_task_tree(p)

    p.execute()




def initialize_paths(p):
    p.combined_block_lists_paths = None # This will be smartly determined in either calibration or allocation

    p.write_global_lulc_seals7_scenarios_overview_and_tifs = True
    p.write_global_lulc_esa_scenarios_overview_and_tifs = True

    p.countries_iso3_path = os.path.join(p.base_data_dir, 'pyramids', 'countries_iso3.gpkg'
                                                                      '')
    # To easily convert between per-ha and per-cell terms, these very accurate ha_per_cell maps are defined.
    p.ha_per_cell_10sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_10sec.tif")
    p.ha_per_cell_300sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_300sec.tif")
    p.ha_per_cell_900sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_900sec.tif")
    p.ha_per_cell_1800sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_1800sec.tif")
    p.ha_per_cell_3600sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_3600sec.tif")

    p.ha_per_cell_paths = {}
    p.ha_per_cell_paths[10.0] = p.ha_per_cell_10sec_path
    p.ha_per_cell_paths[300.0] = p.ha_per_cell_300sec_path
    p.ha_per_cell_paths[900.0] = p.ha_per_cell_900sec_path
    p.ha_per_cell_paths[1800.0] = p.ha_per_cell_1800sec_path
    p.ha_per_cell_paths[3600.0] = p.ha_per_cell_3600sec_path

    # The ha per cell paths also can be used when writing new tifs as the match path.
    p.match_10sec_path = p.ha_per_cell_10sec_path
    p.match_300sec_path = p.ha_per_cell_300sec_path
    p.match_900sec_path = p.ha_per_cell_900sec_path
    p.match_1800sec_path = p.ha_per_cell_1800sec_path
    p.match_3600sec_path = p.ha_per_cell_3600sec_path

    p.match_paths = {}
    p.match_paths[10.0] = p.match_10sec_path
    p.match_paths[300.0] = p.match_300sec_path
    p.match_paths[900.0] = p.match_900sec_path
    p.match_paths[1800.0] = p.match_1800sec_path
    p.match_paths[3600.0] = p.match_3600sec_path

    p.match_float_paths = p.match_paths.copy()


    p.ha_per_cell_column_10sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_10sec.tif")
    p.ha_per_cell_column_300sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_300sec.tif")
    p.ha_per_cell_column_900sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_900sec.tif")
    p.ha_per_cell_column_1800sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_1800sec.tif")
    p.ha_per_cell_column_3600sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_3600sec.tif")

    p.ha_per_cell_column_paths = {}
    p.ha_per_cell_column_paths[10.0] = p.ha_per_cell_column_10sec_path
    p.ha_per_cell_column_paths[300.0] = p.ha_per_cell_column_300sec_path
    p.ha_per_cell_column_paths[900.0] = p.ha_per_cell_column_900sec_path
    p.ha_per_cell_column_paths[1800.0] = p.ha_per_cell_column_1800sec_path
    p.ha_per_cell_column_paths[3600.0] = p.ha_per_cell_column_3600sec_path

    p.luh_data_dir = os.path.join(p.base_data_dir, 'luh2', 'raw_data')

    p.luh_scenario_states_paths = {}
    p.luh_scenario_states_paths['rcp26_ssp1'] = os.path.join(p.luh_data_dir, 'rcp26_ssp1', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp126-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp34_ssp4'] = os.path.join(p.luh_data_dir, 'rcp34_ssp4', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-GCAM-ssp434-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp45_ssp2'] = os.path.join(p.luh_data_dir, 'rcp45_ssp2', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp60_ssp4'] = os.path.join(p.luh_data_dir, 'rcp60_ssp4', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-GCAM-ssp460-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp70_ssp3'] = os.path.join(p.luh_data_dir, 'rcp70_ssp3', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-AIM-ssp370-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp85_ssp5'] = os.path.join(p.luh_data_dir, 'rcp85_ssp5', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MAGPIE-ssp585-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['historical'] = os.path.join(p.luh_data_dir, 'historical', r"states.nc")

    p.ha_per_cell_column_10sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_10sec.tif")
    p.ha_per_cell_column_300sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_300sec.tif")
    p.ha_per_cell_column_900sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_900sec.tif")
    p.ha_per_cell_column_1800sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_1800sec.tif")

    p.visualization_base_path = os.path.join(p.model_base_data_dir, "GTAP_Visualizations_Data")
    p.visualization_base_path = os.path.join(p.model_base_data_dir, "GTAP_Visualizations_Data")
    # p.visualization_base_path = os.path.join(p.script_dir, "visualization")
    p.visualization_config_file_path = os.path.join(p.model_base_data_dir, 'GTAP_Visualizations_Data', 'default_config\config.yaml')



def project_aoi(p):
    #START HERE: Working but still fails to link correctly at C:\Files\Research\base_data_jaj_workstation2022\seals\static_regressors when loading CSV of paths for allocation.
    # also bring the tiles files back into project_aoi and clarify that it's a project-level task.

    path_verbosity = False
    if p.run_this:
        if isinstance(p.aoi, str):

            if p.aoi == 'global':
                p.aoi_path = p.countries_iso3_path
                # p.aoi_path = p.DataRef(p.countries_iso3_path, graft_dir=graft_dir, verbose=path_verbosity).path
                p.aoi_label = 'global'
                p.bb_exact = hb.global_bounding_box
                p.bb = p.bb_exact

                p.aoi_ha_per_cell_coarse_path = p.ha_per_cell_paths[p.coarse_resolution_arcseconds]
                p.aoi_ha_per_cell_fine_path = p.ha_per_cell_paths[p.fine_resolution_arcseconds]

            elif isinstance(p.aoi, str):
                if len(p.aoi) == 3: # Then it might be an ISO3 code. For now, assume so.
                    generated_path = os.path.join(p.cur_dir, 'aoi.gpkg')

                    p.aoi_path = generated_path
                    # p.aoi_path = p.DataRef(generated_path, graft_dir=graft_dir, verbose=path_verbosity).path
                    if not hb.path_exists(p.aoi_path):
                        hb.extract_features_in_shapefile_by_attribute(p.countries_iso3_path, p.aoi_path, 'iso3', p.aoi.upper())
                    p.aoi_label = p.aoi
                else:
                    p.aoi_path = p.aoi
                    # p.aoi_path = p.DataRef(p.aoi, graft_dir=graft_dir).path
                    if not hb.path_exists(p.aoi_path):
                        p.aoi_label = hb.file_root(p.aoi_path)

                p.bb_exact = hb.get_bounding_box(p.aoi_path)
                p.bb = hb.get_pyramid_compatible_bb_from_vector_and_resolution(p.aoi_path, p.coarse_resolution_arcseconds)

                p.aoi_ha_per_cell_fine_path = os.path.join(p.cur_dir, p.aoi, 'pyramids', 'aoi_ha_per_cell_fine.tif')
                if not hb.path_exists(p.aoi_ha_per_cell_fine_path):
                    hb.clip_raster_by_bb(p.ha_per_cell_10sec_path, p.bb, p.aoi_ha_per_cell_fine_path)

                p.aoi_ha_per_cell_coarse_path = os.path.join(p.cur_dir, p.aoi, 'pyramids', 'aoi_ha_per_cell_coarse.tif')
                if not hb.path_exists(p.aoi_ha_per_cell_coarse_path):
                    hb.clip_raster_by_bb(p.ha_per_cell_900sec_path, p.bb, p.aoi_ha_per_cell_coarse_path)

        else:
            raise NameError('Unable to interpret p.aoi.')
    else: ## CURRENTLY IDENTICAL to above because this is related to project initialization.
        if isinstance(p.aoi, str):

            if p.aoi == 'global':
                p.aoi_path = p.countries_iso3_path
                # p.aoi_path = p.DataRef(p.countries_iso3_path, graft_dir=graft_dir, verbose=path_verbosity).path
                p.aoi_label = 'global'
                p.bb_exact = hb.global_bounding_box
                p.bb = p.bb_exact

                p.aoi_ha_per_cell_coarse_path = p.ha_per_cell_paths[p.coarse_resolution_arcseconds]
                p.aoi_ha_per_cell_fine_path = p.ha_per_cell_paths[p.fine_resolution_arcseconds]

            elif isinstance(p.aoi, str):
                if len(p.aoi) == 3:  # Then it might be an ISO3 code. For now, assume so.
                    generated_path = os.path.join(p.cur_dir, 'aoi.gpkg')

                    p.aoi_path = generated_path
                    # p.aoi_path = p.DataRef(generated_path, graft_dir=graft_dir, verbose=path_verbosity).path
                    if not hb.path_exists(p.aoi_path):
                        hb.extract_features_in_shapefile_by_attribute(p.countries_iso3_path, p.aoi_path, 'iso3', p.aoi.upper())
                    p.aoi_label = p.aoi
                else:
                    p.aoi_path = p.aoi
                    # p.aoi_path = p.DataRef(p.aoi, graft_dir=graft_dir).path
                    if not hb.path_exists(p.aoi_path):
                        p.aoi_label = hb.file_root(p.aoi_path)

                p.bb_exact = hb.get_bounding_box(p.aoi_path)
                p.bb = hb.get_pyramid_compatible_bb_from_vector_and_resolution(p.aoi_path, p.coarse_resolution_arcseconds)

                p.aoi_ha_per_cell_fine_path = os.path.join(p.cur_dir, p.aoi, 'pyramids', 'aoi_ha_per_cell_fine.tif')
                if not hb.path_exists(p.aoi_ha_per_cell_fine_path):
                    hb.clip_raster_by_bb(p.ha_per_cell_10sec_path, p.bb, p.aoi_ha_per_cell_fine_path)

                p.aoi_ha_per_cell_coarse_path = os.path.join(p.cur_dir, p.aoi, 'pyramids', 'aoi_ha_per_cell_coarse.tif')
                if not hb.path_exists(p.aoi_ha_per_cell_coarse_path):
                    hb.clip_raster_by_bb(p.ha_per_cell_900sec_path, p.bb, p.aoi_ha_per_cell_coarse_path)

def build_complete_run_task_tree(p):

    ## ADD TASKS to project_flow task tree, then below set if they should run and/or be skipped if existing.
    p.luh2_extraction = p.add_task(seals_process_luh2.luh2_extraction,                                                      run=1, skip_if_dir_exists=0)
    p.luh2_difference_from_base_year = p.add_task(seals_process_luh2.luh2_difference_from_base_year,                        run=1, skip_if_dir_exists=0)
    p.luh2_as_seals7_proportion = p.add_task(seals_process_luh2.luh2_as_seals7_proportion,                                  run=1, skip_if_dir_exists=0)
    p.seals7_difference_from_base_year = p.add_task(seals_process_luh2.seals7_difference_from_base_year,                    run=1, skip_if_dir_exists=0)
    p.available_land_task = p.add_task(gtap_invest_main.available_land,                                                     run=0, skip_if_dir_exists=1)
    p.gtap1_aez_task = p.add_task(gtap_invest_main.gtap1_aez,                                                               run=1, skip_if_dir_exists=1)
    p.luh_projections_by_region_aez_task = p.add_task(gtap_invest_main.luh_projections_by_region_aez,                       run=1, skip_if_dir_exists=0)
    p.gtap_results_joined_with_luh_change_task = p.add_task(gtap_invest_main.gtap_results_joined_with_luh_change,           run=1, skip_if_dir_exists=0)
    p.pes_policy_identification_task = p.add_task(gtap_invest_main.pes_policy_identification,                               run=1, skip_if_dir_exists=0)
    p.pes_policy_endogenous_land_shock_task = p.add_task(gtap_invest_main.pes_policy_endogenous_land_shock,                 run=1, skip_if_dir_exists=0)
    p.protect_30_by_30_endogenous_land_shock_task = p.add_task(gtap_invest_main.protect_30_by_30_endogenous_land_shock,     run=1, skip_if_dir_exists=1)  # VERY SLOW. # TODOO Make this draw from base data.

    # SEALS tasks
    p.allocation_task = p.add_iterator(seals_main.allocation, run_in_parallel=False,                                        run=1, skip_if_dir_exists=0)  # CAREFUL Make these all change together.
    p.allocation_zones_task = p.add_iterator(seals_main.allocation_zones, parent=p.allocation_task, run_in_parallel=True,   run=1, skip_if_dir_exists=0)
    p.prepare_lulc_task = p.add_task(seals_main.prepare_lulc, parent=p.allocation_zones_task,                               run=1, skip_if_dir_exists=0)
    p.allocation_task = p.add_task(seals_main.allocation, parent=p.allocation_zones_task,                                   run=1, skip_if_dir_exists=0)

    # Re-aggregation of SEALS
    p.stitched_lulc_simplified_scenarios_task = p.add_task(seals_main.stitched_lulc_simplified_scenarios,                   run=1, skip_if_dir_exists=0)
    p.stitched_lulc_esa_scenarios_task = p.add_task(seals_main.stitched_lulc_esa_scenarios,                                 run=1, skip_if_dir_exists=0)

    # GTAP2 tasks
    p.land_cover_change_analysis_task = p.add_task(gtap_invest_main.land_cover_change_analysis,                             run=1, skip_if_dir_exists=0) # Not sure if needed
    p.carbon_biophysical_task = p.add_task(gtap_invest_main.carbon_biophysical,                                             run=1, skip_if_dir_exists=0)
    p.carbon_shock_task = p.add_task(gtap_invest_main.carbon_shock,                                                         run=1, skip_if_dir_exists=0)
    p.water_yield_biophysical_task = p.add_task(gtap_invest_main.water_yield_biophysical,                                   run=1, skip_if_dir_exists=0)
    p.water_yield_shock_task = p.add_task(gtap_invest_main.water_yield_shock,                                               run=1, skip_if_dir_exists=0)
    p.pollination_biophysical_task = p.add_task(gtap_invest_main.pollination_biophysical,                                   run=1, skip_if_dir_exists=0)
    p.pollination_shock_task = p.add_task(gtap_invest_main.pollination_shock,                                               run=1, skip_if_dir_exists=0)
    p.marine_fisheries_shock_task = p.add_task(gtap_invest_main.marine_fisheries_shock,                                     run=1, skip_if_dir_exists=1) # long filename error
    p.coastal_protection_biophysical_task = p.add_task(gtap_invest_main.coastal_protection_biophysical,                     run=1, skip_if_dir_exists=1)
    p.coastal_protection_shock_task = p.add_task(gtap_invest_main.coastal_protection_shock,                                 run=1, skip_if_dir_exists=1)
    p.gtap2_combined_shockfile_task = p.add_task(gtap_invest_main.gtap2_combined_shockfile,                                 run=1, skip_if_dir_exists=0)
    p.gtap2_process_shockfiles_task = p.add_task(gtap_invest_main.gtap2_process_shockfiles,                                 run=1, skip_if_dir_exists=0)
    p.gtap2_shockfile_analysis_task = p.add_task(gtap_invest_main.gtap2_shockfile_analysis,                                 run=1, skip_if_dir_exists=0)
    p.gtap2_aez_task = p.add_task(gtap_invest_main.gtap2_aez,                                                               run=1, skip_if_dir_exists=0)
    p.gtap2_extracts_from_solution_task = p.add_task(gtap_invest_main.gtap2_extracts_from_solution,                         run=1, skip_if_dir_exists=0)
    p.gtap2_valdation_extracts_from_solution_task = p.add_task(gtap_invest_main.gtap2_valdation_extracts_from_solution,     run=1, skip_if_dir_exists=0) # Not updated
    p.gtap2_results_as_tables_task = p.add_task(gtap_invest_main.gtap2_results_as_tables,                                   run=1, skip_if_dir_exists=0)
    p.output_visualizations_task = p.add_task(gtap_invest_main.output_visualizations,                                       run=1, skip_if_dir_exists=0)
    p.output_visualizations_secondary_task = p.add_task(gtap_invest_main.output_visualizations_secondary,                   run=1, skip_if_dir_exists=0)
    p.output_visualizations_tertiary_task = p.add_task(gtap_invest_main.output_visualizations_tertiary,                     run=1, skip_if_dir_exists=0)
    p.chris_style_output_visualizations_task = p.add_task(visualization.chris_style_output_visualizations,                  run=0, skip_if_dir_exists=0)


    p.execute()



def build_just_figs_task_tree(p):

    ## ADD TASKS to project_flow task tree, then below set if they should run and/or be skipped if existing.
    p.luh2_extraction = p.add_task(seals_process_luh2.luh2_extraction,                                                      run=0, skip_if_dir_exists=0)
    p.luh2_difference_from_base_year = p.add_task(seals_process_luh2.luh2_difference_from_base_year,                        run=0, skip_if_dir_exists=0)
    p.luh2_as_seals7_proportion = p.add_task(seals_process_luh2.luh2_as_seals7_proportion,                                  run=0, skip_if_dir_exists=0)
    p.seals7_difference_from_base_year = p.add_task(seals_process_luh2.seals7_difference_from_base_year,                    run=0, skip_if_dir_exists=0)
    p.available_land_task = p.add_task(gtap_invest_main.available_land,                                                     run=0, skip_if_dir_exists=1)
    p.gtap1_aez_task = p.add_task(gtap_invest_main.gtap1_aez,                                                               run=0, skip_if_dir_exists=1)
    p.luh_projections_by_region_aez_task = p.add_task(gtap_invest_main.luh_projections_by_region_aez,                       run=0, skip_if_dir_exists=0)
    p.gtap_results_joined_with_luh_change_task = p.add_task(gtap_invest_main.gtap_results_joined_with_luh_change,           run=0, skip_if_dir_exists=0)
    p.pes_policy_identification_task = p.add_task(gtap_invest_main.pes_policy_identification,                               run=0, skip_if_dir_exists=0)
    p.pes_policy_endogenous_land_shock_task = p.add_task(gtap_invest_main.pes_policy_endogenous_land_shock,                 run=0, skip_if_dir_exists=0)
    p.protect_30_by_30_endogenous_land_shock_task = p.add_task(gtap_invest_main.protect_30_by_30_endogenous_land_shock,     run=0, skip_if_dir_exists=1)  # VERY SLOW. # TODOO Make this draw from base data.

    # # SEALS tasks
    # p.allocation_task = p.add_iterator(seals_main.allocation, run_in_parallel=False,                                        run=0, skip_if_dir_exists=0)  # CAREFUL Make these all change together.
    # p.allocation_zones_task = p.add_iterator(seals_main.allocation_zones, parent=p.allocation_task, run_in_parallel=True,   run=0, skip_if_dir_exists=0)
    # p.prepare_lulc_task = p.add_task(seals_main.prepare_lulc, parent=p.allocation_zones_task,                               run=0, skip_if_dir_exists=0)
    # p.allocation_task = p.add_task(seals_main.allocation, parent=p.allocation_zones_task,                                   run=0, skip_if_dir_exists=0)

    # Re-aggregation of SEALS
    p.stitched_lulc_simplified_scenarios_task = p.add_task(seals_main.stitched_lulc_simplified_scenarios,                   run=0, skip_if_dir_exists=0)
    p.stitched_lulc_esa_scenarios_task = p.add_task(seals_main.stitched_lulc_esa_scenarios,                                 run=0, skip_if_dir_exists=0)

    # GTAP2 tasks
    p.land_cover_change_analysis_task = p.add_task(gtap_invest_main.land_cover_change_analysis,                             run=0, skip_if_dir_exists=0) # Not sure if needed
    p.carbon_biophysical_task = p.add_task(gtap_invest_main.carbon_biophysical,                                             run=0, skip_if_dir_exists=0)
    p.carbon_shock_task = p.add_task(gtap_invest_main.carbon_shock,                                                         run=0, skip_if_dir_exists=0)
    p.water_yield_biophysical_task = p.add_task(gtap_invest_main.water_yield_biophysical,                                   run=0, skip_if_dir_exists=0)
    p.water_yield_shock_task = p.add_task(gtap_invest_main.water_yield_shock,                                               run=0, skip_if_dir_exists=0)
    p.pollination_biophysical_task = p.add_task(gtap_invest_main.pollination_biophysical,                                   run=0, skip_if_dir_exists=0)
    p.pollination_shock_task = p.add_task(gtap_invest_main.pollination_shock,                                               run=0, skip_if_dir_exists=0)
    p.marine_fisheries_shock_task = p.add_task(gtap_invest_main.marine_fisheries_shock,                                     run=0, skip_if_dir_exists=1) # long filename error
    p.coastal_protection_biophysical_task = p.add_task(gtap_invest_main.coastal_protection_biophysical,                     run=0, skip_if_dir_exists=1)
    p.coastal_protection_shock_task = p.add_task(gtap_invest_main.coastal_protection_shock,                                 run=0, skip_if_dir_exists=1)
    p.gtap2_combined_shockfile_task = p.add_task(gtap_invest_main.gtap2_combined_shockfile,                                 run=0, skip_if_dir_exists=0)
    p.gtap2_process_shockfiles_task = p.add_task(gtap_invest_main.gtap2_process_shockfiles,                                 run=0, skip_if_dir_exists=0)
    p.gtap2_shockfile_analysis_task = p.add_task(gtap_invest_main.gtap2_shockfile_analysis,                                 run=0, skip_if_dir_exists=0)
    p.gtap2_aez_task = p.add_task(gtap_invest_main.gtap2_aez,                                                               run=0, skip_if_dir_exists=0)
    p.gtap2_extracts_from_solution_task = p.add_task(gtap_invest_main.gtap2_extracts_from_solution,                         run=1, skip_if_dir_exists=0)
    p.gtap2_valdation_extracts_from_solution_task = p.add_task(gtap_invest_main.gtap2_valdation_extracts_from_solution,     run=1, skip_if_dir_exists=0) # Not updated
    p.gtap2_results_as_tables_task = p.add_task(gtap_invest_main.gtap2_results_as_tables,                                   run=1, skip_if_dir_exists=0)
    p.output_visualizations_task = p.add_task(gtap_invest_main.output_visualizations,                                       run=1, skip_if_dir_exists=0)
    p.output_visualizations_secondary_task = p.add_task(gtap_invest_main.output_visualizations_secondary,                   run=1, skip_if_dir_exists=0)
    p.output_visualizations_tertiary_task = p.add_task(gtap_invest_main.output_visualizations_tertiary,                     run=1, skip_if_dir_exists=0)
    p.chris_style_output_visualizations_task = p.add_task(visualization.chris_style_output_visualizations,                  run=0, skip_if_dir_exists=0)


    p.execute()
