import os
import hazelbean as hb

main = ''
if __name__ == '__main__':

    #### These next few lines should be the only computer-specific things to set. Everything is relative to these (or the source code dir)

    # A ProjectFlow object is created from the Hazelbean library to organize directories and enable parallel processing.
    # project-level variables are assigned as attributes to the p object (such as in p.base_data_dir = ... below)
    p = hb.ProjectFlow('..\\..\\projects\\gtap_invest_pnas')

    # Set the base data dir. The model will check here to see if it has everything it needs to run.
    # If anything is missing, it will download it. You can use the same base_data dir across multiple projects.
    p.base_data_dir = os.path.join('C:\\', 'Users', 'jajohns', 'Files', 'Research', 'base_data') # This is where the minimum set of downloaded files goes.

    ### DISABLED FOR NOW
    # In order for GTAP-InVEST to download using the google_cloud_api service, you need to have a valid credentials JSON file.
    # Identify its location here. If you don't have one, email jajohns@umn.edu. The data are freely available but are very, very large
    # (and thus expensive to host), so I limit access via credentials.
    # p.data_credentials_path = '..\\api_key_credentials.json'

    ### DISABLED FOR NOW
    # There are different versions of the base_data in gcloud, but best open-source one is 'seals_public_2022-03-01'
    # p.input_bucket_name = 'seals_public_2022-03-01'

    # Set the area of interest. If set as a country-ISO3 code, all data will be generated based
    # that countries boundaries (as defined in the base data). Other options include setting it to
    # 'global' or a specific shapefile.
    p.aoi = 'global'

    # Set the training start year and end year. These years will be used for calibrating the model. Once calibrated, project forward
    # from the base_year (which could be the same as the training_end_year but not necessarily).
    p.training_start_year = 2000
    p.training_end_year = 2014
    p.base_year = 2014


    ### Econ model run type
    # In order to apply this code to the magpie model, I set this option to either
    # use the GTAP-shited LUH data (as was done in the WB feedback model)
    # or to instead use the outputs of some other extraction functions with
    # no shifting logic. This could be scaled to different interfaces
    # when models have different input points.
    p.is_gtap1_run = True
    p.is_magpie_run = False
    p.is_calibration_run = False
    p.is_standard_seals_run = False

    # For GTAP-enabled runs, we project the economy from the latest GTAP reference year to the year in which a
    # policy is made so that we can apply the policy to a future date. Set that policy year here. (Only affects runs if p.is_gtap_run is True)
    if p.is_gtap1_run:
        p.policy_base_year = 2021
        p.base_years = [p.base_year, p.policy_base_year]
    elif p.is_magpie_run:
        p.base_years = [p.base_year]
    elif p.is_calibration_run:
        p.base_years = [p.base_year]

    # Define terminal year for simulation. This is only partly implemented but will be finished when switching to GTAP-AEZ-RD.
    p.scenario_years = [2030]

    # Define which meso-level LUC and Climate scenarios will be used.
    # Full set of options:  ['rcp26_ssp1', 'rcp34_ssp4', 'rcp45_ssp2', 'rcp60_ssp4', 'rcp70_ssp3', 'rcp85_ssp5'
    p.luh_scenario_labels = [
        'rcp45_ssp2',
    ]

    # SEALS has two resolutions: fine and coarse. In most applications, fine is 10 arcseconds (~300m at equator, based on ESACCI)
    # and coarse is based on IAM results that are 900 arcseconds (LUH2) or 1800 arcseconds (MAgPIE). Note that there is a coarser-yet
    # scale possible from e.g. GTAP-determined endogenous LUC. This is excluded in the base SEALS config.
    p.fine_resolution_arcseconds = 10.0 # MUST BE FLOAT
    p.coarse_resolution_arcseconds = 900.0 # MUST BE FLOAT

    # To run a much faster version for code-testing purposes, enable test_mode. Selects a much smaller set of scenarios and spatial tiles.
    p.test_mode = False

    # For the intitial magpie run, we enforced that the amount of ag land in ESA had to match that in Magpie. This enables that option.
    p.adjust_baseline_to_match_magpie_2015 = False # NYI in GTAP-InVEST


    ############# Below here shouldn't need editing, but it may explain what's happening ###########

    # Configure the logger that captures all the information generated.
    p.L = hb.get_logger('run_gtap_invest')

    p.L.info('Created ProjectFlow object at ' + p.project_dir + '\n    '
           'from script ' +  p.calling_script + '\n    '
           'with base_data set at ' + p.base_data_dir)

    import gtap_invest_main
    import seals_main
    import seals_process_luh2
    from gtap_invest.visualization import visualization



    # initialize and set all basic variables. Sadly this is still needed even for a SEALS run until it's extracted.
    gtap_invest_main.initialize_paths(p)


    p.adjust_baseline_to_match_magpie_2015 = False

    # Run configuration options
    p.num_workers = 14 # None sets it to max available.
    p.reporting_level = 3
    p.output_writing_level = 5 # >=2 writes chunk-baseline lulc
    p.build_overviews_and_stats = 0 # For later fast-viewing, this can be enabled to write ovr files and geotiff stats files.
    p.write_projected_coarse_change_chunks = 1 # in the SEALS allocation, for troubleshooting, it can be useful to see what was the coarse allocation input.

    # TASK to figure out: draw a task tree consistent both with magpie needing esa-magpie 2015 calibration AND
    # gtap needing an extra base-year of 2021 AND gtap being a 3-layer allocation with SSP2 (which should be made interchangeable with Magpie)



    # TODOO For magpie integration: make this an override.
    p.baseline_labels = ['baseline']
    p.baseline_coarse_state_paths = {}
    p.baseline_coarse_state_paths['baseline'] = {}
    p.baseline_coarse_state_paths['baseline'][p.base_year] = os.path.join(p.input_dir, "SSP2_BiodivPol_LPJmL5_2021-05-21_15.08.06", "cell.land_0.5_share_to_seals_SSP2_BiodivPol_LPJmL5.nc")

    # These are the POLICY scenarios. The model will iterate over these as well.
    p.gtap_combined_policy_scenario_labels = ['BAU', 'BAU_rigid', 'PESGC', 'SR_Land', 'PESLC', 'SR_RnD_20p', 'SR_Land_PESGC', 'SR_PESLC',  'SR_RnD_20p_PESGC', 'SR_RnD_20p_PESGC_30']
    p.gtap_collapse_scenarios = ['BAU', 'fishcol', 'polcol', 'amazoncol']
    # NOTE: in pnas version SR_RnD_PESLC fails to solve.
    # p.gtap_combined_policy_scenario_labels = ['BAU', 'BAU_rigid', 'PESGC', 'SR_Land', 'PESLC', 'SR_RnD_20p', 'SR_Land_PESGC', 'SR_PESLC',  'SR_RnD_20p_PESGC', 'SR_RnD_PESLC', 'SR_RnD_20p_PESGC_30']
    p.gtap_just_bau_label = ['BAU']
    p.gtap_just_pesgc_label = ['PESGC']
    p.gtap_bau_and_pesgc_label = ['BAU', 'PESGC']
    p.gtap_bau_and_combined_labels = ['BAU', 'SR_RnD_20p_PESGC']
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
        'SSP2_ClimPol_LPJmL5',
        # 'SSP2_NPI_base_LPJmL5',
    ]

    p.global_esa_lulc_paths_by_year = {}
    p.global_esa_lulc_paths_by_year[p.training_start_year] = os.path.join(p.base_data_dir, "lulc", "esa", "lulc_esa_" + str(p.training_start_year) + '.tif')
    p.global_esa_lulc_paths_by_year[p.base_year] = os.path.join(p.base_data_dir, "lulc", "esa", "lulc_esa_" + str(p.base_year) + '.tif')

    p.training_start_year_lulc_path = p.global_esa_lulc_paths_by_year[p.training_start_year]
    p.training_end_year_lulc_path = p.global_esa_lulc_paths_by_year[p.base_year]
    p.base_year_lulc_path = p.global_esa_lulc_paths_by_year[p.base_year]

    p.global_esa_seals7_lulc_paths_by_year = {}
    p.global_esa_seals7_lulc_paths_by_year[p.training_start_year] = os.path.join(p.base_data_dir, "lulc", "esa", "seals7", "lulc_esa_seals7_" + str(p.training_start_year) + '.tif')
    p.global_esa_seals7_lulc_paths_by_year[p.base_year] = os.path.join(p.base_data_dir, "lulc", "esa", "seals7", "lulc_esa_seals7_" + str(p.base_year) + '.tif')


    # The model relies on both ESACCI defined for the base year and a simplified LULC (7 classes) for the SEALS downscaling step, defined here).
    # p.base_year_lulc_path = os.path.join(hb.SEALS_BASE_DATA_DIR, 'lulc_esa', 'full', 'ESACCI-LC-L4-LCCS-Map-300m-P1Y-' + str(p.base_year) + '-v2.0.7.tif')
    p.lulc_training_start_year_10sec_path = p.global_esa_seals7_lulc_paths_by_year[p.training_start_year]
    p.base_year_simplified_lulc_path = p.global_esa_seals7_lulc_paths_by_year[p.base_year]


    # A bit of a diveation from SEALS, which gets this from the AOI
    p.bb = [-180.0, -90.0, 180.0, 90.0]

    p.luh_data_dir = os.path.join(p.base_data_dir, 'luh2', 'raw_data')

    p.luh_scenario_states_paths = {}
    p.luh_scenario_states_paths['rcp26_ssp1'] = os.path.join(p.luh_data_dir, 'rcp26_ssp1', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp126-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp34_ssp4'] = os.path.join(p.luh_data_dir, 'rcp34_ssp4', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-GCAM-ssp434-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp45_ssp2'] = os.path.join(p.luh_data_dir, 'rcp45_ssp2', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp60_ssp4'] = os.path.join(p.luh_data_dir, 'rcp60_ssp4', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-GCAM-ssp460-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp70_ssp3'] = os.path.join(p.luh_data_dir, 'rcp70_ssp3', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-AIM-ssp370-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp85_ssp5'] = os.path.join(p.luh_data_dir, 'rcp85_ssp5', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MAGPIE-ssp585-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['historical'] = os.path.join(p.luh_data_dir, 'historical', r"states.nc")




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
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'][2030]['PESGC'] = p.luh_scenario_states_paths['rcp45_ssp2']
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'][2050]['PESGC'] = p.luh_scenario_states_paths['rcp45_ssp2']

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

    if p.is_magpie_run:
        p.scenario_coarse_state_paths = p.magpie_scenario_coarse_state_paths
    elif p.is_gtap1_run:
        p.scenario_coarse_state_paths = p.gtap_scenario_coarse_state_paths
    elif p.is_calibration_run:
        p.scenario_coarse_state_paths = p.luh_scenario_coarse_state_paths

    # NOTE REALLY STUPIUD ERROR: Must have a BAU in the scenarios list or it fails.
    if p.test_mode:
        if p.is_magpie_run:
            p.policy_scenario_labels = p.magpie_test_policy_scenario_labels
        elif p.is_gtap1_run:
            p.policy_scenario_labels = p.gtap_bau_and_pesgc_label
        elif p.is_calibration_run:
            p.policy_scenario_labels = p.luh_labels
    else:
        if p.is_magpie_run:
            p.policy_scenario_labels = p.magpie_policy_scenario_labels
        elif p.is_gtap1_run:
            p.policy_scenario_labels = p.gtap_combined_policy_scenario_labels
        elif p.is_calibration_run:
            p.policy_scenario_labels = p.luh_labels

    if p.is_gtap1_run:
        # HACK, because I don't yet auto-generate the cmf files and other GTAP modelled inputs, and instead just take the files out of the zipfile Uris
        # provides, I still have to follow his naming scheme. This list comprehension converts a policy_scenario_label into a gtap1 or gtap2 label.
        p.gtap1_scenario_labels = [str(p.policy_base_year) + '_' + str(p.scenario_years[0])[2:] + '_' + i + '_noES' for i in p.policy_scenario_labels]
        p.gtap2_scenario_labels = [str(p.policy_base_year) + '_' + str(p.scenario_years[0])[2:] + '_' + i + '_allES' for i in p.policy_scenario_labels]



    # This is a zipfile I received from URIS that has all the packaged GTAP files ready to run. Extract these to a project dir.
    p.gtap_aez_invest_release_string = '04_20_2021_GTAP_AEZ_INVEST'
    p.gtap_aez_invest_zipfile_path = os.path.join(p.model_base_data_dir, 'gtap_aez_invest_releases', p.gtap_aez_invest_release_string + '.zip')
    p.gtap_aez_invest_proc_zipfile_path = os.path.join(p.model_base_data_dir, 'gtap_aez_invest_releases', '00_InVEST_proc.zip')
    p.gtap_aez_invest_code_dir = os.path.join(p.script_dir, 'gtap_aez', p.gtap_aez_invest_release_string)
    p.gtap_aez_invest_proc_code_dir = os.path.join(p.script_dir, 'gtap_aez', '00_InVEST_proc')

    # Associate each luh, year, and policy scenario with a set of seals input parameters. This can be used if, for instance, the policy you
    # are analyzing involves focusing land-use change into certain types of gridcells.
    p.gtap_pretrained_coefficients_path_dict = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'] = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030] = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2050] = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['BAU'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['BAU'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['BAU_rigid'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['BAU_rigid'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
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
    p.magpie_pretrained_coefficients_path_dict['baseline'][2015]['baseline'] = os.path.join(p.model_base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')

    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'] = {}
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050] = {}
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_BiodivPol_LPJmL5'] = os.path.join(p.model_base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_BiodivPol_ClimPol_LPJmL5'] = os.path.join(p.model_base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5'] = os.path.join(p.model_base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_ClimPol_LPJmL5'] = os.path.join(p.model_base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_NPI_base_LPJmL5'] = os.path.join(p.model_base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')



    # Provided by GTAP team.
    # TODOO This is still based on the file below, which was from Purdue. It is a vector of 300sec gridcells and should be replaced with continuous vectors
    p.gtap37_aez18_input_vector_path = os.path.join(p.model_base_data_dir, "region_boundaries\GTAPv10_AEZ18_37Reg.shp")

    # GTAP-InVEST has two resolutions: fine (based on ESACCI) and coarse (based on LUH), though actually the coarse does change when using magpie 30min.
    p.fine_resolution = hb.get_cell_size_from_path(p.match_10sec_path)
    p.coarse_resolution = hb.get_cell_size_from_path(p.scenario_coarse_state_paths[p.luh_scenario_labels[0]][p.scenario_years[0]][p.policy_scenario_labels[0]])
    p.coarse_arcseconds = hb.pyramid_compatible_resolution_to_arcseconds[p.coarse_resolution]

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


    p.coarse_ha_per_cell_path = p.ha_per_cell_paths[p.coarse_arcseconds]
    p.coarse_match_path = p.coarse_ha_per_cell_path
    p.coarse_match = hb.ArrayFrame(p.coarse_match_path)

    # Sometimes runs fail mid run. This checks for that and picks up where there is a completed file for that zone.
    p.skip_created_downscaling_zones = 1

    # The SEALS-simplified classes are defined here, which can be iterated over. We also define what classes are shifted by GTAP's endogenous land-calcualtion step.
    p.class_indices = [1, 2, 3, 4, 5] # These are the indices of classes THAT CAN EXPAND/CONTRACT
    p.nonchanging_class_indices = [6, 7] # These add other lulc classes that might have an effect on LUC but cannot change themselves (e.g. water, barren)
    p.regression_input_class_indices = p.class_indices + p.nonchanging_class_indices

    p.class_labels = ['urban', 'cropland', 'grassland', 'forest', 'nonforestnatural',]
    p.nonchanging_class_labels = ['water', 'barren_and_other']
    p.regression_input_class_labels = p.class_labels + p.nonchanging_class_labels

    p.shortened_class_labels = ['urban', 'crop', 'past', 'forest', 'other',]

    p.class_indices_that_differ_between_ssp_and_gtap = [2, 3, 4,]
    p.class_labels_that_differ_between_ssp_and_gtap = ['cropland', 'grassland', 'forest',]

    # A little awkward, but I used to use integers and list counting to keep track of the actual lulc class value. Now i'm making it expicit with dicts.
    p.class_indices_to_labels_correspondence = dict(zip(p.class_indices, p.class_labels))
    p.class_labels_to_indices_correspondence = dict(zip(p.class_labels, p.class_indices))

    # Used for (optional) calibration of seals.
    p.calibrate = False
    p.num_generations = 2

    if p.is_gtap1_run:
        p.pretrained_coefficients_path_dict = p.gtap_pretrained_coefficients_path_dict
    elif p.is_magpie_run:
        p.pretrained_coefficients_path_dict = p.magpie_pretrained_coefficients_path_dict
    elif p.is_calibration_run:
        p.pretrained_coefficients_path_dict = 'use_generated' # TODOO Make this point somehow to the generated one.


    # If calibrate of SEALS is done, here are some starting coefficient guesses to speed it up.
    p.spatial_regressor_coefficients_path = os.path.join(p.input_dir, "spatial_regressor_starting_coefficients.xlsx")

    p.static_regressor_paths = {}
    p.static_regressor_paths['sand_percent'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'sand_percent.tif')
    p.static_regressor_paths['silt_percent'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'silt_percent.tif')
    p.static_regressor_paths['soil_bulk_density'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'soil_bulk_density.tif')
    p.static_regressor_paths['soil_cec'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'soil_cec.tif')
    p.static_regressor_paths['soil_organic_content'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'soil_organic_content.tif')
    p.static_regressor_paths['strict_pa'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'strict_pa.tif')
    p.static_regressor_paths['temperature_c'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'temperature_c.tif')
    p.static_regressor_paths['travel_time_to_market_mins'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'travel_time_to_market_mins.tif')
    p.static_regressor_paths['wetlands_binary'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'wetlands_binary.tif')
    p.static_regressor_paths['alt_m'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'alt_m.tif')
    p.static_regressor_paths['carbon_above_ground_mg_per_ha_global'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'carbon_above_ground_mg_per_ha_global.tif')
    p.static_regressor_paths['clay_percent'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'clay_percent.tif')
    p.static_regressor_paths['ph'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'ph.tif')
    p.static_regressor_paths['pop'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'pop.tif')
    p.static_regressor_paths['precip_mm'] = os.path.join(p.model_base_data_dir, 'static_regressors', 'precip_mm.tif')

    p.global_esa_lulc_paths_by_year = {}
    p.global_esa_lulc_paths_by_year[p.training_start_year] = os.path.join(p.base_data_dir, "lulc", "esa", "lulc_esa_" + str(p.training_start_year) + '.tif' )
    p.global_esa_lulc_paths_by_year[p.base_year] = os.path.join(p.base_data_dir, "lulc", "esa", "lulc_esa_" + str(p.base_year) + '.tif' )

    p.global_esa_seals7_lulc_paths_by_year = {}
    p.global_esa_seals7_lulc_paths_by_year[p.training_start_year] = os.path.join(p.base_data_dir, "lulc", "esa", "seals7", "lulc_esa_seals7_" + str(p.training_start_year) + '.tif')
    p.global_esa_seals7_lulc_paths_by_year[p.base_year] = os.path.join(p.base_data_dir, "lulc", "esa", "seals7", "lulc_esa_seals7_" + str(p.base_year) + '.tif')


    # SEALS results will be tiled on top of output_base_map_path, filling in areas potentially outside of the zones run (e.g., filling in small islands that were skipped_
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

    # If a a subset is defined, set its tiles here.
    if run_1deg_subset:
        p.processing_block_size = 1.0  # arcdegrees
        p.subset_of_blocks_to_run = [15526]  # mn. Has urban displaced by natural error.
        p.subset_of_blocks_to_run = [15708]  # slightly more representative  yet heterogenous zone.
        p.subset_of_blocks_to_run = [15526, 15708]  # combined.

        # p.subset_of_blocks_to_run = [
        #     15526, 15526 + 180 * 1, 15526 + 180 * 2,
        #     15527, 15527 + 180 * 1, 15527 + 180 * 2,
        #     15528, 15528 + 180 * 1, 15528 + 180 * 2,
        # ]  # 3x3 mn tiles
        p.force_to_global_bb = False
    elif run_5deg_subset:
        p.processing_block_size = 5.0  # arcdegrees
        p.subset_of_blocks_to_run = [476, 476 + 1 + (36 * 2), 476 + 3 + (36 * 4), 476 + 9 + (36 * 8), 476 + 1 + (36 * 25)]  # Montana
        p.force_to_global_bb = False
    elif magpie_subset:
        p.processing_block_size = 5.0  # arcdegrees
        p.subset_of_blocks_to_run = [476, 476 + 1 + (36 * 2)]
        # p.subset_of_blocks_to_run = [476 + 9 + (36 * 8)]  # Montana
        # p.subset_of_blocks_to_run = [476]
        p.force_to_global_bb = False
    else:
        p.subset_of_blocks_to_run = None
        p.processing_block_size = 5.0  # arcdegrees
        p.force_to_global_bb = True

    # Sometimes runs fail mid run. This checks for that and picks up where there is a completed file for that zone.
    p.skip_created_downscaling_zones = 1

    # As it calibrates, optionally write each calibration allocation step
    p.write_calibration_generation_arrays = 1

    ## ADD TASKS to project_flow task tree, then below set if they should run and/or be skipped if existing.
    p.luh2_extraction = p.add_task(seals_process_luh2.luh2_extraction,                                                      run=0, skip_if_dir_exists=0)
    p.luh2_difference_from_base_year = p.add_task(seals_process_luh2.luh2_difference_from_base_year,                        run=0, skip_if_dir_exists=0)
    p.luh2_as_seals7_proportion = p.add_task(seals_process_luh2.luh2_as_seals7_proportion,                                  run=0, skip_if_dir_exists=0)
    p.seals7_difference_from_base_year = p.add_task(seals_process_luh2.seals7_difference_from_base_year,                    run=0, skip_if_dir_exists=0)
    p.align_land_available_inputs_task = p.add_task(gtap_invest_main.align_land_available_inputs,                           run=0, skip_if_dir_exists=0)
    p.available_land_task = p.add_task(gtap_invest_main.available_land,                                                     run=0, skip_if_dir_exists=1)
    p.gtap1_aez_task = p.add_task(gtap_invest_main.gtap1_aez,                                                               run=0, skip_if_dir_exists=1)
    p.gtap1_extracts_from_solution_task = p.add_task(gtap_invest_main.gtap1_extracts_from_solution,                         run=0, skip_if_dir_exists=0)
    p.luh_projections_by_region_aez_task = p.add_task(gtap_invest_main.luh_projections_by_region_aez,                       run=0, skip_if_dir_exists=0)
    p.gtap_results_joined_with_luh_change_task = p.add_task(gtap_invest_main.gtap_results_joined_with_luh_change,           run=0, skip_if_dir_exists=0)
    p.pes_policy_identification_task = p.add_task(gtap_invest_main.pes_policy_identification,                               run=0, skip_if_dir_exists=0)
    p.pes_policy_endogenous_land_shock_task = p.add_task(gtap_invest_main.pes_policy_endogenous_land_shock,                 run=0, skip_if_dir_exists=0)
    p.protect_30_by_30_endogenous_land_shock_task = p.add_task(gtap_invest_main.protect_30_by_30_endogenous_land_shock,     run=0, skip_if_dir_exists=1)  # VERY SLOW. # TODOO Make this draw from base data.

    # SEALS tasks
    p.scenarios_task = p.add_iterator(seals_main.allocations, run_in_parallel=False,                                        run=0, skip_if_dir_exists=0)  # CAREFUL Make these all change together.
    p.allocation_zones_task = p.add_iterator(seals_main.allocation_zones, parent=p.scenarios_task, run_in_parallel=True,    run=0, skip_if_dir_exists=0)
    p.prepare_lulc_task = p.add_task(seals_main.prepare_lulc, parent=p.allocation_zones_task,                               run=0, skip_if_dir_exists=0)
    p.allocation_task = p.add_task(seals_main.allocation, parent=p.allocation_zones_task,                                   run=0, skip_if_dir_exists=0)

    # Re-aggregation of SEALS
    p.stitched_lulcs_task = p.add_task(seals_main.stitched_lulc_simplified_scenarios,                                      run=0, skip_if_dir_exists=0)
    p.map_esa_simplified_back_to_esa_task = p.add_task(seals_main.stitched_lulc_esa_scenarios,                             run=0, skip_if_dir_exists=0)

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
    p.gtap2_extracts_from_solution_task = p.add_task(gtap_invest_main.gtap2_extracts_from_solution,                         run=0, skip_if_dir_exists=0)
    p.gtap2_valdation_extracts_from_solution_task = p.add_task(gtap_invest_main.gtap2_valdation_extracts_from_solution,     run=0, skip_if_dir_exists=0) # Not updated
    p.gtap2_results_as_tables_task = p.add_task(gtap_invest_main.gtap2_results_as_tables,                                   run=1, skip_if_dir_exists=0)
    p.output_visualizations_task = p.add_task(gtap_invest_main.output_visualizations,                                       run=1, skip_if_dir_exists=0)
    p.output_visualizations_secondary_task = p.add_task(gtap_invest_main.output_visualizations_secondary,                   run=1, skip_if_dir_exists=0)
    p.output_visualizations_tertiary_task = p.add_task(gtap_invest_main.output_visualizations_tertiary,                     run=1, skip_if_dir_exists=0)
    p.chris_style_output_visualizations_task = p.add_task(visualization.chris_style_output_visualizations,                  run=0, skip_if_dir_exists=0)


    p.execute()


    end = 1