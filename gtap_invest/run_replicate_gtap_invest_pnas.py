import os
import hazelbean as hb
import gtap_invest_initialize_project


main = ''
if __name__ == '__main__':

    #### These next three lines should be the only computer-specific things to set. Everything is relative to these (or the source code dir)

    # A ProjectFlow object is created from the Hazelbean library to organize directories and enable parallel processing.
    # project-level variables are assigned as attributes to the p object (such as in p.base_data_dir = ... below)
    p = hb.ProjectFlow(r'../../projects/run_test_gtap_invest_pnas')

    # Set the base data dir. The model will check here to see if it has everything it needs to run.
    # If anything is missing, it will download it. You can use the same base_data dir across multiple projects.
    p.base_data_dir = os.path.join('C:\\', 'Users', 'jajohns', 'Files', 'Research', 'base_data') # This is where the minimum set of downloaded files goes.

    # ### NYI
    # # In order for SEALS to download using the google_cloud_api service, you need to have a valid credentials JSON file.
    # # Identify its location here. If you don't have one, email jajohns@umn.edu. The data are freely available but are very, very large
    # # (and thus expensive to host), so I limit access via credentials.
    # p.data_credentials_path = '..\\api_key_credentials.json'
    #
    # # There are different versions of the base_data in gcloud, but best open-source one is 'seals_public_2022-03-01'
    # p.input_bucket_name = 'seals_public_2022-03-01'

    # Set the area of interest. If set as a country-ISO3 code, all data will be generated based
    # that countries boundaries (as defined in the base data). Other options include setting it to
    # 'global' or a specific shapefile, or iso3 code. Good small examples include RWA, BTN
    p.aoi = 'global'


    # Set the training start year and end year. These years will be used for calibrating the model. Once calibrated, project forward
    # from the base_year (which could be the same as the training_end_year but not necessarily).
    p.training_start_year = 2000
    p.training_end_year = 2014
    p.base_year = 2014

    # In order to apply this code to the magpie model, I set this option to either
    # use the GTAP-shited LUH data (as was done in the WB feedback model)
    # or to instead use the outputs of some other extraction functions with
    # no shifting logic. This could be scaled to different interfaces
    # when models have different input points.
    p.is_magpie_run = False
    p.is_gtap1_run = True
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
    p.luh_scenario_labels = [
        'rcp45_ssp2',
    ]
    # p.luh_scenario_labels = [
    #     'rcp26_ssp1',
    #     'rcp45_ssp2',
    #     'rcp70_ssp3',
    #     'rcp34_ssp4',
    #     'rcp60_ssp4',
    #     'rcp85_ssp5',
    # ]



    # SEALS has two resolutions: fine and coarse. In most applications, fine is 10 arcseconds (~300m at equator, based on ESACCI)
    # and coarse is based on IAM results that are 900 arcseconds (LUH2) or 1800 arcseconds (MAgPIE). Note that there is a coarser-yet
    # scale possible from e.g. GTAP-determined endogenous LUC. This is excluded in the base SEALS config.
    p.fine_resolution_arcseconds = 10.0 # MUST BE FLOAT
    p.coarse_resolution_arcseconds = 900.0 # MUST BE FLOAT
    p.processing_resolution_arcseconds = 3600.0 # MUST BE FLOAT
    p.processing_block_size = 1.0 # In degrees

    p.calibration_parameters_source = os.path.join(p.base_data_dir, 'seals', 'default_inputs', 'trained_coefficients_seals_manuscript_global_3600sec.csv') # One of 'calibration_task', 'PATH_TO_COMBINED_PARAMETERS_CSV'

    # Can be used in specific scenarios to e.g., not allow expansion of cropland into forest by overwriting the default calibration. Note that I haven't set exactly how this
    # will work if it is specific to certain zones or a single-zone that overrides all. The latter would probably be easier.
    # If the DF has a value, override. If it is None or "", keep from parameters source.
    p.calibration_parameters_override_dict = {}
    # p.calibration_parameters_override_dict['rcp45_ssp2'][2030]['BAU'] = os.path.join(p.input_dir, 'calibration_overrides', 'prevent_cropland_expansion_into_forest.xlsx')

    p.force_to_global_bb = False

    # To run a much faster version for code-testing purposes, enable test_mode. Selects a much smaller set of scenarios and spatial tiles.
    p.test_mode = False

    # WARNING NOT SURE IF MADE CONSISTENT
    p.luh_scenario_labels = ['rcp45_ssp2']
    p.null_policy_scenario_labels = ['no_policy']

    p.magpie_policy_scenario_labels = [
        # 'SSP2_BiodivPol_LPJmL5',
        # 'SSP2_BiodivPol_ClimPol_LPJmL5',
        'SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5',
        'SSP2_ClimPol_LPJmL5',
        # 'SSP2_NPI_base_LPJmL5',
    ]

    p.gtap_five_scenario_labels = ['BAU', 'BAU_rigid', 'PESGC', 'SR_Land', 'PESLC', 'SR_RnD_20p', 'SR_RnD_20p_PESGC', 'SR_RnD_20p_PESGC_30']
    p.gtap_combined_policy_scenario_labels = ['BAU', 'BAU_rigid', 'PESGC', 'SR_Land', 'PESLC', 'SR_RnD_20p', 'SR_Land_PESGC', 'SR_PESLC', 'SR_RnD_20p_PESGC', 'SR_RnD_PESLC', 'SR_RnD_20p_PESGC_30']


    if p.test_mode:
        if p.is_magpie_run:
            p.policy_scenario_labels = p.magpie_policy_scenario_labels
        elif p.is_gtap1_run:
            p.policy_scenario_labels = p.gtap_five_scenario_labels
            # p.policy_scenario_labels = p.gtap_bau_and_30_labels
        elif p.is_calibration_run:
            p.policy_scenario_labels = p.null_policy_scenario_labels
    else:
        if p.is_magpie_run:
            p.policy_scenario_labels = p.magpie_policy_scenario_labels
        elif p.is_gtap1_run:
            p.policy_scenario_labels = p.gtap_combined_policy_scenario_labels
        elif p.is_calibration_run:
            p.policy_scenario_labels = p.null_policy_scenario_labels

    # Choose which set of tasks to run. Including tasks such as calibrate() to the task tree GREATLY increases run time, even for test runs
    p.run_type = 'complete_run' # Can choose from 'allocation_run, 'complete_run', 'extract_calibration_from_project', 'calculate_carbon'

    # In order to apply this code to the magpie model, I set this option to either
    # use the GTAP-shited LUH data (as was done in the WB feedback model)
    # or to instead use the outputs of some other extraction functions with
    # no shifting logic. This could be scaled to different interfaces
    # when models have different input points.
    p.is_magpie_run = False

    # For the intitial magpie run, we enforced that the amount of ag land in ESA had to match that in Magpie. This enables that option.
    p.adjust_baseline_to_match_magpie_2015 = False

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

    # Provided by GTAP team.
    # TODOO This is still based on the file below, which was from Purdue. It is a vector of 300sec gridcells and should be replaced with continuous vectors
    p.gtap37_aez18_input_vector_path = os.path.join(p.model_base_data_dir, "region_boundaries\GTAPv10_AEZ18_37Reg.shp")


    p.reporting_level = 0
    p.calibration_reporting_level = 0
    p.output_writing_level = 5  # >=2 writes chunk-baseline lulc
    p.build_overviews_and_stats = 0  # For later fast-viewing, this can be enabled to write ovr files and geotiff stats files.
    p.write_projected_coarse_change_chunks = 0  # in the SEALS allocation, for troubleshooting, it can be useful to see what was the coarse allocation input.
    p.num_workers = 14  # None sets it to max available.


    # If you want to use R to post-process the GTAP sl4 files, include your path to the Rscript.exe file on your system here.
    p.r_executable_path = r"C:\Program Files\R\R-4.2.1\bin\Rscript.exe"
    # p.r_executable_path = r"C:\Program Files\R\R-4.0.3\bin\Rscript.exe"


    gtap_invest_initialize_project.run(p)
