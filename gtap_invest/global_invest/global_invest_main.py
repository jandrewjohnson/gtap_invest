import hazelbean as hb
import os, sys, math
import pandas as pd
import numpy as np



recompile_cython = 0
if recompile_cython:
    # NOTE, successful recompilation assumes a strict definitino of where the project run_script is relative to the src dir.
    old_cwd = os.getcwd()
    os.chdir('global_invest')
    cython_command = "python compile_cython_functions.py build_ext -i clean"
    returned = os.system(cython_command)
    if returned:
        raise NameError('Cythonization failed.')
    os.chdir(old_cwd)


# import gtap_invest.global_invest
# import gtap_invest.global_invest.carbon_biophysical

def align_invest_inputs(p):
    p.lulc_input_path = os.path.join(p.model_base_data_dir, 'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif')

    p.match_int_path = p.lulc_input_path
    default_ndv = -9999.  # Metadata had it wrong

    # First reclass the data (doing first at modis resolution because faster)
    p.depth_to_root_rest_layer_input_path = os.path.join(p.model_base_data_dir, 'wateryield', 'isric', 'depth_to_root_restricting_layer.tif')
    p.depth_to_root_rest_layer_reclassed_path = os.path.join(p.project_dir, 'depth_to_root_rest_layer_reclassed.tif')
    rules = {-9999.0: 2400.0}
    if not os.path.exists(p.depth_to_root_rest_layer_reclassed_path):
        hb.reclassify_flex(p.depth_to_root_rest_layer_input_path, rules, p.depth_to_root_rest_layer_reclassed_path,
                           target_datatype=7, target_nodata=default_ndv, compress=True)  # NOTE, datatype must be 7 due to use of float64 in cython

    # Resample to ESA resolution.
    p.depth_to_root_rest_layer_output_path = os.path.join(p.project_dir, 'depth_to_root_rest_layer_10s.tif')
    if not os.path.exists(p.depth_to_root_rest_layer_output_path):
        hb.resample_to_match(p.depth_to_root_rest_layer_reclassed_path, p.match_int_path, p.depth_to_root_rest_layer_output_path,
                             resample_method='near', output_data_type=6, compress=True, ndv=-9999.0,
                             add_overviews=True, calc_raster_stats=True)


    # Resample potential evapo transpiration (reference evapotranspiration in this case) to ESA resolution.
    p.reference_evapotranspiration_input_path = os.path.join(p.model_base_data_dir, 'wateryield', 'cgiar_csi', 'pet.tif')
    p.reference_evapotranspiration_output_path = os.path.join(p.project_dir, 'reference_evapotranspiration_10s.tif')
    if not os.path.exists(p.reference_evapotranspiration_output_path):
        hb.resample_to_match(p.reference_evapotranspiration_input_path, p.match_int_path, p.reference_evapotranspiration_output_path,
                             resample_method='near', output_data_type=6, compress=True, ndv=default_ndv,
                             add_overviews=True, calc_raster_stats=True)



    # Resample baseline precip to ESA resolution.
    p.precipitation_2015_input_path = os.path.join(p.model_base_data_dir, 'wateryield', 'bio_12.tif')
    p.precipitation_2015_output_path = os.path.join(p.project_dir, 'precipitation_2015_10s.tif')
    if not os.path.exists(p.precipitation_2015_output_path):
        hb.resample_to_match(p.precipitation_2015_input_path, p.match_int_path, p.precipitation_2015_output_path,
                             resample_method='near', output_data_type=6, compress=True, ndv=-9999.0,
                             add_overviews=True, calc_raster_stats=True)

    # Resample ssp precip to ESA resolution.
    p.precipitation_ssp1_input_path = os.path.join(p.model_base_data_dir, r"climate_change_projections\hd26bi50\hd26bi5012.tif")
    p.precipitation_ssp1_output_path = os.path.join(p.project_dir, 'precipitation_ssp1_rcp26_10s.tif')
    if not os.path.exists(p.precipitation_ssp1_output_path):
        hb.resample_to_match(p.precipitation_ssp1_input_path, p.match_int_path, p.precipitation_ssp1_output_path,
                             resample_method='near', output_data_type=6, compress=True, ndv=-9999.0,
                             add_overviews=True, calc_raster_stats=True)

    p.precipitation_ssp3_input_path = os.path.join(p.model_base_data_dir, r"climate_change_projections\hd60bi50\hd60bi5012.tif")
    p.precipitation_ssp3_output_path = os.path.join(p.project_dir, 'precipitation_ssp3_rcp60_10s.tif')
    if not os.path.exists(p.precipitation_ssp3_output_path):
        hb.resample_to_match(p.precipitation_ssp3_input_path, p.match_int_path, p.precipitation_ssp3_output_path,
                             resample_method='near', output_data_type=6, compress=True, ndv=-9999.0,
                             add_overviews=True, calc_raster_stats=True)

    p.precipitation_ssp5_input_path = os.path.join(p.model_base_data_dir, r"climate_change_projections\hd85bi50\hd85bi5012.tif")
    p.precipitation_ssp5_output_path = os.path.join(p.project_dir, 'precipitation_ssp5_rcp85_10s.tif')
    if not os.path.exists(p.precipitation_ssp5_output_path):
        hb.resample_to_match(p.precipitation_ssp5_input_path, p.match_int_path, p.precipitation_ssp5_output_path,
                             resample_method='near', output_data_type=6, compress=True, ndv=-9999.0,
                             add_overviews=True, calc_raster_stats=True)

    # "C:\OneDrive\Projects\cge\gtap_invest\projects\align_base_data\Globio4_RCP2.6_SSP1.tif"
    # "C:\OneDrive\Projects\cge\gtap_invest\projects\align_base_data\Globio4_current.tif"
    # "C:\OneDrive\Projects\cge\gtap_invest\projects\align_base_data\Globio4_CPP8.5_SSP5.tif"
    # "C:\OneDrive\Projects\cge\gtap_invest\projects\align_base_data\Globio4_RCP7.0_SSP3.tif"
    # p.precipitation_ssp5_input_path = os.path.join(p.model_base_data_dir, r"climate_change_projections\hd85bi50\hd85bi5012.tif")
    # p.precipitation_ssp5_output_path = os.path.join(p.project_dir, 'precipitation_ssp5_rcp85_10s.tif')
    # if not os.path.exists(p.precipitation_ssp5_output_path):
    #     hb.resample_to_match(p.precipitation_ssp5_input_path, p.match_int_path, p.precipitation_ssp5_output_path,
    #                          resample_method='near', output_data_type=6, compress=True, ndv=-9999.0,
    #                          add_overviews=True, calc_raster_stats=True)

    # First reclass the data (doing first at modis resolution because faster)
    p.pawc_input_path = os.path.join(p.model_base_data_dir, 'wateryield', 'pawc_30s.tif')
    p.pawc_reclassed_path = os.path.join(p.project_dir, 'pawc_30s_reclassed.tif')
    rules = {-9999.0: 0.11814513414577} # #0.11814513414577 mean

    if not os.path.exists(p.pawc_reclassed_path):
        hb.reclassify_flex(p.pawc_input_path, rules, p.pawc_reclassed_path,
                           target_datatype=7, target_nodata=default_ndv, compress=True) # NOTE, datatype must be 7 due to use of float64 in cython

    # Resample pawc to ESA resolution.)
    p.pawc_output_path = os.path.join(p.project_dir, 'pawc_10s.tif')
    if not os.path.exists(p.pawc_output_path):

        hb.resample_to_match(p.pawc_reclassed_path, p.match_int_path, p.pawc_output_path,
                             resample_method='near', output_data_type=6, compress=True, src_ndv=default_ndv, ndv=default_ndv,
                             add_overviews=True, calc_raster_stats=True)

# LEFT UNDONE: Making precip layer global extent (missing bottom 30 degrees)
# path_1 = r"C:\OneDrive\Projects\cge\gtap_invest\base_data\wateryield\cgiar_csi\pet2.tif"
# hb.set_geotransform_to_tuple(path_1, desired_geotransform=hb.geotransform_global_30s)
# hb.add_rows_or_cols_to_geotiff(temp_path, r_above, r_below, c_above, c_below)
#
# for filename in hb.list_filtered_paths_recursively(p.project_dir, include_extensions='.tif'):
#     print(filename)
#     # hb.describe_path(filename)
#     print(hb.get_raster_info(filename)['geotransform'])
    # hb.make_path_global_pyramid()



def carbon(lulc_path, carbon_zones_path, carbon_table_path, output_path):
    lookup_table_df = pd.read_csv(carbon_table_path, index_col=0)
    lookup_table = np.float32(lookup_table_df.values)
    print('Running global invest carbon model with look-up table: ' + str(lookup_table_df))
    row_names = {int(v): int(c) for c, v in enumerate(lookup_table_df.index)}
    col_names = {int(v): int(c) for c, v in enumerate(lookup_table_df.columns)}


    print('lulc_path', lulc_path)
    print('carbon_zones_path', carbon_zones_path)
    print('lookup_table', lookup_table)
    print('row_names', row_names)
    print('col_names', col_names)

    import gtap_invest
    import gtap_invest.global_invest
    import gtap_invest.global_invest.carbon_biophysical
    from gtap_invest.global_invest.carbon_biophysical import write_carbon_table_to_array
    base_raster_path_band = [(lulc_path, 1), (carbon_zones_path, 1), (lookup_table, 'raw'), (row_names, 'raw'), (col_names, 'raw')]
    hb.raster_calculator_hb(base_raster_path_band, write_carbon_table_to_array, output_path, 6, -9999, hb.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS_HB)

