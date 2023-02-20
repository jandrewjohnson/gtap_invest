import os
import geopandas as gpd
import hazelbean as hb
import pandas as pd


def joined_region_vectors(p):
    p.current_task_documentation = """
Merge all possible sources of country names, attributes, ids, etc. from Natural Earth Vector and GADM and a marine boundaries map.
This was primarily an exercise to make sure there weren't missing regions, breakaway zones, disputed regions, etc, that were causeing
exclusions in the data.

nev_admin0_allattr.gpkg includes all attributes from nev, gadm, fao, gtap, etc., but linked to a relatively simple nev vector file.
gadm_level0_simp10sec_all_attr.gpkg is similar to nev but has a much more detailed vector definition of boundaries.
comprehensive_region_attributes is the above with no vector produced.
    """
    hb.write_to_file(p.current_task_documentation, os.path.join(p.cur_dir, 'data_note.md'))

    # These are all the raw input I'm considering here, though countries.shp has some legacy modifications of the NEV that I want to keep (e.g. keeping FAO codes).
    p.ne_110m_admin_0_countries_input_path = r"C:\Files\Research\base_data\cartographic\naturalearth_v3\110m_cultural\ne_110m_admin_0_countries.shp"
    p.ne_50m_admin_0_countries_input_path = r"C:\Files\Research\base_data\cartographic\naturalearth_v3\50m_cultural\ne_50m_admin_0_countries.shp"
    p.ne_10m_admin_0_countries_input_path = r"C:\Files\Research\base_data\cartographic\naturalearth_v3\10m_cultural\ne_10m_admin_0_countries.shp"
    p.countries_input_path = r"C:\Files\Research\base_data\pyramids\countries.shp"
    p.countries_marine_input_path = r"C:\Files\Research\base_data\pyramids\country_marine_polygons.shp"
    p.gadm36_input_path = r"C:\Files\Research\base_data\cartographic\gadm\gadm36_level0.gpkg"
    p.gadm36_level0_simp10sec_input_path = r"C:\Files\Research\base_data\cartographic\gadm\gadm36_level0_simp10sec.gpkg"

    p.nev_admin0_allattr_path = os.path.join(p.cur_dir, 'nev_admin0_allattr.gpkg')

    if p.run_this:
        # Load NEV and GADM
        ne_10m_admin_0_countries_input = gpd.read_file(p.ne_10m_admin_0_countries_input_path)
        countries_input = gpd.read_file(p.countries_input_path)

        # GADM is insanely detailed. I used QGIS Simplify (area algorithm) to create thsi file. Going to use this instead.
        gadm36_level0_simp10sec_input = gpd.read_file(p.gadm36_level0_simp10sec_input_path)

        # NEV had a typo on South Sudan, fix it
        countries_input.at[196, 'iso3'] = 'SSD'

        # Decided to collapse Western Sahara into MAR
        gadm36_level0_simp10sec_input.at[68, 'GID_0'] = 'MAR'

        ## NOTE: There are many other organizations of countries, e.g. sovereighnty, but I am going to go with just ADM0_A3 because it is unique and comprehensive.
        # SOV_A3_uniques = np.unique(ne_10m_admin_0_countries_input['SOV_A3'])
        # ADM0_A3_uniques = np.unique(ne_10m_admin_0_countries_input['ADM0_A3'])
        # GU_A3_uniques = np.unique(ne_10m_admin_0_countries_input['GU_A3'])
        # SU_A3_uniques = np.unique(ne_10m_admin_0_countries_input['SU_A3'])
        # BRK_A3_uniques = np.unique(ne_10m_admin_0_countries_input['BRK_A3'])

        # For intuition, just print out the set differences.
        output_dict = hb.compare_sets(countries_input, gadm36_level0_simp10sec_input['GID_0'])


        # Decided to merge all the attributes into two different shapefiles and as a CSV.
        # First merge both but drop geometries so that it is a CSV.
        merged_df = pd.merge(gadm36_level0_simp10sec_input[[i for i in gadm36_level0_simp10sec_input.columns if i != 'geometry']],
                             ne_10m_admin_0_countries_input[[i for i in ne_10m_admin_0_countries_input.columns if i != 'geometry']],
                             how='outer', left_on='GID_0', right_on='ADM0_A3')
        merged_df.to_csv(os.path.join(p.cur_dir, 'comprehensive_region_attributes.csv'))

        # Merge again but keeping the geometry from GADM
        p.gadm_level0_simp10sec_allattr_path = os.path.join(p.cur_dir, 'gadm_level0_simp10sec_allattr.gpkg')
        if not hb.path_exists(p.gadm_level0_simp10sec_allattr_path):
            merged_gdf = gadm36_level0_simp10sec_input.merge(countries_input[[i for i in countries_input.columns if i != 'geometry']],
                                 how='outer', left_on='GID_0', right_on='iso3')
            merged_gdf.to_file(p.gadm_level0_simp10sec_allattr_path, driver='GPKG')

        # Merge again but keeping the geometry from NEV

        if not hb.path_exists(p.nev_admin0_allattr_path):

            merged_gdf = countries_input.merge(gadm36_level0_simp10sec_input[[i for i in gadm36_level0_simp10sec_input.columns if i != 'geometry']],
                                 how='outer', left_on='iso3', right_on='GID_0')

            merged_gdf = merged_gdf[merged_gdf['iso3'].notnull()]
            merged_gdf = merged_gdf[merged_gdf['NAME_0'] != 'Morocco']
            cols_to_drop = ['adm0_a3_un', 'adm0_a3_wb', 'formal_fr', 'iso', 'iso2_cull', 'iso3_cull', 'iso_3digit', 'iso_a3',
             'iso_a3_eh', 'iso_n3', 'lastcensus',
             'name_ar', 'name_bn', 'name_cap', 'name_de', 'name_el', 'name_en', 'name_es', 'name_fr',
             'name_hi', 'name_hu', 'name_id', 'name_it', 'name_ja', 'name_ko', 'name_nl', 'name_pl', 'name_pt', 'name_ru',
             'name_sort', 'name_sv', 'name_tr', 'name_vi', 'name_zh',  'note_adm0', 'note_brk',
             'olympic', 'pop_rank', 'pop_year', 'sovereignt',
             'un_vehicle', 'wiki1', 'wikidataid', 'wikipedia', 'woe_note',]


            merged_gdf = merged_gdf[[i for i in merged_gdf.columns if i not in cols_to_drop]]

            merged_gdf.to_file(p.nev_admin0_allattr_path, driver='GPKG')

            hb.write_to_file(p.current_task_documentation, os.path.join(p.cur_dir, 'data_note.md'))


def gtap_vector_pyramid(p):
    p.current_task_documentation = """
Generate a vector file that contains all AEZs, Regions, and other processing units. Then add a pyramid_id column that will be a unique
index for raster stats later on that corresponds with the generated regions. In the case of 37 Regs and 18 AEZs, there are 341 regions.
"""
    hb.write_to_file(p.current_task_documentation, os.path.join(p.cur_dir, 'data_note.md'))

    # TODOO Where to put these? Not really a run config...
    # To easily convert between per-ha and per-cell terms, these very accurate ha_per_cell maps are defined.
    p.ha_per_cell_10sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_10sec.tif")
    p.ha_per_cell_300sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_300sec.tif")
    p.ha_per_cell_900sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_900sec.tif")

    # The ha per cell paths also can be used when writing new tifs as the match path.
    p.match_10sec_path = p.ha_per_cell_10sec_path
    p.match_300sec_path = p.ha_per_cell_300sec_path
    p.match_900sec_path = p.ha_per_cell_900sec_path

    # this will be the most commonly used vector path.
    p.gtap37_aez18_path = os.path.join(p.cur_dir, "GTAP37_AEZ18.gpkg")

    p.zone_ids_raster_path = os.path.join(p.cur_dir, 'GTAP37_AEZ18_ids_10sec.tif')
    p.zone_ids_raster_300sec_path = os.path.join(p.cur_dir, 'GTAP37_AEZ18_ids_300sec.tif')
    p.zone_ids_raster_900sec_path = os.path.join(p.cur_dir, 'GTAP37_AEZ18_ids_900sec.tif')

    if p.run_this:

        # Process vector file so that it is pyramidal.
        if not hb.path_exists(p.gtap37_aez18_path):
            hb.make_vector_path_global_pyramid(p.gtap37_aez18_input_vector_path, p.gtap37_aez18_path, pyramid_index_columns=['GTAP37v10', 'AEZ'])

        # Once pyramidal, rasterize it to the unique values at 10sec.
        if not hb.path_exists(p.zone_ids_raster_path):
            hb.rasterize_to_match(p.gtap37_aez18_path, p.match_10sec_path, p.zone_ids_raster_path, 'pyramid_id', ndv=-9999, datatype=5)

        # And again at 5min
        if not hb.path_exists(p.zone_ids_raster_300sec_path):
            hb.rasterize_to_match(p.gtap37_aez18_path, p.match_300sec_path, p.zone_ids_raster_300sec_path, 'pyramid_id', ndv=-9999, datatype=5)

        # And again at 15min
        if not hb.path_exists(p.zone_ids_raster_900sec_path):
            hb.rasterize_to_match(p.gtap37_aez18_path, p.match_900sec_path, p.zone_ids_raster_900sec_path, 'pyramid_id', ndv=-9999, datatype=5)



def gtap_marine_regions(p):
    p.current_task_documentation = """
This contains an accurate marine boundareis file ready for aggregation of e.g., the coastal protection model. 
Also joins all attributes from gtap_vector_pyramid and joined_region_vector
    """
    # NEED TO MERGE marine with ISO3 first.
    # TODOO, deal with the fact taht pyramiding a vector file leads to extraneous columns where only the LAST one was written, eg gtap140 is irrelevant now in gtap37 unless somehow coerced into a list.

    # Convert marine shapefile to GPKG
    p.marine_zones_input_path = os.path.join(hb.BASE_DATA_DIR, r"cartographic\marineregions\EEZ_land_union_v2_201410\EEZ_land_v2_201410.shp")
    p.marine_zones_path = os.path.join(p.cur_dir, 'marine_zones.gpkg')
    # "C:\Files\Research\cge\gtap_invest\projects\feedback_with_policies\intermediate\gtap_marine_regions\countries_all.gpkg"
    p.admin0_terrestrial_and_marine_comprehensive_path = os.path.join(p.cur_dir, 'admin0_terrestrial_and_marine_comprehensive.gpkg')
    p.gtap37_marine_pyramid_path = os.path.join(p.cur_dir, 'gtap37_marine_pyramid.gpkg')
    if p.run_this:
        if not hb.path_exists(p.marine_zones_path):
            marine_zones_input = gpd.read_file(p.marine_zones_input_path)
            marine_zones_input.to_file(p.marine_zones_path, driver='GPKG')

        # Overlay via union the marine shapefile with the NEV shapefile. NOTE that this will create mismatched country boundaries that we don't wont (and thus tons of small slivers)
        p.just_marine_mixed_boundaries_path = os.path.join(p.cur_dir, 'just_marine_mixed_boundaries.gpkg')
        if not hb.path_exists(p.just_marine_mixed_boundaries_path):
            # LEARNING POINT Tried to do this with the GPD version of difference, but unlike in Q, this just deleted everything rather than slicing out the non-overlap.  Thus i did it via union and then drop.
            nev_admin0_allattr = gpd.read_file(p.nev_admin0_allattr_path, driver='GPKG')
            marine_zones = gpd.read_file(p.marine_zones_path)
            # LEARNING POINT: In GPD with UNION OVERLAY ERROR: if unspecified error, try merging only on the minimum columns needed.
            just_marine_mixed_boundaries = gpd.overlay(nev_admin0_allattr[['iso3', 'geometry']], marine_zones[['ISO_3digit',  'geometry']], how='union')
            just_marine_mixed_boundaries.to_file(p.just_marine_mixed_boundaries_path, driver='GPKG')

        # Get rid of the wrong (marine based) terrestrial boundaries by dropping where iso3 (which is only in NEV) is na. This makes it so the file is ONLY the ocean boundaries.
        p.just_marine_path = os.path.join(p.cur_dir, 'just_marine.gpkg')
        if not hb.path_exists(p.just_marine_path):
            just_marine_mixed_boundaries = gpd.read_file(p.just_marine_mixed_boundaries_path)
            just_marine = just_marine_mixed_boundaries[just_marine_mixed_boundaries['iso3'].isna()]
            just_marine.to_file(p.just_marine_path, driver='GPKG')

        # There are also several files with 'ISO_3digit' set to '--'. I manually reasigned these via proximity and wikipedia.
        p.just_marine_empty_dropped_path = os.path.join(p.cur_dir, 'just_marine_empty_dropped.gpkg')
        if not hb.path_exists(p.just_marine_empty_dropped_path):
            just_marine = gpd.read_file(p.just_marine_path)

            just_marine['fid'] = just_marine.index
            just_marine.loc[just_marine['fid']==242, 'ISO_3digit'] = 'TTO'
            just_marine.loc[just_marine['fid']==241, 'ISO_3digit'] = 'MRT'
            just_marine.loc[just_marine['fid']==240, 'ISO_3digit'] = 'SOM'
            just_marine.loc[just_marine['fid']==235, 'ISO_3digit'] = 'CHN'
            just_marine.loc[just_marine['fid']==236, 'ISO_3digit'] = 'AUS'
            just_marine.loc[just_marine['fid']==8, 'ISO_3digit'] = 'PNG'
            just_marine.loc[just_marine['fid']==7, 'ISO_3digit'] = 'AUS'
            just_marine.loc[just_marine['fid']==6, 'ISO_3digit'] = 'NGA'
            just_marine.loc[just_marine['fid']==5, 'ISO_3digit'] = 'JAM'
            just_marine.loc[just_marine['fid']==4, 'ISO_3digit'] = 'PGA' # Sprately islands...
            just_marine.loc[just_marine['fid']==3, 'ISO_3digit'] = 'JPN'
            just_marine.loc[just_marine['fid']==2, 'ISO_3digit'] = 'JPN'
            just_marine.loc[just_marine['fid']==1, 'ISO_3digit'] = 'JPN'
            just_marine.loc[just_marine['fid']==0, 'ISO_3digit'] = 'JPN'
            just_marine = just_marine[just_marine['fid'] != 245]

            just_marine.to_file(p.just_marine_empty_dropped_path, driver='GPKG')

        # Merge the marine boundaries back in with the correct NEV terrestrial boundaries. This produces an outer-merge sparse table.
        p.nev_admin0_marine_outer_path = os.path.join(p.cur_dir, 'nev_admin0_marine_outer.gpkg')
        if not hb.path_exists(p.nev_admin0_marine_outer_path):
            nev_admin0_allattr = gpd.read_file(p.nev_admin0_allattr_path, driver='GPKG')
            just_marine = gpd.read_file(p.just_marine_empty_dropped_path, driver='GPKG')

            just_marine['is_marine'] = 1
            just_marine['iso3_marine_tagged'] = just_marine['ISO_3digit'].map(str) + '_MARINE'

            nev_admin0_marine_outer = pd.concat([just_marine, nev_admin0_allattr], ignore_index=True, sort=False)
            # nev_admin0_marine_outer = gpd.overlay(nev_admin0_allattr, just_marine[['ISO_3digit', 'is_marine', 'iso3_marine_tagged', 'geometry']], how='union')

            nev_admin0_marine_outer.to_file(p.nev_admin0_marine_outer_path, driver='GPKG')

        # Fill the iso3 of the otherwise empty outer-merge marine boundaries with the ISO_3digit value.
        p.nev_admin0_marine_prefill_path = os.path.join(p.cur_dir, 'nev_admin0_marine_prefill.gpkg')
        if not hb.path_exists(p.nev_admin0_marine_prefill_path):
            nev_admin0_marine = gpd.read_file(p.nev_admin0_marine_outer_path, driver='GPKG')
            nev_admin0_marine.loc[nev_admin0_marine['iso3'].isna(), 'iso3'] = nev_admin0_marine['ISO_3digit']
            nev_admin0_marine.to_file(p.nev_admin0_marine_prefill_path, driver='GPKG')

        # Create a new, sorted DF ready for dissolving while also fixing missing ISO codes.
        p.nev_admin0_marine_path = os.path.join(p.cur_dir, 'nev_admin0_marine.gpkg')
        if not hb.path_exists(p.nev_admin0_marine_path):
            nev_admin0_marine = gpd.read_file(p.nev_admin0_marine_prefill_path, driver='GPKG')

            # LEARNING POINT: The only way I was able to do a dissolve where it selected the filled value instead of the empty value was to use the aggfunc 'first' option
            # However, this required manually rewriting the GPKG so the objects were in the right order. To do this, I split the DF into notna and isna and concated.
            df = nev_admin0_marine[nev_admin0_marine['id'].notna()]
            df = df.sort_values(axis=0, by='iso3')

            df2 = nev_admin0_marine[nev_admin0_marine['id'].isna()]
            df = pd.concat([df, df2], axis=0)

            # Rewrite via apply function all the non-iso3 names to the nearest neighbor and wikipedia entries.
            def replace_func(x):
                replacement_dict = {}
                replacement_dict['ANT'] = 'VEN'
                replacement_dict['BES'] = 'VEN'
                replacement_dict['BVT'] = 'NOR'
                replacement_dict['CCK'] = 'AUS'
                replacement_dict['CPT'] = 'FRA'
                replacement_dict['CW'] = 'VEN'
                replacement_dict['CXR'] = 'IDN'
                replacement_dict['ESH'] = 'MAR'
                replacement_dict['GLP'] = 'FRA'
                replacement_dict['GUF'] = 'FRA'
                replacement_dict['MNP++'] = 'GUM'
                replacement_dict['MTQ'] = 'FRA'
                replacement_dict['MYT'] = 'FRA'
                replacement_dict['PSE'] = 'PSX'
                replacement_dict['REU'] = 'FRA'
                replacement_dict['SJM'] = 'NOR'
                replacement_dict['TKL'] = 'NZL'
                if x in replacement_dict.keys():
                    return replacement_dict[x]
                else:
                    return x

            df['iso3'] = df.apply(lambda x: x['iso3'] if x['nev_name'] else replace_func(x['iso3']), axis=1)

            df.loc[df['iso3_marine_tagged'].isna(), 'iso3_marine_tagged_filled'] = df['iso3']
            df.loc[df['iso3_marine_tagged'].notna(), 'iso3_marine_tagged_filled'] = df['iso3'].map(str) + '_MARINE'
            df['to_dissolve'] = df['iso3_marine_tagged_filled']
            df = df.dissolve('to_dissolve')

            df = df[df.geometry.area > .0000000001]

            df.to_file(p.nev_admin0_marine_path, driver='GPKG')


        # Dissolve file so that polygons include both the marine and terrestrial boundaries.
        p.nev_admin0_marine_dissolved_path = os.path.join(p.cur_dir, 'nev_admin0_marine_dissolved.gpkg')
        if not hb.path_exists(p.nev_admin0_marine_dissolved_path):
            df = gpd.read_file(p.nev_admin0_marine_path, driver='GPKG')
            df['to_dissolve'] = df['iso3']

            # LEARNING POINT, had to doe a shapely fix non-noded intersection by simplifying geometry a very tiny amount.
            df['geometry'] = df.geometry.simplify(0.0000833333333333333, preserve_topology=False)
            df = df.dissolve('to_dissolve')
            df.to_file(p.nev_admin0_marine_dissolved_path, driver='GPKG')

        p.gtap140_aez18_input_path = r"C:\Files\Research\cge\gtap_invest\base_data\region_boundaries\GTAPv10_AEZ18_37Reg.shp"
        gtap37_aez18 = gpd.read_file(p.gtap37_aez18_path, driver='GPKG')
        gtap140_aez18_input = gpd.read_file(p.gtap140_aez18_input_path)

        gtap140_gtap37_correspondence, categories = hb.extract_correspondence_and_categories_dicts_from_df_cols(gtap140_aez18_input, 'GTAP140v9a', 'GTAP37v10')

        # Dissolve file so that polygons include both the marine and terrestrial boundaries.
        p.nev_admin0_marine_with_isogtap226_path = os.path.join(p.cur_dir, 'nev_admin0_marine_with_isogtap226.gpkg')
        if not hb.path_exists(p.nev_admin0_marine_with_isogtap226_path):
            df = gpd.read_file(p.nev_admin0_marine_path, driver='GPKG')
            df2 = gpd.read_file(p.gtap140_aez18_input_path)
            df2['isogtap226'] = df2['GTAP'].str.upper()
            df2 = df2[[i for i in df2.columns if i not in ['geometry', 'admin', 'AEZ', 'GTAP']]].drop_duplicates()

            df = df.merge(df2[[i for i in df2.columns if i not in ['geometry', 'admin', 'AEZ', 'GTAP']]], how='outer', left_on='iso3', right_on='isogtap226')

            df.to_file(p.nev_admin0_marine_with_isogtap226_path, driver='GPKG')


        if not hb.path_exists(p.admin0_terrestrial_and_marine_comprehensive_path):
            df = gpd.read_file(p.nev_admin0_marine_with_isogtap226_path, driver='GPKG')
            def gtap141v9p_replace_func(x):
                replacement_dict = {}
                replacement_dict['ABW'] = 'VEN'
                replacement_dict['AIA'] = 'VEN'
                replacement_dict['ASM'] = 'USA'
                replacement_dict['ATF'] = 'FRA'
                replacement_dict['BMU'] = 'GBR'
                replacement_dict['COK'] = 'NZL'
                replacement_dict['CYM'] = 'GBR'
                replacement_dict['FRO'] = 'DEN'
                replacement_dict['GGY'] = 'GBR'
                replacement_dict['GIB'] = 'GBR'
                replacement_dict['GUF'] = 'XSM'
                replacement_dict['HMD'] = 'AUS'
                replacement_dict['IOT'] = 'GBR'
                replacement_dict['ITA'] = 'ITA'
                replacement_dict['JEY'] = 'GBR'
                replacement_dict['MAF'] = 'FRA'
                replacement_dict['MAR'] = 'MRT'
                replacement_dict['MSR'] = 'GBR'
                replacement_dict['NFK'] = 'AUS'
                replacement_dict['NIU'] = 'NZL'
                replacement_dict['PCN'] = 'GBR'
                replacement_dict['PSE'] = 'ISR'
                replacement_dict['PYF'] = 'FRA'
                replacement_dict['ROU'] = 'XEE'
                replacement_dict['SDN'] = 'SDS'
                replacement_dict['SGS'] = 'GBR'
                replacement_dict['SHN'] = 'GBR'
                replacement_dict['SPM'] = 'FRA'
                replacement_dict['SRB'] = 'XEE'
                replacement_dict['SXM'] = 'NLD'
                replacement_dict['TCA'] = 'GBR'
                replacement_dict['UMI'] = 'USA'
                replacement_dict['VGB'] = 'GBR'
                replacement_dict['VIR'] = 'USA'
                replacement_dict['WLF'] = 'FRA'
                replacement_dict['SRB'] = 'XER'
                replacement_dict['ROU'] = 'XER'
                replacement_dict['MNE'] = 'XER'
                replacement_dict['LSO'] = 'XSC'

                replacement_dict['ALD'] = 'FIN'
                replacement_dict['ATA'] = ''
                replacement_dict['ATG'] = 'VEN'
                replacement_dict['BJN'] = 'COL'
                replacement_dict['BLM'] = 'VEN'
                replacement_dict['BRB'] = 'FRA'
                replacement_dict['CLP'] = 'MEX'
                replacement_dict['CNM'] = 'CYP'
                replacement_dict['COM'] = 'MOZ'
                replacement_dict['CPV'] = 'MRT'
                replacement_dict['CSI'] = 'AUS'
                replacement_dict['CUW'] = 'VEN'
                replacement_dict['CYN'] = 'CYP'
                replacement_dict['DMA'] = 'FRA'
                replacement_dict['ESB'] = 'CYP'
                replacement_dict['FSM'] = 'PNG'
                replacement_dict['GRD'] = 'VEN'
                replacement_dict['GUM'] = 'JPN'
                replacement_dict['IMN'] = 'GBR'
                replacement_dict['IOA'] = 'IDN'
                replacement_dict['KAB'] = 'KAZ'
                replacement_dict['KAS'] = 'IND'
                replacement_dict['KIR'] = 'FJI'
                replacement_dict['KNA'] = 'VEN'
                replacement_dict['KOS'] = 'SRB'
                replacement_dict['LCA'] = 'FRA'
                replacement_dict['MAC'] = 'CHN'
                replacement_dict['MCO'] = 'FRA'
                replacement_dict['MDV'] = 'IND'
                replacement_dict['MHL'] = 'PNG'
                replacement_dict['MLT'] = 'ITA'
                replacement_dict['MNP'] = 'JPN'
                replacement_dict['MUS'] = 'FRA'
                replacement_dict['NRU'] = 'PNG'
                replacement_dict['PGA'] = 'PHL'
                replacement_dict['PLW'] = 'PHL'
                replacement_dict['PSX'] = 'PHL'
                replacement_dict['SAH'] = 'MAR'
                replacement_dict['SCR'] = 'PHL'
                replacement_dict['SER'] = 'HND'
                replacement_dict['SOL'] = 'SOM'
                replacement_dict['SSD'] = 'SDN'
                replacement_dict['STP'] = 'NGA'
                replacement_dict['SYC'] = 'MDG'
                replacement_dict['TON'] = 'FJI'
                replacement_dict['TUV'] = 'PNG'
                replacement_dict['USG'] = 'CUB'
                replacement_dict['VAT'] = 'ITA'
                replacement_dict['VCT'] = 'VEN'
                replacement_dict['WSB'] = 'CYP'
                replacement_dict['WSM'] = 'FJI'
                if x in replacement_dict.keys():
                    return replacement_dict[x]
                else:
                    return '0'
            def gtap37v10_replace_func(x):
                replacement_dict = {}
                replacement_dict['ROU'] = 'EU27'
                replacement_dict['SRB'] = 'EU27'
                replacement_dict['MNE'] = 'Oth_CEE_CIS'
                replacement_dict['MAR'] = 'S_S_AFR'
                replacement_dict['SDN'] = 'S_S_AFR'
                replacement_dict['LSO'] = 'S_S_AFR'
                replacement_dict['GUF'] = 'S_o_Amer'
                replacement_dict['DEN'] = 'EU27'
                replacement_dict['FJI'] = 'Oceania'
                replacement_dict['CUB'] = 'C_C_Amer'
                replacement_dict['MRT'] = 'S_S_AFR'
                replacement_dict['PNG'] = 'Oceania'
                replacement_dict['SOM'] = 'S_S_AFR'

                if x in replacement_dict.keys():
                    return replacement_dict[x]
                else:
                    if x in gtap140_gtap37_correspondence.keys():
                        return gtap140_gtap37_correspondence[x]
                    else:
                        return '-9999'

            df['GTAP140v9a'] = df.apply(lambda x: x['GTAP140v9a'] if x['GTAP140v9a'] else gtap141v9p_replace_func(x['iso3']), axis=1)
            df['GTAP141v9p'] = df.apply(lambda x: x['GTAP141v9p'] if x['GTAP141v9p'] else gtap141v9p_replace_func(x['iso3']), axis=1)
            df['GTAP37v10'] = df.apply(lambda x: x['GTAP37v10'] if x['GTAP37v10'] else gtap37v10_replace_func(x['GTAP141v9p']), axis=1)

            front_few = ['iso3', 'nev_name', 'GTAP140v9a', 'GTAP37v10']
            drop_cols = ['name_alt', 'name_ar', 'name_bn', 'name_cap', 'name_de', 'name_el', 'name_en', 'name_es', 'name_fr', 'name_hir', 'name_hu', 'name_id', 'name_it', 'name_ja', 'name_ko', 'name_nl', 'name_pl', 'name_pt', 'name_ru', 'name_sv', 'name_zh', 'wiki1', 'wikidataie', 'wikipedia', 'woe_id_eh', 'woe_note']
            df = df[front_few + [i for i in df.columns if i not in front_few and i not in drop_cols]]

            df = df[df['geometry'] != None]
            df = df.sort_values('iso3')
            df.to_file(p.admin0_terrestrial_and_marine_comprehensive_path, driver='GPKG')

        # For later usage in aggregating marine statistucs, also make a pyramidal index for gtap37 regions.
        if not hb.path_exists(p.gtap37_marine_pyramid_path):
            hb.make_vector_path_global_pyramid(p.admin0_terrestrial_and_marine_comprehensive_path, p.gtap37_marine_pyramid_path, pyramid_index_columns=['GTAP37v10'])

        p.aez18_path = os.path.join(p.cur_dir, 'aez18.gpkg')
        if not hb.path_exists(p.aez18_path):
            df = gpd.read_file(p.gtap37_aez18_path, driver='GPKG')

            df = df[['gtap37v10_pyramid_id', 'aez_pyramid_id', 'geometry']]
            df = df.dissolve('aez_pyramid_id')
            df.to_file(p.aez18_path, driver='GPKG')

