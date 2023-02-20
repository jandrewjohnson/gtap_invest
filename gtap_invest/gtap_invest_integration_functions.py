import os
import subprocess
from tkinter import E
import hazelbean as hb
import time
import sys
import pandas as pd
import numpy as np
import math

from gtap_invest import harpy

L = hb.get_logger('gtap_invest_integration_functions')


def extract_vertical_csvs_from_multidimensional_sl4_csv(raw_csv_path, output_dir, output_label):
    chunks = []

    chunk_start = 4
    chunk_end = 0
    with open(raw_csv_path, 'r') as fp:
        fp_list = list(fp)
        for c, line in enumerate(fp_list):
            if c >= chunk_start:  # Skip the first few lines to allow relative back tracing.
                if line == '\n' or line == ' \n':
                    if fp_list[c - 1] == '\n' or fp_list[c - 1] == ' \n' or c + 1 == len(fp_list):
                        chunk_end = c
                        chunks.append(fp_list[chunk_start: chunk_end + 1])
                        chunk_start = c + 1

    # START HERE, this got super difficlut but it's POSSIBLE i could contimnue for ndim=2,3
    unused = []
    showing = []
    matrices = []
    vars_that_were_only_showing = []
    singular_dict = {}
    one_dim_dict = {}
    one_dim_col1 = []
    one_dim_col2 = []
    one_dim_col3 = []
    one_dim_col4 = []

    two_dim_col1 = []
    two_dim_col2 = []
    two_dim_col3 = []
    two_dim_col4 = []
    two_dim_col5 = []
    two_dim_col6 = []
    two_dim_col7 = []
    two_dim_col8 = []

    for c, chunk in enumerate(chunks):
        if chunk[0][0:11] == ' ! Variable':
            var_name = chunk[0].split(' ')[3]
            if '(' not in chunk[1]:  # then its singular
                var_dims = ['singular']
                size = 1
            else:
                var_dims = chunk[1].split('(')[1].split(')')[0].split(':')
                size = [int(i) for i in chunk[1].split(' of size ')[1].split('x')]

            n_dims = len(var_dims)

            if var_dims[0] == 'singular':
                val = float(chunk[3].split(',')[0].replace(' ', ''))

                if val and var_name:
                    singular_dict[var_name] = val

            elif n_dims == 1:
                for chunk_row in chunk[3:]:
                    if len(chunk_row.split(',')) > 1:
                        one_dim_col1.append(var_dims[0])
                        one_dim_col2.append(var_name)
                        one_dim_col3.append(chunk_row.split(',')[0].replace(' ', ''))
                        one_dim_col4.append(chunk_row.split(',')[1].replace(' ', ''))
            elif n_dims == 2:
                header = chunk[3]
                var_name = header.split(',')[0].split('(')[0]
                if ')' in header:
                    dim1_name = header.split('(')[1].split(')')[0].split(':')[0]
                    dim2_name = header.split('(')[1].split(')')[0].split(':')[1]
                    dim2_values = [i for i in header.split(')')[1].replace(' ', '').replace('\n', '').split(',') if len(i) > 0]

                    for chunk_row in chunk[4:]:

                        dim1_value = chunk_row.split(',')[0]
                        for c, value_pre in enumerate([i for i in chunk_row.split(',')[1:] if len(i.replace(' ', '').replace('\n', '')) > 0]):
                            value = value_pre.replace(' ', '')
                            dim2_value = dim2_values[c]

                            two_dim_col1.append(dim1_name)
                            two_dim_col2.append(dim1_value)
                            two_dim_col3.append(dim2_name)
                            two_dim_col4.append(dim2_value)
                            two_dim_col5.append(var_name)
                            two_dim_col6.append(value)

        elif chunk[0][0:11] == ' ! Showing ':
            size = str(chunk[0].split(' ')[-1].replace('\n', ''))
            var_name = chunks[c - 1][0].split(' ')[3]
            vars_that_were_only_showing.append((size, var_name))

            if size == '1':
                var_dims = ['singular']
                value = float(chunk[1].split(',')[0].replace(' ', ''))
                singular_dict[var_name] = value
            else:
                if 'x' in size:
                    unused.append('one here needed to be split still.' + str(size))
                else:

                    var_name = chunks[c - 1][0].split(' ')[3]
                    var_dims = chunks[c - 1][1].split(' ')[3]

                    value = float(chunk[1].split(',')[1].replace(' ', ''))

                    for chunk_row in chunk[3:]:
                        if len(chunk_row.split(',')) > 1:
                            one_dim_col1.append(var_dims[0])
                            one_dim_col2.append(var_name)
                            one_dim_col3.append(chunk_row.split(',')[0].replace(' ', ''))
                            one_dim_col4.append(chunk_row.split(',')[1].replace(' ', ''))
        else:
            unused.append(chunk)

    one_dim_dict = {'dim_name': one_dim_col1, 'dim_value': one_dim_col3, 'var_name': one_dim_col2, 'value': one_dim_col4}
    two_dim_dict = {'dim1_name': two_dim_col1, 'dim1_value': two_dim_col2, 'dim2_name': two_dim_col3, 'dim2_value': two_dim_col4, 'var_name': two_dim_col5, 'value': two_dim_col6}

    singular_df = pd.DataFrame(data={'var_name': list(singular_dict.keys()), 'value': list(singular_dict.values())})
    # singular_df = pd.DataFrame(index=list(singular_dict.keys()), data=list(singular_dict.values()))
    one_dim_df = pd.DataFrame(one_dim_dict)
    two_dim_df = pd.DataFrame(two_dim_dict)

    singular_df_path = os.path.join(output_dir, output_label + '_singular_vars.csv')
    one_dim_df_path = os.path.join(output_dir, output_label + '_one_dim_vars.csv')
    two_dim_df_path = os.path.join(output_dir, output_label + '_two_dim_vars.csv')

    singular_df.to_csv(singular_df_path, index=False)
    one_dim_df.to_csv(one_dim_df_path, index=False)
    two_dim_df.to_csv(two_dim_df_path, index=False)


    if len(unused) > 0:
        L.critical('There exist unused data in the SL4 that wasnt able to be extracted because it didnt get extracted by the manual approach.')


def extract_raw_csv_from_sl4(sl4_path, csv_path, vars_to_extract=None, additional_options=None):

    if additional_options is None:
        additional_options = []

    if vars_to_extract is not None:
        mapfile_path = os.path.join(os.path.split(sl4_path)[0], 'extract_vars.map')

        vars_to_extract = [i for i in vars_to_extract]
        extract_vars_string = '\n'.join(vars_to_extract)

        # START HERE: Clean this, then just switch back to uris' tables. he did it much better than i could

        extract_vars_string = """qgdp
  1
  2 ; """
        hb.write_to_file(extract_vars_string, mapfile_path)

        gtap_sl4_path_no_extension = os.path.splitext(sl4_path)[0]

        extraction_command = 'sltoht -ses -map=' + mapfile_path + ' ' + gtap_sl4_path_no_extension + ' ' + csv_path
        os.system(extraction_command)

        # NOT SURE WHY but sss option only works with some subset of vars. Probably due to larger dimensionality when more vars included.
        csv_sss_path = hb.suri(csv_path, 'sss')
        extraction_command = 'sltoht -SSS -map=' + mapfile_path + ' ' + gtap_sl4_path_no_extension + ' ' + csv_sss_path
        os.system(extraction_command)



    else:
        gtap_sl4_path_no_extension = os.path.splitext(sl4_path)[0]


        # extraction_command = 'sltoht ' + ' '.join(additional_options) + ' ' + sl4_path + ' ' + csv_path
        # extraction_command = 'sltoht -ses ' + gtap_sl4_path_no_extension

        # Works to get old-style all results but doesn't do multiple solutions
        extraction_command = 'sltoht -ses ' + gtap_sl4_path_no_extension + ' ' + csv_path

        extraction_command = 'sltoht -ses ' + gtap_sl4_path_no_extension + ' ' + csv_path
        print ('Running ses: ' + str(extraction_command))
        os.system(extraction_command)


    return

def gtap_shockfile_to_df(input_path):
    """Reads a GTAP-style text file and returns it as a 1-col (plus 1 col index) DataFrame.
    Still must merge these in if considering multiple shockfiles."""
    with open(input_path) as rf:
        indices = []
        col = []
        for line in rf:
            if ';' in line:
                n_entries = int(line.split(' ')[0])
                indices = list(range((n_entries)))
            else:
                col.append(float(line))
        df = pd.DataFrame(index=indices, data=col, columns=[hb.file_root(input_path)])
    return df

def run_gtap_cmf(run_label, call_list):
    
    ### USED BY PNAS CODE
    # print ('call_list', os.path.abspath(call_list[0]), os.path.abspath(call_list[2]))
    # print ('call_list nonabs', call_list)

    # old_cwd = os.getcwd()
    # os.chdir(os.path.split(call_list[0])[0])
    # print (os.getcwd())
    # call_list = [os.path.split(call_list[0])[1], call_list[1], os.path.split(call_list[2])[1]]

    proc = subprocess.Popen(call_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)

    to_log = ""
    full_log_output = []
    for line in iter(proc.stdout.readline, ''):
        cleaned_line = str(line).replace('b\'\'', '').replace('b\'', '').replace('\\r\\n\'', '').replace('b\"', '').replace('\\r\\n\"', '')
        cleaned_line.rstrip("\r\n")
        if len(cleaned_line) > 0:            
            
            # I have no idea why, but triggering the stdout into the logger makes it not show up, but print does work. Thus, I just append it to a list and report later.
            # print (run_label + ': ' + cleaned_line)
            to_log = cleaned_line
            # to_log = run_label + ': ' + cleaned_line
            
            print (to_log)
            
            full_log_output.append(to_log)
            # L.info(run_label + ': ' + cleaned_line) # THIS WONT WORK

        poll = proc.poll()

        sys.stdout.flush()

        if poll is not None:
            time.sleep(3)
            break

    proc.stdout.close()
    # L.info(hb.pp(full_log_output, return_as_string=True))


    L.info('Finished run_gtap_cmf for ' + run_label)
    if proc.wait() != 0:
        raise RuntimeError("%r failed, exit status: %d" % (str(call_list), proc.returncode))
    L.info('Finished run_gtap_cmf for ' + run_label)
    proc.terminate()
    proc = None


def generate_policy_baseline_cmf_file(src_cmf_path, output_cmf_path, run_on_validation=False):
    output_lines = []
    with open(src_cmf_path) as rf:
        for line in rf:
            if 'gtp1414' in line:
                line = line.replace('gtp1414\\', '')


            if run_on_validation:
                if 'allES' in line and 'SUPP' not in line:
                    line = line.replace('allES', 'allES_validation')

                if 'file "X2030_' in line:
                    line = line.replace('.txt', '_validation.txt')

            output_lines.append(line)


    with open(output_cmf_path, 'w') as wf:
        wf.writelines(output_lines)


def generate_postsims_r_script_file(src_r_script_path, output_r_script_path, working_dir=None):
    output_lines = []
    if working_dir is None:
        working_dir = os.path.split(src_r_script_path)[0]

    with open(src_r_script_path) as rf:

        for c, line in enumerate(rf):
            output_lines.append(line)


        output_lines[1:1] = [
            '\n',
            '\n',
            '# AUTOMATICALLY ADDED BY GTAP-INVEST script\n',
            'setwd(\"' + os.path.abspath(working_dir).replace('\\', '/') + '\")\n',
            'packages < - c("ggplot2", "readxl")\n',
            'installed_packages < - packages % in % rownames(installed.packages())\n',
            'if (any(installed_packages == FALSE)) {\n',
            'install.packages(packages[!installed_packages])\n',
            '}\n',
            'invisible(lapply(packages, library, character.only = TRUE))\n',
        ]


        # output_lines.insert(1, '\n')
        # output_lines.insert(1, 'setwd(\"' + os.path.abspath(working_dir).replace('\\', '/') + '\")\n')
        # output_lines.insert(1, '# AUTOMATICALLY ADDED BY GTAP-INVEST script\n')
        # output_lines.insert(1, 'packages < - c("ggplot2", "readxl")\n')
        # output_lines.insert(1, 'installed_packages < - packages % in % rownames(installed.packages())\n')
        # output_lines.insert(1, 'if (any(installed_packages == FALSE)) {\n')
        # output_lines.insert(1, 'install.packages(packages[!installed_packages])\n')
        # output_lines.insert(1, '}\n')
        # output_lines.insert(1, 'invisible(lapply(packages, library, character.only = TRUE))\n')

        # # Package names
        # packages < - c("ggplot2", "readxl", "dplyr", "tidyr", "ggfortify", "DT", "reshape2", "knitr", "lubridate", "pwr", "psy", "car", "doBy", "imputeMissings", "RcmdrMisc", "questionr", "vcd", "multcomp", "KappaGUI", "rcompanion", "FactoMineR", "factoextra", "corrplot", "ltm", "goeveg", "corrplot", "FSA", "MASS", "scales", "nlme", "psych", "ordinal", "lmtest", "ggpubr", "dslabs", "stringr", "assist", "ggstatsplot", "forcats", "styler", "remedy", "snakecaser", "addinslist", "esquisse", "here", "summarytools", "magrittr", "tidyverse", "funModeling", "pander", "cluster", "abind")
        #
        # # Install packages not yet installed
        # installed_packages < - packages % in % rownames(installed.packages())
        # if (any(installed_packages == FALSE)) {
        # install.packages(packages[!installed_packages])
        # }
        #
        # # Packages loading
        # invisible(lapply(packages, library, character.only = TRUE))
        # output_lines.insert(1, '\n')
        # output_lines.insert(1, '\n')

    with open(output_r_script_path, 'w') as wf:
        for line in output_lines:
            if "C:/Files/Research/cge/gtap_invest/gtap_invest_dev" not in line:
                wf.writelines(line)


def generate_procs_r_script_file(src_r_script_path, output_r_script_path, working_dir=None):
    output_lines = []

    if working_dir is None:
        working_dir = os.path.split(src_r_script_path)[0]

    with open(src_r_script_path) as rf:

        for c, line in enumerate(rf):

            # Do line-by-line replacements
            if '20210204_gtap2_shockfile' in line:
                line = line.replace('20210204_gtap2_shockfile', 'gtap2_shockfile')
                line = line[:-1] + '  # AUTOMATICALLY ADDED BY GTAP-InVEST script' + line[-1:]
                # final_shocks <- final_shocks - 100
            # if 'final_shocks <- final_shocks - 100' in line:
            #     line = line.replace('final_shocks <- final_shocks - 100', '')
            #     line = line[:-1] + '  # AUTOMATICALLY ADDED BY GTAP-InVEST script' + line[-1:]
            if 'aggregate(shocks[,3:nscen],' in line:
                line = line.replace('aggregate(shocks[,3:nscen],', 'aggregate(shocks[,3:(nscen+2)],')
                line = line[:-1] + '  # AUTOMATICALLY ADDED BY GTAP-InVEST script' + line[-1:]
            if 'aggregate(shocks[,2:nscen],' in line: #reg_shocks <- aggregate(shocks[,2:(nscen+1)], by=list(REG = shocks$REG), FUN ="sum")
                line = line.replace('aggregate(shocks[,2:nscen],', 'aggregate(shocks[,2:(nscen+1)],')
                line = line[:-1] + '  # AUTOMATICALLY ADDED BY GTAP-InVEST script' + line[-1:]
            if 'shock_final_data <- (shock_data - 100) * shock_yield_ratio$KlienYieldRatio/100 * shock_wgt_data$aez_wght' in line:
                line = line.replace('shock_final_data <- (shock_data - 100) * shock_yield_ratio$KlienYieldRatio/100 * shock_wgt_data$aez_wght', 'shock_final_data <- (shock_data - 1) * shock_wgt_data$aez_wght * 100')
                line = line[:-1] + '  # AUTOMATICALLY ADDED BY GTAP-InVEST script' + line[-1:]
            output_lines.append(line)

        # Do insertions on the top of the file.
        output_lines.insert(1, '\n')
        output_lines.insert(1, '\n')
        output_lines.insert(1, 'setwd(\"' + os.path.abspath(working_dir).replace('\\', '/') + '\")\n')
        output_lines.insert(1, '# AUTOMATICALLY ADDED BY GTAP-InVEST script\n')
        output_lines.insert(1, '\n')
        output_lines.insert(1, '\n')

    with open(output_r_script_path, 'w') as wf:
        for line in output_lines:
            if "C:/Files/Research/cge/gtap_invest/gtap_invest_dev" not in line:
                wf.writelines(line)

default_erwin_style_cmf_dict = {
            'auxiliary files': 'gtapv7',
            'check-on-read elements': 'warn',
            'cpu': 'yes',
            'start with MMNZ': '500000000',
            'File GTAPSETS': '\"<^data_dir^>\SETS.har\"', 
            'File GTAPDATA': '\"<^data_dir^>\Basedata.har\"',
            'File GTAPPARM': '\"<^data_dir^>\Default.prm\"',
            # 'File GTAPSUPP': '\"<^data_dir^>\Basedata.har\"',
            'File GTAPSUPP': '\"<^data_dir^>\MapFile.har\"',
            'Updated File GTAPDATA': '\"<^output_dir^>\<^experiment_name^>.UPD\"',
            'File GTAPVOL': '\"<^output_dir^>\<^experiment_name^>-VOL.har\"',
            'File WELVIEW': '\"<^output_dir^>\<^experiment_name^>-WEL.har\"',
            'File GTAPSUM': '\"<^output_dir^>\<^experiment_name^>-SUM.har\"',
            'Solution File': '\"<^output_dir^>\<^experiment_name^>.sl4\"',
            'log file': '\"<^output_dir^>\<^experiment_name^>.log\"',
            'Method': 'Gragg',
            'Steps': '2 4 6',
            'Exogenous': [
                'pop',
                'psaveslack' ,
                'pfactwld',
                'profitslack',
                'incomeslack ',
                'endwslack',
                'cgdslack',
                'tradslack',
                'ams' ,
                'atm',
                'atf',
                'ats',
                'atd',
                'aosec',
                'aoreg',
                'avasec',
                'avareg',
                'aintsec',
                'aintreg',
                'aintall',
                'afcom',
                'afsec',
                'afreg',
                'afecom',
                'afesec',
                'afereg',
                'aoall',
                'afall',
                'afeall',
                'au',
                'dppriv',
                'dpgov',
                'dpsave',
                'to',
                'tinc',
                'tpreg',
                'tm',
                'tms',
                'tx',
                'txs',
                'qe',
                'qesf',
            ], # DONT FORGET REST ENDOGENOUS STATMENT
            'Verbal Description': 'verbal_description_default_text',
            'xSets': {'AGCOM': ['Agri commodities', '(pdr, wht, gro, v_f, osd, c_b, pfb, ocr, ctl, oap, rmk, wol)'],
                      'AGCOM_SM' : ['smaller agri commodities', '(pdr, wht, gro)'],       
                      },
            # SUGGESTED CHANGE: Use xSet [name of set] read elements from file [file path] header "FOUR LETTER HEADER";
            'xSubsets': ['AGCOM is subset of COMM', 'AGCOM is subset of ACTS', 'AGCOM_SM is subset of COMM', 'AGCOM_SM is subset of ACTS'],
            # xSubset now will be defined based on the same xset file.
            
            
            'Shock': {'name': 'agri_productivity increases 20p',
                      'shortname': 'agpr20',
                      'shock_string': 'Shock aoall(AGCOM_SM, reg) = uniform 20;'}
                
    }


def generate_erwin_style_gtap_cmf_file_for_scenario(inputs_dict, 
                                                    experiment_name,                                                    
                                                    data_dir,
                                                    output_dir,     
                                                    generated_cmf_path,                                                
                                                    # model_dir='..\\mod', 
                                                    # solution_output_dir='..\\out', 
                                                    # cmfs_dir='.\\cmf', 
                                                    # data_dir='..\\data', 
                                                    # aggregation_name='65x141'
                                                    ):
    
    reserved_keys = ['xSets', 'xSubsets', 'Exogenous', 'Shock']
    output_list = []
    for k, v in inputs_dict.items():
        if k not in reserved_keys:
            if '<^experiment_name^>' in v:   
                v = v.replace('<^experiment_name^>', experiment_name)  
            if '<^data_dir^>' in v:
                v = v.replace('<^data_dir^>', data_dir)           
            if '<^output_dir^>' in v:
                v = v.replace('<^output_dir^>', output_dir)         
            output_list.append(str(k) + '=' + str(v) + ';\n')
                
        elif k == 'xSets':
            for xset_name, xset_values_list in v.items():                    
                output_list.append('xSet ' + str(xset_name) + ' # ' + str(xset_values_list[0]) + ' # ' + str(xset_values_list[1]) + ';\n')
        elif k == 'xSubsets':
            for xSubset_name in v:                    
                output_list.append('xSubset ' + str(xSubset_name) + ';\n')
        elif k == 'Exogenous':
            if isinstance(v, list):
                output_list.append('Exogenous\n')
                for i in v:
                    output_list.append('          ' + str(i) + '\n')
                output_list.append('          ;\n')
                output_list.append('Rest endogenous;\n')
        elif k == 'Shock':
            output_list.append(v['shock_string'])
                   
    with open(generated_cmf_path, 'w') as wf:
        for line in output_list:
            wf.writelines(line)

    
    # hb.write_to_file(output_list, generated_cmf_path)                   
        
    
    # -p1=%DATd% -p2=%SOLd%
    5



### NOTE: I made a bunch of CMD line calls to extract hars before finding HARPY. These are kept with the suffix _cmd as reference
def har_to_indexed_dfs(input_har_path, output_index_path):
    """Convert all information in input_har_path into several CSVs that can be programatically rewritten back to a conformant har.
    All paths are written relative to output_index_path (either parallel to or in a newly created output dir)
    
    """
    output_dir = os.path.split(output_index_path)[0]
    label = hb.file_root(output_index_path)
    
    # Will write a separate _index.csv file for fast access of underlying HAR entries, each saved as their own csv.
    har_index_path = output_index_path
    har_sets_path = os.path.join(output_dir, label + '_sets.csv')
    har_csv_dir = os.path.join(output_dir, label)
    hb.create_directories(har_csv_dir)
    
    # Read the harfile using harpy
    InFile = harpy.HarFileObj(input_har_path)

    # Get a list of all headers to iterate through 
    HeadsOnFile = InFile.getHeaderArrayNames()
    
    # Define the data recorded to the har_index
    header_data = {
        'header': [],
        'long_name': [],
        'shape': [],
        'dim_names': [],
        'ndims': [],
        'dtype': [],
        'coefficient_name': [],
        }
    
    # Iterate through individual HAR entires
    set_names_dict = {}
    for header in HeadsOnFile:
        
        # Get a specific header from the InFile
        DataHead = InFile[header]
        
        # Draw most values from the actual array (to avoid problems with missing set names etc.)
        shape = DataHead.array.shape
        ndims = len(DataHead.array.shape)
        dtype = DataHead.array.dtype
        
        if 'float' in str(dtype):
            dtype = 'RE'
        
        # Record to the data structure.
        header_data['header'].append(header)
        header_data['long_name'].append(DataHead.long_name)
        
        # Render shape string.
        # in addition to the python types, like shape, there is also the string type for what is written the CSV. This is for pretty rendering but won't load programatically.
        # Maybe I should add a column for python types?
        shape_string = str(shape).replace('(','').replace(')','').replace(' ','')
        if shape_string.endswith(','):
            shape_string = shape_string[:-1]        
        # Kept with HARfile notation that dimensions are split with *
        shape_string = shape_string.replace(',', '*')            
        if shape_string == '':
            shape_string = 1            
        header_data['shape'].append(shape_string)
        
        # IMPORTANT HARFILE NOTE:         
        """Sets are stored on Header Array files as part of real matrices, or individually. In the latter case, the set is stored as an array of strings.
        The difficulty for ViewHAR is to guess (a) which string arrays contain sets, and (b) what are the names of these sets.
        The longname part of the header should be used to record the name of the set, according to the following convention:
        Set IND sectors in the model
        The first word should be Set and the second word should be the name of the set. ViewHAR ignores all words after the first two.
        """
        
        # Extract dim_names. a little harder because not all entries in a HAR file have a value here.           
        if len(DataHead.setNames) > 0:
            if DataHead.setNames[0] and DataHead.setElements[0]: # check that it's not just a list of None
                for c, set_name in enumerate(DataHead.setNames):
                    if set_name in set_names_dict:
                        assert DataHead.setElements[c] == set_names_dict[set_name] # sets with same names have to have same elements.                        
                    elif len(DataHead.setElements) != len(DataHead.setNames):
                        raise NameError('There should be exactly 1 set name for each setElements list.')
                    else:
                        set_names_dict[set_name] = DataHead.setElements[c]
                try:  
                    fstring = str(DataHead.setNames)[1:-1].replace('\'', '').replace(',', '*').replace(' ', '')
                    header_data['dim_names'].append(fstring) # The string manipulation here makes it look nice in excel
                except:
                    header_data['dim_names'].append('')  
            else:
                header_data['dim_names'].append('')  
        else:
            header_data['dim_names'].append('')  
        

        if isinstance(DataHead.setElements, list):   
            if len(DataHead.setNames) > 0:
                if DataHead.setNames[0] is not None:
                    dim_names = [i for i in DataHead.setNames]
                else:
                    dim_names = []
            else:
                dim_names = []
        else:
            if DataHead.setNames is not None:
                dim_names = str(DataHead.setNames).replace(',', '*').replace(' ', '')
            else:
                dim_names = ''
                
        # Finish adding other attributes to the data dict.     
        header_data['ndims'].append(ndims)
        header_data['dtype'].append(dtype)
        header_data['coefficient_name'].append(DataHead.coeff_name)
        
        # Separate from writing the index.csv, we also will write the specific headers csv spreadsheet. This is straightforward
        # for 1 and 2 dimensions, but for three+ we need to stack vertically
        current_header_data_path = os.path.join(har_csv_dir, header + '.csv')
        
        implied_numpy_type = ''
        
            
        
        skip = False
        if len(shape) == 0:
            # row_index = DataHead.setElements[0]
            columns = [header]
            data_array = np.asarray([[DataHead.array]]).T # Pandas requires it to be a 2d array to write, even tho singular
            
            # Test to see if it can be coerced into a float or int
            try:
                nt = np.float32(data_array[[0]])
                implied_numpy_type = 'float32'
            except:
                print ('unable to coerce')
            
            df_data = pd.DataFrame(index=row_index, columns=columns, data=data_array)
            df_data.to_csv(current_header_data_path, index=False)
        elif len(DataHead.array.shape) == 1:     
            row_index = DataHead.setElements[0]
            columns = [header]   
            data_array = np.asarray([DataHead.array]).T # Pandas requires it to be a 2d array to write, even tho 1dim
            # Test to see if it can be coerced into a float or int
            
            try:
                nt = np.float32(data_array[0, 0])
                
                implied_length = len(data_array[0, 0])
                if implied_length == 12:
                    implied_numpy_type = np.float32
                elif implied_length > 12:
                    implied_numpy_type = np.float64
                else:
                    implied_numpy_type = np.in64
                
            except:
                implied_numpy_type = 'string'
                if isinstance(data_array[0, 0], np.float32):
                    implied_length = 12
                elif isinstance(data_array[0, 0], np.float64):
                    implied_length = 24 # No clue what it actually is for har or if it is even possible.
                else:   
                    implied_length = len(data_array[0, 0])
            
            
            
            df_data = pd.DataFrame(index=row_index, columns=columns, data=data_array)
            
            dtype_dict = {header: implied_numpy_type}
            
            df_data.to_csv(current_header_data_path, index=False)
        elif len(DataHead.array.shape) == 2:          
            row_index = DataHead.setElements[0]
            columns = DataHead.setElements[1]
            data_array = np.asarray(DataHead.array) # Pandas requires it to be a 2d array to write
            df_data = pd.DataFrame(index=row_index, columns=columns, data=data_array)
            df_data.to_csv(current_header_data_path)
        elif len(DataHead.array.shape) >= 3:   
            
            ### When going beyond 2 dims, add the leading n - 2 dimensions as a stacked multiindex.    
            # All but the last index will be uses as nested row_indices
            row_indices = [i for i in DataHead.setElements[:-1]]
            
            # The last index will be columns
            columns = DataHead.setElements[-1]            
             
            # Read the raw 3dim array
            data_array = np.asarray(DataHead.array) 
            
            # The number of rows will be equal to the product of the length of all row indices
            n_index_rows = math.prod([len(i) for i in row_indices])
            
            ### Reshape the array to be 2d so that it is n_index_rows tall and n_cols across
            # LEARNING POINT: -1 notation just means "whatever's left over". So when we take a 65*65*141 array and reshape using m * n = 4225, we call .reshape(4225, -1) which would have been equivilent to .reshape(4225, 141) but is more flexible.        
            array_2d = data_array.reshape(n_index_rows, -1)
            
            # Create a pandas multiindex from the product of the two row indices.
            row_multi_index = pd.MultiIndex.from_product(row_indices, names=dim_names[:-1])
                             
            # Create the dataframe
            df_data = pd.DataFrame(index=row_multi_index, columns=columns, data=array_2d)

            df_data.to_csv(current_header_data_path)


    
    # Only the base data seems to be distributed with Sets.har files. sl4s do not have this. Thus only run if there is something that populates set_names_dict
    if len(set_names_dict) > 0:
        for set_name, set_elements in set_names_dict.items():
            print('Adding header for set: ', set_name, set_elements)


            header_data['header'].append(set_name)
            header_data['long_name'].append('Set ' + set_name) # NOTE: This is literally defined by ViewHAR and is used in TABLO that the first word set means the second word set_name is a set. Subsequent words are ignored.
            header_data['shape'].append(len(set_elements))
            header_data['dim_names'].append(set_name)
            header_data['ndims'].append(1)
            header_data['dtype'].append('<U12')
            header_data['coefficient_name'].append('')
        
            columns = [set_name]   
            data_array = np.asarray(set_elements).T # Pandas requires it to be a 2d array to write, even tho 1dim
            # Test to see if it can be coerced into a float or int
            
            # try:
            #     nt = np.float32(data_array[0, 0])
                
            #     implied_length = len(data_array[0, 0])
            #     if implied_length == 12:
            #         implied_numpy_type = np.float32
            #     elif implied_length > 12:
            #         implied_numpy_type = np.float64
            #     else:
            #         implied_numpy_type = np.in64
                
            # except:
            #     implied_numpy_type = 'string'
            #     if isinstance(data_array[0, 0], np.float32):
            #         implied_length = 12
            #     elif isinstance(data_array[0, 0], np.float64):
            #         implied_length = 24 # No clue what it actually is for har or if it is even possible.
            #     else:   
            #         implied_length = len(data_array[0, 0])
            
            
            # df_data = pd.DataFrame(index=row_multi_index, columns=columns, data=array_2d)
            
            current_header_data_path = os.path.join(har_csv_dir, set_name + '.csv')
            df_data = pd.DataFrame(columns=columns, data=data_array)
            df_data.to_csv(current_header_data_path, index=False)
            
    df_index = pd.DataFrame(data=header_data)
    df_index.to_csv(har_index_path, index=False)
        
        # hb.python_object_to_csv(set_names_dict, har_sets_path, csv_type='2d_odict_list')
        # df_sets = pd.DataFrame(data=set_names_dict) # Can't export with pandas cause different lengths


def indexed_dfs_to_har(input_indexed_dfs_path, output_har_path):
    index_df_dir, index_df_filename = os.path.split(input_indexed_dfs_path)
    index_name = os.path.splitext(index_df_filename)[0]
    index_df = pd.read_csv(input_indexed_dfs_path, index_col=0)
    
    # Prior to writing ARRAYS, we need to load the relevant sets to get the actually n-dim array shape.
    # So iterate through the headers looking for things that follow the ViewHAR convention of the long_name being Set NAME   
    sets_data = {} 
    for header in index_df.index:
        index_df_row = index_df.loc[index_df.index == header]
        long_name = index_df_row['long_name'].values[0]
        split_name = long_name.split(' ')
        if len(split_name) >= 2:
            annotation = split_name[0]
            possible_set_label = split_name[1]
            if annotation == 'Set':
                print('Found set', possible_set_label)
                set_data = pd.read_csv(os.path.join(index_df_dir, index_name, possible_set_label + '.csv'))
                sets_data[possible_set_label] = list(set_data.values)
    # Create a new Harfile object.
    Harfile = harpy.HarFileObj(output_har_path)
    
    # Based on the headers listed in the index CSV, add new headers to the Harfile.
    for header in index_df.index:
        skip_write = False
        data_df_path = os.path.join(index_df_dir, index_name, header + '.csv')
        if hb.path_exists(data_df_path) and len(header) <= 4: # Note that HAR Files are hard-coded to assume this is 4. However, some dimension labels in GTAP are more than 4, like TARTYPE, so you cant infer between each other. 
            
            if header == 'STIM':
                print('header', header)

            # Load columns of index_df for use in writing.
            index_df_row = index_df.loc[index_df.index == header]
            long_name = index_df_row['long_name'].values[0]
            shape = index_df_row['shape'].values[0]
            dtype = index_df_row['dtype'].values[0]
            coefficient_name = index_df_row['coefficient_name'].values[0]  
            if not isinstance(coefficient_name, str):
                coefficient_name = ''
            dim_names = list(index_df_row['dim_names'].values)            
            
            # Given the name of the header as loaded from the index, open the CSV file that contains that header's data.
            
            data_df = pd.read_csv(data_df_path)
            
            if len(data_df) == 0:
                skip_write = True
            set_elements = []           
            if len(dim_names) > 0:
                
                # Some headers don't have dim names which pandas will interpret as nans. Check for that.
                try: 
                    tried_is_nan = math.isnan(dim_names[0])
                except: 
                    tried_is_nan = False
                    
                # If it's not nan, can read it directly
                if not tried_is_nan:
                    
                    # To keep with the look-and-feel of ViewHAR, dimensions are stored split with a *
                    dim_names_as_list = dim_names[0].split('*')
                    for dim_name in dim_names_as_list:
                        cur_col = sets_data[dim_name]
                        cur_values = [i[0] for i in cur_col]
                        set_elements.append(cur_values)
                        # as_list = list(sets_df[dim_name].dropna()) # NOTE Because this was loaded from non-rectangular DF, need to manually trim the excess rows once we know what column it is.
                        # set_elements.append(as_list)
                    zipped =  dict(zip(dim_names_as_list, set_elements)) # BUG!!! Dicts cannot store duplicate strings in their keys. Thus it drops the second REG in COMM*REG*REG. # Fixed this by not getting the shapes from the zipped, but instead directly from the elements list.               
                    implied_shape = [len(i) for i in set_elements]
                    
                    # Because indices past dim1 are recorded as columns, need to add them on to the before the reshape and then drop
                    # FUTURE feature: I Think I could have just used pandas multi-indexes more smartly to avoid having to store dimensions as data.                
                    if len(implied_shape) >= 2:
                        n_indices_stored_in_rows = len(implied_shape) - 1
                    else:
                        n_indices_stored_in_rows = 0
            
                    # Select the non-dimension data
                    unshaped = data_df.values 
                    unshaped = unshaped[:, n_indices_stored_in_rows:]
                    
                    # Reshape it to the shape implied by the list of elements.
                    array_entries = unshaped.reshape(implied_shape)

                # If it is nan, just store a blank value for now.
                else:
                    dim_names_as_list = ['']
                    if header in data_df:
                        array_entries = data_df[header].values
                        set_elements = None
                        dim_names = ['']
                        zipped = None
                    else:
                        try:
                            array_entries = data_df['0'].values
                            set_elements = None
                            dim_names = ['']
                            zipped = None     
                            skip_write = True        
                        except:
                            array_entries = np.float32(0)
                            set_elements = None
                            dim_names = ['']
                            zipped = None       
                            
                            skip_write = True                 

            else:
                raise NameError('wtf')   
                     
            verbose = True
            if verbose:
                print ('indexed_dfs_to_har is writing ' + str(header), dtype)
                
            # Create a new Header object from the values loaded from the DFs. 
            har_dtype = dtype
            if not skip_write:
                Header = harpy.HeaderArrayObj.HeaderArrayFromData(array_entries, coefficient_name, long_name, dim_names_as_list, zipped, har_dtype)
                    
                # Add it to the Harfile object
                Harfile[header] = Header
        
    # Once all headers have been added, write it to disk.
    Harfile.writeToDisk()

        
def har_to_txt_cmd(input_har_path, output_txt_path, gempack_utilities_dir=None):
    """Write context of har_path to txt_path. Returns nothing. if gempack_utilities_dir=None, use the default."""
    if gempack_utilities_dir is None:
        gempack_utilities_dir = os.path.join('C:\\', 'GP')
    command = os.path.join(gempack_utilities_dir, 'har2txt.exe') + ' ' + input_har_path + ' ' + output_txt_path
    os.system(command)

def har_to_txt_stloht(input_har_path, output_csv_path, gempack_utilities_dir=None):
    """Write context of har_path to txt_path. Returns nothing. if gempack_utilities_dir=None, use the default."""
    if gempack_utilities_dir is None:
        gempack_utilities_dir = os.path.join('C:\\', 'GP')
    command = os.path.join(gempack_utilities_dir, 'sltoht.exe') + ' ' + input_har_path + ' ' + output_csv_path
    os.system(command)


def get_headers_from_har(har_path=None, txt_path=None, gempack_utilities_dir=None, output_headers_path=None):
    """  DEPRACATED cause it used a silly file-creation logic. Kept for humor value.    
    
    Uses har2txt facility to write a fill txt file of the har, uses the txt to get the headers, then returns them.
    If output_txt_path is specified, will check to see if that exists alread and read from there instead of the har_path.
    If output_headers_path is set, writes a simple txt file with 1 header per line.
    In retrospect, this probably wasnt that useful. Should have just always called 
    """
    if har_path is None and txt_path is None:
        raise NameError('At least one of har_path and txt_path must be defined.')
    
    if gempack_utilities_dir is None:
        gempack_utilities_dir = os.path.join('C:\\', 'GP')        
    
    if txt_path is None:
        txt_path = hb.path_replace_extension(har_path, 'txt')
        
    if har_path is None:
        if hb.path_exists(txt_path):
            har_string = hb.read_path_as_string(txt_path) 
        else:
            raise NameError('txt_path ' + str(txt_path) + ' does not exist from call get_headers_from_har.')
    else:
        if not hb.path_exists(har_path):
            raise NameError('har_path ' + str(har_path) + ' does not exist from call get_headers_from_har.')
        
        if hb.path_exists(txt_path):
            har_string = hb.read_path_as_string(txt_path) 
        else:
            command = os.path.join(gempack_utilities_dir, 'har2txt.exe') + ' ' + har_path + ' ' + txt_path
            os.system(command)
            har_string = hb.read_path_as_string(txt_path) 
            
    headers_list = get_headers_from_hartxt_path(har_string, output_headers_path)
    
    return headers_list
                
def get_headers_from_hartxt_path(hartxt_path, headers_txt_path=None):

    har_string = hb.read_path_as_string(hartxt_path) 
    
    headers = []    
    for line in har_string.split('\n'):
        if line.endswith(';'):
            headers.append(line)

    if headers_txt_path:
        with open(headers_txt_path, 'w') as fp:
            for line in headers:
                fp.write(str(line) + '\n')
                
    return headers

def hartxt_to_dataframe(hartxt_path, csv_path=None):  
    """ABANDONED IN FAVOR OF HARPY"""
    har_string = hb.read_path_as_string(hartxt_path) 
    headers = get_headers_from_hartxt_path(hartxt_path)
    
    for line in har_string.split('\n'):
        if line.endswith(';'):
            
            # First split on key phrase header
            left_of_header, right_of_header = line.split(' Header ')
            left_list = left_of_header.split(' ')
            
            if 'Strings' in left_list:
                shape = [int(left_list[3])]
            elif 'Real' in left_list:
                left_of_real = []
                for i in left_list:
                    if i != 'Real':
                        left_of_real.append(str(i))
                    else:
                        break
                shape = [int(i) for i in left_of_real]
            else:
                raise NameError('There apparently are more types in hars than I found.')
            
            
            right_list = right_of_header.split(' ')
            header_name = right_list[0].replace('\"', '')
            long_name = ' '.join([i.replace('\"', '') for i in right_list[2:]])
            
            # headers.append(line)
    
    
def extract_sets_and_correspondences_from_har(input_har_path, output_txt_path):
    

    """From HAR documentation:
    
    Data for economic models (for example, input-output tables or parameters such as elasticities) are often held within GEMPACK on files called Header Array or HAR files. Header Array files contain one or more arrays each containing data values. An individual array of data on a Header Array file is accessed by supplying the unique 4-character identifier (or Header) for that array of values.

The data values held by an individual array can be either all real numbers, all integer numbers or all character strings. Depending on the type of data that is to be stored, the number of dimensions allowed varies.

The dimension limits for Header Arrays are:

•For real numbers: up to and including 7 dimensions

•For integer numbers: up to and including 2 dimensions

•For character strings: only one dimensional arrays are allowed, each string being of the same length

Headers for arrays on any one file must be unique since the header acts as a label or primary key to identify the associated array.

Once written, an array contains not just the numbers in the array itself but also self-describing data, including the type of data values, dimensions of the array and a descriptive "long name" of up to 70 characters. There may also be labels for the set elements, which appear in ViewHAR as row and column labels.

Header Array files have the advantage that you can access any array just by referring to the header which uniquely identifies the array in the file. The format of the arrays and other details are all taken care of automatically by the software.

Headers consist of four characters which are usually letters (A to Z, a to z) or digits (0 to 9). Different arrays must have different headers. The case (upper or lower) of a header is not significant. (For example, you cannot have one array with header 'ABCD' and another on the same file with header 'AbCd'.). Headers starting with letters 'XX' are reserved for internal program use; an error is returned if you choose a header starting with 'XX'.


    """


    command = 'c:\\GP\\har2txt.exe ' + input_har_path + ' ' + output_txt_path
    os.system(command)

    # START HERE: able to extract data using har2txt. Now just use that to make a new mapping file. Maybe get it by reading the aggregation file? Identify the canonical REG ACT SET etc and then the subset of smaller to rewrite a new har?

def write_sets_and_correspondences_txt_to_har(input_txt_path, output_har_path):


    command = 'c:\\GP\\txt2har.exe ' + input_txt_path + ' ' + output_har_path
    os.system(command)
    
    
    
    
def directory_of_hars_to_indexed_dfs(input_dir, output_dir=None, produce_hars_from_csvs=None, verbose=True):
    
    if output_dir is None:
        output_dir = input_dir
    hb.create_directories(output_dir)
    
    hars_to_look_for = hb.list_filtered_paths_nonrecursively(input_dir, include_extensions='.har')
    
    for har_filename in hars_to_look_for:
        if verbose:
            print('Extracting ' + str(har_filename))
            
        # Write har to CSVs
        har_index_path = os.path.join(output_dir, hb.file_root(har_filename) + '.csv')     
         
        if hb.path_exists(har_filename):
            hb.create_directories(output_dir)
            
            if not hb.path_exists(har_index_path, verbose=True): # Minor note, could add more robust file validation to check for ALL the implied files to exist.
            
                # Extract the har to the indexed DF format.
                har_to_indexed_dfs(har_filename, har_index_path)       
                
            if produce_hars_from_csvs:

                # For validation (and actual use in the model), create a new har from the indexed dir.
                validation_har_path = hb.path_replace_extension(har_index_path, '.har')
                indexed_dfs_to_har(har_index_path, validation_har_path)


def get_set_labels_from_index_path(input_path):
    df = pd.read_csv(input_path)
    df_sets = df.loc[df.long_name.str.startswith('Set ')]
    set_labels = list(df_sets['header'])
    
    return set_labels