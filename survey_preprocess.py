#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 6 16:25:39 2023

@author: beauz
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # suppress the harmless but annoying warnings that come with pd.append()

import os
import pandas as pd
import copy
import functions.preprocessing_functions as funk

output_path = '/Users/beauz/Library/CloudStorage/GoogleDrive-zhangb42@msu.edu/My Drive/SBRI Outputs/'

base_21, factors_21 = funk.basic_cleaning(year=2021, filename='SBRI_2021_raw.csv', subfolder='raw')
base_22, factors_22 = funk.basic_cleaning(year=2022, filename='SBRI_2023_Feb_14.csv', subfolder='raw')

# funk.get_file(truncated_base_df=truncated_base_df, select='Matrix', subfolder='output')
scores_to_eval = ['Comp_Hope', 'Comp_Trust', 'Comp_Tech', 'Comp_Belonging', 'OP_Total', 'Index', 'Metric']
var_21 = funk.create_latent_variables(path=output_path, df=base_21, score_list=scores_to_eval, subfolder='output_21')
var_22 = funk.create_latent_variables(path=output_path, df=base_22, score_list=scores_to_eval, subfolder='output')

[ev_21, vec_21, loading_21] = funk.custom_factor_analysis(df=base_21, rotation='varimax', extraction='principal')
[ev_22, vec_22, loading_22] = funk.custom_factor_analysis(df=base_22, rotation='varimax', extraction='principal')

with pd.ExcelWriter(os.path.join(os.getcwd(), 'SBRI_21&22_comparisons.xlsx')) as writer:
    base_21.to_excel(excel_writer=writer, sheet_name='raw_21')
    base_22.to_excel(excel_writer=writer, sheet_name='raw_22')
    var_21.to_excel(excel_writer=writer, sheet_name='latent_variables_21')
    var_22.to_excel(excel_writer=writer, sheet_name='latent_variables_22')