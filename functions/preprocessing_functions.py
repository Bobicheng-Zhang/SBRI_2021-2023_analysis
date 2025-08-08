#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 6 16:29:41 2023

@author: beauz
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import copy
import scipy.stats as stats
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)  # suppress the harmless but annoying warnings that come with pd.append()

def figure_customization(object):
    """
    Adds some very specific customized settings to plots such as color, font size, legends handles, etc. This function does not actually plot, just adds flair.
    Arguments:
        object: any Matplotlib subplot (axes) object

    Returns:

    """

    object.spines[['top', 'right']].set_visible(False)
    object.spines[['left', 'bottom']].set_linewidth(3)
    object.xaxis.set_tick_params(width=3)
    object.yaxis.set_tick_params(width=3)
    object.tick_params(axis='both', which='major', labelsize=24)


def basic_cleaning(year, filename, subfolder, exclusion_criterion=None):
    """
    Read in SBRI data and perform some basic cleaning. First, a copy of the questions and their corresponding labels used for analysis are saved; personal
    information is removed here and for all later stages of analysis. Second, for all questions relating to demographic information, responses provided are
    extracted and appended to each respondent's row with empty cells removed such that for each demographic category of each respondent (row),
    there is only one cell that includes all the options that respondent selected, one or otherwise. Then all the columns from which these data are
    extracted are dropped from the output. Third, questions with Likert-scale response options (i.e. all questions with 'matrix' in the label) are coded (
    dummy, effect, reverse or otherwise). Lastly, if a value is provided to exclusion criterion, respondent who missed more than the provided number of
    questions will be dropped before returning the outputs.
     
    Arguments
     year (int): year in which the survey was distributed; this was added because the date in CSV filename is always the save data, not necessarily in the
     same year
     filename (string): absolute name of the data file in single quotes
     subfolder (string): absolute path where any exports of this function will be stored
     exclusion_criterion (int): none by default; if a reasonable number is given, then the to-be-returned base_df will exclude based on this argument

    Return
     base_df (Pandas DataFrame): the new base data structure for most of the future analyses, with fewer, but all numerically coded columns
     factors_df (Pandas DataFrame): the data structure with only the matrix questions that could be used for later factor analysis

    Export
     q_labels_(year): all the question texts removed from base_df and saved in a CSV
     emails_(year): just emails, nothing else

    """

    if subfolder not in os.listdir(os.getcwd()):
        os.mkdir(subfolder)

    # simplify the data set by removing data not used for analysis
    raw_df = pd.read_csv(filename)  # read in the raw .csv file and store as a df
    question_texts = raw_df.iloc[0]  # save the first row in case we need it
    question_texts.to_csv(os.path.join(os.getcwd(), subfolder, 'q_labels_' + str(year) + '.csv'))  # save a copy of all the question and corresponding labels
    if year == 2022:  # only do this for 22 b/c the raw 21 data did not have the question texts as this row
        base_df = raw_df.tail(-1)  # then drop it since it's not actual data
        emails = base_df[base_df['Q82'].notna()]
        emails['Q82'].to_csv(os.path.join(os.getcwd(), subfolder, 'emails_' + str(year) + '.csv'))  # save a copy of those who left their email addresses
    # base_df = base_df.drop(['Q82'], axis=1)  # drop the column with email addresses

    # these are some cleaning done during exploration of 22 data including locating duplicates and various grant groups
    # ==========================================================================================================================================================
    # for col in base_df.columns:
    #     values_per_col = base_df.loc[:, col].unique()
    #     if len(values_per_col) == 1:  # if a column only has one value
    #         base_df = base_df.drop(col, axis=1)  # drop the column
    #
    # no_id_df = copy.deepcopy(base_df)
    # # no_id_df = no_id_df.drop(['IPAddress', 'LocationLatitude', 'LocationLongitude'], axis=1)
    # no_id_df = no_id_df.drop(['LocationLatitude', 'LocationLongitude'], axis=1)
    # no_id_df = no_id_df.drop(no_id_df.iloc[:, 6:18], axis=1)
    # no_id_df = no_id_df.drop(['Q96', 'respondent_group', 'Q_DataPolicyViolations'], axis=1)
    # no_id_df.to_csv(os.path.join(os.getcwd(), subfolder, 'SBRI_2022_No_ID_raw.csv'))
    # # add a little future-proofing
    # if len(base_df.loc[:, 'IPAddress'].unique()) <= len(base_df):  # if not every respondent has a unique IP address
    #     duplicates_df = base_df[base_df.loc[:, 'IPAddress'].duplicated(keep=False)]  # find those duplicate IP addresses
    #     duplicates_df = duplicates_df.sort_values(by='IPAddress')
    #     duplicates_df.to_csv(os.path.join(os.getcwd(), subfolder, 'duplicates_both_times.csv'))  # save to csv
    #     print('Warning: repetitive IP address found, starting from row number ' + str(duplicates_df.index[0]) + ' ; duplicates saved to subfolder.')

    # find those 59 respondents who should be excluded
    # idiots_who_lied = base_df[(base_df['Q5_taken EC Survey'] == 'No') & (base_df['respondent_group'] == 'ECSBSS2022_6')]
    # check for inconsistencies between group and grant year
    # ecs_1 = base_df[base_df.loc[:, 'respondent_group'] == 'ECSBSS2022_1']
    # ecs_2 = base_df[base_df.loc[:, 'respondent_group'] == 'ECSBSS2022_2']
    # ecs_3 = base_df[base_df.loc[:, 'respondent_group'] == 'ECSBSS2022_3']
    # ecs_4 = base_df[base_df.loc[:, 'respondent_group'] == 'ECSBSS2022_4']
    # ecs_5 = base_df[base_df.loc[:, 'respondent_group'] == 'ECSBSS2022_5']
    # ecs_6 = base_df[base_df.loc[:, 'respondent_group'] == 'ECSBSS2022_6']
    #
    # group_by_grant_year = pd.DataFrame({
    #     'group': ['ECSBSS2022_1', 'ECSBSS2022_2', 'ECSBSS2022_3', 'ECSBSS2022_4', 'ECSBSS2022_5', 'ECSBSS2022_6'],
    #     'only_2020': [len(ecs_1[(ecs_1['Q1_Grant Year_1'].notna()) & ecs_1['Q1_Grant Year_2'].isna()]),
    #                   len(ecs_2[(ecs_2['Q1_Grant Year_1'].notna()) & ecs_2['Q1_Grant Year_2'].isna()]),
    #                   len(ecs_3[(ecs_3['Q1_Grant Year_1'].notna()) & ecs_3['Q1_Grant Year_2'].isna()]),
    #                   len(ecs_4[(ecs_4['Q1_Grant Year_1'].notna()) & ecs_4['Q1_Grant Year_2'].isna()]),
    #                   len(ecs_5[(ecs_5['Q1_Grant Year_1'].notna()) & ecs_5['Q1_Grant Year_2'].isna()]),
    #                   len(ecs_6[(ecs_6['Q1_Grant Year_1'].notna()) & ecs_6['Q1_Grant Year_2'].isna()])],
    #     'only_2022': [len(ecs_1[(ecs_1['Q1_Grant Year_1'].isna()) & ecs_1['Q1_Grant Year_2'].notna()]),
    #                   len(ecs_2[(ecs_2['Q1_Grant Year_1'].isna()) & ecs_2['Q1_Grant Year_2'].notna()]),
    #                   len(ecs_3[(ecs_3['Q1_Grant Year_1'].isna()) & ecs_3['Q1_Grant Year_2'].notna()]),
    #                   len(ecs_4[(ecs_4['Q1_Grant Year_1'].isna()) & ecs_4['Q1_Grant Year_2'].notna()]),
    #                   len(ecs_5[(ecs_5['Q1_Grant Year_1'].isna()) & ecs_5['Q1_Grant Year_2'].notna()]),
    #                   len(ecs_6[(ecs_6['Q1_Grant Year_1'].isna()) & ecs_6['Q1_Grant Year_2'].notna()])],
    #     'both_years': [len(ecs_1[(ecs_1['Q1_Grant Year_1'].notna()) & ecs_1['Q1_Grant Year_2'].notna()]),
    #                    len(ecs_2[(ecs_2['Q1_Grant Year_1'].notna()) & ecs_2['Q1_Grant Year_2'].notna()]),
    #                    len(ecs_3[(ecs_3['Q1_Grant Year_1'].notna()) & ecs_3['Q1_Grant Year_2'].notna()]),
    #                    len(ecs_4[(ecs_4['Q1_Grant Year_1'].notna()) & ecs_4['Q1_Grant Year_2'].notna()]),
    #                    len(ecs_5[(ecs_5['Q1_Grant Year_1'].notna()) & ecs_5['Q1_Grant Year_2'].notna()]),
    #                    len(ecs_6[(ecs_6['Q1_Grant Year_1'].notna()) & ecs_6['Q1_Grant Year_2'].notna()])]})
    #
    # group_by_grant_year.to_csv(os.path.join(os.getcwd(), subfolder, 'group_by_grant_year.csv'))
    # ==========================================================================================================================================================

    demo_cols_to_drop = ['Demo_main_1', 'Demo_main_5', 'Demo2_main_2', 'Demo2_main_3', 'Demo2_main_5', 'Demo2_main_7', 'Demo_main_8']
    demo_df = pd.DataFrame()  # make a new df just to store all the shortened demo before assigning them to base
    for a in base_df.iloc:  # for each respondent
        demo_region = []
        demo_sector = []
        demo_degree = []
        demo_disable = []
        demo_race = []
        demo_gender = []
        demo_lgbtq = []

        for b in a.index:
            if pd.notna(a[b]):
                if 'Demo_main_1' in b:
                    demo_region += [a[b]]
                if 'Demo_main_5' in b:
                    demo_sector += [a[b]]
                if 'Demo2_main_2' in b:
                    demo_degree += [a[b]]
                if 'Demo2_main_3' in b:
                    demo_disable += [a[b]]
                if 'Demo2_main_5' in b:
                    demo_race += [a[b]]
                if 'Demo2_main_7' in b:
                    demo_gender += [a[b]]
                if 'Demo2_main_8' in b:
                    demo_lgbtq += [a[b]]

        demo_df = demo_df.append(dict(Region=demo_region, Industry=demo_sector, Education=demo_degree, Disability=demo_disable,
                                      Race=demo_race, Gender=demo_gender, LGBTQ=demo_lgbtq), ignore_index=True)

    for c in demo_cols_to_drop:
        if c in base_df.columns:
            base_df = base_df.drop(c, axis=1)

    base_df[['Region', 'Industry', 'Education', 'Disability', 'Race', 'Gender', 'LGBTQ']] = demo_df  # get the concatenated columns

    ## Create a df with just the matrix questions for later factor analysis
    matrix_cols = []
    for col in base_df.columns:
        if 'matrix' in col:
            matrix_cols += [col]
    factors_df = base_df[matrix_cols]
    # matrix_df.to_csv(os.path.join(os.getcwd(), subfolder, 'SBRI_Matrix.csv'))

    ## Drop those who did not respond to five or more questions
    qs_missed = []
    for resp in factors_df.iloc:
        qs_missed += [(resp.isna().sum())]
    base_df = base_df.assign(Questions_Missed=qs_missed)
    problem_children = base_df[base_df['Questions_Missed'] > exclusion_criterion]
    # problem_children.to_csv(os.path.join(os.getcwd(), subfolder, 'SBRI_' + str(year) + '_Problem_Children.csv'))
    base_df = base_df[base_df['Questions_Missed'] <= exclusion_criterion]
    base_df.to_csv(os.path.join(os.getcwd(), subfolder, 'SBRI_' + str(year) + 'analysis_base.csv'))

    return base_df, factors_df


def get_file(truncated_base_df, select, subfolder):
    """
    Spit out semi-cleaned, not coded data structures based on input to the filter argument.
    Arguments
     truncated_base_df (Pandas DataFrame):
     select (string): argument to specify which subset of the data, or what form of data to save/return
        this could be 'Demographics', 'Qualitative', 'OPB'
     subfolder (string): absolute path where any output of this function will be stored
    Return
    Save
     OPB_Qs_SBRI_2022: WFBN_tal_2_matrix_3, Bhea_Orgres_1_matrix_1 to Bhea_Orgres_1_matrix_5; all demo_main_1, demo_main_2
     SBRI_Demo: this file includes all demographic columns
     SBRI_Qual: this file includes all questions that required user-input other than selecting an option, e.g. text, list, open
    """
    if subfolder not in os.listdir(os.getcwd()):
        os.mkdir(subfolder)

    if select == 'Demographics':
        demo_cols = []
        for col in truncated_base_df.columns:
            if 'Demo' in col:
                demo_cols += [col]
        demo_df = truncated_base_df[demo_cols]
        demo_df.to_csv(os.path.join(os.getcwd(), subfolder, 'SBRI_Demo.csv'))

    if select == 'Qualitative':
        qual_cols = []
        for col in truncated_base_df.columns:
            if 'Demo' in col or 'TEXT' in col or 'list' in col or 'open' in col:
                qual_cols += [col]
        qual_df = truncated_base_df[qual_cols]
        qual_df.to_csv(os.path.join(os.getcwd(), subfolder, 'SBRI_Qual.csv'))

    if select == 'OPB':
        opb_cols = []
        for col in truncated_base_df.columns:
            if 'Demo' in col or 'Bhea_Orgres_1_matrix' in col:
                opb_cols += [col]

        opb_cols += ['WFBN_tal_2_matrix_3']
        opb_df = truncated_base_df[opb_cols]
        for row in opb_df.iloc:
            if pd.notna(row['Demo_main_3_mc_4_TEXT']) and len(row['Demo_main_3_mc_4_TEXT']) > 2:
                row['Demo_main_3_mc_4_TEXT'] = np.NaN

        opb_df.to_csv(os.path.join(os.getcwd(), subfolder, 'OPB_Qs_SBRI_2022.csv'))


def custom_factor_analysis(df, rotation, extraction):
    """
    Takes in one of the factor dfs and performs custom factor analysis to find the appropriate clusters of survey questions based on factor loadings
    Arguments
     df:
    """
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
    from factor_analyzer.factor_analyzer import calculate_kmo

    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    kmo_all, kmo_model = calculate_kmo(df)

    fa = FactorAnalyzer(n_factors=11, rotation=rotation, method=extraction)
    fa.fit(df)
    eigen_values, vectors = fa.get_eigenvalues()

    loadings = pd.DataFrame(fa.loadings_, index=df.columns)

    return [eigen_values, vectors, loadings]


def create_latent_variables(path, df, score_list, subfolder):
    """
    Code all columns with question labels with 'matrix' in the label; then reverse code select columns. Then performance for each respondent is scored based on
    their deviation from the mean.
    Args:
        path (string): absolute path to where the subfolder will be created if it doesn't already exist
        df (Pandas DataFrame):
        score_list (list):
        subfolder (string): name of the folder where generated outputs will be saved
    """

    df['Demo_main_3_mc'] = df['Demo_main_3_mc'].replace('Number', 'Physical')
    df['Demo_main_3_mc'] = df['Demo_main_3_mc'].replace('NA - Virtual Only', 'Virtual')

    # code likert-scale responses as numeric values
    df = df.replace('Strongly Agree', 5)
    df = df.replace('Strongly Disagree', 1)
    df = df.replace('Neither Agree nor Disagree', 3)
    df = df.replace('Agree', 4)
    df = df.replace('Disagree', 2)

    cols_to_reverse = ['Bhea_Orgres_1_matrix_4', 'Bhea_Orgres_1_matrix_5',
                       'Bhea_Opres_2_matrix_3', 'Bhea_Opres_5_matrix_2',
                       'Mind_belon_4_matrix_4', 'Mind_Creat_1_matrix_3']
    # cols_to_reverse = ['Bhea_Orgres_1_matrix_2', 'Bhea_Orgres_1_matrix_3',
    #                    'Bhea_Opres_2_matrix_3', 'Bhea_Opres_5_matrix_2',
    #                    'Mind_belon_4_matrix_4', 'Mind_Creat_1_matrix_3']
    for col in cols_to_reverse:
        df.loc[:, col] = 6 - df.loc[:, col]  # reverse code these columns

    effect_mapping = {'Very negatively affected': -2,
                      'Negatively affected': -1,
                      'Unaffected': 0,
                      'Positively affected (e.g., experiencing growth)': 1,
                      'Very positively affected': 2}
    cleaned_df = df.assign(V2_Econ_Condition=df['Adapt_main_2_mc'].map(effect_mapping))
    # cleaned_df = cleaned_df.assign(V4_COVID=cleaned_df['Adapt_main_2_mc'].map(effect_mapping))
    # cleaned_df.to_csv(os.path.join(os.getcwd(), subfolder, 'SBRI_2022_data_2021_scoring.csv'))

    comp_df = copy.deepcopy(cleaned_df)

    ## Compute V4 adapt
    cols_for_adapt_new = []
    for col in comp_df.columns:
        if 'Adapt_main_3_mc' in col and 'TEXT' not in col and '16' not in col:
            cols_for_adapt_new += [col]

    cols_for_adapt_barrier = []
    for col in comp_df.columns:
        if 'Adapt_main_4_mc' in col and 'TEXT' not in col:
            cols_for_adapt_barrier += [col]
    comp_df.loc[:, 'V4_Adapt_New'] = comp_df[cols_for_adapt_new].notna().sum(axis=1)  # V4_Adapt_New
    comp_df.loc[:, 'V4_Adapt_Barrier'] = comp_df[cols_for_adapt_barrier].notna().sum(axis=1)  # V4_Adapt_Barrier

    ## Compute SBRI metric
    # recode bhea sales and profit
    comp_df = comp_df.replace('15%+ Decrease', -4)
    comp_df = comp_df.replace('10-14% Decrease', -3)
    comp_df = comp_df.replace('5-9% Decrease', -2)
    comp_df = comp_df.replace('0-4% Decrease', -1)
    comp_df = comp_df.replace('Neither Up nor Down', 0)
    comp_df = comp_df.replace('0-4% Increase', 1)
    comp_df = comp_df.replace('5-9% Increase', 2)
    comp_df = comp_df.replace('10-14% Increase', 3)
    comp_df = comp_df.replace('15%+ Increase', 4)

    cols_for_tech_security = []
    for col in comp_df.columns:
        if 'Tech_tool_3_mc' in col and 'TEXT' not in col:
            cols_for_tech_security += [col]
    comp_df.loc[:, 'V3_Tech_Security'] = comp_df[cols_for_tech_security].notna().sum(axis=1)
    comp_df = comp_df.drop(cols_for_tech_security, axis=1)

    cols_for_hope = ['Bhea_Orgres_1_matrix_3', 'Bhea_Opres_5_matrix_1', 'Mind_Hope_3_matrix_1', 'Mind_Hope_3_matrix_2', 'Mind_Hope_3_matrix_3',
                     'Mind_FO_5_matrix_1', 'Mind_FO_5_matrix_2']
    # cols_for_hope = ['Bhea_Orgres_1_matrix_1', 'Bhea_Opres_5_matrix_1', 'Mind_Hope_3_matrix_1', 'Mind_Hope_3_matrix_2', 'Mind_Hope_3_matrix_3',
    #                  'Mind_FO_5_matrix_1', 'Mind_FO_5_matrix_2']
    cols_for_trust = ['WFBN_INFO_1_matrix_1', 'WFBN_INFO_1_matrix_2', 'Mind_Trust_2_matrix_1', 'Mind_Trust_2_matrix_2', 'Mind_Trust_2_matrix_3']
    cols_for_tech = ['Tech_rc_1_matrix_1', 'Tech_rc_1_matrix_2', 'Tech_rc_1_matrix_3', 'Tech_rc_2_matrix_1', 'Adapt_main_1_matrix_3']
    cols_for_belo = ['Mind_Creat_1_matrix_1', 'Mind_Creat_1_matrix_2', 'Mind_belon_4_matrix_1', 'Mind_belon_4_matrix_2', 'Mind_belon_4_matrix_3']
    cols_for_netw = ['WFBN_advnet_5_matrix_1', 'WFBN_advnet_5_matrix_2', 'WFBN_advnet_5_matrix_3']
    cols_for_eff = ['Bhea_Opres_2_matrix_1', 'Bhea_Opres_2_matrix_2', 'Tech_rc_2_matrix_2']
    cols_for_adapt = ['Adapt_main_1_matrix_2', 'Adapt_main_1_matrix_3', 'Adapt_main_1_matrix_4']
    # cols_for_adapt = ['Adapt_main_1_matrix_2', 'Adapt_main_1_matrix_3', 'Adapt_main_1_matrix_1']
    cols_for_turn = ['Bhea_Orgres_1_matrix_4', 'Bhea_Orgres_1_matrix_5']
    # cols_for_turn = ['Bhea_Orgres_1_matrix_2', 'Bhea_Orgres_1_matrix_3']
    cols_for_fund = ['Bhea_Opres_5_matrix_1', 'Bhea_Opres_5_matrix_2']
    cols_for_empl = ['WFBN_tal_2_matrix_1', 'WFBN_tal_2_matrix_2']
    cols_for_dest = ['Mind_Creat_1_matrix_3', 'Mind_belon_4_matrix_4']

    comp_df.loc[:, 'Comp_Hope'] = comp_df[cols_for_hope].sum(axis=1)
    comp_df.loc[:, 'Comp_Trust'] = comp_df[cols_for_trust].sum(axis=1)
    comp_df.loc[:, 'Comp_Tech'] = comp_df[cols_for_tech].sum(axis=1)
    comp_df.loc[:, 'Comp_Belonging'] = comp_df[cols_for_belo].sum(axis=1)
    cols_for_core = ['Comp_Hope', 'Comp_Trust', 'Comp_Tech', 'Comp_Belonging']

    comp_df.loc[:, 'Comp_Core_Total'] = comp_df[cols_for_core].sum(axis=1)
    comp_df.loc[:, 'Comp_OP_Network'] = comp_df[cols_for_netw].sum(axis=1)
    comp_df.loc[:, 'Comp_OP_Efficiency'] = comp_df[cols_for_eff].sum(axis=1)
    comp_df.loc[:, 'Comp_OP_Adapt'] = comp_df[cols_for_adapt].sum(axis=1)
    comp_df.loc[:, 'Comp_OP_Turnover'] = comp_df[cols_for_turn].sum(axis=1)
    comp_df.loc[:, 'Comp_OP_Funding'] = comp_df[cols_for_fund].sum(axis=1)
    comp_df.loc[:, 'Comp_OP_Employ'] = comp_df[cols_for_empl].sum(axis=1)
    comp_df.loc[:, 'Comp_OP_Dest'] = comp_df[cols_for_dest].sum(axis=1)
    cols_for_op = ['Comp_OP_Network', 'Comp_OP_Efficiency', 'Comp_OP_Adapt', 'Comp_OP_Turnover', 'Comp_OP_Funding', 'Comp_OP_Employ', 'Comp_OP_Dest']

    cols_for_index = cols_for_core + cols_for_op + ['Bhea_Opres_2_matrix_3']
    comp_df.loc[:, 'SBRI_Index_Total'] = comp_df[cols_for_index].sum(axis=1)
    comp_df.loc[:, 'SBRI_OP_CAP'] = comp_df[cols_for_op].sum(axis=1)
    comp_df.loc[:, 'SBRI_Metric'] = comp_df.loc[:, 'V4_Adapt_New'] + comp_df.loc[:, 'Bhea_Profit_4_list'] + \
                                    comp_df.loc[:, 'V3_Tech_Security'] + comp_df.loc[:, 'V2_Econ_Condition'] - comp_df.loc[:, 'V4_Adapt_Barrier']
    # comp_df.loc[:, 'SBRI_Metric'] = comp_df.loc[:, 'V4_Adapt_New'] + comp_df.loc[:, 'Bhea_Profit_4_list'] + \
    #                                 comp_df.loc[:, 'V3_Tech_Security'] + comp_df.loc[:, 'V4_COVID'] - comp_df.loc[:, 'V4_Adapt_Barrier']

    # comp_df.to_csv(os.path.join(os.getcwd(), subfolder, 'SBRI_2022_Comp_Scores.csv'))

    comp_df = comp_df.reset_index()
    for score in score_list:
        sd_score = comp_df.loc[:, score].std()
        mean_score = comp_df.loc[:, score].mean()
        eval = []

        for row in comp_df.iloc:
            if row[score] <= mean_score - 2 * sd_score:
                eval += ['Underperforming']
            elif mean_score - 2 * sd_score <= row[score] <= mean_score - sd_score:
                eval += ['Below Average']
            elif mean_score - sd_score <= row[score] <= mean_score + sd_score:
                eval += ['Average']
            elif mean_score + sd_score <= row[score] <= mean_score + 2 * sd_score:
                eval += ['Above Average']
            elif row[score] >= mean_score + 2 * sd_score:
                eval += ['Outperforming']
            elif pd.isna(row[score]):
                eval += ['NA']

        eval_s = pd.Series(eval)
        new_col = score + '_Eval'
        comp_df.loc[:, new_col] = eval_s
        comp_df.to_csv(os.path.join(os.getcwd(), subfolder, 'SBRI_2022_Comp_Scores.csv'))
        # comp_df.to_csv(os.path.join(os.getcwd(), subfolder, 'SBRI_2021_Comp_Scores.csv'))

        # comp_df[comp_df['Q82'].notna()].to_csv(os.path.join(os.getcwd(), subfolder, 'SBRI_2022_Comp_Scores_Email.csv'))

    return comp_df


def group_level_analysis(comp_22, comp_21, subfolder):
    """

    :param composite_variables:
    :param subfolder:
    :return:
    """

    ## Compare SBRI index from 2021 and 2022
    # stats.shapiro(df_2021['Index'])
    # stats.shapiro(composite_variables['SBRI_Index_Total'])
    # stats.mannwhitneyu(df_2021['Index'], composite_variables['SBRI_Index_Total'])
    #
    # fig, p = plt.subplots(ncols=2, figsize=(14, 8))
    # box_2021 = df_2021['Index'].to_list()
    # box_2022 = composite_variables['SBRI_Index_Total'].to_list()
    # p[0].boxplot([box_2021, box_2022], notch=False, widths=[len(box_2021) * 0.0005, len(box_2022) * 0.0005],
    #              patch_artist=True, boxprops=dict(lw=3, fc='w'),
    #              whiskerprops=dict(lw=3), medianprops=dict(lw=3, c='g'),
    #              meanprops=dict(markerfacecolor='r', markeredgecolor='r', markersize=10),
    #              capprops=dict(lw=3), showmeans=True, showfliers=False)
    # p[1].boxplot([box_2021, box_2022], notch=False, widths=[len(box_2021) * 0.0005, len(box_2022) * 0.0005],
    #              patch_artist=True, boxprops=dict(lw=3, fc='w'),
    #              whiskerprops=dict(lw=3), medianprops=dict(lw=3, c='g'),
    #              meanprops=dict(markerfacecolor='r', markeredgecolor='r', markersize=10),
    #              capprops=dict(lw=3), showmeans=True, showfliers=True)
    # for i in range(2):
    #     p[i].set_xticks([1, 2]), p[i].set_xticklabels(['2021', '2022'], fontsize=24)
    #     p[i].set_ylabel('SBRI Index Scores', fontsize=24)
    #     figure_customization(p[i])
    # plt.tight_layout()
    # plt.savefig(os.path.join(os.getcwd(), subfolder, 'SBRI_Index_Comparison_21~22.png'))

    ## Run correlation between SBRI metric and revenue
    no_na_22 = comp_22[(comp_22['Demo_main_6_list'].notna()) & (comp_22['SBRI_Metric'].notna())]
    no_na_21 = comp_21[(comp_21['Demo_main_6_list'].notna()) & (comp_21['SBRI_Metric'].notna())]
    revenue_mapping = {'Less than $99k': 1, '$100k - $199K': 2, '$200k - $299k': 3,
                       '$300k - $399k': 4, '$400k - $499k': 5, '$500k - $699k': 6,
                       '$700k - $1M': 7, '$1M - $1.49M': 8, '$1.5M - $1.9M': 9,
                       '$2M - $2.9M': 10, '$3M - $5.0M': 11, 'Greater than $5.0M': 12}
    no_na_22 = no_na_22.assign(Recoded_Revenue=no_na_22['Demo_main_6_list'].map(revenue_mapping))
    no_na_21 = no_na_21.assign(Recoded_Revenue=no_na_21['Demo_main_6_list'].map(revenue_mapping))

    no_na_22 = no_na_22[no_na_22['Recoded_Revenue'] < 11]
    no_na_21 = no_na_21[no_na_21['Recoded_Revenue'] < 11]

    no_na_22[['Recoded_Revenue', 'SBRI_Metric']].corr(method='spearman')
    mean_by_revenue = no_na_22.groupby(['Recoded_Revenue']).mean()
    mean_by_revenue_21 = no_na_21.groupby(['Recoded_Revenue']).mean()
    legend_elements = [Patch(fc='orange', ec='k', lw=2, label='2021'), Patch(fc='b', ec='k', lw=2, label='2022')]

    fig, q = plt.subplots(figsize=(16, 9))
    q.bar(mean_by_revenue.index, no_na_22.groupby(['Recoded_Revenue']).count()['SBRI_Metric'], fc='w', edgecolor='b', lw=4)
    q.bar(mean_by_revenue_21.index, no_na_21.groupby(['Recoded_Revenue']).count()['SBRI_Metric'], fc='w', edgecolor='orange', lw=3)
    q.set_yticks([100, 200, 300, 400]), q.set_yticklabels(['100', '200', '300', '400'], fontsize=18)
    q.set_ylabel('Number of Samples per Revenue Group', fontsize=24)
    q.tick_params(axis='both', which='major', labelsize=24)

    q2 = q.twinx()
    q2.plot(mean_by_revenue.index, mean_by_revenue['SBRI_Index_Total'], lw=4, c='b')
    q2.plot(mean_by_revenue_21.index, mean_by_revenue_21['SBRI_Index_Total'], lw=3, c='orange')
    plt.axhline(lw=2, color='grey', linestyle='--')
    q.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), q.set_xticklabels(list(revenue_mapping)[0:10], fontsize=16, rotation=30)
    q2.set_yticks([0, 50, 100, 150, 200]), q2.set_yticklabels(['0','50', '100', '150', '200'], fontsize=18)
    q2.set_ylabel('Average SBRI Index Scores', fontsize=24)
    q.spines[['top', 'left', 'right', 'bottom']].set_linewidth(3)
    q.xaxis.set_tick_params(width=3)
    q.yaxis.set_tick_params(width=3)
    q2.tick_params(axis='both', which='major', labelsize=24)
    q2.legend(handles=legend_elements, frameon=False, fontsize=20, bbox_to_anchor=(.2, .5, .6, .5))
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), subfolder, 'SBRI_Index_by_Revenue.png'))

    # ## Describe Adapt_main_1_matrix_1
    # valid_adapt1 = composite_variables[composite_variables['Adapt_main_1_matrix_1'].notna()]
    # box_r = pd.to_numeric(valid_adapt1['Adapt_main_1_matrix_1']).to_list()
    # fig, r = plt.subplots(figsize=(6, 8))
    # r.boxplot(box_r, widths=0.4, patch_artist=True, boxprops=dict(lw=3, fc='w'),
    #           whiskerprops=dict(lw=3), medianprops=dict(lw=3, c='g'),
    #           meanprops=dict(markerfacecolor='r', markeredgecolor='r', markersize=10),
    #           capprops=dict(lw=3), showmeans=True, showfliers=True)
    # r.xaxis.set_tick_params(top=False, bottom=False)
    # r.set_xticklabels([])
    # r.set_yticks([1, 2, 3, 4, 5]), r.set_yticklabels(['SD', 'Disagree', 'Neither', 'Agree', 'SA'])
    # figure_customization(r)
    # plt.tight_layout()
    # plt.savefig(os.path.join(os.getcwd(), subfolder, 'Adapt_main_1_matrix_1.png'))
    #
    # ## Compare loan receiver from 2021 and 2022
    # loan_2021 = composite_variables[composite_variables['Q2_Loan Year'] == '2021']['SBRI_Index_Total'].to_list()
    # loan_2022 = composite_variables[composite_variables['Q2_Loan Year'] == '2022']['SBRI_Index_Total'].to_list()
    # stats.mannwhitneyu(loan_2021, loan_2022)
    # fig, s = plt.subplots(figsize=(8, 8))
    # s.boxplot([loan_2021, loan_2022], widths=[len(loan_2021)*0.005, len(loan_2022)*0.005],
    #           patch_artist=True, boxprops=dict(lw=3, fc='w'),
    #           whiskerprops=dict(lw=3), medianprops=dict(lw=3, c='g'),
    #           meanprops=dict(markerfacecolor='r', markeredgecolor='r', markersize=10),
    #           capprops=dict(lw=3), showmeans=True, showfliers=True)
    # s.set_xticks([1, 2]), s.set_xticklabels(['2021 Loan', '2022 Loan'])
    # s.set_ylabel('SBRI Index Scores', fontsize=24)
    # figure_customization(s)
    # plt.tight_layout()
    # plt.savefig(os.path.join(os.getcwd(), subfolder, 'Index_Loan_Year.png'))

    # ## Compare loan receiver from 2021 and 2022
    # grant_2020 = composite_variables[(composite_variables['Q1_Grant Year_1'] == '2020') & (composite_variables['Q1_Grant Year_2'] != '2022')][
    #     'SBRI_Index_Total'].to_list()
    # grant_both = composite_variables[(composite_variables['Q1_Grant Year_1'] == '2020') & (composite_variables['Q1_Grant Year_2'] == '2022')][
    #     'SBRI_Index_Total'].to_list()
    # grant_2022 = composite_variables[(composite_variables['Q1_Grant Year_1'] != '2020') & (composite_variables['Q1_Grant Year_2'] == '2022')][
    #     'SBRI_Index_Total'].to_list()
    # stats.mannwhitneyu(grant_2020, grant_2022)
    # fig, s = plt.subplots(figsize=(8, 8))
    # s.boxplot([grant_2020, grant_2022, grant_both], widths=[len(grant_2020) * 0.002, len(grant_2022) * 0.002, len(grant_both) * 0.002],
    #           patch_artist=True, boxprops=dict(lw=3, fc='w'),
    #           whiskerprops=dict(lw=3), medianprops=dict(lw=3, c='g'),
    #           meanprops=dict(markerfacecolor='r', markeredgecolor='r', markersize=10),
    #           capprops=dict(lw=3), showmeans=True, showfliers=True)
    # s.set_xticks([1, 2, 3]), s.set_xticklabels(['2020 Grant', '2022 Grant', 'Both Years'])
    # s.set_ylabel('SBRI Index Scores', fontsize=24)
    # figure_customization(s)
    # plt.tight_layout()
    # plt.savefig(os.path.join(os.getcwd(), subfolder, 'Index_Grant_Year.png'))

    return
