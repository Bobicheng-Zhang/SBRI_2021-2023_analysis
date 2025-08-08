# SBRI 2023 Public README

## Preliminary Data Cleaning
1. First, all personal information, including IP address, longitude, latitude, email address, and any other information that could be used to identify the respondent was removed from the raw data.
2. Second, demographic information, such as birth sex, self-identified gender, race/ethnicity, region/location, industry/sector was compiled, truncated on a per-respondent basis and counted.
   1. For industry/sector, those that selected only ‘Other Services’ remained in this category while those that selected this option in addition to a pre-defined option were assigned to a ‘Multiple’ category.
3. Then, reported revenue (in the form of $) and profit difference (in the form of % increase or decrease since the previous year) categories were coded numerically from 1 to 12, with 1 corresponding to the lowest revenue/profit change category and 12 the highest.
4. All questions with Likert-style options were coded numerically from 1 to 5, with 1 corresponding to ‘Strongly Disagree’ and 5 ‘Strongly Agree’ for most, and the reverse mapping for the rest.
   1. Questions in 2021 that were reverse-coded: 'Bhea_Orgres_1_matrix_2'**,** 'Bhea_Orgres_1_matrix_3'**,** 'Bhea_Opres_2_matrix_3', 'Bhea_Opres_5_matrix_2'**,** 'Mind_belon_4_matrix_4'**,** 'Mind_Creat_1_matrix_3'
   2. Questions in 2022 that were reversed-coded: 'Bhea_Orgres_1_matrix_4'**,** 'Bhea_Orgres_1_matrix_5'**,** 'Bhea_Opres_2_matrix_3'**,** 'Bhea_Opres_5_matrix_2'**,** 'Mind_belon_4_matrix_4'**,** 'Mind_Creat_1_matrix_3'
5. **Respondents which missed more than 5 Likert-style responses were included in the calculation of demographic information but were excluded from further analysis.**

## Exploratory FA
1. After removing respondents from both years which missed more than 5 Likert-style responses, we start by assessing whether the SBRI datasets are suitable for factor analysis.
2. For data from both years, we temporarily removed all missing responses and performed Barttlet’s Test of Sphericity to assess correlations between current variables.
   1. ~[The Barttlet’s Test of Sphericity](https://www.statisticshowto.com/bartletts-test/)~ has been shown to be sensitive to chance-level results; thus performing this test prior to factor extraction can help inform us of whether FA is viable here (Tobias & Carlson, 1969).
   2. In such a test, the null hypothesis states that all existing variables in a data set are orthogonal, or uncorrelated, in which case any dimension reduction technique would be unnecessary. On the other hand, if correlations are detected, we can use them to compress the data by reducing the number of variables.
   3. **We found** ***p*** **< .001 for both years suggesting correlations between variables, thereby rejecting the null hypothesis.**
3. Next, we performed the Kaiser-Meyer-Olkin (KMO) Test to measure the sampling adequacy of each variable in our data, which provides another indicator for viability of FA (Dziuban & Shirkey, 1974).
   1. **We found a KMO measure of .84 for our 2021 dataset and .81 for 2022 dataset; this suggests that they are both “meritorious” (Kaiser, 1974) and therefore suitable for FA.**
4. Then, we performed exploratory factor analysis with Varimax rotation and principle component extraction method.
   1. After running parallel analysis (Horn, 1965) for both 2021 and 2022 datasets, factor 9 and factor 10 are tentatively removed as their supra-threshold eigenvalues are likely due to chance.
5. For each factor, absolute loadings below the 90th percentile are removed; and for each question that has retained loading on more than one factors, only the loading with the highest absolute value is kept.
6. This approach has actually helped to find badly-phrased questions because they would be the ones left with no loadings or have their highest loading removed.

[Horn_PA_21.pdf](README/Horn_PA_21.pdf)<!-- {"embed":"true", "preview":"true"} -->
[Horn_PA_22.pdf](README/Horn_PA_22.pdf)<!-- {"embed":"true", "preview":"true"} -->
