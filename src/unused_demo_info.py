import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
'''Perform a chi-squared test for independence (compared to general population) for each combination of `['Gender', 'Race', 'Ethnicity']` with 'Presence recorded within data (viz. 'Updated_All_Vital_Sign_Data...')'. Exits pipeline because it's not modifying the data.'''
'''Purpose: To get an initial understanding of association of GBM with demographic characteristics within the data.'''
'''! Limitations:'''
'''    a. These results are not adjusted for confounding factors, potential or actual (most importantly time range: GBM data is cumulative across time, while gen. pop. data is snapshot);'''
'''    b. The GBM data source may not be representative of all patients diagnosed with GBM in the general population (GBM data only captures NIH patients);'''
'''    c. Controlled/multivariate analyses are required for more definitive conclusions;'''
'''    d. Since i couldn't find NIH (as opposed to US) general pop. data, definitions of 'Race', 'Ethnicity' could differ between the GBM and general pop. datasets.'''
'''0. Variables: ['Gender', 'Race', 'Ethnicity']; Presence recorded within data'''
'''1. Hypotheses:'''
'''  a. Null hypotheses: There is no association of one of ['Gender', 'Race', 'Ethnicity'] between the population present in GBM data versus the population not present in GBM data.'''
'''  b. Alternative hypotheses: There is such an association.'''
'''2. Assumptions:'''
'''  1. ! Random sampling: The `genpop` frequencies (renamed such below) are from the 2020 US Census and is thus a random sample.  However, the `GBM` frequencies (from our data) are (from what i've been told) pulled from NIH patient records (i.e. records of patients who have gone to NIH), which means it's only a random sample (census) from the population of people who have gone to the NIH as patients, no from the general US population.  So, with respect to the `genpop` frequencies, the `GBM` frequencies are biased (and vice versa).'''
'''  2. Independence of observations: The records were pulled indiscriminately (w.r.t. ['Gender', 'Race', 'Ethnicity']).'''
'''  3. Proportions normally distributed: Being recorded in the data or not is categorical; being male or female is categorical.  There is only mutual exclusivity between the levels in each of the variables.'''
'''  4. Sample size: With at least tens of thousands of data points, the sample size seems to be large enough (>30).'''
'''  5. Expected counts not less than 5: [Will check later]'''
# 3. Contingency tables:

# The three columns for which chi-square tests will be run
test_columns = ['Gender', 'Race', 'Ethnicity']

# Gender
# Gen. pop. data link: https://www.census.gov/data/tables/2020/demo/age-and-sex/2020-age-sex-composition.html
# Gen. pop. data name: Age and Sex Composition in the United States: 2020, Table 1. Population by Age and Sex: 2020 [<1.0 MB]
genpop_gender_dict = {'Male': 159461, 'Female': 165807}

# Race
# Gen. pop. data link: https://www.census.gov/library/visualizations/interactive/race-and-ethnicity-in-the-united-state-2010-and-2020-census.html
# Gen. pop. data name: Race and Ethnicity in the United States: 2010 Census and 2020 Census
# Note that GBM data spans from around 1990 to 2022, and in 2000 the Census began including the option to select more than one race categories, meaning 2000 census race data is not directly comparable to past censal race data, and thus not past self-identified data on race in the GBM dataset. (See: https://web.archive.org/web/20090831085310/http://quickfacts.census.gov/qfd/meta/long_68178.htm)
genpop_race_dict = {'White': 204277273,  # White alone
                    'Black/African Amer': 41104200,  # Black or African American alone
                    'Am Indian/Alaska Nat': 3727135,  # American Indian and Alaska Native alone
                    'Asian': 19886049,  # Asian alone
                    'Hawaiian/Pac. Island': 689966,  # Native Hawaiian and Other Pecific Islander alone
                    np.nan: 27915715,  # Some Other Race alone
                    'Multiple Race': 33848943}  # Two or More Races

# Ethnicity
# Gen. pop. data link: https://www.census.gov/library/visualizations/interactive/race-and-ethnicity-in-the-united-state-2010-and-2020-census.html
# Gen. pop. data name: Race and Ethnicity in the United States: 2010 Census and 2020 Census
genpop_ethnic_dict = {'Hispanic or Latino': 62080044,  # same as Census
                    'Not Hispanic or Latino': 269369237,  # same as Census
                    np.nan: 0}  # not on Census

genpop_dicts = [genpop_gender_dict, genpop_race_dict, genpop_ethnic_dict]

for col, genpop_dict in zip(test_columns, genpop_dicts):
    # Gender, in GBM data, doesn't have NA values
    is_col_gender = col == 'Gender'

    # Get one 'column' of the contingency table representing the level 'not present in data'
    genpop_df = pd.DataFrame.from_dict(genpop_dict, orient='index',
                                    columns=['genpop_counts'])
    genpop_df = genpop_df.reset_index(names=col)

    # Get the other column of the contingency table representing the level 'present in data'
    GBM_df = (df.groupby(col, dropna=is_col_gender).size()
            .reset_index(name='GBM_counts'))

    if not is_col_gender:
        # Making sure the `{col}` column in both dataframes can hold NA values
        genpop_df[col] = genpop_df[col].astype(object)
        GBM_df[col] = GBM_df[col].astype(object)

    # Put the two columns side-by-side, merging on their `{col}` columns, and including the union of the keys in `{col}`
    merged_df = pd.merge(genpop_df, GBM_df, how='outer', on=col)

    # Extract the numeric values (as array) from the dataframe
    merged_array = merged_df[['genpop_counts', 'GBM_counts']].values

    # Perform chi-squared test
    chi2, p, dof, expected = chi2_contingency(merged_array)

    # Display and write-to-file the test results
    with open('demo_chisquared_results.txt', 'a') as file:
        # Go to beginning of file
        file.seek(0,0)
        # Truncate previous results
        file.truncate()

        line = str('-' * 30 + f'Chi-square test results for {col}' + '-' * 30 + '\n')
        file.write(line)
        print(line)

        line = f'Observed frequencies table:\n{merged_df.to_string()}\n'
        file.write(line)
        print(line)

        line = f'Chi-squared statistic:\t{chi2}\n'
        file.write(line)
        print(line)

        line = f'Degrees of Freedom:\t{dof}\n'
        file.write(line)
        print(line)

        line = f'p-value:\t{p:.3e}\n'
        file.write(line)
        print(line)

        line = f'Expected frequencies table:\n{np.array2string(expected.astype(int))}\n'
        file.write(line)
        print(line)

        # Interpret p-value, with alpha = 0.05
        alpha = 0.05
        is_significant = p < alpha

        # Check Assumption 5:
        is_assumption_5_met = np.all(expected > 5)
        line = f'Are all expected frequencies greater than 5?\t{is_assumption_5_met}\n'
        file.write(line)
        print(line)

        if is_assumption_5_met:
            caveat = ['', '']
        else:
            caveat[0] = 'Due to one expected count being less than 5, bear in mind that the null hypothesis could have been '
            if is_significant:
                caveat[1] = 'falsely rejected in this case.'
            else:
                caveat[1] = 'incorrect in this case, and should have been rejected.'

        # Determine which categorical we are talking about
        if col == 'Gender':
            col_predicate = 'being male or female'
        elif col == 'Race':
            col_predicate = 'being of a certain race'
        elif col == 'Ethnicity':
            col_predicate = 'being Hispanic/Latino or not'

        if is_significant:
            line = f'There is a significant association between being recorded as having GBM and {col_predicate}.\n{"".join(caveat)}'
            file.write(line)
            print(line)
        else:
            line = f'The null hypothesis (that there is no association between being recorded as having GBM and {col_predicate}) fails to be rejected.\n{"".join(caveat)}'
            file.write(line)
            print(line)