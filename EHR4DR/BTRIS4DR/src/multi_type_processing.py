import numpy as np
import pandas as pd

def verify_consistent_demography(filepath_dx: str='../intermediates/dx_demo.pkl',
                                 filepath_vitals: str='../intermediates/vitals_demo.pkl'):
    '''Verify that the (non-null) demographic information between the two data-types Diagnoses and Vitals are consistent. The existence of this function is just for peace of mind, and not indicative of the quality of the data sources from which this function draws.'''

    # Load and deserialize the two named dataframes
    dx = pd.read_pickle(filepath_dx)
    vitals = pd.read_pickle(filepath_vitals)

    # Turn `Subject` into each's index to allow for concatenation
    dx, vitals = dx.set_index('Subject'), vitals.set_index('Subject')

    # Make column-naming consistent
    vitals = vitals.rename(columns={'Ethnicity': 'Ethnic_group'})

    # Differentiate column names so they aren't collapsed in the concatenation
    dx.columns = [col + '_dx' for col in dx.columns]
    vitals.columns = [col + '_vitals' for col in vitals.columns]

    # Concatenate the two dataframes, as promised
    combined = pd.concat([dx, vitals], axis='columns', join='inner')

    # Select columns we wish to cross-reference
    combined = combined[['Race_dx', 'Race_vitals', 'Gender_dx', 'Gender_vitals', 'Ethnic_group_dx', 'Ethnic_group_vitals']]

    # Cast to `str` type to be able to compare cell-by-cell
    combined = combined.astype(str)

    # Weird that i have to do this.
    combined = combined.replace({'nan': 'NULL'})

    # Because `NaN != NaN`. Comparing strings is easier than NA values
    combined = combined.fillna('NULL')

    # Perform verification (for each demographic category) and save results
    with open('../results/inconsistent_demography.txt', 'a') as file:
        for characteristic in ['Race', 'Gender', 'Ethnic_group']:
            line = '-' * 20 + f'Inconsistent {characteristic} info between Vitals and Diagnoses data-types' + '-' * 20 + '\n'
            file.write(line)
            print(line)

            selected = combined[[f'{characteristic}_dx', f'{characteristic}_vitals']]
            combined[f'Different {characteristic}?'] = selected[f'{characteristic}_dx'] != selected[f'{characteristic}_vitals']
            inconsistent = selected[combined[f"Different {characteristic}?"]]
            line = f'Rows where {characteristic} is different for the two columns:\n{"Empty DataFrame" if inconsistent.empty else inconsistent}\n'
            file.write(line)
            print(line)

            n_different = combined[f'Different {characteristic}?'].sum()
            if n_different != 0:
                line = f'There are a total of {n_different} Subjects who are discrepant in terms of recorded {characteristic} across the two spreadsheets/data-types.'
                file.write(line)
                print(line)
            else:
                line = f'In terms of recorded {characteristic}, all Subjects across the two spreadsheets/data-types have consistent information.'
                file.write(line)
                print(line)

            file.write('\n\n')
