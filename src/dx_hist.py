import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from util import resolve_path

FILEPATH_DIAGNOSTICS = 'C:/Users/Admin_Calvin/Microwave_Documents/NIH/data/Updated_All_Dx_Data_For_Glioblastoma_Subjects_sent_to_Zhu_1-18-2022.xlsx - All_Diagnosis.csv'

from gensim.models.keyedvectors import KeyedVectors
#BIO_WORD_VECS = KeyedVectors.load_word2vec_format('C:/Users/Admin_Calvin/Microwave_Documents/NIH/biowordvec/bio_embedding_intrinsic', binary=True)


def preprocess_dx_df(filepath: str=FILEPATH_DIAGNOSTICS) -> pd.DataFrame:
    '''Does not exit pipe.'''

    # Define available column types
    column_types = {
        # 'Row No': 'Int64',
        'Subject': 'str',
        'Date_of_death': 'str',  # to be parsed
        'Race': 'string',  # cannot have `np.nan` as a category level
        'Gender': 'string',  # cannot have `np.nan` as a category level
        'Ethnic_group': 'string',  # cannot have `np.nan` as a category level
        'Age at Time of Diagnosis': 'Int64',
        'Date': 'str',  # to be parsed
        'Diagnosis Type': 'string',  # cannot have `np.nan` as a category level
        'ICD if Available': 'string',  # cannot have `np.nan` as a category level
        'Main Diagnosis Text': 'str',
        'Secondary Diagnosis Text': 'str',
        # 'Review Note': 'str'  # all blanks
    }

    # Select desired columns
    selected_columns = list(column_types.keys())

    # Specify file-wide NA values
    na_values = ['', 'NULL', 'NONE', 'Unknown']

    # Import data
    df = pd.read_csv(filepath, usecols=selected_columns,
                     dtype=column_types, na_values=na_values)

    # Parse datetimes
    df['Date_of_death'] = pd.to_datetime(df['Date_of_death'], format='mixed')
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')

    # In `Gender` column, correct misspelled values
    gender_replacements = { 'M': 'Male' }
    df['Gender'] = df['Gender'].replace(gender_replacements)

    # In `Ethnic_group` column, fix misspelled and missing values
    ethnic_replacements = {
        'Not Hispanic or Lati': 'Not Hispanic or Latino',
        'N': None,  # Is 'N' considered a missing value? BTRIS doesn't know, so better be safe
        'Not Reported': None
    }
    df['Ethnic_group'] = df['Ethnic_group'].replace(ethnic_replacements)

    # In the `...Diagnosis Text` columns (given `ICD if Available=='NULL'`), fix missing values
    diagnoses_replacements = {
        'UNKNOWN': None,
        'UNKOWN': None,
        'NOT SPECIFIED': None,
        'DIAGNOSIS': None,  # (w/o ICD given,) this is not specific enough
    }
    df['Main Diagnosis Text'] = df['Main Diagnosis Text'].replace(diagnoses_replacements)
    df['Secondary Diagnosis Text'] = df['Secondary Diagnosis Text'].replace(diagnoses_replacements)

    # Cast nullable categorical columns into 'category' type, the non-nullable ones having already been cast
    nullable_cols = ['Race', 'Gender', 'Ethnic_group',
                    'Diagnosis Type', 'ICD if Available']
    for col in nullable_cols:
        df[col] = df[col].astype('category')

    '''This line is for `verify_consistent_demography()` in `multi_type_processing`.'''
    # df.drop_duplicates(subset='Subject').to_pickle('../intermediates/dx_demo.pkl')

    '''Note: No need to filter rows based on unique `Subject` values (the focus, at least in terms of comorbidities, is on ALL diagnoses).'''

    print(f'Unique Subjects in this D.F.:\t{df["Subject"].unique().shape[0]}')  # TK Define this (1210) to be number of unique patients across all data types

    return df



# Set text sizes for the two plots
PLOT_SMALL_SIZE = 8
PLOT_MEDIUM_SIZE = 10
PLOT_BIGGER_SIZE = 12
plt.rc('font', size=PLOT_SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=PLOT_SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=PLOT_MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=PLOT_SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=PLOT_SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=PLOT_SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=PLOT_BIGGER_SIZE)  # fontsize of the figure title

def visualize_conditions_barplot(df: pd.DataFrame, n_included: int=20,
                            filepath_10: str='../lookups/Section111ValidICD10-Jan2023-DupFixed.xlsx',
                            filepath_9: str='../lookups/Section111ValidICD9-Jan2023.xlsx'):
    '''Simple barplot for frequency of co-morbidities. Exits pipeline because it's not modifying the data.'''

    # Filter out (perhaps not all) rows recording procedures done
    df = df[df['Diagnosis Type'] != 'Procedure']

    # Sort bars by frequency of `ICD if Available` (imperfect)
    conditions = df.groupby(by=['ICD if Available'], dropna=True).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

    # We are looking at CO-morbidities; filter out GARD==2491 (GBM) row
    conditions = conditions[conditions['ICD if Available'] != '2491']

    '''Collectively, the first four patterns match 1778 (out of 1778 total), unique (after `strip()` and pad) codes. TK cite all the links or download all this stuff or whatever'''
    icd_9_pattern = r'^(?:\d|E\d?)\d{2}\.?\d{0,2}$'
    # Link: https://www.cms.gov/Medicare/Coding/ICD10/downloads/032310_ICD10_Slides.pdf
    # Test: https://www.cms.gov/medicare/coordination-benefits-recovery-overview/icd-code-lists
    # Note: Overlaps with `cpt_pattern`; Excludes 'V' (supplementary classification codes)
    icd_10_cm_pattern = r'^[A-TV-Z]\d[A-Z\d](?:\.?[A-Z\d]{0,4})|U0(?:70|71|99)$'
    # Link: https://www.cms.gov/Medicare/Coding/ICD10/downloads/032310_ICD10_Slides.pdf
    # Test: https://www.cms.gov/medicare/coordination-benefits-recovery-overview/icd-code-lists
    # Note: Overlaps with `hcpcs_sin_modifiers_pattern`
    icd_9_proc_pattern = r'^\d{1,2}(?:\.\d{1,2})?$'
    # Test: http://www.icd9data.com/2012/Volume3/default.htm
    icd_10_pcs_pattern = r'^[A-HJ-NP-Z\d]{7}$'  # format-valid but maybe not semantic-valid
    # Link: https://www.cms.gov/files/document/2022-official-icd-10-pcs-coding-guidelines-updated-december-1-2021.pdf
    cpt_pattern = r'\d{4}[FT\d]'
    # Link: https://en.wikipedia.org/wiki/Current_Procedural_Terminology#Types_of_code
    # Note: Overlaps with `icd_9_pattern`
    hcpcs_sin_modifiers_pattern = r'[A-CEGHJ-MP-V]\d{4}'
    # Link: https://www.aapc.com/codes/hcpcs-codes-range/
    # Note: Pattern is not even specific to codes presented in [Link]. Overlaps with `icd_10_cm_pattern`

    # Standardize code-strings by stripping whitespace, and, since some codes have removed prefixed '0's, left-padding with '0'
    conditions['ICD if Available'] = conditions['ICD if Available'].astype(str).apply(lambda code: re.sub(r'^(\d\..*)$', r'0\1', code.strip()))

    # Filter out the rows which do not record diagnoses. (Note that there is some ambiguity between non-diagnoses/procedures vs. diagnoses, if going off of `ICD if Available` alone.)
    conditions = conditions[conditions['ICD if Available'].str.match('|'.join([icd_9_pattern, icd_10_cm_pattern]))]

    # Remove periods ('.') because that's how the codes are, in the lookup tables.
    conditions['ICD if Available'] = conditions['ICD if Available'].str.replace('.', '')

    # Speaking of lookup tables, load them (ICD9, ICD10)
    icd10_df = pd.read_excel(resolve_path(filepath_10), header=0, usecols='A:B', index_col=0, dtype=str)
    icd9_df = pd.read_excel(resolve_path(filepath_9), header=0, usecols='A:B', index_col=0, dtype=str)

    # Concatenate (rbind) the two DataFrames, and...
    lookup_df = pd.concat([icd10_df, icd9_df], axis='index')
    # ...Combine their columns to create a single-column lookup DataFrame
    lookup_df = lookup_df.iloc[:, 0].combine_first(lookup_df.iloc[:, 1])

    def lookup_name(code: str) -> str:
        '''Return the human-readable description of the diagnosis (based on the code). If a corresponding description doesn't exist for the code entered, it's probably a non-diagnosis/procedure code (e.g. a 'V'-prefixed ICD-9 code, mentioned above), and deemed unimportant.'''
        try:
            return lookup_df.loc[code]
        except KeyError:
            return ''

    # Create new column of diagnoses names based on lookups
    conditions['Dx Name based on ICD'] = conditions['ICD if Available'].astype(str).apply(lookup_name)

    # Filter out blank (i.e. not found) diagnoses names
    conditions = conditions[conditions['Dx Name based on ICD'] != '']

    # Keep only the `n_included` most frequent diagnoses to display in the barplot
    conditions = conditions.head(n_included)
    plt.figure()
    # Create the barplot
    ax = sns.barplot(data=conditions, x='counts', y='Dx Name based on ICD')
    plt.title(f'Frequency of top {n_included} Comorbidities')
    plt.xlabel('Frequency')
    plt.ylabel('Comorbidities')

    # Loop through each bar
    for p in ax.patches:
        # Get the width of the bar (which is the count)
        width = p.get_width()
        # Annotate said bar with the count
        ax.text(x=width + 0.1, y=p.get_y() + p.get_height() / 2,
                    s=f'{int(width)}', ha='left', va='center')

    # Automatically adjust the figure size
    plt.tight_layout()
    # Save this first figure as a PNG file
    # plt.show()
    plt.savefig(resolve_path('../plots/dx_hist_condtns.png'))
    # TK idk why the bars become not sorted descending anymore



def deduplicate_gbm_dx(df: pd.DataFrame):
    '''Keep only rows with first GBM diagnoses. Exits pipe because further processing is done elsewhere.'''

    def keep_gbm_dx_rows_via_regex(df: pd.DataFrame) -> pd.DataFrame:
        '''Naive method: Identify, by eye, and mark all the GBM synonyms (synonyms/typos which were found comprehensively (in the xlsx)).'''

        rows_GBM = df.copy()

        '''Tests whether GBM diagnoses exclusion was too restrictive.'''
        # # Read the file content
        # with open(resolve_path('../intermediates/bob.txt'), 'r') as file:
        #     content = file.read()
        # # Split the content using the $ separator and store it in a list
        # list_GBM = [re.escape(s) for s in content.split('$')]
        # # Pattern it using OR logic
        # match_pattern = '|'.join(list_GBM)
        # # Match the pattern in the diagnoses
        # rows_GBM['gbm_dx_marker'] = rows_GBM['Main Diagnosis Text'].str.match(match_pattern, case=False)

        '''TK Link (Grade IV Astrocytoma == GBM): https://stanfordhealthcare.org/content/shc/en/medical-conditions/brain-and-nerves/astrocytoma/about-this-condition/stages.html/'''

        # Matches 2879 (out of 19887) rows and 1025 (out of 1210) unique patients
        # 1025 is to be expected because only 1033 is obtained after removing the generic 'neoplasm's and 'tumor's, without even excluding e.g. neuroblastomas and non-astrocytic gliomas.
        contain_pattern = r'gi?li?o?(?:(?:bo?l?astr?|sarc)oma|matosis cerebri)|gbm|gmb|(?:anaplastic [a-z]*|GRADE IV |malignant )astrocyt?oma|(?:glioma )?(?:high grade|grade 3or 4)(?: glioma)?'
        # TK include further justification (links) e.g for gliosarcoma and gliomatosis cerebri
        # "High grade" (the same as "grade 3or 4") glioma included because (before 2021) GBM is grade 4 glioma (TK cite), and to include as many GBM diagnoses as possible.

        rows_GBM['gbm_dx_marker'] = rows_GBM['Main Diagnosis Text'].str.contains(contain_pattern, case=False, regex=True)

        # GBM == GARD2491 TK include other codes?
        filter_result = rows_GBM[rows_GBM['gbm_dx_marker'] |
                                (rows_GBM['ICD if Available'] == '2491')]

        print(f'Total Rows in this D.F.:\t{len(rows_GBM)}\nRows recording GBM diagnosis:\t{len(filter_result)}\nSubjects with GBM diagnosis:\t{len(filter_result.drop_duplicates(subset="Subject"))}')

        return filter_result

    '''This line is not enough; there may be multiple (differently named) GBM diagnoses per patient'''
    rows_GBM = keep_gbm_dx_rows_via_regex(df)

    # Select only the useful columns
    rows_GBM = rows_GBM[['Subject', 'Date_of_death', 'Date', 'Age at Time of Diagnosis']]

    # For clarity, rename `Date` column
    rows_GBM = rows_GBM.rename(columns={'Date': 'Date_of_Diagnosis'})

    # De-space `Age at Time of Diagnosis` column name (so they won't get affected by `split(' ')` later on)
    rows_GBM = rows_GBM.rename(
        columns={'Age at Time of Diagnosis': 'Age_at_Time_of_Diagnosis'})

    # Sort by date (ascending), so earliest GBM diagnosis date comes first
    rows_GBM = rows_GBM.sort_values(by='Date_of_Diagnosis', ascending=True)

    # Keep first occurrences of duplicated (by `Subject`) rows
    rows_GBM = rows_GBM.drop_duplicates(subset='Subject', keep='first')

    # Set index to `Subject` ID, to concatenate with similar DataFrames later
    rows_GBM = rows_GBM.set_index('Subject', drop=True)

    # Serialize the DataFrame
    rows_GBM.to_pickle(resolve_path('../intermediates/only_GBM_dx_dates.pkl'))

df = preprocess_dx_df()
print(f'Percent of total right-censored:\t{df.drop_duplicates(subset="Subject")["Date_of_death"].isna().sum()/len(df.drop_duplicates(subset="Subject")["Date_of_death"]) * 100:.2f}%')
print(df.info())

#Standard pipeline (preprocess %>% deduplicate)
deduplicate_gbm_dx(preprocess_dx_df())

# Visualization pipline (preprocess %>% visualize)
# visualize_conditions_barplot(preprocess_dx_df())



