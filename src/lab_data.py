import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from util import resolve_path, impute_multinominal, CSV_DIR

FILENAME_LABS='Updated_All_Lab_Data_sent_to_Zhu_1-18-2023.xlsx - All_Lab_Data_Final.csv'

def preprocess_labs_df(filepath: str=CSV_DIR+FILENAME_LABS) -> pd.DataFrame:
    '''Does not exit pipe.'''

    # Define the desired column types, inferring levels for categoricals
    column_types = {
        # 'Row No': 'Int64',
        # 'Data Type': 'str',
        'Subject': 'str',
        'Order_Name': 'category',
        # 'Status': 'category',
        'Collected_Datetime': 'str',  # to be parsed
        # 'btris_cluster_id': 'category',
        'btris_cluster_label': 'category',
        # 'Result_Name': 'str',
        # 'Result_Value_Text': 'str',
        'Result_Value_Numeric': 'float64',
        'Result_Value_Name': 'str',
        'Result_Note': 'str',
        'Unit_of_Measure': 'str',
        'Range': 'str',
        # 'Order_Category': 'category',
        # 'Priority': 'category',
        # 'Lab Code': 'category',
        # 'Pt Type': 'category',
        # 'Reported_Date_Time': 'str'
    }

    # Select desired columns
    selected_cols = list(column_types.keys())

    # Specify file-wide NA values
    na_values = ['', 'NULL']

    # Import data
    df = pd.read_csv(filepath, usecols=selected_cols, dtype=column_types, na_values=na_values)

    # Parse datetimes
    date_format = '%m/%d/%y %H:%M'
    df['Collected_Datetime'] = pd.to_datetime(df['Collected_Datetime'], format=date_format)
    # df['Reported_Date_Time'] = pd.to_datetime(df['Reported_Date_Time'], format=date_format)

    return df



def standardize_units(preprocessed_df: pd.DataFrame) -> pd.DataFrame:
	'''For each different `Unit_of_Measure`, manually write out its full name (not necessarily its official name, but the names are consistent among each other) BEFOREHAND.  Misspellings/alternate spellings of the same unit (e.g. "mcL" and "uL") unified at this stage.  Then standardize the units, applying conversions as needed.'''
	
	# Load the manual unit standardizations
	full_names = pd.read_csv(resolve_path('../intermediates/units_of_measure.csv'), header=0, usecols=['Original_Name', 'Full_Name'], dtype='str')
	
	# Lowercase and strip whitespace
	full_names['Full_Name'] = full_names['Full_Name'].str.strip().lower()

	# Deduplicate the manually standardized units
	dedup_names = full_names['Full_Name'].drop_duplicates().sort_values(by=['Full_Name'])
	
	# Separate the numerator and denominator (if not a fractional unit, populate only the numerator)
	dedup_names['num'], dedup_names['den'] = dedup_names['Full_Name'].str.split('per',1)  # TK this may not work
	cols = ['num', 'den']
	
	'''Calculate the (total, in case of rational units) power of 10 of the unit (e.g. the power of 10 of "mL" is -3)'''
	def get_exponent(str: val) -> int:
	val = str(val)
	# a recognized SI or power-of-10 prefix
	prefix_pattern = re.compile(r'\b(kilo|thousands|millions|deci|100 milli|centi|milli|micro|nano|pico|femto)')
	match = prefix_pattern.search(val)
	if match:
		prefix = match.group(1)
		if 'deci' == prefix or '100 milli' == prefix:
			return -1
		elif 'centi' == prefix:
			return -2
		elif 'milli' == prefix:
			return -3
		elif 'micro' == prefix:
			return -6
		elif 'nano' == prefix:
			return -9
		elif 'pico' == prefix:
			return -12
		elif 'femto' == prefix:
			return -15
		elif 'cent' == prefix:
			return 2
		elif 'kilo' == prefix or 'thousands' == prefix:
			return 3
		elif 'millions' == prefix:
			return 6
		else:  # Just in case
			return 0  
	else:
		return 0  # 10^0 = 1
	# Create new columns based on the above function
	for col in cols:
		dedup_names[col+'_exp'] = dedup_names[col].apply(get_exponent)
		
	'''Find the base unit (e.g. "millimoles" becomes "mole").'''
	def deprefix_and_depluralize(str: s) -> str:
		# List of SI prefixes and large units to remove (order of removal matters)
		large_units = ['thousands', 'millions']
		si_prefixes = [
			'yotta', 'zetta', 'exa', 'peta', 'tera', 'giga', 'mega', 'kilo',
			'hecto', 'deca', 'deci', 'centi', 'milli', 'micro', 'nano', 'pico',
			'femto', 'atto', 'zepto', 'yocto'
		]
		# Step 1: Remove large units, case-insensitive
		large_unit_pattern = re.compile(r'\b(' + '|'.join(large_units) + r')')
		s = large_unit_pattern.sub('', s)
		# Step 2: Remove SI prefixes, case-insensitive
		si_prefix_pattern = re.compile(r'\b(' + '|'.join(si_prefixes) + r')')
		s = si_prefix_pattern.sub('', s)
		# Step 3: Depluralize while excluding certain words, case-insensitive
		excluded_words = ['celsius', 'copies']
		depluralize_pattern = re.compile(r'\b(?!' + '|'.join(excluded_words) + r')(\w+)(s)')
		s = depluralize_pattern.sub(r'\1', s)
		return s.strip()
	# Create new columns based on the above function
	for col in cols:
		dedup_names[col+'_base'] = dedup_names[col].apply(deprefix_and_depluralize)
		
	# Calculate total power of 10 of unit
	dedup_names['tot_exp'] = dedup_names['num_exp'] - dedup_names['den_exp']
		
	'''Out of all the units with the same underlying base unit (e.g. "milligrams per liter" and "milligrams per 100 milliliters" both have the same underlying base unit of "grams per liter"), get the one with the largest positive power of 10'''
	'''Shift the others' powers by said power (if any conversions are to be done, it's best to multiply, as opposed to divide, by a positive integer, to avoid floating point imprecision), obtaining the multiplier for each smaller unit.'''
	dedup_names.groupby(['num_base', 'den_base']).idxmax() # TK fix
	
	
	'''(Or, if unit is not a power of 10 (e.g. "milligrams per 24 hours"), obtain the multiplier directly (e.g. for "millimeters per 24 hours" and "millimeters per hour", the multipliers will be 24 and 1, respectively).)''' # TK fix
	per_day_rows = dedup_names[dedup_names['Full_Name'].str.contains('per 24 hours')].view()
	per_day_rows['multiplier'] = 24
	# TK deal with minutes as well
	
	
	
def fill_null_numerics_with_text(converted_df: pd.DataFrame) -> pd.DataFrame:
	'''There are generally more non-null `Result_Value_Text`s than `Result_Value_Numeric`, so use them (as categorical variables) in case of missing numeric data.'''
	

exclude_alternate_units(preprocess_labs_df())


def prepare_indep_lab_df(df: pd.DataFrame):
    '''Exits pipe because further processing is done elsewhere.'''

    # TK do we have to take into account whether Collected_Datetime is before Date_of_diagnosis??
    #^try 1 week before/after

    # Select desired columns
    df = df[['Subject', 'Collected_Datetime', 'btris_cluster_label', 'Result_Value_Text', 'Result_Value_Name']]

    # Fill in (at least some of) `Result_Value_Text`'s missing values with `Result_Value_Name`
    df['Result_Value_Text'] = df['Result_Value_Text'].fillna(df['Result_Value_Name'])

    # Pivot so that there's one row per `Subject`, with `Result_Value_Text` filling down each row (under their corresponding `btris_cluster_label`). The `Collected_Datetime` gets stored in a separate "level" (TK) at the (otherwise) same location as its corresponding `Result_Value_Text` measurement.
    pivoted = df.pivot(index='Subject', columns='btris_cluster_label', values=['Result_Value_Text', 'Collected_Datetime'])

    # TK what should we do with missing values? should we fill them in with random values from `Range`?

    # TK what about the textual but sort of ordinal (or even non-ordinal) data? how do we encode them ordinally?

    # Serialize the DataFrame
    pivoted.to_pickle(resolve_path('../intermediates/explanatory_labs.pkl'))
    
    
    
# Standard pipeline (preprocess %>% exclude_alternate %>% impute) (YES imputation)
# Multiple (5 times) imputation
print('Performing multiple imputation...')
df = preprocess_labs_df()
for i in range(5):
    impute_multinominal(df[['Subject', 'Gender', 'Race', 'Ethnicity']], str(i), 'explanatory_demo_imputed')
    print(f'Imputation {str(i)} done.')
print('Performing multiple imputation done.')

