import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from util import resolve_path

FILEPATH_LABS='C:/Users/Admin_Calvin/Sync/Updated_All_Lab_Data_sent_to_Zhu_1-18-2023.csv'

def preprocess_labs_df(filepath: str=FILEPATH_LABS) -> pd.DataFrame:
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
        'Result_Value_Text': 'str',
        # 'Result_Value_Numeric': 'float64',
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



def prepare_indep_lab_df(df: pd.DataFrame):
    '''Exits pipe because further processing is done elsewhere.'''

    # TK do we have to take into account whether Collected_Datetime is before Date_of_diagnosis??

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

