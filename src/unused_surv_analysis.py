# TK implement this actually
def impute_missing_demo_data(df: pd.DataFrame) -> pd.DataFrame:
    '''TK Does not exit pipe. Imputation can be improved by having multiple imputated variants as inputs to the model.'''
    X = df.values
    cols_demo = ['Gender', 'Race', 'Ethnicity']
    col_indices_demo = [list(df.columns).index(col) for col in cols_demo]
    imputer = MissForest()
    imputed_array = imputer.fit_transform(X, cat_vars=col_indices_demo)
    imputer_df = pd.DataFrame(imputed_array, columns=df.columns, index=df.index)
    imputer_df.to_pickle('../intermediates/explanatory_demo_imputed.pkl')
    return imputer_df

data_xt = impute_missing_demo_data(data_xt)







# TK how bout we just integrate this into their respective functions (of the three above)
# From: https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Stratification
def fix_violations(data_x: pd.DataFrame,
                   data_y: pd.DataFrame):
    pass





# TK temporary measure
def mask_filtering_out_small_drugs():
    df = pd.read_csv(resolve_path('../onehot_encoded_drugs.csv'))
    with_100_or_more = pd.DataFrame(df.agg('sum'), columns=['patient_count']).drop(
        index=['Unnamed: 3', 'Unnamed: 0'])
    with_100_or_more = with_100_or_more[with_100_or_more['patient_count'] >= 100]
    return with_100_or_more.index

with open(resolve_path('../intermediates/prelim_candidates.json'), 'w') as file:
    json.dump(pd.Series(coefs[coefs[alpha_min] <= 0][alpha_min].sort_values(ascending=True).index).apply(
        lambda name: query_getRxTermDisplayName(name)).tolist(), file)


import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
# Link: https://github.com/epsilon-machine/missingpy/issues/38#issuecomment-1614864022
from missingpy import MissForest
