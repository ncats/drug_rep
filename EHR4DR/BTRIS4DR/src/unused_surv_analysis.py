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




# Link: https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Cox%20residuals.html#Assessing-Cox-model-fit-using-residuals-(work-in-progress)
def display_summary(data_x: pd.DataFrame,
                    data_y: pd.DataFrame):
    entire = pd.concat([data_y, data_x], axis='columns', join='outer')
    cph = CoxPHFitter(penalizer=0.9)
    # False (sksurv) == Right-censored () == 0 (lifelines)
    cph.fit(entire, duration_col='Survival_in_days',
            event_col='Status', show_progress=True)
    # Display summary
    cph.print_summary()
    cph.plot()


# Link: https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html
def validate_hazard_proportionality(data_x: pd.DataFrame,
                                    data_y: pd.DataFrame):
    '''Computes statistics and generates plots that are used to check the hazard proportionality assumption, printing everything along with some advice (to correct for non-proportionality).'''

    print('\n\nChecking proportionality of hazards requirement...')

    entire = pd.concat([data_y, data_x], axis='columns', join='outer')
    cph = CoxPHFitter(penalizer=0.05, l1_ratio=0.9)
    # False (sksurv) == Right-censored () == 0 (lifelines)
    cph.fit(entire, duration_col='Survival_in_days',
            event_col='Status', show_progress=True)

    # Summary of the data
    cph.print_summary(model="untransformed variables", decimals=3)
    cph.check_assumptions(entire, p_value_threshold=0.05, show_plots=True)

    scaled_schoenfeld_residuals = cph.compute_residuals(
        training_DataFrame=entire, kind='scaled_schoenfeld')
    scaled_schoenfeld_residuals.to_csv(
        resolve_path('../results/sclschresid.csv'))

    print(scaled_schoenfeld_residuals)

    print('Checking proportionality of hazards requirement done.\n\n')


# From: https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Cox%20residuals.html#Deviance-residuals
def examine_outliers(data_x: pd.DataFrame,
                     data_y: pd.DataFrame):

    print('\n\nExamining outliers...')

    entire = pd.concat([data_y, data_x], axis='columns', join='outer')
    cph = CoxPHFitter(penalizer=0.05, l1_ratio=0.9)
    # False (sksurv) == Right-censored () == 0 (lifelines)
    cph.fit(entire, duration_col='Survival_in_days',
            event_col='Status', show_progress=True)

    r = cph.compute_residuals(entire, 'deviance')
    r.head()
    r.plot.scatter(
        x='Survival_in_days', y='deviance', alpha=0.75,
        c=np.where(r['arrest'], '#008fd5', '#fc4f30')
    )

    r = r.join(entire.drop(['week', 'arrest'], axis=1))
    plt.scatter(r['prio'], r['deviance'], color=np.where(
        r['arrest'], '#008fd5', '#fc4f30'))
    r = cph.compute_residuals(entire, 'delta_beta')
    r.head()
    r = r.join(entire[['week', 'arrest']])
    r.head()
    plt.scatter(r['week'], r['prio'], color=np.where(
        r['arrest'], '#008fd5', '#fc4f30'))

    print('Examining outliers done.\n\n')


# From: https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Cox%20residuals.html#Martingale-residuals
def detect_nonlinearity(data_x: pd.DataFrame,
                        data_y: pd.DataFrame):

    print('\n\nDetecting nonlinearity...')

    entire = pd.concat([data_y, data_x], axis='columns', join='outer')
    cph = CoxPHFitter(penalizer=0.05, l1_ratio=0.9)
    # False (sksurv) == Right-censored () == 0 (lifelines)
    cph.fit(entire, duration_col='Survival_in_days',
            event_col='Status', show_progress=True)

    r = cph.compute_residuals(entire, 'martingale')
    r.head()
    r.plot.scatter(
        x='week', y='martingale', alpha=0.75,
        c=np.where(r['arrest'], '#008fd5', '#fc4f30')
    )

    print('Detecting nonlinearity done.\n\n')


# TK how bout we just integrate this into their respective functions (of the three above)
# From: https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Stratification
def fix_violations(data_x: pd.DataFrame,
                   data_y: pd.DataFrame):
    pass


display_summary(*format_4_assumption_tests())
validate_hazard_proportionality(*format_4_assumption_tests())
examine_outliers(*format_4_assumption_tests())
detect_nonlinearity(*format_4_assumption_tests())


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
