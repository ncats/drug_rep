import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from util import resolve_path

from lifelines import CoxPHFitter

from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV, KFold
import warnings
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import set_config

'''Citation: Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.'''

set_config(display='text')  # displays text representation of estimators
# TK (rewrite) to incorporate one-hot encoding
def format_4_assumption_tests(filepath_dx: str='../intermediates/only_GBM_dx_dates.pkl',
                              filepath_meds: str='../intermediates/explanatory_drugs.pkl',
                              filepath_demo: str='../intermediates/explanatory_demo.pkl'):
    '''Per `lifelines` requirements, condense?TK and format explanatory and response variable inputs for assumption testing.'''

    print('Formatting data in preparation for assumption diagnostics...')

    # Load and deserialize diagnoses DataFrame
    rows_GBM = pd.read_pickle(resolve_path(filepath_dx))

    # Load and deserialize drugs DataFrame
    med_hist = pd.read_pickle(resolve_path(filepath_meds))

    # Load and deserialize demographics DataFrame
    demo_info = pd.read_pickle(resolve_path(filepath_demo))

    '''Method 1: Simply removing missing demographic information'''
    print(f'Number of rows in demo:\t{len(demo_info)}')
    demo_info = demo_info.dropna()
    print(f'Number of rows in demo after dropping nulls:\t{len(demo_info)}')
    # TK fill in lab_data stuff

    # Set-intersect indices to get common patients between drugs and diagnostic datasets
    common_med_dx_index = med_hist.index.intersection(rows_GBM.index)

    # Set-intersect indices again to get common patients between all three of drugs, diagnostic, and demographic dataseta
    common_index = common_med_dx_index.intersection(demo_info.index)
    print(f'Potential Subjects:\t{len(common_med_dx_index)}\t\tActual Subjects:\t{len(common_index)}')

    # Filter out non-intersected rows
    rows_GBM = rows_GBM.loc[common_index]
    med_hist = med_hist.loc[common_index]
    demo_info = demo_info.loc[common_index]

    # One-hot encode all the categorical variables
    demo_info_t = OneHotEncoder().fit_transform(X=demo_info)

    # Concatenate demographic and drug DataFrames into explanatory DataFrame
    data_x = pd.concat(
        [med_hist, demo_info_t, rows_GBM['Age_at_Time_of_Diagnosis']], axis='columns', join='outer')

    # Exclude features with low variance (according to `lifelines` fitting warnings)
    data_x = data_x.drop(columns=['Order_Name=2599', 'Order_Name=317398'])
    # RxCUI 2599 = clonidine transdermal patch
    # RxCUI 317398 = lamotrigine 150 mg tablet

    # Exclude medicational features without therapeutic effects
    data_x = data_x.drop(columns=['Order_Name=dry mouth treatment',
                                  'Order_Name=optichamber spacer device',
                                  'Order_Name=saline nasal rinse kit',
                                  'Order_Name=saliva substitute solution',
                                  'Order_Name=skin test read order'])

    # The patients with no (i.e. missing) death-date by end-data-collection date are right-censored (`Status == False`)
    rows_GBM['Status'] = pd.isna(rows_GBM['Date_of_death'])
    n_censored, n_unique = rows_GBM['Status'].sum(), rows_GBM['Status'].shape[0]
    with open(resolve_path('../results/n_right_censored.txt'), 'w') as file:
        line = f'Right-censored: {n_censored}\tTotal: {n_unique}\tPercent right-censored: {n_censored / n_unique * 100:.2f}%'
        print(line)
        file.write(line)
    '''Link: "Five-year relative survival was lowest for glioblastoma (6.8%)" (PMID: 31675094)'''
    END_DX_DATE = pd.to_datetime('2022/01/13')  # Confirmed by BTRIS
    # Fill NAs in `Date_of_death` column with the end-data-collection date to get a survival duration anyways
    rows_GBM['Date_of_death'] = rows_GBM['Date_of_death'].fillna(END_DX_DATE)
    # Calculate duration between (1st) GBM diagnosis date and death/end-data-collection date
    rows_GBM['Survival_in_days'] = (rows_GBM['Date_of_death'] - \
        rows_GBM['Date_of_Diagnosis']).dt.days

    print('Formatting data in preparation for assumption diagnostics done.')

    print(data_x.head(7))
    print(data_x.info())

    return [data_x, rows_GBM[['Status', 'Survival_in_days']]]


# Link: https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Cox%20residuals.html#Assessing-Cox-model-fit-using-residuals-(work-in-progress)
def display_summary(data_x: pd.DataFrame,
                    data_y: pd.DataFrame,
                    alpha: float):

    entire = pd.concat([data_y, data_x], axis='columns', join='outer')
    entire = entire.astype(int)
    cph = CoxPHFitter(penalizer=alpha, l1_ratio=0.9)
    # Fit model
    print('Fitting model...')
    cph.fit(entire, duration_col='Survival_in_days')
    print('Fitting model done.')
    # Display summary
    print('Displaying summary...')
    # TK shows an extra table (below the one shown in the example) with ['cmp to', 'z', 'p', '-log2(p)']
    cph.print_summary()
    # Serialize summary
    cph.summary.to_pickle(resolve_path('../results/summary.pkl'))
    print('Displaying summary done.')
    print('Plotting coefficients...')
    cph.plot()
    plt.show()
    plt.savefig(resolve_path('../plots/coefficients.png'))
    print('Plotting coefficients done.')

    return cph


# Link: https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html
def validate_hazard_proportionality(data_x: pd.DataFrame,
                                    data_y: pd.DataFrame,
                                    fitted_cph):
    '''Computes statistics and generates plots that are used to check the hazard proportionality assumption, printing everything along with some advice (to correct for non-proportionality).'''

    print('\n\nChecking proportionality of hazards requirement...')

    entire = pd.concat([data_y, data_x], axis='columns', join='outer')

    # Summary of the data
    fitted_cph.print_summary(model="untransformed variables", decimals=3)
    fitted_cph.check_assumptions(entire, p_value_threshold=0.05, show_plots=True)

    # Compute residuals
    scaled_schoenfeld_residuals = fitted_cph.compute_residuals(
        training_DataFrame=entire, kind='scaled_schoenfeld')
    scaled_schoenfeld_residuals.to_csv(
        resolve_path('../results/sclschresid.csv'))

    print(scaled_schoenfeld_residuals)

    print('Checking proportionality of hazards requirement done.\n\n')


# From: https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Cox%20residuals.html#Deviance-residuals
def examine_outliers(data_x: pd.DataFrame,
                     data_y: pd.DataFrame,
                     fitted_cph):

    print('\n\nExamining outliers...')

    entire = pd.concat([data_y, data_x], axis='columns', join='outer')

    r = fitted_cph.compute_residuals(entire, 'deviance')
    r.head()
    r.plot.scatter(
        x='Survival_in_days', y='deviance', alpha=0.75,
        c=np.where(r['arrest'], '#008fd5', '#fc4f30')
    )

    r = r.join(entire.drop(['Survival_in_days', 'arrest'], axis='columns'))
    plt.scatter(r['prio'], r['deviance'], color=np.where(
        r['arrest'], '#008fd5', '#fc4f30'))
    r = fitted_cph.compute_residuals(entire, 'delta_beta')
    r.head()
    r = r.join(entire[['Survival_in_days', 'arrest']])
    r.head()
    plt.scatter(r['Survival_in_days'], r['prio'], color=np.where(
        r['arrest'], '#008fd5', '#fc4f30'))
    plt.show()

    print('Examining outliers done.\n\n')


# From: https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Cox%20residuals.html#Martingale-residuals
def detect_nonlinearity(data_x: pd.DataFrame,
                        data_y: pd.DataFrame,
                        fitted_cph):

    print('\n\nDetecting nonlinearity...')

    entire = pd.concat([data_y, data_x], axis='columns', join='outer')

    r = fitted_cph.compute_residuals(entire, 'martingale')
    r.head()
    r.plot.scatter(
        x='Survival_in_days', y='martingale', alpha=0.75,
        c=np.where(r['arrest'], '#008fd5', '#fc4f30')
    )

    print('Detecting nonlinearity done.\n\n')


# ==============================================================================


def format_4_surv_analysis(data_x: pd.DataFrame, rows_GBM: pd.DataFrame):
    '''Per `sksurv` requirements, convert the dependent variables into a structured array with [`Status`, `Survival_in_days`] as fields. TK i converted this too early (TK did i?) (assumption diagnostics work better with un-converted DataFrames than structured arrays)'''
    dtypes = np.dtype([('Status', '?'), ('Survival_in_days', '<f8')])
    data_y = np.array([tuple(value) for value in rows_GBM.values],
                      dtype=dtypes)
    # data_xt = data_xt.astype(float)

    return [data_x, data_y]


def recover_drug_name_else_return(maybe_rxcui: str) -> str:
    with open(resolve_path('../results/after_&_before_normalization.json'), 'r') as file:
        try:
            maybe_drug_name = json.load(file)[maybe_rxcui.replace('Order_Name=', '')].replace('$$$', '').capitalize()
            return maybe_drug_name
        except KeyError:
            return maybe_rxcui


def save_coefficients_vs_alpha_plot(data_x: pd.DataFrame,
                                    data_y: pd.DataFrame,
                                    # filepath_y: str = '../data_y.pkl',
                                    # filepath_x: str='../data_x.pkl',
                                    n_highlight: int=7):
    '''Perform preliminary survival analysis (i.e. to get an idea of the most important features). Exits pipeline because it's not modifying the data nor generating any results. From: https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html#Penalized-Cox-Models'''

    # Load and deserialize data
    # data_x, data_y = pd.read_pickle(resolve_path(filepath_x)), pd.read_pickle(resolve_path(filepath_y))

    '''Weigh l1 (i.e. LASSO) penalty at 90%. Set alpha to be decreased until 1% of its original value (where feature coefficients are all zero).'''
    print('Fitting (preliminary) to Cox PH model...')
    cox_elastic_net = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01)
    cox_elastic_net.fit(data_x, data_y)

    coefficients_elastic_net = pd.DataFrame(
        cox_elastic_net.coef_, index=data_x.columns,
        columns=np.round(cox_elastic_net.alphas_, 5)
    )
    print('Fitting (preliminary) to Cox PH model done.')

    # Elastic-Net-penalized Cox Proportional Hazard Model
    '''Use this because we have many features, and we want to select the most influential'''
    def plot_coefficients(coefs, n_highlight):
        _, ax = plt.subplots(figsize=(9, 5))
        alphas = coefs.columns
        for row in coefs.itertuples():
            ax.semilogx(alphas, row[1:], '.-', label=row.Index)

        # Labels will be at left-end of graph
        alpha_min = alphas.min()
        # Get the top (wrt abs value) coefficients
        top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)

        # Labeling the graphs of the top coefficients
        for feature_name in top_coefs.index:
            coef = coefs.loc[feature_name, alpha_min]
            # Check whether the order had been marked
            marked_4_interest = feature_name.startswith('$$$')
            line_label = recover_drug_name_else_return(feature_name).replace(
                '$$$', '').split(' ')[0]
            plt.text(alpha_min, coef, ('$$$ ' if marked_4_interest else '') + (line_label if line_label != '' else feature_name) + '        ',
                horizontalalignment='right', verticalalignment='center')

        ax.set_title('Feature coefficients over a range of possible penalty strengths')
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.grid(True)
        ax.set_title('Coefficients of features for various values of penalty strength')
        ax.set_xlabel('penalty strength $\\alpha$')
        ax.set_ylabel('Coefficient')
        plt.savefig(resolve_path('../plots/coefs_over_alphas.png'))
        plt.show()

    plot_coefficients(coefficients_elastic_net, n_highlight)


# ==============================================================================


# ALPHA_MIN_RATIO = 0.23  # Lowest successful percentage of alpha (2 s.f.)
ALPHA_MIN_RATIO = 0.01  # Lowest successful percentage of alpha (2 s.f.)
def choose_penalty_strength(data_x: pd.DataFrame,
                            data_y:  pd.DataFrame,
                            alpha_min_ratio: float=ALPHA_MIN_RATIO):
    '''Choose a penalty strength, and with it, the features. Exits pipe to avoid repeating expensive gridsearches. From: https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html#Choosing-penalty-strength-alpha'''

    print('Choosing penalty strength via cross-validation...')

    # Load and deserialize data
    # data_x, data_y = pd.read_pickle(resolve_path(filepath_x)), pd.read_pickle(resolve_path(filepath_y))

    '''Max iterations can be lower since we are only interested in alphas, not coefficients.'''
    coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=alpha_min_ratio, max_iter=100))
    warnings.simplefilter('ignore', UserWarning)
    warnings.simplefilter('ignore', FitFailedWarning)
    coxnet_pipe.fit(data_x, data_y)

    '''5-fold cross-validation estimating concordance for each estimated alpha'''
    estimated_alphas = coxnet_pipe.named_steps['coxnetsurvivalanalysis'].alphas_
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    gcv = GridSearchCV(
        make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
        param_grid={'coxnetsurvivalanalysis__alphas': [[v] for v in estimated_alphas]},
        cv=cv,
        error_score=0.5,
        n_jobs=1,
    ).fit(data_x, data_y)

    print(gcv.cv_results_)

    # Serialize results and visualization information
    pd.DataFrame(gcv.cv_results_).to_pickle(resolve_path('../results/cv_results.pkl'))
    with open(resolve_path('../results/best_params.json'), 'w', encoding='ascii') as file:
        json.dump(gcv.best_params_, file, ensure_ascii=False, indent=4)
    best_model = gcv.best_estimator_.named_steps['coxnetsurvivalanalysis']
    pd.DataFrame(best_model.coef_, index=data_x.columns, columns=['coefficient']).to_pickle(resolve_path('../results/best_coefs.pkl'))

    print('Choosing penalty strength via cross-validation done.')


# ==============================================================================


def save_concord_idx_vs_alphas_plot(filepath_cv_results: str='../results/cv_results.pkl'):
    '''Visualize cross-validation results. Exits pipeline because it's not modifying the data nor results. From: https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html#Choosing-penalty-strength-$$\alpha$$'''

    print('Generating concordance vs. alphas plot...')

    cv_results = pd.read_pickle(resolve_path(filepath_cv_results))
    with open(resolve_path('../results/best_params.json'), 'r') as file:
        best_alpha = json.loads(file.read())['coxnetsurvivalanalysis__alphas'][0]
    alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    _, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.set_title('Mean concordance index over a range of penalty strengths (with best strength indicated)')
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale('log')
    ax.set_ylabel('Concordance index')
    ax.set_xlabel('penalty strength $\\alpha$')
    best_mean = mean[alphas[alphas == best_alpha].index[0]]
    # Label the vertical line
    ax.axvline(best_alpha, color='C1')
    ax.text(x=best_alpha, y=0.383, s=f'$\\alpha$={best_alpha:.4f}', horizontalalignment='center', verticalalignment='top')
    # Label the horizontal line
    ax.axhline(best_mean, color='dimgrey')
    ax.text(x=0.001, y=best_mean, s=f'{best_mean:.2f}  ', verticalalignment='center', horizontalalignment='right')
    ax.axhline(0.5, color='grey', linestyle='--')
    ax.grid(True)
    plt.savefig(resolve_path('../plots/concord_idx_over_alphas.png'))

    print('Generating concordance vs. alphas plot done.')



def save_features_and_coeffs_plot(filepath_best_coefs: str ='../results/best_coefs.pkl',
                                  n_shown: int = 20):
    '''From: https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html#Choosing-penalty-strength-alpha'''

    print('Generating features and coefficients plot...')

    best_coefs = pd.read_pickle(resolve_path(filepath_best_coefs))
    non_zero = (best_coefs.iloc[:, 0] != 0).sum()
    print(f'Number of non-zero coefficients: {non_zero}')

    non_zero_coefs = best_coefs.query('coefficient != 0').copy()

    print(non_zero_coefs['coefficient'])

    # TK temporary measure to get drugs with more observations than less
    # larger_drugs_indices = non_zero_coefs.index.isin(mask_filtering_out_small_drugs())
    # non_zero_coefs = non_zero_coefs.loc[larger_drugs_indices]

    # For human-readability, convert RxCUIs to pre-normalized drug names
    non_zero_coefs['feature_names'] = ''
    for feature_rxcui in non_zero_coefs.index:
        marked_4_interest = feature_rxcui.startswith('$$$')
        name = ' '.join(recover_drug_name_else_return(
            feature_rxcui).replace('$$$', '').split(' ')[0:4])
        non_zero_coefs.loc[feature_rxcui, 'feature_names'] = ('$$$' if marked_4_interest else '') + name if name != '' else feature_rxcui

    # Arrange the bars by the absolute value of the coefficients
    coef_order = non_zero_coefs['coefficient'].abs().sort_values(ascending=True).index

    _, ax = plt.subplots(figsize=(10, 8))
    ordered_coefs = non_zero_coefs.loc[coef_order]
    ordered_coefs = ordered_coefs.reset_index().set_index('feature_names')
    ordered_coefs.plot.barh(ax=ax, legend=False)
    ax.set_title('Features and their coefficients, sorted by coeff.')
    ax.set_ylabel('Feature')
    ax.set_xlabel('coefficient')
    ax.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig(resolve_path('../plots/features_&_coefs.png'))

    print('Generating features and coefficients plot done.')



def save_prognosis_plot_for_specific_feature(level: str,
                                             filepath_best_params: str='../results/best_params.json',
                                             filepath_x: str='data_x.pkl',
                                             filepath_y: str='data_y.pkl'):
    '''From: https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html#Survival-and-Cumulative-Hazard-Function'''

    print('Generating prognosis plot for specific feature...')

    best_params = json.loads(resolve_path(filepath_best_params))
    data_x = pd.read_pickle(filepath_x)
    data_y = pd.read_pickle(filepath_y)

    coxnet_pred = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True))
    coxnet_pred.set_params(**best_params)
    coxnet_pred.fit(data_x, data_y)

    surv_fns = coxnet_pred.predict_survival_function(data_x)

    time_points = np.quantile(data_y['t.tdm'], np.linspace(0, 0.6, 100))
    legend_handles = []
    legend_labels = []
    _, ax = plt.subplots(figsize=(9, 6))
    for fn, label in zip(surv_fns, data_x.loc[:, level].astype(int)):
        (line,) = ax.step(time_points, fn(time_points), where='post', color=f'C{label}', alpha=0.5)
        if legend_handles.shape[0] <= label:
            name = 'positive' if label == 1 else 'negative'
            legend_labels.append(name)
            legend_handles.append(line)

    ax.set_title(f'Survival probability vs. Time, for level {str(level)}')
    ax.legend(legend_handles, legend_labels)
    ax.set_xlabel('time (days)')
    ax.set_ylabel('Survival probability')
    ax.grid(True)

    plt.savefig(f'../plots/prog_for_level_{str(level)}.png')

    print('Generating prognosis plot for specific feature done.')


# TK Run
with open(resolve_path('../results/best_params.json'), 'r') as file:
    best_alpha = json.loads(file.read())['coxnetsurvivalanalysis__alphas'][0]
fitted_cph = display_summary(*format_4_assumption_tests(), alpha=best_alpha)
validate_hazard_proportionality(*format_4_assumption_tests(), fitted_cph)
# examine_outliers(*format_4_assumption_tests(), fitted_cph)
# detect_nonlinearity(*format_4_assumption_tests(), fitted_cph)

# save_coefficients_vs_alpha_plot(*format_4_surv_analysis(*format_4_assumption_tests()))
# choose_penalty_strength(*format_4_surv_analysis(*format_4_assumption_tests()))
# save_concord_idx_vs_alphas_plot()
# save_features_and_coeffs_plot()
#save_prognosis_plot_for_specific_feature(level='')

