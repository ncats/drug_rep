import os
from urllib.parse import urlencode, quote_plus
import requests
import pandas as pd



CODE_DIR = '~/Repos/drug_rep/src/'
CSV_DIR = '~/Repos/drug_rep/data/'



def resolve_path(relative_path: str) -> str:
    '''Resolve the relative file path using os.path.abspath()'''
    return os.path.abspath(os.path.join(CODE_DIR, relative_path))
    


def query_getRxTermDisplayName(search_str: str) -> str:
    '''Helper function: Query the RxNorm getRxTermDisplayName API, which returns "[s]trings to support auto-completion in a user interface."  Used in actuality to re-obtain concept names from RxCUIs.'''
    rxnorm_url = 'https://rxnav.nlm.nih.gov/REST/rxcui/'
    encoded_rxnorm_payload = urlencode({'prop':'names'}, quote_via=quote_plus)
    # Make a GET request:
    print('Making request to RxNorm...')
    response = requests.get(rxnorm_url + str(search_str) + '/allProperties.json' + '?' + encoded_rxnorm_payload)
    # Check the response status code:
    if response.status_code == 200:
        try:
            # Parse the response as JSON:
            '''Note: Naming systems may not be consistent.'''
            name = response.json()['propConceptGroup']['propConcept'][0]['propValue']
            print('Request succeeded.')
            # Assuming every JSON object with a 'propValue' key has a name as value
            return str(name)
        except KeyError:
            return ''
    else:
        print('Request failed with status code:', response.status_code)
        return ''



def query_findRxcuiByString(search_str: str, verbose: bool=False) -> str:
    '''Helper function: Query the RxNorm findRxcuiByString API, which returns "[c]oncepts with a specified name".'''
    # rxnorm_url = 'http://localhost:4000/REST/rxcui.json'
    rxnorm_url = 'https://rxnav.nlm.nih.gov/REST/rxcui.json'
    # Encode the payload
    rxnorm_payload = dict(name=search_str, search=2)
    # Parameter `search` (precision): 2: Best match (exact or normalized)
    encoded_rxnorm_payload = urlencode(rxnorm_payload, quote_via=quote_plus)
    # Make a GET request:
    print('Making request to RxNorm...')
    response = requests.get(rxnorm_url + '?' + encoded_rxnorm_payload)
    # Check the response status code:
    if response.status_code == 200:
        try:
            # Parse the response as JSON:
            rxcui = response.json()['idGroup']['rxnormId'][0]
            print('Request succeeded.')
            # Assuming every JSON object with a 'rxnormId' key has a RxCUI
            return str(rxcui)
        except KeyError:
            return ''
    else:
        print('Request failed with status code:', response.status_code)
        return ''
        
        
        
def impute_multinominal(selected_cols: pd.DataFrame,
                        outfile_suffix: str,
                        outfile_name: str):

    import os
    os.environ['R_HOME'] = '/usr/lib/R'
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    # Set index to `Subject` ID, to concatenate with similar DataFrames later
    selected_cols = selected_cols.set_index('Subject')
    
    # Import the R packages
    missForest = importr('missForest')
    pandas2ri.activate()

    # Convert pandas DataFrame to R data.frame
    try:
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_data = ro.conversion.py2rpy(selected_cols)
    except Exception as e:
        print(f"Error converting pandas DataFrame to R data.frame: {e}")

    # Apply missForest imputation
    imputed_data_r = missForest.missForest(r_data)

    # Convert the imputed R data.frame back to a pandas DataFrame
    try:
        with localconverter(ro.default_converter + pandas2ri.converter):
            imputed_data = ro.conversion.py2rpy(imputed_data_r[0])
    except Exception as e:
        print(f"Error converting R data.frame to pandas DataFrame: {e}")

    # Serialize the DataFrame
    pd.DataFrame(imputed_data).to_pickle(resolve_path(
        f'../intermediates/{outfile_name}{file_suffix}.pkl'))
