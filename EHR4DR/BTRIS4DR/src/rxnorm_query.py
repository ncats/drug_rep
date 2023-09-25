'''RxNorm API Queries Module'''
from urllib.parse import urlencode, quote_plus
import requests


def query_getRxTermDisplayName(search_str: str) -> str:
    '''Re-obtain concept names from RxCUIs.'''
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
    '''Helper function: Query the RxNorm findRxcuiByString API.'''
    rxnorm_url = 'https://rxnav.nlm.nih.gov/REST/rxcui.json'
    # Encode the payload
    rxnorm_payload = dict(name=search_str, search=2)
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


