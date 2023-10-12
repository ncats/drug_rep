# TK TODO: one final bar graph of the medications themselves, not by categories. but that requires NER

# print('Requiring GPU...')
# spacy.require_gpu()
# spacy.prefer_gpu()
# print('Loading entity-linking model...')
# #nlp = spacy.load('en_core_sci_scibert')
# nlp = spacy.load('en_core_sci_lg')

# # Add the abbreviation pipe to the spacy pipeline.
# print('Adding abbreviation detector to pipe...')
# nlp.add_pipe("abbreviation_detector")
# # Add the entity-linker pipe to the spacy pipeline.
# print('Adding entity linker to pipe...')
# nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True,
#                                         "linker_name": "umls"})

'''The pipeline is as follows: transformer, tagger, attribute_ruler, lemmatizer, parser, ner, abbreviation_detector, scispacy_linker'''

# Case where there are multiple entities per one cell
# Case where extra words confuse entity-linking (e.g. 'Adult -' in 'Sodium Phosphate Enema Fleets Adult -')

# Side case where 'GBM' is linked to 'Glomerular Basement Membrane' and not 'glioblastoma'

# '''Recognize and link drug entity names to UMLS Metathesaurus, creating the one-hot dataframe in the process.'''
# unique_subjects = confirmed_df['Subject'].unique()
# onehot_encoded_medications = pd.DataFrame()
# onehot_encoded_medications['Subject'] = unique_subjects

# print('Analysing input doc...')
# doc = nlp('DiphenhydrAMINE 50 mg capsule')
# for entity in doc.ents:
#     linker = nlp.get_pipe("scispacy_linker")
#     for umls_ent in entity._.kb_ents:
#         print(linker.kb.cui_to_entity[umls_ent[0]])


# ==============================================================================


# # Suggestions: RxNorm and NLM MetaMap
# # Use NER on `Order Name` to parse out drug names

# '''Initialize a MIMIC pipeline with an bc4chemd and an i2b2 NER model (and OntoNotes NER as control)'''
# '''Said MIMIC pipeline is as follows: "tokenize", "pos", "lemma", "depparse"'''
# # Run one-time (refer to `download_method` param below):
# #stanza.download('en', package='mimic', processors={'ner': ['bc4chemd', 'i2b2', 'ontonotes', 'conll03']})
# nlp = stanza.Pipeline('en', package='mimic', processors={'ner': ['bc4chemd', 'i2b2', 'ontonotes', 'conll03']}, use_gpu=True, download_method=None)

# # Read lookup TSV file
# filepath_Inxight = ''
# # med_unii_lookup = pd.read_csv('../inxight/frdb-drugs.tsv', sep='\t', header=0)
# #
# # Ideal solution: Pretrain(?) NER with domain-specific (to Inxight) "dictionary"/"ontology" so Stanza can directly recognize GBM medication free text into entities searchable through Inxight

# labels = ['\t\ttoken: ', '\t\tchem_ner: ',
#           '\t\ttreat_ner: ', '\t\tvernac_ner: ', '\t\tother_ner: ']

# for cell in confirmed_df['Order_Name']:
#     doc = nlp(cell)
#     is_anything_printed = 0
#     # Go through all "sentences"
#     for sent in doc.sentences:
#         bc4chemd_advantage_on_i2b2 = 0
#         ontonotes_advantage_on_conll03 = 0
#         ontonotes_advantage_on_i2b2 = 0
#         for token in sent.tokens:
#             # print out all tokens and the tokens as recognized named entities
#             #print(zip(labels, list(token.text, *token.multi_ner)))
#             # Calculate which NER is better
#             # if token.multi_ner[0] != 'O' and token.multi_ner[1] == 'O':
#             #     bc4chemd_advantage_on_i2b2 += 1
#             #     print(f'\t\ttoken: {token.text}\t\tchem_ner: {token.multi_ner[0]}\t\ttreat_ner: {token.multi_ner[1]}')
#             on = token.multi_ner[2]
#             if on != 'O' and token.multi_ner[3] == 'O' and 'QUANTITY' not in on and 'CARDINAL' not in on and 'PERCENT' not in on and 'TIME' not in on and 'PERSON' not in on:
#                 ontonotes_advantage_on_conll03 += 1
#                 print(
#                     f'\t\ttoken: {token.text}\t\tvernac_ner: {token.multi_ner[2]}\t\tother_ner: {token.multi_ner[3]}')
#             # if 'ORG' in token.multi_ner[2] and token.multi_ner[1] == 'O':
#             #     ontonotes_advantage_on_i2b2 += 1
#             #     print(
#             #         f'\t\ttoken: {token.text}\t\tvernac_ner: {token.multi_ner[2]}\t\ttreat_ner: {token.multi_ner[1]}')
#         if bc4chemd_advantage_on_i2b2 > 0 and ontonotes_advantage_on_conll03 > 0:
#             print('bc4chemd_advantage_on_i2b2: ', bc4chemd_advantage_on_i2b2,
#               '\tontonotes_advantage_on_conll03: ', ontonotes_advantage_on_conll03, '\ttotal: ', sent.tokens.shape[0])
#             is_anything_printed += 1
#     if is_anything_printed > 0:
#         print('===^^^prev sent^^^===')
#         print('\n')
#         print('===vvvnext sentvvv===')
#     print("yeah")

# for row in confirmed_df[['Subject', 'Order_Name', 'drug_ingredients']].head(10).itertuples(index=False):
#     second_two_cells = dict(order=nlp(row.Order_Name),
#                             ingred_CSV=nlp(row.drug_ingredients))
    # for cell in cells:
    #     i2b2_ent_candidates = []
    #     bc4chemd_ent_candidates = []
    #     controls = []
    #     for ent in cell.ents:
    #         if ent['type'] == 'TREATMENT':
    #             i2b2_ent_candidates.append(ent)
    #         elif ent['type'] == 'CHEMICAL':
    #             bc4chemd_ent_candidates.append(ent)
    #         elif ent['type'] == 'QUANTITY':
    #             controls.append(ent)
    #     '''BC4ChemD sometimes mis-recognizes units (e.g. 'mcg/mL', 'mmol/mL') as (S-)CHEMICALs'''
    #     satisfiers = find_satisfying_2combs(bc4chemd_ent_candidates,
    #                            controls, check_entity_text_intersection)
    #     print(satisfiers)
    #     # Replace candidates list with filtered list (delenda listed in `satisfiers`):
    #     bc4chemd_ent_candidates = [ent for ent in bc4chemd_ent_candidates if ent != delendum for delendum in zip(*satisfiers)[0]]

# TK [findRxcuiByString API requests here]

# Case where none of the NERs identify anything (e.g. 'Sunblock')
# Case where the domain NER identify multiple entities in one cell (i.e. more than 1 of B- or S-)

# Known ungettable specific entities:
# - Inj should be Injection aka not part of a CHEMICAL
# - Enzastaurin aka LY317615 is a TREATMENT (i think)
# - 5MG/GM should be 5 MG/GM (gram???) aka a QUANTITY

# OntoNotes identifies quantities, percentages, cardinals, ordinals, and times whereas ConLL doesn't at all

# ConLL can identify ORGs (i think 'organizations') which could correlate with brand names? (But is made irrelevant by i2b2)
# OntoNote sometimes identifies medications (e.g. 'Morphine Sulfate Inj') as persons
# Non-domain NERs (OntoNote nor ConLL) aren't good enough for controls.
# BC4ChemD can't really identify brands

# Perform one survival analysis with "primary ingredient", and one with CSV of ingredients IN LEXICOGRAPHIC ORDER
# Currently can't distinguish between different dosages

# Accept only alphanumeric in Names (non-alphanumeric in quantities, percentages, cardinals, ordinals, and times okay) (e.g. no parentheses, no slashes)

'''Ingredients in medication-order'''
'''Cells are comma-separated names, sometimes related (describing the same ingredient), sometimes mutually exclusive (different ingredients).'''
'''Prioritizes names at the front of the comma-separated string.'''

'''Full medication-order names (may contain brand names)'''


import itertools

import pandas as pd  # for list combinatorics


def check_entity_text_intersection(ent1, ent2):
    '''Helper function: Check intersections for both permutations (of the two entity arguments).'''
    if (ent1['start_char'] <= ent2['start_char']  # `ent1` comes before `ent2`
        and ent2['start_char'] <= ent1['end_char']):  # `ent2` begins before `ent1` ends
        return True
    if (ent2['start_char'] <= ent1['start_char']  # `ent2` comes before `ent1`
        and ent1['start_char'] <= ent2['end_char']):  # `ent1` begins before `ent2` ends
        return True


def find_satisfying_2combs(ents1, ents2, check_binary):
    '''Helper function: Find all combinations (order doesn't matter) which satisfy a binary predicate.'''
    satisfiers = []
    for twople in list(itertools.product(ents1, ents2)):
        if check_binary(*twople): satisfiers.append(twople)


# ==============================================================================


# # TK use this to indicate the drugs NOT meant for tumor/cancer
# # HTTP Request to NCATS Stitcher
# response = requests.get('https://stitcher.ncats.io/api/stitches/latest/' + med_unii)
# if response.status_code != 200:
#    print('Request failed with status code:', response.status_code)
# stitcher_base_json = response.json()

# # Get conditions jsons
# base_conditions = stitches_base_json['sgroup']['properties']['conditions']
# conditions_base64 = [base_conditions[i]['value'] for i in base_conditions]

# # Decode (base 64)
# conditions_decoded = base64.b64decode(conditions_array).decode('utf-8')
# conditions_parsed = json.loads(conditions_decoded)

# # Get condition names as list
# med_conditions = [value['name'] for value in conditions_jsons]


def obtain_onehot_drugs_encoding(filepath_ingreds: str='processed_ingredients.pkl',
                                 filepath_orders: str='processed_orders.pkl'):
    '''One-hot encode drugs based on (non-)administration. Exits pipe because further processing is done elsewhere.'''

    '''Loading RxCUI dataframes'''
    # Get dataframes from serialized forms
    processed_ingredients = pd.read_pickle(filepath_ingreds)
    processed_orders = pd.read_pickle(filepath_orders)

    # Combine said dataframes into one, collapsing into one column per patient
    combined_ingreds_and_orders = pd.concat([processed_ingredients, processed_orders], axis='index')

    # def parse_str_to_tuple(cell: str):
    #     '''Convert string representation of tuples to actual tuples (because APPARENTLY pandas can't handle tuples (as elements) in dataframes).'''
    #     try:
    #         na_corrected_cell = re.sub(r'(?<=, )nan(?=\))', 'None', str(cell))
    #         twople = ast.literal_eval(na_corrected_cell)
    #         return (twople[0], np.nan) if twople[1] is None else twople
    #     except ValueError:
    #         return np.nan
    # combined_ingreds_and_orders = combined_ingreds_and_orders.applymap(parse_str_to_tuple)

    '''Medication one-hot encoding'''
    # Keep a list of (1-column) dataframes (to be concatenated at the very end, because updating a dict with dataframes and then concatenating is more efficient than repeatedly inserting columns into a dataframe)
    onehot_encoded_drugs_cols = {}

    # Initialize a dictionary to keep track of drug frequency
    drug_freq_dict = {}

    # Loop over each patient
    for patient in combined_ingreds_and_orders.columns:
        # Loop over the 2-ples in each column
        for (_, rxcui) in combined_ingreds_and_orders[patient].dropna():
            # Update the dictionary (increment the corresponding value)
            drug_freq_dict[rxcui] = drug_freq_dict.get(rxcui, 0) + 1
            # Check if `rxcui` is already a column in `onehot_encoded_drugs`
            if rxcui not in onehot_encoded_drugs_cols.keys():
                # If not, initialize a dataframe with the given unique patient ID as its column
                new_col = pd.DataFrame(0, columns=[rxcui], index=combined_ingreds_and_orders.columns, dtype=int)
                # One-hot for (first) occurrence
                new_col.loc[patient, rxcui] = 1
                onehot_encoded_drugs_cols.update({rxcui: new_col})
            else:
                # If so, increment the element (will coerce all nonzeroes to 1s later)
                onehot_encoded_drugs_cols[rxcui].loc[patient, rxcui] += 1

    # Concatenate all the (1-column) dataframes in the dict, as promised
    onehot_encoded_drugs = pd.concat(onehot_encoded_drugs_cols.values(), axis='columns')

    # Remove column of unidentified drugs (i.e. '')
    # TK what's the deal with the one unnamed column (where there's supposed to be rxcuis)?
    onehot_encoded_drugs = onehot_encoded_drugs.drop([''], axis='columns')

    # Coerce all nonzero elements (number of orders per patient) to 1, creating a true ONE-hot encoding
    onehot_encoded_drugs = onehot_encoded_drugs.applymap(lambda n: 1 if n > 0 else 0)

    # Sort (in decreasing order by frequency) the drug frequency dictionary
    drug_freq_df = pd.DataFrame.from_dict(drug_freq_dict, columns=['counts'], orient='index', dtype=int)
    drug_freq_df = drug_freq_df.sort_values(by='counts', ascending=False)
    drug_freq_df = drug_freq_df.reset_index(names='rxcui')

    # Re-obtain concept names using the above function
    drug_freq_df['concepts'] = drug_freq_df['rxcui'].apply(query_getRxTermDisplayName)

    # Sort (in decreasing order by frequency) the columns of the one-hot drug encoding
    onehot_encoded_drugs = onehot_encoded_drugs[drug_freq_df.index.tolist()]

    # Serialize the drug frequencies
    drug_freq_df.to_csv('./results/drug_freq_df.csv', index=False)

    # Serialize the one-hot encoding
    onehot_encoded_drugs.to_pickle('./intermediates/onehot_encoded_drugs.pkl')


def split_ingredient_column(confirmed_df: pd.DataFrame) -> pd.DataFrame:
    '''Does not exit pipe. Since the `drug_ingredients` column consists of CSVs as values, string-split the `drug_ingredients` columns into separate single-column DataFrames, similar to MS Excel's 'Text to Columns' functionality, then replace the unsplit `drug_ingredients` column with the split columns.  Semantically, there is no difference between orders and their constituent ingredients.'''

    print('Splitting `drug_ingredients` column...')

    # Rename `Order_Name` column to make column accessing-by-label easier
    confirmed_df = confirmed_df.rename(columns={'Order_Name': 'drug_'})

    # Split the CSV column
    split_ingredients = confirmed_df['drug_ingredients'].astype(
        str).str.split(', ', expand=True)

    # Rename the split columns to be human-readable
    split_ingredients = split_ingredients.rename(
        columns={col: f'drug_{col}' for col in split_ingredients.columns})

    # Obtain the DataFrame less the unsplit CSV column
    no_ingredients = confirmed_df.drop(columns='drug_ingredients')

    # Join the rest of the DataFrame with the split columns
    rejoined_ingredients = no_ingredients.join(split_ingredients)

    print('Splitting `drug_ingredients` column done.')

    return rejoined_ingredients

    # TK DEPRECATED '''Due to the conflated concept of `drug_ingredients`, certain supposed mutually exclusive ingredients (but nevertheless distinct values in (what used to be) `drug`) are synonyms of other ingredients, or meaningless descriptors (e.g. 'Vaccine', 'Investigational Agent'). It is the case that some synonyms/descriptors always appeared in the same CSVs as their antecedents. This method finds all such co-occurrent `drug` values via hashing.'''

# Melt the DataFrame from wide to tall (so that `OneHotEncoding` will correctly group/collapse drugs (which appear in different columns that are melted into a single column))
drugs = drugs.melt(id_vars='Subject',
                          value_name='drug')[['Subject', 'drug']]

# Drop all missing values (artifacts from `split_ingredient_column`)
drugs = drugs.dropna()

# ; Maintains of-interest and pre-GBM marking
split_ingredients_cols = [
    col.astype(str).apply(lambda x: (
        '$$$ ' if '$$$' in x else '') + ('*** ' if '***' in x else ''))
    for _, col in pivoted_ingreds.items()
]
# TK fix the above lol...it's splitting after prefixing (which is the reverse). it's supposed to check for a prefix, then split, then prefix
# TK figure out string concatenation BASED ON BOOLEAN SERIES/COLUMNS

# Concatenate all those single-column DataFrames into a new DataFrame
expanded_ingreds = pd.concat(
    split_ingredients_cols, axis='columns', join='outer')



def process_ingredients(ingredient: str):
    '''Get ingredient RxCUIs via raw-inputted (no non-RxNorm NER) RxNorm.'''
    '''Check whether the ingredient had been marked.'''
    split_ingredient = ingredient.split(' ')
    mark_pos = 0
    marked_4_pre_GBM = False
    marked_4_interest = False
    if len(split_ingredient) >= 2:
        for i in range(2):
            marked_4_pre_GBM = split_ingredient[i] == '***'
            marked_4_interest = split_ingredient[i] == '$$$'
            if marked_4_pre_GBM or marked_4_interest:
                mark_pos = i
                break
    if marked_4_pre_GBM or marked_4_interest:
        # For querying, un-split while excluding marking prefix
        ingredient = ' '.join(split_ingredient[mark_pos:])

    print(f'Querying for {ingredient}...')

    wait()

    ingred_rxcui = query_findRxcuiByString(ingredient)
    # Mark queried rxcui to be consistent with markedness of query term
    return ('$$$ ' if marked_4_interest else '') + ('*** ' if marked_4_pre_GBM else '') + ingred_rxcui



'''Create a multi-index of (['rxcui', 'text'], *) for each column'''
# Create a multi-index to combine both (processed and original) DataFrames as two levels of each column of each ingredient DataFrame
processed_ingredients.columns = pd.MultiIndex.from_product(
    [['rxcui'], processed_ingredients.columns])
expanded_ingreds.columns = pd.MultiIndex.from_product(
    [['text'], expanded_ingreds.columns])
# Create a multi-index to combine both (processed and original) DataFrames as two levels of each column of each order DataFrame
normalized_drugs.columns = pd.MultiIndex.from_product(
    [['rxcui'], normalized_drugs.columns])
pivoted_order.columns = pd.MultiIndex.from_product(
    [['text'], pivoted_order.columns])

'''Concatenate the 'rxcui' and the 'text' levels'''
# Concatenate the ingredients levels
combined_ingreds = pd.concat(
    [expanded_ingreds, processed_ingredients], axis='columns')
# Concatenate the orders levels
combined_orders = pd.concat(
    [pivoted_order, normalized_drugs], axis='columns')

# Column-bind `combined_ingreds` and `combined_orders` along the `Subject` index
combined_drugs = combined_ingreds.join(combined_orders, how='outer')



def wait():
    '''RxNorm API is rate-limited to 20 requests per second (https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html#OnlineService)'''
    global start_time
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time < 0.5:
        # If not enough time has passed since the last half-second,
        # then wait until a full half-second has passed
        time.sleep(0.5 - elapsed_time)
    # Reset the start_time
    start_time = time.time()

# Strip extraneous text further (of course, leaving kept text contiguous).  Note that match-group labeling is 1-based (not 0-based)
def process_string(string: str):
    match = re.search(r'Nonform Additive: (.*mg)', string)
    return match.group(1) if match else re.sub(r'(.*?mg(?:/mL|\stablet|\scapsule)?).*', r'\1', string)

post_df['Order_Name'] = post_df['Order_Name'].apply(process_string)




# Escape metacharacters BEFORE joining with a pipe (meta)character
for i, name in enumerate(notanda):
    notanda[i] = re.escape(name)

# Create OR-logic regex pattern from list
or_pattern = '|'.join(notanda)

# Stripping (right; which doesn't make sense) extraneous text to aid (a redo of) RxNorm normalization
right_stripping = {
    'albumin 25% infusion 0.25 gs / ml': 'albumin 25% infusion 0.25 g / ml',
    'albumin 5% infusion 0.05 gs / ml': 'albumin 5% infusion 0.05 g / ml',
}
