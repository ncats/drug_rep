from gensim.models.keyedvectors import KeyedVectors

# # BIO_WORD_VECS = KeyedVectors.load_word2vec_format('bio_embedding_intrinsic', binary=True)
# # print('Vectors loaded.')
# for cell in cells:
#     result = BIO_WORD_VECS.similar_by_word(cell.split()[0].lower())
#     most_similar_key, similarity = result[0]  # look at the first match
#     print(f"{most_similar_key}: {similarity:.4f}")
# get_vecattr()
# doesnt_match()

from matplotlib import pyplot as plt
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd

from util import resolve_path

from lifelines.datasets import load_rossi
rossi = load_rossi()
print(rossi.head(7))
print(rossi.info())
cph = CoxPHFitter()

cph.fit(rossi, 'week', 'arrest')

# cv_results = pd.read_pickle(resolve_path('../results/cv_results.pkl'))
# print(cv_results)

# best_model_coefs = pd.read_pickle(resolve_path('../results/best_coefs.pkl'))
# print(best_model_coefs[best_model_coefs['coefficient'] != 0])


