from matplotlib import pyplot as plt
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer




# '''Loading RxCUI dataframes'''
# # Get dataframes from serialized forms
# processed_ingredients = pd.read_csv('processed_ingredients.csv')
# processed_orders = pd.read_csv('processed_orders.csv')

# print(processed_ingredients)
# print(processed_ingredients.info())

# print(processed_orders)
# print(processed_orders.info())