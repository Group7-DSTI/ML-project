import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import words as english_words
import string
import re
from sklearn.model_selection import learning_curve
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
nltk.download('punkt')
nltk.download('words')

def preprocess_data(train_file_path, test_file_path, valid_file_path, valid_sub_file_path):
    # Lecture des fichiers
    df = pd.read_csv(train_file_path, delimiter='\t', encoding='latin1')
    test_data = pd.read_csv(test_file_path, delimiter='\t', encoding='latin1')
    validation_data = pd.read_csv(valid_file_path, delimiter='\t', encoding='latin1')
    v1 = pd.read_csv(valid_sub_file_path, header=1, names=['prediction_id', 'predicted_score'])
    
    # Fusion avec les données de validation
    validation_data = pd.merge(v1, validation_data, left_on='prediction_id', right_on='domain1_predictionid')
    
    # Suppression des colonnes inutiles dans le DataFrame d'entraînement
    df = df.drop(columns=['rater1_domain1','rater2_domain1','rater3_domain1','rater1_domain2','rater2_domain2',
                          'rater1_trait1','rater1_trait2','rater1_trait3','rater1_trait4','rater1_trait5',
                          'rater1_trait6','rater2_trait1','rater2_trait2','rater2_trait3','rater2_trait4',
                          'rater2_trait5','rater2_trait6','rater3_trait1','rater3_trait2','rater3_trait3',
                          'rater3_trait4','rater3_trait5','rater3_trait6'])
    
    return df, test_data, validation_data