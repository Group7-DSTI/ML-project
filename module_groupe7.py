from FilePaths import FilePaths
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


def preprocess_data():
    """
    Read and preprocess data from the specified files.

    Returns:
    - df (pd.DataFrame): The training data DataFrame.
    - test_data (pd.DataFrame): The testing data DataFrame.
    - validation_data (pd.DataFrame): The preprocessed validation data DataFrame.
    """
    fp = FilePaths()
    
    # Reading files
    df = pd.read_csv(fp.train_file_path, delimiter='\t', encoding='latin1')
    test_data = pd.read_csv(fp.test_file_path, delimiter='\t', encoding='latin1')
    validation_data = pd.read_csv(fp.valid_file_path, delimiter='\t', encoding='latin1')
    v1 = pd.read_csv(fp.valid_sub_file_path, header=1, names=['prediction_id', 'predicted_score'])

    # Merging with validation data
    validation_data = pd.merge(v1, validation_data, left_on='prediction_id', right_on='domain1_predictionid')

    # Deleting useless columns in the dataframe
    df = df.drop(columns=['rater1_domain1', 'rater2_domain1', 'rater3_domain1', 'rater1_domain2', 'rater2_domain2',
                          'rater1_trait1', 'rater1_trait2', 'rater1_trait3', 'rater1_trait4', 'rater1_trait5',
                          'rater1_trait6', 'rater2_trait1', 'rater2_trait2', 'rater2_trait3', 'rater2_trait4',
                          'rater2_trait5', 'rater2_trait6', 'rater3_trait1', 'rater3_trait2', 'rater3_trait3',
                          'rater3_trait4', 'rater3_trait5', 'rater3_trait6'])

    return df, test_data, validation_data



def evaluate_models_1(validation_data, df, models):
    """
    Evaluate different regression models using TF-IDF vectors as input features.

    Parameters:
    - validation_data (DataFrame): DataFrame containing validation data.
    - df (DataFrame): DataFrame containing training data.
    - models (dict): Dictionary containing regression models to be evaluated.

    Returns:
    None (prints evaluation metrics for each model).
    """
    X = df['essay']
    y = df['domain1_score']
    vectorizer = TfidfVectorizer(max_features=10)

    X_train = vectorizer.fit_transform(X)
    X_test = vectorizer.transform(validation_data['essay'])
    y_train = y
    y_test = validation_data['predicted_score']

    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test) 
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        score = model.score(X_train, y_train)
        print('----------------------')
        print("Metrics:")
        print(f"MSE: {mse}")
        print(f"RMSE: {np.sqrt(mse)}")
        print(f"R²: {r2}")
        print(f"SCORE: {score}%")
        print('----------------------\n')

def over_samplingData(df):
    """
    Oversamples the data to balance the distribution of samples across categories.

    Args:
    - df (pd.DataFrame): The input DataFrame containing the data to be oversampled.

    Returns:
    - df2 (pd.DataFrame): The DataFrame with oversampled data.
    """
    # Calculate the average number of samples per category
    moyenne_par_categorie = df.groupby('essay_set').size().mean()
    
    # Get the exact size of category 8
    taille_exacte_8 = len(df[df['essay_set'] == 8])
    
    # Calculate the difference to achieve the average size
    reste = int(moyenne_par_categorie - taille_exacte_8)
    
    # Duplicate samples from category 8 to reach the average size
    duplique1 = df[df['essay_set'] == 8].sample(n=reste, replace=True)
    
    # Concatenate the original DataFrame with the duplicated samples
    df2 = pd.concat([df, duplique1])
    
    # Shuffle the DataFrame
    df2 = df2.sample(frac=1, random_state=42)
    
    return df2

def plot_distribution_essay_set(df, df2):
    """
    Plots the distribution of essay types before and after oversampling.

    Args:
    - df (pd.DataFrame): The original DataFrame containing the data.
    - df2 (pd.DataFrame): The DataFrame with oversampled data.

    Returns:
    - None: Displays the histogram plot.
    """
    # Plot the histogram of essay types in the original DataFrame
    plt.hist(df['essay_set'], bins=20, alpha=1, label='base')
    
    # Plot the histogram of essay types in the DataFrame with oversampled data
    plt.hist(df2['essay_set'], bins=30, alpha=1, label='final')
    
    # Set plot title and labels
    plt.title('Distribution of essay types')
    plt.xlabel('Type')
    plt.ylabel('Frequency')
    
    # Display legend
    plt.legend()
    
    # Show the plot
    plt.show()


def plot_max_scores_distribution(df2):
    """
    Plots the distribution of maximum scores for each essay set.

    Args:
    - df2 (pd.DataFrame): The DataFrame with oversampled data.

    Returns:
    - None: Displays the subplots of histograms.
    """
    # Calculate the maximum scores for each essay set
    max_scores = df2.groupby('essay_set')['domain1_score'].max()
    
    # Create subplots grid
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    
    # Iterate over essay set values and plot histograms
    for i, essay_set_value in enumerate(range(1, 9)):
        ax = axs[i // 4, i % 4]
        ax.hist(df2['domain1_score'][df2['essay_set'] == essay_set_value], bins=(max_scores[essay_set_value]) + 1, edgecolor='black')
        ax.set_title(f"Essay Set {essay_set_value}")
    
    # Adjust layout for better presentation
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def overSamplerSubCategorie(data, colonneDuType):
    """
    Applies random oversampling to balance the distribution of domain1_score within subcategories.

    Args:
    - data (pd.DataFrame): The DataFrame containing the data.
    - colonneDuType (str): The column representing the subcategory for oversampling.

    Returns:
    - pd.DataFrame: Resampled DataFrame with a balanced distribution of domain1_score.
    """
    over_sampler = RandomOverSampler(random_state=42)
    df_resampled = pd.DataFrame()

    # Iterate over unique values in the specified subcategory column
    for essay_set_value in data[colonneDuType].unique():
        data_subset = data[data[colonneDuType] == essay_set_value]
        X = data_subset.drop('domain1_score', axis=1)
        y = data_subset['domain1_score']
        
        # Apply random oversampling
        X_resampled, y_resampled = over_sampler.fit_resample(X, y)
        
        # Create a resampled DataFrame subset
        df_resampled_subset = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='domain1_score')], axis=1)
        
        # Concatenate the resampled subset to the overall resampled DataFrame
        df_resampled = pd.concat([df_resampled, df_resampled_subset], ignore_index=True)
    
    return df_resampled

def plot_scores_distribution(df3):
    """
    Plots the distribution of domain1_score for each essay set.

    Args:
    - df3 (pd.DataFrame): DataFrame containing the data.

    Returns:
    - None
    """
    max_scores = df3.groupby('essay_set')['domain1_score'].max()
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Distribution des différentes notes')

    # Iterate over each essay set and plot the distribution of domain1_score
    for i, essay_set_value in enumerate(range(1, 9)):
        ax = axs[i // 4, i % 4]
        ax.hist(df3['domain1_score'][df3['essay_set'] == essay_set_value], bins=(max_scores[essay_set_value])+1, edgecolor='black')
        ax.set_title(f"Essay Set de type {essay_set_value}")
        ax.set_xlabel('Note')
        ax.set_ylabel('Fréquence')

    plt.tight_layout()
    plt.show()


def plot_essay_set_histograms(df, df3):
    """
    Plots histograms of essay_set for two DataFrames and compares them.

    Args:
    - df (pd.DataFrame): First DataFrame containing essay_set data.
    - df3 (pd.DataFrame): Second DataFrame containing essay_set data.

    Returns:
    - None
    """
    plt.hist(df['essay_set'], bins=20, alpha=1, label='base')
    plt.hist(df3['essay_set'], bins=30, alpha=1, label='final')
    plt.title('Data distribution of essay_set ')
    plt.xlabel('Type')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def balance_categories(df, essay_set, random_seed=42):
    """
    Balance the categories of scores within the given DataFrame based on the specified essay_set column.

    Args:
    - df (pd.DataFrame): The DataFrame containing the data to be balanced.
    - essay_set (str): The column representing different essay sets.
    - random_seed (int): Seed for reproducibility in random sampling.

    Returns:
    - pd.DataFrame: The balanced DataFrame with an equal number of samples for each category.
    """
    # Get unique categories
    categories = df[essay_set].unique()
    
    # Initialize a list to store balanced data
    balanced_dataframe = []
    
    # Calculate the mean sample size for balancing
    mean_sample_size = int(df.groupby(essay_set).size().mean())
    
    # Balance each category
    for cat in categories:
        cat_data = df[df[essay_set] == cat]
        # Use replace=True to allow sampling more elements than present in the population
        balanced_dataframe.append(cat_data.sample(mean_sample_size, replace=True, random_state=random_seed))
    
    # Concatenate the balanced dataframes
    balanced_dataframe = pd.concat(balanced_dataframe)
    balanced_dataframe = balanced_dataframe.sample(frac=1, random_state=random_seed)
    
    return balanced_dataframe


def plot_histograms_comparison(df_base, df_final):
    """
    Plot histograms comparing the distribution of essay_set in two DataFrames and the distribution of domain1_score.

    Args:
    - df_base (pd.DataFrame): The first DataFrame for comparison.
    - df_final (pd.DataFrame): The second DataFrame for comparison.

    Returns:
    - None: Displays the histograms.
    """
    # Plot histogram for essay_set in both DataFrames
    plt.hist(df_base['essay_set'], bins=20, alpha=1, label='base')
    plt.hist(df_final['essay_set'], bins=30, alpha=1, label='final')
    plt.title('Histograms of Data')
    plt.xlabel('Type')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Plot histogram for domain1_score in the final DataFrame
    max_scores = df_final.groupby('essay_set')['domain1_score'].max()
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Distribution of Different Scores')
    
    for i, essay_set_value in enumerate(range(1, 9)):
        ax = axs[i // 4, i % 4]
        ax.hist(df_final['domain1_score'][df_final['essay_set'] == essay_set_value], bins=(max_scores[essay_set_value])+1, edgecolor='black')
        ax.set_title(f"Essay Set {essay_set_value}")
    
    plt.tight_layout()
    plt.show()




def evaluate_models_2(df,validation_data, models, max_features=10):
    """
    Evaluate machine learning models on the given dataset.

    Args:
        df (pd.DataFrame): The DataFrame containing essay and score data.
        models (dict): A dictionary of machine learning models to be evaluated.
        max_features (int): The maximum number of features for the TfidfVectorizer.

    Returns:
        list: A list of dictionaries containing evaluation results for each model.
    """
    X = df['essay']
    y = df['domain1_score']

    # Vectorize the essays using TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(X)
    X_test = vectorizer.transform(validation_data['essay'])
    y_train = y
    y_test = validation_data['predicted_score']

    results = []

    # Evaluate each model
    for model_name, model in models.items():
        model_results = {}
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        score_train = model.score(X_train, y_train)
        score_test = model.score(X_test, y_test)

        # Populate the results dictionary
        model_results['Modèle'] = model_name
        model_results['MSE'] = mse
        model_results['RMSE'] = rmse
        model_results['R² (Score de détermination)'] = r2
        model_results['Score Entraînement'] = score_train
        model_results['Score Test'] = score_test

        results.append(model_results)

    return results

def calculate_vector(essay,df):

    row = df[df['essay'] == essay]
    #essay_id = row['essay_id'].iloc[0]
    #essay_set = row['essay_set'].iloc[0]

    # Tokenisation du texte en mots
    words = nltk.word_tokenize(essay)
    num_words = len(words)

    # Nombre de mots commençant par une majuscule
    capital_words = sum(1 for word in words if word[0].isupper())
    capitalization_score = capital_words / num_words if num_words > 0 else 0
    
    # Nombre de mots uniques
    unique_words_count = len(set(words))
    style_score = unique_words_count / num_words if num_words > 0 else 0
    
    # Nombre de paragraphes
    num_paragraphs = essay.count('\n\n')
    
    # Nombre de caractères de ponctuation
    punctuation_count = sum(1 for char in essay if char in string.punctuation)
    punctuation_score = punctuation_count / len(essay) if len(essay) > 0 else 0
    
    # Score d'orthographe
    orthographe_score = sum(word.lower() not in english_word_set for word in words) / num_words if num_words > 0 else 0
    
    # Calcul du nombre de phrases et de la longueur moyenne des phrases
    sentences = nltk.sent_tokenize(essay)
    num_sentences = len(sentences)
    avg_sentence_length = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences) / num_sentences if num_sentences > 0 else 0
    structure_phrase_score = num_sentences / avg_sentence_length if avg_sentence_length > 0 else 0
    
    # Score de convention (utilisation correcte de la ponctuation)
    conventions_score = (1 - punctuation_score) * 0.5 
    
    # Score de champ lexical
    lexical_richness_score = unique_words_count / num_words if num_words > 0 else 0
    
    # Score d'efficacité des phrases
    sentence_efficiency_score = 1 / structure_phrase_score if structure_phrase_score > 0 else 0
    
    # Nombre de mots considérés comme péjoratifs
    pejorative_terms = {'bad', 'ugly', 'hate', 'stupid'}
    num_pejorative_terms = sum(1 for word in words if word.lower() in pejorative_terms)
    pejorative_term_score = num_pejorative_terms / num_words if num_words > 0 else 0
    
    # Taux d'utilisation excessive de la ponctuation
    overusage_punctuation = punctuation_count / num_words if num_words > 0 else 0
    
    mentions = re.findall(r'@(\w+)', essay)
    nombre_carac_special = len(mentions)

    # Création du vecteur
    vecteur_complexite = [
        #essay_set,
        #essay_id,
        nombre_carac_special,
        num_words,
        punctuation_score,
        orthographe_score,
        structure_phrase_score,
        conventions_score,
        lexical_richness_score,
        sentence_efficiency_score,
        pejorative_term_score,
        punctuation_count,
        overusage_punctuation,
        capitalization_score,
        style_score,
        num_paragraphs
    ]
    return vecteur_complexite

def calculate_vector_concatene(essay):
    """
    Args:
        essay (str): The essay text.

    Returns:
        list: A concatenated feature vector containing both linguistic and count-based features.
    """
    vecteur_texte = calculate_vector(essay,df4)
    X = vectorizer.transform([essay])
    vecteur_count = list(X.toarray()[0])
    vecteur_concatene = vecteur_texte + vecteur_count
    
    return vecteur_concatene


def vectorize_essays(data):
    vectors = []
    for essay in data['essay']:
        vector = calculate_vector(essay,df4)
        vectors.append(vector)
    return np.array(vectors)

def vectorize_essays_concatene(data):
    vectors = []
    for essay in data['essay']:
        vector = calculate_vector_concatene(essay)
        vectors.append(vector)
    return vectors



def evaluate_models(models, X_train, y_train, X_test, y_test):
    """
    Evaluate the performance of machine learning models on a given dataset.

    Parameters:
    - models (dict): A dictionary containing machine learning models.
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data labels.
    - X_test (array-like): Test data features.
    - y_test (array-like): Test data labels.

    Returns:
    - results (list of dict): List of dictionaries containing evaluation results for each model.
    """
    results = []

    for model_name, model in models.items():
        model_results = {}
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        score_train = model.score(X_train, y_train)
        score_test = model.score(X_test, y_test)

        model_results['Model'] = model_name
        model_results['MSE'] = mse
        model_results['RMSE'] = rmse
        model_results['R² (Coefficient of Determination)'] = r2
        model_results['Training Score'] = score_train
        model_results['Test Score'] = score_test

        results.append(model_results)

    return results


def evaluate_models_second_dataset(models, X_train1, y_train1, X_test1, y_test1):
    """
    Evaluate the performance of machine learning models on a second dataset.

    Parameters:
    - models (dict): A dictionary containing machine learning models.
    - X_train1 (array-like): Training data features for the second dataset.
    - y_train1 (array-like): Training data labels for the second dataset.
    - X_test1 (array-like): Test data features for the second dataset.
    - y_test1 (array-like): Test data labels for the second dataset.

    Returns:
    - results (list of dict): List of dictionaries containing evaluation results for each model on the second dataset.
    """
    results = []

    for model_name, model in models.items():
        model_results = {}
        model.fit(X_train1, y_train1)
        y_pred = model.predict(X_test1)
        mse = mean_squared_error(y_test1, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test1, y_pred)
        score_train = model.score(X_train1, y_train1)
        score_test = model.score(X_test1, y_test1)

        model_results['Model'] = model_name
        model_results['MSE'] = mse
        model_results['RMSE'] = rmse
        model_results['R² (Coefficient of Determination)'] = r2
        model_results['Training Score'] = score_train
        model_results['Test Score'] = score_test

        results.append(model_results)

    return results


def plot_learning_curve(estimator, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Plots learning curves of an estimator for various training set sizes.

    This function plots the learning curve of an estimator for different training set sizes. 
    It uses learning_curve from scikit-learn to compute the train and test scores for 
    different training set sizes.

    Parameters:
        estimator (object): The estimator object to use.
        X (array-like): The training data.
        y (array-like): The target values.
        cv (int, cross-validation generator or None, optional): If int, the number of folds.
            Otherwise, if callable, a cross-validation generator or an object implementing the 
            split interface. For str, use 'predefined' to use the default splits. 
            Defaults to None.
        n_jobs (int, optional): The number of CPUs to use to do the cross-validation. 
            Defaults to None, meaning 1.
        train_sizes (array-like): The training set sizes for which to compute the learning curve.
            Defaults to np.linspace(.1, 1.0, 5).

    Returns:
        matplotlib.pyplot object: The plot object.

    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
