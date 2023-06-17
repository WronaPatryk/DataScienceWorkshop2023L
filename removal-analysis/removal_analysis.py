import time

import pandas as pd
from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import random

random.seed(42)

def remove_na_values(df: DataFrame, drop_column_threshold=0.15):
    null_var = df.isnull().sum() / df.shape[0]
    drop_columns = null_var[null_var > drop_column_threshold].keys()
    df = df.drop(columns=drop_columns)
    df = df.dropna()
    return df

def remove_columns_ints(df: DataFrame, exclude_columns: list):
    result = df.drop(columns=exclude_columns)
    return result


def train_models(models: list, df: DataFrame):
    results = pd.DataFrame(columns=['model', 'acc', 'tpr'])
    for model in tqdm(models):
        x = df.iloc[:, 4:]
        y = df.iloc[:, 0]

        print(x.shape)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        model.fit(x_train, y_train)
        # y_pred_proba = model.predict_proba(x_test)
        y_pred = model.predict(x_test)

        # auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')
        acc = accuracy_score(y_test, y_pred)
        tpr = recall_score(y_test, y_pred, average='micro')

        results = results.append({'model': str(model), 'acc': acc, 'tpr': tpr}, ignore_index=True)
    return results, models


if __name__ == '__main__':
    models = [
        # KNeighborsClassifier(3, weights="distance"),
        # GaussianNB(),
        # BernoulliNB(),
        # ComplementNB(),
        # MultinomialNB(),
        RandomForestClassifier(),
        # LogisticRegression(),
        # LinearDiscriminantAnalysis(),
        # MLPClassifier(),
        # GradientBoostingClassifier() #likely to require approx. 30min of training time
    ]

    #df = remove_na_values(pd.read_csv('data\data.csv'))
    df = pd.read_csv('data\data.csv')

    agg_features = pd.read_excel('removal-analysis\\aggregated_features.xlsx', header=None).iloc[:,1].to_list()

    for feature in tqdm(agg_features):
        df_trimmed = remove_columns(df, [feature])


        df_trimmed = remove_na_values(df_trimmed)

        results, models = train_models(models, df_trimmed)

        results.to_csv(f'removal-analysis\classification_results\{int(time.time())}_{feature}.csv', index=False)

