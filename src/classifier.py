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
import numpy as np


def remove_na_values(df: DataFrame, drop_column_threshold=0.15):
    null_var = df.isnull().sum() / df.shape[0]
    drop_columns = null_var[null_var > drop_column_threshold].keys()
    df = df.drop(columns=drop_columns)
    df = df.dropna()
    return df


def false_rejection_rate(y_true, y_pred):
    classes, counts = np.unique(y_true, return_counts=True)
    individual_rates = {
        user: float(len([1 for t, p in zip(y_true, y_pred) if t == user and t != p])) / count
        for user, count in zip(classes, counts)
    }
    return np.mean(list(individual_rates.values()))


def false_acceptance_rate(y_true, y_pred):
    return false_rejection_rate(y_pred, y_true)


def train_models(models: list, df: DataFrame):
    results = pd.DataFrame(columns=['model', 'acc', 'tpr', 'far', 'frr'])
    x = df.iloc[:, 4:]
    y = df.iloc[:, 0]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    for model in tqdm(models):
        model.fit(x_train, y_train)
        # y_pred_proba = model.predict_proba(x_test)
        y_pred = model.predict(x_test)

        # auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')
        acc = accuracy_score(y_test, y_pred)
        tpr = recall_score(y_test, y_pred, average='micro')

        far = false_acceptance_rate(y_test, y_pred)
        frr = false_rejection_rate(y_test, y_pred)

        results = results.append({'model': str(model), 'acc': acc, 'tpr': tpr, 'far': far, 'frr': frr}, ignore_index=True)
    return results, models, (x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    models = [
        KNeighborsClassifier(3, weights="distance"),
        GaussianNB(),
        BernoulliNB(),
        ComplementNB(),
        MultinomialNB(),
        RandomForestClassifier(),
        LogisticRegression(),
        LinearDiscriminantAnalysis(),
        MLPClassifier(),
        GradientBoostingClassifier() #likely to require approx. 30min of training time
    ]

    df = remove_na_values(pd.read_csv('..\data\data.csv'))

    results, models, _ = train_models(models, df)

    results.to_csv(f'classification_results/{int(time.time())}.csv', index=False)