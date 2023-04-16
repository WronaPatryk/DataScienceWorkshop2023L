import time

import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


def main(models: list, df: DataFrame):
    results = {}
    for model in models:
        model_results = pd.DataFrame(columns=['user', 'acc', 'tpr', 'auc', 'tnr'])
        for user in df["data_donor_email"].unique():
            user_df = df[df["data_donor_email"] == user]
            x = user_df.iloc[:, 4:]
            y = user_df.iloc[:, 1]

            (x_train, y_train), (x_test, y_test) = train_test_split((x, y), test_size=0.2, random_state=0)

            model.fit(x_train, y_train)
            y_pred_proba = model.predict_proba(x_test)[:, 1]
            y_pred = model.predict(x_test)

            auc = roc_auc_score(y_test, y_pred_proba)
            acc = accuracy_score(y_test, y_pred)
            tpr = recall_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            tnr = tn / (tn + fp)

            model_results.append([user, acc, tpr, auc, tnr])

        results[model] = model_results
    return results


if __name__ == '__main__':
    models = [
        KNeighborsClassifier(3, weights="distance"),
        GaussianNB(),
        BernoulliNB(),
        ComplementNB(),
        MultinomialNB()
    ]

    df = pd.read_csv('data/data.csv')

    results = main(models, df)

    for model, result in results.items():
        result.to_csv(f'results/{model}_{int(time.time())}.csv')


