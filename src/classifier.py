import time

import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def remove_na_values(df: DataFrame, drop_column_threshold=0.15):
    null_var = df.isnull().sum() / df.shape[0]
    drop_columns = null_var[null_var > drop_column_threshold].keys()
    df = df.drop(columns=drop_columns)
    df = df.dropna()
    return df


def main(models: list, df: DataFrame):
    results = pd.DataFrame(columns=['model', 'acc', 'tpr'])
    for model in models:
        x = df.iloc[:, 4:]
        y = df.iloc[:, 0]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        model.fit(x_train, y_train)
        # y_pred_proba = model.predict_proba(x_test)
        y_pred = model.predict(x_test)

        # auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')
        acc = accuracy_score(y_test, y_pred)
        tpr = recall_score(y_test, y_pred, average='micro')

        results = results.append({'model': str(model), 'acc': acc, 'tpr': tpr}, ignore_index=True)
    return results


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

    results = main(models, df)

    results.to_csv(f'classification_results/{int(time.time())}.csv', index=False)