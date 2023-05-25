import pandas as pd
import time
import sklearn.ensemble
from classifier import train_models, remove_na_values
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    models = [RandomForestClassifier(random_state=0)]

    df = remove_na_values(pd.read_csv('data\data.csv'))

    results, models = train_models(models, df)
    rf = models[0]

    f_i = list(zip(df.columns[4:], rf.feature_importances_))
    f_i.sort(key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(f_i, columns=['Column_Name','Feature importance'])
    df.to_csv(f'src\\feature_importances_classifier{int(time.time())}.csv', index=False)
    print(f_i)