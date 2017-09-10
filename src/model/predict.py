import os
import numpy as np
import pandas as pd
import datetime as dt

import sklearn
from sklearn.externals import joblib

def predict_prob(testing):
    """
    predict the probability of each class...
    """
    df = model.predict_proba(testing.drop('vessel_number', axis = 1))
    df = pd.DataFrame([row[:,1] for row in df]).transpose()
    df = df.set_index(testing.vessel_number)
    df.columns = model.classes_

    df = df.unstack().reset_index().sort_values(['vessel_number'])
    df.rename(columns={
                       'level_0' : 'FishingType',
                       'vessel_number' : 'Track#',
                       0 : 'Prob'
                       }, inplace=True)
    df = df[['Track#', 'FishingType', 'Prob']]
    return df

def predict_class(df):
    """
    predict a binary class...
    """
    df = pd.DataFrame(model.predict(
                               df.drop('vessel_number', axis = 1)),
                               index=df.vessel_number)
    df = pd.get_dummies(df)

    df.rename(columns={
                       '0_longliner' : 'longliner',
                       '0_seiner' : 'seiner',
                       '0_support' : 'support',
                       '0_trawler' : 'trawler'
                      }, inplace=True)

    df = df.unstack().reset_index().sort_values(['vessel_number'])
    df.rename(columns={
                       'level_0' : 'FishingType',
                       'vessel_number' : 'Track#',
                       0 : 'Prob'
                       }, inplace=True)
    df = df[['Track#', 'FishingType', 'Prob']]
    return df

if __name__ == '__main__':
    model = joblib.load((os.environ['MODEL_FOLDER'] + "model_3.pkl"))

    testing = pd.read_csv((os.environ['PROJECT_FOLDER'] +
                            "src/data/" + "testing.csv"))

    predictions = predict_prob(testing)
        #predictions = predict_class(testing)

    predictions.to_csv((os.environ['PROJECT_FOLDER'] + "scoring.csv"),
                       index=False, header=False)
