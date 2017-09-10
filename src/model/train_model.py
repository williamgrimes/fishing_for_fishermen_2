import os

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.pyplot.switch_backend('agg')

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics.scorer import make_scorer
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib


def prepare_train_test(df):
    """
    prepare data for training by dropping others and
    creating an X and y set.
    """
    df = df[df.vessel_type != 'other']
    df = df.dropna()

    X = df.drop(['vessel_type', 'vessel_number'], axis=1)
    y = df.vessel_type
    return df, X, y

def prepare_train_test_categories(df):
    """
    prepare data for training by dropping others and
    creating an X and y set, where y has dummies.
    """
    df = training
    df = df[df.vessel_type != 'other']
    df = df.dropna()

    X = df.drop(['vessel_type', 'vessel_number'], axis=1)
    y = df.vessel_type
    y_enc = pd.get_dummies(pd.factorize(y)[0])
    y_lab = pd.factorize(y)[1]
    y_enc.columns = y_lab
    y = y_enc
    y_lab = y_lab.tolist()
    return df, X, y, y_lab


def plot_roc_curves(y_lab, fpr, tpr, roc_auc):
    """
    plots roc curves for each class
    """
    for x in y_lab:
        plt.figure()
        plt.plot(fpr[x], tpr[x], label='ROC curve (area = %0.2f)' % roc_auc[x])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('plots/roc_class_'+ str(x)+'.png')


def best_config(X, y):
    """
    find best random forest configuration
    """
    #clf = RandomForestClassifier()
    clf = ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy')
    param_grid = {
                  'n_estimators': [10, 50, 100, 120, 130, 140, 150, 160, 170,\
                                   180, 190, 200, 250, 300, 400, 500,\
                                   1000, 2000, 3000],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'min_samples_leaf': [1, 10, 50, 100, 200]
                 }

    custom_scorer = make_scorer(custom_roc_scorer, greater_is_better=True)

    time_start = datetime.now()
    grid_search = GridSearchCV(clf,
                               param_grid,
                               cv = 5,
                               n_jobs = -1,
                               scoring=custom_scorer
                               )
    grid_search.fit(X, y)

    time_elapsed = datetime.now() - time_start
    print('Time elpased (hh:mm:ss.ms) {}'.format(time_elapsed))

    print("best score: " + str(grid_search.best_score_))

    model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    importance = pd.DataFrame({'feature':X.columns,
                                'importance':np.round(
                                 model.feature_importances_, 3)
                              })
    importance = importance.sort_values('importance',
                                          ascending=False).set_index('feature')

    return model, best_params, importance

def roc_auc_each_class(ground_truth, predictions):
    """
    returns a dictionary of AuC score from ROC of each class
    """
    labels = ground_truth.columns.tolist()
    predictions = pd.DataFrame(predictions, columns=labels).astype(float)
    n_classes = len(labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for x in labels:
        fpr[x], tpr[x], _ = roc_curve(ground_truth[x], predictions[x])
        roc_auc[x] = auc(fpr[x], tpr[x])
    return roc_auc


def custom_roc_scorer(ground_truth, predictions):
    """
    creates as core based on weighted average of roc values
    described in the topcoder competition

    https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16978&pm=14691
    """
    roc_auc = roc_auc_each_class(ground_truth, predictions)
    score = (0.4 *roc_auc.get('trawler')) + (0.3 *roc_auc.get('seiner')) + \
            (0.2 *roc_auc.get('longliner')) + (0.1 * roc_auc.get('support'))
    score = (score * 2 -1) * 10000000
    return score

def generate_model(X, y):
    """
    find best random forest configuration
    """
    clf = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy',
                      max_depth=100, max_features='log2', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=10,
                      min_weight_fraction_leaf=0.0, n_estimators=10000, n_jobs=-1,
                      oob_score=False, random_state=None, verbose=0, warm_start=False)


    model = clf.fit(X, y)
    importance = pd.DataFrame({'feature':X.columns,
                                'importance':np.round(
                                 model.feature_importances_, 3)
                              })
    importance = importance.sort_values('importance',
                                          ascending=False).set_index('feature')
    return model, importance

if __name__ == '__main__':
    training = pd.read_csv((os.environ['PROJECT_FOLDER'] +
                            "src/data/" + "training.csv"))

    training, X, y, y_lab = prepare_train_test_categories(training)

    model, importance = generate_model(X, y)
    model.classes_ = y_lab

    joblib.dump(model, str(os.environ['MODEL_FOLDER'] + "model_3.pkl"))
    importance.to_csv(str(os.environ['MODEL_FOLDER'] + "importances_3.csv"))

    #best_model, best_params, importance = best_config(X, y)
    #importance.to_csv(str(os.environ['MODEL_FOLDER'] + "importances.csv"))
    #model = best_model.fit(X, y)
