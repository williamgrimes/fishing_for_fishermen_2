from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns

import os

def define_clfs_params(grid_size):

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3)
            }

    large_grid = {
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }

    small_grid = {
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }

    test_grid = {
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }

    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def prepare_train_test_categories(df):
    """
    prepare data for training by dropping others and
    creating an X and y set, where y has dummies.
    """
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

def roc_auc_scorer(ground_truth, predictions):
    """
    use simpleloops best classifier and parameter combination
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

    score = (0.4 *roc_auc.get('trawler')) + (0.3 *roc_auc.get('seiner')) + \
            (0.2 *roc_auc.get('longliner')) + (0.1 * roc_auc.get('support'))
    score = (score * 2 -1) * 10000000
    return score


def clf_loop(models_to_run, clfs, grid, X, y, y_lab):
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'weighted_roc_score'))
    for n in range(1, 2):
        # create training and valdation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    model = clf.fit(X_train, y_train)
                    #predict_proba(X_test)[:,1]
                    y_pred_probs = model.predict_proba(X_test)
                    y_pred_probs = pd.DataFrame([row[:,1] for row in y_pred_probs]).transpose()
                    #y_pred_probs = y_pred_probs.set_index(testing.vessel_number)
                    y_pred_probs.columns = y_lab
                    # you can also store the model, feature importances, and prediction scores
                    # we're only storing the metrics for now
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                       roc_auc_scorer(y_test, y_pred_probs)
                                                       ]
                except IndexError:
                    print('IndexError:')
                    continue
    return results_df

def main():
    grid_size = 'small'
    clfs, grid = define_clfs_params(grid_size)
    #models_to_run=['RF','DT','KNN', 'ET', 'AB', 'GB', 'LR', 'NB']
    models_to_run=['RF','DT','KNN', 'ET']

    df = pd.read_csv((os.environ['PROJECT_FOLDER'] +
                      "src/data/" + "training.csv"))

    df = df[df.vessel_type != 'other']

    df, X, y, y_lab = prepare_train_test_categories(df)

    # reduced features
    #f =['sea_temperature_median', 'water_surface_elevation_median', \
    #    'oceanic_depth_std', 'oceanic_depth_mad', 'sog_mad', 'sog_std', \
    #    'oceanic_depth_median', 'speed_mad', 'salinity_median', \
    #    'sog_median', 'distance_to_shore_max', 'speed_std', \
    #    'distance_to_shore_std', 'choloro_conc_median', 'speed_median', \
    #    'distance_to_port_max', 'distance_to_shore_mad', \
    #    'distance_to_shore_median', 'ratio_in_eez', 'salinity_std', \
    #    'distance_to_port_std', 'distance_to_port_median', 'thermo_depth_std', \
    #    'choloro_conc_std', 'salinity_mad', 'distance_to_port_mad', \
    #    'sea_temperature_std', 'course_diff_mad', 'thermo_depth_mad', \
    #    'choloro_conc_mad']

    #X = X[f] # reduced features

    results_df = clf_loop(models_to_run, clfs,grid, X,y, y_lab)
    results_df.to_csv('results/small_simpleloop_output_reduced.csv', index=False)

if __name__ == '__main__':
    main()
