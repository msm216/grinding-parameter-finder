import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import time

from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_prepare import Feature, Label


# ******************************************************************************************************************** #

# show the influence of changing hyperparameters on the model
def dtree_hp_tester(depth=20, leaf=20):
    # split the data into training/validation sets and test set
    X_tv, X_test, y_tv, y_test = train_test_split(
        Feature, Label, random_state=0
    )
    # K-Folds cross-validator
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    # range of hyperparameters
    depth_list = np.arange(1, depth + 1, 1).tolist()
    leaf_list = np.arange(2, leaf + 2, 1).tolist()
    # score of cross-vali
    scores_d = []
    scores_l = []
    # score on test
    evas_d = []
    evas_l = []
    # evaluate max_depth
    for d in depth_list:
        dtree = tree.DecisionTreeRegressor(max_depth=d)
        score = np.mean(cross_val_score(dtree, X_tv, y_tv, cv=kfold))
        dtree.fit(X_tv, y_tv)
        eva = dtree.score(X_test, y_test)
        scores_d.append(score)
        evas_d.append(eva)
    # evaluate min_samples_split
    for l in leaf_list:
        dtree = tree.DecisionTreeRegressor(min_samples_split=l)
        score = np.mean(cross_val_score(dtree, X_tv, y_tv, cv=kfold))
        dtree.fit(X_tv, y_tv)
        eva = dtree.score(X_test, y_test)
        scores_l.append(score)
        evas_l.append(eva)

    # visualization
    plt.figure(figsize=(12, 8))
    plt.title('Influence of Hyperparameters by Dicision Tree')
    x_ticks = np.arange(0, 21, 2)
    plt.xticks(x_ticks)
    plt.ylabel('Coefficient of Determination R^2')
    plt.plot(depth_list, scores_d, 'co-', label='train: max_depth')
    plt.plot(depth_list, evas_d, 'c^-', label='eva: max_depth')
    plt.plot(leaf_list, scores_l, 'ro-', label='train: min_leaf')
    plt.plot(leaf_list, evas_l, 'r^-', label='eva: min_leaf')
    plt.legend()
    plt.grid()
    plt.show()

#
dtree_hp_tester()


# ******************************************************************************************************************** #


# search full of the grid
def gscv_tree(feature=Feature, label=Label, depth_num=10, leaf_num=20):
    # split the data into training/validation sets and test set
    X_train, X_test, y_train, y_test = train_test_split(
        Feature, Label, random_state=0
    )
    print("size of training data: {}\nsize of test data: {}\n"
          .format(X_train.shape[0], X_test.shape[0]))
    # K-Folds cross-validator
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    # initialize
    depth_list = np.arange(1, depth_num + 1, 1).tolist()
    leaf_list = np.arange(2, leaf_num + 2, 1).tolist()
    # parameter grid
    param_grid_dtree = {
        'max_depth': depth_list,
        'min_samples_leaf': leaf_list
    }
    # timer on
    print("Start training with GridSearchCV...")
    start = time()
    # initialize the decision tree regressor
    dtree = tree.DecisionTreeRegressor()
    # take through the search
    gs_dtree = GridSearchCV(
        dtree, param_grid_dtree, cv=kfold
    ).fit(X_train, y_train)
    # score on test data with the best configuration, to be compared with r2_score
    score_gs = gs_dtree.score(X_test, y_test)

    # retrain the decision tree with the best parameters
    dtree = tree.DecisionTreeRegressor(**gs_dtree.best_params_).fit(X_train, y_train)
    test_score = dtree.score(X_test, y_test)
    # prediction method
    pred = dtree.predict
    # metrics of Decision Tree
    mse_dtree = mean_squared_error(y_test, pred(X_test))
    mae_dtree = mean_absolute_error(y_test, pred(X_test))

    # report the training result
    print("Done, training took {:.2f}sec.".format((time() - start)))
    print(
        "best score by training: {:.2f}%".format(
            round(gs_dtree.best_score_ * 100, 2)
        )
    )
    # print("best configuration: {}".format(gs_dtree.best_params_))
    print("best estimator: {}".format(gs_dtree.best_estimator_))
    # confirm the end-valuations
    print(
        "score on test data through GridSearchCV: {:.2f}%".format(
            round(score_gs * 100, 2)
        )
    )
    print(
        "score on test data with best configuration: {:.2f}%".format(
            round(test_score * 100, 2)
        )
    )
    print(
        "score on test data through r2_score: {:.2f}%".format(
            round(r2_score(y_test, pred(X_test)) * 100, 2)
        )
    )
    print("MSE: ", mse_dtree)
    print("MAE: ", mae_dtree)

    # visualize the tree
    d = gs_dtree.best_params_['max_depth']
    l = gs_dtree.best_params_['min_samples_leaf']
    plt.figure(figsize=(20, 10))
    plt.title("max_depth: {}, min_samples_leaf: {}".format(d, l), fontsize=16)
    tree.plot_tree(dtree, filled=True, fontsize=14)
    plt.plot()

    # ********** visualize the result of random search **********#

    # call the results
    results = pd.DataFrame(gs_dtree.cv_results_)
    # generate an empty 20x20 array
    scores = np.zeros(shape=(depth_num, leaf_num))
    # write the mean test scores from rs_results into rs_scores
    for i in range(len(results)):
        serie = results[[
            'param_min_samples_leaf',
            'param_max_depth',
            'mean_test_score'
        ]].iloc[i]
        l = serie[0] - 2
        d = serie[1] - 1
        value = serie[2]
        # change target value
        scores[d][l] = value
    # evenly round the values in the array
    scores = np.around(scores, decimals=2)

    # visualize the grid (matplotlib)
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(scores)
    # set up axes
    ax.set_xticks(np.arange(len(leaf_list)))
    ax.set_yticks(np.arange(len(depth_list)))
    # set up axes label
    ax.set_xticklabels(leaf_list)
    ax.set_yticklabels(depth_list)
    # Loop over data dimensions and create text annotations. depth循环在外，所以depth是y轴
    for i in range(len(depth_list)):
        for j in range(len(leaf_list)):
            text = ax.text(j, i, scores[i, j],
                           ha='center', va='center', color='w')

    ax.set_title("Grid of Score over Hyperparameters for Decision Tree")
    ax.set_xlabel('min_samples_leaf')
    ax.set_ylabel('max_depth')
    fig.tight_layout()
    plt.show()

    return pred, score_gs


# best score by training: 78.56%
# best hyperparameter: {'max_depth': 6, 'min_samples_leaf': 3}
# score on test data with best configuration: 84.76%
pred_dtree, score_dtree = gscv_tree()
print(80 * "=")

# ******************************************************************************************************************** #


# search quarter of the grid
def rscv_dtree(feature=Feature, label=Label, depth_num=10, leaf_num=20):
    # split the data into training/validation sets and test set
    X_train, X_test, y_train, y_test = train_test_split(
        Feature, Label, random_state=0
    )
    print("size of training data: {}\nsize of test data: {}\n"
          .format(X_train.shape[0], X_test.shape[0]))
    # K-Folds cross-validator
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    # initialize
    depth_list = np.arange(1, depth_num + 1, 1).tolist()
    leaf_list = np.arange(2, leaf_num + 2, 1).tolist()
    # parameter grid
    param_grid_dtree = {
        'max_depth': depth_list,
        'min_samples_leaf': leaf_list
    }
    # timer on
    print("Start training with RandomizedSearchCV...")
    start = time()
    # initialize the decision tree regressor
    dtree = tree.DecisionTreeRegressor()
    # define iteration number
    it = int(depth_num * 0.5) * int(leaf_num * 0.5)
    print("with {} iteration.".format(it))
    # take through the search
    rs_dtree = RandomizedSearchCV(
        dtree, param_distributions=param_grid_dtree, n_iter=it, cv=kfold
    ).fit(X_train, y_train)
    # score on test data with the best configuration
    score_rs = rs_dtree.score(X_test, y_test)

    # retrain the decision tree with the best parameters
    dtree = tree.DecisionTreeRegressor(**rs_dtree.best_params_).fit(X_train, y_train)
    test_score = dtree.score(X_test, y_test)
    # prediction method
    pred = dtree.predict
    # metrics of Decision Tree
    mse_dtree = mean_squared_error(y_test, pred(X_test))
    mae_dtree = mean_absolute_error(y_test, pred(X_test))

    # report the training result
    print("Done, training took {:.2f}sec.".format((time() - start)))
    print(
        "best score by training: {:.2f}%".format(
            round(rs_dtree.best_score_ * 100, 2)
        )
    )
    # print("best configuration: {}".format(gs_dtree.best_params_))
    print("best estimator: {}".format(rs_dtree.best_estimator_))
    # confirm the end-valuations
    print(
        "score on test data through GridSearchCV: {:.2f}%".format(
            round(score_rs * 100, 2)
        )
    )
    print(
        "score on test data with best configuration: {:.2f}%".format(
            round(test_score * 100, 2)
        )
    )
    print(
        "score on test data through r2_score: {:.2f}%".format(
            round(r2_score(y_test, pred(X_test)) * 100, 2)
        )
    )
    print("MSE: ", mse_dtree)
    print("MAE: ", mae_dtree)

    # visualize the tree
    d = rs_dtree.best_params_['max_depth']
    l = rs_dtree.best_params_['min_samples_leaf']
    plt.figure(figsize=(20, 10))
    plt.title("max_depth: {}, min_samples_leaf: {}".format(d, l), fontsize=16)
    tree.plot_tree(dtree, filled=True, fontsize=14)
    plt.plot()

    # ********** visualize the result of grid search **********#

    # call the results
    results = pd.DataFrame(rs_dtree.cv_results_)
    # generate an empty 20x20 array
    scores = np.zeros(shape=(depth_num, leaf_num))
    # write the mean test scores from rs_results into rs_scores
    for i in range(len(results)):
        serie = results[[
            'param_min_samples_leaf',
            'param_max_depth',
            'mean_test_score'
        ]].iloc[i]
        l = serie[0] - 2
        d = serie[1] - 1
        value = serie[2]
        # change target value
        scores[d][l] = value
    # evenly round the values in the array
    scores = np.around(scores, decimals=2)

    # visualize the grid (matplotlib)
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(scores)
    # set up axes
    ax.set_xticks(np.arange(len(leaf_list)))
    ax.set_yticks(np.arange(len(depth_list)))
    # set up axes label
    ax.set_xticklabels(leaf_list)
    ax.set_yticklabels(depth_list)
    # Loop over data dimensions and create text annotations. depth循环在外，所以depth是y轴
    for i in range(len(depth_list)):
        for j in range(len(leaf_list)):
            text = ax.text(j, i, scores[i, j],
                           ha='center', va='center', color='w')

    ax.set_title("Grid of Score over Hyperparameters for Decision Tree")
    ax.set_xlabel('min_samples_leaf')
    ax.set_ylabel('max_depth')
    fig.tight_layout()
    plt.show()

    # return pred, score_rs


rscv_dtree()
print(80 * "=")
