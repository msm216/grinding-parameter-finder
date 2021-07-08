import pandas as pd
import numpy as np
import mglearn

from time import time

from sklearn import linear_model
from sklearn import tree
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from data_prepare import X_train, X_test, y_train, y_test


# ************************************** initialize the prognosis methods ******************************************** #


# start = timeit.default_timer()
start = time()

# K-Folds cross-validator
kfold = KFold(n_splits=5, shuffle=True, random_state=0)


# Linear Regression
print("Star training Linear Regression...")
lreg = linear_model.LinearRegression()
# specify the parameters to be searched as a dictionary
param_grid_lreg = {
    "fit_intercept": [True],
    "normalize": [False],
    "copy_X": [True],
    "n_jobs": [1],
}
# take through the search
gs_lreg = GridSearchCV(lreg, param_grid_lreg, cv=kfold)
gs_lreg.fit(X_train, y_train)
score_lreg = gs_lreg.score(X_test, y_test)  # <--- score on test data
# retrain the model with best parameters
lreg = linear_model.LinearRegression(**gs_lreg.best_params_).fit(X_train, y_train)
pred_lreg = lreg.predict  # <--- for the function picking!
# metrics of Linear Regression
mse_lreg = mean_squared_error(y_test, pred_lreg(X_test))
mae_lreg = mean_absolute_error(y_test, pred_lreg(X_test))
r2_lreg = r2_score(y_test, pred_lreg(X_test))
print("MSE: ", mse_lreg)
print("MAE: ", mae_lreg)


# Decision Tree
print("Star training Decision Tree...")
dtree = tree.DecisionTreeRegressor()
# specify the parameters to be searched as a dictionary
depth_list = np.arange(1, 11, 1).tolist()
leaf_list = np.arange(1, 21, 1).tolist()
param_grid_dtree = {"max_depth": depth_list, "min_samples_leaf": leaf_list}
# take through the search
gs_dtree = GridSearchCV(dtree, param_grid_dtree, cv=kfold)
gs_dtree.fit(X_train, y_train)
score_dtree = gs_dtree.score(X_test, y_test)  # <--- score on test data
# retrain the decision tree with the best parameters
dtree = tree.DecisionTreeRegressor(**gs_dtree.best_params_).fit(X_train, y_train)
pred_dtree = dtree.predict  # <--- for the function picking!
# metrics of Decision Tree
mse_dtree = mean_squared_error(y_test, pred_dtree(X_test))
mae_dtree = mean_absolute_error(y_test, pred_dtree(X_test))
r2_dtree = r2_score(y_test, pred_dtree(X_test))
print("MSE: ", mse_dtree)
print("MAE: ", mae_dtree)


# ********************************************************************************** #
# Grid of Dicision Tree
cv_results = pd.DataFrame(gs_dtree.cv_results_)
plt.figure(figsize=(10, 10))
plt.title("Grid of Dicision Tree")
# mittlere Scores der Validierung
scores_dtree = np.array(cv_results.mean_test_score).reshape(10, 20)
# scores_dtree = np.array(results.mean_test_score).reshape(5, 10)
scores_image = mglearn.tools.heatmap(
    scores_dtree,
    ylabel="max_depth",
    yticklabels=param_grid_dtree["max_depth"],
    xlabel="min_samples_leaf",
    xticklabels=param_grid_dtree["min_samples_leaf"],
    cmap="viridis",
)

# visualize the Dicision Tree
de = gs_dtree.best_params_['max_depth']
le = gs_dtree.best_params_['min_samples_leaf']
plt.figure(figsize=(20, 12))
plt.suptitle("GridSearchCV, max_depth: {}, min_samples_leaf: {}".format(de, le), fontsize=16)
tree.plot_tree(dtree, filled=True, fontsize=14)
plt.show()
# ********************************************************************************** #


# Multi-Layer-Perceptron
print("Star training Multi-Layer-Perceptron...")
mlp = MLPRegressor()
# specify the parameters to be searched as a dictionary
param_grid_mlp = {
    "activation": ["relu"],
    "solver": ["lbfgs"],
    "alpha": [0.1, 0.001, 0.0001],
    "hidden_layer_sizes": [(19, 16), (5, 2)],
    "max_iter": [1000],
    "random_state": [0],
    "tol": [1e-07, 1e-06, 1e-05, 1e-04],
}
# take through the search
gs_mlp = GridSearchCV(mlp, param_grid_mlp, cv=kfold)
gs_mlp.fit(X_train, y_train)
score_mlp = gs_mlp.score(X_test, y_test)  # <--- score on test data
# retrain the model with best parameters
mlp = MLPRegressor(**gs_mlp.best_params_).fit(X_train, y_train)
pred_mlp = mlp.predict  # <--- for the function picking!
# metrics of Multi-Layer-Perceptron
mse_mlp = mean_squared_error(y_test, pred_mlp(X_test))
mae_mlp = mean_absolute_error(y_test, pred_mlp(X_test))
r2_mlp = r2_score(y_test, pred_mlp(X_test))
print("MSE: ", mse_mlp)
print("MAE: ", mae_mlp)


print(80 * "=")
print(
    "Initiation of models has taken {:.2f} seconds.".format((time() - start))
)
print(80 * "=")


# *********************************** select the suitable prognosis model ******************************************** #

# make a DataFrame of accuracies and the functions
df_inf = pd.DataFrame(
    [
        ["lreg", score_lreg, pred_lreg],
        ["dtree", score_dtree, pred_dtree],
        ["mlp", score_mlp, pred_mlp],
    ],
    columns=["Name", "Score", "Funktion"],
)

# rewrite the index start with 1
df_inf.index = np.arange(1, len(df_inf["Name"]) + 1, 1)
# turn the expression of accuracy into percentage
df_inf.loc[:, "Score"] = df_inf.loc[:, "Score"].apply(lambda x: round(x * 100, 2))
df_inf.loc[:, "Score"] = df_inf.loc[:, "Score"].apply(
    lambda x: "{:.2f}{}".format(x, "%")
)

# function selector
def select_func():
    # pick the model with the highest accuracy by default automatically
    r2_list = df_inf.loc[:, "Score"].tolist()  # starts with 0!
    auto_index = r2_list.index(max(r2_list))
    method_auto = df_inf.iloc[auto_index].loc["Funktion"]
    name_auto = df_inf.iloc[auto_index].loc["Name"]
    # method = None
    # name = ''    # initial local variable 'name'
    # manual selection
    my_num = input("Select a prediction model: ")  # <--- input here!
    if len(my_num) == 0:
        name = name_auto
        print("No input, the model stays by default: {}".format(name))
        method = method_auto
    else:
        # generate a list of index
        idx_list = list(range(1, len(df_inf) + 1, 1))
        # adjust the input value as index of the function list
        idx = min(idx_list, key=lambda x: abs(x - float(my_num)))
        method = df_inf.loc[idx, "Funktion"]
        name = df_inf.loc[idx, "Name"]
        print("Chosen: {}".format(name))
    return method, name


# information to be displayed
df_display = df_inf[["Name", "Score"]]
# a list of method names
name_list = df_inf.loc[:, "Name"].tolist()

# number of models
method_num = len(df_display)

print(
    "There are {0} prediction models available:\n"
    "[1] Lineare Regression ({1[0]})\n"
    "[2] Decision Tree ({1[1]})\n"
    "[3] Multi-Layer-Perceptron ({1[2]})".format(method_num, name_list)
)
print("with the generalization accuracies: \n", df_display)
print(50 * "=")

# pick the function
pred_method, method_name = select_func()

# show the parameter setting of the estimator
print("Configuration of the prediction model:")
print(pred_method)
print(80 * "=")


if __name__ == "__main__":
    # x-axis
    r = len(X_test) + 1
    # y-axis
    y_pred = pred_method(X_test)
    # plot the prediction result
    plt.plot(np.arange(1, r), y_pred, "go-", label="predict")
    plt.plot(np.arange(1, r), y_test, "co-", label="real")
    plt.legend()
    plt.show()
