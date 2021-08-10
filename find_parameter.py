import os
import pandas as pd
import numpy as np
import itertools


from data_prepare import sql_contr
from model_initialize import pred_method, method_name
from data_prepare import scaler
from fake_bench import new_measuring_results, write_to_sql


# find the fitting value from the parameter lists, which is closest to the input value
def get_list(num, array):
    if len(num) == 0:
        # transform the array to list directly
        the_list = array.tolist()
    else:
        # returns the index of the minimal value in array
        idx = (np.abs(array - float(num))).argmin()
        # build a new list with one single value
        the_list = [array[idx]]
    return the_list

# ***************************************** parameter pre-selection ************************************************** #

# parameter selector, find the fitting value from in lists, which is closest to the input value

# pre-selection of parameter "Drehzahl"
array_d = np.arange(1000, 5001, 500)  # numpy.ndarray
num_d = input("Select a Werkzeugdrehzahl between 1000 and 5000: ")
Drehzahl = get_list(num_d, array_d)
print("The requested Werkzeugdrehzahl is: ", num_d, "[1/min]")
print("List for the parameter search:\n", Drehzahl)

print(50 * "=")

# pre-selection of parameter "Vorschub"
array_v = np.arange(50, 201, 10)  # numpy.ndarray
num_v = input("Select a Vorschubgeschwindigkeit between 50 and 200: ")
Vorschub = get_list(num_v, array_v)
print("The requested Vorschubgeschwindigkeit is: ", num_v, "[mm/s]")
print("List for the parameter search:\n", Vorschub)

print(50 * "=")

# pre-selection of parameter "Kraft"
array_k = np.arange(10, 31, 1)  # numpy.ndarray
num_k = input("Select a Anpresskraft between 10 and 30: ")
Kraft = get_list(num_k, array_k)
print("The requested Anpresskraft is: ", num_k, "[N]")
print("List for the parameter search:\n", Kraft)

print(50 * "=")

# pre-selection of parameter "Winkel"
array_w = np.arange(1, 6, 1)  # numpy.ndarray
num_w = input("Select a Anstellwinkel between 1 and 5: ")
Winkel = get_list(num_w, array_w)
print("The requested Anstellwinkel is: ", num_w, "[°]")
print("List for the parameter search:\n", Winkel)

print(80 * "=")


# ********************************************** parameter finder **************************************************** #

# original parameter lists
# Drehzahl = [1000, 3000, 5000]
# Vorschub = [50, 100, 200]
# Kraft = [10, 20, 30]
# Winkel = [1, 3, 5]

# iterates over all possible parameter options and finds the parameter constellation
# with the smallest deviation from the input
def find_parameter(estimator, expectation, *parameter_lists):
    dev_min = np.inf
    parameter = []
    estimation = float()
    for d, v, k, w in itertools.product(*parameter_lists):
        #prediction = estimator(np.array([[d/10000, v/500, k/50, w/10]]))
        pred = estimator(scaler.transform([[d, v, k, w]]))
        deviation = abs(pred - expectation)
        if deviation < dev_min:
            parameter = [d, v, k, w]
            estimation = pred[0]
            dev_min = deviation
    return parameter, estimation    # list

print("Bitte geben Sie Materialabtrag in [mm] der Reihe nach.")
print("Die Eingabe leer lassen und Enter druecken, um weiter zu gehen.")

# run the parameter-finder for multiple times, put the results into a list
p_list = []
e_list = []
# list of all inputs
in_list = []

while True:
    target = input("Zielabtrag: ")
    if len(target) == 0:
        break
    else:
        para, esti = find_parameter(pred_method, float(target), Drehzahl, Vorschub, Kraft, Winkel)
        p_list.append(para)
        e_list.append(esti)
        Winkel = [para[3]]
        in_list.append(target)

# DataFrame with full information
df_log = pd.DataFrame(p_list, columns=['Drehzahl', 'Vorschub', 'Kraft', 'Winkel'])
df_log["Abtrag"] = e_list
df_log['Input'] = in_list
df_log['Methode'] = method_name
# DataFrame for the recommend
df_rec = df_log[["Drehzahl", "Vorschub", "Kraft", "Winkel"]]

print("Following parameter have been recommended:\n", df_log)
print(80 * "=")


# *********************************************** write into csv ***************************************************** #

# write the parameter into a csv-file
df_log.to_csv("run_history.csv", index=True, mode="a", header=False)
df_rec.to_csv("p_recommend.csv", index=True, header=False)
print(
    "The recommended process parameter have been saved into csv-file under:\n"
    + os.getcwd()
)

# ************************************************ work on bench ***************************************************** #

print("Virtual measuring results:\n", new_measuring_results)

if sql_contr == True:
    write_to_sql(new_measuring_results)
    print("New data has been written into MySQL.")
else:
    print("Done, MySQL database not updated.")
