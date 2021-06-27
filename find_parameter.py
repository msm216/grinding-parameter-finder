import os
import pandas as pd
import numpy as np
import itertools

from model_initialize import pred_method, method_name


# ***************************************** parameter pre-selection ************************************************** #

# parameter selector, find the fitting value from in lists, which is closest to the input value

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


# pre-selection of parameter "Drehzahl"
array_d = np.arange(1000, 5001, 500)  # numpy.ndarray
num_d = input("Waehlen Sie eine Werkzeugdrehzahl zw. 1000 und 5000: ")
Drehzahl = get_list(num_d, array_d)
print("Die gewuenschte Werkzeugdrehzahl ist: ", num_d, "[1/min]")
print("Daraus erflogt die Liste zur Parametersuche:\n", Drehzahl)

print(50 * "=")

# pre-selection of parameter "Vorschub"
array_v = np.arange(50, 201, 10)  # numpy.ndarray
num_v = input("Waehlen Sie eine Vorschubgeschwindigkeit zw. 50 und 200: ")
Vorschub = get_list(num_v, array_v)
print("Das gewuenschte Vorschubgeschwindigkeit ist: ", num_v, "[mm/s]")
print("Daraus erflogt die Liste zur Parametersuche:\n", Vorschub)

print(50 * "=")

# pre-selection of parameter "Kraft"
array_k = np.arange(10, 31, 1)  # numpy.ndarray
num_k = input("Waehlen Sie eine Anpresskraft zw. 10 und 30: ")
Kraft = get_list(num_k, array_k)
print("Die gewuenschte Anpresskraft ist: ", num_k, "[N]")
print("Daraus erflogt die Liste zur Parametersuche:\n", Kraft)

print(50 * "=")

# pre-selection of parameter "Winkel"
array_w = np.arange(1, 6, 1)  # numpy.ndarray
num_w = input("Waehlen Sie einen Anstellwinkel zw. 1 und 5: ")
Winkel = get_list(num_w, array_w)
print("Der gewuenschte Anstellwinkel ist: ", num_w, "[°]")
print("Daraus erflogt die Liste zur Parametersuche:\n", Winkel)

print(100 * "=")


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
    for d, v, k, w in itertools.product(*parameter_lists):
        prediction = estimator(np.array([[d / 10000, v / 500, k / 50, w / 10]]))
        deviation = abs(prediction - expectation)
        if deviation < dev_min:
            parameter = [d, v, k, w]
            dev_min = deviation
    return parameter  # list


print("Bitte geben Sie Materialabtrag in [mm] der Reihe nach.")
print("Die Eingabe leer lassen und Enter druecken, um Eingabe aufzuhoeren.")

# run the parameter-finder for multiple times, put the results into a list
p_list = []
# list of all inputs
in_list = []
while True:
    the_target = input("Zielabtrag: ")
    if len(the_target) == 0:
        break
    else:
        res_list = find_parameter(
            pred_method, float(the_target), Drehzahl, Vorschub, Kraft, Winkel
        )
        p_list.append(res_list)
        Winkel = [res_list[3]]
        in_list.append(the_target)

# DataFrame with full information
df_01 = pd.DataFrame(p_list, columns=["Drehzahl", "Vorschub", "Kraft", "Winkel"])
df_01["Abtrag"] = df_01.apply(
    lambda x: pred_method(
        np.array(
            [
                [
                    x["Drehzahl"] / 10000,
                    x["Vorschub"] / 500,
                    x["Kraft"] / 50,
                    x["Winkel"] / 10,
                ]
            ]
        )
    )[0],
    axis=1,
)

df_01["Input"] = in_list
df_01["Methode"] = method_name
# DataFrame for the recommend
df_02 = df_01[["Drehzahl", "Vorschub", "Kraft", "Winkel"]]

print("Empfohlen wird folgende Prozessparameter:\n", df_01)
print(100 * "=")


# *********************************************** write into csv ***************************************************** #

# write the parameter into a csv-file
df_01.to_csv("run_history.csv", index=True, mode="a", header=False)
df_02.to_csv("p_recommend.csv", index=True, header=False)
print("Die Prozessparameter wird als CSV-Datei gespeichert unter:\n" + os.getcwd())
