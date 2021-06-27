import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_prepare import data, lower_than

# **********************************************************************************************************************#

# detect indexes of samples lower than 0.005
ind_bad = lower_than(data, ["Abtrag"], 0.01)
# drop the negatives, reset the index
data_gut = data.drop(ind_bad, axis=0).reset_index(drop=True)


if __name__ == "__main__":

    print(
        "Der gute Datensatz besitzt insgesamt {} Datenpunkte.".format(data_gut.shape[0])
    )
    print(100 * "=")
    print("4 zufaellige Datentupel:")
    print(100 * "=")
    print(data_gut.sample(4))
    print(100 * "=")

    # amount of good measurement based on other attributes:
    fig_1, ax = plt.subplots(2, 2, figsize=(9, 8))  # 2 x 2 subplot grid
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.suptitle("Verteilung der guten Messungen")
    sns.countplot("Drehzahl", hue="Qualität", data=data, ax=ax[0, 0])
    ax[0, 0].set_title("Messqualität beim Drehzahl")
    ax[0, 0].set_ylim([0, 100])
    sns.countplot("Vorschub", hue="Qualität", data=data, ax=ax[0, 1])
    ax[0, 1].set_title("Messqualität beim Vorschub")
    ax[0, 1].set_ylim([0, 100])
    sns.countplot("Kraft", hue="Qualität", data=data, ax=ax[1, 0])
    ax[1, 0].set_title("Messqualität beim Kraft")
    ax[1, 0].set_ylim([0, 100])
    sns.countplot("Winkel", hue="Qualität", data=data, ax=ax[1, 1])
    ax[1, 1].set_title("Messqualität beim Winkel")
    ax[1, 1].set_ylim([0, 100])
    plt.show()

    # Show the correlation between parameters and material removal
    data_heat = pd.DataFrame(
        data_gut, columns=["Drehzahl", "Vorschub", "Kraft", "Winkel", "Abtrag"]
    )
    colormap = plt.cm.RdBu
    plt.figure(figsize=(8, 8))
    plt.title("Korrelation der Parametern und Abtrag", y=1.05, size=15)
    sns.heatmap(
        data_heat.astype(float).corr(),
        linewidths=1,
        vmax=1.0,
        vmin=-1.0,
        square=True,
        cmap=colormap,
        linecolor="white",
        annot=True,
    )
    plt.show()

    # linear regression
    fig_8, ax = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.suptitle("linear regression across levels of a categorical variable")
    sns.regplot(x=data_gut["Drehzahl"], y=data_gut["Abtrag"], fit_reg=True, ax=ax[0, 0])
    ax[0, 0].set_title("Abtrag bei Drehzahl")
    sns.regplot(x=data_gut["Vorschub"], y=data_gut["Abtrag"], fit_reg=True, ax=ax[0, 1])
    ax[0, 1].set_title("Abtrag bei Vorschub")
    sns.regplot(x=data_gut["Kraft"], y=data_gut["Abtrag"], fit_reg=True, ax=ax[1, 0])
    ax[1, 0].set_title("Abtrag bei Kraft")
    sns.regplot(x=data_gut["Winkel"], y=data_gut["Abtrag"], fit_reg=True, ax=ax[1, 1])
    ax[1, 1].set_title("Abtrag bei Winkel")
    plt.show()
