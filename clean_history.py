import pandas as pd

# reset the csv-file
def clean_csv(csvName):
    df = pd.DataFrame(
        columns=[
            "Drehzahl",
            "Vorschub",
            "Kraft",
            "Winkel",
            "Abtrag",
            "Input",
            "Methode",
        ]
    )
    df.to_csv(str(csvName), index=True, header=True)
    print("csv-file '", str(csvName), "' has been resetet.")


if __name__ == "__main__":
    clean_csv("run_history.csv")
    clean_csv("p_recommend.csv")
