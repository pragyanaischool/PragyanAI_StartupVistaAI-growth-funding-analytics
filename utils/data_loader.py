import pandas as pd

def load_data():

    df = pd.read_csv("data/startup_data.csv")

    df["Startup Age"] = 2026 - df["Year Founded"]

    df["Funding Efficiency"] = (
        df["Revenue (M USD)"]
        / df["Funding Amount (M USD)"]
    )

    return df
