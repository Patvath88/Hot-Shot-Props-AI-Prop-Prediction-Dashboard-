from train_model import train_stat
import pandas as pd


def main():
    df = pd.read_csv("data/model_dataset.csv")

    train_stat(df, "points", "points")
    train_stat(df, "rebounds", "rebounds")
    train_stat(df, "assists", "assists")

    print("All models trained successfully.")


if __name__ == "__main__":
    main()