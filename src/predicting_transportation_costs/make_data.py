import pandas as pd
import numpy as np
from predicting_transportation_costs import config as cfg

def main() :
    df = pd.read_csv(cfg.DATA_DIR / "train-test.csv")
    df["load_id"] = df["load_id"].astype("string")

    df_test= df[df["date"] > "2025-09-01"].reset_index(drop=True)
    df_train= df[df["date"] <= "2025-09-01"].reset_index(drop=True)

    df_train["delivery_lat"] = pd.to_numeric(df_train["delivery_lat"], errors="coerce").astype("float64")
    df_train = df_train[df_train["delivery_lat"].notna()]
    df_train= df_train[df_train["market_index"].notna()]

    df_train.to_parquet(cfg.DATA_DIR / "train.parquet")
    df_test.to_parquet(cfg.DATA_DIR / "test.parquet")
    return

if __name__ == "__main__":
    main()